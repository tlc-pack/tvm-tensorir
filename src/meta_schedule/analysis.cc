/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include "../tir/schedule/analysis.h"

#include <numeric>

#include "./analysis.h"
#include "./utils.h"

namespace tvm {
namespace meta_schedule {

/**************** TIR Nodes ****************/
using tir::BlockNode;
using tir::ForNode;

bool IsTrivialBinding(const tir::ScheduleState& self, const tir::StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  tir::BlockRealize realize = tir::GetBlockRealize(block_sref);
  Array<tir::StmtSRef> loops = tir::GetLoops(block_sref);
  const Array<PrimExpr>& bindings = realize->iter_values;
  if (loops.size() != bindings.size()) {
    return false;
  }
  int n = loops.size();
  for (int i = 0; i < n; ++i) {
    const PrimExpr& bind = bindings[i];
    const ForNode* loop = TVM_SREF_TO_FOR(loop, loops[i]);
    if (bind.as<tir::VarNode>() != loop->loop_var.get()) {
      return false;
    }
  }
  return true;
}

bool IsSubrootBlock(const tir::ScheduleState& self, const tir::StmtSRef& block_sref) {
  tir::StmtSRef parent_block_sref = GetScopeRoot(block_sref).value();
  return parent_block_sref->parent == nullptr;
}

bool IsLeafBlock(const tir::ScheduleState& self, const tir::StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  bool no_child = true;
  tir::PreOrderVisit(block->body, [&no_child](const ObjectRef& obj) -> bool {
    if (!no_child) {
      return false;
    }
    if (obj->IsInstance<tir::BlockNode>()) {
      no_child = false;
      return false;
    }
    return true;
  });
  return no_child;
}

Array<Integer> GetBlockVarTypes(const tir::ScheduleState& self, const tir::StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  Array<Integer> result;
  for (const tir::IterVar& iter_var : block->iter_vars) {
    int iter_type = iter_var->iter_type;
    result.push_back(iter_type);
  }
  return result;
}

bool IsSpatial(const tir::ScheduleState& self, const tir::StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  for (const tir::IterVar& iter_var : block->iter_vars) {
    if (iter_var->iter_type != tir::IterVarType::kDataPar) {
      return false;
    }
  }
  return true;
}

bool IsOutputBlock(const tir::ScheduleState& self, const tir::StmtSRef& block_sref) {
  tir::StmtSRef parent_sref = tir::GetScopeRoot(block_sref).value();
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  const BlockNode* parent = TVM_SREF_TO_BLOCK(parent, parent_sref);
  if (parent_sref->parent == nullptr) {
    const tir::PrimFuncNode* func = tir::GetRootPrimFunc(self, parent_sref);
    for (const tir::BufferRegion& write : block->writes) {
      for (const auto& kv : func->buffer_map) {
        if (write->buffer.get() == kv.second.get()) {
          return true;
        }
      }
    }
  } else {
    for (const tir::BufferRegion& write : block->writes) {
      for (const tir::BufferRegion& parent_write : parent->writes) {
        if (write->buffer.get() == parent_write->buffer.get()) {
          return true;
        }
      }
    }
  }
  return false;
}

int CountOp(const tir::ScheduleState& self, const tir::StmtSRef& block_sref, const Op& op) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  int count = 0;
  tir::PostOrderVisit(block->body, [&count, &op](const ObjectRef& obj) {
    if (const auto* call = obj.as<tir::CallNode>()) {
      if (call->op.same_as(op)) {
        ++count;
      }
    }
  });
  return count;
}

bool HasBranch(const tir::ScheduleState& self, const tir::StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  bool has_branch = false;
  arith::Analyzer analyzer;
  auto f_visit = [&has_branch, &analyzer](const ObjectRef& obj) -> bool {
    if (has_branch) {
      // stop visiting
      return false;
    }
    if (const auto* realize = obj.as<tir::BlockRealizeNode>()) {
      // Case 1: BlockRealize
      if (!analyzer.CanProve(realize->predicate == 1)) {
        has_branch = true;
      }
    } else if (obj->IsInstance<tir::IfThenElseNode>() || obj->IsInstance<tir::SelectNode>()) {
      // Case 2: IfThenElse / Select
      has_branch = true;
    } else if (const auto* call = obj.as<tir::CallNode>()) {
      // Case 3: Call
      static const Op& op_if_then_else = Op::Get("tir.if_then_else");
      if (call->op.same_as(op_if_then_else)) {
        has_branch = true;
      }
    }
    return !has_branch;
  };
  tir::PreOrderVisit(tir::GetBlockRealize(block_sref), f_visit);
  return has_branch;
}

Optional<Array<Bool>> GetReadPattern(const Array<tir::IterVar>& block_vars,
                                     const Array<PrimExpr>& read_axes) {
  // Maps a block var to its index
  std::unordered_map<const tir::VarNode*, int> block_var_to_idx;
  for (const tir::IterVar& iter_var : block_vars) {
    if (iter_var->iter_type == tir::IterVarType::kDataPar) {
      int index = block_var_to_idx.size();
      block_var_to_idx[iter_var->var.get()] = index;
    }
  }
  bool surjective = true;
  bool injective = true;
  bool ordered = true;
  // `read_which_block_var[i] = j` maps non-constant read-axis[i] to block_var[j]
  std::vector<int> read_which_block_var;
  // Number of times that a block var is mapped to
  std::vector<int> block_var_mapped_times(block_var_to_idx.size(), 0);
  // Enumerate each index, collect the read axis -> block var mapping info
  for (const PrimExpr& idx : read_axes) {
    if (IsConstInt(idx)) {
      continue;
    }
    // Check if it matches a block var
    if (Optional<tir::Var> opt_var = IsVarPlusMinusConst(idx)) {
      tir::Var var = opt_var.value();
      if (block_var_to_idx.count(var.get())) {
        int index = block_var_to_idx.at(var.get());
        read_which_block_var.push_back(index);
        ++block_var_mapped_times[index];
        continue;
      }
    }
    // If not, the mapping does not exist
    return NullOpt;
  }
  // Check `block_var_mapped_times` to determine if the mapping is injective and surjective
  for (int times : block_var_mapped_times) {
    // If there is a block var that doesn't have corresponding any read axis
    if (times == 0) {
      surjective = false;
    }
    // If there is a block var that has more than 2 corresponding load axes
    if (times >= 2) {
      injective = false;
    }
  }
  // Check `read_which_block_var` to determine if the mapping is in order
  for (size_t i = 1; i < read_which_block_var.size(); ++i) {
    if (read_which_block_var[i - 1] > read_which_block_var[i]) {
      ordered = false;
      break;
    }
  }
  return Array<Bool>{Bool(surjective), Bool(injective), Bool(ordered)};
}

bool IsElementWiseMatch(const tir::ScheduleState& self, const tir::StmtSRef& producer_sref,
                        const tir::StmtSRef& consumer_sref) {
  // Assume consumer is the only consumer of the producer
  tir::StmtSRef parent_sref = tir::GetScopeRoot(producer_sref).value();
  const BlockNode* producer = TVM_SREF_TO_BLOCK(producer, producer_sref);
  const BlockNode* consumer = TVM_SREF_TO_BLOCK(consumer, consumer_sref);
  if (producer->writes.empty()) {
    return false;
  }
  // Cond 1: size of the read/write regions match
  std::unordered_set<const tir::BufferNode*> buffer_produced;
  {
    std::vector<tir::BufferRegion> producer_reads, producer_writes;
    std::vector<tir::BufferRegion> consumer_reads, consumer_writes;
    tir::RelaxRegion(producer_sref, parent_sref, &producer_reads, &producer_writes);
    tir::RelaxRegion(consumer_sref, parent_sref, &consumer_reads, &consumer_writes);
    const Array<Range>& region = producer_writes.at(0)->region;
    // Cond 1.1: check all producer's write regions share the same shape
    for (const tir::BufferRegion& write : producer_writes) {
      buffer_produced.insert(write->buffer.get());
      if (!DomainEqual(write->region, region)) {
        return false;
      }
    }
    // Cond 1.2: check all consumer's write regions share the same shape
    for (const tir::BufferRegion& write : consumer_writes) {
      if (!DomainEqual(write->region, region)) {
        return false;
      }
    }
    // Cond 1.3: check if the consumer reads the entire region the producer produces
    for (const tir::BufferRegion& write : producer_writes) {
      for (const tir::BufferRegion& read : consumer_reads) {
        if (write->buffer.get() == read->buffer.get()) {
          if (!DomainEqual(write->region, read->region)) {
            return false;
          }
        }
      }
    }
  }
  // Cond 2: The read is elementwise
  const Array<tir::IterVar>& block_vars = consumer->iter_vars;
  for (const tir::BufferRegion& read : consumer->reads) {
    if (!buffer_produced.count(read->buffer.get())) {
      continue;
    }
    Array<PrimExpr> read_axes;
    read_axes.reserve(read->region.size());
    for (const Range& range : read->region) {
      if (IsConstInt(range->extent)) {
        read_axes.push_back(range->min);
      } else {
        return false;
      }
    }
    if (Optional<Array<Bool>> access = GetReadPattern(block_vars, read_axes)) {
      ICHECK_EQ(access.value().size(), 3);
      bool surjective = access.value()[0];
      bool injective = access.value()[1];
      bool order = access.value()[2];
      if (!surjective || !injective || !order) {
        return false;
      }
    }
  }
  // TODO(@junrushao1994): examine region cover here or defer to TIR compute_at?
  return true;
}

bool NeedsMultiLevelTiling(const tir::ScheduleState& self, const tir::StmtSRef& block_sref) {
  // Right now it only works with trivial binding
  if (!IsTrivialBinding(self, block_sref)) {
    return false;
  }
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  // Assume complete/reduction block
  if (block->writes.size() != 1) {
    return false;
  }
  if (block->reads.empty()) {
    return false;
  }
  std::vector<int> n_missing_block_vars;
  for (const tir::BufferRegion& region : block->reads) {
    int n_missing = 0;
    std::unordered_set<const tir::VarNode*> vars_in_load;
    for (const Range& range : region->region) {
      if (!IsConstInt(range->extent)) {
        return false;
      }
      tir::PostOrderVisit(range->min, [&vars_in_load](const ObjectRef& obj) {
        if (const auto* var = obj.as<tir::VarNode>()) {
          vars_in_load.insert(var);
        }
      });
    }
    for (const tir::IterVar& block_var : block->iter_vars) {
      if (block_var->iter_type == tir::IterVarType::kDataPar) {
        if (!vars_in_load.count(block_var->var.get())) {
          ++n_missing;
        }
      }
    }
    n_missing_block_vars.push_back(n_missing);
  }
  bool is_spatial = IsSpatial(self, block_sref);
  int min_n_missing = *std::min_element(n_missing_block_vars.begin(), n_missing_block_vars.end());
  int sum_n_missing = std::accumulate(n_missing_block_vars.begin(), n_missing_block_vars.end(), 0);
  if (is_spatial && min_n_missing == 0) {
    return false;
  }
  if (sum_n_missing >= 2) {
    return true;
  }
  if (sum_n_missing == 0) {
    return false;
  }
  return !IsSpatial(self, block_sref);
}

bool IsStrictlyInlineable(const tir::ScheduleState& self, const tir::StmtSRef& block_sref) {
  static const Op& op_tir_exp = Op::Get("tir.exp");
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  // Const tensors are strictly inlineable
  if (block->reads.empty()) {
    return true;
  }

  if (HasBranch(self, block_sref)) {
    return false;
  }
  if (CountOp(self, block_sref, op_tir_exp)) {
    return false;
  }
  // Check if it is ordered-injective mapping
  for (const tir::BufferRegion& region : block->reads) {
    Array<PrimExpr> read_axes;
    read_axes.reserve(region->region.size());
    for (const Range& range : region->region) {
      if (!IsConstInt(range->extent)) {
        return false;
      } else {
        read_axes.push_back(range->min);
      }
    }
    if (Optional<Array<Bool>> access = GetReadPattern(block->iter_vars, read_axes)) {
      ICHECK_EQ(access.value().size(), 3);
      bool injective = access.value()[1];
      bool order = access.value()[2];
      if (!order || !injective) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

class AutoTensorizeComparator : public tir::TensorizeComparator {
 public:
  AutoTensorizeComparator() : tir::TensorizeComparator(false) {}

  bool VisitStmt(const tir::Stmt& n, const tir::Stmt& rhs) override {
    if (n.same_as(rhs)) return true;
    tir::Stmt lhs = n;
    if (lhs->type_index() != rhs->type_index()) {
      return false;
    }
    bool equal = tir::StmtComparator::VisitStmt(lhs, rhs);
    ICHECK(equal || !assert_mode_) << "Statements are not matching between:\n"
                                   << n << "\nand\n"
                                   << rhs;
    return equal;
  }

  bool CompareBuffer(const tir::Buffer& lhs, const tir::Buffer& rhs) override {
    if (lhs.same_as(rhs)) return true;
    // Remap both buffer itself and buffer data
    // Skip buffer shape
    bool equal = DefEqual(lhs, rhs) && DefEqual(lhs->data, rhs->data) &&
                 lhs->buffer_type == rhs->buffer_type && CompareType(lhs->dtype, rhs->dtype);
    if (equal) rhs_buffer_map_[rhs] = lhs;
    return equal;
  }
};

Optional<TensorizeInfo> GetTensorizeLoopMapping(const tir::ScheduleState& self,
                                                const tir::StmtSRef& block_sref,
                                                const tir::PrimFunc& desc_func) {
  // Try to do tiling automatically if possible
  // Now the heuristic is that if block's block var binding is constant + loop var,
  // in other words, with tir.block(..., vi=Ci+i, vj=Cj+j, vk=Ck+k), then we split and reorder
  // i, j, k according to the loops outside desc_block
  // Collect the loops outside block
  arith::Analyzer analyzer;
  const tir::BlockRealize& block = tir::GetBlockRealize(block_sref);
  // Step 1. Analyze desc_func, extract its block, loops and loop vars
  const tir::BlockRealizeNode* desc_block = nullptr;
  std::vector<const tir::ForNode*> desc_loops;
  std::unordered_set<const tir::VarNode*> desc_loop_vars;
  {
    auto f_visit = [&desc_block, &desc_loops, &desc_loop_vars,
                    &analyzer](const ObjectRef& obj) -> bool {
      // Extract the block
      if (const auto* block = obj.as<tir::BlockRealizeNode>()) {
        desc_block = block;
        return false;
      }
      // Extract the loops
      if (const auto* loop = obj.as<tir::ForNode>()) {
        desc_loops.push_back(loop);
        desc_loop_vars.insert(loop->loop_var.get());
        if (!analyzer.CanProve(loop->min == 0)) {
          return false;
        }
      }
      return true;
    };
    const auto* desc_body =
        Downcast<tir::BlockRealize>(desc_func->body)->block->body.as<tir::BlockRealizeNode>();
    ICHECK(desc_body);
    tir::PostOrderVisit(desc_body->block->body, f_visit);
    std::reverse(desc_loops.begin(), desc_loops.end());
    ICHECK(desc_block);
  }
  // Step 2. Check if `desc_block` matches `block`
  // Ignore the scope of buffers when comparing, since we can do cache_read/write
  if (!AutoTensorizeComparator().VisitStmt(block, GetRef<tir::BlockRealize>(desc_block))) {
    return NullOpt;
  }
  // Step 3. Extract the loops on top of the block. It is a mirror step of Step 1
  std::vector<const tir::ForNode*> block_loops;
  std::unordered_set<const tir::VarNode*> block_loop_vars;
  {
    for (const tir::StmtSRefNode* loop_sref = block_sref->parent;; loop_sref = loop_sref->parent) {
      const auto* loop = loop_sref->StmtAs<tir::ForNode>();
      if (loop == nullptr || loop->body->IsInstance<tir::SeqStmtNode>()) {
        break;
      }
      block_loops.push_back(loop);
      block_loop_vars.insert(loop->loop_var.get());
      if (!analyzer.CanProve(loop->min == 0)) {
        return NullOpt;
      }
    }
    std::reverse(block_loops.begin(), block_loops.end());
  }
  // Step 4. Map from block loops to desc block loops
  ObjectPtr<TensorizeInfoNode> ret = make_object<TensorizeInfoNode>();
  int n_block_vars = block->iter_values.size();
  int n_desc_vars = desc_block->iter_values.size();
  int offset = n_block_vars - n_desc_vars;
  if (offset < 0) {
    return NullOpt;
  }
  // We align the block and desc block's bindings from the right side
  // block     (v0=..., v1=..., v2=...)
  //                    ^ i_block
  // desc_block(        v1=..., v2=...)
  //                    ^ i_desc
  for (int i_desc = 0, i_block = offset; i_desc < n_desc_vars; ++i_desc, ++i_block) {
    // For each block var binding, we find
    const PrimExpr& block_bind = block->iter_values[i_block];
    const PrimExpr& desc_bind = desc_block->iter_values[i_desc];
    // Step 4.1. Find the corresponding loop of the i-th block var of block
    const tir::ForNode* block_loop = nullptr;
    for (int i = 0, n = block_loops.size(); i < n; ++i) {
      // Check if block_bind = block_loops[i]->loop_var + stuff-irrelevant-of-loop-vars
      PrimExpr r = analyzer.Simplify(block_bind - block_loops[i]->loop_var);
      if (!tir::StmtExprContainsVar(r, block_loop_vars)) {
        block_loop = block_loops[i];
        break;
      }
    }
    if (block_loop == nullptr) {
      return NullOpt;
    }
    // Step 4.2. Find the corresponding loop of the i-th block var of desc
    const tir::ForNode* desc_loop = nullptr;
    for (int i = 0, n = desc_loops.size(); i < n; ++i) {
      // Check if desc_bind = loops[i]->loop_var + stuff-irrelevant-of-loop-vars
      PrimExpr r = analyzer.Simplify(desc_bind - desc_loops[i]->loop_var);
      if (!tir::StmtExprContainsVar(r, desc_loop_vars)) {
        desc_loop = desc_loops[i];
        break;
      }
    }
    if (block_loop == nullptr) {
      return NullOpt;
    }
    // Step 4.3. Check divisibility of loop extents
    PrimExpr block_extent = analyzer.Simplify(block_loop->extent);
    PrimExpr desc_extent = analyzer.Simplify(desc_loop->extent);
    if (const auto* int_block_extent = block_extent.as<IntImmNode>()) {
      if (const auto* int_desc_extent = desc_extent.as<IntImmNode>()) {
        if (int_block_extent->value % int_desc_extent->value != 0) {
          return NullOpt;
        }
      } else {
        return NullOpt;
      }
    } else {
      return NullOpt;
    }
    // Step 4.4. Maps the result of Step 4.1 to Step 4.2
    const tir::StmtSRef& block_loop_sref = self->stmt2ref[block_loop];
    auto it = ret->loop_map.find(block_loop_sref);
    if (it == ret->loop_map.end()) {
      ret->loop_map.Set(block_loop_sref, GetRef<tir::For>(desc_loop));
    } else if ((*it).second.get() != desc_loop) {
      return NullOpt;
    }
  }
  for (int i = 0, n = desc_loops.size(); i < n; ++i) {
    ret->desc_loop_indexer.Set(GetRef<tir::For>(desc_loops[i]), Integer(i));
  }
  return TensorizeInfo(ret);
}

double CountFlop(const tir::PrimFunc& func) {
  struct TResult {
    using TTable = std::unordered_map<int32_t, double>;

    TResult() = default;

    explicit TResult(const tvm::DataType& dtype) { Add(dtype); }

    void Add(const tvm::DataType& dtype) { data_[DataType2Int(dtype)] += 1; }

    TResult operator+=(const TResult& rhs) {
      for (const auto& kv : rhs.data_) {
        data_[kv.first] += kv.second;
      }
      return *this;
    }

    TResult operator*=(int64_t rhs) {
      for (auto& kv : data_) {
        kv.second *= rhs;
      }
      return *this;
    }

    TResult MaxWith(const TResult& rhs) {
      for (const auto& kv : rhs.data_) {
        double& v = data_[kv.first];
        if (v < kv.second) {
          v = kv.second;
        }
      }
      return *this;
    }

    struct DType {
      uint8_t code : 8;
      uint8_t bits : 8;
      uint16_t lanes : 16;
    };
    static_assert(sizeof(DType) == 4, "Incorrect size of DType");

    static String Int2Str(int32_t dtype) {
      union {
        DType dst;
        int32_t src;
      } converter;
      converter.src = dtype;
      static std::string type_code_tab[] = {"int", "uint", "float", "handle", "bfloat"};
      std::ostringstream os;
      os << type_code_tab[converter.dst.code];
      os << static_cast<int>(converter.dst.bits);
      if (converter.dst.lanes != 1) {
        os << "x" << static_cast<int>(converter.dst.lanes);
      }
      return os.str();
    }

    static int32_t DataType2Int(const tvm::DataType& dtype) {
      union {
        DType src;
        int32_t dst;
      } converter;
      converter.src.code = dtype.code();
      converter.src.bits = dtype.bits();
      converter.src.lanes = dtype.lanes();
      return converter.dst;
    }

    TTable data_;
  };

  class FlopCounter : public tir::ExprFunctor<TResult(const PrimExpr& n)>,
                      public tir::StmtFunctor<TResult(const tir::Stmt& n)> {
   public:
    ~FlopCounter() {}

    TResult VisitExpr(const PrimExpr& expr) override { return ExprFunctor::VisitExpr(expr); }
    TResult VisitStmt(const tir::Stmt& stmt) override { return StmtFunctor::VisitStmt(stmt); }

    TResult VisitStmt_(const tir::IfThenElseNode* branch) override {
      TResult cond = VisitExpr(branch->condition);
      cond += VisitStmt(branch->then_case).MaxWith(VisitStmt(branch->else_case));
      return cond;
    }

    TResult VisitStmt_(const tir::BufferStoreNode* store) override {
      TResult result = VisitExpr(store->value);
      for (const PrimExpr& e : store->indices) {
        result += VisitExpr(e);
      }
      return result;
    }

    TResult VisitStmt_(const tir::SeqStmtNode* seq) override {
      TResult result;
      for (const tir::Stmt& stmt : seq->seq) {
        result += VisitStmt(stmt);
      }
      return result;
    }

    TResult VisitStmt_(const tir::BlockRealizeNode* block) override {
      return VisitStmt(block->block->body);
    }

    TResult VisitStmt_(const tir::BlockNode* block) override {
      TResult result;
      if (block->init.defined()) {
        result += VisitStmt(block->init.value());
      }
      result += VisitStmt(block->body);
      return result;
    }

    TResult VisitStmt_(const tir::ForNode* loop) override {
      TResult result = VisitStmt(loop->body);
      const auto* int_imm = loop->extent.as<IntImmNode>();
      ICHECK(int_imm) << "TypeError: Expect the extent of a loop to be IntImm, but gets: "
                      << loop->extent->GetTypeKey();
      result *= int_imm->value;
      return result;
    }

#define TVM_META_SCHEDULE_VISIT_BINARY(Node) \
  TResult VisitExpr_(const Node* op) final { \
    TResult result(op->dtype);               \
    result += VisitExpr(op->a);              \
    result += VisitExpr(op->b);              \
    return result;                           \
  }
    TVM_META_SCHEDULE_VISIT_BINARY(tir::AddNode);
    TVM_META_SCHEDULE_VISIT_BINARY(tir::SubNode);
    TVM_META_SCHEDULE_VISIT_BINARY(tir::MulNode);
    TVM_META_SCHEDULE_VISIT_BINARY(tir::DivNode);
    TVM_META_SCHEDULE_VISIT_BINARY(tir::ModNode);
    TVM_META_SCHEDULE_VISIT_BINARY(tir::FloorDivNode);
    TVM_META_SCHEDULE_VISIT_BINARY(tir::FloorModNode);
    TVM_META_SCHEDULE_VISIT_BINARY(tir::MinNode);
    TVM_META_SCHEDULE_VISIT_BINARY(tir::MaxNode);
    TVM_META_SCHEDULE_VISIT_BINARY(tir::EQNode);
    TVM_META_SCHEDULE_VISIT_BINARY(tir::NENode);
    TVM_META_SCHEDULE_VISIT_BINARY(tir::LTNode);
    TVM_META_SCHEDULE_VISIT_BINARY(tir::LENode);
    TVM_META_SCHEDULE_VISIT_BINARY(tir::GTNode);
    TVM_META_SCHEDULE_VISIT_BINARY(tir::GENode);
    TVM_META_SCHEDULE_VISIT_BINARY(tir::AndNode);
    TVM_META_SCHEDULE_VISIT_BINARY(tir::OrNode);
#undef TVM_META_SCHEDULE_VISIT_BINARY
    TResult VisitExpr_(const tir::CastNode* op) override { return VisitExpr(op->value); }
    TResult VisitExpr_(const tir::VarNode* op) override { return TResult(); }
    TResult VisitExpr_(const tir::SizeVarNode* op) override { return TResult(); }
    TResult VisitExpr_(const tir::BufferLoadNode* op) override { return TResult(); }
    TResult VisitExpr_(const IntImmNode* op) override { return TResult(); }
    TResult VisitExpr_(const FloatImmNode* op) override { return TResult(); }

    TResult VisitExpr_(const tir::NotNode* op) override {
      TResult result(op->dtype);
      result += VisitExpr(op->a);
      return result;
    }

    TResult VisitExpr_(const tir::SelectNode* op) override {
      TResult cond = VisitExpr(op->condition);
      cond += VisitExpr(op->true_value).MaxWith(VisitExpr(op->false_value));
      return cond;
    }

    TResult VisitExpr_(const tir::CallNode* op) override {
      TResult ret;
      for (const auto& x : op->args) {
        ret += VisitExpr(x);
      }
      return ret;
    }
  };
  if (auto flop_count_expr = func->GetAttr<PrimExpr>("flop_ct")) {
    arith::Analyzer analyzer;
    PrimExpr flop_count = analyzer.Simplify(flop_count_expr.value());
    if (const auto* flop_count_imm = flop_count.as<IntImmNode>()) {
      return static_cast<double>(flop_count_imm->value);
    } else {
      LOG(FATAL) << "ValueError: Unable to evaluate flop count";
    }
  }
  TResult result = FlopCounter().VisitStmt(func->body);
  double cnt = 0.0;
  int i32 = TResult::DataType2Int(tvm::DataType::Int(32));
  int i64 = TResult::DataType2Int(tvm::DataType::Int(64));
  int u1 = TResult::DataType2Int(tvm::DataType::UInt(1));
  for (const auto& kv : result.data_) {
    if (kv.first != i32 && kv.first != i64 && kv.first != u1) {
      cnt += kv.second;
    }
  }
  return cnt;
}

std::pair<int64_t, int64_t> GetCumulativeSpaceAndReductionLength(const tir::ScheduleState& self,
                                                                 const tir::StmtSRef& block_sref) {
  Array<tir::StmtSRef> loops = tir::GetLoops(block_sref);
  int64_t cum_space_len = 1, cum_reduce_len = 1;
  /*
   * Return (-1, -1) if
   *   1. there is some loop with type other than kDataPar and kCommReduce;
   *   2. there is some loop which is dynamic.
   */
  for (const tir::StmtSRef& loop_sref : loops) {
    tir::IterVarType type = GetLoopIterType(self, loop_sref);
    if (type == tir::kDataPar) {
      int64_t extent = GetLoopIntExtent(loop_sref);
      if (extent != -1) {
        cum_space_len *= extent;
      } else {
        return std::make_pair(-1, -1);
      }
    } else if (type == tir::kCommReduce) {
      int64_t extent = GetLoopIntExtent(loop_sref);
      if (extent != -1) {
        cum_reduce_len *= extent;
      } else {
        return std::make_pair(-1, -1);
      }
    } else {
      return std::make_pair(-1, -1);
    }
  }
  return std::make_pair(cum_space_len, cum_reduce_len);
}

bool NeedsRFactorOrCrossThreadReduction(const tir::ScheduleState& self,
                                        const tir::StmtSRef& block_sref,
                                        int64_t max_parallel_extent,
                                        int64_t basic_parallel_extent) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  Array<tir::StmtSRef> loops = tir::GetLoops(block_sref);

  // Cond 1. The block is a reduction block and has trivial binding.
  if (ReductionBlock(self, block_sref, GetScopeRoot(block_sref).value()) &&
      !IsTrivialBinding(self, block_sref)) {
    return false;
  }

  // Cond 2. Every the loop axis must be either spatial axis or reduction axis.
  for (const tir::StmtSRef& loop_sref : loops) {
    const tir::IterVarType& type = GetLoopIterType(self, loop_sref);
    if (type != tir::kDataPar && type != tir::kCommReduce) {
      return false;
    }
  }

  // Cond 3. Whether there is at least one reduction loop.
  // Cond 4. The loops are continuous, and the body of the innermost loop is exactly the block.
  bool has_reduction_loop = false;
  for (int i = 0; i < static_cast<int>(loops.size()); ++i) {
    // Cond 3.
    if (GetLoopIterType(self, loops[i]) == tir::kCommReduce) {
      has_reduction_loop = true;
    }

    // Cond 4.
    const ForNode* loop_i = TVM_SREF_TO_FOR(loop_i, loops[i]);
    if (i < static_cast<int>(loops.size()) - 1) {
      const ForNode* loop_i1 = TVM_SREF_TO_FOR(loop_i1, loops[i + 1]);
      if (loop_i->body.get() != loop_i1) {
        return false;
      }
    } else {
      const auto* block_realize = loop_i->body.as<tir::BlockRealizeNode>();
      if (!block_realize || block_realize->block.get() != block) {
        return false;
      }
    }
  }
  if (!has_reduction_loop) {
    return false;
  }

  // Cond 5. Can successfully calculating the cumulative loop length.
  int64_t cum_space_len, cum_reduce_len;
  std::tie(cum_space_len, cum_reduce_len) = GetCumulativeSpaceAndReductionLength(self, block_sref);
  if (cum_space_len == -1 || cum_reduce_len == -1) {
    return false;
  }

  // Cond 6.
  if (NeedsMultiLevelTiling(self, block_sref)) {
    // Do not use rfactor/cross-thread-reduction if we have enough parallelism on spatial loops.
    if (cum_space_len >= cum_reduce_len || cum_space_len > max_parallel_extent) {
      return false;
    } else {
      return true;
    }
  } else if (cum_reduce_len > 1) {
    // Always try rfactor/cross-thread-reduction for other reduction blocks.
    return cum_reduce_len > basic_parallel_extent;
  }

  return false;
}

bool HasCacheWriteBlock(const Schedule& sch, const BlockRV& block_rv, const int& i) {
  static tir::InstructionKind cache_write = tir::InstructionKind::Get("CacheWrite");
  for (const Instruction& inst : sch->trace().value()->insts) {
    if (inst->kind.same_as(cache_write)) {
      CHECK_EQ(inst->inputs.size(), 1);
      const BlockRV& input_rv = Downcast<BlockRV>(inst->inputs[0]);
      if (block_rv.same_as(input_rv)) {
        if (Downcast<Integer>(inst->attrs[0])->value == i) {
          return true;
        }
      }
    }
  }
  return false;
}

bool IsConstShiftEqual(const tir::Var& var, const PrimExpr& expr) {
  arith::PVar<PrimExpr> x;
  arith::PVar<IntImm> c;

  if (((x + c).Match(expr) || (x - c).Match(expr) || (c + x).Match(expr) || x.Match(expr)) &&
      x.Eval().same_as(var)) {
    return true;
  }
  return false;
}

int GetNumberCommonOutputDims(const tir::Block& producer, const tir::Block& consumer) {
  // The producer and consumer are required to both have only one output buffer.
  if (producer->writes.size() != 1 || consumer->writes.size() != 1) {
    return 0;
  }

  const tir::Buffer& buf_producer = producer->writes[0]->buffer;
  const tir::Buffer& buf_consumer = consumer->writes[0]->buffer;
  const Array<PrimExpr>& shape_buf_producer = buf_producer->shape;
  const Array<PrimExpr>& shape_buf_consumer = buf_consumer->shape;
  // Step 1. Collect the BufferLoads with source buffer `buf_producer` and the BufferStores with
  // destination buffer `buf_consumer` in the consumer block.
  Array<tir::BufferLoad> buffer_loads;
  Array<tir::BufferStore> buffer_stores;
  tir::PreOrderVisit(consumer->body, [&buf_producer, &buf_consumer, &buffer_loads,
                                      &buffer_stores](const ObjectRef& node) {
    if (const auto* buffer_load = node.as<tir::BufferLoadNode>()) {
      if (buffer_load->buffer.same_as(buf_producer)) {
        buffer_loads.push_back(GetRef<tir::BufferLoad>(buffer_load));
        return false;
      }
    }
    if (const auto* buffer_store = node.as<tir::BufferStoreNode>()) {
      if (buffer_store->buffer.same_as(buf_consumer)) {
        buffer_stores.push_back(GetRef<tir::BufferStore>(buffer_store));
        // Do not return, since there might be BufferLoads in the body of a BufferStore.
      }
    }
    return true;
  });
  // Step 2. Collect the block vars of the consumer block.
  std::unordered_set<const tir::VarNode*> consumer_iter_vars;
  for (const tir::IterVar& iter_var : consumer->iter_vars) {
    consumer_iter_vars.insert(iter_var->var.get());
  }
  // Step 3. Let `n_common` grow up from 0. For each `n_common`, check whether this dimension is a
  // "common" one. If not, we terminate the loop.
  arith::Analyzer analyzer;
  int n_common;
  for (n_common = 0; n_common < static_cast<int>(shape_buf_producer.size()) &&
                     n_common < static_cast<int>(shape_buf_consumer.size());
       ++n_common) {
    // If the shapes of the two buffers don't match, this dimension cannot be common.
    if (!tir::is_zero(
            analyzer.Simplify(shape_buf_producer[n_common] - shape_buf_consumer[n_common]))) {
      break;
    }
    // Collect the vars in the indices of the BufferStores.
    std::vector<tir::Var> vars;
    for (const tir::BufferStore& buffer_store : buffer_stores) {
      tir::PreOrderVisit(buffer_store->indices[n_common], [&vars](const ObjectRef& node) {
        if (const auto* var = node.as<tir::VarNode>()) {
          vars.push_back(GetRef<tir::Var>(var));
          return false;
        }
        return true;
      });
    }
    // If this dimension is a common dimension, there should be only one variable in the index.
    // Furthermore, this variable should be the corresponding variable of one consumer's block var.
    if (vars.size() != 1 || !consumer_iter_vars.count(vars[0].get())) {
      break;
    }
    // Check whether the indices of BufferLoads and BufferStores are of const shift pattern.
    bool const_shift = true;
    for (const tir::BufferStore& buffer_store : buffer_stores) {
      if (!IsConstShiftEqual(vars[0], buffer_store->indices[n_common])) {
        const_shift = false;
        break;
      }
    }
    if (!const_shift) {
      break;
    }
    for (const tir::BufferLoad& buffer_load : buffer_loads) {
      if (!IsConstShiftEqual(vars[0], buffer_load->indices[n_common])) {
        const_shift = false;
        break;
      }
    }
    if (!const_shift) {
      break;
    }
  }
  return n_common;
}

TVM_REGISTER_NODE_TYPE(TensorizeInfoNode);

TVM_REGISTER_GLOBAL("meta_schedule.analysis.IsTrivialBinding").set_body_typed(IsTrivialBinding);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.IsSubrootBlock").set_body_typed(IsSubrootBlock);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.IsLeafBlock").set_body_typed(IsLeafBlock);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.GetLoopIterType").set_body_typed(tir::GetLoopIterType);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.GetBlockVarTypes").set_body_typed(GetBlockVarTypes);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.IsSpatial").set_body_typed(IsSpatial);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.IsOutputBlock").set_body_typed(IsOutputBlock);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.CountOp").set_body_typed(CountOp);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.HasBranch").set_body_typed(HasBranch);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.IsElementWiseMatch").set_body_typed(IsElementWiseMatch);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.NeedsMultiLevelTiling")
    .set_body_typed(NeedsMultiLevelTiling);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.IsStrictlyInlineable")
    .set_body_typed(IsStrictlyInlineable);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.GetTensorizeLoopMapping")
    .set_body_typed(GetTensorizeLoopMapping);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.CountFlop").set_body_typed(CountFlop);

}  // namespace meta_schedule
}  // namespace tvm
