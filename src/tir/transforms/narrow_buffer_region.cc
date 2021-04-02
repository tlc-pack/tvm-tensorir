/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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

/*!
 * \file buffer_flatten.cc
 * \brief Narrow the buffer size into its exactly need.
 */

#include <tvm/arith/int_set.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/analysis.h>

namespace tvm {
namespace tir {

template <class K, class V>
using SMap = std::unordered_map<K, V, ObjectPtrHash, ObjectPtrEqual>;
template <class K>
using SSet = std::unordered_set<K, ObjectPtrHash, ObjectPtrEqual>;

using NDIntSet = std::vector<arith::IntSet>;

inline bool StrStartsWith(const String& str, const String& prefix) {
  int n = prefix.size();
  if (static_cast<int>(str.size()) < n) {
    return false;
  }
  const char* data = str.data();
  return std::equal(data, data + n, prefix.data());
}

arith::IntSet IntSetFromMinExtent(const PrimExpr& min, const PrimExpr& extent) {
  return arith::IntSet::FromRange(Range::FromMinExtent(min, extent));
}

void NDIntSetUnionWith(NDIntSet* lhs, const NDIntSet& rhs) {
  ICHECK_EQ(lhs->size(), rhs.size());
  int ndim = rhs.size();
  for (int i = 0; i < ndim; ++i) {
    arith::IntSet& int_set = lhs->at(i);
    int_set = arith::Union({int_set, rhs.at(i)});
  }
}

Array<Range> NDIntSet2Region(const NDIntSet& nd_int_set) {
  Integer one(1);
  Array<Range> result;
  result.reserve(nd_int_set.size());
  for (const arith::IntSet& int_set : nd_int_set) {
    PrimExpr min = int_set.min();
    PrimExpr max = int_set.max();
    result.push_back(Range(/*begin=*/min, /*end=*/max + one));
  }
  return result;
}

NDIntSet NDIntSetFromShape(const Array<PrimExpr>& shape) {
  NDIntSet result;
  for (const PrimExpr& extent : shape) {
    result.push_back(IntSetFromMinExtent(Integer(0), extent));
  }
  return result;
}

NDIntSet NDIntSetEmpty(int ndim) {
  return std::vector<arith::IntSet>(ndim, arith::IntSet::Nothing());
}

bool IsThreadBound(const For& loop) {
  if (loop->kind != ForKind::kThreadBinding) {
    return false;
  }
  ICHECK(loop->thread_binding.defined());
  std::string thread_tag = loop->thread_binding.value()->thread_tag;
  if (StrStartsWith(thread_tag, "threadIdx")) {
    return true;
  }
  if (StrStartsWith(thread_tag, "vthread")) {
    return true;
  }
  return false;
}



/*! \brief Helper class to mutate the buffer access. */
class BufferAccessRewriter : public StmtExprMutator {
 public:
  using FRewriteBufferAccess = std::function<void(Buffer* buffer, Array<PrimExpr>* indices)>;
  using FRewriteBufferRegion = std::function<void(Buffer* buffer, Region* region)>;

  static Stmt Rewrite(const Stmt& stmt, const FRewriteBufferAccess& f_access_rewrite,
                      const FRewriteBufferRegion& f_region_rewrite) {
    BufferAccessRewriter rewriter(f_access_rewrite, f_region_rewrite);
    return rewriter.VisitStmt(stmt);
  }

 private:
  explicit BufferAccessRewriter(const FRewriteBufferAccess& f_access_rewrite,
                                  const FRewriteBufferRegion& f_region_rewrite)
      : f_access_rewrite_(f_access_rewrite), f_region_rewrite_(f_region_rewrite) {}

  Stmt VisitStmt_(const BufferStoreNode* _op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_op));
    BufferStoreNode* op = store.CopyOnWrite();
    f_access_rewrite_(&op->buffer, &op->indices);
    return std::move(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* _op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_op));
    BufferLoadNode* op = load.CopyOnWrite();
    f_access_rewrite_(&op->buffer, &op->indices);
    return std::move(load);
  }

  Stmt VisitStmt_(const BlockNode* _op) final {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(_op));
    BlockNode* op = block.CopyOnWrite();
    auto f_rewrite_buffer_region =
        [this](const Array<BufferRegion>& regions) -> Array<BufferRegion> {
      Array<BufferRegion> new_regions;
      new_regions.reserve(regions.size());
      for (const auto& read : regions) {
        BufferRegion buffer_region = Downcast<BufferRegion>(read);
        BufferRegionNode* p = buffer_region.CopyOnWrite();
        f_region_rewrite_(&p->buffer, &p->region);
        new_regions.push_back(buffer_region);
      }
      return new_regions;
    };
    Array<BufferRegion> reads = f_rewrite_buffer_region(op->reads);
    Array<BufferRegion> writes = f_rewrite_buffer_region(op->writes);
    op->reads = std::move(reads);
    op->writes = std::move(writes);
    return std::move(block);
  }

  const FRewriteBufferAccess& f_access_rewrite_;
  const FRewriteBufferRegion& f_region_rewrite_;
};

/*! \brief Alloc the used region of each buffers. */
class BufferNarrower : public StmtExprMutator {
 public:
  static Stmt Narrow(const PrimFunc& f) {
    SMap<Buffer, BufferInfo> buffer_info;
    for (const auto& kv : f->buffer_map) {
      const Buffer& buffer = kv.second;
      BufferInfo info(buffer->shape.size());
      info.accessed_region = NDIntSetFromShape(buffer->shape);
      info.region = NDIntSet2Region(info.accessed_region);
      info.new_buffer = buffer;
      info.is_arg = true;
      buffer_info.emplace(buffer, info);
    }
    BufferNarrower narrower(std::move(buffer_info));
    Stmt stmt = narrower.VisitStmt(f->body);
    stmt = BufferAccessRewriter::Rewrite(
        /*stmt=*/stmt,
        /*f_access_rewrite=*/
        std::bind(&BufferNarrower::RewriteBufferAccess,  //
                  &narrower,                             //
                  std::placeholders::_1,                 //
                  std::placeholders::_2),
        /*f_region_rewrite=*/
        std::bind(&BufferNarrower::RewriteBufferRegion,  //
                  &narrower,                             //
                  std::placeholders::_1,                 //
                  std::placeholders::_2));
    return stmt;
  }

 private:
  struct BufferInfo {
    NDIntSet accessed_region;
    Array<Range> region;
    Buffer new_buffer;
    const ForNode* alloc_site;
    bool is_arg;

    explicit BufferInfo(int ndim)
        : accessed_region(NDIntSetEmpty(ndim)),
          region{nullptr},
          new_buffer{nullptr},
          alloc_site{nullptr},
          is_arg(false) {}
  };

  explicit BufferNarrower(SMap<Buffer, BufferInfo> buffer_info)
      : buffer_info_(std::move(buffer_info)),
        ancestor_loops_{},
        var_substitutes_{},
        reduction_loop_vars_{} {}

  Stmt VisitStmt_(const ForNode* loop) final {
    // Step 1. Handle block vars in `min` and `extent`
    PrimExpr min = this->VisitExpr(loop->min);
    PrimExpr extent = this->VisitExpr(loop->extent);
    // Step 3. Visit recursively
    ancestor_loops_.push_back(GetRef<For>(loop));
    Stmt body = this->VisitStmt(loop->body);
    ancestor_loops_.pop_back();
    // Step 5. Make the new loop
    if (loop->kind == ForKind::kThreadBinding && reduction_loop_vars_.count(loop->loop_var)) {
      // do nothing, because the loop is going to be removed
    } else {
      body = For(/*loop_var=*/loop->loop_var,
                 /*min=*/min,
                 /*extent=*/extent,
                 /*kind=*/loop->kind,
                 /*body=*/std::move(body),
                 /*thread_binding=*/loop->thread_binding,
                 /*annotations=*/loop->annotations);
    }
    return body;
  }

  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    const auto* block = realize->block.get();
    ICHECK(!block->init.defined());
    // Step 1. Update "block vars => loop vars" for substitution, add reduction loop vars
    ICHECK_EQ(block->iter_vars.size(), realize->iter_values.size());
    for (int i = 0, n = block->iter_vars.size(); i < n; ++i) {
      IterVar block_var = block->iter_vars[i];
      PrimExpr v = this->VisitExpr(realize->iter_values[i]);
      var_substitutes_.emplace(block_var->var, v);
      if (block_var->iter_type == kCommReduce) {
        for (const Var& var : Vars(v)) {
          this->reduction_loop_vars_.insert(var);
        }
      }
    }
    // Step 2. Update BufferInfo
    for (const Buffer& buffer : block->alloc_buffers) {
      BufferInfo info(buffer->shape.size());
      if (!ancestor_loops_.empty()) {
        info.alloc_site = ancestor_loops_[ancestor_loops_.size() - 1].get();
      } else {
        info.alloc_site = nullptr;
      }
      buffer_info_.emplace(buffer, info);
    }
    // Step 3. Visit recursively
    Stmt body = this->VisitStmt(block->body);
    // Step 4. Update the read/write buffer regions
    Array<BufferRegion> reads = VisitBufferRegions(block->reads);
    Array<BufferRegion> writes = VisitBufferRegions(block->writes);
    // Step 5. Handle predicate
    PrimExpr predicate = this->VisitExpr(realize->predicate);
    // Step 6. Root allocation
    Array<Buffer> alloc_buffers = AllocBuffer(block);
    // Step 7. Create new blocks
    return BlockRealize(/*iter_values=*/{},
                        /*predicate=*/std::move(predicate),
                        /*block=*/
                        Block({},                                          //
                              /*reads=*/std::move(reads),                  //
                              /*writes=*/std::move(writes),                //
                              /*name_hint=*/block->name_hint,              //
                              /*body=*/std::move(body),                    //
                              /*init=*/block->init,                        //
                              /*alloc_buffers=*/std::move(alloc_buffers),  //
                              /*match_buffers=*/block->match_buffers,      //
                              /*annotations=*/block->annotations           //
                              ));
  }

  PrimExpr VisitExpr_(const VarNode* var) final {
    auto it = var_substitutes_.find(GetRef<Var>(var));
    if (it != var_substitutes_.end()) {
      return it->second;
    }
    return GetRef<Var>(var);
  }

  Array<BufferRegion> VisitBufferRegions(const Array<BufferRegion>& buffer_regions) {
    // Calculate `new_buffer_regions` by recursively visiting min/extent of each range
    Array<BufferRegion> new_buffer_regions;
    new_buffer_regions.reserve(buffer_regions.size());
    for (const BufferRegion& buffer_region : buffer_regions) {
      const Buffer& buffer = buffer_region->buffer;
      const Array<Range>& region = buffer_region->region;
      Array<Range> new_region;
      new_region.reserve(region.size());
      for (const Range& range : region) {
        new_region.push_back(Range::FromMinExtent(/*min=*/this->VisitExpr(range->min),
                                                  /*extent=*/this->VisitExpr(range->extent)));
      }
      new_buffer_regions.push_back(BufferRegion(buffer, new_region));
    }
    // Calculate `info.accessed_region`
    for (const BufferRegion& buffer_region : new_buffer_regions) {
      const Buffer& buffer = buffer_region->buffer;
      auto it = buffer_info_.find(buffer);
      ICHECK(it != buffer_info_.end());
      BufferInfo& info = it->second;
      if (info.is_arg) {
        continue;
      }
      std::unordered_map<const VarNode*, arith::IntSet> dom_map;
      {
        const ForNode* alloc_site = info.alloc_site;
        // Every loop will be relaxed if the lca is the root
        bool need_relax = (alloc_site == nullptr);
        for (const For& loop : this->ancestor_loops_) {
          const VarNode* loop_var = loop->loop_var.get();
          if (need_relax || (buffer->scope == "shared" && IsThreadBound(loop))) {
            dom_map[loop_var] = IntSetFromMinExtent(loop->min, loop->extent);
          }
          if (loop.get() == alloc_site) {
            need_relax = true;
          }
        }
      }
      NDIntSet int_set;
      int_set.reserve(buffer_region->region.size());
      for (const Range& range : buffer_region->region) {
        int_set.push_back(arith::EvalSet(range, dom_map));
      }
      NDIntSetUnionWith(&info.accessed_region, int_set);
    }
    return new_buffer_regions;
  }

  Array<Buffer> AllocBuffer(const BlockNode* block) {
    const Array<Buffer>& buffers = block->alloc_buffers;
    Array<Buffer> result;
    result.reserve(buffers.size());
    for (const Buffer& buffer : buffers) {
      auto it = buffer_info_.find(buffer);
      ICHECK(it != buffer_info_.end());
      BufferInfo& info = it->second;
      if (info.is_arg) {
        ICHECK(info.region.defined());
        ICHECK(info.new_buffer.defined());
        continue;
      } else {
        ICHECK(!info.region.defined());
        ICHECK(!info.new_buffer.defined());
      }
      // Calculate `info.region`
      if (std::any_of(info.accessed_region.begin(), info.accessed_region.end(),
                      [](const arith::IntSet& int_set) { return int_set.IsNothing(); })) {
        // If the buffer can not be narrow, return the current buffer
        info.region.reserve(buffer->shape.size());
        for (const auto& extent : buffer->shape) {
          info.region.push_back(Range::FromMinExtent(0, extent));
        }
        info.new_buffer = buffer;
        result.push_back(buffer);
      } else {
        info.region = NDIntSet2Region(info.accessed_region);
        // Calculate `info.new_buffer`
        Array<PrimExpr> shape;
        shape.reserve(info.region.size());
        for (const Range& range : info.region) {
          shape.push_back(range->extent);
        }
        ObjectPtr<BufferNode> new_buffer = make_object<BufferNode>(*buffer.get());
        new_buffer->shape = std::move(shape);
        info.new_buffer = Buffer(std::move(new_buffer));
        result.push_back(info.new_buffer);
      }


    }
    return result;
  }

  void RewriteBufferAccess(Buffer* buffer, Array<PrimExpr>* indices) const {
    auto it = buffer_info_.find(*buffer);
    ICHECK(it != buffer_info_.end());
    const BufferInfo& info = it->second;
    ICHECK(info.new_buffer.defined());
    ICHECK(info.region.defined());
    ICHECK_GE(indices->size(), info.region.size());
    int ndim = info.region.size();
    Array<PrimExpr> new_indices;
    new_indices.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      new_indices.push_back((*indices)[i] - info.region[i]->min);
    }
    *buffer = info.new_buffer;
    *indices = std::move(new_indices);
  }

  void RewriteBufferRegion(Buffer* buffer, Region* region) const {
    auto it = buffer_info_.find(*buffer);
    ICHECK(it != buffer_info_.end());
    const BufferInfo& info = it->second;
    ICHECK(info.new_buffer.defined());
    ICHECK(info.region.defined());
    ICHECK_GE(region->size(), info.region.size());
    Region new_region;
    new_region.reserve(info.region.size());
    for (size_t i = 0; i < info.region.size(); ++i) {
      const Range& range = (*region)[i];
      new_region.push_back(Range::FromMinExtent(range->min - info.region[i]->min, range->extent));
    }
    *buffer = info.new_buffer;
    *region = std::move(new_region);
  }

  /*! \brief Collective information about each buffer */
  SMap<Buffer, BufferInfo> buffer_info_;
  /*! \brief The loops from the current node up to the root */
  std::vector<For> ancestor_loops_;
  /*! \brief The map from block vars to the expr value */
  SMap<Var, PrimExpr> var_substitutes_;
  /*! \brief Loop variables that are bound to reduction block vars */
  SSet<Var> reduction_loop_vars_;
};

PrimFunc NarrowBufferRegion(PrimFunc f) {
  PrimFuncNode* fptr = f.CopyOnWrite();
  fptr->body = BufferNarrower::Narrow(f);
  return f;
}

namespace transform {

Pass NarrowBufferRegion() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return NarrowBufferRegion(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.NarrowBufferRegion", {});
}

TVM_REGISTER_GLOBAL("tir.transform.NarrowBufferRegion").set_body_typed(NarrowBufferRegion);
}  // namespace transform

}  // namespace tir
}  // namespace tvm