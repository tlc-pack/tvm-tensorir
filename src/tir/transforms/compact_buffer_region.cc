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
 * \file compact_buffer_region.cc
 * \brief Compact the buffer size into its exact need.
 */

#include <tvm/arith/int_set.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../support/utils.h"

namespace tvm {
namespace tir {

using NDIntSet = std::vector<arith::IntSet>;

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
  if (support::StrStartsWith(thread_tag, "threadIdx")) {
    return true;
  }
  if (support::StrStartsWith(thread_tag, "vthread")) {
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

/*! \brief Collect the access region of each buffer. */
class BufferAccessRegionCollector : public StmtExprVisitor {
 public:
  static std::unordered_map<const BufferNode*, Region> Collect(const PrimFunc& f) {
    std::unordered_map<const BufferNode*, BufferAccessInfo> buffer_info;
    for (const auto& kv : f->buffer_map) {
      const Buffer& buffer = kv.second;
      BufferAccessInfo info(buffer->shape.size());
      info.accessed_region = NDIntSetFromShape(buffer->shape);
      info.is_arg = true;
      buffer_info.emplace(buffer.get(), info);
    }
    BufferAccessRegionCollector collector(std::move(buffer_info));
    collector(f->body);
    std::unordered_map<const BufferNode*, Region> ret;
    for (const auto& kv : collector.buffer_info_) {
      const BufferNode* buffer = kv.first;
      const BufferAccessInfo& info = kv.second;
      Region region = NarrowBufferRegionFromNDIntSet(info.accessed_region, buffer->shape);
      ret.emplace(buffer, std::move(region));
    }
    return ret;
  }

 private:
  struct BufferAccessInfo {
    /*! \brief The buffer access region, which can be updated during visiting. */
    NDIntSet accessed_region;
    /*! \brief The inner most loop outside the buffer allocation site. */
    const ForNode* alloc_site = nullptr;
    /*! \brief Mark whether the buffer is an arg (defined by function match_buffer). */
    bool is_arg = false;

    explicit BufferAccessInfo(int ndim) : accessed_region(NDIntSetEmpty(ndim)) {}
  };

  explicit BufferAccessRegionCollector(
      std::unordered_map<const BufferNode*, BufferAccessInfo> buffer_info)
      : buffer_info_(std::move(buffer_info)) {}

  void VisitStmt_(const ForNode* loop) final {
    ancestor_loops_.push_back(loop);
    StmtExprVisitor::VisitStmt_(loop);
    ancestor_loops_.pop_back();
  }

  void VisitStmt_(const BlockNode* op) final {
    ICHECK(!op->init.defined());
    // Step 1. Update BufferAccessInfo
    for (const Buffer& buffer : op->alloc_buffers) {
      BufferAccessInfo info(buffer->shape.size());
      if (!ancestor_loops_.empty()) {
        info.alloc_site = ancestor_loops_[ancestor_loops_.size() - 1];
      } else {
        info.alloc_site = nullptr;
      }
      buffer_info_.emplace(buffer.get(), info);
    }
    // Step 2. Update the read/write buffer regions
    VisitBufferRegions(op->reads);
    VisitBufferRegions(op->writes);
    // Step 3. Visit recursively
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitBufferRegions(const Array<BufferRegion>& buffer_regions) {
    // Calculate `info.accessed_region`
    for (const BufferRegion& buffer_region : buffer_regions) {
      const BufferNode* buffer = buffer_region->buffer.get();
      auto it = buffer_info_.find(buffer);
      ICHECK(it != buffer_info_.end());
      BufferAccessInfo& info = it->second;
      if (info.is_arg) {
        continue;
      }
      std::unordered_map<const VarNode*, arith::IntSet> dom_map;
      {
        const ForNode* alloc_site = info.alloc_site;
        // Every loop will be relaxed if the lca is the root
        bool need_relax = (alloc_site == nullptr);
        for (const ForNode* loop : ancestor_loops_) {
          const VarNode* loop_var = loop->loop_var.get();
          if (need_relax || (buffer->scope == "shared" && IsThreadBound(GetRef<For>(loop)))) {
            dom_map[loop_var] = IntSetFromMinExtent(loop->min, loop->extent);
          }
          if (loop == alloc_site) {
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
  }

  static Region NarrowBufferRegionFromNDIntSet(const NDIntSet& nd_int_set,
                                               const Array<PrimExpr>& original_shape) {
    Integer one(1);
    Array<Range> result;
    result.reserve(nd_int_set.size());
    for (size_t i = 0; i < nd_int_set.size(); ++i) {
      const arith::IntSet& int_set = nd_int_set[i];
      if (int_set.IsNothing()) {
        result.push_back(Range(/*begin=*/0, /*end=*/original_shape[i]));
      } else {
        PrimExpr min = int_set.min();
        PrimExpr max = int_set.max();
        result.push_back(Range(/*begin=*/min, /*end=*/max + one));
      }
    }
    return result;
  }

  /*! \brief Collective information about each buffer. */
  std::unordered_map<const BufferNode*, BufferAccessInfo> buffer_info_;
  /*! \brief The loops from the current node up to the root. */
  std::vector<const ForNode*> ancestor_loops_;
};

/*! \brief Reallocate the buffers with minial region. */
class BufferCompactor : public StmtExprMutator {
 public:
  static Stmt Compact(const PrimFunc& f,
                      const std::unordered_map<const BufferNode*, Region>& regions) {
    std::unordered_map<const BufferNode*, BufferAllocInfo> buffer_info;
    for (const auto& kv : regions) {
      const BufferNode* buffer = kv.first;
      Region region = kv.second;
      buffer_info.emplace(buffer, BufferAllocInfo(std::move(region)));
    }
    BufferCompactor compactor(std::move(buffer_info));
    Stmt stmt = compactor(f->body);
    stmt = BufferAccessRewriter::Rewrite(
        /*stmt=*/stmt,
        /*f_access_rewrite=*/
        std::bind(&BufferCompactor::RewriteBufferAccess,  //
                  &compactor,                             //
                  std::placeholders::_1,                  //
                  std::placeholders::_2),
        /*f_region_rewrite=*/
        std::bind(&BufferCompactor::RewriteBufferRegion,  //
                  &compactor,                             //
                  std::placeholders::_1,                  //
                  std::placeholders::_2));
    return stmt;
  }

 private:
  struct BufferAllocInfo {
    /*! \brief The buffer access region. */
    Region region;
    /*!
     * \brief The reallocated buffer with minimal size.
     * \note The value if NullOpt if the buffer do not need reallocate (e.g arg buffer).
     */
    Optional<Buffer> new_buffer = NullOpt;

    explicit BufferAllocInfo(Region region) : region(std::move(region)) {}
  };

  explicit BufferCompactor(std::unordered_map<const BufferNode*, BufferAllocInfo> buffer_info)
      : buffer_info_(std::move(buffer_info)) {}

  Stmt VisitStmt_(const BlockNode* op) final {
    ICHECK(!op->init.defined());
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BlockNode>();
    ICHECK(op != nullptr);

    Array<Buffer> alloc_buffers = AllocBuffer(op);
    auto n = CopyOnWrite(op);
    n->alloc_buffers = std::move(alloc_buffers);
    return Stmt(n);
  }

  Array<Buffer> AllocBuffer(const BlockNode* block) {
    const Array<Buffer>& buffers = block->alloc_buffers;
    Array<Buffer> result;
    result.reserve(buffers.size());
    for (const Buffer& buffer : buffers) {
      auto it = buffer_info_.find(buffer.get());
      ICHECK(it != buffer_info_.end());
      BufferAllocInfo& info = it->second;
      Array<PrimExpr> shape;
      shape.reserve(info.region.size());
      for (const Range& range : info.region) {
        shape.push_back(range->extent);
      }
      ObjectPtr<BufferNode> n = make_object<BufferNode>(*buffer.get());
      n->shape = std::move(shape);
      info.new_buffer = Buffer(std::move(n));
      result.push_back(info.new_buffer.value());
    }
    return result;
  }

  void RewriteBufferAccess(Buffer* buffer, Array<PrimExpr>* indices) const {
    auto it = buffer_info_.find(buffer->get());
    ICHECK(it != buffer_info_.end());
    const BufferAllocInfo& info = it->second;
    if (!info.new_buffer.defined()) {
      // new_buffer is empty if and only if it's an arg buffer
      // Then the buffer and indices do not need to change
      return;
    }
    ICHECK_GE(indices->size(), info.region.size());
    int ndim = info.region.size();
    Array<PrimExpr> new_indices;
    new_indices.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      new_indices.push_back((*indices)[i] - info.region[i]->min);
    }
    *buffer = info.new_buffer.value();
    *indices = std::move(new_indices);
  }

  void RewriteBufferRegion(Buffer* buffer, Region* region) const {
    auto it = buffer_info_.find(buffer->get());
    ICHECK(it != buffer_info_.end());
    const BufferAllocInfo& info = it->second;
    if (!info.new_buffer.defined()) {
      // new_buffer is empty if and only if it's an arg buffer
      // Then the buffer and region do not need to change
      return;
    }
    ICHECK_GE(region->size(), info.region.size());
    Region new_region;
    new_region.reserve(info.region.size());
    for (size_t i = 0; i < info.region.size(); ++i) {
      const Range& range = (*region)[i];
      new_region.push_back(Range::FromMinExtent(range->min - info.region[i]->min, range->extent));
    }
    *buffer = info.new_buffer.value();
    *region = std::move(new_region);
  }

  /*! \brief The allocation information about each buffer. */
  std::unordered_map<const BufferNode*, BufferAllocInfo> buffer_info_;
};

PrimFunc CompactBufferAllocation(PrimFunc f) {
  PrimFuncNode* fptr = f.CopyOnWrite();
  std::unordered_map<const BufferNode*, Region> region = BufferAccessRegionCollector::Collect(f);
  fptr->body = BufferCompactor::Compact(f, region);
  return f;
}

namespace transform {

Pass CompactBufferAllocation() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return CompactBufferAllocation(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.CompactBufferAllocation", {});
}

TVM_REGISTER_GLOBAL("tir.transform.CompactBufferAllocation")
    .set_body_typed(CompactBufferAllocation);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
