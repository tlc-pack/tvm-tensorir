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

/*!
 * \brief Detecting the LCA of buffer access points and
 *        where the buffer should be allocated
 * \file locate_buffer_allocation.cc
 */

#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

class BufferAllocationLocator : public StmtExprMutator {
 public:
  explicit BufferAllocationLocator(const PrimFunc& func) {
    Map<Buffer, Stmt> buffer_lac = DetectBufferAccessLCA(func);
    for (const auto& pair : buffer_lac) {
      const Buffer& buffer = pair.first;
      const StmtNode* stmt = pair.second.get();
      auto it = alloc_buffers_.find(stmt);
      if (it == alloc_buffers_.end()) {
        alloc_buffers_[stmt] = {buffer};
      } else {
        it->second.push_back(buffer);
      }
    }
  }

 private:
  Stmt VisitStmt_(const ForNode* op) final {
    auto it = alloc_buffers_.find(op);
    if (it != alloc_buffers_.end()) {
      for (const Buffer& buf : it->second) {
        buffer_alloc_outer_.Set(buf->data, buf);
      }
    }
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<ForNode>();
    ICHECK(op != nullptr);
    if (it != alloc_buffers_.end()) {
      for (const Buffer& buf : it->second) {
        buffer_alloc_outer_.erase(buf->data);
      }
      Stmt body = InjectOpaqueBlock(op->body, it->second);
      auto n = CopyOnWrite(op);
      n->body = std::move(body);
      return Stmt(n);
    } else {
      return GetRef<Stmt>(op);
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    Array<Buffer> alloc_buffers;
    auto it = alloc_buffers_.find(op);
    if (it != alloc_buffers_.end()) {
      alloc_buffers = it->second;
      for (const Buffer& buf : it->second) {
        buffer_alloc_outer_.Set(buf->data, buf);
      }
    } else {
      alloc_buffers = {};
    }
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<BlockNode>();
    ICHECK(op != nullptr);

    // Ignore buffer allocated inside the block when getting access region.
    if (it != alloc_buffers_.end()) {
      for (const Buffer& buf : it->second) {
        buffer_alloc_outer_.erase(buf->data);
      }
    }

    if (alloc_buffers.same_as(op->alloc_buffers)) {
      return GetRef<Stmt>(op);
    } else {
      // Recalculate block access region
      auto access = GetBlockAccessRegion(GetRef<Block>(op), buffer_alloc_outer_);
      auto reads = access[0];
      auto writes = access[1];
      for (const auto& opaque_access : access[2]) {
        reads.push_back(opaque_access);
        writes.push_back(opaque_access);
      }

      auto n = CopyOnWrite(op);
      n->alloc_buffers = std::move(alloc_buffers);
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    ICHECK(false) << "Internal Error: BufferRealizeNode is not allowed in TensorIR.";
    return StmtMutator::VisitStmt_(op);
  }

  Stmt InjectOpaqueBlock(const Stmt& body, const std::vector<Buffer>& alloc_buffers) {
    ICHECK(!alloc_buffers.empty());
    Block opaque_block(/*iter_vars=*/{},
                       /*reads=*/{},
                       /*writes=*/{},
                       /*name_hint=*/"",
                       /*body=*/body,
                       /*init=*/NullOpt,
                       /*alloc_buffers=*/alloc_buffers);
    auto access = GetBlockAccessRegion(opaque_block, buffer_alloc_outer_);
    auto n = CopyOnWrite(opaque_block.get());
    n->reads = access[0];
    n->writes = access[1];
    for (const auto& opaque_access : access[2]) {
      n->reads.push_back(opaque_access);
      n->writes.push_back(opaque_access);
    }
    BlockRealize realize({}, Bool(true), Block(n));
    return std::move(realize);
  }

  std::map<const StmtNode*, std::vector<Buffer>> alloc_buffers_;
  Map<Var, Buffer> buffer_alloc_outer_;
};

PrimFunc LocateBufferAllocation(PrimFunc func) {
  auto fptr = func.CopyOnWrite();
  BufferAllocationLocator locator(func);
  fptr->body = locator(fptr->body);
  return func;
}

namespace transform {

Pass LocateBufferAllocation() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LocateBufferAllocation(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LocateBufferAllocation", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LocateBufferAllocation").set_body_typed(LocateBufferAllocation);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
