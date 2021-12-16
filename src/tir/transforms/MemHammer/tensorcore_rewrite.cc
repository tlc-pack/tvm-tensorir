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
#include "rewrite_rule.h"
namespace tvm {
namespace tir {
/*!
 * \brief Tile the 2 innermost loops to extent=16. This helps further tensor core rewrite.
 * \param stmt The stmt
 * \return A pair. The first is the stmt after transformation.
 *         The second is the compute location where we may add write cache.
 */
std::pair<Stmt, For> TileWmmaBlock(Stmt stmt) {
  Stmt body = stmt;
  std::vector<const ForNode*> loops;
  while (const ForNode* loop = body.as<ForNode>()) {
    loops.push_back(loop);
    body = loop->body;
  }
  arith::Analyzer analyzer;
  PrimExpr extent_last1 = loops[loops.size() - 1]->extent,
           extent_last2 = loops[loops.size() - 2]->extent;

  if (!analyzer.CanProve(floormod(extent_last1, 16) == 0) ||
      !analyzer.CanProve(floormod(extent_last2, 16) == 0)) {
    return std::make_pair(stmt, For());
  }
  std::vector<Var> new_loop_vars;
  Array<PrimExpr> factor{floordiv(extent_last2, 16), floordiv(extent_last1, 16), 16, 16};
  new_loop_vars.reserve(4);
  for (int i = 0; i < 4; i++) {
    new_loop_vars.push_back(
        loops[loops.size() - (i + 1) % 2 - 1]->loop_var.copy_with_suffix(std::to_string(i / 2)));
  }
  Map<Var, PrimExpr> substitue_value;
  substitue_value.Set(loops[loops.size() - 2]->loop_var, new_loop_vars[0] * 16 + new_loop_vars[2]);
  substitue_value.Set(loops[loops.size() - 1]->loop_var, new_loop_vars[1] * 16 + new_loop_vars[3]);
  body = Substitute(body, substitue_value);
  for (int i = 3; i >= 0; i--) {
    body = For(new_loop_vars[i], 0, factor[i], ForKind::kSerial, body);
  }
  For compute_location = Downcast<For>(body);
  for (int i = static_cast<int>(loops.size()) - 3; i >= 0; i--) {
    body = For(loops[i]->loop_var, loops[i]->min, loops[i]->extent, loops[i]->kind, body,
               loops[i]->thread_binding, loops[i]->annotations);
  }
  return std::make_pair(body, compute_location);
}

/*!
 * \brief Rewrite the data copy that stores to wmma fragment with wmma::load_matrix_sync
 * \param stmt The stmt to rewrite
 * \return The stmt after rewrite
 */
Stmt RewriteWmmaLoad(Stmt stmt) {
  Array<MatchBufferRegion> match_buffers;
  Stmt body = stmt;
  Map<Var, Range> var_range;
  std::vector<const ForNode*> loops;
  while (const ForNode* loop = body.as<ForNode>()) {
    loops.push_back(loop);
    body = loop->body;
  }
  for (int i = 1; i <= 2; i++) {
    const ForNode* loop = loops[loops.size() - i];
    var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
  }
  const BufferStoreNode* buf_store = TVM_TYPE_AS(buf_store, body, BufferStoreNode);
  const BufferLoadNode* buf_load = TVM_TYPE_AS(buf_load, buf_store->value, BufferLoadNode);
  Buffer src_buffer = buf_load->buffer;
  Buffer tgt_buffer = buf_store->buffer;

  DataType dtype = DataType::Float(16);
  Var new_src_var("src", PointerType(PrimType(dtype), src_buffer.scope()));
  Type int32 = PrimType(DataType::Int(32));
  Buffer new_src_buffer(new_src_var, dtype, {Integer(16), Integer(16)},
                        {Var("s1", int32), Var("s0", int32)}, Var("src_elem_offset", int32), "src",
                        128, 16, kDefault);
  auto read_int_set = arith::EvalSet(buf_load->indices, AsIntSet(var_range));
  Array<Range> read_region;
  for (const auto& int_set : read_int_set) {
    read_region.push_back(int_set.CoverRange(Range()));
  }
  match_buffers.push_back(MatchBufferRegion(new_src_buffer, BufferRegion(src_buffer, read_region)));
  Var new_tgt_var("tgt", PointerType(PrimType(dtype), tgt_buffer.scope()));
  Buffer new_tgt_buffer(new_tgt_var, dtype, {Integer(16), Integer(16)}, {},
                        Var("tgt_elem_offset", int32), "tgt", 128, 16, kDefault);
  auto write_int_set = arith::EvalSet(buf_store->indices, AsIntSet(var_range));
  Array<Range> write_region;
  for (const auto& int_set : write_int_set) {
    write_region.push_back(int_set.CoverRange(Range()));
  }
  match_buffers.push_back(
      MatchBufferRegion(new_tgt_buffer, BufferRegion(tgt_buffer, write_region)));

  PrimExpr frag_index = floordiv(new_tgt_buffer->elem_offset, 256) +
                        floordiv(floormod(new_tgt_buffer->elem_offset, 256), 16);

  auto new_src_pointer = Call(
      runtime::DataType::Handle(), builtin::tvm_access_ptr(),
      {TypeAnnotation(new_src_buffer->dtype), new_src_buffer->data, new_src_buffer->elem_offset,
       new_src_buffer->strides[new_src_buffer->strides.size() - 2] * 16, 1});

  Stmt wmma_body = Evaluate(
      Call(runtime::DataType::Handle(), builtin::tvm_load_matrix_sync(),
           {new_tgt_buffer->data, 16, 16, 16, frag_index, new_src_pointer,
            new_src_buffer->strides[new_src_buffer->strides.size() - 2], StringImm("row_major")}));
  wmma_body = BlockRealize(
      {}, Bool(true), Block({}, {}, {}, "wmma_load", wmma_body, NullOpt, {}, match_buffers, {}));
  for (int i = static_cast<int>(loops.size()) - 3; i >= 0; i--) {
    wmma_body = For(loops[i]->loop_var, loops[i]->min, loops[i]->extent, loops[i]->kind, wmma_body,
                    loops[i]->thread_binding, loops[i]->annotations);
  }
  return wmma_body;
}

/*!
 * \brief Rewrite the data copy that loads from wmma fragment with wmma::store_matrix_sync
 * \param stmt The stmt to rewrite
 * \return The stmt after rewrite
 */
Stmt RewriteWmmaStore(Stmt stmt) {
  Array<MatchBufferRegion> match_buffers;
  Stmt body = stmt;
  Map<Var, Range> var_range;
  std::vector<const ForNode*> loops;
  while (const ForNode* loop = body.as<ForNode>()) {
    loops.push_back(loop);
    body = loop->body;
  }
  for (int i = 1; i <= 2; i++) {
    const ForNode* loop = loops[loops.size() - i];
    var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
  }
  const BufferStoreNode* buf_store = TVM_TYPE_AS(buf_store, body, BufferStoreNode);
  const BufferLoadNode* buf_load = TVM_TYPE_AS(buf_load, buf_store->value, BufferLoadNode);
  Buffer src_buffer = buf_load->buffer;
  Buffer tgt_buffer = buf_store->buffer;

  DataType dtype = DataType::Float(32);
  Type int32 = PrimType(DataType::Int(32));
  Var new_src_var("src", PointerType(PrimType(dtype), src_buffer.scope()));
  Buffer new_src_buffer(new_src_var, dtype, {Integer(16), Integer(16)}, {},
                        Var("src_elem_offset", int32), "src", 128, 16, kDefault);
  auto read_int_set = arith::EvalSet(buf_load->indices, AsIntSet(var_range));
  Array<Range> read_region;
  for (const auto& int_set : read_int_set) {
    read_region.push_back(int_set.CoverRange(Range()));
  }
  match_buffers.push_back(MatchBufferRegion(new_src_buffer, BufferRegion(src_buffer, read_region)));
  Var new_tgt_var("tgt", PointerType(PrimType(dtype), tgt_buffer.scope()));
  Buffer new_tgt_buffer(new_tgt_var, dtype, {Integer(16), Integer(16)},
                        {Var("s1", int32), Var("s0", int32)}, Var("tgt_elem_offset", int32), "tgt",
                        128, 16, kDefault);
  auto write_int_set = arith::EvalSet(buf_store->indices, AsIntSet(var_range));
  Array<Range> write_region;
  for (const auto& int_set : write_int_set) {
    write_region.push_back(int_set.CoverRange(Range()));
  }
  match_buffers.push_back(
      MatchBufferRegion(new_tgt_buffer, BufferRegion(tgt_buffer, write_region)));

  PrimExpr frag_index = floordiv(new_src_buffer->elem_offset, 256) +
                        floordiv(floormod(new_src_buffer->elem_offset, 256), 16);

  auto new_tgt_pointer = Call(runtime::DataType::Handle(), builtin::tvm_access_ptr(),
                              {TypeAnnotation(new_tgt_buffer->dtype), new_tgt_buffer->data,
                               new_tgt_buffer->elem_offset, new_tgt_buffer->strides[0] * 16, 2});

  Stmt wmma_body = Evaluate(Call(runtime::DataType::Handle(), builtin::tvm_store_matrix_sync(),
                                 {new_src_buffer->data, 16, 16, 16, frag_index, new_tgt_pointer,
                                  new_tgt_buffer->strides[0], StringImm("row_major")}));
  wmma_body = BlockRealize(
      {}, Bool(true), Block({}, {}, {}, "wmma_store", wmma_body, NullOpt, {}, match_buffers, {}));
  for (int i = static_cast<int>(loops.size()) - 3; i >= 0; i--) {
    wmma_body = For(loops[i]->loop_var, loops[i]->min, loops[i]->extent, loops[i]->kind, wmma_body,
                    loops[i]->thread_binding, loops[i]->annotations);
  }
  return wmma_body;
}

Stmt SharedToWmma::Rewrite(const Stmt& stmt, const Map<String, ObjectRef>& constraints,
                           Map<String, ObjectRef>* output) const {
  Stmt after_tiling = TileWmmaBlock(stmt).first;
  BufferRegion read_region = Downcast<BufferRegion>(constraints["read_region"]);
  output->Set("wmma_use", read_region->buffer);
  return RewriteWmmaLoad(after_tiling);
}

Stmt WmmaToShared::Rewrite(const Stmt& stmt, const Map<String, ObjectRef>& constraints,
                           Map<String, ObjectRef>* output) const {
  Stmt after_tiling = TileWmmaBlock(stmt).first;
  BufferRegion write_region = Downcast<BufferRegion>(constraints["write_region"]);
  output->Set("wmma_use", write_region->buffer);
  return RewriteWmmaStore(after_tiling);
}

class WmmaToGlobalRewriter : public StmtExprMutator {
 public:
  WmmaToGlobalRewriter(const SeqStmtNode* tgt_stmt, const Map<String, ObjectRef>& constraints)
      : tgt_stmt_(tgt_stmt), constraints_(constraints) {}

 private:
  Stmt VisitStmt_(const SeqStmtNode* op) final {
    if (op == tgt_stmt_) {
      ICHECK_EQ(op->seq.size(), 2);
      Stmt wmma_to_shared = RewriteWmmaStore(op->seq[0]);
      Stmt shared_to_global = CoalescedAccess().Rewrite(op->seq[1], constraints_, nullptr);
      return SeqStmt({wmma_to_shared, shared_to_global});
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }

  const SeqStmtNode* tgt_stmt_;
  const Map<String, ObjectRef>& constraints_;
};

std::pair<Stmt, SeqStmt> InsertCacheStage(Stmt stmt, bool is_write_cache, String storage_scope,
                                          For compute_location, const Array<For>& outer_loops,
                                          Buffer* alloc_buffer);

Stmt WmmaToGlobal::Rewrite(const Stmt& stmt, const Map<String, ObjectRef>& constraints,
                           Map<String, ObjectRef>* output) const {
  Stmt body;
  For compute_location;
  std::tie(body, compute_location) = TileWmmaBlock(stmt);
  SeqStmt seq;
  Array<For> outer_loops = Downcast<Array<For>>(constraints["outer_loops"]);
  Buffer cache_buffer;
  // Step 1. add a shared memory cache
  std::tie(body, seq) =
      InsertCacheStage(body, true, "shared.dyn", compute_location, outer_loops, &cache_buffer);
  output->Set("alloc_buffer", cache_buffer);
  output->Set("wmma_use", cache_buffer);
  // Step 2. do coalesced rewrite and tensor core rewrite respectively for 2 parts
  WmmaToGlobalRewriter rewriter(seq.get(), constraints);
  return rewriter(body);
}

}  // namespace tir
}  // namespace tvm
