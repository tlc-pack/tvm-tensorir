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
#include "../../../runtime/thread_storage_scope.h"
namespace tvm{
namespace tir{
/*!
 * \brief Fuse consecutive loops
 * \param stmt the outer-most loop
 * \return the fused loop
 */
Stmt FuseNestLoops(const Stmt& stmt){
  std::vector<const ForNode*> loops;
  Stmt body = stmt;
  while (const ForNode* loop = body.as<ForNode>()) {
    loops.push_back(loop);
    body = loop->body;
  }
  Var fused_var = loops[0]->loop_var.copy_with_suffix("_fused");
  Array<PrimExpr> substitute_value;
  substitute_value.resize(loops.size());
  PrimExpr tot = fused_var;
  for (int i = static_cast<int>(loops.size()) - 1; i >= 0; i--) {
    substitute_value.Set(i, floormod(tot, loops[i]->extent));
    tot = floordiv(tot, loops[i]->extent);
  }
  auto f_substitute = [&](const Var& v) -> Optional<PrimExpr> {
    for (int i = 0; i < static_cast<int>(loops.size()); i++) {
      if (v.same_as(loops[i]->loop_var)) {
        return substitute_value[i];
      }
    }
    return NullOpt;
  };
  PrimExpr fused_extent = 1;
  for (int i = 0; i < static_cast<int>(loops.size()); i++) {
    fused_extent *= loops[i]->extent;
  }
  Stmt new_stmt = Substitute(body, f_substitute);
  new_stmt = For(fused_var, 0, fused_extent, ForKind::kSerial, new_stmt);
  return new_stmt;
}

/*!
 * \brief a combination of split, bind, vectorize,
 *        a helper function to perform coalesced load/store
 * \param stmt the stmt to do transformation
 * \param constraints The constraints, including thread extents, vector bytes, and data bits.
 * \return The stmt after transformation
 */
Stmt SplitBindVectorize(const Stmt& stmt, const Map<String, ObjectRef>& constraints){
  Stmt body = stmt;
  const ForNode* loop = body.as<ForNode>();
  PrimExpr vector_bytes = Downcast<PrimExpr>(constraints.Get("vector_bytes").value_or(Integer
                                                                                        (1)));
  PrimExpr threadIdx_x_ = Downcast<PrimExpr>(constraints.Get("threadIdx.x").value_or(Integer(1)));
  PrimExpr threadIdx_y_ = Downcast<PrimExpr>(constraints.Get("threadIdx.y").value_or(Integer(1)));
  PrimExpr threadIdx_z_ = Downcast<PrimExpr>(constraints.Get("threadIdx.z").value_or(Integer(1)));
  PrimExpr tot_threads = threadIdx_x_ * threadIdx_y_ * threadIdx_z_;
  PrimExpr data_bits_ = Downcast<PrimExpr>(constraints["data_bit"]);
  PrimExpr vector_len = max(1, vector_bytes * 8 / data_bits_);
  if (!loop || !is_zero(indexmod(loop->extent, (vector_len * tot_threads)))) {
    LOG(FATAL) << "the number of elements must be a multiple of thread num";
  }
  PrimExpr outer_loop_extent = indexdiv(loop->extent, tot_threads * vector_len);
  Array<PrimExpr> factors{outer_loop_extent};
  std::vector<std::string> thread_axis;
  //generate thread binding loops
  int new_loop_num = 2;
  if (is_one(threadIdx_z_)) {
    factors.push_back(threadIdx_z_);
    thread_axis.push_back("threadIdx.z");
    new_loop_num++;
  }
  if (is_one(threadIdx_y_)) {
    factors.push_back(threadIdx_y_);
    thread_axis.push_back("threadIdx.y");
    new_loop_num++;
  }
  if (is_one(threadIdx_x_)) {
    factors.push_back(threadIdx_x_);
    thread_axis.push_back("threadIdx.x");
    new_loop_num++;
  }
  //generate vectorized loop
  factors.push_back(vector_len);
  std::vector<Var> new_loop_vars;
  new_loop_vars.reserve(new_loop_num);
  for (int i = 0; i < new_loop_num; i++) {
    new_loop_vars.push_back(loop->loop_var.copy_with_suffix("_" + std::to_string(i)));
  }
  
  PrimExpr substitute_value = 0;
  for (int i = 0; i < new_loop_num; i++) {
    substitute_value *= factors[i];
    substitute_value += new_loop_vars[i];
  }
  body = Substitute(loop->body, [&](const Var& v) -> Optional<PrimExpr> {
    if (v.same_as(loop->loop_var)) {
      return substitute_value;
    } else {
      return NullOpt;
    }
  });
  
  For new_loop = For(new_loop_vars[new_loop_num - 1], 0, vector_len, ForKind::kVectorized, body);

  for (int i = new_loop_num - 2; i >= 1; i--) {
    new_loop =
        For(new_loop_vars[i], 0, factors[i], ForKind::kThreadBinding, new_loop,
            IterVar(Range(nullptr), Var(thread_axis[i - 1]), kThreadIndex, thread_axis[i - 1]));
  }

  new_loop = For(new_loop_vars[0], 0, outer_loop_extent, ForKind::kSerial, new_loop);
  return std::move(new_loop);
}

Stmt CoalescedAccess::Rewrite(const Stmt& stmt, const Map<String, ObjectRef>& constraints, Map<String, ObjectRef>*
                                                                                               output) const{
  Stmt after_fuse = FuseNestLoops(stmt);
  Stmt after_split = SplitBindVectorize(after_fuse, constraints);
  return after_split;
}

/*!
 * \brief Get the index mapping of a specific stmt
 * \param stmt The stmt
 * \param constraints The constraints, including the write region that is required to calculate
 * the index mapping
 * \return The mapping
 */
std::pair<Array<PrimExpr>, Array<Var>> GetMapping(const Stmt& stmt, const Map<String, ObjectRef>& constraints) {
  Stmt body = stmt;
  Array<Var> loop_vars;
  while (const ForNode* loop = body.as<ForNode>()) {
    loop_vars.push_back(loop->loop_var);
    body = loop->body;
  }
  const BufferStoreNode* buf_store = TVM_TYPE_AS(buf_store, body, BufferStoreNode);
  BufferRegion write_region = Downcast<BufferRegion>(constraints["write_region"]);
  const Array<PrimExpr>& write_index = buf_store->indices;
  ICHECK(write_region->region.size() == write_index.size() &&
         write_region->buffer.same_as(buf_store->buffer));
  Array<PrimExpr> result;
  arith::Analyzer analyzer;
  for (int i = 0; i < static_cast<int>(write_region->region.size()); i++) {
    PrimExpr pattern = analyzer.Simplify(write_index[i] - write_region->region[i]->min);
    if (!is_zero(pattern)) {
      result.push_back(pattern);
    }
  }
  return std::make_pair(result, loop_vars);
}

Stmt InverseMapping::Rewrite(const Stmt& stmt, const Map<String, ObjectRef>& constraints, Map<String, ObjectRef>*
                                                                                              output)const {
  Stmt body = stmt;
  Map<Var, Range> var_range;
  Array<PrimExpr> loop_vars;
  // Step 1. Get index mapping
  std::pair<Array<PrimExpr>, Array<Var>> mapping = GetMapping(stmt, constraints);
  Array<PrimExpr> mapping_pattern = mapping.first;
  while (const ForNode* loop = body.as<ForNode>()) {
    var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    loop_vars.push_back(loop->loop_var);
    body = loop->body;
  }
  // Step 2. Get Inverse mapping
  arith::Analyzer analyzer;
  Array<arith::IterSumExpr> iter_map =
      arith::DetectIterMap(mapping_pattern, var_range, Bool(true), true, &analyzer);
  CHECK_EQ(iter_map.size(), loop_vars.size());
  Map<Var, PrimExpr> inverse_mapping = arith::InverseAffineIterMap(iter_map, loop_vars);
  // Step 3. Generate new body
  BufferRegion read_region = Downcast<BufferRegion>(constraints["read_region"]);
  BufferRegion write_region = Downcast<BufferRegion>(constraints["write_region"]);
  Array<PrimExpr> write_index;
  Array<PrimExpr> read_index;
  Array<Var> new_loop_vars;
  Map<Var, PrimExpr> substitute_map;
  // Step 3.1 construct target buffer indices
  for (int i = 0, j = 0; i < static_cast<int>(write_region->region.size()); i++) {
    if (is_zero(write_region->region[i]->extent)) {
      write_index.push_back(write_region->region[i]->min);
    } else {
      Var var = runtime::Downcast<Var>(loop_vars[j]).copy_with_suffix("_inverse");
      new_loop_vars.push_back(var);
      substitute_map.Set(runtime::Downcast<Var>(loop_vars[j++]), var);
      write_index.push_back(write_region->region[i]->min + var);
    }
  }
  // Step 3.2 construct source buffer indices
  for (int i = 0, j = 0; i < static_cast<int>(read_region->region.size()); i++) {
    if (is_zero(read_region->region[i]->extent)) {
      read_index.push_back(read_region->region[i]->min);
    } else {
      read_index.push_back(
          read_region->region[i]->min +
          Substitute(inverse_mapping[Downcast<Var>(loop_vars[j++])], substitute_map));
    }
  }
  BufferLoad new_buf_load = BufferLoad(read_region->buffer, read_index);
  BufferStore new_buf_store = BufferStore(write_region->buffer, new_buf_load, write_index);
  Stmt ret = new_buf_store;
  //Step 3.3 construct loop body
  for (int i = static_cast<int>(new_loop_vars.size()) - 1; i >= 0; i--) {
    PrimExpr extent = write_region->region[i]->extent;
    ret = For(new_loop_vars[i], 0, extent, ForKind::kSerial, ret);
  }
  return ret;
}
}// namespace tir
}// namespace tvm
