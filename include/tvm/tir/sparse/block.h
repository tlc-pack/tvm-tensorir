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
 * \file tvm/tir/sparse/block.h
 * \brief Sparse Block in Sparse TIR.
 */

#ifndef TVM_TIR_SPARSE_BLOCK_H_
#define TVM_TIR_SPARSE_BLOCK_H_

#include <tvm/tir/sparse/format.h>

namespace tvm {

namespace tir {

namespace sparse {

/*!
 * \brief Class of sparse block.
 * \example
 * with tir.sp.block([i, j, k], [False, False, True]) as [vi, vj, vk]:
 *     pass
 * with tir.sp.block([i, j, k], [False, False, True], [(0, 1), (2,)]) as [vi, vj, vk]:
 *     pass
 */
class SparseBlockNode : public Object {
 public:
  AxisRef root;
  Array<Axis> axes;
  Array<Array<int>> fused_groups;
  Array<bool> is_reduce_axis;
  static constexpr const char* _type_key = "tir.sp.SparseBlockNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(SparseBlockNode, Object);
};

class SparseBlock : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(SparseBlock, ObjectRef, SparseBlockNode);
}


}  // namespace sparse

}  // namespace tir

}  // namespace tvm

#endif  // TVM_TIR_SPRASE_BLOCK_H_
