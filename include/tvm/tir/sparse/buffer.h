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
 * \file tvm/tir/sparse/buffer.h
 * \brief Sparse buffer data structure in Sparse TIR.
 */
#ifndef TVM_TIR_SPARSE_BUFFER_H_
#define TVM_TIR_SPRASE_BUFFER_H_

#include <tvm/tir/sparse/format.h>
#include <tvm/tir/buffer.h>

namespace tvm {

namespace tir {

namespace sparse {

/*!
 * \brief Class of sparse buffer.
 */
class SparseBufferNode : public Object {
 public:
  /* Root of Axis Dependency Tree. */
  AxisRef root;
  /* Axes */
  Array<Axis> axes;
  /* Number of dimensions */
  int ndim;
  /* Buffer corresponding to flattened value */
  Buffer data;
  /* Buffer corresponding to indices pointer */
  Array<Buffer> indptr;
  /* Buffer of column indices */
  Array<Buffer> indices;

  static constexpr const char* _type_key = "tir.sp.SparseBufferNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(SparseBufferNode, Object);
};

/*!
 * \brief Managed reference to SparseBufferNode.
 * \sa SparseBufferNode
 */
class SparseBuffer : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(SparseBuffer, ObjectRef, SparseBufferNode);
};

}  // namespace sparse

}  // namespace tir

}  // namespace tvm

#endif  // TVM_TIR_SPARSE_BUFFER_H_
