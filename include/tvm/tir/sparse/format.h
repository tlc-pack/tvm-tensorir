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
 * \file tvm/tir/sparse/format.h
 * \brief Sparse format in Sparse TIR.
 */

#ifndef TVM_TIR_SPARSE_FORMAT_H_
#define TVM_TIR_SPRASE_FORMAT_H_

#include <tvm/ir/expr.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/tir/var.h>

#include <string>

namespace tvm {

namespace tir {

namespace sparse {

/*!
 * \brief Base type for axis in sparse formats.
 */
class AxisNode : public Object {
 public:
  /* name of current axis. */
  String name;
  /* length of current axis. For sparse axis, length refers to the upperbound of
   * the current axis. */
  PrimExpr length;
  static constexpr const char* _type_key = "Axis";
  TVM_DECLARE_BASE_OBJECT_INFO(AxisNode, Object);
};

/*!
 * \brief Managed reference to AxisNode.
 * \sa AxisNode
 */
class Axis : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Axis, ObjectRef, AxisNode);
};

/*!
 * \brief Root of Axis Dependency Tree.
 */
class RootAxisNode : public Object {
 public:
  static constexpr const char* _type_key = "RootAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(RootAxisNode, Object);
};

/*!
 * \brief Managed reference to RootAxisNode.
 * \sa RootAxisNode
 */
class RootAxis : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(RootAxis, ObjectRef, RootAxisNode);
};

/*!
 * \brief Dense axis whose column indices are consecutive.
 */
class DenseAxisNode : public AxisNode {
 public:
  static constexpr const char* _type_key = "DenseAxis";
  TVM_DECLARE_BASE_OBJECT_INFO(DenseAxisNode, AxisNode);
};

/*!
 * \brief Managed reference to DenseAxisNode.
 * \sa DenseAxisNode
 */
class DenseAxis : public Axis {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(DenseAxis, Axis, DenseAxisNode);
};

/*!
 * \brief Dense axis with fixed length per row.
 */
class DenseFixedAxisNode : public DenseAxisNode {
 public:
  static constexpr const char* _type_key = "DenseFixedAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(DenseFixedAxisNode, DenseAxisNode);
};

/*!
 * \brief Managed reference to DenseFixedAxisNode.
 * \sa DenseFixedAxisNode
 */
class DenseFixedAxis : public DenseAxis {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(DenseFixedAxis, DenseAxis, DenseFixedAxisNode);
};

class DenseVariableAxisNode : public DenseAxisNode {
 public:
  static constexpr const char* _type_key = "DenseVariableAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(DenseVariableAxisNode, DenseAxisNode);
};

/*!
 * \brief Dense axis whose length is dependent on its predecessors on the axis
 * dependency tree.
 */
class DenseVariableAxis : public DenseAxis {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(DenseVariableAxis, DenseAxis,
                                DenseVariableAxisNode);
};

/*!
 * \brief Sparse axis whose column indices is not consecutive.
 */
class SparseAxisNode : public AxisNode {
 public:
  static constexpr const char* _type_key = "SparseAxis";
  TVM_DECLARE_BASE_OBJECT_INFO(SparseAxisNode, AxisNode);
};

/*!
 * \brief Managed reference to SparseAxisNode.
 * \sa SparseAxisNode
 */
class SparseAxis : public Axis {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(SparseAxis, Axis, SparseAxisNode);
};

/*!
 * \brief Sparse axis with fixed number of non-zero columns per row.
 */
class SparseFixedAxisNode : public SparseAxisNode {
 public:
  /* (fixed) number of columns of current sparse axis. */
  PrimExpr num_cols;
  static constexpr const char* _type_key = "SparseFixedAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(SparseFixedAxisNode, SparseAxisNode);
};

/*!
 * \brief Managed reference to SparseFixedAxisNode.
 * \sa SparseFixedAxisNode
 */
class SparseFixedAxis : public SparseAxis {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(SparseFixedAxis, SparseAxis,
                                SparseFixedAxisNode);
};

/*!
 * \brief Sparse axis with variable number of non-zero columns per row.
 */
class SparseVariableAxisNode : public SparseAxisNode {
 public:
  static constexpr const char* _type_key = "SparseVariabledAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(SparseVariableAxisNode, SparseAxisNode);
};

/*!
 * \brief Managed reference to SparseVariableAxisNode.
 * \sa SparseVariableAxisNode
 */
class SparseVariableAxis : public SparseAxis {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(SparseVariableAxis, SparseAxis,
                                SparseVariableAxisNode);
};

/*!
 * \brief Reference of Axis on Axis Dependency Tree.
 */
class AxisRefNode : public Object {
 public:
  // parent refers to the parent axis of current axis tree.
  Optional<AxisRef> parent;
  Axis axis;
  Array<AxisRef> children;
  static constexpr const char* _type_key = "tir.sp.AxisRefNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(AxisRefNode, Object);
};

/*!
 * \brief Managed reference to AxisRefNode.
 * \sa AxisRefNode
 */
class AxisRef : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(AxisRef, ObjectRef, AxisRefNode);
};

}  // namespace sparse

}  // namespace tir

}  // namespace tvm

#endif  // TVM_TIR_SPRASE_FORMAT_H_
