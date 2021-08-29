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
  static constexpr const char* _type_key = "Axis";
  TVM_DECLARE_BASE_OBJECT_INFO(AxisNode, Object);
};

class Axis : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Axis, ObjectRef, AxisNode);
};

class RootAxisNode : public Object {
 public:
  static constexpr const char* _type_key = "RootAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(RootAxisNode, Object);
};

class RootAxis : public Object {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(RootAxis, ObjectRef, RootAxisNode);
};

class DenseAxisNode : public AxisNode {
 public:
  static constexpr const char* _type_key = "DenseAxis";
  TVM_DECLARE_BASE_OBJECT_INFO(DenseAxisNode, AxisNode);
};

class DenseAxis : public Axis {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(DenseAxis, Axis, DenseAxisNode);
};

class DenseFixedAxisNode : public DenseAxisNode {
 public:
  static constexpr const char* _type_key = "DenseFixedAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(DenseFixedAxisNode, DenseAxisNode);
};

class DenseFixedAxis : public DenseAxis {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(DenseFixedAxis, DenseAxis, DenseFixedAxisNode);
};

class DenseVariableAxisNode : public DenseAxisNode {
 public:
  static constexpr const char* _type_key = "DenseVariableAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(DenseVariableAxisNode, DenseAxisNode);
};

class DenseVariableAxis : public DenseAxis {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(DenseVariableAxis, DenseAxis,
                                DenseVariableAxisNode);
};

class SparseAxisNode : public AxisNode {
 public:
  static constexpr const char* _type_key = "SparseAxis";
  TVM_DECLARE_BASE_OBJECT_INFO(SparseAxisNode, AxisNode);
};

class SparseAxis : public Axis {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(SparseAxis, Axis, SparseAxisNode);
};

class SparseFixedAxisNode : public SparseAxisNode {
 public:
  static constexpr const char* _type_key = "SparseFixedAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(SparseFixedAxisNode, SparseAxisNode);
};

class SparseFixedAxis : public SparseAxis {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(SparseFixedAxis, SparseAxis,
                                SparseFixedAxisNode);
};

class SparseVariableAxisNode : public SparseAxisNode {
 public:
  static constexpr const char* _type_key = "SparseVariabledAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(SparseVariableAxisNode, SparseAxisNode);
};

class SparseVariableAxis : public SparseAxis {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(SparseVariableAxis, SparseAxis,
                                SparseVariableAxisNode);
};

/*!
 * \brief An object that refers to axis 
 */
class AxisRefNode : public Object {
 public:
  const AxisNode* axis;
  AxisRefNode* parent;

  static constexpr const char* _type_key = "tir.sp.AxisRefNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(AxisRefNode, Object);

  /*! \brief Reset the object inplace to the invalid state. */
  void Reset() {
    this->axis = nullptr;
    this->parent = nullptr;  
  }

  /*!
   * \brief Get the referenced axis with proper type checking.
   * It serves the same purpose as `ObjectRef::as`, but does not acquire strong reference to `axis`.
   * \tparam AxisType The type that `this->axis` to be downcasted to.
   * \return nullptr if type check fails, otherwise the casted result for `this->axis`
   */
  template <typename AxisType>
  const AxisType* AxisAs() const {
    if (axis != nullptr && axis->IsInstance<AxisType>()) {
      return static_cast<const AxisType*>(axis);
    } else {
      return nullptr;
    }
  }
};

class AxisRef : public ObjectRef {
 public:
 /*!
  * \brief The constructor
  * \param axis The corresponding axis node.
  * \param parent The parent AxisRef.
  */
 TVM_DLL explicit AxisRef(const AxisNode* axis, AxisRefNode* parent);

 /*! \return The mutable pointer to AxisRefNode */
 AxisRefNode* get() const {
   return static_cast<AxisRefNode*>(data_.get());
 }

 TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(AxisRef, ObjectRef, AxisRefNode);
};

}  // namespace sparse

}  // namespace tir

}  // namespace tvm

#endif  // TVM_TIR_SPRASE_FORMAT_H_
