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
#ifndef TVM_META_SCHEDULE_ARG_INFO_H_
#define TVM_META_SCHEDULE_ARG_INFO_H_

#include <tvm/node/node.h>
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/tir/function.h>

namespace tvm {
namespace meta_schedule {

/*! \brief The argument information. */
class ArgInfoNode : public runtime::Object {
 public:
  virtual ~ArgInfoNode() = default;

  virtual ObjectRef AsJSON() const = 0;

  static constexpr const char* _type_key = "meta_schedule.ArgInfo";
  TVM_DECLARE_BASE_OBJECT_INFO(ArgInfoNode, runtime::Object);
};

/*!
 * \brief Managed reference to ArgInfoNode
 * \sa ArgInfoNode
 */
class ArgInfo : public runtime::ObjectRef {
 public:
  /*!
   * \brief Get the argument information from PrimFunc.
   * \param func The PrimFunc to get argument information from.
   * \return An array of the argument information derived.
   */
  TVM_DLL static Array<ArgInfo, void> FromPrimFunc(const tir::PrimFunc& func);
  /*!
   * \brief Parse the argument information from a json object.
   * \param json The json object to parse.
   * \return The argument information parsed.
   */
  TVM_DLL static ArgInfo FromJSON(const ObjectRef& json_obj);
  TVM_DEFINE_OBJECT_REF_METHODS(ArgInfo, runtime::ObjectRef, ArgInfoNode);
};

/*! \brief The tensor argument information. */
class TensorArgInfoNode : public ArgInfoNode {
 public:
  /*! \brief The data type of the tensor. */
  runtime::DataType dtype;
  /*! \brief The shape of the tensor. */
  runtime::ShapeTuple shape;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("shape", &shape);
  }

  static constexpr const char* _type_key = "meta_schedule.TensorArgInfo";
  TVM_DECLARE_BASE_OBJECT_INFO(TensorArgInfoNode, ArgInfoNode);

 public:
  ObjectRef AsJSON() const;
};

/*!
 * \brief Managed reference to TensorArgInfoNode
 * \sa TensorArgInfoNode
 */
class TensorArgInfo : public ArgInfo {
 public:
  /*!
   * \brief Constructor of TensorArgInfo.
   * \param dtype The data type of the tensor argument.
   * \param shape The shape tuple of the tensor argument.
   */
  TVM_DLL explicit TensorArgInfo(runtime::DataType dtype, runtime::ShapeTuple shape);
  TVM_DLL static TensorArgInfo FromJSON(const ObjectRef& json_obj);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TensorArgInfo, ArgInfo, TensorArgInfoNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_ARG_INFO_H_
