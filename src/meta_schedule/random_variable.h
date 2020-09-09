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
#ifndef SRC_META_SCHEDULE_RANDOM_VARIABLE_H_
#define SRC_META_SCHEDULE_RANDOM_VARIABLE_H_

#include <tvm/tir/schedule.h>

namespace tvm {
namespace meta_schedule {

/*! \brief A random variable that evaluates to a TIR block */
class BlockRVNode : public Object {
 public:
  /*! \brief Name of the loop variable */
  String name;
  /*! \brief Value of the loop variable */
  mutable Optional<tir::StmtSRef> block;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("block", &block);
  }

  static constexpr const char* _type_key = "meta_schedule.BlockRV";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockRVNode, Object);
};

/*!
 * \brief Managed reference to BlockRVNode
 * \sa BlockRVNode
 */
class BlockRV : public ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param name Name of the loop variable
   * \param block Value of the loop variable
   */
  explicit BlockRV(String name, Optional<tir::StmtSRef> block);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(BlockRV, ObjectRef, BlockRVNode);
};

/*! \brief A random variable that evaluates to a TIR loop axis */
class LoopRVNode : public Object {
 public:
  /*! \brief Name of the loop variable */
  String name;
  /*! \brief Value of the loop variable */
  mutable Optional<tir::StmtSRef> loop;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("loop", &loop);
  }

  static constexpr const char* _type_key = "meta_schedule.LoopRV";
  TVM_DECLARE_FINAL_OBJECT_INFO(LoopRVNode, Object);
};

/*!
 * \brief Managed reference to LoopRVNode
 * \sa LoopRVNode
 */
class LoopRV : public ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param name Name of the loop variable
   * \param loop Value of the loop variable
   */
  explicit LoopRV(String name, Optional<tir::StmtSRef> loop);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(LoopRV, ObjectRef, LoopRVNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_RANDOM_VARIABLE_H_
