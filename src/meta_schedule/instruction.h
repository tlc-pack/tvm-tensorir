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
#ifndef SRC_META_SCHEDULE_INSTRUCTION_H_
#define SRC_META_SCHEDULE_INSTRUCTION_H_

#include "./random_variable.h"

namespace tvm {
namespace meta_schedule {

/**************** Instruction ****************/

/*! \brief Base class for all meta scheduling instrructions */
class InstructionNode : public Object {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "meta_schedule.Instruction";
  TVM_DECLARE_BASE_OBJECT_INFO(InstructionNode, Object);
};

/*!
 * \brief Managed reference to InstructionNode
 * \sa InstructionNode
 */
class Instruction : public ObjectRef {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Instruction, ObjectRef, InstructionNode);

 protected:
  /*! \brief Constructor. The node should never be constructed directly. */
  Instruction() = default;
};

/**************** SampleTileFactorInst ****************/

/*! \brief An instruction to sample possible tiling factors */
class SampleTileFactorInstNode : public InstructionNode {
 public:
  /*! \brief The loop to be tiled */
  LoopRV loop;
  /*! \brief The uniform distribution to be sampled from */
  Array<Integer> where;
  /*! \brief The output variables it creates */
  Array<tir::Var> outputs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("loop", &loop);
    v->Visit("where", &where);
    v->Visit("outputs", &outputs);
  }

  static constexpr const char* _type_key = "meta_schedule.SampleTileFactorInst";
  TVM_DECLARE_FINAL_OBJECT_INFO(SampleTileFactorInstNode, InstructionNode);
};

/*!
 * \brief Managed reference to SampleTileFactorInstNode
 * \sa SampleTileFactorInstNode
 */
class SampleTileFactorInst : public Instruction {
 public:
  /*!
   * \brief Constructor
   * \param loop The loop to be tiled
   * \param where The uniform distribution to be sampled from
   * \param outputs The output variables it creates
   */
  explicit SampleTileFactorInst(LoopRV loop, Array<Integer> where, Array<tir::Var> outputs);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(SampleTileFactorInst, Instruction,
                                            SampleTileFactorInstNode);

 protected:
  /*! \brief Constructor. The node should never be constructed directly. */
  SampleTileFactorInst() = default;
};

/**************** GetBlockInst ****************/

/*! \brief An instruction to retrieve a block using its name */
class GetBlockInstNode : public InstructionNode {
 public:
  /*! \brief The name used for retrieval */
  String name;
  /*! \brief The output of the instruction */
  BlockRV output;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("output", &output);
  }

  static constexpr const char* _type_key = "meta_schedule.GetBlockInst";
  TVM_DECLARE_FINAL_OBJECT_INFO(GetBlockInstNode, InstructionNode);
};

/*!
 * \brief Managed reference to GetBlockInstNode
 * \sa GetBlockInstNode
 */
class GetBlockInst : public Instruction {
 public:
  /*!
   * \brief Constructor
   * \param name The name used for retrieval
   * \param output The output of the instruction
   */
  explicit GetBlockInst(String name, BlockRV output);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(GetBlockInst, Instruction, GetBlockInstNode);

 protected:
  /*! \brief Constructor. The node should never be constructed directly. */
  GetBlockInst() = default;
};

/**************** GetAxesInst ****************/

/*! \brief An instruction to retrieve nested loop axes on top of a block */
class GetAxesInstNode : public InstructionNode {
 public:
  /*! \brief The block used for retriving the axes */
  BlockRV block;
  /*! \brief The nested loop axes on top of the block, from outer to inner */
  Array<LoopRV> outputs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("block", &block);
    v->Visit("outputs", &outputs);
  }

  static constexpr const char* _type_key = "meta_schedule.GetAxesInst";
  TVM_DECLARE_FINAL_OBJECT_INFO(GetAxesInstNode, InstructionNode);
};

/*!
 * \brief Managed reference to GetAxesInstNode
 * \sa GetAxesInstNode
 */
class GetAxesInst : public Instruction {
 public:
  /*!
   * \brief Constructor
   * \param block The block used for retriving the axes
   * \param The nested loop axes on top of the block, from outer to inner
   */
  explicit GetAxesInst(BlockRV block, Array<LoopRV> outputs);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(GetAxesInst, Instruction, GetAxesInstNode);

 protected:
  /*! \brief Constructor. The node should never be constructed directly. */
  GetAxesInst() = default;
};

/**************** SplitInst ****************/

/*! \brief An instruction to split a loop by a set of factors */
class SplitInstNode : public InstructionNode {
 public:
  /*! \brief The loop to be split */
  LoopRV loop;
  /*! \brief The factors used to do tiling */
  Array<PrimExpr> factors;
  /*! \brief The output variables */
  Array<LoopRV> outputs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("loop", &loop);
    v->Visit("factors", &factors);
    v->Visit("outputs", &outputs);
  }

  static constexpr const char* _type_key = "meta_schedule.SplitInst";
  TVM_DECLARE_FINAL_OBJECT_INFO(SplitInstNode, InstructionNode);
};

/*!
 * \brief Managed reference to SplitInstNode
 * \sa SplitInstNode
 */
class SplitInst : public Instruction {
 public:
  /*!
   * \brief Constructor
   * \param loop The loop to be split
   * \param factors The factors used to do the split
   * \param outputs The output variables
   */
  explicit SplitInst(LoopRV loop, Array<PrimExpr> factors, Array<LoopRV> outputs);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(SplitInst, Instruction, SplitInstNode);

 protected:
  /*! \brief Constructor. The node should never be constructed directly. */
  SplitInst() = default;
};

/**************** ReorderInst ****************/

/*! \brief An instruction to reorder the given axes */
class ReorderInstNode : public InstructionNode {
 public:
  /*! \brief The order of axes after the reordering, from outer to inner */
  Array<LoopRV> after_axes;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("after_axes", &after_axes); }

  static constexpr const char* _type_key = "meta_schedule.ReorderInst";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReorderInstNode, InstructionNode);
};

/*!
 * \brief Managed reference to ReorderInstNode
 * \sa ReorderInstNode
 */
class ReorderInst : public Instruction {
 public:
  /*!
   * \brief Constructor
   * \param after_axes The order of axes after the reordering, from outer to inner
   */
  explicit ReorderInst(Array<LoopRV> after_axes);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ReorderInst, Instruction, ReorderInstNode);

 protected:
  /*! \brief Constructor. The node should never be constructed directly. */
  ReorderInst() = default;
};

/**************** DecomposeReductionInst ****************/

/*! \brief An instruction for decompose_reduction in TIR */
class DecomposeReductionInstNode : public InstructionNode {
 public:
  /*! \brief The block to be decomposed */
  BlockRV block;
  /*! \brief The loop to be decomposed at */
  LoopRV loop;
  /*! \brief The output variable */
  BlockRV output;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("block", &block);
    v->Visit("loop", &loop);
    v->Visit("output", &output);
  }

  static constexpr const char* _type_key = "meta_schedule.DecomposeReductionInst";
  TVM_DECLARE_FINAL_OBJECT_INFO(DecomposeReductionInstNode, InstructionNode);
};

/*!
 * \brief Managed reference to DecomposeReductionInstNode
 * \sa DecomposeReductionInstNode
 */
class DecomposeReductionInst : public Instruction {
 public:
  /*!
   * \brief Constructor
   * \param block The block to be decomposed
   * \param loop The loop to be decomposed at
   * \param output The output variable
   */
  explicit DecomposeReductionInst(BlockRV block, LoopRV loop, BlockRV output);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(DecomposeReductionInst, Instruction,
                                            DecomposeReductionInstNode);

 protected:
  /*! \brief Constructor. The node should never be constructed directly. */
  DecomposeReductionInst() = default;
};

/**************** GetOnlyConsumerInst ****************/

/*! \brief An instruction for decompose_reduction in TIR */
class GetOnlyConsumerInstNode : public InstructionNode {
 public:
  /*! \brief The producer block */
  BlockRV block;
  /*! \brief The only consumer block of the producer block */
  BlockRV output;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("block", &block);
    v->Visit("output", &output);
  }

  static constexpr const char* _type_key = "meta_schedule.GetOnlyConsumerInst";
  TVM_DECLARE_FINAL_OBJECT_INFO(GetOnlyConsumerInstNode, InstructionNode);
};

/*!
 * \brief Managed reference to GetOnlyConsumerInstNode
 * \sa GetOnlyConsumerInstNode
 */
class GetOnlyConsumerInst : public Instruction {
 public:
  /*!
   * \brief Constructor
   * \param block The producer block
   * \param output The only consumer block of the producer block
   */
  explicit GetOnlyConsumerInst(BlockRV block, BlockRV output);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(GetOnlyConsumerInst, Instruction,
                                            GetOnlyConsumerInstNode);

 protected:
  /*! \brief Constructor. The node should never be constructed directly. */
  GetOnlyConsumerInst() = default;
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_INSTRUCTION_H_
