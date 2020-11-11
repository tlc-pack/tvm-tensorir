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

#include <tvm/ir/attrs.h>
#include <tvm/tir/var.h>

namespace tvm {
namespace meta_schedule {

class ScheduleNode;

/**************** Random variables ****************/

/*! \brief A random variable that evaluates to a TIR block */
class BlockRVNode : public runtime::Object {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}
  static constexpr const char* _type_key = "meta_schedule.BlockRV";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockRVNode, Object);
};

/*!
 * \brief Managed reference to BlockRVNode
 * \sa BlockRVNode
 */
class BlockRV : public runtime::ObjectRef {
 public:
  /*! \brief Constructor */
  BlockRV();
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BlockRV, ObjectRef, BlockRVNode);
};

/*! \brief A random variable that evaluates to a TIR loop axis */
class LoopRVNode : public runtime::Object {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}
  static constexpr const char* _type_key = "meta_schedule.LoopRV";
  TVM_DECLARE_FINAL_OBJECT_INFO(LoopRVNode, Object);
};

/*!
 * \brief Managed reference to LoopRVNode
 * \sa LoopRVNode
 */
class LoopRV : public runtime::ObjectRef {
 public:
  /*! \brief Constructor */
  LoopRV();
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(LoopRV, ObjectRef, LoopRVNode);
};

/*! \brief A random variable that evaluates to a TIR block */
class BufferRVNode : public runtime::Object {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}
  static constexpr const char* _type_key = "meta_schedule.BufferRV";
  TVM_DECLARE_FINAL_OBJECT_INFO(BufferRVNode, Object);
};

/*!
 * \brief Managed reference to BufferRVNode
 * \sa BufferRV
 */
class BufferRV : public runtime::ObjectRef {
 public:
  /*! \brief Constructor */
  BufferRV();
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BufferRV, ObjectRef, BufferRVNode);
};

/**************** InstAttrs ****************/

class InstAttrs;

/*! \brief Attributes of an instruction */
class InstAttrsNode : public Object {
 public:
  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  virtual Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                           const Array<ObjectRef>& inputs) const = 0;

  virtual String GetName() const = 0;

  virtual void Export(Array<ObjectRef>* record,
                      const Optional<Array<ObjectRef>>& decision) const = 0;

  static constexpr const char* _type_key = "meta_schedule.InstAttrs";
  TVM_DECLARE_BASE_OBJECT_INFO(InstAttrsNode, Object);
};

/*!
 * \brief Managed reference to InstAttrsNode
 * \sa InstAttrsNode
 */
class InstAttrs : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(InstAttrs, ObjectRef, InstAttrsNode);
};

/**************** Instruction ****************/

/*! \brief Base class for all meta scheduling instrructions */
class InstructionNode : public Object {
 public:
  /*! \brief The input random variables it consumers */
  Array<ObjectRef> inputs;
  /*! \brief The output random variables it produces */
  Array<ObjectRef> outputs;
  /*! \brief The attributes of the instruction */
  InstAttrs inst_attrs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("inputs", &inputs);
    v->Visit("outputs", &outputs);
    v->Visit("inst_attrs", &inst_attrs);
  }

  Array<ObjectRef> Export(const Map<ObjectRef, String>& rv_names,
                          const Optional<Array<ObjectRef>>& decision) const;

  static constexpr const char* _type_key = "meta_schedule.Instruction";
  TVM_DECLARE_FINAL_OBJECT_INFO(InstructionNode, Object);
};

/*!
 * \brief Managed reference to InstructionNode
 * \sa InstructionNode
 */
class Instruction : public ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param inputs The input random variables it consumers
   * \param outputs The output random variables it produces
   * \param inst_attrs The attributes of the instruction
   */
  explicit Instruction(Array<ObjectRef> inputs, Array<ObjectRef> outputs, InstAttrs inst_attrs);

  /*!
   * \brief Apply an instruction to the specific schedule, and return the outputs
   * \param sch The schedule to be applied
   * \param inst_attrs Attributes of the instruction
   * \param inputs The inputs to the instruction
   * \return The outputs of the instruction applied
   */
  static Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const InstAttrs& inst_attrs,
                                          const Array<ObjectRef>& inputs);

  static Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& record,
                                          Map<String, ObjectRef>* named_rvs);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Instruction, ObjectRef, InstructionNode);

 protected:
  /*! \brief Constructor. The node should never be constructed directly. */
  Instruction() = default;
};

/**************** Sampling ****************/

/*! \brief Attrs of the instruction to sample perfect tile factors */
struct SamplePerfectTileAttrs : public InstAttrsNode {
  /*! \brief The number of loops after tiling */
  int n_splits;
  /*! \brief The maximum factor in the innermost loop */
  int max_innermost_factor;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("n_splits", &n_splits);
    v->Visit("max_innermost_factor", &max_innermost_factor);
  }

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param n_splits The number of loops after tiling
   * \param loop The loop to be tiled
   * \param max_innermost_factor The maximum factor in the innermost loop
   * \param outputs Outputs of the instruction
   * \return The instruction created
   */
  static Instruction MakeInst(int n_splits, const LoopRV& loop, int max_innermost_factor,
                              const Array<tir::Var>& outputs);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "SamplePerfectTile"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.SamplePerfectTileAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(SamplePerfectTileAttrs, InstAttrsNode);
};

/*! \brief Attrs of the instruction to sample tiling factors */
struct SampleTileFactorAttrs : public InstAttrsNode {
  /*! \brief The number of loops after tiling */
  int n_splits;
  /*! \brief The distribution to be sampled from */
  Array<Integer> where;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("n_splits", &n_splits);
    v->Visit("where", &where);
  }

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param n_splits The number of loops after tiling
   * \param loop The loop to be tiled
   * \param where The distribution to be sampled from
   * \param outputs Outputs of the instruction
   * \return The instruction created
   */
  static Instruction MakeInst(int n_splits, const LoopRV& loop, const Array<Integer>& where,
                              const Array<tir::Var>& outputs);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "SampleTileFactor"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.SampleTileFactorAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(SampleTileFactorAttrs, InstAttrsNode);
};

/*! \brief Attrs of the instruction to sample fusible loops */
struct SampleFusibleLoopsAttrs : public InstAttrsNode {
  /*! \brief Type of the loop */
  Array<Integer> loop_types;
  /*! \brief The maximum extent of loops */
  int max_extent;
  /*! \brief Whether to include the last loop that makes the extent larger then `max_extent`*/
  bool include_overflow_loop;
  /*! \brief The order of fusion, can be 'outer_to_inner' (0) or 'inner_to_outer' (1) */
  int order;
  /*! \brief The mode of the fusion, can be 'max' (0) or 'rand' (1) */
  int mode;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("loop_types", &loop_types);
    v->Visit("max_extent", &max_extent);
    v->Visit("include_overflow_loop", &include_overflow_loop);
    v->Visit("order", &order);
    v->Visit("mode", &mode);
  }

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param loops The loops to be fused
   * \param loop_types Type of the loop
   * \param max_extent The maximum extent of loops
   * \param include_overflow_loop Whether to include the last loop that makes the extent larger then
   * `max_extent`
   * \param order The order of fusion, can be 'outer_to_inner' (0) or 'inner_to_outer' (1)
   * \param mode The mode of the fusion, can be 'max' (0) or 'rand' (1)
   * \param output The output of the instruction
   * \return The instruction created
   */
  static Instruction MakeInst(const Array<LoopRV>& loops, const Array<Integer>& loop_types,
                              int max_extent, bool include_overflow_loop, int order, int mode,
                              const tir::Var& output);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "SampleFusibleLoops"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.SampleFusibleLoopsAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(SampleFusibleLoopsAttrs, InstAttrsNode);
};

/**************** Block/Loop Relationship ****************/

/*! \brief Attrs of the instruction that gets the only consumer of a specific block */
struct GetOnlyConsumerAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The block to be queried
   * \param output The output of the query
   * \return The instruction created
   */
  static Instruction MakeInst(const BlockRV& block, const BlockRV& output);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "GetOnlyConsumer"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.GetOnlyConsumerAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(GetOnlyConsumerAttrs, InstAttrsNode);
};

/*! \brief Attrs of the instruction that gets a specific block by its name */
struct GetBlockAttrs : public InstAttrsNode {
  /*! \brief The name of the block */
  String name;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("name", &name); }

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param name The name of the block
   * \param output The output of the query
   * \return The instruction created
   */
  static Instruction MakeInst(const String& name, const BlockRV& output);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "GetBlock"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.GetBlockAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(GetBlockAttrs, InstAttrsNode);
};

/*! \brief Attrs of the instruction that gets loop axes on top of a specifc block */
struct GetAxesAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The name of the block
   * \param outputs The outputs of the query
   * \return The instruction created
   */
  static Instruction MakeInst(const BlockRV& block, const Array<LoopRV>& outputs);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "GetAxes"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.GetAxesAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(GetAxesAttrs, InstAttrsNode);
};

/*! \brief Attrs of the instruction that gets the buffers the block reads */
struct GetReadBuffersAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The name of the block
   * \param outputs The outputs of the query
   * \return The instruction created
   */
  static Instruction MakeInst(const BlockRV& block, const Array<BufferRV>& outputs);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "GetReadBuffers"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.GetReadBuffersAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(GetReadBuffersAttrs, InstAttrsNode);
};

/*! \brief Attrs of the instruction that gets the buffers the block writes */
struct GetWriteBuffersAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The name of the block
   * \param outputs The outputs of the query
   * \return The instruction created
   */
  static Instruction MakeInst(const BlockRV& block, const Array<BufferRV>& outputs);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "GetWriteBuffers"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.GetWriteBuffersAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(GetWriteBuffersAttrs, InstAttrsNode);
};

struct GetRootBlocksAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param outputs The outputs of the instruction
   * \return The instruction created
   */
  static Instruction MakeInst(const Array<BlockRV>& outputs);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "GetRootBlocks"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.GetRootBlocksAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(GetRootBlocksAttrs, InstAttrsNode);
};

struct GetLeafBlocksAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param outputs The outputs of the instruction
   * \return The instruction created
   */
  static Instruction MakeInst(const Array<BlockRV>& outputs);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "GetLeafBlocks"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.GetLeafBlocksAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(GetLeafBlocksAttrs, InstAttrsNode);
};

/**************** Scheduling Primitives ****************/

struct MarkLoopTypeAttrs : public InstAttrsNode {
  /*! \brief The loop annotation */
  String mark;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("mark", &mark); }

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param loops The loops to be mark
   * \param mark The annotation
   * \param first_n To mark the first n loops
   * \param last_n To mark the last n loops
   * \return The instruction created
   */
  static Instruction MakeInst(const Array<LoopRV>& loops, const String& mark, const PrimExpr& first,
                              const PrimExpr& last_n);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "MarkLoopType"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.MarkLoopTypeAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(MarkLoopTypeAttrs, InstAttrsNode);
};

struct MarkBlockTypeAttrs : public InstAttrsNode {
  /*! \brief The loop annotation */
  String mark;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("mark", &mark); }

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The block to be marked
   * \param mark The annotation
   * \return The instruction created
   */
  static Instruction MakeInst(const BlockRV& block, const String& mark);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "MarkBlockType"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.MarkBlockTypeAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(MarkBlockTypeAttrs, InstAttrsNode);
};

struct FuseAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param loops The loops to be fused
   * \param output The output of the instruction
   * \return The instruction created
   */
  static Instruction MakeInst(const Array<LoopRV>& loops, const LoopRV& output);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "Fuse"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.FuseAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(FuseAttrs, InstAttrsNode);
};

/*! \brief Attrs of the instruction that applies loop splitting */
struct SplitAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param loop The loop to be split
   * \param factors Thee splitting factors
   * \param outputs The outputs of the query
   * \return The instruction created
   */
  static Instruction MakeInst(const LoopRV& loop, const Array<Optional<PrimExpr>>& factors,
                              const Array<LoopRV>& outputs);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "Split"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.SplitAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(SplitAttrs, InstAttrsNode);
};

/*! \brief Attrs of the instruction that applies loop reordering */
struct ReorderAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param after_axes The axes to be reordered
   * \return The instruction created
   */
  static Instruction MakeInst(const Array<LoopRV>& after_axes);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "Reorder"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.ReorderAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReorderAttrs, InstAttrsNode);
};

/*! \brief Attrs of the instruction that applies reverse_compute_at */
struct ComputeAtAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The block to be moved
   * \param loop The loop to be moved to
   * \return The instruction created
   */
  static Instruction MakeInst(const BlockRV& block, const LoopRV& loop);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "ComputeAt"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.ComputeAtAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(ComputeAtAttrs, InstAttrsNode);
};

/*! \brief Attrs of the instruction that applies reverse_compute_at */
struct ReverseComputeAtAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The block to be moved
   * \param loop The loop to be moved to
   * \return The instruction created
   */
  static Instruction MakeInst(const BlockRV& block, const LoopRV& loop);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "ReverseComputeAt"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.ReverseComputeAtAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReverseComputeAtAttrs, InstAttrsNode);
};

/*! \brief Attrs of the instruction that applies compute_inline */
struct ComputeInlineAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The block to be computed inline
   * \return The instruction created
   */
  static Instruction MakeInst(const BlockRV& block);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "ComputeInline"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.ComputeInlineAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(ComputeInlineAttrs, InstAttrsNode);
};

/*! \brief Attrs of the instruction that applies compute_inline */
struct ReverseComputeInlineAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The block to be reverse computed inline
   * \return The instruction created
   */
  static Instruction MakeInst(const BlockRV& block);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "ReverseComputeInline"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.ReverseComputeInlineAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReverseComputeInlineAttrs, InstAttrsNode);
};

/*! \brief Attrs of the instruction that applies cache_write */
struct CacheReadAttrs : public InstAttrsNode {
  /*! \brief The storage scope of the instruction cache_write */
  String storage_scope;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("storage_scope", &storage_scope); }

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param buffer The buffer to be cached
   * \param storage_scope The storage scope of the instruction
   * \param output The output of the instruction
   * \return The instruction created
   */
  static Instruction MakeInst(const BufferRV& buffer, const String& storage_scope,
                              const BlockRV& output);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "CacheRead"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.CacheReadAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(CacheReadAttrs, InstAttrsNode);
};

/*! \brief Attrs of the instruction that applies cache_write */
struct CacheWriteAttrs : public InstAttrsNode {
  /*! \brief The storage scope of the instruction cache_write */
  String storage_scope;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("storage_scope", &storage_scope); }

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The buffer to be cached
   * \param storage_scope The storage scope of the instruction
   * \param output The output of the instruction
   * \return The instruction created
   */
  static Instruction MakeInst(const BlockRV& block, const String& storage_scope,
                              const BlockRV& output);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "CacheWrite"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.CacheWriteAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(CacheWriteAttrs, InstAttrsNode);
};

/*! \brief Attrs of the instruction that applies blockize */
struct BlockizeAttrs : public InstAttrsNode {
  /*! \brief The execution scope of the instruction blockize */
  String exec_scope;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("exec_scope", &exec_scope); }

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param loop The loop to be blockized
   * \param exec_scope The execution scope
   * \param output The output of the instruction
   * \return The instruction created
   */
  static Instruction MakeInst(const LoopRV& block, const String& exec_scope, const BlockRV& output);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "Blockize"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.BlockizeAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockizeAttrs, InstAttrsNode);
};

/*! \brief Attrs of the instruction that applies decompose_reduction */
struct DecomposeReductionAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The reduction block to be decomposed
   * \param loop The loop to be decomposed at
   * \param output The output of the instruction
   * \return The instruction created
   */
  static Instruction MakeInst(const BlockRV& block, const LoopRV& loop, const BlockRV& output);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "DecomposeReduction"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.DecomposeReductionAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(DecomposeReductionAttrs, InstAttrsNode);
};

/*! \brief Attrs of an NOP that indicates entrance of post processing */
struct EnterPostProcAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The block to be queried
   * \param output The output of the query
   * \return The instruction created
   */
  static Instruction MakeInst();

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch,
                                   const Array<ObjectRef>& inputs) const override;

  void Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const override;

  static InstAttrs Import(const Array<ObjectRef>& record);

  static String Name() { return "EnterPostProc"; }
  String GetName() const override { return Name(); }
  static constexpr const char* _type_key = "meta_schedule.attrs.EnterPostProcAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(EnterPostProcAttrs, InstAttrsNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_INSTRUCTION_H_
