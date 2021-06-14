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

#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/var.h>

namespace tvm {
namespace meta_schedule {

using BlockRV = tir::BlockRV;
using BlockRVNode = tir::BlockRVNode;
using LoopRV = tir::LoopRV;
using LoopRVNode = tir::LoopRVNode;
using ExprRV = tir::ExprRV;

class Schedule;

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
  virtual Array<ObjectRef> Apply(const Schedule& sch, const Array<Optional<ObjectRef>>& inputs,
                                 const Optional<ObjectRef>& decisions) const = 0;

  // We intentionally consider sampling instructions as pure too
  virtual bool IsPure() const = 0;

  static constexpr const char* _type_key = "meta_schedule.InstAttrs";
  TVM_DECLARE_BASE_OBJECT_INFO(InstAttrsNode, Object);
  friend class InstructionNode;

 private:
  virtual void Serialize(Array<ObjectRef>* record, const Optional<ObjectRef>& decision) const = 0;

  virtual void AsPython(std::ostream& os, const Array<String>& inputs, const Array<String>& outputs,
                        const Optional<ObjectRef>& decision) const = 0;

  virtual String GetName() const = 0;
};

/*!
 * \brief Managed reference to InstAttrsNode
 * \sa InstAttrsNode
 */
class InstAttrs : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(InstAttrs, ObjectRef, InstAttrsNode);
};

#define TVM_META_SCHEDULE_DEFINE_INST_ATTRS(Cls, TypeKey, InstName, Pure)                       \
 private:                                                                                       \
  String GetName() const override { return String(_name); }                                     \
  void Serialize(Array<ObjectRef>* record, const Optional<ObjectRef>& decision) const override; \
  static InstAttrs Deserialize(const Array<ObjectRef>& record, Optional<ObjectRef>* decision);  \
                                                                                                \
 public:                                                                                        \
  static constexpr const char* _name = InstName;                                                \
  static constexpr const char* _type_key = TypeKey;                                             \
  Array<ObjectRef> Apply(const Schedule& sch, const Array<Optional<ObjectRef>>& inputs,         \
                         const Optional<ObjectRef>& decision) const override;                   \
  bool IsPure() const override { return Pure; }                                                 \
  void AsPython(std::ostream& os, const Array<String>& inputs, const Array<String>& outputs,    \
                const Optional<ObjectRef>& decision) const override;                            \
  friend class InstructionNode;                                                                 \
  TVM_DECLARE_FINAL_OBJECT_INFO(Cls, InstAttrsNode);

/**************** Instruction ****************/

/*! \brief Base class for all meta scheduling instrructions */
class InstructionNode : public Object {
 public:
  /*! \brief The input random variables it consumers */
  Array<Optional<ObjectRef>> inputs;
  /*! \brief The output random variables it produces */
  Array<ObjectRef> outputs;
  /*! \brief The attributes of the instruction */
  InstAttrs inst_attrs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("inputs", &inputs);
    v->Visit("outputs", &outputs);
    v->Visit("inst_attrs", &inst_attrs);
  }

  Array<ObjectRef> Serialize(const Map<ObjectRef, String>& rv_names,
                             const Optional<ObjectRef>& decision) const;

  static Array<ObjectRef> Deserialize(const Array<ObjectRef>& record,
                                      Map<String, ObjectRef>* named_rvs, const Schedule& sch);

  void AsPython(std::ostream& os, const Map<ObjectRef, String>& rv_names,
                const Optional<ObjectRef>& decision) const;

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
  explicit Instruction(Array<Optional<ObjectRef>> inputs, Array<ObjectRef> outputs,
                       InstAttrs inst_attrs);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Instruction, ObjectRef, InstructionNode);

 protected:
  /*! \brief Constructor. The node should never be constructed directly. */
  Instruction() = default;
};

/**************** Sampling ****************/

/*! \brief Attrs of the instruction to sample perfect tile factors */
struct SamplePerfectTileAttrs : public InstAttrsNode {
  /*! \brief The number of loops after tiling */
  int n;
  /*! \brief The maximum factor in the innermost loop */
  int max_innermost_factor;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("n", &n);
    v->Visit("max_innermost_factor", &max_innermost_factor);
  }

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param n The number of loops after tiling
   * \param loop The loop to be tiled
   * \param max_innermost_factor The maximum factor in the innermost loop
   * \param outputs Outputs of the instruction
   * \return The instruction created
   */
  static Instruction Make(const LoopRV& loop, int n, int max_innermost_factor,
                          const Array<tir::Var>& outputs);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(SamplePerfectTileAttrs,
                                      "meta_schedule.attrs.SamplePerfectTileAttrs",
                                      "SamplePerfectTile", true);
};

/*! \brief Attrs of the instruction to sample from a categorical distribution */
struct SampleCategoricalAttrs : public InstAttrsNode {
  /*! \brief The candidates */
  Array<Integer> candidates;
  /*! \brief The probability distribution of the candidates */
  Array<FloatImm> probs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("candidates", &candidates);
    v->Visit("probs", &probs);
  }

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param candidates The candidates
   * \param probs The probability distribution of the candidates
   * \param output The output the instruction
   * \return The instruction created
   */
  static Instruction Make(const Array<Integer>& candidates, const Array<FloatImm>& probs,
                          const tir::Var& output);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(SampleCategoricalAttrs,
                                      "meta_schedule.attrs.SampleCategoricalAttrs",
                                      "SampleCategorical", true);
};

/*! \brief Attrs of the instruction to sample a compute-at location from a block */
struct SampleComputeLocationAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \return The instruction created
   */
  static Instruction Make(const BlockRV& block, const LoopRV& output);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(SampleComputeLocationAttrs,
                                      "meta_schedule.attrs.SampleComputeLocationAttrs",
                                      "SampleComputeLocation", true);
};

/**************** Block/Loop Relationship ****************/

/*! \brief Attrs of the instruction that gets the producers of a specific block */
struct GetProducersAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The block to be queried
   * \param output The output of the query
   * \return The instruction created
   */
  static Instruction Make(const BlockRV& block, const Array<BlockRV>& outputs);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(GetProducersAttrs,                        //
                                      "meta_schedule.attrs.GetProducersAttrs",  //
                                      "GetProducers", true);
};

/*! \brief Attrs of the instruction that gets the consumers of a specific block */
struct GetConsumersAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The block to be queried
   * \param output The output of the query
   * \return The instruction created
   */
  static Instruction Make(const BlockRV& block, const Array<BlockRV>& outputs);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(GetConsumersAttrs,                        //
                                      "meta_schedule.attrs.GetConsumersAttrs",  //
                                      "GetConsumers", true);
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
  static Instruction Make(const String& name, const BlockRV& output);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(GetBlockAttrs,                        //
                                      "meta_schedule.attrs.GetBlockAttrs",  //
                                      "GetBlock", true);
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
  static Instruction Make(const BlockRV& block, const Array<LoopRV>& outputs);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(GetAxesAttrs,                        //
                                      "meta_schedule.attrs.GetAxesAttrs",  //
                                      "GetAxes", true);
};

/**************** Scheduling Primitives ****************/

struct MarkLoopAttrs : public InstAttrsNode {
  /*! \brief The loop annotation key */
  String ann_key;
  /*! \brief The loop annotation value */
  String ann_val;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("ann_key", &ann_key);
    v->Visit("ann_val", &ann_val);
  }

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param loops The loop to be mark
   * \param ann_key The loop annotation key
   * \param ann_val The loop annotation value
   * \return The instruction created
   */
  static Instruction Make(const LoopRV& loop, const String& ann_key, const PrimExpr& ann_val);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(MarkLoopAttrs,                        //
                                      "meta_schedule.attrs.MarkLoopAttrs",  //
                                      "MarkLoop", false);
};

struct MarkBlockAttrs : public InstAttrsNode {
  /*! \brief The loop annotation key */
  String ann_key;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("ann_key", &ann_key); }

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The block to be marked
   * \param ann_key The loop annotation key
   * \param ann_val The loop annotation value
   * \return The instruction created
   */
  static Instruction Make(const BlockRV& block, const String& ann_key, const PrimExpr& ann_val);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(MarkBlockAttrs,                        //
                                      "meta_schedule.attrs.MarkBlockAttrs",  //
                                      "MarkBlock", false);
};

struct FuseAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param loops The loops to be fused
   * \param output The output of the instruction
   * \return The instruction created
   */
  static Instruction Make(const Array<LoopRV>& loops, const LoopRV& output);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(FuseAttrs,                        //
                                      "meta_schedule.attrs.FuseAttrs",  //
                                      "Fuse", false);
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
  static Instruction Make(const LoopRV& loop, const Array<Optional<PrimExpr>>& factors,
                          const Array<LoopRV>& outputs);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(SplitAttrs,                        //
                                      "meta_schedule.attrs.SplitAttrs",  //
                                      "Split", false);
};

/*! \brief Attrs of the instruction that applies loop reordering */
struct ReorderAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param after_axes The axes to be reordered
   * \return The instruction created
   */
  static Instruction Make(const Array<LoopRV>& after_axes);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(ReorderAttrs,                        //
                                      "meta_schedule.attrs.ReorderAttrs",  //
                                      "Reorder", false);
};

/*! \brief Attrs of the instruction that applies reverse_compute_at */
struct ComputeAtAttrs : public InstAttrsNode {
  /*! \brief If true, keep the trivial loops whose extent is 1 */
  bool preserve_unit_loop;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The block to be moved
   * \param loop The loop to be moved to
   * \return The instruction created
   */
  static Instruction Make(const BlockRV& block, const LoopRV& loop, bool preserve_unit_loop);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(ComputeAtAttrs,                        //
                                      "meta_schedule.attrs.ComputeAtAttrs",  //
                                      "ComputeAt", false);
};

/*! \brief Attrs of the instruction that applies reverse_compute_at */
struct ReverseComputeAtAttrs : public InstAttrsNode {
  /*! \brief If true, keep the trivial loops whose extent is 1 */
  bool preserve_unit_loop;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The block to be moved
   * \param loop The loop to be moved to
   * \return The instruction created
   */
  static Instruction Make(const BlockRV& block, const LoopRV& loop, bool preserve_unit_loop);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(ReverseComputeAtAttrs,                        //
                                      "meta_schedule.attrs.ReverseComputeAtAttrs",  //
                                      "ReverseComputeAt", false);
};

/*! \brief Attrs of the instruction that applies compute_inline */
struct ComputeInlineAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The block to be computed inline
   * \return The instruction created
   */
  static Instruction Make(const BlockRV& block);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(ComputeInlineAttrs,                        //
                                      "meta_schedule.attrs.ComputeInlineAttrs",  //
                                      "ComputeInline", false);
};

/*! \brief Attrs of the instruction that applies reverse_compute_inline */
struct ReverseComputeInlineAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The block to be reverse computed inline
   * \return The instruction created
   */
  static Instruction Make(const BlockRV& block);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(ReverseComputeInlineAttrs,                        //
                                      "meta_schedule.attrs.ReverseComputeInlineAttrs",  //
                                      "ReverseComputeInline", false);
};

/*! \brief Attrs of the instruction that applies cache_write */
struct CacheReadAttrs : public InstAttrsNode {
  /*! \brief The index of the buffer in block's write region */
  int i;
  /*! \brief The storage scope of the instruction cache_write */
  String storage_scope;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("i", &i);
    v->Visit("storage_scope", &storage_scope);
  }

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The buffer to be cached
   * \param i The index of the buffer in block's read region
   * \param storage_scope The storage scope of the instruction
   * \param output The output of the instruction
   * \return The instruction created
   */
  static Instruction Make(const BlockRV& block, int i, const String& storage_scope,
                          const BlockRV& output);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(CacheReadAttrs,                        //
                                      "meta_schedule.attrs.CacheReadAttrs",  //
                                      "CacheRead", false);
};

/*! \brief Attrs of the instruction that applies cache_write */
struct CacheWriteAttrs : public InstAttrsNode {
  /*! \brief The index of the buffer in block's write region */
  int i;
  /*! \brief The storage scope of the instruction cache_write */
  String storage_scope;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("i", &i);
    v->Visit("storage_scope", &storage_scope);
  }

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The buffer to be cached
   * \param i The index of the buffer in block's write region
   * \param storage_scope The storage scope of the instruction
   * \param output The output of the instruction
   * \return The instruction created
   */
  static Instruction Make(const BlockRV& block, int i, const String& storage_scope,
                          const BlockRV& output);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(CacheWriteAttrs,                        //
                                      "meta_schedule.attrs.CacheWriteAttrs",  //
                                      "CacheWrite", false);
};

/*! \brief Attrs of the instruction that applies blockize */
struct BlockizeAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param loop The loop to be blockized
   * \param output The output of the instruction
   * \return The instruction created
   */
  static Instruction Make(const LoopRV& block, const BlockRV& output);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(BlockizeAttrs,                        //
                                      "meta_schedule.attrs.BlockizeAttrs",  //
                                      "Blockize", false);
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
  static Instruction Make(const BlockRV& block, const Optional<LoopRV>& loop,
                          const BlockRV& output);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(DecomposeReductionAttrs,                        //
                                      "meta_schedule.attrs.DecomposeReductionAttrs",  //
                                      "DecomposeReduction", false);
};

/*! \brief Attrs of the instruction that applies decompose_reduction */
struct TensorizeAttrs : public InstAttrsNode {
  /*! \brief Name of the tensor intrinsic */
  String tensor_intrin_name;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("tensor_intrin_name", &tensor_intrin_name); }

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The reduction block to be decomposed
   * \param loop The loop to be decomposed at
   * \param output The output of the instruction
   * \return The instruction created
   */
  static Instruction Make(const LoopRV& loop, const String& tensor_intrin_name);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(TensorizeAttrs,                        //
                                      "meta_schedule.attrs.TensorizeAttrs",  //
                                      "Tensorize", false);
};

/*! \brief Attrs of the instruction that applies parallel */
struct ParallelAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param loop The loop to be parallelized
   * \return The instruction created
   */
  static Instruction Make(const LoopRV& loop);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(ParallelAttrs,                        //
                                      "meta_schedule.attrs.ParallelAttrs",  //
                                      "parallel", false);
};

/*! \brief Attrs of the instruction that applies vectorize */
struct VectorizeAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param loop The loop to be vectorized
   * \return The instruction created
   */
  static Instruction Make(const LoopRV& loop);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(VectorizeAttrs,                        //
                                      "meta_schedule.attrs.VectorizeAttrs",  //
                                      "vectorize", false);
};

/*! \brief Attrs of the instruction that applies unroll */
struct UnrollAttrs : public InstAttrsNode {
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param loop The loop to be vectorized
   * \return The instruction created
   */
  static Instruction Make(const LoopRV& loop);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(UnrollAttrs,                        //
                                      "meta_schedule.attrs.UnrollAttrs",  //
                                      "unroll", false);
};

/*! \brief Attrs of the instruction that applies rfactor */
struct RFactorAttrs : public InstAttrsNode {
  /*! \brief The factor_axis in the result buffer */
  int factor_axis;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("factor_axis", &factor_axis); }

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param loop The loop to be applied rfactor
   * \param factor_axis The factor_axis in the result buffer
   * \param output The output of the instruction
   * \return The instruction created
   */
  static Instruction Make(const LoopRV& loop, int factor_axis, const BlockRV& output);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(RFactorAttrs,                        //
                                      "meta_schedule.attrs.RFactorAttrs",  //
                                      "rfactor", false);
};

/*! \brief Attrs of the instruction that applies bind */
struct BindAttrs : public InstAttrsNode {
  String thread_axis;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("thread_axis", &thread_axis); }

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param loop The loop to be vectorized
   * \return The instruction created
   */
  static Instruction Make(const LoopRV& loop, const String& thread_axis);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(BindAttrs,                        //
                                      "meta_schedule.attrs.BindAttrs",  //
                                      "bind", false);
};

/*! \brief Attrs of the instruction that applies set_scope */
struct SetScopeAttrs : public InstAttrsNode {
  /*! \brief The index of the buffer in block's write region */
  int i;
  /*! \brief The storage scope to be set */
  String storage_scope;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("i", &i);
    v->Visit("storage_scope", &storage_scope);
  }

  /*!
   * \brief Set the storage scope of a buffer, where the buffer is given as the i-th write buffer
   *        of the input block
   * \param block The producer of the buffer
   * \param i The index of the buffer in block's write region
   * \param storage_scope The The storage scope to be set
   * \return The instruction created
   */
  static Instruction Make(const BlockRV& block, int i, const String& storage_scope);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(SetScopeAttrs,                        //
                                      "meta_schedule.attrs.SetScopeAttrs",  //
                                      "SetScope", false);
};

/*! \brief Attrs of the instruction that applies storage_align */
struct StorageAlignAttrs : public InstAttrsNode {
  /*! \brief The index of the buffer in block's write region */
  int buffer_index;
  /*! The dimension to be specified for alignment */
  int axis;
  /*! The factor multiple of alignment */
  int factor;
  /*! The required offset factor */
  int offset;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("buffer_index", &buffer_index);
    v->Visit("axis", &axis);
    v->Visit("factor", &factor);
    v->Visit("offset", &offset);
  }

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The producer block of the buffer
   * \param buffer_index The index of the buffer in block's write region
   * \param axis The dimension to be specified for alignment
   * \param factor The factor multiple of alignment
   * \param offset The required offset factor
   * \return The instruction created
   */
  static Instruction Make(const BlockRV& block, int buffer_index, int axis, int factor, int offset);

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(StorageAlignAttrs,                        //
                                      "meta_schedule.attrs.StorageAlignAttrs",  //
                                      "storage_align", false);
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
  static Instruction Make();

  TVM_META_SCHEDULE_DEFINE_INST_ATTRS(EnterPostProcAttrs,                        //
                                      "meta_schedule.attrs.EnterPostProcAttrs",  //
                                      "EnterPostProc", false);
};

#undef TVM_META_SCHEDULE_DEFINE_INST_ATTRS

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_INSTRUCTION_H_
