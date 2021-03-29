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
#ifndef TVM_TIR_SCHEDULE_CONCRETE_SCHEDULE_H_
#define TVM_TIR_SCHEDULE_CONCRETE_SCHEDULE_H_
#include <tvm/arith/analyzer.h>
#include <tvm/tir/schedule/schedule.h>

#include <memory>

namespace tvm {
namespace meta_schedule {
class ScheduleNode;
}  // namespace meta_schedule
}  // namespace tvm

namespace tvm {
namespace tir {

class ConcreteScheduleNode : public ScheduleNode {
  friend class Schedule;
  friend class meta_schedule::ScheduleNode;

 public:
  using TSymbolTable = Map<ObjectRef, ObjectRef>;

 protected:
  /*! \brief The internal state of scheduling */
  ScheduleState state_;
  /*! \brief A symbol table that maps random variables to concrete StmtSRef/Integers */
  TSymbolTable symbol_table_;
  /*! \brief A persistent stateless arithmetic analyzer. */
  std::unique_ptr<arith::Analyzer> analyzer_;

 public:
  void VisitAttrs(tvm::AttrVisitor* v) {
    // `state_` is not visited
    // `symbol_table_` is not visited
    // `analyzer_` is not visitied
  }

  virtual ~ConcreteScheduleNode() = default;

  static constexpr const char* _type_key = "tir.ConcreteSchedule";
  TVM_DECLARE_BASE_OBJECT_INFO(ConcreteScheduleNode, ScheduleNode);

 public:
  ScheduleState state() const final { return state_; }

  Schedule Copy() const override;

  void Seed(int64_t seed = -1) override {
    // do nothing
  }

 public:
  /******** Lookup random variables ********/
  Block Get(const BlockRV& block_rv) const final;

  For Get(const LoopRV& loop_rv) const final;

  int64_t Get(const Var& var_rv) const final;

  PrimExpr Get(const ExprRV& expr_rv) const final;

  StmtSRef GetSRef(const BlockRV& block_rv) const final;

  StmtSRef GetSRef(const LoopRV& loop_rv) const final;

  void RemoveRV(const BlockRV& block_rv) final;

  void RemoveRV(const LoopRV& loop_rv) final;

  void RemoveRV(const VarRV& var_rv) final;

  using ScheduleNode::GetSRef;

 public:
  /******** Sampling ********/

  Array<Var> SamplePerfectTile(const LoopRV& loop_rv,     //
                               int n,                     //
                               int max_innermost_factor,  //
                               Optional<Array<Integer>> decision = NullOpt) override {
    LOG(FATAL) << "NotImplemented";
    throw;
  }

  Var SampleCategorical(const Array<Integer>& candidates,  //
                        const Array<FloatImm>& probs,      //
                        Optional<Integer> decision = NullOpt) override {
    LOG(FATAL) << "NotImplemented";
    throw;
  }

  LoopRV SampleComputeLocation(const BlockRV& block_rv,
                               Optional<Integer> decision = NullOpt) override {
    LOG(FATAL) << "NotImplemented";
    throw;
  }

 public:
  /******** Block/Loop relation ********/

  BlockRV GetBlock(const String& name) override;

  Array<LoopRV> GetAxes(const BlockRV& block_rv) override;

  Array<BlockRV> GetChildBlocks(const BlockRV& block_rv) override;

  Array<BlockRV> GetChildBlocks(const LoopRV& loop_rv) override;

  Array<BlockRV> GetProducers(const BlockRV& block_rv) override;

  Array<BlockRV> GetConsumers(const BlockRV& block_rv) override;

  /******** Schedule: loops ********/

  LoopRV Fuse(const Array<LoopRV>& loop_rvs) override;

  Array<LoopRV> Split(const LoopRV& loop_rv, const Array<Optional<ExprRV>>& factor_rvs) override;

  void Reorder(const Array<LoopRV>& order) override;

  /******** Schedule: compute location ********/

  void ComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv, bool preserve_unit_loop) override;

  void ReverseComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv,
                        bool preserve_unit_loop) override;

  void ComputeInline(const BlockRV& block_rv) override;

  void ReverseComputeInline(const BlockRV& block_rv) override;

  /******** Schedule: parallelize / annotate ********/

  void Vectorize(const LoopRV& loop_rv) override;

  void Parallel(const LoopRV& loop_rv) override;

  void Unroll(const LoopRV& loop_rv) override;

  void Bind(const LoopRV& loop_rv, const IterVar& thread) override;

  void Bind(const LoopRV& loop_rv, const String& thread) override;

  void DoubleBuffer(const BlockRV& block_rv) override;

  void Pragma(const LoopRV& loop_rv, const String& pragma_type,
              const ExprRV& pragma_value) override;

  /******** Schedule: cache read/write ********/

  BlockRV CacheRead(const BlockRV& block_rv, int i, const String& storage_scope) override;

  BlockRV CacheWrite(const BlockRV& block_rv, int i, const String& storage_scope) override;

  /******** Schedule: reduction ********/

  BlockRV RFactor(const LoopRV& loop_rv, int factor_axis) override;

  BlockRV DecomposeReduction(const BlockRV& block_rv, const Optional<LoopRV>& loop_rv) override;

  void MergeReduction(const BlockRV& init_block_rv, const BlockRV& update_block_rv) override;

  /******** Schedule: blockize / tensorize ********/

  BlockRV Blockize(const LoopRV& loop_rv) override;

  void Tensorize(const LoopRV& loop_rv, const TensorIntrin& intrin) override;

  void Tensorize(const LoopRV& loop_rv, const String& intrin_name) override;

  /******** Utility functions ********/
 protected:
  void MakeCopy(ScheduleState* new_state, TSymbolTable* new_symbol_table) const;

  void RemoveFromSymbolTable(const ObjectRef& rv);

  template <class T>
  inline Array<T> SetRV(const Array<StmtSRef>& srefs) {
    Array<T> result;
    result.reserve(srefs.size());
    for (const StmtSRef& sref : srefs) {
      T rv;
      this->symbol_table_.Set(rv, sref);
      result.push_back(rv);
    }
    return result;
  }

  template <class T>
  inline T SetRV(const StmtSRef& sref) {
    T rv;
    this->symbol_table_.Set(rv, sref);
    return rv;
  }

  inline Var SetRV(int64_t number) {
    Var rv;
    this->symbol_table_.Set(rv, Integer(number));
    return rv;
  }

  inline Array<Var> SetRV(const Array<Integer>& numbers) {
    Array<Var> result;
    result.reserve(numbers.size());
    for (int64_t number : numbers) {
      Var rv;
      this->symbol_table_.Set(rv, Integer(number));
      result.push_back(rv);
    }
    return result;
  }

  template <class T>
  inline Array<StmtSRef> FromRV(const Array<T>& rvs) {
    Array<StmtSRef> result;
    result.reserve(rvs.size());
    for (const T& rv : rvs) {
      result.push_back(this->GetSRef(rv));
    }
    return result;
  }
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_CONCRETE_SCHEDULE_H_
