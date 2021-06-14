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
#include "../utils.h"

namespace tvm {
namespace tir {

Array<StmtSRef> GetBlocks(const ScheduleState& self, const String& name) {
  Array<StmtSRef> result;
  for (const auto& kv : self->block_info) {
    const StmtSRef& block_sref = kv.first;
    const auto* block = TVM_SREF_TO_BLOCK(block, block_sref);
    if (block->name_hint == name) {
      result.push_back(block_sref);
    }
  }
  return result;
}

Array<StmtSRef> GetAxes(const ScheduleState& self, const StmtSRef& block_sref) {
  std::vector<StmtSRef> result;
  for (StmtSRefNode* parent = block_sref->parent; parent && parent->stmt->IsInstance<ForNode>();
       parent = parent->parent) {
    result.push_back(GetRef<StmtSRef>(parent));
  }
  return {result.rbegin(), result.rend()};
}

Array<StmtSRef> GetChildBlocks(const ScheduleState& self, const StmtSRef& parent_sref,
                               bool inclusive) {
  struct Collector : public StmtVisitor {
   private:
    void VisitStmt_(const BlockNode* block) final { result.push_back(self->stmt2ref.at(block)); }

   public:
    explicit Collector(const ScheduleState& self) : self(self) {}

    const ScheduleState& self;
    Array<StmtSRef> result;
  };
  Collector collector(self);
  if (inclusive) {
    collector(GetRef<Stmt>(parent_sref->stmt));
  } else if (parent_sref->stmt->IsInstance<ForNode>()) {
    const auto* loop = static_cast<const ForNode*>(parent_sref->stmt);
    collector(loop->body);
  } else if (parent_sref->stmt->IsInstance<BlockNode>()) {
    const auto* block = static_cast<const BlockNode*>(parent_sref->stmt);
    collector(block->body);
  }
  return std::move(collector.result);
}

Array<StmtSRef> GetProducers(const ScheduleState& self, const StmtSRef& block_sref) {
  Array<Dependency> pred_edges = self->GetBlockScope(GetScopeRoot(block_sref))  //
                                     ->GetDepsByDst(block_sref);
  Array<StmtSRef> results;
  results.reserve(pred_edges.size());
  for (const Dependency& edge : pred_edges) {
    if (edge->kind == DepKind::kRAW || edge->kind == DepKind::kWAW) {
      results.push_back(edge->src);
    }
  }
  return results;
}

Array<StmtSRef> GetConsumers(const ScheduleState& self, const StmtSRef& block_sref) {
  Array<Dependency> succ_edges = self->GetBlockScope(GetScopeRoot(block_sref))  //
                                     ->GetDepsBySrc(block_sref);
  Array<StmtSRef> results;
  results.reserve(succ_edges.size());
  for (const Dependency& edge : succ_edges) {
    if (edge->kind == DepKind::kRAW || edge->kind == DepKind::kWAW) {
      results.push_back(edge->dst);
    }
  }
  return results;
}

struct GetBlockTraits : public UnpackedInstTraits<GetBlockTraits> {
  static constexpr const char* kName = "GetBlock";
  static constexpr bool kIsPure = true;

 private:
  static constexpr size_t kNumInputs = 0;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, String name) { return sch->GetBlock(name); }

  static String UnpackedAsPython(Array<String> outputs, String name) {
    PythonAPICall py("get_block");
    py.Attr("name", name);
    py.Output(outputs[0]);
    return py.Str();
  }

  template <typename>
  friend struct UnpackedInstTraits;
};

struct GetAxesTraits : public UnpackedInstTraits<GetAxesTraits> {
  static constexpr const char* kName = "GetAxes";
  static constexpr bool kIsPure = true;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static Array<LoopRV> UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv) {
    return sch->GetAxes(block_rv);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv) {
    PythonAPICall py("get_axes");
    py.Input("block", block_rv);
    py.Outputs(outputs);
    return py.Str();
  }

  template <typename>
  friend struct UnpackedInstTraits;
};

struct GetChildBlocksTraits : public UnpackedInstTraits<GetChildBlocksTraits> {
  static constexpr const char* kName = "GetChildBlocks";
  static constexpr bool kIsPure = true;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static Array<BlockRV> UnpackedApplyToSchedule(Schedule sch, ObjectRef block_or_loop_rv) {
    if (const auto* block = block_or_loop_rv.as<BlockRVNode>()) {
      return sch->GetChildBlocks(GetRef<BlockRV>(block));
    }
    if (const auto* loop = block_or_loop_rv.as<LoopRVNode>()) {
      return sch->GetChildBlocks(GetRef<LoopRV>(loop));
    }
    LOG(FATAL) << "TypeError: Expected Block or Loop, but gets: " << block_or_loop_rv->GetTypeKey();
    throw;
  }

  static String UnpackedAsPython(Array<String> outputs, String block_or_loop_rv) {
    PythonAPICall py("get_child_blocks");
    py.Input("block_or_loop", block_or_loop_rv);
    py.Outputs(outputs);
    return py.Str();
  }

  template <typename>
  friend struct UnpackedInstTraits;
};

struct GetProducersTraits : public UnpackedInstTraits<GetProducersTraits> {
  static constexpr const char* kName = "GetProducers";
  static constexpr bool kIsPure = true;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static Array<BlockRV> UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv) {
    return sch->GetProducers(block_rv);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv) {
    PythonAPICall py("get_producers");
    py.Input("block", block_rv);
    py.Outputs(outputs);
    return py.Str();
  }

  template <typename>
  friend struct UnpackedInstTraits;
};

struct GetConsumersTraits : public UnpackedInstTraits<GetConsumersTraits> {
  static constexpr const char* kName = "GetConsumers";
  static constexpr bool kIsPure = true;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static Array<BlockRV> UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv) {
    return sch->GetConsumers(block_rv);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv) {
    PythonAPICall py("get_consumers");
    py.Input("block", block_rv);
    py.Outputs(outputs);
    return py.Str();
  }

  template <typename>
  friend struct UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND(GetBlockTraits);
TVM_REGISTER_INST_KIND(GetAxesTraits);
TVM_REGISTER_INST_KIND(GetChildBlocksTraits);
TVM_REGISTER_INST_KIND(GetProducersTraits);
TVM_REGISTER_INST_KIND(GetConsumersTraits);

}  // namespace tir
}  // namespace tvm
