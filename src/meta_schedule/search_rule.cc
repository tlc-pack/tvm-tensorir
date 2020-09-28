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
#include "./analysis.h"
#include "./search.h"
#include "./utils.h"

namespace tvm {
namespace meta_schedule {

String AsHybrid(const Schedule& sch) {
  const auto* f = runtime::Registry::Get("hybrid.AsHybrid");
  String s = (*f)(sch->sch->func, false);
  return s;
}

class AlwaysInline {
 public:
  AlwaysInline() = default;

  RulePackedArgs operator()(Schedule sch, BlockRV block_rv) {
    static const Op& op_exp = Op::Get("tir.exp");
    if (HasReduceBlockVar(sch, block_rv) || IsOutputBlock(sch, block_rv)) {
      return RulePackedArgs(sch);
    }
    if (HasBranch(sch, block_rv) || CountOp(sch, block_rv, op_exp)) {
      return RulePackedArgs(sch);
    }
    if (Optional<Array<Bool>> access = InspectLoadIndices(sch, block_rv)) {
      CHECK_EQ(access.value().size(), 3);
      bool injective = access.value()[1];
      bool order = access.value()[2];
      if (!order || !injective) {
        return RulePackedArgs(sch);
      }
    } else {
      return RulePackedArgs(sch);
    }
    sch->ComputeInline(block_rv);
    return RulePackedArgs(/*proceed=*/{}, /*ignored=*/{sch});
  }

  static SearchRule MakeRule() {
    auto invoke = [](Schedule sch, BlockRV block) -> RulePackedArgs {
      AlwaysInline rule;
      return rule(sch, block);
    };
    return SearchRule("always_inline", invoke);
  }
};

class MultiLevelTiling {
 public:
  String tiling_structure;

  explicit MultiLevelTiling(String tiling_structure)
      : tiling_structure(std::move(tiling_structure)) {}

  RulePackedArgs operator()(Schedule sch, BlockRV block_rv) {
    // Right now it only works with a leaf block with a single statement
    if (NeedsMultiLevelTiling(sch, block_rv)) {
      DoMultiLevelTiling(sch, block_rv, tiling_structure);
      return RulePackedArgs(/*proceed=*/{}, /*ignored=*/{sch});
    }
    return RulePackedArgs(sch);
  }

  static SearchRule MakeRule(String tiling_structure) {
    auto invoke = [tiling_structure](Schedule sch, BlockRV block) -> RulePackedArgs {
      MultiLevelTiling rule(tiling_structure);
      return rule(sch, block);
    };
    return SearchRule("multi_level_tiling", invoke);
  }
};

class MultiLevelTilingWithFusion {
 public:
  String tiling_structure;

  explicit MultiLevelTilingWithFusion(String tiling_structure)
      : tiling_structure(std::move(tiling_structure)) {}

  RulePackedArgs operator()(Schedule sch, BlockRV block_rv) {
    // Rule out the possibility that it does not need multi-level tiling
    if (!NeedsMultiLevelTiling(sch, block_rv)) {
      return RulePackedArgs(sch);
    }
    // Get the only consumer
    Optional<BlockRV> opt_consumer_rv = sch->GetOnlyConsumer(block_rv);
    if (!opt_consumer_rv.defined()) {
      return RulePackedArgs(sch);
    }
    // Check elementwise-match
    BlockRV consumer_rv = opt_consumer_rv.value();
    if (HasReduceBlockVar(sch, block_rv) && HasReduceBlockVar(sch, consumer_rv)) {
      return RulePackedArgs(sch);
    }
    if (!IsElementWiseMatch(sch, block_rv, consumer_rv)) {
      return RulePackedArgs(sch);
    }
    DoMultiLevelTiling(sch, block_rv, tiling_structure);
    LOG(INFO) << "We can do multi-level-tiling with fusion!";
    // TODO(@junrushao1994): add fusion
    return RulePackedArgs(/*proceed=*/{}, /*ignored=*/{sch});
  }

  static SearchRule MakeRule(String tiling_structure) {
    auto invoke = [tiling_structure](Schedule sch, BlockRV block) -> RulePackedArgs {
      MultiLevelTilingWithFusion rule(tiling_structure);
      return rule(sch, block);
    };
    return SearchRule("multi_level_tiling_with_fusion", invoke);
  }
};

class AddCacheWrite {
 public:
  AddCacheWrite() = default;

  RulePackedArgs operator()(Schedule sch, BlockRV block_rv) {
    if (!NeedsMultiLevelTiling(sch, block_rv)) {
      return RulePackedArgs(sch);
    }
    // The only consumer will not be fused
    if (Optional<BlockRV> opt_consumer_rv = sch->GetOnlyConsumer(block_rv)) {
      BlockRV consumer_rv = opt_consumer_rv.value();
      if (!HasReduceBlockVar(sch, block_rv) || !HasReduceBlockVar(sch, consumer_rv)) {
        if (IsElementWiseMatch(sch, block_rv, consumer_rv)) {
          return RulePackedArgs(sch);
        }
      }
    }
    // Add a cache write
    sch->CacheWrite(block_rv, "local");
    return RulePackedArgs(/*proceed=*/{}, /*ignored=*/{sch});
  }

  static SearchRule MakeRule() {
    auto invoke = [](Schedule sch, BlockRV block) -> RulePackedArgs {
      AddCacheWrite rule;
      return rule(sch, block);
    };
    return SearchRule("multi_level_tiling", invoke);
  }
};

TVM_REGISTER_GLOBAL("meta_schedule.rule.AlwaysInline").set_body_typed(AlwaysInline::MakeRule);
TVM_REGISTER_GLOBAL("meta_schedule.rule.AddCacheWrite").set_body_typed(AddCacheWrite::MakeRule);
TVM_REGISTER_GLOBAL("meta_schedule.rule.MultiLevelTilingWithFusion")
    .set_body_typed(MultiLevelTilingWithFusion::MakeRule);
TVM_REGISTER_GLOBAL("meta_schedule.rule.MultiLevelTiling")
    .set_body_typed(MultiLevelTiling::MakeRule);

}  // namespace meta_schedule
}  // namespace tvm
