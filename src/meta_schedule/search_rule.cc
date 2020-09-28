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

TVM_REGISTER_GLOBAL("meta_schedule.rule.MultiLevelTiling")
    .set_body_typed(MultiLevelTiling::MakeRule);
TVM_REGISTER_GLOBAL("meta_schedule.rule.MultiLevelTilingWithFusion")
    .set_body_typed(MultiLevelTilingWithFusion::MakeRule);

}  // namespace meta_schedule
}  // namespace tvm
