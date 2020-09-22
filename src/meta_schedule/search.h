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
#ifndef SRC_META_SCHEDULE_SEARCH_H_
#define SRC_META_SCHEDULE_SEARCH_H_

#include "./measure.h"
#include "./schedule.h"

namespace tvm {
namespace meta_schedule {

/********** SearchSpace **********/

class SearchSpaceNode : public runtime::Object {
 public:
  virtual ~SearchSpaceNode() = default;
  virtual Schedule SampleByReplay(const SearchTask& task) = 0;
  virtual Array<Schedule> GetSupport(const SearchTask& task) = 0;

  static constexpr const char* _type_key = "meta_schedule.SearchSpace";
  TVM_DECLARE_BASE_OBJECT_INFO(SearchSpaceNode, Object);
};

class SearchSpace : public runtime::ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SearchSpace, ObjectRef, SearchSpaceNode);
};

/********** SearchStrategy **********/

class SearchStrategyNode : public Object {
 public:
  virtual ~SearchStrategyNode() = default;
  virtual Schedule Search(const SearchTask& task, const SearchSpace& space,
                          const ProgramMeasurer& measurer, int verbose) = 0;

  static constexpr const char* _type_key = "meta_schedule.SearchStrategy";
  TVM_DECLARE_BASE_OBJECT_INFO(SearchStrategyNode, Object);
};

class SearchStrategy : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SearchStrategy, ObjectRef, SearchStrategyNode);
};

/********** RulePackedArgs **********/

class RulePackedArgsNode : public Object {
 public:
  Array<Schedule> proceed;
  Array<Schedule> skipped;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("proceed", &proceed);
    v->Visit("skipped", &skipped);
  }

  static constexpr const char* _type_key = "meta_schedule.RulePackedArgs";
  TVM_DECLARE_FINAL_OBJECT_INFO(RulePackedArgsNode, Object);
};

class RulePackedArgs : public ObjectRef {
 public:
  explicit RulePackedArgs(Schedule schedule);

  explicit RulePackedArgs(Array<Schedule> proceed, Array<Schedule> skipped);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(RulePackedArgs, ObjectRef, RulePackedArgsNode);
};

/********** SearchRule **********/

class SearchRuleNode : public Object {
 public:
  using FApply = runtime::TypedPackedFunc<RulePackedArgs(Schedule, BlockRV)>;

  String name;

  FApply apply_;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("name", &name); }

  RulePackedArgs Apply(Schedule schedule, BlockRV block) const;

  RulePackedArgs Apply(RulePackedArgs schedules, BlockRV block) const;

  static constexpr const char* _type_key = "meta_schedule.SearchRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(SearchRuleNode, Object);
};

class SearchRule : public ObjectRef {
 public:
  explicit SearchRule(String name);

  explicit SearchRule(String name, runtime::PackedFunc apply);

  explicit SearchRule(String name, SearchRuleNode::FApply apply);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SearchRule, ObjectRef, SearchRuleNode);
};

TVM_DLL SearchRule ComposeSequential(String name, Array<SearchRule> rules);

/********** Search **********/

TVM_DLL Schedule AutoTune(SearchTask task, SearchSpace space, SearchStrategy strategy,
                          ProgramBuilder builder, ProgramRunner runner,
                          Array<MeasureCallback> measure_callbacks, int verbose);

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SEARCH_H_
