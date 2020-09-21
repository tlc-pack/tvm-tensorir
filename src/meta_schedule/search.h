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

/********** Search **********/

TVM_DLL Schedule AutoTune(SearchTask task, SearchSpace space, SearchStrategy strategy,
                          ProgramBuilder builder, ProgramRunner runner,
                          Array<MeasureCallback> measure_callbacks, int verbose);

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SEARCH_H_
