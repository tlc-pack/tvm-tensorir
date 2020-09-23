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
#include "../measure.h"
#include "../search.h"

namespace tvm {
namespace meta_schedule {

/********** Definition for Replay **********/

class ReplayNode : public SearchStrategyNode {
 public:
  int batch_size;
  int num_iterations;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("batch_size", &batch_size);
    v->Visit("num_iterations", &num_iterations);
  }

  Optional<Schedule> Search(const SearchTask& task, const SearchSpace& space,
                            const ProgramMeasurer& measurer, int verbose) override;

  static constexpr const char* _type_key = "meta_schedule.Replay";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReplayNode, SearchStrategyNode);
};

class Replay : public SearchStrategy {
 public:
  explicit Replay(int batch_size, int num_iterations);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Replay, SearchStrategy, ReplayNode);
};

/********** Constructor **********/

Replay::Replay(int batch_size, int num_iterations) {
  ObjectPtr<ReplayNode> n = make_object<ReplayNode>();
  n->batch_size = batch_size;
  n->num_iterations = num_iterations;
  data_ = std::move(n);
}

/********** Search **********/

Optional<Schedule> ReplayNode::Search(const SearchTask& task, const SearchSpace& space,
                                      const ProgramMeasurer& measurer, int verbose) {
  measurer->Reset();
  for (int iter_id = 0; iter_id < num_iterations;) {
    Array<MeasureInput> measure_inputs;
    measure_inputs.reserve(batch_size);
    for (int batch_id = 0; batch_id < batch_size && iter_id < num_iterations;
         ++batch_id, ++iter_id) {
      measure_inputs.push_back(MeasureInput(task, space->SampleByReplay(task)));
    }
    measurer->BatchMeasure(measure_inputs, this->batch_size, verbose);
  }
  return measurer->best_sch;
}

/********** FFI **********/

struct Internal {
  static Replay New(int batch_size, int num_iterations) {
    return Replay(batch_size, num_iterations);
  }
};

TVM_REGISTER_NODE_TYPE(ReplayNode);
TVM_REGISTER_GLOBAL("meta_schedule.Replay").set_body_typed(Internal::New);

}  // namespace meta_schedule
}  // namespace tvm
