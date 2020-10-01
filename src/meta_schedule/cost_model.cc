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

#include "./cost_model.h"  // NOLINT(build/include)

namespace tvm {
namespace meta_schedule {

/**************** Constructors ****************/

RandomModel::RandomModel() { data_ = make_object<RandomModelNode>(); }

/**************** RandomModel ****************/

void RandomModelNode::Update(const Array<MeasureInput>& inputs,
                             const Array<MeasureResult>& results) {}

std::vector<double> RandomModelNode::Predict(const SearchTask& task,
                                             const Array<Schedule>& states) {
  return sampler_.SampleUniform(states.size(), 0.0, 1.0);
}

/**************** FFI ****************/

struct Internal {
  static RandomModel NewRandomModel() { return RandomModel(); }
  static void CostModelUpdate(CostModel model, Array<MeasureInput> inputs,
                              Array<MeasureResult> results) {
    model->Update(inputs, results);
  }
  static Array<FloatImm> CostModelPredict(CostModel model, SearchTask task,
                                          Array<Schedule> states) {
    std::vector<double> scores = model->Predict(task, states);
    Array<FloatImm> ret;
    for (double score : scores) {
      ret.push_back(FloatImm(DataType::Float(64), score));
    }
    return ret;
  }
};

TVM_REGISTER_OBJECT_TYPE(CostModelNode);
TVM_REGISTER_NODE_TYPE(RandomModelNode);

TVM_REGISTER_GLOBAL("meta_schedule.RandomModel").set_body_typed(Internal::NewRandomModel);
TVM_REGISTER_GLOBAL("meta_schedule.CostModelUpdate").set_body_typed(Internal::CostModelUpdate);
TVM_REGISTER_GLOBAL("meta_schedule.CostModelPredict").set_body_typed(Internal::CostModelPredict);

}  // namespace meta_schedule
}  // namespace tvm
