//
// Created by jinho on 2021/5/10.
//
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

#include <tvm/arith/analyzer.h>
#include <tvm/auto_scheduler/feature.h>
#include <tvm/support/parallel_for.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <cmath>

#include "../schedule.h"
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

struct DoubleNDArrayPusher {
  explicit DoubleNDArrayPusher(const std::vector<int64_t>& shape)
      : array(runtime::NDArray::Empty(/*shape=*/shape, /*dtype=*/DLDataType{kDLFloat, 64, 1},
                                      /*ctx=*/DLContext{kDLCPU, 0})),
        back(static_cast<double*>(array->data)) {}

  template <class TIter>
  void Push(TIter begin, TIter end) {
    while (begin != end) {
      *back = *begin;
      ++back;
      ++begin;
    }
  }

  void PushRepeat(int n, double value) {
    while (n-- > 0) {
      *back = value;
      ++back;
    }
  }

  runtime::NDArray Done() {
    int64_t* shape = array->shape;
    int64_t array_size = 1;
    for (int i = 0, ndim = array->ndim; i < ndim; ++i) {
      array_size *= shape[i];
    }
    int64_t written_size = back - static_cast<double*>(array->data);
    ICHECK_EQ(array_size, written_size);
    return std::move(array);
  }

  runtime::NDArray array;
  double* back;
};

runtime::NDArray GetPerStoreFeaturesWorkerFunc(const Schedule& sch, int max_n_bufs) {
  std::vector<float> feature;

  const std::string& name = "main";
  GlobalVar global_var(name);

  auto pass_ctx = tvm::transform::PassContext::Current();
  auto f = GetOnlyFunc(sch->mod());
  f = WithAttr(std::move(f), "global_symbol", runtime::String(name));

  bool noalias = pass_ctx->GetConfig<Bool>("tir.noalias", Bool(true)).value();
  if (noalias) {
    f = WithAttr(std::move(f), "tir.noalias", Bool(true));
  }
  auto mod = IRModule(Map<GlobalVar, BaseFunc>({{global_var, f}}));

  auto pass_list = Array<tvm::transform::Pass>();
  pass_list.push_back(tir::transform::PartialBufferFlatten());
  pass_list.push_back(tir::transform::Simplify());
  const auto& optimize = tir::transform::Sequential(pass_list);
  mod = optimize(std::move(mod));
  const auto& it = mod->functions.find(global_var);
  ICHECK(it != mod->functions.end());
  const auto& prim_func = (*it).second.as<tir::PrimFuncNode>();
  auto_scheduler::GetPerStoreFeature(prim_func->body, Integer(64), max_n_bufs, &feature);
  DoubleNDArrayPusher pusher({int64_t(feature[0]), 164});
  pusher.Push(feature.begin() + 1, feature.end());
  return pusher.Done();
}
Array<runtime::NDArray> PerStoreFeatureBatched(const Array<Schedule>& schs,
                                               int max_num_buffer_access_features) {
  int n = schs.size();
  std::vector<runtime::NDArray> result;
  result.resize(n);
  auto worker = [&result, &schs, &max_num_buffer_access_features](int thread_id, int i) {
    result[i] = GetPerStoreFeaturesWorkerFunc(schs[i], max_num_buffer_access_features);
  };
  support::parallel_persist_for(0, n, worker);
  return result;
}

TVM_REGISTER_GLOBAL("meta_schedule.PerStoreFeatureBatched").set_body_typed(PerStoreFeatureBatched);

}  // namespace meta_schedule
}  // namespace tvm
