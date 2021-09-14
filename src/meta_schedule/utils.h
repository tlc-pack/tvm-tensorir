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
#ifndef TVM_META_SCHEDULE_UTILS_H_
#define TVM_META_SCHEDULE_UTILS_H_
#include <tvm/ir/module.h>
#include <tvm/support/parallel_for.h>
#include <tvm/support/random_engine.h>
#include <tvm/tir/schedule/trace.h>

#include <vector>

#include "../printer/text_printer.h"
#include "../support/array.h"
#include "./arg_info.h"
#include "./builder.h"
#include "./runner.h"
#include "./search_strategy.h"
#include "./space_generator.h"
#include "./tune_context.h"

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Fork a random state into another, i.e. PRNG splitting.
 * The given random state is also mutated.
 * \param rand_state The random state to be forked
 * \return The forked random state
 */
inline support::LinearCongruentialEngine::TRandState ForkSeed(
    support::LinearCongruentialEngine::TRandState* rand_state) {
  return support::LinearCongruentialEngine(rand_state).ForkSeed();
}

/*!
 * \brief Fork a random state into another ones, i.e. PRNG splitting.
 * The given random state is also mutated.
 * \param rand_state The random state to be forked
 * \param n The number of forks
 * \return The forked random states
 */
inline std::vector<support::LinearCongruentialEngine::TRandState> ForkSeed(
    support::LinearCongruentialEngine::TRandState* rand_state, int n) {
  std::vector<support::LinearCongruentialEngine::TRandState> results;
  results.reserve(n);
  for (int i = 0; i < n; ++i) {
    results.push_back(support::LinearCongruentialEngine(rand_state).ForkSeed());
  }
  return results;
}

inline tir::PrimFunc FindEntryFunc(const IRModule& mod) {
  // Priority 1: PrimFunc marked as `tir::attr::kIsEntryFunc`
  int num_prim_func = 0;
  const tir::PrimFuncNode* main_func = nullptr;
  const tir::PrimFuncNode* last_func = nullptr;
  for (const auto& kv : mod->functions) {
    GlobalVar gv = kv.first;
    BaseFunc base_func = kv.second;
    if (const auto* func = base_func.as<tir::PrimFuncNode>()) {
      last_func = func;
      if (func->HasNonzeroAttr(tir::attr::kIsEntryFunc)) {
        return GetRef<tir::PrimFunc>(func);
      }
      if (gv->name_hint == "main") {
        main_func = func;
      }
      ++num_prim_func;
    }
  }
  // Priority 2: PrimFunc whose name is `main`
  if (main_func != nullptr) {
    return GetRef<tir::PrimFunc>(main_func);
  }
  // Priority 3: The only PrimFunc in the IRModule
  if (num_prim_func == 0) {
    LOG(FATAL) << "ValueError: Cannot find any PrimFunc in the given IRModule: "
               << tir::AsTVMScript(mod);
  }
  if (num_prim_func > 1) {
    LOG(FATAL) << "ValueError: Multiple PrimFuncs exist in the IRModule, but none of them are "
                  "annotated with `kIsEntryFunc`, i.e. `tir.is_entry_func`"
               << tir::AsTVMScript(mod);
  }
  return GetRef<tir::PrimFunc>(last_func);
}

inline int SampleInt(support::LinearCongruentialEngine::TRandState* rand_state, int min_inclusive,
                     int max_exclusive) {
  // Move to better places
  if (min_inclusive + 1 == max_exclusive) {
    return min_inclusive;
  }
  support::LinearCongruentialEngine rand_(rand_state);
  std::uniform_int_distribution<int> dist(min_inclusive, max_exclusive - 1);
  return dist(rand_);
}

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_UTILS_H_
