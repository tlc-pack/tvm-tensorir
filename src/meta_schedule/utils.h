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

#include <dmlc/memory_io.h>
#include <tvm/driver/driver_api.h>
#include <tvm/meta_schedule/arg_info.h>
#include <tvm/meta_schedule/builder.h>
#include <tvm/meta_schedule/cost_model.h>
#include <tvm/meta_schedule/database.h>
#include <tvm/meta_schedule/feature_extractor.h>
#include <tvm/meta_schedule/measure_callback.h>
#include <tvm/meta_schedule/mutator.h>
#include <tvm/meta_schedule/postproc.h>
#include <tvm/meta_schedule/runner.h>
#include <tvm/meta_schedule/schedule_rule.h>
#include <tvm/meta_schedule/search_strategy.h>
#include <tvm/meta_schedule/space_generator.h>
#include <tvm/meta_schedule/task_scheduler.h>
#include <tvm/meta_schedule/tune_context.h>
#include <tvm/support/parallel_for.h>

#include <random>
#include <string>
#include <vector>

#include "../printer/text_printer.h"
#include "../support/array.h"
#include "../support/base64.h"
#include "../support/utils.h"
#include "../tir/schedule/utils.h"

namespace tvm {
namespace meta_schedule {

/*! \brief The type of the random state */
using TRandState = support::LinearCongruentialEngine::TRandState;

/*!
 * \brief Read lines from a json file.
 * \param path The path to the json file.
 * \param allow_missing Whether to create new file when the given path is not found.
 * \return An array containing lines read from the json file.
 */
inline Array<String> JSONFileReadLines(const String& path, bool allow_missing) {
  std::ifstream is(path);
  if (is.good()) {
    Array<String> results;
    for (std::string str; std::getline(is, str);) {
      results.push_back(str);
    }
    return results;
  }
  CHECK(allow_missing) << "ValueError: File doesn't exist: " << path;
  std::ofstream os(path);
  CHECK(os.good()) << "ValueError: Cannot create new file: " << path;
  return {};
}

/*!
 * \brief Append a line to a json file.
 * \param path The path to the json file.
 * \param line The line to append.
 */
inline void JSONFileAppendLine(const String& path, const std::string& line) {
  std::ofstream os(path, std::ofstream::app);
  CHECK(os.good()) << "ValueError: Cannot open the file to write: " << path;
  os << line << std::endl;
}

/*!
 * \brief Get the base64 encoded result of a string.
 * \param str The string to encode.
 * \return The base64 encoded string.
 */
inline std::string Base64Encode(std::string str) {
  std::string result;
  dmlc::MemoryStringStream m_stream(&result);
  support::Base64OutStream b64stream(&m_stream);
  static_cast<dmlc::Stream*>(&b64stream)->Write(str);
  b64stream.Finish();
  return result;
}

/*!
 * \brief Get the base64 decoded result of a string.
 * \param str The string to decode.
 * \return The base64 decoded string.
 */
inline std::string Base64Decode(std::string str) {
  std::string result;
  dmlc::MemoryStringStream m_stream(&str);
  support::Base64InStream b64stream(&m_stream);
  b64stream.InitPosition();
  static_cast<dmlc::Stream*>(&b64stream)->Read(&result);
  return result;
}

/*!
 * \brief Parse lines of json string into a json object.
 * \param lines The lines of json string.
 * \return Array of json objects parsed.
 * \note The function calls the python-side json parser in runtime registry.
 */
inline Array<ObjectRef> JSONStr2Obj(const Array<String>& lines) {
  static const runtime::PackedFunc* f_to_obj =
      runtime::Registry::Get("meta_schedule.batch_json_str2obj");
  ICHECK(f_to_obj) << "IndexError: Cannot find the packed function "
                      "`meta_schedule.batch_json_str2obj` in the global registry";
  return (*f_to_obj)(lines);
}

/*!
 * \brief Serialize a json object into a json string.
 * \param json_obj The json object to serialize.
 * \return A string containing the serialized json object.
 * \note The function calls the python-side json obj serializer in runtime registry.
 */
inline String JSONObj2Str(const ObjectRef& json_obj) {
  static const runtime::PackedFunc* f_to_str = runtime::Registry::Get("meta_schedule.json_obj2str");
  ICHECK(f_to_str) << "IndexError: Cannot find the packed function "
                      "`meta_schedule.json_obj2str` in the global registry";
  return (*f_to_str)(json_obj);
}

/*!
 * \brief Converts a structural hash code to string
 * \param hash_code The hash code
 * \return The string representation of the hash code
 */
inline String SHash2Str(Workload::THashCode hash_code) { return std::to_string(hash_code); }

/*!
 * \brief Find the entry function of the given IRModule, i.e, functions marked by
 * `tir::attr::kIsEntryFunc`, whose name is `main` or being the only PrimeFunc.
 * \param mod The IRModule to find the entry function.
 * \return The entry function.
 */
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
 *  The given random state is also mutated.
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

/*!
 * \brief Get deep copy of an IRModule.
 * \param mod The IRModule to make a deep copy.
 * \return The deep copy of the IRModule.
 */
inline IRModule DeepCopyIRModule(IRModule mod) {
  return Downcast<IRModule>(LoadJSON(SaveJSON(mod)));
}

/*!
 * \brief Get the BlockRV from a block StmtSRef
 * \param sch The schedule
 * \param block_sref The block StmtSRef
 * \param global_var_name The global variable name
 * \return The BlockRV
 */
inline tir::BlockRV GetRVFromSRef(const tir::Schedule& sch, const tir::StmtSRef& block_sref,
                                  const String& global_var_name) {
  const tir::BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  return sch->GetBlock(block->name_hint, global_var_name);
}

/*!
 * \brief Get the number of cores in CPU
 * \param target The target
 * \return The number of cores.
 */
inline int GetTargetNumCores(const Target& target) {
  int num_cores = target->GetAttr<Integer>("num-cores").value_or(-1);
  if (num_cores == -1) {
    static const auto* f_cpu_count = runtime::Registry::Get("meta_schedule.cpu_count");
    ICHECK(f_cpu_count)
        << "ValueError: Cannot find the packed function \"meta_schedule._cpu_count\"";
    num_cores = (*f_cpu_count)(false);
    LOG(FATAL) << "Target does not have attribute \"num_cores\", pyhsical core number must be "
                  "defined! Example: Local target ....";
  }
  return num_cores;
}

/*!
 * \brief Apply the trace and postprocessors to an IRModule
 * \param mod The IRModule to be applied
 * \param trace The trace to apply to the IRModule
 * \param rand_state The random seed
 * \param postprocs The postprocessors to apply to the IRModule
 * \return The schedule created, or NullOpt if any postprocessor fails
 */
inline Optional<tir::Schedule> ApplyTrace(const IRModule& mod, const tir::Trace& trace,
                                          TRandState* rand_state,
                                          const Array<Postproc>& postprocs) {
  tir::Schedule sch =
      tir::Schedule::Traced(mod,
                            /*rand_state=*/ForkSeed(rand_state),
                            /*debug_mode=*/0,
                            /*error_render_level=*/tir::ScheduleErrorRenderLevel::kNone);
  trace->ApplyToSchedule(sch, /*remove_postproc=*/true);
  sch->EnterPostproc();
  for (const Postproc& proc : postprocs) {
    if (!proc->Apply(sch)) {
      return NullOpt;
    }
  }
  return sch;
}

/*!
 * \brief Get the string representation for the given schedule's IRModule.
 * \param sch The given schedule.
 * \return The string representation created.
 */
inline String Repr(const tir::Schedule& sch) { return tir::AsTVMScript(sch->mod()); }

/*!
 * \brief Create a sampling function that does multinomial sampling.
 * \param rand_state The random state.
 * \param weights The weights for multinomial sampling.
 * \return The multinomial sampling function.
 */
inline std::function<int()> MakeMultinomial(
    support::LinearCongruentialEngine::TRandState& rand_state, const std::vector<double>& weights) {
  std::vector<double> sums;
  sums.reserve(weights.size());
  double sum = 0.0;
  for (double w : weights) {
    sums.push_back(sum += w);
  }
  std::uniform_real_distribution<double> dist(0.0, sum);
  auto sampler = [rand_state, dist = std::move(dist), sums = std::move(sums)]() mutable -> int {
    support::LinearCongruentialEngine rand_(&rand_state);
    double p = dist(rand_);
    int idx = std::lower_bound(sums.begin(), sums.end(), p) - sums.begin();
    int n = sums.size();
    CHECK_LE(0, idx);
    CHECK_LE(idx, n);
    return (idx == n) ? (n - 1) : idx;
  };
  return sampler;
}

/*!
 * \brief Create a sampler function that picks mutators according to the mass function
 * \param rand_state The random state for sampling
 * \return The sampler created
 */
inline std::function<Optional<Mutator>()> MakeMutatorSampler(
    double p_mutate, const Map<Mutator, FloatImm>& mutator_probs,
    support::LinearCongruentialEngine::TRandState& rand_state) {
  CHECK(0.0 <= p_mutate && p_mutate <= 1.0)  //
      << "ValueError: Probability should be within [0, 1], "
      << "but get `p_mutate = " << p_mutate << '\'';
  std::vector<Optional<Mutator>> mutators;
  std::vector<double> masses;
  mutators.push_back(NullOpt);
  masses.push_back(1.0 - p_mutate);
  double total_mass_mutator = 0.0;
  for (const auto& kv : mutator_probs) {
    const Mutator& mutator = kv.first;
    double mass = kv.second->value;
    CHECK_GE(mass, 0.0) << "ValueError: Probability of mutator '" << mutator
                        << "' is ill-formed: " << mass;
    total_mass_mutator += mass;
    mutators.push_back(kv.first);
    masses.push_back(mass * p_mutate);
  }
  // Normalize the sum to 1.0
  if (total_mass_mutator == 0.0) {
    masses[0] = 1.0;
    for (int i = 1, n = masses.size(); i < n; ++i) {
      masses[i] = 0.0;
    }
  } else if (total_mass_mutator != 1.0) {
    for (int i = 1, n = masses.size(); i < n; ++i) {
      masses[i] /= total_mass_mutator;
    }
  }
  auto idx_sampler = MakeMultinomial(rand_state, masses);
  return [idx_sampler = std::move(idx_sampler),
          mutators = std::move(mutators)]() -> Optional<Mutator> {
    int i = idx_sampler();
    return mutators[i];
  };
}

/*! \brief The postprocessed built of a trace */
struct CachedTrace {
  /*! \brief The trace */
  const tir::TraceNode* trace;
  /*! \brief The schedule the trace creates */
  tir::Schedule sch;
  /*! \brief The string representation of the schedule */
  String repr;
  // Todo: Challenges in deduplication: remove unit loop / simplify pass
  /*! \brief The normalized score, the higher the better */
  double score;

  static bool Compare(const CachedTrace& lhs, const CachedTrace& rhs) {
    return lhs.score > rhs.score;
  }
};

/*!
 * \brief Predict the normalized score of each candidate.
 * \param candidates The candidates for prediction
 * \param task The search task
 * \param space The search space
 * \return The normalized score in the prediction
 */
inline std::vector<double> PredictNormalizedScore(const std::vector<CachedTrace>& cached_traces,
                                                  const TuneContext& tune_context,
                                                  const CostModel& cost_model,
                                                  Array<ArgInfo> args_info) {
  Array<MeasureCandidate> measure_inputs;
  measure_inputs.reserve(cached_traces.size());
  for (const CachedTrace& cached_trace : cached_traces) {
    measure_inputs.push_back(MeasureCandidate(cached_trace.sch, args_info));
  }

  std::vector<double> scores = cost_model->Predict(tune_context, measure_inputs);
  // Normalize the score
  // TODO(@junrushao1994): use softmax + temperature to replace simple normalization to [0.0, +oo)
  for (double& score : scores) {
    score = std::max(0.0, score);
  }
  return scores;
}

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_UTILS_H_
