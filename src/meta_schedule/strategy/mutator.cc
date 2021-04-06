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
#include "./mutator.h"  // NOLINT(build/include)

#include "../analysis.h"
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

/********** Constructor **********/

Mutator::Mutator(String name, FApply apply) {
  ObjectPtr<MutatorNode> n = make_object<MutatorNode>();
  n->name = std::move(name);
  n->apply_ = std::move(apply);
  data_ = std::move(n);
}

/********** Mutator **********/

Optional<Trace> MutatorNode::Apply(const SearchTask& task, const Trace& trace, Sampler* sampler) {
  return apply_(task, trace, sampler);
}

/********** MutateTileSize **********/

class MutatorTileSize {
 public:
  MutatorTileSize() = default;

  static std::vector<int> CastDecision(const ObjectRef& obj) {
    const auto* arr = obj.as<runtime::ArrayNode>();
    CHECK(arr) << "TypeError: Expects ArrayNode, but gets: " << obj->GetTypeKey();
    return AsVector<ObjectRef, int>(GetRef<Array<ObjectRef>>(arr));
  }

  /*!
   * \brief Find instruction `SamplePerfectTile` whose extent > 1 and n_splits > 1
   * \param trace The trace from which to find the instructions
   * \return All the candidate instructions
   */
  std::vector<Instruction> FindCandidates(const Trace& trace) {
    std::vector<Instruction> candidates;
    candidates.reserve(trace->decisions.size());
    for (const auto& kv : trace->decisions) {
      const Instruction& inst = kv.first;
      if (const auto* attrs = inst->inst_attrs.as<SamplePerfectTileAttrs>()) {
        if (attrs->n <= 1) {
          continue;
        }
        std::vector<int> tiles = CastDecision(kv.second);
        int64_t prod = 1;
        for (int item : tiles) {
          prod *= item;
        }
        if (prod > 1) {
          candidates.push_back(inst);
        }
      }
    }
    return candidates;
  }

  Optional<Trace> Apply(const SearchTask& task, const Trace& trace, Sampler* sampler) {
    // Find instruction `SamplePerfectTile` whose extent > 1 and n_splits > 1
    std::vector<Instruction> candidates = FindCandidates(trace);
    if (candidates.empty()) {
      return NullOpt;
    }
    const Instruction& inst = candidates[sampler->SampleInt(0, candidates.size())];
    std::vector<int> tiles = CastDecision(trace->decisions.at(inst));
    int n_splits = tiles.size();
    // Choose two loops
    int x = sampler->SampleInt(0, n_splits);
    int y;
    if (tiles[x] == 1) {
      // need to guarantee that tiles[x] * tiles[y] > 1
      std::vector<int> idx;
      idx.reserve(n_splits);
      for (int i = 0; i < n_splits; ++i) {
        if (tiles[i] > 1) {
          idx.push_back(i);
        }
      }
      y = idx[sampler->SampleInt(0, idx.size())];
    } else {
      // sample without replacement
      y = sampler->SampleInt(0, n_splits - 1);
      if (y >= x) {
        ++y;
      }
    }
    // make sure x < y
    CHECK_NE(x, y);
    if (x > y) {
      std::swap(x, y);
    }
    // Case 1. None of x and y are innermost loop
    int len_x, len_y;
    if (y != n_splits - 1) {
      do {
        std::vector<int> result = sampler->SamplePerfectTile(2, tiles[x] * tiles[y]);
        len_x = result[0];
        len_y = result[1];
      } while (len_y == tiles[y]);
    } else {
      // Case 2. y is the innermost loop
      std::vector<int> len_y_space;
      int limit = inst->inst_attrs.as<SamplePerfectTileAttrs>()->max_innermost_factor;
      int prod = tiles[x] * tiles[y];
      for (len_y = 1; len_y <= limit; ++len_y) {
        if (len_y != tiles[y] && prod % len_y == 0) {
          len_y_space.push_back(len_y);
        }
      }
      if (len_y_space.empty()) {
        return NullOpt;
      }
      len_y = len_y_space[sampler->SampleInt(0, len_y_space.size())];
      len_x = prod / len_y;
    }
    tiles[x] = len_x;
    tiles[y] = len_y;
    return trace->WithDecision(inst, AsArray<int, ObjectRef>(tiles), /*remove_postproc=*/true);
  }
};

Mutator MutateTileSize() {
  auto f_apply = [](SearchTask task, Trace trace, void* sampler) -> Optional<Trace> {
    MutatorTileSize mutator;
    return mutator.Apply(task, trace, static_cast<Sampler*>(sampler));
  };
  return Mutator("mutate_tile_size", f_apply);
}

/********** MutateComputeLocation **********/

class MutatorComputeLocation {
 public:
  struct Candidate {
    /*! \brief The SampleComputeLocation instruction */
    Instruction inst;
    /*! \brief The candidate compute locations */
    std::vector<int> locs;

    explicit Candidate(Instruction inst, std::vector<int> locs)
        : inst(std::move(inst)), locs(std::move(locs)) {}
  };

  /*!
   * \brief Find instruction `SampleComputeLocation`
   * \param trace The trace from which to find the instructions
   * \param workload The workload
   * \return All the candidate instructions together with the candidate compute locations
   */
  std::vector<Candidate> FindCandidates(const Trace& trace, const tir::PrimFunc& workload) {
    std::vector<Candidate> candidates;
    Schedule sch(workload);
    auto f_provide_decision = [&trace, &candidates, &sch](
                                  const Instruction& inst,
                                  const Array<Optional<ObjectRef>>& inputs) -> Optional<ObjectRef> {
      Optional<ObjectRef> decision = trace->decisions.Get(inst);
      if (inst->inst_attrs->IsInstance<SampleComputeLocationAttrs>()) {
        // The decision made
        int decided = Downcast<Integer>(decision)->value;
        // Extract the inputs
        ICHECK_EQ(inputs.size(), 1);
        BlockRV block_rv = Downcast<BlockRV>(inputs[0]);
        tir::StmtSRef block_sref = sch->GetSRef(block_rv);
        // Extract locations that can be computed at
        Array<tir::StmtSRef> loop_srefs = CollectComputeLocation(sch->state(), block_sref);
        std::vector<int> locs{-2, -1};
        {
          int i = 0;
          for (const tir::StmtSRef& loop_sref : loop_srefs) {
            int64_t extent = GetLoopIntExtent(loop_sref);
            if (extent != 1 && extent != -1) {
              locs.push_back(i);
            }
            ++i;
          }
        }
        // Remove `decided`
        std::vector<int>::iterator rm = std::find(locs.begin(), locs.end(), decided);
        if (rm != locs.end()) {
          locs.erase(rm);
        }
        // Add the candidate
        ICHECK(!locs.empty());
        candidates.emplace_back(inst, std::move(locs));
      }
      return decision;
    };
    trace->Apply(sch, f_provide_decision);
    return candidates;
  }

  Optional<Trace> Apply(const SearchTask& task, const Trace& trace, Sampler* sampler) {
    std::vector<Candidate> candidates = FindCandidates(trace, task->workload);
    if (candidates.empty()) {
      return NullOpt;
    }
    const Candidate& candidate = candidates[sampler->SampleInt(0, candidates.size())];
    int loc = candidate.locs[sampler->SampleInt(0, candidate.locs.size())];
    return trace->WithDecision(candidate.inst, Integer(loc), /*remove_postproc=*/true);
  }
};

Mutator MutateComputeLocation() {
  auto f_apply = [](SearchTask task, Trace trace, void* sampler) -> Optional<Trace> {
    MutatorComputeLocation mutator;
    return mutator.Apply(task, trace, static_cast<Sampler*>(sampler));
  };
  return Mutator("mutate_compute_location", f_apply);
}

/********** MutateAutoUnroll **********/

class MutatorAutoUnroll {
 public:
  MutatorAutoUnroll() = default;

  struct Candidate {
    /*! \brief The SampleCategorical instruction */
    Instruction inst;
    /*! \brief The weights of the categorical distribution */
    std::vector<double> weights;
    /*! \brief The original decision */
    int ori_decision;

    explicit Candidate(Instruction inst, std::vector<double> weights, int ori_decision)
        : inst(std::move(inst)), weights(std::move(weights)), ori_decision(ori_decision) {}
  };

  /*!
   * \brief Find instruction `SampleCategorical` whose output is used by `auto_unroll`
   * \param trace The trace from which to find the instructions
   * \return All the candidate instructions
   */
  std::vector<Candidate> FindCandidates(const Trace& trace) {
    std::vector<Candidate> candidates;
    for (int i = 0; i < static_cast<int>(trace->insts.size()); ++i) {
      const Instruction& mark_inst = trace->insts[i];
      // Step 1. Find the `MarkBlockAttr` whose attr_key is `auto_unroll`
      //         and whose unroll depth is a `tir::VarNode`.
      if (const auto* mark_attr = mark_inst->inst_attrs.as<MarkBlockAttrs>()) {
        ICHECK_EQ(mark_inst->inputs.size(), 2);
        if (mark_attr->ann_key != tir::attr::auto_unroll_explicit &&
            mark_attr->ann_key != tir::attr::auto_unroll_implicit) {
          continue;
        }
        const auto* sample_output = mark_inst->inputs[1].as<tir::VarNode>();
        if (!sample_output) {
          continue;
        }
        // Step 2. Back to find the corresponding `SampleCategorical` instruction.
        for (int j = i - 1; j >= 0; --j) {
          const Instruction& sample_inst = trace->insts[j];
          if (sample_inst->outputs.size() == 1 && sample_inst->outputs[0].get() == sample_output) {
            const auto* sample_attr = sample_inst->inst_attrs.as<SampleCategoricalAttrs>();
            if (!sample_attr) {
              // The unroll depth is not created by a `SampleCategorical`. So skip.
              break;
            }
            ICHECK_EQ(sample_attr->candidates.size(), sample_attr->probs.size());
            int decision = Downcast<Integer>(trace->decisions.Get(sample_inst))->value;
            // Step 3. Remove the current decision from the sampling candidates.
            std::vector<double> weights = AsVector<FloatImm, double>(sample_attr->probs);
            weights.erase(weights.begin() + decision);
            // Step 4. Add a new candidate if `weights` is not empty.
            if (!weights.empty()) {
              candidates.emplace_back(sample_inst, weights, decision);
            }
            break;
          }
        }
      }
    }
    return candidates;
  }

  Optional<Trace> Apply(const SearchTask& task, const Trace& trace, Sampler* sampler) {
    std::vector<Candidate> candidates = FindCandidates(trace);
    if (candidates.empty()) {
      return NullOpt;
    }
    const Candidate& candidate = candidates[sampler->SampleInt(0, candidates.size())];
    int result = sampler->MakeMultinomial(candidate.weights)();
    if (result >= candidate.ori_decision) {
      result++;
    }
    return trace->WithDecision(candidate.inst, Integer(result), /*remove_postproc=*/true);
  }
};

Mutator MutateAutoUnroll() {
  auto f_apply = [](SearchTask task, Trace trace, void* sampler) -> Optional<Trace> {
    MutatorAutoUnroll mutator;
    return mutator.Apply(task, trace, static_cast<Sampler*>(sampler));
  };
  return Mutator("mutate_unroll_depth", f_apply);
}

/********** MutateParallel **********/

class MutatorParallel {
 public:
  int max_jobs_per_core;
  mutable std::atomic<int> warned_num_cores_missing;

  explicit MutatorParallel(int max_jobs_per_core)
      : max_jobs_per_core(max_jobs_per_core), warned_num_cores_missing(0) {}

  MutatorParallel(const MutatorParallel& other)
      : max_jobs_per_core(other.max_jobs_per_core),
        warned_num_cores_missing(other.warned_num_cores_missing.load()) {}

  struct Candidate {
    /*! \brief The MarkBlock instruction */
    Instruction inst;
    /*! \brief The extent candidates */
    std::vector<int> extent_candidates;

    explicit Candidate(Instruction inst, std::vector<int> extent_candidates)
        : inst(std::move(inst)), extent_candidates(std::move(extent_candidates)) {}
  };

  /*!
   * \brief Find instruction `MarkBlock` with annotation key `auto_parallel`
   * \param trace The trace from which to find the instructions
   * \return All the candidate instructions
   */
  Candidate FindCandidates(const Trace& trace, const tir::PrimFunc& workload,
                           const int& max_extent) const {
    Schedule sch(workload);
    std::set<int> extent_candidates;
    extent_candidates.insert(1);
    for (size_t i = 0; i < trace->insts.size(); i++) {
      const Instruction& mark_inst = trace->insts[i];
      // Step 1. Find the `MarkBlockAttr` whose ann_key is `auto_parallel_extent`
      //         and whose parallel extent is given by an integer.
      if (const auto* mark_attr = mark_inst->inst_attrs.as<MarkBlockAttrs>()) {
        ICHECK_EQ(mark_inst->inputs.size(), 2);
        if (mark_attr->ann_key != tir::attr::auto_parallel_extent ||
            !mark_inst->inputs[1]->IsInstance<IntImmNode>()) {
          continue;
        }
        tir::StmtSRef root_sref = tir::GetBlocks(sch->state(), "root")[0];
        // Step 2. For all the leaf blocks ,fetch the loops above it.
        // Furthermore, get their loop types.
        for (const auto& block_sref : tir::GetChildBlocks(sch->state(), root_sref)) {
          Array<tir::StmtSRef> loop_srefs = tir::GetAxes(sch->state(), block_sref);
          std::vector<int> loop_types;
          for (const tir::StmtSRef& loop_sref : loop_srefs) {
            loop_types.emplace_back(GetLoopIterType(sch->state(), loop_sref));
          }
          // Step 3. Get the original parallel extent.
          int ori_extent = mark_inst->inputs[1].as<IntImmNode>()->value;
          // Step 4. Find extent candidates.
          int prod_extent = 1;
          for (int i = 0; i < static_cast<int>(loop_srefs.size()) &&
                          loop_types[i] == tir::IterVarType::kDataPar;
               ++i) {
            const tir::StmtSRef& loop_sref = loop_srefs[i];
            if (HasAnyAnn(loop_sref)) {
              break;
            }
            // Check if the loop extent is valid
            int64_t extent = GetLoopIntExtent(loop_sref);
            if (extent == -1) {
              break;
            }
            // Then we can fuse it in. Moreover, if extent is not 1 and extent does not
            // equal the original extent, then it is a valid candidate.
            if (extent != 1) {
              prod_extent *= extent;
              if (prod_extent > max_extent) {
                break;
              }
              if (prod_extent != ori_extent) {
                extent_candidates.insert(prod_extent);
              }
            }
            // Check if we need to break.
            if (!HasSingleChild(loop_sref)) {
              break;
            }
          }
        }
        return Candidate(mark_inst,
                         std::vector<int>(extent_candidates.begin(), extent_candidates.end()));
      }
    }
    return Candidate(Instruction({}, {}, InstAttrs()), {});
  }

  Optional<Trace> Apply(const SearchTask& task, const Trace& trace, Sampler* sampler) const {
    int max_extent =
        GetTargetNumCores(task->target, &warned_num_cores_missing) * max_jobs_per_core - 1;
    Candidate candidate = FindCandidates(trace, task->workload, max_extent);
    if (candidate.extent_candidates.empty()) {
      return NullOpt;
    }
    const BlockRV& block = Downcast<BlockRV>(candidate.inst->inputs[0]);
    const std::vector<int>& extent_candidates = candidate.extent_candidates;
    const int& parallel_size = extent_candidates[sampler->SampleInt(0, extent_candidates.size())];

    std::vector<Instruction> new_insts;
    for (int i = 0; i < static_cast<int>(trace->insts.size()) &&
                    !trace->insts[i]->inst_attrs->IsInstance<EnterPostProcAttrs>();
         ++i) {
      new_insts.emplace_back(trace->insts[i]);
    }
    for (Instruction& new_inst : new_insts) {
      if (new_inst.same_as(candidate.inst)) {
        new_inst = MarkBlockAttrs::Make(block, tir::attr::auto_parallel_extent, parallel_size);
        break;
      }
    }
    return Trace(new_insts, trace->decisions);
  }
};

Mutator MutateParallel(const int& max_jobs_per_core) {
  MutatorParallel mutator(max_jobs_per_core);
  auto f_apply = [mutator](SearchTask task, Trace trace, void* sampler) -> Optional<Trace> {
    return mutator.Apply(task, trace, static_cast<Sampler*>(sampler));
  };
  return Mutator("mutate_parallel", f_apply);
}

/********** FFI **********/

struct Internal {
  /*!
   * \brief FFI function for MutatorNode::Apply
   * \sa MutatorNode::Apply
   */
  static Optional<Trace> Apply(Mutator mutator, SearchTask task, Trace trace,
                               Optional<Integer> seed) {
    Sampler seeded;
    if (seed.defined()) {
      seeded.Seed(seed.value());
    }
    return mutator->Apply(task, trace, &seeded);
  }
};

TVM_REGISTER_NODE_TYPE(MutatorNode);
TVM_REGISTER_GLOBAL("meta_schedule.mutator.Apply").set_body_typed(Internal::Apply);
TVM_REGISTER_GLOBAL("meta_schedule.mutator.MutateTileSize").set_body_typed(MutateTileSize);
TVM_REGISTER_GLOBAL("meta_schedule.mutator.MutateComputeLocation")
    .set_body_typed(MutateComputeLocation);
TVM_REGISTER_GLOBAL("meta_schedule.mutator.MutateAutoUnroll").set_body_typed(MutateAutoUnroll);
TVM_REGISTER_GLOBAL("meta_schedule.mutator.MutateParallel").set_body_typed(MutateParallel);

}  // namespace meta_schedule
}  // namespace tvm
