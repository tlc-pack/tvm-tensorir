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
  try {
    return apply_(task, trace, sampler);
  } catch (dmlc::Error& error) {
    return NullOpt;
  }
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
        if (attrs->n_splits <= 1) {
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
        CHECK_EQ(inputs.size(), 1);
        BlockRV block_rv = Downcast<BlockRV>(inputs[0]);
        tir::StmtSRef block_sref = sch->Eval(block_rv);
        // Extract locations that can be computed at
        Array<tir::StmtSRef> loop_srefs = CollectComputeLocation(sch->sch, block_sref);
        std::vector<int> locs{-2, -1};
        {
          int i = 0;
          for (const tir::StmtSRef& loop_sref : loop_srefs) {
            int64_t extent = GetLoopIntExtent(loop_sref).value_or(-1)->value;
            if (extent != 1) {
              locs.push_back(i);
            }
            ++i;
          }
        }
        // Remove `decided`
        std::vector<int>::iterator rm = std::find(locs.begin(), locs.end(), decided);
        CHECK(rm != locs.end());
        locs.erase(rm);
        // Add the candidate
        CHECK(!locs.empty());
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

}  // namespace meta_schedule
}  // namespace tvm
