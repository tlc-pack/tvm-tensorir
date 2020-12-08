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

Optional<Schedule> MutatorNode::Apply(const SearchTask& task, const Schedule& sch,
                                      Sampler* sampler) {
  return apply_(task, sch, sampler);
}

/********** MutateTileSize **********/

class MutatorTileSize {
 public:
  MutatorTileSize() = default;

  Optional<Schedule> Apply(const SearchTask& task, const Schedule& sch, Sampler* sampler) {
    // Find instruction `SamplePerfectTile` whose extent > 1 and n_splits > 1
    std::vector<Instruction> candidates;
    candidates.reserve(sch->decisions.size());
    for (const auto& kv : sch->decisions) {
      const Instruction& inst = kv.first;
      if (const auto* attrs = inst->inst_attrs.as<SamplePerfectTileAttrs>()) {
        if (attrs->n_splits <= 1) {
          continue;
        }
        std::vector<int> tiles = AsVector<ObjectRef, int>()(kv.second);
        int64_t prod = 1;
        for (int item : tiles) {
          prod *= item;
        }
        if (prod > 1) {
          candidates.push_back(inst);
        }
      }
    }
    if (candidates.empty()) {
      return NullOpt;
    }
    const Instruction& inst = candidates[sampler->SampleInt(0, candidates.size())];
    std::vector<int> tiles = AsVector<ObjectRef, int>()(sch->decisions.at(inst));
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
    Schedule new_sch = sch->Copy(sampler->ForkSeed());
    tiles[x] = len_x;
    tiles[y] = len_y;
    new_sch->MutateDecision(inst, AsArray<int, ObjectRef>()(tiles));
    // TODO(@junrushao1994): recover
    // std::unordered_set<Instruction, ObjectPtrHash, ObjectPtrEqual> fusible_samplings;
    // for (const auto& kv : new_sch->decisions) {
    //   if (kv.first->inst_attrs->IsInstance<SampleFusibleLoopsAttrs>()) {
    //     fusible_samplings.insert(kv.first);
    //   }
    // }
    // for (const Instruction& fusible_sample : fusible_samplings) {
    //   new_sch->MutateDecision(fusible_sample, NullOpt);
    // }
    new_sch->ReplayDecision();
    return new_sch;
  }
};

Mutator MutateTileSize() {
  auto f_apply = [](SearchTask task, Schedule sch, void* sampler) -> Optional<Schedule> {
    MutatorTileSize mutator;
    return mutator.Apply(task, sch, static_cast<Sampler*>(sampler));
  };
  return Mutator("mutate_tile_size", f_apply);
}

/********** FFI **********/

struct Internal {
  /*!
   * \brief FFI function for MutatorNode::Apply
   * \sa MutatorNode::Apply
   */
  static Optional<Schedule> Apply(Mutator mutator, SearchTask task, Schedule sch,
                                  Optional<Integer> seed) {
    Sampler seeded;
    if (seed.defined()) {
      seeded.Seed(seed.value());
    }
    return mutator->Apply(task, sch, &seeded);
  }
};

TVM_REGISTER_NODE_TYPE(MutatorNode);
TVM_REGISTER_GLOBAL("meta_schedule.mutator.Apply").set_body_typed(Internal::Apply);
TVM_REGISTER_GLOBAL("meta_schedule.mutator.MutateTileSize").set_body_typed(MutateTileSize);

}  // namespace meta_schedule
}  // namespace tvm
