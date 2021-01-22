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

/*!
 * \file src/target/target_tag.cc
 * \brief Target tag registry
 */
#include <tvm/ir/expr.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/tag.h>
#include <tvm/target/target.h>

#include "../node/attr_registry.h"

namespace tvm {

TVM_REGISTER_NODE_TYPE(TargetTagNode);

TVM_REGISTER_GLOBAL("target.TargetTagListTags").set_body_typed(TargetTag::ListTags);
TVM_REGISTER_GLOBAL("target.TargetTagAddTag").set_body_typed(TargetTag::AddTag);

/**********  Registry-related code  **********/

using TargetTagRegistry = AttrRegistry<TargetTagRegEntry, TargetTag>;

TargetTagRegEntry& TargetTagRegEntry::RegisterOrGet(const String& target_tag_name) {
  return TargetTagRegistry::Global()->RegisterOrGet(target_tag_name);
}

Optional<Target> TargetTag::Get(const String& target_tag_name) {
  const TargetTagRegEntry* reg = TargetTagRegistry::Global()->Get(target_tag_name);
  if (reg == nullptr) {
    return NullOpt;
  }
  return Target(reg->tag_->config);
}

Map<String, Target> TargetTag::ListTags() {
  Map<String, Target> result;
  for (const String& tag : TargetTagRegistry::Global()->ListAllNames()) {
    result.Set(tag, TargetTag::Get(tag).value());
  }
  return result;
}

Target TargetTag::AddTag(String name, Map<String, ObjectRef> config, bool override) {
  TargetTagRegEntry& tag = TargetTagRegEntry::RegisterOrGet(name).set_name();
  CHECK(override || tag.tag_->config.empty())
      << "Tag \"" << name << "\" has been previously defined as: " << tag.tag_->config;
  tag.set_config(config);
  return Target(config);
}

/**********  Register Target tags  **********/

// `arch` is determined according to https://developer.nvidia.com/cuda-gpus
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
// `shared_memory_per_block`: Maximum amount of shared memory per thread block
// `registers_per_block`: Maximum number of 32-bit registers per thread block
// `max_threads_per_block`: Maximum number of threads per block (always 1024)
// `thread_warp_size`: warp size (always 32)
// `vector_unit_bytes`: 16 (?)

TVM_REGISTER_TARGET_TAG("nvidia/rtx2080ti")
    .set_config({
        {"kind", String("cuda")},
        {"arch", String("sm_75")},
        {"shared_memory_per_block", Integer(49152)},
        {"registers_per_block", Integer(65536)},
        {"max_threads_per_block", Integer(1024)},
        {"vector_unit_bytes", Integer(16)},
        {"thread_warp_size", Integer(32)},
    });

TVM_REGISTER_TARGET_TAG("nvidia/jetson-agx-xavier")
    .set_config({
        {"kind", String("cuda")},
        {"arch", String("sm_72")},
        {"shared_memory_per_block", Integer(49152)},
        {"registers_per_block", Integer(65536)},
        {"max_threads_per_block", Integer(1024)},
        {"vector_unit_bytes", Integer(16)},
        {"thread_warp_size", Integer(32)},
    });

TVM_REGISTER_TARGET_TAG("raspberry-pi/4b")
    .set_config({
        {"kind", String("llvm")},
        {"mtriple", String("armv8l-linux-gnueabihf")},
        {"mcpu", String("cortex-a72")},
        {"mattr", Array<String>{"+neon"}},
        {"num_cores", Integer(4)},
    });

TVM_REGISTER_TARGET_TAG("raspberry-pi/4b-64")
    .set_config({
        {"kind", String("llvm")},
        {"mtriple", String("aarch64-linux-gnu")},
        {"mcpu", String("cortex-a72")},
        {"mattr", Array<String>{"+neon"}},
        {"num_cores", Integer(4)},
    });

}  // namespace tvm
