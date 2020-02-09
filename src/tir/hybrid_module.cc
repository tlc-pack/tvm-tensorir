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
 * \file module.cc
 * \brief The global module in hybrid script.
 */

#include <tvm/tir/hybrid_module.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <utility>
#include <iostream>

namespace tvm {
namespace tir {

GlobalVar::GlobalVar(std::string name_hint) {
  auto n = make_object<GlobalVarNode>();
  n->name_hint = std::move(name_hint);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(GlobalVarNode);

TVM_REGISTER_GLOBAL("make.TirGlobalVar")
.set_body_typed([](std::string name){
  return GlobalVar(std::move(name));
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<GlobalVarNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const GlobalVarNode*>(ref.get());
  p->stream << "TirGlobalVar(" << node->name_hint << ")";
});

Module::Module(Map<GlobalVar, Function> functions) {
  auto n = make_object<ModuleNode>();
  n->functions = std::move(functions);
  n->global_var_map_ = {};
  for (const auto &kv : n->functions) {
    // set global var map
    CHECK(n->global_var_map_.count(kv.first->name_hint) == 0)
      << "Duplicate global function name " << kv.first->name_hint;
    n->global_var_map_.Set(kv.first->name_hint, kv.first);
  }
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ModuleNode);

TVM_REGISTER_GLOBAL("make.TirModule")
.set_body_typed([](Map<GlobalVar, Function> funcs) {
  return Module(std::move(funcs));
});

}  // namespace tir
}  // namespace tvm
