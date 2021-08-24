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

#include "./transform.h"

namespace tvm {
namespace tir {

/******** Annotation ********/
Block WithAnnotation(const BlockNode* block, const String& attr_key, const ObjectRef& attr_value) {
  Map<String, ObjectRef> annotations = block->annotations;
  annotations.Set(attr_key, attr_value);
  ObjectPtr<BlockNode> new_block = make_object<BlockNode>(*block);
  new_block->annotations = std::move(annotations);
  return Block(new_block);
}

/******** Buffer Related ********/
Buffer WithScope(const Buffer& buffer, const String& scope) {
  auto n = make_object<BufferNode>(*buffer.get());
  auto new_ptr = make_object<VarNode>(*n->data.get());
  const auto* ptr_type = new_ptr->type_annotation.as<PointerTypeNode>();
  ICHECK(ptr_type);
  new_ptr->type_annotation = PointerType(ptr_type->element_type, scope);
  n->data = Var(new_ptr->name_hint + "_" + scope, new_ptr->type_annotation);
  n->name = buffer->name + "_" + scope;
  return Buffer(n);
}

}  // namespace tir
}  // namespace tvm
