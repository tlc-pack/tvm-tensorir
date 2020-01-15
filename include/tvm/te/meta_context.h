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
 * \file tvm/te/meta_context.h
 * \brief Meta data context for TePrinter
 */

#ifndef TVM_META_CONTEXT_H
#define TVM_META_CONTEXT_H

#include <tvm/node/serialization.h>

namespace tvm {
namespace te {

class TextMetaDataContext {
 public:
  std::string GetMetaNode(const ObjectRef& node) {
    auto it = meta_repr_.find(node);
    if (it != meta_repr_.end()) {
      return it->second;
    }
    std::string type_key = node->GetTypeKey();
    CHECK(!type_key.empty());
    Array<ObjectRef>& mvector = meta_data_[type_key];
    auto index = static_cast<int64_t>(mvector.size());
    mvector.push_back(node);
    std::ostringstream doc;
    doc << "meta[" << type_key << "][" << index << "]";
    meta_repr_[node] = doc.str();
    return meta_repr_[node];
  }

  std::string GetMetaSection() const {
    if (meta_data_.empty()) return std::string();
    return SaveJSON(Map<std::string, ObjectRef>(meta_data_.begin(), meta_data_.end()));
  }

  bool empty() const {
    return meta_data_.empty();
  }

 private:
  std::unordered_map<std::string, Array<ObjectRef> > meta_data_;
  std::unordered_map<ObjectRef, std::string, ObjectHash, ObjectEqual> meta_repr_;
};

}  // namespace te
}  // namespace tvm

#endif  // TVM_META_CONTEXT_H
