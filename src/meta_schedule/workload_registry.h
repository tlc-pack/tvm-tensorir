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
#ifndef SRC_META_SCHEDULE_WORKLOAD_REGISTERY_H_
#define SRC_META_SCHEDULE_WORKLOAD_REGISTERY_H_

#include <tvm/ir/module.h>

#include <unordered_map>
#include <vector>

namespace tvm {
namespace meta_schedule {

/*! \brief The class of workload tokens. */
class WorkloadTokenNode : public runtime::Object {
 public:
  /*! \brief The workload's IRModule. */
  IRModule mod;
  /*! \brief The workload's structural hash. */
  String shash;
  /*! \brief The workload's token id. */
  int64_t token_id_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("mod", &mod);
    v->Visit("shash", &shash);
    // `token_id_` is not visited
  }

  /*!
   * \brief Export the workload token to a JSON string.
   * \return An array containing the structural hash and the base64 json string.
   */
  ObjectRef AsJSON() const;

  static constexpr const char* _type_key = "meta_schedule.WorkloadToken";
  TVM_DECLARE_FINAL_OBJECT_INFO(WorkloadTokenNode, runtime::Object);
};

/*!
 * \brief Managed reference to WorkloadTokenNode.
 *  \sa WorkloadTokenNode
 */
class WorkloadToken : public runtime::ObjectRef {
 public:
  /*!
   * \brief Constructor of WorkloadToken.
   * \param mod The workload's IRModule.
   * \param shash The workload's structural hash.
   * \param token_id The workload's token id.
   */
  TVM_DLL explicit WorkloadToken(IRModule mod, String shash, int64_t token_id);

  /*!
   * \brief Create a workload token from a JSON string.
   * \param json_obj The ObjectRef containing the json string.
   * \param token_id The workload's token id.
   * \return The created workload token.
   */
  TVM_DLL static WorkloadToken FromJSON(const ObjectRef& json_obj, int64_t token_id);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(WorkloadToken, runtime::ObjectRef, WorkloadTokenNode);
};

/*! \brief The class for workload registry. */
class WorkloadRegistryNode : public runtime::Object {
 public:
  /*! \brief The workload registry's storage path. */
  String path;
  /*! \brief The map from workload to its corresponding workload token id. */
  std::unordered_map<IRModule, int64_t, tvm::StructuralHash, tvm::StructuralEqual> mod2token_id_;
  /*! \brief The vector of workload tokens. */
  std::vector<WorkloadToken> workload_tokens_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("path", &path);
    // `mod2token_id_` is not visited
    // `workload_tokens_` is not visited
  }

  static constexpr const char* _type_key = "meta_schedule.WorkloadRegistry";
  TVM_DECLARE_FINAL_OBJECT_INFO(WorkloadRegistryNode, runtime::Object);

 public:
  /*!
   * \brief Look up or add one IRModule in the workload registry.
   * \param mod The IRModule to look up or add.
   * \return The corresponding workload token.
   */
  TVM_DLL WorkloadToken LookupOrAdd(const IRModule& mod);

  /*!
   * \brief Get the size of the workload registry.
   * \return The size of the registry.
   */
  TVM_DLL int64_t Size() const;

  /*!
   * \brief Get the workload token given its id.
   * \param token_id The id of the workload token.
   * \return The requested workload token.
   */
  TVM_DLL WorkloadToken At(int64_t token_id) const;
};

/*!
 * \brief Managed reference to WorkloadRegistryNode.
 * \sa WorkloadRegistryNode
 */
class WorkloadRegistry : public runtime::ObjectRef {
 public:
  TVM_DLL WorkloadRegistry(String path, bool allow_missing);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(WorkloadRegistry, runtime::ObjectRef,
                                                    WorkloadRegistryNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_WORKLOAD_REGISTERY_H_
