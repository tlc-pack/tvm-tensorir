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
#ifndef SRC_META_SCHEDULE_MEASURE_CALLBACK_H_
#define SRC_META_SCHEDULE_MEASURE_CALLBACK_H_

#include "./measure.h"

namespace tvm {
namespace meta_schedule {

/*!
 * \brief A callback node used to append newly measured records to a speicifc log file
 */
class RecordToFileNode : public MeasureCallbackNode {
 public:
  /*! \brief Name of the log file to be written */
  String filename;
  /*! \brief Name of the task */
  String task_name;
  /*! \brief Serialized JSON-like object for the target */
  Map<String, ObjectRef> target;
  /*! \brief Serialized JSON-like object for the target_host */
  Map<String, ObjectRef> target_host;
  /*! \brief Base64-encoded PrimFunc */
  String prim_func_b64;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("filename", &filename);
    v->Visit("task_name", &task_name);
    v->Visit("target", &target);
    v->Visit("target_host", &target_host);
    v->Visit("prim_func_b64", &prim_func_b64);
  }

  void Init(const SearchTask& task) override;

  void Callback(const Array<MeasureInput>& inputs, const Array<MeasureResult>& results) override;

  static constexpr const char* _type_key = "meta_schedule.RecordToFile";
  TVM_DECLARE_FINAL_OBJECT_INFO(RecordToFileNode, MeasureCallbackNode);
};

/*!
 * \brief Managed reference to RecordToFileNode
 * \sa RecordToFileNode
 */
class RecordToFile : public MeasureCallback {
 public:
  /*! \brief Default constructor */
  RecordToFile();

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(RecordToFile, MeasureCallback,
                                                    MeasureCallbackNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_MEASURE_CALLBACK_H_
