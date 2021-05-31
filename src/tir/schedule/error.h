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
#ifndef TVM_TIR_SCHEDULE_ERROR_H_
#define TVM_TIR_SCHEDULE_ERROR_H_

#include <tvm/tir/schedule/state.h>

namespace tvm {
namespace tir {

/*! \brief Error that happens during TensorIR scheduling */
class ScheduleError : public tvm::runtime::Error {
 public:
  /*! \brief Base constructor */
  ScheduleError() : tvm::runtime::Error("") {}
  /*! \brief The IRModule that is applied on */
  virtual IRModule mod() const = 0;
  /*! \brief The locations of interest that we want to point out */
  virtual Array<ObjectRef> LocationsOfInterest() const = 0;
  /*!
   * \brief Returns an error string template for rendering, corresponds to the "detail" mode.
   * \sa ScheduleErrorRenderLevel
   * \note Format of the template:
   * "Some normal error string and curly bracket-enclosed numbers like {0}, which indicates the
   * 0-th element of the interested location."
   */
  virtual String DetailRenderTemplate() const = 0;
  /*!
   * \brief Returns an error string without needing to render, corresponds to the "fast" mode
   * \sa ScheduleErrorRenderLevel
   */
  virtual String FastErrorString() const = 0;
  /*! \brief Render the ScheduleError with the template provided by `DetailRenderTemplate` */
  String RenderReport(const String& primitive) const;
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_ERROR_H_
