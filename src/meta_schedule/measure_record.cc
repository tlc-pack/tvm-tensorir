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
#include "./measure_record.h"  // NOLINT(build/include)

#include <tvm/node/node.h>

#include <algorithm>

#include "./utils.h"

namespace tvm {
namespace meta_schedule {

/********** Constructors **********/

MeasureInput::MeasureInput(SearchTask task, Schedule sch, Optional<Array<IntImm>> variant) {
  ObjectPtr<MeasureInputNode> n = make_object<MeasureInputNode>();
  n->task = std::move(task);
  n->sch = std::move(sch);
  n->variant = std::move(variant);
  data_ = std::move(n);
}

BuildResult::BuildResult(String filename, int error_no, String error_msg, double time_cost) {
  ObjectPtr<BuildResultNode> n = make_object<BuildResultNode>();
  n->filename = std::move(filename);
  n->error_no = error_no;
  n->error_msg = std::move(error_msg);
  n->time_cost = time_cost;
  data_ = std::move(n);
}

MeasureResult::MeasureResult(Array<FloatImm> costs, int error_no, String error_msg, double all_cost,
                             double timestamp) {
  ObjectPtr<MeasureResultNode> n = make_object<MeasureResultNode>();
  n->costs = std::move(costs);
  n->error_no = error_no;
  n->error_msg = std::move(error_msg);
  n->all_cost = all_cost;
  n->timestamp = timestamp;
  data_ = std::move(n);
}

/********** Member methods **********/

MeasureInput MeasureInputNode::Copy() const {
  ObjectPtr<MeasureInputNode> n = make_object<MeasureInputNode>();
  n->task = task;
  n->sch = sch;
  n->variant = variant;
  return MeasureInput(n);
}

MeasureResult MeasureResultNode::Copy() const {
  ObjectPtr<MeasureResultNode> n = make_object<MeasureResultNode>();
  n->costs = costs;
  n->error_no = error_no;
  n->error_msg = error_msg;
  n->all_cost = all_cost;
  n->timestamp = timestamp;
  return MeasureResult(n);
}

double MeasureResultNode::MeanCost() const { return FloatArrayMean(this->costs); }

/********** Printing functions **********/

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MeasureResultNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const MeasureResultNode*>(ref.get());
      if (node->error_no == static_cast<int>(MeasureErrorNO::kNoError)) {
        p->stream << "MeasureResult(cost:[";
        auto old_config = p->stream.precision(4);
        for (size_t i = 0; i < node->costs.size(); ++i) {
          auto pf = node->costs[i].as<FloatImmNode>();
          ICHECK(pf != nullptr);
          p->stream << pf->value;
          if (i != node->costs.size() - 1) {
            p->stream << ",";
          }
        }
        p->stream.precision(old_config);
        p->stream << "], ";
        p->stream << "error_no:" << 0 << ", "
                  << "all_cost:" << node->all_cost << ", "
                  << "Tstamp:" << node->timestamp << ")";
      } else {
        p->stream << "MeasureResult("
                  << "error_type:"
                  << MeasureErrorNOToStr(static_cast<MeasureErrorNO>(node->error_no)) << ", "
                  << "error_msg:" << node->error_msg << ", "
                  << "all_cost:" << node->all_cost << ", "
                  << "Tstamp:" << node->timestamp << ")";
      }
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BuildResultNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const BuildResultNode*>(ref.get());
      p->stream << "BuildResult(" << node->filename << ", " << node->error_no << ", "
                << node->time_cost << ")";
    });

/********** FFI **********/

struct Internal {
  /********** Constructors **********/
  /*!
   * \brief Constructor of MeasureInput
   * \param task The task to be measured
   * \param state Concrete schedule of the task
   * \return The MeasureInput constructed
   * \sa MeasureInput::MeasureInput
   */
  static MeasureInput MeasureInputNew(SearchTask task, Schedule sch,
                                      Optional<Array<IntImm>> variant = NullOpt) {
    return MeasureInput(task, sch, variant);
  }
  /*!
   * \brief Constructor of BuildResult
   * \param filename The filename of built binary file.
   * \param error_no The error code.
   * \param error_msg The error message if there is any error.
   * \param time_cost The time cost of build.
   * \return The BuildResult constructed
   * \sa BuildResult::BuildResult
   */
  static BuildResult BuildResultNew(String filename, int error_no, String error_msg,
                                    double time_cost) {
    return BuildResult(filename, error_no, error_msg, time_cost);
  }
  /*!
   * \brief Constructor of MeasureResult
   * \param costs The time costs of execution.
   * \param error_no The error code.
   * \param error_msg The error message if there is any error.
   * \param all_cost The time cost of build and run.
   * \param timestamp The time stamps of this measurement.
   * \return The MeasureResult constructed
   * \sa MeasureResult::MeasureResult
   */
  static MeasureResult MeasureResultNew(Array<FloatImm> costs, int error_no, String error_msg,
                                        double all_cost, double timestamp) {
    return MeasureResult(costs, error_no, error_msg, all_cost, timestamp);
  }
  /*!
   * \brief The average cost
   * \sa MeasureResultNode::MeanCost
   */
  static double MeasureResultMeanCost(MeasureResult measure_result) {
    return measure_result->MeanCost();
  }
};

TVM_REGISTER_NODE_TYPE(MeasureInputNode);
TVM_REGISTER_NODE_TYPE(BuildResultNode);
TVM_REGISTER_NODE_TYPE(MeasureResultNode);

TVM_REGISTER_GLOBAL("meta_schedule.MeasureInput").set_body_typed(Internal::MeasureInputNew);
TVM_REGISTER_GLOBAL("meta_schedule.BuildResult").set_body_typed(Internal::BuildResultNew);
TVM_REGISTER_GLOBAL("meta_schedule.MeasureResult").set_body_typed(Internal::MeasureResultNew);
TVM_REGISTER_GLOBAL("meta_schedule.MeasureResultMeanCost")
    .set_body_typed(Internal::MeasureResultMeanCost);

}  // namespace meta_schedule
}  // namespace tvm
