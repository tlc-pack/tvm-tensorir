/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Helper functions to write schedule
 */

#ifndef TVM_TENSORIR_SCHEDULE_HELPER_H_
#define TVM_TENSORIR_SCHEDULE_HELPER_H_

#include "schedule.h"

namespace tvm {
namespace tensorir {

// Inline all elemwise and broadcast blocks
void InlineAllInjective(Schedule sch);

}  // namespace tensorir
}  // namespace tvm

#endif  // TVM_TENSORIR_SCHEDULE_HELPER_H_

