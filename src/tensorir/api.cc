/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief API registration
 */

#include <sstream>
#include <tvm/api_registry.h>
#include "schedule.h"
#include "tree_builder.h"

namespace tvm {
namespace tensorir {

TVM_REGISTER_API("tensorir.schedule.CreateSchedule")
.set_body_typed(ScheduleNode::make);

TVM_REGISTER_API("tensorir.schedule.PrintTreeNode")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  std::ostringstream os;
  PrintTreeNode(os, args[0], 0);
  *rv = os.str();
});

TVM_REGISTER_API("tensorir.schedule.ScheduleBlocks")
.set_body_method(&Schedule::blocks);

TVM_REGISTER_API("tensorir.schedule.ScheduleAxis")
.set_body_method(&Schedule::axis);

//TVM_REGISTER_API("tensorir.schedule.ScheduleSplit")
//.set_body_method(&Schedule::split);
//
//TVM_REGISTER_API("tensorir.schedule.ScheduleFuse")
//.set_body_method(&Schedule::fuse);
//
//TVM_REGISTER_API("tensorir.schedule.ScheduleReorder")
//.set_body_method(&Schedule::reorder);
//
//TVM_REGISTER_API("tensorir.schedule.ScheduleComputeInline")
//.set_body_method(&Schedule::compute_inline);

TVM_REGISTER_API("tensorir.schedule.ScheduleComputeAt")
.set_body_method(&Schedule::compute_at);

//TVM_REGISTER_API("tensorir.schedule.ScheduleComputeRoot")
//.set_body_method(&Schedule::compute_root);

//TVM_REGISTER_API("tensorir.schedule.ScheduleBind")
//.set_body([](TVMArgs args, TVMRetValue* rv) {
//  *rv = args[0].operator Schedule().bind(args[0], args[1]);
//});

} // namespace tensorir
} // namespace tvm

