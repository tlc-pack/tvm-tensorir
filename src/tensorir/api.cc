/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief API registration
 */

#include <tvm/api_registry.h>
#include <tvm/operation.h>
#include <sstream>
#include "schedule.h"
#include "tree_builder.h"
#include "schedule_helper.h"

namespace tvm {
namespace tensorir {

TVM_REGISTER_API("tensorir.schedule.CreateSchedule")
.set_body_typed(ScheduleNode::make);

TVM_REGISTER_API("tensorir.tree_node.PrintTreeNode")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  std::ostringstream os;
  PrintTreeNode(os, args[0], 0);
  *rv = os.str();
});

// Schedule member functions
TVM_REGISTER_API("tensorir.schedule.ScheduleBlocks")
.set_body_method(&Schedule::blocks);

TVM_REGISTER_API("tensorir.schedule.ScheduleAxis")
.set_body_method(&Schedule::axis);

TVM_REGISTER_API("tensorir.schedule.ScheduleOutputBlocks")
.set_body_method(&Schedule::output_blocks);

TVM_REGISTER_API("tensorir.schedule.ScheduleReductionBlocks")
.set_body_method(&Schedule::reduction_blocks);

TVM_REGISTER_API("tensorir.schedule.SchedulePredecessor")
.set_body_method(&Schedule::predecessor);

TVM_REGISTER_API("tensorir.schedule.ScheduleSuccessor")
.set_body_method(&Schedule::successor);

TVM_REGISTER_API("tensorir.schedule.ScheduleSplit")
.set_body_method(&Schedule::split);

TVM_REGISTER_API("tensorir.schedule.ScheduleSplitNParts")
.set_body_method(&Schedule::split_nparts);

TVM_REGISTER_API("tensorir.schedule.ScheduleFuse")
.set_body_method(&Schedule::fuse);

TVM_REGISTER_API("tensorir.schedule.ScheduleReorder")
.set_body_method(&Schedule::reorder);

TVM_REGISTER_API("tensorir.schedule.ScheduleUnroll")
.set_body_method(&Schedule::unroll);

TVM_REGISTER_API("tensorir.schedule.ScheduleComputeInline")
.set_body_method(&Schedule::compute_inline);

TVM_REGISTER_API("tensorir.schedule.ScheduleComputeAt")
.set_body_method(&Schedule::compute_at);

TVM_REGISTER_API("tensorir.schedule.ScheduleComputeAfter")
.set_body_method(&Schedule::compute_after);

TVM_REGISTER_API("tensorir.schedule.ScheduleComputeRoot")
.set_body_method(&Schedule::compute_root);

TVM_REGISTER_API("tensorir.schedule.ScheduleBind")
.set_body_method(&Schedule::bind);

TVM_REGISTER_API("tensorir.schedule.ScheduleBlockize")
.set_body_method(&Schedule::blockize);

TVM_REGISTER_API("tensorir.schedule.ScheduleUnblockize")
.set_body_method(&Schedule::unblockize);

TVM_REGISTER_API("tensorir.schedule.ScheduleTensorize")
.set_body_method(&Schedule::tensorize);

TVM_REGISTER_API("tensorir.schedule.ScheduleUntensorize")
.set_body_method(&Schedule::untensorize);

TVM_REGISTER_API("tensorir.schedule.ScheduleAnnotate")
.set_body_method(&Schedule::annotate);

TVM_REGISTER_API("tensorir.schedule.ScheduleToHalide")
.set_body_method(&Schedule::ToHalide);

TVM_REGISTER_API("tensorir.schedule.ScheduleCheckFatherLink")
.set_body_method(&Schedule::CheckFatherLink);

TVM_REGISTER_API("tensorir.schedule.ScheduleInlineAllInjective")
.set_body_typed(InlineAllInjective);

// maker
TVM_REGISTER_API("make.TensorRegion")
.set_body_typed(TensorRegionNode::make);

TVM_REGISTER_API("make.BlockTreeNode")
.set_body_typed(BlockTreeNodeNode::make);

TVM_REGISTER_API("make.AxisTreeNode")
.set_body_typed<AxisTreeNode(Var, Expr, Expr, int, Array<ScheduleTreeNode>)>([](
    Var loop_var, Expr min, Expr extent,
    int axis_type,
    Array<ScheduleTreeNode> children) {
  return AxisTreeNodeNode::make(loop_var,
                                min,
                                extent,
                                static_cast<AxisType>(axis_type),
                                children);
});

TVM_REGISTER_API("make._TensorIntrinsic")
.set_body_typed(TensorIntrinsicNode::make);

// other member functions
TVM_REGISTER_API("_TensorRegion_MakeView")
.set_body_method(&TensorRegion::MakeView);

TVM_REGISTER_API("_TensorIntrinsic_Instantiate")
.set_body_method(&TensorIntrinsic::Instantiate);

}  // namespace tensorir
}  // namespace tvm
