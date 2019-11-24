/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief TE API registration
 */

#include <tvm/te/transform.h>
#include <tvm/api_registry.h>
#include <tvm/te/ir.h>

namespace tvm {
namespace te {


TVM_REGISTER_API("ir_pass.TeLower")
.set_body_typed(TeLower);


// maker
TVM_REGISTER_API("make.TensorRegion")
.set_body_typed(TensorRegionNode::make);

TVM_REGISTER_API("make.BufferAllocate")
.set_body_typed(BufferAllocateNode::make);

TVM_REGISTER_API("make.BufferLoad")
.set_body_typed(BufferLoadNode::make);

TVM_REGISTER_API("make.BufferStore")
.set_body_typed(BufferStoreNode::make);

TVM_REGISTER_API("make.Loop")
.set_body_typed(LoopNode::make);

TVM_REGISTER_API("make.TeBlock")
.set_body_typed(BlockNode::make);

TVM_REGISTER_API("make.TeFunction")
.set_body_typed(FunctionNode::make);

}  // namespace te
}  // namespace tvm