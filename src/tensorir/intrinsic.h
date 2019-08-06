/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Tensor intrinsics for tensorize
 */

#ifndef TVM_TENSORIR_INTRINSIC_H_
#define TVM_TENSORIR_INTRINSIC_H_

#include <string>
#include "intrinsic.h"
#include "tree_node.h"

namespace tvm {
namespace tensorir {

using runtime::PackedFunc;
using runtime::TypedPackedFunc;

// A tensor intrinsic replaces a block to another block
class TensorIntrinsic;
class TensorIntrinsicNode : public Node {
 public:
  Operation op;             // semantic form
  TypedPackedFunc<NodeRef(Array<TensorRegion>, Array<TensorRegion>)> intrin_func;
  std::string name;

  void VisitAttrs(AttrVisitor *v) final {
    PackedFunc packed = intrin_func.packed();
    v->Visit("op", &op);
    v->Visit("intrin_func", &packed);
    v->Visit("name", &name);
  }

  TVM_DLL static TensorIntrinsic make(
      Operation op,
      TypedPackedFunc<NodeRef(Array<TensorRegion>, Array<TensorRegion>)> intrin_func,
      std::string name);

  static constexpr const char *_type_key = "tensorir.TensorIntrinsic";
  TVM_DECLARE_NODE_TYPE_INFO(TensorIntrinsicNode, Node);
};

class TensorIntrinsic : public NodeRef {
 public:
  // Generate intrinsic code for the given inputs and outputs region
  ScheduleTreeNode Instantiate(Array<TensorRegion> inputs, Array<TensorRegion> outputs) const;

  TVM_DEFINE_NODE_REF_METHODS(TensorIntrinsic, NodeRef, TensorIntrinsicNode);
};


}  // namespace tensorir
}  // namespace tvm

#endif  // TVM_TENSORIR_INTRINSIC_H_
