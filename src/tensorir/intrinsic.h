/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Tensor intrinsics
 */

#ifndef TVM_TENSORIR_INTRINSIC_H_
#define TVM_TENSORIR_INTRINSIC_H_

#include "tree_node.h"

namespace tvm {
namespace tensorir {

using runtime::PackedFunc;

// A tensor intrinsic replaces a block to another block
class TensorIntrinsic;
class TensorIntrinsicNode : public Node {
 public:
  Operation op;             // semantic form
  PackedFunc intrin_func;   // (Array<TensorRegion>, Array<TensorRegion>) -> ScheduleTreeNode or Stmt,
                            // todo(lmzheng): use TypedPackedFunc？
  std::string name;

  void VisitAttrs(AttrVisitor *v) final {
    v->Visit("op", &op);
    //v->Visit("intrin_func", &intrin_func);  // todo(lmzheng): fix AttrVisitor
    v->Visit("name", &name);
  }

  TVM_DLL static TensorIntrinsic make(Operation op, PackedFunc intrin_func, std::string name);

  static constexpr const char *_type_key = "tensorir.TensorIntrinsic";
  TVM_DECLARE_NODE_TYPE_INFO(TensorIntrinsicNode, Node);
};

class TensorIntrinsic : public NodeRef {
 public:
  TensorIntrinsic() {}
  explicit TensorIntrinsic(NodePtr<Node> n): NodeRef(n) {}

  const TensorIntrinsicNode* operator->() const;
  ScheduleTreeNode Instantiate(Array<TensorRegion> inputs, Array<TensorRegion> outputs) const;

  using ContainerType = TensorIntrinsicNode;
};


} // namespace tensorir
} // namespace tvm


#endif // TVM_TENSORIR_INTRINSIC_H_
