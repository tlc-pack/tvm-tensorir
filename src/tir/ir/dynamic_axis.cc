#include <tvm/runtime/registry.h>
#include <tvm/tir/dynamic_axis.h>

namespace tvm {
namespace tir {


DyAxis::DyAxis(String name, Array<IntImm> possible_values) {
  auto n = make_object<DyAxisNode>();
  n->name_hint = std::move(name);
  n->dtype = DataType::Int(32);
  n->possible_values = std::move(possible_values);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("tir.DyAxis")
    .set_body_typed([](String name, Array<IntImm> possible_values) {
      DyAxis ret(name, possible_values);
      LOG(INFO) << ret;
      return ret;
    });


TVM_REGISTER_NODE_TYPE(DyAxisNode);


TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<DyAxisNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const DyAxisNode*>(node.get());
      p->stream << "Dynamic Axis " << op->name_hint;
      p->stream << " : [";
      for (const IntImm& I : op->possible_values) {
        p->stream << I->value << ", ";
      }
      p->stream << "]";
    });


}  // namespace tir
}  // namespace tvm
