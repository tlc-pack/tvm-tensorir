// <bojian/TVM-SymbolicTuning>
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/dynamic_axis.h>
#include <tvm/tir/dynamic_axis_functor.h>


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


namespace {

void canProveForEachDyAxis(arith::Analyzer& analyzer, PrimExpr predicate, bool* const can_prove,
                           const std::unordered_set<const DyAxisNode*>::iterator& dyaxes_iter,
                           const std::unordered_set<const DyAxisNode*>::iterator& dyaxes_end) {
  if (dyaxes_iter == dyaxes_end) {
    bool analyzer_result = analyzer.CanProve(predicate);
    if (!analyzer_result) {
      LOG(WARNING) << "Unable to show that (" << predicate << ") is always true";
    }
    (*can_prove) &= analyzer_result;
    return;
  }
  const DyAxisNode* const dy_axis = *dyaxes_iter;
  std::unordered_set<const DyAxisNode*>::iterator dyaxes_next_iter = dyaxes_iter;
  ++dyaxes_next_iter;

  for (const IntImm& v : dy_axis->possible_values) {
    DyAxisSubstituter dyaxis_substituter(dy_axis, v);
    canProveForEachDyAxis(analyzer, dyaxis_substituter(predicate), can_prove,
                          dyaxes_next_iter, dyaxes_end);
  }
}


}  // namespace anonymous


bool canProveForAllDyAxes(arith::Analyzer& analyzer, PrimExpr predicate) {
  DyAxisFinder dyaxis_finder;
  // find all the dynamic axes within predicate
  dyaxis_finder(predicate);
  bool can_prove = true;

  canProveForEachDyAxis(analyzer, predicate, &can_prove,
                        dyaxis_finder.dy_axes.begin(),
                        dyaxis_finder.dy_axes.end());
  return can_prove;
}


}  // namespace tir
}  // namespace tvm
