// <bojian/DietCodes>
#include <dmlc/parameter.h>

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/dynamic_axis.h>
#include <tvm/tir/dynamic_axis_functor.h>


namespace tvm {
namespace tir {


DynamicAxis::DynamicAxis(String name, Array<IntImm> possible_values) {
  auto n = make_object<DynamicAxisNode>();
  n->name_hint = std::move(name);
  n->dtype = DataType::Int(32);
  n->possible_values = std::move(possible_values);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("tir.DynamicAxis")
    .set_body_typed([](String name, Array<IntImm> possible_values) {
      DynamicAxis ret(name, possible_values);
      LOG(INFO) << ret;
      return ret;
    });


TVM_REGISTER_NODE_TYPE(DynamicAxisNode);


TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<DynamicAxisNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const DynamicAxisNode*>(node.get());
      p->stream << "Dynamic Axis " << op->name_hint;
      p->stream << " : [";
      for (const IntImm& I : op->possible_values) {
        p->stream << I->value << ", ";
      }
      p->stream << "]";
    });


namespace {

void canProveForEachDynamicAxis(arith::Analyzer& analyzer, PrimExpr predicate, bool* const can_prove,
                                const std::unordered_set<const DynamicAxisNode*>::iterator& dyaxes_iter,
                                const std::unordered_set<const DynamicAxisNode*>::iterator& dyaxes_end) {
  if (dyaxes_iter == dyaxes_end) {
    bool analyzer_result = analyzer.CanProve(predicate);
    if (!analyzer_result && dmlc::GetEnv("DIETCODE_DEBUG_TRACE", 0)) {
      LOG(WARNING) << "Unable to show that (" << predicate << ") is always true";
    }
    (*can_prove) &= analyzer_result;
    return;
  }
  const DynamicAxisNode* const dy_axis = *dyaxes_iter;
  std::unordered_set<const DynamicAxisNode*>::iterator dyaxes_next_iter = dyaxes_iter;
  ++dyaxes_next_iter;

  for (const IntImm& v : dy_axis->possible_values) {
    DynamicAxisSubstituter dynamic_axis_substituter(dy_axis, v);
    canProveForEachDynamicAxis(analyzer, dynamic_axis_substituter(predicate), can_prove,
                               dyaxes_next_iter, dyaxes_end);
  }
}


}  // namespace anonymous


bool canProveForAllDynamicAxes(arith::Analyzer& analyzer, PrimExpr predicate) {
  DynamicAxisFinder dynamic_axis_finder;
  // find all the dynamic axes within predicate
  dynamic_axis_finder(predicate);
  bool can_prove = true;

  canProveForEachDynamicAxis(analyzer, predicate, &can_prove,
                             dynamic_axis_finder.dy_axes.begin(),
                             dynamic_axis_finder.dy_axes.end());
  return can_prove;
}


}  // namespace tir
}  // namespace tvm
