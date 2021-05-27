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
                                const std::unordered_set<const DynamicAxisNode*>::iterator& dyn_axes_iter,
                                const std::unordered_set<const DynamicAxisNode*>::iterator& dyn_axes_end) {
  if (dyn_axes_iter == dyn_axes_end) {
    bool analyzer_result = analyzer.CanProve(predicate);
    if (!analyzer_result && dmlc::GetEnv("DIETCODE_DEBUG_TRACE", 0)) {
      LOG(WARNING) << "Unable to show that (" << predicate << ") is always true";
    }
    (*can_prove) &= analyzer_result;
    return;
  }
  const DynamicAxisNode* const dyn_axis = *dyn_axes_iter;
  std::unordered_set<const DynamicAxisNode*>::iterator dyn_axes_next_iter = dyn_axes_iter;
  ++dyn_axes_next_iter;

  for (const IntImm& v : dyn_axis->possible_values) {
    DynamicAxisReplacer dynamic_axis_replacer(
        [dyn_axis, &v](const DynamicAxisNode* op) -> PrimExpr {
          if (op == dyn_axis) {
            return v;
          } else {
            return GetRef<DynamicAxis>(op);
          }
        });
    canProveForEachDynamicAxis(analyzer, dynamic_axis_replacer(predicate), can_prove,
                               dyn_axes_next_iter, dyn_axes_end);
  }
}


Array<PrimExpr> InitializeDynamicArgs(Array<PrimExpr> args,
                                      Array<String> shape_vars,
                                      Map<Array<IntImm>, FloatImm> shape_freq) {
  std::unordered_map<std::string, std::set<int>> dyn_axes_info;
  for (const std::pair<Array<IntImm>, FloatImm>& shape_freq_pair : shape_freq) {
    CHECK(shape_vars.size() == shape_freq_pair.first.size());
    for (size_t i = 0; i < shape_vars.size(); ++i) {
      dyn_axes_info[std::string(shape_vars[i])].insert(shape_freq_pair.first[i]->value);
    }
  }
  Array<PrimExpr> new_args;
  DynamicAxisReplacer dynamic_axis_replacer(
      [dyn_axes_info](const DynamicAxisNode* op) ->PrimExpr {
        auto dyn_axes_info_iter = dyn_axes_info.find(std::string(op->name_hint));
        if (dyn_axes_info_iter != dyn_axes_info.end()) {
          std::vector<IntImm> possible_values(dyn_axes_info_iter->second.begin(),
                                              dyn_axes_info_iter->second.end());
          return DynamicAxis(dyn_axes_info_iter->first, possible_values);
        } else {
          return GetRef<DynamicAxis>(op);
        }
      });

  for (const PrimExpr& arg : args) {
    new_args.push_back(dynamic_axis_replacer(arg));
  }
  return new_args;
}


}  // namespace anonymous


bool canProveForAllDynamicAxes(arith::Analyzer& analyzer, PrimExpr predicate) {
  DynamicAxisFinder dynamic_axis_finder;
  // find all the dynamic axes within predicate
  dynamic_axis_finder(predicate);
  bool can_prove = true;

  canProveForEachDynamicAxis(analyzer, predicate, &can_prove,
                             dynamic_axis_finder.dyn_axes.begin(),
                             dynamic_axis_finder.dyn_axes.end());
  return can_prove;
}


TVM_REGISTER_GLOBAL("InitializeDynamicArgs").set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = InitializeDynamicArgs(args[0], args[1], args[2]);
});


}  // namespace tir
}  // namespace tvm
