// <bojian/DietCode>
#pragma once

#include <tvm/tir/var.h>


namespace tvm {
namespace tir {


class DynamicAxisNode : public VarNode {
 public:
  Array<IntImm> possible_values;

  static constexpr const char* _type_key = "tir.DynamicAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(DynamicAxisNode, VarNode);
};


class DynamicAxis : public Var {
 public:
  TVM_DLL DynamicAxis(String name, Array<IntImm> possible_values);

  TVM_DEFINE_OBJECT_REF_METHODS(DynamicAxis, Var, DynamicAxisNode);
};


}  // namespace tir
}  // namespace tvm
