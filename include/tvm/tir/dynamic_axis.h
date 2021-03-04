// <bojian/TVM-SymbolicTuning>
#pragma once

#include <tvm/tir/var.h>


namespace tvm {
namespace tir {


class DyAxisNode : public VarNode {
 public:
  Array<IntImm> possible_values;

  static constexpr const char* _type_key = "tir.DyAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(DyAxisNode, VarNode);
};


class DyAxis : public Var {
 public:
  TVM_DLL DyAxis(String name, Array<IntImm> possible_values);

  TVM_DEFINE_OBJECT_REF_METHODS(DyAxis, Var, DyAxisNode);
};


}  // namespace tir
}  // namespace tvm
