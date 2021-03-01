// <bojian/TVM-SymbolicTuning>

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

  const DyAxisNode* operator->() const {
    return get();
  }
  const DyAxisNode* get() const {
    return static_cast<const DyAxisNode*>(data_.get());
  }

  using ContainerType = DyAxisNode;
};


}  // namespace tir
}  // namespace tvm
