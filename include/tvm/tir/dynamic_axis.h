// <bojian/TVM-SymbolicTuning>

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

  const DynamicAxisNode* operator->() const {
    return get();
  }
  const DynamicAxisNode* get() const {
    return static_cast<const DynamicAxisNode*>(data_.get());
  }

  using ContainerType = DynamicAxisNode;
};


}  // namespace tir
}  // namespace tvm
