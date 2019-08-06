/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Extra utilities for tvm::Node
 */

#ifndef TVM_TENSORIR_NODE_UTIL_H_
#define TVM_TENSORIR_NODE_UTIL_H_

#include <tvm/base.h>
#include <tvm/tensor.h>
#include <unordered_set>
#include <unordered_map>

namespace tvm {

template <typename Key, typename Value>
struct StdNodeMap_T { using type = std::unordered_map<Key, Value, NodeHash, NodeEqual>; };
template <typename Key>
struct StdNodeSet_T { using type = std::unordered_set<Key, NodeHash, NodeEqual>; };

// Specialization for Tensor, because Tensor overrides operator= and std::hash,
// we don't want to use the default NodeHash and NodeEqual
template <>
struct StdNodeSet_T<Tensor> { using type = std::unordered_set<Tensor>; };

template <typename Value>
struct StdNodeMap_T<Tensor, Value> { using type = std::unordered_map<Tensor, Value>; };

// Wrap std::unordered_map and std::unordered_set for NodeRef
// It is recommended to use this class when the key is NodeRef (e.g Expr, Tensor, Stmt)
// It will do specialization to pick the correct hash and equal functor for the key.
template <typename Key, typename Value>
using StdNodeMap = typename StdNodeMap_T<Key, Value>::type;

template <typename Key>
using StdNodeSet = typename StdNodeSet_T<Key>::type;

// Macro to define common methods for a mutable NodeRef
#define TVM_DEFINE_MUTABLE_NODE_REF_METHODS(TypeName, BaseType, NodeName)  \
  TypeName() {}                                                            \
  explicit TypeName(::tvm::NodePtr<::tvm::Node> n) : BaseType(n) {}        \
  const NodeName* operator->() const {                                     \
    return static_cast<const NodeName*>(node_.get());                      \
  }                                                                        \
  NodeName* operator->() {                                                 \
    return static_cast<NodeName*>(node_.get());                            \
  }                                                                        \
  operator bool() const { return this->defined(); }                        \
  using ContainerType = NodeName;

// Macro to define a mutable NodeRef class
#define TVM_DEFINE_MUTABLE_NODE_REF(TypeName, BaseType, NodeName)       \
  class TypeName : public BaseType {                                    \
   public:                                                              \
    TVM_DEFINE_MUTABLE_NODE_REF_METHODS(TypeName, BaseType, NodeName);  \
  };                                                                    \

}  // namespace tvm

#endif  // TVM_TENSORIR_NODE_UTIL_H_
