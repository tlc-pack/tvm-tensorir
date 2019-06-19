/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Extra utilities for tvm::Node
 */

#ifndef TVM_TENSORIR_NODE_UTIL_H_
#define TVM_TENSORIR_NODE_UTIL_H_

#include <unordered_set>
#include <unordered_map>
#include <tvm/base.h>

#include <tvm/tensor.h>

namespace tvm {

// wrap std::unordered_map and std::unordered_set for NodeRef
template <typename Key, typename Value>
struct StdNodeMap_T { using type = std::unordered_map<Key, Value, NodeHash, NodeEqual>; };

// specialization for Tensor, because Tensor overrides operator= and std::hash
template <typename Value>
struct StdNodeMap_T<Tensor, Value> { using type = std::unordered_map<Tensor, Value>; };

template <typename Key, typename Value>
using StdNodeMap = typename StdNodeMap_T<Key, Value>::type;

template <typename Key>
struct StdNodeSet_T { using type = std::unordered_set<Key, NodeHash, NodeEqual>; };

// specialization for Tensor, because Tensor overrides operator= and std::hash
template <>
struct StdNodeSet_T<Tensor> { using type = std::unordered_set<Tensor>; };

template <typename Key>
using StdNodeSet = typename StdNodeSet_T<Key>::type;

#define TVM_DEFINE_MUTABLE_NODE_REF(TypeName, BaseType, NodeName)  \
  class TypeName : public BaseType {                               \
   public:                                                         \
    NodeName* operator->(){                                        \
      return static_cast<NodeName*>(node_.get());                  \
    }                                                              \
    TVM_DEFINE_NODE_REF_METHODS(TypeName, BaseType, NodeName);     \
  };                                                               \

} // namespace tvm

#endif // TVM_TENSORIR_NODE_UTIL_H_
