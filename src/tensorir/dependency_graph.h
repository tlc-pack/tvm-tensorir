/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Dependency between blocks
 */

#ifndef TVM_TENSORIR_DEPENDENCY_GRAPH_H_
#define TVM_TENSORIR_DEPENDENCY_GRAPH_H_

#include <list>
#include <tvm/base.h>
#include <tvm/ir.h>
#include <tvm/node/container.h>
#include <tvm/tensor.h>
#include "tree_node.h"

namespace tvm {
namespace tensorir {

// Two-level vector : the first level is for multiple access, the second level is for multiple dimension.
// This might be extended to handle the tensor region.
using AccessSet = int;  // currently it is useless

// Dependency type
enum EdgeType : int {
  kRAW,
  kWAW,
  kWAR,
  kRAR,
  kUnknown
};

class Edge;
class EdgeNode : public Node {
 public:
  BlockTreeNode dst;
  EdgeType type;
  Array<Array<Expr> > access; // Todo(lmzheng) : define type

  void VisitAttrs(AttrVisitor* v) final {
  }

  TVM_DLL static Edge make(BlockTreeNode dst, EdgeType type);

  static constexpr const char* _type_key = "tensorir.EdgeNode";
  TVM_DECLARE_NODE_TYPE_INFO(EdgeNode, Node);
};

TVM_DEFINE_MUTABLE_NODE_REF(Edge, NodeRef, EdgeNode);


/*!
 * \brief A bipartite graph to store
 * read/write access information between tensors and statements
 */
class ReadWriteGraph {
 public:
  // build 4 indexes for query
  using QueryByTensorMap = StdNodeMap<Tensor, StdNodeMap<BlockTreeNode, std::shared_ptr<AccessSet> > >;
  using QueryByBlockMap   = StdNodeMap<BlockTreeNode, StdNodeMap<Tensor, std::shared_ptr<AccessSet> > >;

  // insert read/write access
  void InsertRead(BlockTreeNode block, Tensor tensor) {
    std::shared_ptr<AccessSet> acc_set = query_by_block_read[block][tensor];
    if (acc_set == nullptr) {
      acc_set = std::make_shared<AccessSet>();
      query_by_block_read[block][tensor] = acc_set;
      query_by_tensor_read[tensor][block] = acc_set;
    }
  }

  void InsertWrite(BlockTreeNode block, Tensor tensor) {
    std::shared_ptr<AccessSet> acc_set = query_by_block_write[block][tensor];
    if (acc_set == nullptr) {
      acc_set = std::make_shared<AccessSet>();
      query_by_block_write[block][tensor] = acc_set;
      query_by_tensor_write[tensor][block] = acc_set;
    }
  }

  // query
  StdNodeMap<Tensor, std::shared_ptr<AccessSet> > GetRead(BlockTreeNode block) const {
    auto iter = query_by_block_read.find(block);
    if (iter != query_by_block_read.end()) {
      return iter->second;
    }
    return StdNodeMap<Tensor, std::shared_ptr<AccessSet> >();
  };

  StdNodeMap<Tensor, std::shared_ptr<AccessSet> > GetWrite(BlockTreeNode block) const {
    auto iter = query_by_block_write.find(block);
    if (iter != query_by_block_write.end()) {
      return iter->second;
    }
    return StdNodeMap<Tensor, std::shared_ptr<AccessSet> >();
  };

  StdNodeMap<BlockTreeNode, std::shared_ptr<AccessSet> > GetReadBy(Tensor tensor) const {
    auto iter = query_by_tensor_read.find(tensor);
    if (iter != query_by_tensor_read.end()) {
      return iter->second;
    }
    return StdNodeMap<BlockTreeNode, std::shared_ptr<AccessSet> >();
  };

  StdNodeMap<BlockTreeNode, std::shared_ptr<AccessSet> > GetWriteBy(Tensor tensor) const {
    auto iter = query_by_tensor_write.find(tensor);
    if (iter != query_by_tensor_write.end()) {
      return iter->second;
    }
    return StdNodeMap<BlockTreeNode, std::shared_ptr<AccessSet> >();
  };

  QueryByTensorMap query_by_tensor_read;
  QueryByTensorMap query_by_tensor_write;
  QueryByBlockMap   query_by_block_read;
  QueryByBlockMap   query_by_block_write;
};

/*!
 * \brief Dependency Graph for elemwise-defined dependency relationship
 */
class DependencyGraph;
class DependencyGraphNode : public Node {
 public:
  StdNodeMap<BlockTreeNode, std::list<Edge> > forward_edges;
  StdNodeMap<BlockTreeNode, std::list<Edge> > backward_edges;

  void VisitAttrs(AttrVisitor* v) final {
  }

  TVM_DLL static DependencyGraph make(ScheduleTreeNode node);

  static constexpr const char* _type_key = "tensorir.DependencyGraph";
  TVM_DECLARE_NODE_TYPE_INFO(DependencyGraphNode, Node);
};

class DependencyGraph : public NodeRef {
 public:
  DependencyGraph() {}
  explicit DependencyGraph(NodePtr<Node> n) : NodeRef(n) {}

  inline const DependencyGraphNode* operator->() const;
  inline DependencyGraphNode* operator->();

  void AddNode(BlockTreeNode op_stmt);
  void AddEdge(BlockTreeNode from, BlockTreeNode to, EdgeType type);
  void InlineNode(BlockTreeNode op_stmt);
  Array<BlockTreeNode> GetSuccessor(BlockTreeNode op_stmt);
  Array<BlockTreeNode> GetPredecessor(BlockTreeNode op_stmt);

  using ContainerType = DependencyGraphNode;
};

const DependencyGraphNode* DependencyGraph::operator->() const {
  return static_cast<const DependencyGraphNode*>(node_.get());;
}

DependencyGraphNode* DependencyGraph::operator->() {
  return static_cast<DependencyGraphNode*>(node_.get());
}

// Debug tool
std::ostream& PrintDependencyGraph(std::ostream &os,  DependencyGraph g);

} // namespace tensorir
} // namespace tvm

#endif // TVM_TENSORIR_DEPENDENCY_GRAPH_H_
