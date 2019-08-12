/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Dependency between blocks
 */

#ifndef TVM_TENSORIR_DEPENDENCY_GRAPH_H_
#define TVM_TENSORIR_DEPENDENCY_GRAPH_H_

#include <tvm/base.h>
#include <tvm/ir.h>
#include <tvm/node/container.h>
#include <tvm/tensor.h>
#include <list>
#include "tree_node.h"

namespace tvm {
namespace tensorir {

// Detailed access information. NOTE: Currently, it is useless.
using AccessSet = int;

// Dependency type
enum EdgeType : int {
  kRAW,
  kWAW,
  kWAR,
  kRAR,
  kUnknown
};

// An edge in the dependency graph
class Edge;
class EdgeNode : public Node {
 public:
  BlockTreeNode dst;
  EdgeType type;

  void VisitAttrs(AttrVisitor* v) final {
  }

  TVM_DLL static Edge make(BlockTreeNode dst, EdgeType type);

  static constexpr const char* _type_key = "tensorir.EdgeNode";
  TVM_DECLARE_NODE_TYPE_INFO(EdgeNode, Node);
};

TVM_DEFINE_MUTABLE_NODE_REF(Edge, NodeRef, EdgeNode);


/*!
 * \brief A bipartite graph to store
 * read/write access information between tensors and blocks
 */
class ReadWriteGraph {
 public:
  // build 4 indexes for query
  using QueryByTensorMap =
    StdNodeMap<Tensor, StdNodeMap<BlockTreeNode, std::shared_ptr<AccessSet> > >;
  using QueryByBlockMap  =
    StdNodeMap<BlockTreeNode, StdNodeMap<Tensor, std::shared_ptr<AccessSet> > >;

  // insert read/write access
  // Block read from a tensor
  void InsertRead(BlockTreeNode block, Tensor tensor) {
    std::shared_ptr<AccessSet> acc_set = query_by_block_read[block][tensor];
    if (acc_set == nullptr) {
      acc_set = std::make_shared<AccessSet>();
      query_by_block_read[block][tensor] = acc_set;
      query_by_tensor_read[tensor][block] = acc_set;
    }
  }
  // Block write to a tensor
  void InsertWrite(BlockTreeNode block, Tensor tensor) {
    std::shared_ptr<AccessSet> acc_set = query_by_block_write[block][tensor];
    if (acc_set == nullptr) {
      acc_set = std::make_shared<AccessSet>();
      query_by_block_write[block][tensor] = acc_set;
      query_by_tensor_write[tensor][block] = acc_set;
    }
  }

  // query
  // Get all tensors read by the block
  StdNodeMap<Tensor, std::shared_ptr<AccessSet> > GetRead(BlockTreeNode block) const {
    auto iter = query_by_block_read.find(block);
    if (iter != query_by_block_read.end()) {
      return iter->second;
    }
    return StdNodeMap<Tensor, std::shared_ptr<AccessSet> >();
  }

  // Get all tensors write by the block
  StdNodeMap<Tensor, std::shared_ptr<AccessSet> > GetWrite(BlockTreeNode block) const {
    auto iter = query_by_block_write.find(block);
    if (iter != query_by_block_write.end()) {
      return iter->second;
    }
    return StdNodeMap<Tensor, std::shared_ptr<AccessSet> >();
  }

  // Get all blocks that read from a tensor
  StdNodeMap<BlockTreeNode, std::shared_ptr<AccessSet> > GetReadBy(Tensor tensor) const {
    auto iter = query_by_tensor_read.find(tensor);
    if (iter != query_by_tensor_read.end()) {
      return iter->second;
    }
    return StdNodeMap<BlockTreeNode, std::shared_ptr<AccessSet> >();
  }

  // Get all blocks that write to a tensor
  StdNodeMap<BlockTreeNode, std::shared_ptr<AccessSet> > GetWriteBy(Tensor tensor) const {
    auto iter = query_by_tensor_write.find(tensor);
    if (iter != query_by_tensor_write.end()) {
      return iter->second;
    }
    return StdNodeMap<BlockTreeNode, std::shared_ptr<AccessSet> >();
  }

  QueryByTensorMap query_by_tensor_read;
  QueryByTensorMap query_by_tensor_write;
  QueryByBlockMap query_by_block_read;
  QueryByBlockMap query_by_block_write;
};

/*!
 * \brief Dependency Graph that stores read/write dependency between BlockTreeNode
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
  // Add a node to the graph
  void AddNode(BlockTreeNode block);
  // Add a dependency edge
  void AddEdge(BlockTreeNode from, BlockTreeNode to, EdgeType type);
  // to support compute_inline which deletes a block in the graph
  void InlineNode(BlockTreeNode op_stmt);
  // Replace dependency
  void CacheReadNode(BlockTreeNode old_block, BlockTreeNode cache_block, Array<BlockTreeNode> relative_blocks);
  void CacheWriteNode(BlockTreeNode old_block, BlockTreeNode cache_block, Array<BlockTreeNode> relative_blocks);
  // Get all blocks that are dependent on block
  Set<BlockTreeNode> GetSuccessor(BlockTreeNode block) const;
  // Get all blocks that this block dependent on
  Set<BlockTreeNode> GetPredecessor(BlockTreeNode block) const;

  TVM_DEFINE_MUTABLE_NODE_REF_METHODS(
      DependencyGraph, NodeRef, DependencyGraphNode);
};

// Debug tool
std::ostream& PrintDependencyGraph(std::ostream &os,  DependencyGraph g);

}  // namespace tensorir
}  // namespace tvm

#endif  // TVM_TENSORIR_DEPENDENCY_GRAPH_H_
