/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Build Schedule Tree from Halide IR
 */

#ifndef TVM_TREE_BUILDER_H_
#define TVM_TREE_BUILDER_H_

#include <tvm/ir_functor_ext.h>
#include <vector>
#include "tree_node.h"
#include "schedule.h"

namespace tvm {
namespace tensorir {

// Build schedule tree from Halide IR
class TreeBuilder : public StmtFunctor<ScheduleTreeNode(const Stmt&)> {
 public:
  Schedule Build(Stmt stmt);

  // statement
  //ScheduleTreeNode VisitStmt_(const LetStmt* op) override;
  //ScheduleTreeNode VisitStmt_(const AttrStmt* op) override;
  //ScheduleTreeNode VisitStmt_(const IfThenElse* op) override;
  ScheduleTreeNode VisitStmt_(const For* op) override;
  ScheduleTreeNode VisitStmt_(const Allocate* op) override;
  //ScheduleTreeNode VisitStmt_(const Store* op) override;
  //ScheduleTreeNode VisitStmt_(const Free* op) override;
  ScheduleTreeNode VisitStmt_(const AssertStmt* op) override;
  //ScheduleTreeNode VisitStmt_(const ProducerConsumer* op) override;
  ScheduleTreeNode VisitStmt_(const Provide* op) override;
  //ScheduleTreeNode VisitStmt_(const Realize* op) override;
  //ScheduleTreeNode VisitStmt_(const Prefetch* op) override;
  //ScheduleTreeNode VisitStmt_(const Block* op) override;
  //ScheduleTreeNode VisitStmt_(const Evaluate* op) override;

 private:
  StdNodeMap<Var, Range> dom_map_;      // The gathered domain information for all iteration vars
  StdNodeMap<Var, size_t> var_order_;
  int var_ct_{0};

  Array<BlockTreeNode> block_list_;
};

// Create input regions for an expression or statement
Array<TensorRegion> CreateInputRegions(const NodeRef& expr_or_stmt);


} // namespace tensorir
} // namespace tvm

#endif // TVM_TENSORIR_UTIL_H_
