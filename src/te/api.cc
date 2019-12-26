/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief TE API registration
 */

#include <tvm/te/transform.h>
#include <tvm/api_registry.h>
#include <tvm/te/schedule.h>

namespace tvm {
namespace te {

TVM_REGISTER_API("ir_pass.TeLower")
.set_body_typed(TeLower);

// schedule
TVM_REGISTER_API("te.schedule.CreateSchedule")
.set_body_typed(Schedule::Create);

TVM_REGISTER_API("te.schedule.Replace")
.set_body_method(&Schedule::Replace);

TVM_REGISTER_API("te.schedule.GetStmtSRef")
.set_body_typed<StmtSRef(Schedule, Stmt)>(
    [](Schedule schedule, Stmt stmt) {
      return schedule->stmt2ref.at(stmt.operator->());
    });

TVM_REGISTER_API("te.schedule.GetStmt")
.set_body_typed<Stmt(StmtSRef)>(
    [](StmtSRef sref) {
      return GetRef<Stmt>(sref->node);
    });

TVM_REGISTER_API("te.schedule.GetBlocksFromTag")
.set_body_typed<Array<StmtSRef>(Schedule, StmtSRef, std::string)>(
    [](Schedule schedule, StmtSRef scope, std::string tag) {
      return schedule.GetBlock(scope, tag);
    });

TVM_REGISTER_API("te.schedule.GetBlocksFromBuffer")
.set_body_typed<Array<StmtSRef>(Schedule, StmtSRef, Buffer)>(
    [](Schedule schedule, StmtSRef scope, Buffer buffer) {
      return schedule.GetBlock(scope, buffer);
    });

// dependency graph
TVM_REGISTER_API("te.schedule.GetSuccessor")
.set_body_typed<Array<StmtSRef>(Schedule, StmtSRef, StmtSRef)>(
    [](Schedule schedule, StmtSRef scope, StmtSRef block) {
      return schedule->dependency_graphs_[scope].GetSuccessor(block);
    });

TVM_REGISTER_API("te.schedule.GetPredecessor")
.set_body_typed<Array<StmtSRef>(Schedule, StmtSRef, StmtSRef)>(
    [](Schedule schedule, StmtSRef scope, StmtSRef block) {
      return schedule->dependency_graphs_[scope].GetPredecessor(block);
    });

// maker
TVM_REGISTER_API("make.TensorRegion")
.set_body_typed<TensorRegion(Buffer, Array<Range>)>(
    [](Buffer buffer, Array<Range> region) {
      return TensorRegion(buffer, region);
    });

TVM_REGISTER_API("make.BufferAllocate")
.set_body_typed<BufferAllocate(Buffer, std::string)>(
    [](Buffer buffer, std::string scope) {
      return BufferAllocate(buffer, scope);
    });

TVM_REGISTER_API("make.BufferLoad")
.set_body_typed<BufferLoad(DataType, Buffer, Array<Expr>)>(
    [](DataType type, Buffer buffer, Array<Expr> indices) {
      return BufferLoad(type, buffer, indices);
    });

TVM_REGISTER_API("make.BufferStore")
.set_body_typed<BufferStore(Buffer, Expr, Array<Expr>)>(
    [](Buffer buffer, Expr value, Array<Expr> indices) {
      return BufferStore(buffer, value, indices);
    });

TVM_REGISTER_API("make.Loop")
.set_body_typed<Loop(Var, Expr, Expr, Array<Annotation>, Stmt)>(
    [](Var loop_var, Expr min, Expr extent,
       Array<Annotation> annotations, Stmt body) {
      return Loop(loop_var, min, extent, annotations, body);
    });

TVM_REGISTER_API("make.TeBlock")
.set_body_typed<Block(Array<IterVar>,
                      Array<Expr>,
                      Array<TensorRegion>,
                      Array<TensorRegion>,
                      Stmt, Expr,
                      Array<BufferAllocate>,
                      Array<Annotation>,
                      std::string)>(
    [](Array<IterVar> iter_vars,
       Array<Expr> values,
       Array<TensorRegion> reads,
       Array<TensorRegion> writes,
       Stmt body,
       Expr predicate,
       Array<BufferAllocate> allocates,
       Array<Annotation> annotations,
       std::string tag) {
      if (!predicate.type().is_bool()) {
        // To support python ir_builder
        CHECK(is_one(predicate));
        predicate = UIntImm::make(Bool(), 1);
      }
      return Block(iter_vars, values, reads, writes,
                   body, predicate, allocates, annotations, tag);
    });

TVM_REGISTER_API("make.TeFunction")
.set_body_typed<Function(Array<Var>, Map<Var, Buffer>,
                         std::string, Stmt)>(
    [](Array<Var> params, Map<Var, Buffer> buffer_map,
       std::string name, Stmt body) {
      return Function(params, buffer_map, name, body);
    });

}  // namespace te
}  // namespace tvm
