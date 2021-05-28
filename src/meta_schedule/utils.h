/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef SRC_META_SCHEDULE_UTILS_H_
#define SRC_META_SCHEDULE_UTILS_H_

#include <tvm/arith/analyzer.h>
#include <tvm/target/target.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/stmt_functor.h>

#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../arith/pattern_match.h"
#include "../tir/schedule/utils.h"

namespace tvm {
namespace meta_schedule {

/**************** Array Handling ****************/

using tir::AsArray;
using tir::AsOptArray;
using tir::AsVector;

/*!
 * \brief Compute mean of a FloatImm array.
 * Taken from Ansor
 * \param float_array The array of floating point numbers to be averaged
 * \return The mean of the given array
 */
inline double FloatArrayMean(const Array<FloatImm>& float_array) {
  double sum = 0;
  if (float_array.empty()) {
    return 0.0;
  }
  for (const FloatImm& x : float_array) {
    sum += x.get()->value;
  }
  return sum / float_array.size();
}

/*!
 * \brief Concatenate the nested vector into a flattened vector
 * \tparam T The element type of the nested vector
 * \param source The nested vector
 * \return The flattened vector
 */
template <class T>
inline std::vector<T> ConcatArray(const std::vector<std::vector<T> >& source) {
  std::vector<T> result;
  for (const std::vector<T>& item : source) {
    result.insert(result.end(), item.begin(), item.end());
  }
  return result;
}

/*!
 * \brief Concatenate the nested vector into a flattened vector
 * \tparam T The element type of the nested vector
 * \param source The nested vector
 * \return The flattened vector
 */
template <class T>
inline Array<T> ConcatArray(const std::vector<Array<T> >& source) {
  Array<T> result;
  for (const Array<T>& item : source) {
    result.insert(result.end(), item.begin(), item.end());
  }
  return result;
}

/**************** Expression Parsing ****************/

/*!
 * \brief Checks if the specific expr is an integer constant
 * \param x The expr to be checked
 * \return A boolean flag indicating if it is a constant integer, or broadcast of constant integer
 */
inline bool IsConstInt(const PrimExpr& x) {
  if (x->IsInstance<tir::IntImmNode>()) {
    return true;
  }
  if (const auto* op = x.as<tir::BroadcastNode>()) {
    return op->value->IsInstance<tir::IntImmNode>();
  }
  return false;
}

/*!
 * \brief Check if an expression consists of a single variable, or a variable +/i an constant
 * \param expr The expression to be checked
 * \return result Output, the var inside if it satisfies the condition; otherwise NullOpt
 */
inline Optional<tir::Var> IsVarPlusMinusConst(const PrimExpr& expr) {
  // match: "var"
  if (const auto* var = expr.as<tir::VarNode>()) {
    return GetRef<tir::Var>(var);
  }
  arith::PVar<tir::Var> var;
  arith::PVar<IntImm> shift;
  // match: "var +/- shift"
  if ((var + shift).Match(expr) || (var - shift).Match(expr) || (shift + var).Match(expr)) {
    return var.Eval();
  }
  return NullOpt;
}

/**************** TIR Misc ****************/

/*!
 * \brief Compare two domains and check if they are equal
 * \param lhs One domain
 * \param rhs The other domain
 * \return A boolean indicating if the two domains are proved to be equal
 */
inline bool DomainEqual(const Array<Range>& lhs, const Array<Range>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  arith::Analyzer analyzer;
  int n = lhs.size();
  for (int i = 0; i < n; ++i) {
    const Range& l = lhs[i];
    const Range& r = rhs[i];
    if (!analyzer.CanProve(l->min == r->min)) {
      return false;
    }
    if (!analyzer.CanProve(l->extent == r->extent)) {
      return false;
    }
  }
  return true;
}

template <class FPredicate>
inline Optional<tir::StmtSRef> FindBlockSRef(const tir::ScheduleState& sch, FPredicate predicate) {
  Optional<tir::StmtSRef> result = NullOpt;
  auto f_visit = [&sch, &result, &predicate](const ObjectRef& obj) -> bool {
    if (result.defined()) {
      return false;
    }
    if (const auto* block = obj.as<tir::BlockNode>()) {
      if (predicate(block)) {
        result = sch->stmt2ref.at(block);
        return false;
      }
    }
    return true;
  };
  for (const auto& kv : sch->mod->functions) {
    const BaseFunc& base_func = kv.second;
    if (const auto* func = base_func.as<tir::PrimFuncNode>()) {
      tir::PreOrderVisit(func->body, f_visit);
    }
  }
  return result;
}

/**************** TIR Annotation ****************/

inline bool HasBinding(const tir::StmtSRef& loop_sref, const String& thread_tag) {
  const auto* loop = TVM_SREF_TO_FOR(loop, loop_sref);
  if (!loop->thread_binding.defined()) {
    return false;
  }
  tir::IterVar binding = loop->thread_binding.value();
  if (binding->iter_type != tir::IterVarType::kThreadIndex) {
    return false;
  }
  return binding->thread_tag == thread_tag;
}

inline Optional<String> GetBinding(const tir::StmtSRef& loop_sref) {
  const auto* loop = TVM_SREF_TO_FOR(loop, loop_sref);
  if (!loop->thread_binding.defined()) {
    return NullOpt;
  }
  tir::IterVar binding = loop->thread_binding.value();
  if (loop->kind == tir::ForKind::kParallel) {
    return String("parallel");
  } else if (loop->kind == tir::ForKind::kVectorized) {
    return String("vectorized");
  } else if (loop->kind == tir::ForKind::kUnrolled) {
    return String("unrolled");
  }
  if (binding->iter_type != tir::IterVarType::kThreadIndex) {
    return NullOpt;
  }
  return binding->thread_tag;
}

inline Optional<String> GetAnn(const tir::StmtSRef& sref, const String& ann_key) {
  const Map<String, ObjectRef>* annotations = nullptr;
  if (const auto* loop = sref->StmtAs<tir::ForNode>()) {
    annotations = &loop->annotations;
  } else if (const auto* block = sref->StmtAs<tir::BlockNode>()) {
    annotations = &block->annotations;
  } else {
    LOG(FATAL) << "TypeError: Unknown type of sref: " << sref->stmt->GetTypeKey();
  }
  for (const auto& ann : *annotations) {
    if (ann.first == ann_key) {
      if (const auto* str_imm = ann.second.as<tir::StringImmNode>()) {
        return str_imm->value;
      }
    }
  }
  return NullOpt;
}

inline bool HasAnn(const tir::StmtSRef& sref, const String& ann_key, const String& ann_val) {
  Optional<String> result = GetAnn(sref, ann_key);
  return result.defined() && result.value() == ann_val;
}

inline bool HasAnyAnn(const tir::StmtSRef& sref) {
  if (const auto* loop = sref->StmtAs<tir::ForNode>()) {
    return !loop->annotations.empty();
  } else if (const auto* block = sref->StmtAs<tir::BlockNode>()) {
    return !block->annotations.empty();
  }
  LOG(FATAL) << "TypeError: Unknown type of sref: " << sref->stmt->GetTypeKey();
  throw;
}

inline void DelAnn(const tir::ScheduleState& sch, const tir::StmtSRef& sref,
                   const String& ann_key) {
  // Extract annotation
  const Map<String, ObjectRef>* annotations = nullptr;
  if (const auto* loop = sref->StmtAs<tir::ForNode>()) {
    annotations = &loop->annotations;
  } else if (const auto* block = sref->StmtAs<tir::BlockNode>()) {
    annotations = &block->annotations;
  } else {
    LOG(FATAL) << "TypeError: Unknown type of sref: " << sref->stmt->GetTypeKey();
  }
  // Remove the annotation
  ICHECK(annotations->find(ann_key) != annotations->end())
      << "IndexError: Cannot find annotation key: " << ann_key;
  Map<String, ObjectRef> new_ann(*annotations);
  new_ann.erase(ann_key);

  // Create the new stmt
  if (const auto* loop = sref->StmtAs<tir::ForNode>()) {
    ObjectPtr<tir::ForNode> n = make_object<tir::ForNode>(*loop);
    n->annotations = std::move(new_ann);
    sch->Replace(sref, tir::For(n), {});
  } else if (const auto* block = sref->StmtAs<tir::BlockNode>()) {
    ObjectPtr<tir::BlockNode> n = make_object<tir::BlockNode>(*block);
    n->annotations = std::move(new_ann);
    tir::Block p(n);
    sch->Replace(sref, p, {{GetRef<tir::Block>(block), p}});
  } else {
    LOG(FATAL) << "TypeError: Unknown type of sref: " << sref->stmt->GetTypeKey();
    throw;
  }
}

inline void AddAnn(const tir::ScheduleState& sch, const tir::StmtSRef& sref, const String& ann_key,
                   const PrimExpr& ann_val) {
  // Extract annotation
  const Map<String, ObjectRef>* annotations = nullptr;
  if (const auto* loop = sref->StmtAs<tir::ForNode>()) {
    annotations = &loop->annotations;
  } else if (const auto* block = sref->StmtAs<tir::BlockNode>()) {
    annotations = &block->annotations;
  } else {
    LOG(FATAL) << "TypeError: Unknown type of sref: " << sref->stmt->GetTypeKey();
  }
  // Check if the annotation already exists
  if (annotations->find(ann_key) != annotations->end()) {
    return;
  }
  // Add the new annotation
  Map<String, ObjectRef> new_ann(*annotations);
  new_ann.Set(ann_key, ann_val);
  // Create the new stmt
  if (const auto* loop = sref->StmtAs<tir::ForNode>()) {
    ObjectPtr<tir::ForNode> n = make_object<tir::ForNode>(*loop);
    n->annotations = std::move(new_ann);
    sch->Replace(sref, tir::For(n), {});
  } else if (const auto* block = sref->StmtAs<tir::BlockNode>()) {
    ObjectPtr<tir::BlockNode> n = make_object<tir::BlockNode>(*block);
    n->annotations = std::move(new_ann);
    tir::Block p(n);
    sch->Replace(sref, p, {{GetRef<tir::Block>(block), p}});
  } else {
    LOG(FATAL) << "TypeError: Unknown type of sref: " << sref->stmt->GetTypeKey();
    throw;
  }
}

/**************** String Manipulation ****************/

using tir::FindCharPos;
using tir::StartsWith;

/**************** Target Hardware Concurrency ****************/

inline int GetTargetNumCores(const Target& target, std::atomic<int>* warned_num_cores_missing) {
  int num_cores = target->GetAttr<Integer>("num_cores").value_or(-1);
  if (num_cores == -1) {
    static const auto* f_cpu_count = runtime::Registry::Get("meta_schedule._cpu_count");
    ICHECK(f_cpu_count)
        << "ValueError: Cannot find the packed function \"meta_schedule._cpu_count\"";
    num_cores = (*f_cpu_count)(false);
    if (warned_num_cores_missing != nullptr && warned_num_cores_missing->fetch_add(1) == 0) {
      LOG(WARNING) << "Warning: Target does not have attribute \"num_cores\", falling back the "
                      "number of CPU cores on the local machine. The inaccuracy in number of "
                      "cores may lead to dramatically inferior performance. Falling back to "
                      "assuming "
                   << num_cores << " CPU core(s)";
    }
  }
  return num_cores;
}

/**************** Module-related ****************/

inline tir::PrimFunc GetOnlyFunc(const IRModule& mod) {
  const Map<GlobalVar, BaseFunc>& funcs = mod->functions;
  CHECK_EQ(funcs.size(), 1);
  for (const auto& kv : funcs) {
    const BaseFunc& base_func = kv.second;
    if (const auto* prim_func = base_func.as<tir::PrimFuncNode>()) {
      return GetRef<tir::PrimFunc>(prim_func);
    }
  }
  throw;
}

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_UTILS_H_
