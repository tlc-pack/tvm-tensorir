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
#include <tvm/tir/schedule.h>
#include <tvm/tir/stmt_functor.h>

#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../arith/pattern_match.h"

namespace tvm {
namespace meta_schedule {

/**************** Array Handling ****************/

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

inline String Repr(const tir::PrimFunc& func) {
  static const auto* f = runtime::Registry::Get("script.AsTVMScript");
  CHECK(f) << "IndexError: global function \"script.AsTVMScript\" not found";
  return (*f)(func, false).operator String();
}

inline PrimExpr GetLoopExtent(const tir::StmtSRef& loop_sref) {
  const auto* loop = loop_sref->GetStmt<tir::LoopNode>();
  CHECK(loop) << "TypeError: Expects LoopNode, but gets: " << loop_sref->stmt->GetTypeKey();
  return loop->extent;
}

inline Optional<Integer> GetLoopIntExtent(const tir::LoopNode* loop) {
  const auto* int_extent = loop->extent.as<IntImmNode>();
  return int_extent ? Integer(int_extent->value) : Optional<Integer>(NullOpt);
}

inline Optional<Integer> GetLoopIntExtent(const tir::StmtSRef& loop_sref) {
  PrimExpr extent = GetLoopExtent(loop_sref);
  const auto* int_extent = extent.as<IntImmNode>();
  return int_extent ? Integer(int_extent->value) : Optional<Integer>(NullOpt);
}

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
inline Optional<tir::StmtSRef> FindBlockSRef(const tir::Schedule& sch, FPredicate predicate) {
  Optional<tir::StmtSRef> result = NullOpt;
  tir::PreOrderVisit(sch->func->body, [&sch, &result, &predicate](const ObjectRef& obj) -> bool {
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
  });
  return result;
}

/**************** TIR Annotation ****************/

inline Optional<String> GetAnn(const tir::StmtSRef& sref, const String& ann_key) {
  const Array<tir::Annotation>* annotations = nullptr;
  if (const auto* loop = sref->GetStmt<tir::LoopNode>()) {
    annotations = &loop->annotations;
  } else if (const auto* block = sref->GetStmt<tir::BlockNode>()) {
    annotations = &block->annotations;
  } else {
    LOG(FATAL) << "TypeError: Unknown type of sref: " << sref->stmt->GetTypeKey();
  }
  for (const tir::Annotation& ann : *annotations) {
    if (ann->attr_key == ann_key) {
      if (const auto* str_imm = ann->value.as<tir::StringImmNode>()) {
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
  if (const auto* loop = sref->GetStmt<tir::LoopNode>()) {
    return !loop->annotations.empty();
  } else if (const auto* block = sref->GetStmt<tir::BlockNode>()) {
    return !block->annotations.empty();
  }
  LOG(FATAL) << "TypeError: Unknown type of sref: " << sref->stmt->GetTypeKey();
  throw;
}

inline void DelAnn(const tir::Schedule& sch, const tir::StmtSRef& sref, const String& ann_key) {
  // Extract annotation
  const Array<tir::Annotation>* annotations = nullptr;
  if (const auto* loop = sref->GetStmt<tir::LoopNode>()) {
    annotations = &loop->annotations;
  } else if (const auto* block = sref->GetStmt<tir::BlockNode>()) {
    annotations = &block->annotations;
  } else {
    LOG(FATAL) << "TypeError: Unknown type of sref: " << sref->stmt->GetTypeKey();
  }
  // Remove the annotation
  Array<tir::Annotation> new_ann;
  int n = annotations->size();
  new_ann.reserve(n - 1);
  for (int i = 0; i < n; ++i) {
    const tir::Annotation& ann = annotations->operator[](i);
    if (ann->attr_key != ann_key) {
      new_ann.push_back(ann);
    }
  }
  CHECK_NE(annotations->size(), new_ann.size())
      << "IndexError: Cannot find annotation key: " << ann_key;
  // Create the new stmt
  if (const auto* loop = sref->GetStmt<tir::LoopNode>()) {
    ObjectPtr<tir::LoopNode> n = make_object<tir::LoopNode>(*loop);
    n->annotations = std::move(new_ann);
    sch->Replace(sref, tir::Loop(n), {});
  } else if (const auto* block = sref->GetStmt<tir::BlockNode>()) {
    ObjectPtr<tir::BlockNode> n = make_object<tir::BlockNode>(*block);
    n->annotations = std::move(new_ann);
    tir::Block p(n);
    sch->Replace(sref, p, {{p, GetRef<tir::Block>(block)}});
  } else {
    LOG(FATAL) << "TypeError: Unknown type of sref: " << sref->stmt->GetTypeKey();
    throw;
  }
}

inline void AddAnn(const tir::Schedule& sch, const tir::StmtSRef& sref, const String& ann_key,
                   const PrimExpr& ann_val) {
  // Extract annotation
  const Array<tir::Annotation>* annotations = nullptr;
  if (const auto* loop = sref->GetStmt<tir::LoopNode>()) {
    annotations = &loop->annotations;
  } else if (const auto* block = sref->GetStmt<tir::BlockNode>()) {
    annotations = &block->annotations;
  } else {
    LOG(FATAL) << "TypeError: Unknown type of sref: " << sref->stmt->GetTypeKey();
  }
  // Check if the annotation already exists
  for (const tir::Annotation& ann : *annotations) {
    if (ann->attr_key == ann_key) {
      return;
    }
  }
  // Add the new annotation
  Array<tir::Annotation> new_ann(*annotations);
  new_ann.push_back(tir::Annotation(ann_key, ann_val));
  // Create the new stmt
  if (const auto* loop = sref->GetStmt<tir::LoopNode>()) {
    ObjectPtr<tir::LoopNode> n = make_object<tir::LoopNode>(*loop);
    n->annotations = std::move(new_ann);
    sch->Replace(sref, tir::Loop(n), {});
  } else if (const auto* block = sref->GetStmt<tir::BlockNode>()) {
    ObjectPtr<tir::BlockNode> n = make_object<tir::BlockNode>(*block);
    n->annotations = std::move(new_ann);
    tir::Block p(n);
    sch->Replace(sref, p, {{p, GetRef<tir::Block>(block)}});
  } else {
    LOG(FATAL) << "TypeError: Unknown type of sref: " << sref->stmt->GetTypeKey();
    throw;
  }
}

/**************** AsArray<TSrc, TDst> ****************/

namespace details {
template <class TSrc, class TDst>
struct AsArrayImpl {};

template <class TSrc>
struct AsArrayImpl<TSrc, TSrc> {
  inline Array<TSrc> operator()(const std::vector<TSrc>& vec) const {
    return Array<TSrc>(vec.begin(), vec.end());
  }
};

template <class TDstObjectRef>
struct AsArrayImpl<int, TDstObjectRef> {
  inline Array<TDstObjectRef> operator()(const std::vector<int>& vec) const {
    Array<TDstObjectRef> result;
    result.reserve(vec.size());
    for (int x : vec) {
      result.push_back(Integer(x));
    }
    return result;
  }
};

template <class TDstObjectRef>
struct AsArrayImpl<double, TDstObjectRef> {
  inline Array<TDstObjectRef> operator()(const std::vector<double>& vec) const {
    Array<TDstObjectRef> result;
    result.reserve(vec.size());
    for (double x : vec) {
      result.push_back(FloatImm(tvm::DataType::Float(64), x));
    }
    return result;
  }
};
}  // namespace details

template <class TSrc, class TDst>
inline Array<TDst> AsArray(const std::vector<TSrc>& vec) {
  return details::AsArrayImpl<TSrc, TDst>()(vec);
}

/**************** AsVector<TSrc, TDst> ****************/

namespace details {

template <class TSrc, class TDst>
struct AsVectorImpl {};

template <class TSrc>
struct AsVectorImpl<TSrc, TSrc> {
  inline std::vector<TSrc> operator()(const Array<TSrc>& vec) const {
    return std::vector<TSrc>(vec.begin(), vec.end());
  }
};

template <class TSrcObjectRef>
struct AsVectorImpl<TSrcObjectRef, int> {
  inline std::vector<int> operator()(const Array<TSrcObjectRef>& vec) const {
    std::vector<int> results;
    for (const TSrcObjectRef& x : vec) {
      const auto* n = x.template as<IntImmNode>();
      CHECK(n) << "TypeError: Expects IntImm, but gets: " << x->GetTypeKey();
      results.push_back(n->value);
    }
    return results;
  }
};

template <class TSrcObjectRef>
struct AsVectorImpl<TSrcObjectRef, int64_t> {
  inline std::vector<int64_t> operator()(const Array<TSrcObjectRef>& vec) const {
    std::vector<int64_t> results;
    for (const TSrcObjectRef& x : vec) {
      const auto* n = x.template as<IntImmNode>();
      CHECK(n) << "TypeError: Expects IntImm, but gets: " << x->GetTypeKey();
      results.push_back(n->value);
    }
    return results;
  }
};

template <class TSrcObjectRef>
struct AsVectorImpl<TSrcObjectRef, double> {
  inline std::vector<double> operator()(const Array<TSrcObjectRef>& array) const {
    std::vector<double> results;
    for (const TSrcObjectRef& x : array) {
      const auto* n = x.template as<FloatImmNode>();
      CHECK(n) << "TypeError: Expects FloatImm, but gets: " << x->GetTypeKey();
      results.push_back(n->value);
    }
    return results;
  }
};
}  // namespace details

template <class TSrc, class TDst>
inline std::vector<TDst> AsVector(const Array<TSrc>& vec) {
  return details::AsVectorImpl<TSrc, TDst>()(vec);
}

/**************** I/O ****************/

/*!
 * \brief An empty output stream
 * Taken from Ansor
 */
class NullStream : public std::ostream {
 public:
  NullStream() : std::ostream(nullptr) {}
  NullStream(const NullStream&) : std::ostream(nullptr) {}
  static NullStream& Global();
};

template <class T>
NullStream& operator<<(NullStream& os, const T& value) {
  return os;
}

/*!
 * \brief Get std cout with verbose control
 * Taken from Ansor
 */
inline std::ostream& StdCout(int verbose, int setting = 1) {
  return verbose >= setting ? std::cout : NullStream::Global();
}

/**************** String Manipulation ****************/

/*!
 * \brief Find all positions that the specific char occurs in the string
 * \param str The string to be examined
 * \param c The specific char
 * \return A list of integers indicating the occurrence position
 */
inline std::vector<int> FindCharPos(const String& str, char c) {
  std::vector<int> result;
  const char* data = str.data();
  int n = str.length();
  for (int i = 0; i < n; ++i) {
    if (data[i] == c) {
      result.push_back(i);
    }
  }
  return result;
}

/**************** Target Hardware Concurrency ****************/

inline int GetTargetNumCores(const Target& target, std::atomic<int>* warned_num_cores_missing) {
  int num_cores = target->GetAttr<Integer>("num_cores").value_or(-1);
  if (num_cores == -1) {
    static const auto* f_cpu_count = runtime::Registry::Get("meta_schedule._cpu_count");
    CHECK(f_cpu_count)
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

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_UTILS_H_
