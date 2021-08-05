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

/*!
 * \file auto_scheduler/utils.h
 * \brief Common utilities.
 */

#ifndef TVM_AUTO_SCHEDULER_UTILS_H_
#define TVM_AUTO_SCHEDULER_UTILS_H_

#include <dmlc/common.h>
#include <tvm/tir/expr.h>

#include <algorithm>
#include <deque>
#include <exception>
#include <future>
#include <iomanip>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

// <bojian/DietCode>
#include <tvm/auto_scheduler/loop_state.h>
#include <tvm/auto_scheduler/search_task.h>
#include <tvm/auto_scheduler/transform_step.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/te/operation.h>
#include <tvm/tir/dynamic_axis_functor.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>

#include <sstream>


namespace std {

/*! \brief Hash function for std::pair */
template <typename T1, typename T2>
struct hash<std::pair<T1, T2>> {
  std::size_t operator()(const std::pair<T1, T2>& k) const {
    return ::dmlc::HashCombine(std::hash<T1>()(k.first), std::hash<T2>()(k.second));
  }
};

/*! \brief Hash function for std::tuple */
template <typename T1, typename T2, typename T3>
struct hash<std::tuple<T1, T2, T3>> {
  std::size_t operator()(const std::tuple<T1, T2, T3>& k) const {
    return ::dmlc::HashCombine(
        ::dmlc::HashCombine(std::hash<T1>()(std::get<0>(k)), std::hash<T2>()(std::get<1>(k))),
        std::hash<T3>()(std::get<2>(k)));
  }
};

}  // namespace std

namespace tvm {
namespace auto_scheduler {

/********** Utilities for Array, std::vector, std::string **********/
/*! \brief Get the first appearance index of elements in an Array */
template <typename T>
inline void GetIndices(const Array<T>& array, const Array<T>& to_locate, Array<Integer>* indices) {
  for (const auto& v : to_locate) {
    auto it = std::find(array.begin(), array.end(), v);
    if (it != array.end()) {
      indices->push_back(it - array.begin());
    } else {
      LOG(FATAL) << "Cannot find the item";
    }
  }
}

/*! \brief Get the first appearance index of an element in an Array */
template <typename T>
inline int GetIndex(const Array<T>& array, const T& to_locate) {
  for (size_t i = 0; i < array.size(); ++i) {
    if (array[i] == to_locate) {
      return i;
    }
  }
  LOG(FATAL) << "Cannot find the item";
  return -1;
}

/*! \brief Delete the item in a std::vector if it exists. */
template <typename T>
inline void FindAndDeleteItem(std::vector<T>* array, const T& to_delete) {
  auto iter = std::find(array->begin(), array->end(), to_delete);
  if (iter != array->end()) {
    array->erase(iter);
  }
}

/*! \brief Compute the product of all elements in a vector */
inline int64_t ElementProduct(const std::vector<int>& array) {
  int64_t ret = 1;
  for (auto x : array) {
    ret *= x;
  }
  return ret;
}

/*! \brief Move elements from multiple vectors to one vector */
template <typename T>
std::vector<T>& ConcatenateMove(std::vector<T>* out, std::vector<T>* in) {
  out->insert(out->end(), std::make_move_iterator(in->begin()), std::make_move_iterator(in->end()));
  return *out;
}

/*! \brief Move elements from multiple vectors to one vector */
template <typename T, typename... Args>
std::vector<T>& ConcatenateMove(std::vector<T>* out, std::vector<T>* first, Args... args) {
  ConcatenateMove(out, first);
  ConcatenateMove(out, args...);
  return *out;
}

/*! \brief Get a random permutation of integers [0, n-1] */
template <typename G>
void RandomPermutation(int n, std::vector<int>* out, G* gen) {
  out->assign(n, 0);
  std::iota(out->begin(), out->end(), 0);
  std::shuffle(out->begin(), out->end(), *gen);
}

/*! \brief Replace a sub-string to another sub-string in a string */
inline void StrReplace(std::string* base, const std::string& from, const std::string& to) {
  auto pos = base->find(from);
  while (pos != std::string::npos) {
    base->replace(pos, from.size(), to);
    pos = base->find(from, pos + to.size());
  }
}

/*! \brief Return whether two int arrays are elementwise-equal */
inline bool IntArrayEqual(const Array<PrimExpr>& arr1, const Array<PrimExpr>& arr2) {
  if (arr1.size() != arr2.size()) {
    return false;
  }


  // <bojian/DietCode>
  std::ostringstream strout;

  for (size_t i = 0; i < arr1.size(); ++i) {
    auto int1 = arr1[i].as<IntImmNode>();
    auto int2 = arr2[i].as<IntImmNode>();

    // <bojian/DietCode>
    // ICHECK(int1 != nullptr);
    // ICHECK(int2 != nullptr);
    if (int1 == nullptr || int2 == nullptr) {
      strout << arr1[i];
      std::string arr1_idx = strout.str();
      strout.str("");
      strout.clear();
      strout << arr2[i];
      std::string arr2_idx = strout.str();
      strout.str("");
      strout.clear();
      if (arr1_idx != arr2_idx) {
        LOG(INFO) << arr1_idx << " != " << arr2_idx;
        return false;
      } else {
        continue;
      }
    }


    if (int1->value != int2->value) {
      return false;
    }
  }
  return true;
}

/********** Utilities for TVM Containers / ByteArray **********/
/*! \brief Compute mean of a FloatImm array */
inline double FloatArrayMean(const Array<PrimExpr>& float_array) {
  double sum = 0;
  if (float_array.empty()) {
    return 0.0;
  }

  for (const auto& x : float_array) {
    auto floatimm = x.as<tir::FloatImmNode>();
    ICHECK(floatimm != nullptr);
    sum += floatimm->value;
  }
  return sum / float_array.size();
}

/*! \brief Return whether a string starts with another substring */
inline bool StrStartsWith(const String& a, const String& b) {
  if (b.size() > a.size()) return false;
  return std::equal(a.c_str(), a.c_str() + b.size(), b.c_str());
}

/*! \brief Return whether a string ends with another substring */
inline bool StrEndsWith(const String& a, const String& b) {
  if (b.size() > a.size()) return false;
  return std::equal(a.c_str() + a.size() - b.size(), a.c_str() + a.size(), b.c_str());
}

/********** Other Utilities **********/
/*! \brief Get an int value from an Expr */
inline int64_t GetIntImm(const PrimExpr& expr) {
  auto pint = expr.as<IntImmNode>();
  ICHECK(pint != nullptr) << "Expect an IntImm but get " << expr;
  return pint->value;
}

/*!
 * \brief Clean the name of an iterator or an op to make it valid in python code.
 * \param str The original name.
 * \param prefix The name prefix to differentiate the same name (e.g., the same iterator names).
 * \return The cleaned name.
 */
inline std::string CleanName(const std::string& str, const std::string& prefix = "") {
  std::string ret = str;
  StrReplace(&ret, ".", "_");
  StrReplace(&ret, "@", "_");
  StrReplace(&ret, "outer", "o");
  StrReplace(&ret, "inner", "i");
  if (prefix != "") {
    return prefix + "_" + ret;
  }
  return ret;
}

/*! \brief An empty output stream */
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

/*! \brief Get std cout with verbose control */
inline std::ostream& StdCout(int verbose, int setting = 1) {
  return verbose >= setting ? std::cout : NullStream::Global();
}

/*! \brief Print multiple chars */
inline std::string Chars(const char& str, int times) {
  std::stringstream ret;
  for (int i = 0; i < times; ++i) {
    ret << str;
  }
  return ret.str();
}

/*! \brief Print the time elapsed */
inline void PrintTimeElapsed(std::chrono::time_point<std::chrono::high_resolution_clock> t_begin,
                             const std::string& info, int verbose) {
  double duration = std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::high_resolution_clock::now() - t_begin)
                        .count();
  StdCout(verbose) << "Time elapsed for " << info << ": " << std::fixed << std::setprecision(2)
                   << duration << " s" << std::endl;
}

/*!
 * \brief Parse shape and axis names from layout string
 */
inline void ParseKernelLayout(const String& layout, Array<PrimExpr>* shape,
                              std::vector<std::string>* axes) {
  int32_t factor = 0;
  std::string axis = "";
  for (char c : std::string(layout)) {
    if (c >= 'A' && c <= 'z') {
      axis += c;
      if (factor != 0) {
        shape->push_back(factor);
        factor = 0;
      }
    } else if (c >= '0' && c <= '9') {
      factor = factor * 10 + c - '0';
      if (!axis.empty()) {
        axes->push_back(axis);
        axis = "";
      }
    } else {
      LOG(FATAL) << "Invalid layout " << layout;
    }
  }
  if (!axis.empty()) {
    axes->push_back(axis);
  }
}

/*! \brief Get the base name before '_' of an axis */
inline std::string AxisBaseName(const std::string& str) { return str.substr(0, str.rfind("_")); }


// <bojian/DietCode>
template<typename T>
inline std::string ArrayToString(const std::vector<T>& Arr) {
  std::ostringstream strout;
  strout << "[";
  for (const T& a : Arr) {
    strout << a << ", ";
  }
  strout << "]";
  return strout.str();
}

template<typename T>
inline std::string ArrayToString(const Array<T>& Arr) {
  std::ostringstream strout;
  strout << "[";
  for (const T& a : Arr) {
    strout << a << ", ";
  }
  strout << "]";
  return strout.str();
}

template<typename T>
inline std::string OptionalArrayToString(const Array<Optional<T>>& Arr) {
  std::ostringstream strout;
  strout << "[";
  for (const Optional<T>& a : Arr) {
    if (a == nullptr) {
      strout << "NULL, ";
    } else {
      strout << a.value() << ", ";
    }
  }
  strout << "]";
  return strout.str();
}

template<typename T>
inline std::string MatrixToString(const Array<Array<T>>& Mat,
                                  const bool flatten = false) {
  std::ostringstream strout;
  strout << "[";
  if (!flatten) {
    strout << std::endl;
  }
  for (const Array<T>& Arr : Mat) {
    if (!flatten) {
      strout << "  ";
    }
    strout << ArrayToString(Arr);
    if (flatten) {
      strout << ", ";
    } else {
      strout << std::endl;
    }
  }
  strout << "]";
  return strout.str();
}


template<typename T>
inline std::string MatrixToString(const std::vector<std::vector<T>>& Mat,
                                  const bool flatten = false) {
  std::ostringstream strout;
  strout << "[";
  if (!flatten) {
    strout << std::endl;
  }
  for (const std::vector<T>& Arr : Mat) {
    if (!flatten) {
      strout << "  ";
    }
    strout << ArrayToString(Arr);
    if (flatten) {
      strout << ", ";
    } else {
      strout << std::endl;
    }
  }
  strout << "]";
  return strout.str();
}



template<typename T>
inline std::string OptionalMatrixToString(const Array<Array<Optional<T>>>& Mat,
                                          const bool flatten = false) {
  std::ostringstream strout;
  strout << "[";
  if (!flatten) {
    strout << std::endl;
  }
  for (const Array<Optional<T>>& Arr : Mat) {
    if (!flatten) {
      strout << "  ";
    }
    strout << OptionalArrayToString(Arr);
    if (flatten) {
      strout << ", ";
    } else {
      strout << std::endl;
    }
  }
  strout << "]";
  return strout.str();
}

template<typename K, typename V>
inline std::string MapToString(const Map<K, V>& Map,
                               const bool flatten = false) {
  std::ostringstream strout;
  strout << "{";
  if (!flatten) {
    strout << std::endl;
  }
  for (const std::pair<K, V>& kv : Map) {
    if (!flatten) {
      strout << "  ";
    }
    strout << kv.first << " : " << kv.second;
    if (flatten) {
      strout << ", ";
    } else {
      strout << std::endl;
    }
  }
  strout << "}";
  return strout.str();
}

template<typename K, typename V, typename H, typename E>
inline std::string MapToString(const std::unordered_map<K, V, H, E>& Map,
                               const bool flatten = false) {
  std::ostringstream strout;
  strout << "{";
  if (!flatten) {
    strout << std::endl;
  }
  for (const std::pair<K, V>& kv : Map) {
    if (!flatten) {
      strout << "  ";
    }
    strout << kv.first << " : " << kv.second;
    if (flatten) {
      strout << ", ";
    } else {
      strout << std::endl;
    }
  }
  strout << "}";
  return strout.str();
}

using runtime::NDArray;

inline NDArray VecToNDArray(const std::vector<float>& vec,
                            const std::vector<int64_t>& shape) {
  int64_t ndarr_size = 1;
  for (const int64_t s : shape) {
    ndarr_size *= s;
  }
  CHECK(vec.size() == static_cast<size_t>(ndarr_size))
      << "Vector size=" << vec.size() << " does not match shape size=" << ndarr_size;
  NDArray ret = NDArray::Empty(shape, DataType::Float(32), {kDLCPU, 0});
  ret.CopyFromBytes(vec.data(), sizeof(float) * ndarr_size);
  return ret;
}

// The following functions are defined in compute_dag.cc.
/*
std::vector<Iterator> GatherAllItersWithSamePrefix(
    const Array<Iterator>& all_iters, const Iterator& iter_0);
 */
Iterator FindIterInInitState(const State& init_state, const Iterator& iter_0);


class SyntheticExprReplacer : public tir::StmtExprMutator {
 private:
  Map<ObjectRef, IntImm> expr_subst_map_;

  PrimExpr VisitExpr_(const ProducerLoadNode* op) override {
    auto producer_subst_map_it = producer_subst_map.find(op->producer);
    if (producer_subst_map_it != producer_subst_map.end()) {
      // LOG(INFO) << "Replacing " << op->producer << " w/ "
      //           << (*producer_subst_map_it).second;
      return ProducerLoad((*producer_subst_map_it).second,
                          op->indices);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

 public:
  Map<DataProducer, te::Tensor> producer_subst_map;

  SyntheticExprReplacer(const Map<ObjectRef, IntImm>& expr_subst_map)
      : expr_subst_map_(expr_subst_map) {
  }

  PrimExpr VisitExpr_(const DynamicAxisNode* op) override final {
    auto expr_subst_map_it = expr_subst_map_.find(op->name_hint);
    if (expr_subst_map_it != expr_subst_map_.end()) {
      return (*expr_subst_map_it).second;
    }
    return tir::StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr(const PrimExpr& expr) override final {
    auto expr_subst_map_it = expr_subst_map_.find(expr);
    if (expr_subst_map_it != expr_subst_map_.end()) {
      return (*expr_subst_map_it).second;
    }
    std::ostringstream strout;
    strout << expr;
    std::string expr_str = strout.str();
    for (const std::pair<ObjectRef, Integer>& kv : expr_subst_map_) {
      strout.str("");
      strout.clear();
      strout << kv.first;
      std::string k_str = strout.str();
      if (expr_str == k_str) {
        // LOG(WARNING) << "Despite not sharing the same address, expr=" << expr
        //              << " and key=" << k_str << " are still deemed equal";
        return kv.second;
      }
    }
    return StmtExprMutator::VisitExpr(expr);
  }
};


// Dispatcher is used to dispatch workload instances to its best matching states.
struct Dispatcher {
  virtual std::unordered_map<size_t, size_t>
  dispatch(const std::vector<float>& scores, const size_t num_states) = 0;
};

struct TopKDispatcher : public Dispatcher {
  virtual std::unordered_map<size_t, size_t>
  dispatch(const std::vector<float>& scores, const size_t num_states) override final;
};


// <bojian/DietCode> Migrated from compute_dag.cc.

using namespace ::tvm::tir;

// Estimate the number of float operations in an expression
class FlopEstimator : public tir::ExprFunctor<double(const PrimExpr& n)> {

  // <bojian/DietCode> Integrate AxisLengthProd as part of the FlopEstimator and
  //                   Add support for dynamic workloads.
 public:
  FlopEstimator(const DynamicAxisReplacer& replacer =
                DynamicAxisReplacer(nullptr))
      : replacer_(replacer) {}
 private:
  arith::Analyzer analyzer_;
  DynamicAxisReplacer replacer_;
  int64_t AxisLengthProd(const Array<tir::IterVar>& axes) {
    int64_t ret = 1.0;
    for (const auto& x : axes) {
      if (const IntImmNode* imm = x->dom->extent.as<IntImmNode>()) {
        ret *= imm->value;
      } else {
        if (const IntImmNode* const imm =
            analyzer_.Simplify(replacer_(x->dom->extent)).as<IntImmNode>()) {
          ret *= imm->value;
          // LOG(INFO) << "ret=" << ret;
        } else {
          // LOG(WARNING) << "Unable to estimate the FLOPs for "
          //              << x->dom->extent;
          return -1.0;
        }
      }
    }
    return ret;
  }

 public:
  double EstimateFlop(const Array<te::Operation>& ops) {
    double ret = 0;
    for (const auto& op : ops) {
      if (auto pop = op.as<te::ComputeOpNode>()) {
        if (pop->attrs.count("FLOP")) {
          // Use user-provided FLOP
          auto pint = pop->attrs["FLOP"].as<IntImmNode>();

          // <bojian/DietCode>
          // ICHECK(pint != nullptr);
          if (pint == nullptr) {
            LOG(FATAL) << "pop->attrs[\"FLOP\"]=" << pop->attrs["FLOP"]
                       << " is NOT an integer";
          } else {
            ret += pint->value;
          }
          // ret += pint->value;
        } else {
          // Estimate by parsing the compute body
          double num_element = AxisLengthProd(pop->axis);
          if (num_element == -1) {
            fail_ = true;
            break;
          }
          cur_type_code_ = pop->output_dtype(0).code();
          double op_per_element = 0;
          for (const auto& x : pop->body) {
            op_per_element += VisitExpr(x);
          }
          ret += num_element * op_per_element;
        }
      } else if (op->IsInstance<te::PlaceholderOpNode>()) {
        {}  // do nothing
      } else {
        LOG(FATAL) << "Invalid op type " << op;
      }
    }

    return fail_ ? -1 : ret;
  }

  double VisitExpr_(const ReduceNode* op) final {
    uint64_t num_iter = 1;
    for (const auto& x : op->axis) {
      if (auto imm = x->dom->extent.as<IntImmNode>()) {
        num_iter *= imm->value;
      } else {

        // <bojian/DietCode>
        if (const IntImmNode* const imm =
            analyzer_.Simplify(replacer_(x->dom->extent)).as<IntImmNode>()) {
          num_iter *= imm->value;
        } else {

          fail_ = true;
          num_iter = -1;

        }
      }
    }
    double body_flop = 0;
    for (size_t i = 0; i < op->combiner->result.size(); ++i) {
      body_flop += VisitExpr(op->combiner->result[i]);
      body_flop += VisitExpr(op->source[i]);
    }
    return num_iter * body_flop;
  }

  double VisitExpr_(const FloatImmNode* op) final { return 0.0; }
  double VisitExpr_(const IntImmNode* op) final { return 0.0; }
  double VisitExpr_(const ProducerLoadNode* op) final { return 0.0; }

  double VisitExpr_(const CastNode* op) final { return VisitExpr(op->value); }
  double VisitExpr_(const VarNode* op) final { return 0.0; }

  double VisitExpr_(const SelectNode* op) final {
    return VisitExpr(op->condition) +
           std::max(VisitExpr(op->true_value), VisitExpr(op->false_value));
  }

#define VisitBinary(Node)                                         \
  double VisitExpr_(const Node* op) final {                       \
    double base = op->dtype.code() == cur_type_code_ ? 1.0 : 0.0; \
    return base + VisitExpr(op->a) + VisitExpr(op->b);            \
  }

#define VisitUnary(Node)                                          \
  double VisitExpr_(const Node* op) final {                       \
    double base = op->dtype.code() == cur_type_code_ ? 1.0 : 0.0; \
    return base + VisitExpr(op->a);                               \
  }

  VisitBinary(AddNode);
  VisitBinary(SubNode);
  VisitBinary(MulNode);
  VisitBinary(DivNode);
  VisitBinary(ModNode);
  VisitBinary(FloorDivNode);
  VisitBinary(FloorModNode);
  VisitBinary(MaxNode);
  VisitBinary(MinNode);
  VisitBinary(EQNode);
  VisitBinary(NENode);
  VisitBinary(LTNode);
  VisitBinary(LENode);
  VisitBinary(GTNode);
  VisitBinary(GENode);
  VisitBinary(AndNode);
  VisitBinary(OrNode);
  VisitUnary(NotNode);

  // undefine macros to avoid potential conflicts
#undef VisitBinary
#undef VisitUnary

  double VisitExpr_(const CallNode* op) final {
    double ret = 0.0;
    for (const auto& x : op->args) {
      ret += VisitExpr(x);
    }
    return ret;
  }

  double VisitExprDefault_(const Object* op) final {
    fail_ = true;
    return -1.0;
  }

 private:
  bool fail_{false};
  int cur_type_code_;
};


void AdaptStateToWorkload(const SearchTask& task, const State& state,
                          const Array<IntImm>& shape_values,
                          const float score,
                          float* const occupancy_penalty,
                          float* const padding_penalty,
                          float* const adapted_score
                          );

// double GetSyntheticWorkloadFlopCtFromState(const SearchTask& task,
//                                            const State& state);
std::pair<double, double>
GetCherryPickedWklInstFlopCtFromState(const SearchTask& task,
                                      const State& state);

double EstimateFlopForInst(const ComputeDAG& compute_dag,
                           // const Array<Step>& transform_steps,
                           const Array<String>& shape_vars,
                           const Array<IntImm>& shape_values);


}  // namespace auto_scheduler
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_UTILS_H_

