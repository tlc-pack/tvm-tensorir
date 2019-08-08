/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Solvers for integer set
 */

#include <tvm/arithmetic.h>
#include <tvm/ir.h>
#include "../tensorir/node_util.h"
#include "int_set.h"

namespace tvm {
namespace arith {

/*!
 * \brief Return int sets with minimal sizes to make `produce` cover `requirement`
 * \param vars The unknown vars to solve
 * \param produces The produced int sets
 * \param requirements The required int sets
 * \return sets The set of unknown vars
 */
Array<IntSet> SolveCover(Array<Var> vars, Array<IntSet> produces, Array<IntSet> requirements) {
  std::vector<Array<IntSet> > to_merge(vars.size());
  StdNodeMap<Var, size_t> var2index;
  Analyzer analyzer;

  for (size_t i = 0; i < vars.size(); ++i) {
    var2index[vars[i]] = i;
  }

  // fit requirements one by one
  CHECK_EQ(produces.size(), requirements.size());
  for (size_t i = 0; i < produces.size(); ++i) {
    const IntervalSetNode* iset = produces[i].as<IntervalSetNode>();
    CHECK(iset != nullptr);

    CHECK(iset->max_value->is_type<Variable>())
      << "The min of produces range must be a single variable";
    Var var = Downcast<Var>(iset->min_value);

    CHECK_GT(var2index.count(var), 0) << "Find irrelevant variable in produces";
    size_t id = var2index[var];

    if (requirements[i]->is_type<IntervalSetNode>()) {
      const IntervalSetNode* require = requirements[i].as<IntervalSetNode>();

      Expr base = require->min_value;
      Expr produces_len = analyzer.Simplify(iset->max_value - iset->min_value + 1);
      Expr extent = analyzer.Simplify(
          (require->max_value - require->min_value + 1 + produces_len - 1) / produces_len);
      Expr strides = produces_len;

      to_merge[id].push_back(StrideSet(base, make_const(base.type(), 1),
                                       Array<Expr>{extent},
                                       Array<Expr>{produces_len}));
    } else {
      LOG(FATAL) << "Only support IntervalSet";
    }
  }

  // merge
  Array<IntSet> ret;
  for (size_t i = 0; i < vars.size(); ++i) {
    if (to_merge[i].size() == 0) {  // return nullptr for unbounded free vars
      ret.push_back(IntSet(NodePtr<Node>(nullptr)));
    } else {
      ret.push_back(Union(to_merge[i]));
    }
  }

  return ret;
}

/*!
 * \brief Return int sets with maximum sizes to make `consume` covered by `requirement`
 * \param vars The unknown vars to solve
 * \param produces The produced int sets
 * \param requirements The required int sets
 * \return sets The set of unknown vars
 */
Array<IntSet> SolveCoverBy(Array<Var> vars, Array<IntSet> produces, Array<IntSet> requirements) {
  std::vector<Array<IntSet> > to_merge(vars.size());
  StdNodeMap<Var, size_t> var2index;
  Analyzer analyzer;

  for (size_t i = 0; i < vars.size(); ++i) {
    var2index[vars[i]] = i;
  }

  // fit requirements one by one
  CHECK_EQ(produces.size(), requirements.size());
  for (size_t i = 0; i < produces.size(); ++i) {
    const IntervalSetNode* iset = produces[i].as<IntervalSetNode>();
    CHECK(iset != nullptr);

    CHECK(iset->max_value->is_type<Variable>())
      << "The min of produces range must be a single variable";
    Var var = Downcast<Var>(iset->min_value);

    CHECK(var2index.count(var)) << "Find irrelevant variable in produces";
    size_t id = var2index[var];

    if (requirements[i]->is_type<IntervalSetNode>()) {
      const IntervalSetNode* require = requirements[i].as<IntervalSetNode>();

      Expr base = require->min_value;
      Expr produces_len = analyzer.Simplify(iset->max_value - iset->min_value + 1);
      Expr extent = analyzer.Simplify(
          (require->max_value - require->min_value + 1 - produces_len) / produces_len + 1);
      Expr strides = produces_len;

      to_merge[id].push_back(StrideSet(base, make_const(base.type(), 1),
                                       Array<Expr>{extent},
                                       Array<Expr>{produces_len}));
    } else {
      LOG(FATAL) << "Only support IntervalSet";
    }
  }

  // merge
  Array<IntSet> ret;
  for (size_t i = 0; i < vars.size(); ++i) {
    if (to_merge[i].size() == 0) {  // return nullptr for unbounded free vars
      ret.push_back(IntSet(NodePtr<Node>(nullptr)));
    } else {
      ret.push_back(Intersect(to_merge[i]));
    }
  }

  return ret;
};

};  // namespace arith
};  // namespace tvm
