/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Solvers for integer set
 */

#include <tvm/arithmetic.h>
#include <tvm/ir.h>
#include "int_set_internal.h"
#include "../tensorir/node_util.h"

namespace tvm {
namespace arith {

/*!
 * \brief Return int sets with minimal sizes to make produce cover requirement
 * \param vars The unknown vars to solve
 * \param produce The produced int sets
 * \param requirement The required int sets
 * \param sets The set of unknown vars
 */
Array<IntSet> SolveCover(Array<Var> vars, Array<IntSet> produce, Array<IntSet> requirement) {
  std::vector<Array<IntSet> > to_merge(vars.size());
  StdNodeMap<Var, size_t> var2index;

  for (size_t i = 0; i < vars.size(); ++i) {
    var2index[vars[i]] = i;
  }

  // fit requirements one by one
  CHECK_EQ(produce.size(), requirement.size());
  for (size_t i = 0; i < produce.size(); ++i) {
    const IntervalSet* iset = produce[i].as<IntervalSet>();
    CHECK(iset != nullptr);

    CHECK(iset->i.min->is_type<Variable>()) << "The min of produce range must be a single variable";
    Var var = Downcast<Var>(iset->i.min);

    CHECK_GT(var2index.count(var), 0) << "Find irrelevant variable in produce";
    size_t id = var2index[var];

    if (requirement[i]->is_type<IntervalSet>()) {
      const IntervalSet* require = requirement[i].as<IntervalSet>();

      Expr base = require->i.min;
      Expr produce_len = simplify(iset->i.max - iset->i.min + 1);
      Expr extent = simplify((require->i.max - require->i.min + 1 + produce_len - 1) / produce_len);
      Expr strides = produce_len;

      to_merge[id].push_back(StrideSet::make(Interval::single_point(base),
                                             Array<Expr>{extent},
                                             Array<Expr>{produce_len}));
    } else {
      LOG(FATAL) << "Only support IntervalSet";
    }
  }

  // merge
  Array<IntSet> ret;
  for (size_t i = 0; i < vars.size(); ++i) {
    if (to_merge[i].size() == 0) { // unbounded free vars
      ret.push_back(IntSet(NodePtr<Node>(nullptr)));
    }
    ret.push_back(Union(to_merge[i]));
  }

  return ret;
}

}; // namespace arith
}; // namespace tvm