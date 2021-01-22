// <bojian/TVM-SymbolicTuning>
#pragma once

#include <sstream>
#include <string>
#include <vector>

#include <tvm/runtime/container.h>


// #define SYMTUNE_DEBUG_TRACE
// #define SYMTUNE_SCHED_OPT


template<typename PrimExprT>
std::string exprs_tostr(
    const std::vector<PrimExprT>& exprs) {
  std::ostringstream strout;
  strout << "[";
  for (const PrimExprT& expr : exprs) {
    strout << expr << ", ";
  }
  strout << "]";

  return strout.str();
}

template<typename PrimExprT>
std::string exprs_tostr(
    const ::tvm::runtime::Array<PrimExprT>& exprs) {
  std::ostringstream strout;
  strout << "[";
  for (const PrimExprT& expr : exprs) {
    strout << expr << ", ";
  }
  strout << "]";

  return strout.str();
}
