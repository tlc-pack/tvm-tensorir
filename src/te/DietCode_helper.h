// <bojian/DietCode>
#pragma once

#include <sstream>
#include <string>
#include <vector>

#include <tvm/runtime/container.h>


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
