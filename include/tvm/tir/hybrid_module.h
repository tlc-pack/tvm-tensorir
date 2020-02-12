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

/*
 * \file tvm/tir/hybrid_module.h
 * \brief Module that stores global functions.
 */

#ifndef TVM_HYBRID_MODULE_H
#define TVM_HYBRID_MODULE_H

#include <tvm/tir/ir.h>
#include <tvm/node/container.h>

namespace tvm {
namespace tir {

class GlobalVar;
/*!
 * \brief Global variable that refers to function definitions
 *
 * GlobalVar is used to enable recursive calls between functions
 */
class GlobalVarNode : public Object {
 public:
  /*! \brief The name of the variable, this only acts as a hint. */
  std::string name_hint;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name_hint", &name_hint);
  }

  static constexpr const char* _type_key = "TirGlobalVar";
  TVM_DECLARE_FINAL_OBJECT_INFO(GlobalVarNode, Object);
};

/*!
 * \brief Managed reference to GlobalVarNode.
 */
class GlobalVar : public ObjectRef {
 public:
  TVM_DLL explicit GlobalVar(std::string name_hint);

  TVM_DEFINE_OBJECT_REF_METHODS(GlobalVar, ObjectRef, GlobalVarNode);
};

class Module;
/*!
 * \brief Module that holds collections of functions.
 */
class ModuleNode : public Object {
 public:
  /*! \brief the name of the module. */
  std::string name;
  /*! \brief a map from GlobalVar to all global functions. */
  Map<GlobalVar, Function> functions;

  /*! \brief default constructor */
  ModuleNode() {}

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("functions", &functions);
  }

  static constexpr const char* _type_key = "TirModule";
  TVM_DECLARE_FINAL_OBJECT_INFO(ModuleNode, Object);

 private:
  /*! \brief A map from string names to global variables. */
  Map<std::string, GlobalVar> global_var_map_;
  friend class Module;
};

/*
 * \brief Managed reference to ModuleNode.
 */
class Module : public ObjectRef {
 public:
  /*!
  * \brief constructor
  * \param functions Functions in the module.
  */
  TVM_DLL explicit Module(std::string name, Map<GlobalVar, Function> functions);

  TVM_DEFINE_OBJECT_REF_METHODS(Module, ObjectRef, ModuleNode);

  /*!
   * \brief append a new Function into current Module
   * \param globalVar the corresponding GlobalVar
   * \param function the Function to be appended
   */
  void append(const GlobalVar& globalVar, const Function& function);
};

}  // namespace tir
}  // namespace tvm

#endif //TVM_HYBRID_MODULE_H
