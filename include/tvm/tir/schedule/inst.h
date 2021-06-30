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
#ifndef TVM_TIR_SCHEDULE_INST_H_
#define TVM_TIR_SCHEDULE_INST_H_

#include <tvm/tir/schedule/schedule.h>

#include <utility>
#include <vector>

namespace tvm {
template <typename, typename>
class AttrRegistry;
namespace tir {

/*!
 * \brief Kind of an instruction, e.g. Split, Reorder, etc
 */
class InstKindNode : public runtime::Object {
 public:
  /*!
   * \brief A functor type used to apply an instruction to a Schedule
   * \return The functor returns an array of random variables
   */
  using FApplyToSchedule = std::function<Array<ObjectRef>(
      /*! \param sch The schedule to be applied on */
      const Schedule& sch,
      /*! \param inputs The input random variables */
      const Array<ObjectRef>& inputs,
      /*! \param attrs Instruction attributes */
      const Array<ObjectRef>& attrs,
      /*! \param decision Decisions made on the instruction */
      const Optional<ObjectRef>& decision)>;
  /*!
   * \brief A functor type used to convert an instruction to python api call
   * \return A string representing the python api call
   */
  using FAsPython = std::function<String(
      /*! \param inputs Names of the input random variables */
      const Array<ObjectRef>& inputs,
      /*! \param attrs Instruction attributes */
      const Array<ObjectRef>& attrs,
      /*! \param decisions Decisions made on the instruction */
      const Optional<ObjectRef>& decision,
      /*! \param outputs Names of the output random variables */
      const Array<String>& outputs)>;
  /*!
   * \brief A functor type used to serialize attributes of an instrcution
   * \return An array, serialized attributes
   */
  using FAttrsAsJSON = std::function<Array<ObjectRef>(
      /*! \brief The attributes to be serialized */
      const Array<ObjectRef>& attrs)>;
  /*!
   * \brief A functor type used to deserialize attributes of an instrcution
   * \return An array, deserialized attributes
   */
  using FAttrsFromJSON = std::function<Array<ObjectRef>(
      /*! \brief The attributes to be serialized */
      const Array<ObjectRef>& attrs_record)>;

 public:
  /*! \brief The name of the instruction */
  String name;
  /*! \brief Indicates if an instructino is pure */
  bool is_pure{false};
  /*! \brief The functor to apply an instrruction onto a schedule class */
  FApplyToSchedule f_apply_to_schedule{nullptr};
  /*! \brief The functor to convert an instruction to python api call */
  FAsPython f_as_python{nullptr};
  /*! \brief The functor to serialize the attributes of an instruction */
  FAttrsAsJSON f_attrs_as_json{nullptr};
  /*! \brief The functor to deserialize the attributes of an instruction */
  FAttrsFromJSON f_attrs_from_json{nullptr};

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("is_pure", &is_pure);
  }

  static constexpr const char* _type_key = "tir.InstKind";
  TVM_DECLARE_FINAL_OBJECT_INFO(InstKindNode, runtime::Object);

 private:
  /*! \brief The index in the attr registry */
  uint32_t reg_index_{0};
  /*! \return The internal attr registry index. */
  uint32_t AttrRegistryIndex() const { return reg_index_; }
  /*! \brief The repr to be printed in registry*/
  String AttrRegistryName() const { return name; }

  friend class InstKindRegEntry;
};

/*!
 * \brief Managed reference to InstKindNode
 * \sa InstKindNode
 */
class InstKind : public runtime::ObjectRef {
 public:
  static InstKind Get(const String& inst_kind_name);
  TVM_DEFINE_OBJECT_REF_METHODS(InstKind, runtime::ObjectRef, InstKindNode);
};

/*!
 * \brief An instruction in the trace
 */
class InstNode : public runtime::Object {
 public:
  /*! \brief Type of the instruction */
  InstKind kind;
  /*! \brief The input random variables of the instruction */
  Array<ObjectRef> inputs;
  /*! \brief The attributes of the instruction */
  Array<ObjectRef> attrs;
  /*! \brief The output random variables of the instruction */
  Array<ObjectRef> outputs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("kind", &kind);
    v->Visit("inputs", &inputs);
    v->Visit("attrs", &attrs);
    v->Visit("outputs", &outputs);
  }

  static constexpr const char* _type_key = "tir.Inst";
  TVM_DECLARE_FINAL_OBJECT_INFO(InstNode, runtime::Object);
};

/*!
 * \brief Managed reference to InstNode
 * \sa InstNode
 */
class Inst : public runtime::ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param kind The kind of the instruction
   * \param inputs The input random variables of the instruction
   * \param attrs The attributes of the instruction
   * \param outputs The output random variables of the instruction
   */
  explicit Inst(InstKind kind, Array<ObjectRef> inputs, Array<ObjectRef> attrs,
                Array<ObjectRef> outputs);

  TVM_DEFINE_OBJECT_REF_METHODS(Inst, runtime::ObjectRef, InstNode);
};

/*!
 * \brief A helper macro to register instruction traits, only used in `TVM_REGISTER_INST_KIND`
 * \sa TVM_REGISTER_INST_KIND
 */
#define TVM_INST_KIND_REGISTER_VAR_DEF \
  static DMLC_ATTRIBUTE_UNUSED ::tvm::tir::InstKindRegEntry& __make_##InstKind

/*!
 * \brief Register the traits of an instruction kind
 * \param InstKindTraits A traits class for an instruction kind
 *
 * Example:
 *
 * \code
 *
 * struct SomeInstKindTraits {
 *   static constexpr const char* kName = "name-of-the-instruction";
 *   static constexpr bool kIsPure = false;
 *
 *   // Convertible to `InstKindNode::FApplyToSchedule`
 *   static Array<ObjectRef> ApplyToSchedule(
 *      const tir::Schedule& sch,
 *      const Array<ObjectRef>& inputs,
 *      const Array<ObjectRef>& attrs,
 *      const Optional<ObjectRef>& decision);
 *
 *   // Convertible to `InstKindNode::FAsPython`
 *   static String AsPython(
 *      const Array<String>& inputs,
 *      const Array<ObjectRef>& attrs,
 *      const Optional<ObjectRef>& decision,
 *      const Array<String>& outputs);
 *
 *   // Convertible to `InstKindNode::FAttrsAsJSON`
 *   static Array<ObjectRef> AttrsAsJSON(
 *      const Array<ObjectRef>& attrs);
 *
 *   // Convertible to `InstKindNode::FAttrsFromJSON`
 *   static Array<ObjectRef> AttrsFromJSON(
 *      const Array<ObjectRef>& attrs_record);
 * };
 *
 * TVM_REGISTER_INST_KIND(SomeInstKindTraits);
 *
 * \endcode
 */
#define TVM_REGISTER_INST_KIND(InstKindTraits)                           \
  TVM_STR_CONCAT(TVM_INST_KIND_REGISTER_VAR_DEF, __COUNTER__) =          \
      ::tvm::tir::InstKindRegEntry::RegisterOrGet(InstKindTraits::kName) \
          .set_name()                                                    \
          .set_is_pure(InstKindTraits::kIsPure)                          \
          .set_apply_to_schedule(InstKindTraits::ApplyToSchedule)        \
          .set_attrs_as_json(InstKindTraits::AttrsAsJSON)                \
          .set_attrs_from_json(InstKindTraits::AttrsFromJSON)            \
          .set_as_python(InstKindTraits::AsPython)

/*!
 * \brief A convenient helper to define an instruction kind. When inherited in curiously recursive
 * template pattern, the derived class `TTraits` only needs to define two unpacked functions. See
 * the example for more details.
 * \tparam TTraits The derived class
 *
 *
 * Example:
 *
 * \code
 *
 * struct SamplePerfectTileTraits : public UnpackedInstTraits<SamplePerfectTileTraits> {
 *   static constexpr const char* kName = "SamplePerfectTile";
 *   static constexpr bool kIsPure = false;
 *
 *  private:
 *   static constexpr size_t kNumInputs = 1;
 *   static constexpr size_t kNumAttrs = 2;
 *   static constexpr size_t kNumDecisions = 1;
 *
 *   // Calling convention:
 *   // - All the arguments must be ObjectRef
 *   // - The 1st argument is Schedule
 *   // - The next `kNumInputs` arguments are input random variables
 *   // - The next `kNumAttrs` arguments are attributes
 *   // - The next argument is decision, if `kNumDecisions == 1`
 *   static Array<Var> UnpackedApplyToSchedule(
 *      Schedule sch,
 *      LoopRV loop_rv,
 *      Integer n,
 *      Integer max_innermost_factor,
 *      Optional<Array<Integer>> decision) {
 *     return sch->SamplePerfectTile(loop_rv, n->value, max_innermost_factor->value, decision);
 *   }
 *
 *   // Calling convention:
 *   // - All the arguments must be ObjectRef
 *   // - The 1st argument is an array containing names of output random variables
 *   // - The next `kNumInputs` arguments are names of input random variables
 *   // - The next `kNumAttrs` arguments are attributes
 *   // - The next argument is decision, if `kNumDecisions == 1`
 *   static String UnpackedAsPython(
 *      Array<String> outputs,
 *      String loop_rv,
 *      Integer n,
 *      Integer max_innermost_factor,
 *      Optional<Array<Integer>> decision) {
 *     PythonAPICall py("sample_perfect_tile");
 *     py.Input("loop", loop_rv);
 *     py.Input("n", n->value);
 *     py.Input("max_innermost_factor", max_innermost_factor->value);
 *     py.Decision(decision);
 *     py.Outputs(outputs);
 *     return py.Str();
 *   }
 *
 *   template <typename>
 *   friend struct UnpackedInstTraits;
 * };
 *
 * TVM_REGISTER_INST_KIND(SamplePerfectTileTraits);
 * \endcode
 */
template <class TTraits>
struct UnpackedInstTraits {
  /*!
   * \brief Unpack the arguments into the calling convention of `TTraits::UnpackedApplyToSchedule`
   * \sa InstKindNode::FApplyToSchedule
   */
  static Array<ObjectRef> ApplyToSchedule(const Schedule& sch, const Array<ObjectRef>& inputs,
                                          const Array<ObjectRef>& attrs,
                                          const Optional<ObjectRef>& decision);

  /*!
   * \brief Unpack the arguments into the calling convention of `TTraits::UnpackedAsPython`
   * \sa InstKindNode::FAsPython
   */
  static String AsPython(const Array<ObjectRef>& inputs, const Array<ObjectRef>& attrs,
                         const Optional<ObjectRef>& decision, const Array<String>& outputs);

  /*! \brief No customized serializer */
  static constexpr std::nullptr_t AttrsAsJSON = nullptr;

  /*! \brief No customized deserializer */
  static constexpr std::nullptr_t AttrsFromJSON = nullptr;

 protected:
  template <size_t delta>
  static TVM_ALWAYS_INLINE void _SetInputs(const runtime::TVMArgsSetter& setter,
                                           const Array<ObjectRef>& inputs);
  template <size_t delta>
  static TVM_ALWAYS_INLINE void _SetAttrs(const runtime::TVMArgsSetter& setter,
                                          const Array<ObjectRef>& attrs);
  template <size_t delta>
  static TVM_ALWAYS_INLINE void _SetDecision(const runtime::TVMArgsSetter& setter,
                                             const Optional<ObjectRef>& decision);
  static TVM_ALWAYS_INLINE Array<ObjectRef> _ConvertOutputs(const TVMRetValue& rv);
};

/*! \brief A helper class to construct schedule API call in python syntax */
class PythonAPICall {
 public:
  /*!
   * \brief Constructor
   * \param method_name The name of the schedule API to be called
   */
  explicit PythonAPICall(String method_name)
      : method_name_(std::move(method_name)), output_(NullOpt) {}
  /*! \brief Add an attribute */
  void Input(String arg_name, int arg);
  /*! \brief Add an attribute */
  void Input(String arg_name, int64_t arg);
  /*! \brief Add an attribute */
  void Input(String arg_name, double arg);
  /*! \brief Add an input random variable */
  void Input(String arg_name, String arg);
  /*! \brief Add an attribute */
  void Input(String arg_name, ObjectRef arg);
  /*! \brief Add a decision */
  void Decision(ObjectRef decision);
  /*! \brief Add a single output random variable */
  void SingleOutput(Array<String> unit_array);
  /*! \brief Add a list of output random variables */
  void OutputList(Array<String> outputs);
  /*! \returns The schedule API call in python syntax */
  String Str() const;

 private:
  String method_name_;
  Optional<String> output_;
  std::vector<String> arg_names_;
  std::vector<String> args_;
};

/********** implementation details **********/

/*! \brief An entry in the registry of InstKind */
class InstKindRegEntry {
 public:
  static InstKindRegEntry& RegisterOrGet(const String& name);

  InstKindRegEntry& set_name() {
    get_mutable()->name = this->name;
    return *this;
  }

  InstKindRegEntry& set_is_pure(bool is_pure) {
    get_mutable()->is_pure = is_pure;
    return *this;
  }

  InstKindRegEntry& set_apply_to_schedule(InstKindNode::FApplyToSchedule f_apply_to_schedule) {
    get_mutable()->f_apply_to_schedule = std::move(f_apply_to_schedule);
    return *this;
  }

  InstKindRegEntry& set_as_python(InstKindNode::FAsPython f_as_python) {
    get_mutable()->f_as_python = std::move(f_as_python);
    return *this;
  }

  InstKindRegEntry& set_attrs_as_json(InstKindNode::FAttrsAsJSON f_attrs_as_json) {
    get_mutable()->f_attrs_as_json = std::move(f_attrs_as_json);
    return *this;
  }

  InstKindRegEntry& set_attrs_from_json(InstKindNode::FAttrsFromJSON f_attrs_from_json) {
    get_mutable()->f_attrs_from_json = std::move(f_attrs_from_json);
    return *this;
  }

 private:
  /*! \brief Private constructor, used only by AttrRegistry */
  explicit InstKindRegEntry(uint32_t reg_index);
  /*! \brief Get the mutable reference to the internal InstKind */
  InstKindNode* get_mutable() const { return const_cast<InstKindNode*>(inst_kind_.get()); }

  /*! \brief The name of the registry entry */
  String name;
  /*! \brief The instruction kind */
  InstKind inst_kind_;
  template <typename, typename>
  friend class ::tvm::AttrRegistry;
  friend class InstKind;
};

// forward declaration
namespace details {

template <typename... Args>
struct _ArgsPacker;

template <>
struct _ArgsPacker<> {
  static constexpr bool checked = true;
};

template <typename TObjectRef, typename... Args>
struct _ArgsPacker<TObjectRef, Args...> {
  static constexpr bool checked =
      std::is_base_of<ObjectRef, TObjectRef>::value && _ArgsPacker<Args...>::checked;
};

template <typename T>
struct _MethodType {};

template <typename TReturn, typename... Args>
struct _MethodType<TReturn(Args...)> {
  using return_type = TReturn;
  using argument_type = _ArgsPacker<Args...>;
};

template <typename T>
struct _NumArgs {};

template <typename TReturn, typename... Args>
struct _NumArgs<TReturn(Args...)> {
  static constexpr size_t value = sizeof...(Args);
};

template <typename>
struct _IsTVMArray : std::false_type {};

template <typename T>
struct _IsTVMArray<runtime::Array<T>> : std::true_type {};

template <typename T>
struct _IsSingleObject
    : std::integral_constant<bool, std::is_base_of<ObjectRef, T>::value && !_IsTVMArray<T>::value> {
};

template <class T>
using ReturnType = typename _MethodType<std::remove_cv_t<T>>::return_type;

template <class T>
static constexpr bool ArgumentAreAllObjects =
    _MethodType<std::remove_cv_t<T>>::argument_type::checked;

template <class T>
static constexpr size_t NumArgs = _NumArgs<std::remove_cv_t<T>>::value;

template <class T>
static constexpr int IsTVMArray = _IsTVMArray<std::remove_cv_t<T>>::value;

template <class T>
static constexpr int IsSingleObject = _IsSingleObject<std::remove_cv_t<T>>::value;

};  // namespace details

template <class TTraits>
Array<ObjectRef> UnpackedInstTraits<TTraits>::ApplyToSchedule(const Schedule& sch,
                                                              const Array<ObjectRef>& inputs,
                                                              const Array<ObjectRef>& attrs,
                                                              const Optional<ObjectRef>& decision) {
  using method_type = decltype(TTraits::UnpackedApplyToSchedule);
  using return_type = details::ReturnType<method_type>;
  static_assert(details::ArgumentAreAllObjects<method_type>,
                "All arguments to `UnpackedApplyToSchedule` must be subclasses of ObjectRef");
  constexpr size_t kNumArgs = details::NumArgs<method_type>;
  constexpr size_t kNumInputs = TTraits::kNumInputs;
  constexpr size_t kNumAttrs = TTraits::kNumAttrs;
  constexpr size_t kNumDecisions = TTraits::kNumDecisions;
  static_assert(kNumArgs == 1 + kNumInputs + kNumAttrs + kNumDecisions,
                "length of argument list mismatch");
  TVMValue tvm_values[kNumArgs];
  int tvm_type_codes[kNumArgs];
  runtime::TVMArgsSetter setter(tvm_values, tvm_type_codes);
  setter(0, sch);
  TTraits::template _SetInputs<1>(setter, inputs);
  TTraits::template _SetAttrs<1 + kNumInputs>(setter, attrs);
  TTraits::template _SetDecision<1 + kNumInputs + kNumAttrs>(setter, decision);
  PackedFunc pf([](const TVMArgs& args, TVMRetValue* rv) -> void {
    using runtime::detail::unpack_call;
    constexpr size_t kNumArgs = details::NumArgs<method_type>;
    ICHECK_EQ(args.size(), kNumArgs);
    unpack_call<return_type, kNumArgs>(nullptr, TTraits::UnpackedApplyToSchedule, args, rv);
  });
  TVMRetValue rv;
  pf.CallPacked(TVMArgs(tvm_values, tvm_type_codes, kNumArgs), &rv);
  return TTraits::_ConvertOutputs(rv);
}

template <class TTraits>
String UnpackedInstTraits<TTraits>::AsPython(const Array<ObjectRef>& inputs,
                                             const Array<ObjectRef>& attrs,
                                             const Optional<ObjectRef>& decision,
                                             const Array<String>& outputs) {
  using method_type = decltype(TTraits::UnpackedAsPython);
  using return_type = details::ReturnType<method_type>;
  static_assert(details::ArgumentAreAllObjects<method_type>,
                "All arguments to `UnpackedAsPython` must be subclasses of ObjectRef");
  constexpr size_t kNumArgs = details::NumArgs<method_type>;
  constexpr size_t kNumInputs = TTraits::kNumInputs;
  constexpr size_t kNumAttrs = TTraits::kNumAttrs;
  constexpr size_t kNumDecisions = TTraits::kNumDecisions;
  static_assert(kNumArgs == 1 + kNumInputs + kNumAttrs + kNumDecisions,
                "length of argument list mismatch");
  TVMValue tvm_values[kNumArgs];
  int tvm_type_codes[kNumArgs];
  runtime::TVMArgsSetter setter(tvm_values, tvm_type_codes);
  setter(0, outputs);
  TTraits::template _SetInputs<1>(setter, inputs);
  TTraits::template _SetAttrs<1 + kNumInputs>(setter, attrs);
  TTraits::template _SetDecision<1 + kNumInputs + kNumAttrs>(setter, decision);
  PackedFunc pf([](const TVMArgs& args, TVMRetValue* rv) -> void {
    using runtime::detail::unpack_call;
    constexpr size_t kNumArgs = details::NumArgs<method_type>;
    ICHECK_EQ(args.size(), kNumArgs);
    unpack_call<return_type, kNumArgs>(nullptr, TTraits::UnpackedAsPython, args, rv);
  });
  TVMRetValue rv;
  pf.CallPacked(TVMArgs(tvm_values, tvm_type_codes, kNumArgs), &rv);
  String result = rv;
  return result;
}

template <class TTraits>
template <size_t delta>
TVM_ALWAYS_INLINE void UnpackedInstTraits<TTraits>::_SetInputs(const runtime::TVMArgsSetter& setter,
                                                               const Array<ObjectRef>& inputs) {
  constexpr size_t kNumInputs = TTraits::kNumInputs;
  ICHECK_EQ(kNumInputs, inputs.size())
      << "ValueError: Incorrect kNumInputs for instruction: " << TTraits::kName;
  const ObjectRef* ptr = inputs.template as<ArrayNode>()->begin();
  for (size_t i = 0; i < kNumInputs; ++i) {
    setter(i + delta, *(ptr + i));
  }
}

template <class TTraits>
template <size_t delta>
TVM_ALWAYS_INLINE void UnpackedInstTraits<TTraits>::_SetAttrs(const runtime::TVMArgsSetter& setter,
                                                              const Array<ObjectRef>& attrs) {
  constexpr size_t kNumAttrs = TTraits::kNumAttrs;
  ICHECK_EQ(kNumAttrs, attrs.size())
      << "ValueError: Incorrect kNumAttrs for instruction: " << TTraits::kName;
  const ObjectRef* ptr = attrs.as<ArrayNode>()->begin();
  for (size_t i = 0; i < kNumAttrs; ++i) {
    setter(i + delta, *(ptr + i));
  }
}

template <class TTraits>
template <size_t delta>
TVM_ALWAYS_INLINE void UnpackedInstTraits<TTraits>::_SetDecision(
    const runtime::TVMArgsSetter& setter, const Optional<ObjectRef>& decision) {
  constexpr size_t kNumDecisions = TTraits::kNumDecisions;
  static_assert(kNumDecisions <= 1, "an instruction is supposed to have at most 1 decision");
  if (kNumDecisions == 1) {
    setter(delta, decision);
  } else {
    ICHECK(!decision.defined());
  }
}

template <class TTraits>
TVM_ALWAYS_INLINE Array<ObjectRef> UnpackedInstTraits<TTraits>::_ConvertOutputs(
    const TVMRetValue& rv) {
  using method_type = decltype(TTraits::UnpackedApplyToSchedule);
  using return_type = details::ReturnType<method_type>;
  constexpr int is_array = details::IsTVMArray<return_type>;
  constexpr int is_single_obj = details::IsSingleObject<return_type>;
  constexpr int is_void = std::is_void<return_type>::value;
  static_assert(is_array || is_single_obj || is_void, "return type not supported");
  static_assert(is_array + is_single_obj + is_void == 1, "internal template error");
  if (is_void) {
    return {};
  } else if (is_single_obj) {
    ObjectRef obj = rv;
    return {obj};
  } else if (is_array) {
    ObjectRef obj = rv;
    const ArrayNode* array = obj.as<ArrayNode>();
    return GetRef<Array<ObjectRef>>(array);
  }
}

}  // namespace tir
}  // namespace tvm

#endif  //  TVM_TIR_SCHEDULE_INST_H_
