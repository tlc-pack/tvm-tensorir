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
 * \file tvm/node/container.h
 * \brief Array/Map container in the DSL graph.
 */
#ifndef TVM_NODE_CONTAINER_H_
#define TVM_NODE_CONTAINER_H_

#include <type_traits>
#include <vector>
#include <initializer_list>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <string>
#include "node.h"
#include "memory.h"

namespace tvm {


/*! \brief array node content in array */
class ArrayNode : public Node {
 public:
  /*! \brief the data content */
  std::vector<NodePtr<Node> > data;

  void VisitAttrs(AttrVisitor* visitor) final {
     // Visitor to array have no effect.
  }

  static constexpr const char* _type_key = "Array";
  TVM_DECLARE_NODE_TYPE_INFO(ArrayNode, Node);
};

/*! \brief The hash function for NodePtr */
struct NodePtrHash {
  size_t operator()(const NodePtr<Node>& n) const {
    return std::hash<Node*>()(n.get());
  }
};

/*! \brief The equal comparator for NodePtr */
struct NodePtrEqual {
  bool operator()(const NodePtr<Node>& a, const NodePtr<Node>& b) const {
    return a.get() == b.get();
  }
};

/*! \brief set node content */
class SetNode : public Node {
 public:
  void VisitAttrs(AttrVisitor* visitors) final {
    // Visitor to set have no effect.
  }

  /*! \brief The corresponding container type */
  using ContainerType = std::unordered_set<
      NodePtr<Node>, NodePtrHash, NodePtrEqual>;

  /*! \brief the data content */
  ContainerType data;

  static constexpr const char* _type_key = "Set";
  TVM_DECLARE_NODE_TYPE_INFO(SetNode, Node);
};

/*! \brief map node content */
class MapNode : public Node {
 public:
  void VisitAttrs(AttrVisitor* visitor) final {
    // Visitor to map have no effect.
  }

  /*! \brief The corresponding container type */
  using ContainerType = std::unordered_map<
      NodePtr<Node>,
      NodePtr<Node>,
      NodePtrHash, NodePtrEqual>;

  /*! \brief the data content */
  ContainerType data;

  static constexpr const char* _type_key = "Map";
  TVM_DECLARE_NODE_TYPE_INFO(MapNode, Node);
};


/*! \brief specialized map node with string as key */
class StrMapNode : public Node {
 public:
  void VisitAttrs(AttrVisitor* visitor) final {
    // Visitor to map have no effect.
  }
  /*! \brief The corresponding conatiner type */
  using ContainerType = std::unordered_map<
      std::string,
      NodePtr<Node> >;

  /*! \brief the data content */
  ContainerType data;

  static constexpr const char* _type_key = "StrMap";
  TVM_DECLARE_NODE_TYPE_INFO(StrMapNode, Node);
};

/*!
 * \brief iterator adapter that adapts TIter to return another type.
 * \tparam Converter a struct that contains converting function
 * \tparam TIter the content iterator type.
 */
template<typename Converter,
    typename TIter>
class IterAdapter {
 public:
  explicit IterAdapter(TIter iter) : iter_(iter) {}
  inline IterAdapter& operator++() {  // NOLINT(*)
    ++iter_;
    return *this;
  }
  inline IterAdapter& operator++(int) {  // NOLINT(*)
    ++iter_;
    return *this;
  }
  inline IterAdapter operator+(int offset) const {  // NOLINT(*)
    return IterAdapter(iter_ + offset);
  }
  inline bool operator==(IterAdapter other) const {
    return iter_ == other.iter_;
  }
  inline bool operator!=(IterAdapter other) const {
    return !(*this == other);
  }
  inline const typename Converter::ResultType operator*() const {
    return Converter::convert(*iter_);
  }

 private:
  TIter iter_;
};

/*!
 * \brief Array container of NodeRef in DSL graph.
 *  Array implements copy on write semantics, which means array is mutable
 *  but copy will happen when array is referenced in more than two places.
 *
 * operator[] only provide const acces, use Set to mutate the content.
 * \tparam T The content NodeRef type.
 */
template<typename T,
    typename = typename std::enable_if<std::is_base_of<NodeRef, T>::value>::type >
class Array : public NodeRef {
 public:
  /*!
   * \brief default constructor
   */
  Array() {
    node_ = make_node<ArrayNode>();
  }
  /*!
   * \brief move constructor
   * \param other source
   */
  Array(Array<T> && other) {  // NOLINT(*)
    node_ = std::move(other.node_);
  }
  /*!
   * \brief copy constructor
   * \param other source
   */
  Array(const Array<T> &other) : NodeRef(other.node_) { // NOLINT(*)
  }
  /*!
   * \brief constructor from pointer
   * \param n the container pointer
   */
  explicit Array(NodePtr<Node> n) : NodeRef(n) {}
  /*!
   * \brief constructor from iterator
   * \param begin begin of iterator
   * \param end end of iterator
   * \tparam IterType The type of iterator
   */
  template<typename IterType>
  Array(IterType begin, IterType end) {
    assign(begin, end);
  }
  /*!
   * \brief constructor from initializer list
   * \param init The initializer list
   */
  Array(std::initializer_list<T> init) { // NOLINT(*)
    assign(init.begin(), init.end());
  }
  /*!
   * \brief constructor from vector
   * \param init The vector
   */
  Array(const std::vector<T>& init) { // NOLINT(*)
    assign(init.begin(), init.end());
  }
  /*!
   * \brief Constructs a container with n elements. Each element is a copy of val
   * \param n The size of the container
   * \param val The init value
   */
  explicit Array(size_t n, const T& val) {
    auto tmp_node = make_node<ArrayNode>();
    for (size_t i = 0; i < n; ++i) {
      tmp_node->data.push_back(val.node_);
    }
    node_ = std::move(tmp_node);
  }
  /*!
   * \brief move assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Array<T>& operator=(Array<T> && other) {
    node_ = std::move(other.node_);
    return *this;
  }
  /*!
   * \brief copy assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Array<T>& operator=(const Array<T> & other) {
    node_ = other.node_;
    return *this;
  }
  /*!
   * \brief reset the array to content from iterator.
   * \param begin begin of iterator
   * \param end end of iterator
   * \tparam IterType The type of iterator
   */
  template<typename IterType>
  void assign(IterType begin, IterType end) {
    auto n = make_node<ArrayNode>();
    for (IterType it = begin; it != end; ++it) {
      n->data.push_back((*it).node_);
    }
    node_ = std::move(n);
  }
  /*!
   * \brief Read i-th element from array.
   * \param i The index
   * \return the i-th element.
   */
  inline const T operator[](size_t i) const {
    return T(static_cast<const ArrayNode*>(node_.get())->data[i]);
  }
  /*! \return The size of the array */
  inline size_t size() const {
    if (node_.get() == nullptr) return 0;
    return static_cast<const ArrayNode*>(node_.get())->data.size();
  }
  /*!
   * \brief copy on write semantics
   *  Do nothing if current handle is the unique copy of the array.
   *  Otherwise make a new copy of the array to ensure the current handle
   *  hold a unique copy.
   *
   * \return Handle to the internal node container(which ganrantees to be unique)
   */
  inline ArrayNode* CopyOnWrite() {
    if (node_.get() == nullptr || !node_.unique())  {
      NodePtr<ArrayNode> n = make_node<ArrayNode>();
      n->data = static_cast<ArrayNode*>(node_.get())->data;
      NodePtr<Node>(std::move(n)).swap(node_);
    }
    return static_cast<ArrayNode*>(node_.get());
  }
  /*!
   * \brief push a new item to the back of the list
   * \param item The item to be pushed.
   */
  inline void push_back(const T& item) {
    ArrayNode* n = this->CopyOnWrite();
    n->data.push_back(item.node_);
  }
  /*!
   * \brief set i-th element of the array.
   * \param i The index
   * \param value The value to be setted.
   */
  inline void Set(size_t i, const T& value) {
    ArrayNode* n = this->CopyOnWrite();
    n->data[i] = value.node_;
  }
  /*!
   * \brief Get the index of the first occurrence of a value.
   * \param value The value to search
   * \return idx The index. Returns std::string::npos if not find
   */
  inline size_t Index(const T& value) const {
    const std::vector<NodePtr<Node> > &data = static_cast<const ArrayNode*>(node_.get())->data;
    for (size_t i = 0; i < data.size(); ++i) {
      if (data[i] == value.node_) {
        return i;
      }
    }
    return std::string::npos;
  }
  /*!
   * \brief Remove the first occurrence of a value.
   * \param value The value to remove
   * \return idx The index of the elements deleted. Returns -1 if not find
   */
  size_t Remove(const T& value) {
    const std::vector<NodePtr<Node> > &old_data = static_cast<const ArrayNode*>(node_.get())->data;
    auto tmp_node = make_node<ArrayNode>();
    size_t find_idx = std::string::npos;

    size_t i = 0;
    for (i = 0; i < old_data.size(); ++i) {
      if (old_data[i] == value.node_) {
        find_idx = i;
        break;
      }
      tmp_node->data.push_back(old_data[i]);
    }
    for (i = i + 1; i < old_data.size(); ++i) {
      tmp_node->data.push_back(old_data[i]);
    }

    node_ = std::move(tmp_node);
    return find_idx;
  }

  /*! \return whether array is empty */
  inline bool empty() const {
    return size() == 0;
  }
  /*! \brief specify container node */
  using ContainerType = ArrayNode;

  struct Ptr2NodeRef {
    using ResultType = T;
    static inline T convert(const NodePtr<Node>& n) {
      return T(n);
    }
  };
  using iterator = IterAdapter<Ptr2NodeRef,
                               std::vector<NodePtr<Node> >::const_iterator>;

  using reverse_iterator = IterAdapter<
      Ptr2NodeRef,
      std::vector<NodePtr<Node> >::const_reverse_iterator>;

  /*! \return begin iterator */
  inline iterator begin() const {
    return iterator(static_cast<const ArrayNode*>(node_.get())->data.begin());
  }
  /*! \return end iterator */
  inline iterator end() const {
    return iterator(static_cast<const ArrayNode*>(node_.get())->data.end());
  }
  /*! \return rbegin iterator */
  inline reverse_iterator rbegin() const {
    return reverse_iterator(static_cast<const ArrayNode*>(node_.get())->data.rbegin());
  }
  /*! \return rend iterator */
  inline reverse_iterator rend() const {
    return reverse_iterator(static_cast<const ArrayNode*>(node_.get())->data.rend());
  }
};

template<typename T,
    typename = typename std::enable_if<std::is_base_of<NodeRef, T>::value>::type >
class Set : public NodeRef {
 public:
  /*!
   * \brief default constructor
   */
  Set() {
    node_ = make_node<SetNode>();
  }
  /*!
   * \brief move constructor
   * \param other source
   */
  Set(Set<T> && other) {  // NOLINT(*)
    node_ = std::move(other.node_);
  }
  /*!
   * \brief copy constructor
   * \param other source
   */
  Set(const Set<T> &other) : NodeRef(other.node_) { // NOLINT(*)
  }
  /*!
   * \brief constructor from pointer
   * \param n the container pointer
   */
  explicit Set(NodePtr<Node> n) : NodeRef(n) {}
  /*!
   * \brief constructor from iterator
   * \param begin begin of iterator
   * \param end end of iterator
   * \tparam IterType The type of iterator
   */
  template<typename IterType>
  Set(IterType begin, IterType end) {
    assign(begin, end);
  }
  /*!
   * \brief constructor from initializer list
   * \param init The initializer list
   */
  Set(std::initializer_list<T> init) { // NOLINT(*)
    assign(init.begin(), init.end());
  }
  /*!
   * \brief constructor from vector
   * \param init The set
   */
  template<typename Hash, typename Equal>
  Set(const std::unordered_set<T, Hash, Equal>& init) { // NOLINT(*)
    assign(init.begin(), init.end());
  }
  /*!
   * \brief move assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Set<T>& operator=(Set<T> && other) {
    node_ = std::move(other.node_);
    return *this;
  }
  /*!
   * \brief copy assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Set<T>& operator=(const Set<T> & other) {
    node_ = other.node_;
    return *this;
  }
  /*!
   * \brief reset the array to content from iterator.
   * \param begin begin of iterator
   * \param end end of iterator
   * \tparam IterType The type of iterator
   */
  template<typename IterType>
  void assign(IterType begin, IterType end) {
    auto n = make_node<SetNode>();
    for (IterType it = begin; it != end; ++it) {
      n->data.insert((*it).node_);
    }
    node_ = std::move(n);
  }

  /*! \return The size of the array */
  inline size_t size() const {
    if (node_.get() == nullptr) return 0;
    return static_cast<const SetNode*>(node_.get())->data.size();
  }
  /*!
   * \brief copy on write semantics
   *  Do nothing if current handle is the unique copy of the array.
   *  Otherwise make a new copy of the array to ensure the current handle
   *  hold a unique copy.
   *
   * \return Handle to the internal node container(which ganrantees to be unique)
   */
  inline SetNode* CopyOnWrite() {
    if (node_.get() == nullptr || !node_.unique())  {
      NodePtr<SetNode> n = make_node<SetNode>();
      n->data = static_cast<SetNode*>(node_.get())->data;
      NodePtr<Node>(std::move(n)).swap(node_);
    }
    return static_cast<SetNode*>(node_.get());
  }
  /*!
   * \brief insert a value to set
   * \param value The value to insert
   * */
  inline void insert(const T& value) {
    SetNode* n = CopyOnWrite();
    n->data.insert(value.node_);
  }

  inline void insert(const Set<T>& other) {
    SetNode* n = CopyOnWrite();
    const SetNode* other_n = static_cast<const SetNode*>(other.node_.get());

    n->data.insert(other_n->data.begin(), other_n->data.end());
  }

  inline bool IsSubset(const Set<T>& other) const {
    for (const auto &x : *this) {
      if (!other.count(x)) {
        return false;
      }
    }
    return true;
  }
  inline bool IsSuperset(const Set<T>& other) const {
    return other.IsSubset(*this);
  }

  Set<T> Intersection(const Set<T>& other) const {
    NodePtr<SetNode> n = make_node<SetNode>();
    for (const auto &x : *this) {
      if (other.count(x)) {
        n->data.insert(x.node_);
      }
    }
    return Set<T>(n);
  }

  Set<T> Difference(const Set<T>& other) const {
    NodePtr<SetNode> n = make_node<SetNode>();
    for (const auto &x : *this) {
      if (!other.count(x)) {
        n->data.insert(x.node_);
      }
    }
    return Set<T>(n);
  }

  Set<T> Union(const Set<T>& other) const {
    NodePtr<SetNode> n = make_node<SetNode>();
    const SetNode* a = static_cast<const SetNode*>(node_.get());
    const SetNode* b = static_cast<const SetNode*>(other.node_.get());
    n->data.insert(a->data.begin(), a->data.end());
    n->data.insert(b->data.begin(), b->data.end());
    return Set<T>(n);
  }

  /*! \return whether array is empty */
  inline bool empty() const {
    return size() == 0;
  }
  /*! \return The number of elements of the key */
  inline int count(const T& value) const {
    return static_cast<const SetNode*>(node_.get())->data.count(value.node_);
  }

  /*! \brief specify container node */
  using ContainerType = SetNode;

  struct Ptr2NodeRef {
    using ResultType = T;
    static inline T convert(const NodePtr<Node>& n) {
      return T(n);
    }
  };
  using iterator = IterAdapter<Ptr2NodeRef,
                               SetNode::ContainerType::const_iterator>;

  /*! \return begin iterator */
  inline iterator begin() const {
    return iterator(static_cast<const SetNode*>(node_.get())->data.begin());
  }
  /*! \return end iterator */
  inline iterator end() const {
    return iterator(static_cast<const SetNode*>(node_.get())->data.end());
  }
};

/*!
 * \brief Map container of NodeRef->NodeRef in DSL graph.
 *  Map implements copy on write semantics, which means map is mutable
 *  but copy will happen when array is referenced in more than two places.
 *
 * operator[] only provide const acces, use Set to mutate the content.
 * \tparam K The key NodeRef type.
 * \tparam V The value NodeRef type.
 */
template<typename K,
    typename V,
    typename = typename std::enable_if<
        std::is_base_of<NodeRef, K>::value ||
            std::is_base_of<std::string, K>::value >::type,
    typename = typename std::enable_if<std::is_base_of<NodeRef, V>::value>::type>
class Map : public NodeRef {
 public:
  /*!
   * \brief default constructor
   */
  Map() {
    node_ = make_node<MapNode>();
  }
  /*!
   * \brief move constructor
   * \param other source
   */
  Map(Map<K, V> && other) {  // NOLINT(*)
    node_ = std::move(other.node_);
  }
  /*!
   * \brief copy constructor
   * \param other source
   */
  Map(const Map<K, V> &other) : NodeRef(other.node_) { // NOLINT(*)
  }
  /*!
   * \brief constructor from pointer
   * \param n the container pointer
   */
  explicit Map(NodePtr<Node> n) : NodeRef(n) {}
  /*!
   * \brief constructor from iterator
   * \param begin begin of iterator
   * \param end end of iterator
   * \tparam IterType The type of iterator
   */
  template<typename IterType>
  Map(IterType begin, IterType end) {
    assign(begin, end);
  }
  /*!
   * \brief constructor from initializer list
   * \param init The initalizer list
   */
  Map(std::initializer_list<std::pair<K, V> > init) { // NOLINT(*)
    assign(init.begin(), init.end());
  }
  /*!
   * \brief constructor from unordered_map
   * \param init The unordered_map
   */
  template<typename Hash, typename Equal>
  Map(const std::unordered_map<K, V, Hash, Equal>& init) { // NOLINT(*)
    assign(init.begin(), init.end());
  }
  /*!
   * \brief move assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Map<K, V>& operator=(Map<K, V> && other) {
    node_ = std::move(other.node_);
    return *this;
  }
  /*!
   * \brief copy assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Map<K, V>& operator=(const Map<K, V> & other) {
    node_ = other.node_;
    return *this;
  }
  /*!
   * \brief reset the array to content from iterator.
   * \param begin begin of iterator
   * \param end end of iterator
   * \tparam IterType The type of iterator
   */
  template<typename IterType>
  void assign(IterType begin, IterType end) {
    NodePtr<MapNode> n = make_node<MapNode>();
    for (IterType i = begin; i != end; ++i) {
      n->data.emplace(std::make_pair(i->first.node_,
                                     i->second.node_));
    }
    node_ = std::move(n);
  }
  /*!
   * \brief Read element from map.
   * \param key The key
   * \return the corresonding element.
   */
  inline const V operator[](const K& key) const {
    return V(static_cast<const MapNode*>(node_.get())->data.at(key.node_));
  }
  /*!
   * \brief Read element from map.
   * \param key The key
   * \return the corresonding element.
   */
  inline const V at(const K& key) const {
    return V(static_cast<const MapNode*>(node_.get())->data.at(key.node_));
  }
  /*! \return The size of the array */
  inline size_t size() const {
    if (node_.get() == nullptr) return 0;
    return static_cast<const MapNode*>(node_.get())->data.size();
  }
  /*! \return The number of elements of the key */
  inline size_t count(const K& key) const {
    if (node_.get() == nullptr) return 0;
    return static_cast<const MapNode*>(node_.get())->data.count(key.node_);
  }
  /*!
   * \brief copy on write semantics
   *  Do nothing if current handle is the unique copy of the array.
   *  Otherwise make a new copy of the array to ensure the current handle
   *  hold a unique copy.
   *
   * \return Handle to the internal node container(which ganrantees to be unique)
   */
  inline MapNode* CopyOnWrite() {
    if (node_.get() == nullptr || !node_.unique())  {
      NodePtr<MapNode> n = make_node<MapNode>();
      n->data = static_cast<const MapNode*>(node_.get())->data;
      NodePtr<Node>(std::move(n)).swap(node_);
    }
    return static_cast<MapNode*>(node_.get());
  }
  /*!
   * \brief set the Map.
   * \param key The index key.
   * \param value The value to be set.
   */
  inline void Set(const K& key, const V& value) {
    MapNode* n = this->CopyOnWrite();
    n->data[key.node_] = value.node_;
  }
  /*!
   * \brief Remove an element
   * \param key The key of the element to remove
   */
  inline void erase(const K& key) {
    MapNode* n = this->CopyOnWrite();
    n->data.erase(key.node_);
  }

  /*! \return whether array is empty */
  inline bool empty() const {
    return size() == 0;
  }
  /*! \brief specify container node */
  using ContainerType = MapNode;

  struct Ptr2NodeRef {
    using ResultType = std::pair<K, V>;
    static inline ResultType convert(const std::pair<
        NodePtr<Node>,
        NodePtr<Node> >& n) {
      return std::make_pair(K(n.first), V(n.second));
    }
  };

  using iterator = IterAdapter<
      Ptr2NodeRef, MapNode::ContainerType::const_iterator>;

  /*! \return begin iterator */
  inline iterator begin() const {
    return iterator(static_cast<const MapNode*>(node_.get())->data.begin());
  }
  /*! \return end iterator */
  inline iterator end() const {
    return iterator(static_cast<const MapNode*>(node_.get())->data.end());
  }
  /*! \return begin iterator */
  inline iterator find(const K& key) const {
    return iterator(static_cast<const MapNode*>(node_.get())->data.find(key.node_));
  }
};

// specialize of string map
template<typename V, typename T1, typename T2>
class Map<std::string, V, T1, T2> : public NodeRef {
 public:
  // for code reuse
  Map() {
    node_ = make_node<StrMapNode>();
  }
  Map(Map<std::string, V> && other) {  // NOLINT(*)
    node_ = std::move(other.node_);
  }
  Map(const Map<std::string, V> &other) : NodeRef(other.node_) { // NOLINT(*)
  }
  explicit Map(NodePtr<Node> n) : NodeRef(n) {}
  template<typename IterType>
  Map(IterType begin, IterType end) {
    assign(begin, end);
  }
  Map(std::initializer_list<std::pair<std::string, V> > init) { // NOLINT(*)
    assign(init.begin(), init.end());
  }

  template<typename Hash, typename Equal>
  Map(const std::unordered_map<std::string, V, Hash, Equal>& init) { // NOLINT(*)
    assign(init.begin(), init.end());
  }
  Map<std::string, V>& operator=(Map<std::string, V> && other) {
    node_ = std::move(other.node_);
    return *this;
  }
  Map<std::string, V>& operator=(const Map<std::string, V> & other) {
    node_ = other.node_;
    return *this;
  }
  template<typename IterType>
  void assign(IterType begin, IterType end) {
    auto n = make_node<StrMapNode>();
    for (IterType i = begin; i != end; ++i) {
      n->data.emplace(std::make_pair(i->first,
                                     i->second.node_));
    }
    node_ = std::move(n);
  }
  inline const V operator[](const std::string& key) const {
    return V(static_cast<const StrMapNode*>(node_.get())->data.at(key));
  }
  inline const V at(const std::string& key) const {
    return V(static_cast<const StrMapNode*>(node_.get())->data.at(key));
  }
  inline size_t size() const {
    if (node_.get() == nullptr) return 0;
    return static_cast<const StrMapNode*>(node_.get())->data.size();
  }
  inline size_t count(const std::string& key) const {
    if (node_.get() == nullptr) return 0;
    return static_cast<const StrMapNode*>(node_.get())->data.count(key);
  }
  inline StrMapNode* CopyOnWrite() {
    if (node_.get() == nullptr || !node_.unique())  {
      NodePtr<StrMapNode> n = make_node<StrMapNode>();
      n->data = static_cast<const StrMapNode*>(node_.get())->data;
      NodePtr<Node>(std::move(n)).swap(node_);
    }
    return static_cast<StrMapNode*>(node_.get());
  }
  inline void Set(const std::string& key, const V& value) {
    StrMapNode* n = this->CopyOnWrite();
    n->data[key] = value.node_;
  }
  inline bool empty() const {
    return size() == 0;
  }
  using ContainerType = StrMapNode;

  struct Ptr2NodeRef {
    using ResultType = std::pair<std::string, V>;
    static inline ResultType convert(const std::pair<
        std::string,
        NodePtr<Node> >& n) {
      return std::make_pair(n.first, V(n.second));
    }
  };

  using iterator = IterAdapter<
      Ptr2NodeRef, StrMapNode::ContainerType::const_iterator>;

  /*! \return begin iterator */
  inline iterator begin() const {
    return iterator(static_cast<const StrMapNode*>(node_.get())->data.begin());
  }
  /*! \return end iterator */
  inline iterator end() const {
    return iterator(static_cast<const StrMapNode*>(node_.get())->data.end());
  }
  /*! \return begin iterator */
  inline iterator find(const std::string& key) const {
    return iterator(static_cast<const StrMapNode*>(node_.get())->data.find(key));
  }
};

}  // namespace tvm
#endif  // TVM_NODE_CONTAINER_H_
