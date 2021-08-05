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
#include "../utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Create a new block with the given annotation added
 * \param block The block with original annotation
 * \param attr_key The annotation key to be added
 * \param attr_value The annotation value to be added
 * \return A new block with the given annotation as its last annotation
 */
Block WithAnnotation(const BlockNode* block, const String& attr_key, const ObjectRef& attr_value) {
  Map<String, ObjectRef> annotations = block->annotations;
  annotations.Set(attr_key, attr_value);
  ObjectPtr<BlockNode> new_block = make_object<BlockNode>(*block);
  new_block->annotations = std::move(annotations);
  return Block(new_block);
}

class StorageAlignAxisOutOfRangeError : public ScheduleError {
 public:
  explicit StorageAlignAxisOutOfRangeError(IRModule mod, Buffer buffer, int axis)
      : mod_(std::move(mod)), buffer_(std::move(buffer)), axis_(axis) {}

  String FastErrorString() const final {
    return "ScheduleError: The input `axis` is out of range. It is required to be in range "
           "[-ndim, ndim) where `ndim` is the number of dimensions of the buffer to set "
           "storage alignment.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    int ndim = static_cast<int>(buffer_->shape.size());
    os << "The buffer to set storage alignment " << buffer_->name << " has " << ndim
       << " dimension(s), so `axis` is required to be in [" << -(ndim) << ", " << ndim
       << ") for storage_align. However, the input `axis` is " << axis_
       << ", which is out of the expected range.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

  static int CheckAndUpdate(const IRModule& mod, const Buffer& buffer, int axis) {
    int ndim = static_cast<int>(buffer->shape.size());
    if (axis < -ndim || axis >= ndim) {
      throw StorageAlignAxisOutOfRangeError(mod, buffer, axis);
    }
    // If axis is negative, convert it to a non-negative one.
    if (axis < 0) {
      axis += ndim;
    }
    return axis;
  }

 private:
  IRModule mod_;
  Buffer buffer_;
  int axis_;
};

class WriteBufferIndexOutOfRangeError : public ScheduleError {
 public:
  explicit WriteBufferIndexOutOfRangeError(IRModule mod, Block block, int buffer_index)
      : mod_(std::move(mod)), block_(std::move(block)), buffer_index_(buffer_index) {}

  String FastErrorString() const final {
    return "ScheduleError: The input `buffer_index` is out of range. It is required to be in range "
           "[0, num_write_regions) where `num_write_regions` is the number of buffer regions "
           "written by the block.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    size_t num_writes = block_->writes.size();
    os << "The block {0} has " << num_writes
       << " write regions, so `buffer_index` is required to be in [0, "
       << num_writes << "). However, the input `buffer_index` is " << buffer_index_
       << ", which is out of the expected range";
    return os.str();
  }

  static Buffer CheckAndGetBuffer(const IRModule& mod, const Block& block, int buffer_index) {
    if (buffer_index < 0 || buffer_index > block->writes.size()) {
      throw WriteBufferIndexOutOfRangeError(mod, block, buffer_index);
    }
    return block->writes[buffer_index]->buffer;
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {mod_}; }

 private:
  IRModule mod_;
  Block block_;
  int buffer_index_;
};

/*!
 * \brief Find the defining site of the buffer in the given block and its ancestors
 * \param block_sref The block sref
 * \param buffer The buffer
 * \return The defining site of the buffer and whether the buffer is allocated (otherwise the
 *         buffer is from match_buffer).
 */
std::pair<StmtSRef, bool> GetBufferDefiningSite(const StmtSRef& block_sref, const Buffer& buffer) {
  // Climb up along the sref tree, and find the block where `buffer` is in alloc_buffers or
  // match_buffers.
  const StmtSRefNode* defining_site_sref = block_sref.get();
  while (defining_site_sref != nullptr) {
    const auto* block = defining_site_sref->StmtAs<BlockNode>();
    // If this sref is not a block sref, skip it.
    if (block == nullptr) {
      defining_site_sref = defining_site_sref->parent;
      continue;
    }
    // Try to find the buffer in `allloc_buffers`
    for (const Buffer& alloc_buffer : block->alloc_buffers) {
      if (buffer.same_as(alloc_buffer)) {
        return {GetRef<StmtSRef>(defining_site_sref), true};
      }
    }
    // We do not allow the buffer being defined in `match_buffer`.
    for (const MatchBufferRegion match_buffer : block->match_buffers) {
      if (buffer.same_as(match_buffer)) {
        return {GetRef<StmtSRef>(defining_site_sref), false};
      }
    }
    defining_site_sref = defining_site_sref->parent;
  }
  // If we cannot find the defining site block, it means that the buffer must be in the function's
  // buffer_map, which isn't an intermediate buffer.
  return {StmtSRef(), false};
}

class NonAllocatedBufferError : public ScheduleError {
 public:
  explicit NonAllocatedBufferError(IRModule mod, Buffer buffer) : mod_(mod), buffer_(buffer) {}

  String FastErrorString() const final {
    return "ScheduleError: The input buffer is not allocated by a block. This means the buffer is "
           " either a function parameter or defined in `match_buffer` of a block.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The input buffer " << buffer_->name
       << " is not allocated by a block. This means the buffer is either a function parameter or "
          "defined in `match_buffer` of a block.";
    return os.str();
  }

  static void CheckBufferAllocated(const IRModule& mod, const StmtSRef& block_sref,
                                   const Buffer& buffer) {
    StmtSRef defining_site_sref;
    bool is_alloc;
    std::tie(defining_site_sref, is_alloc) = GetBufferDefiningSite(block_sref, buffer);
    if (!defining_site_sref.defined() || !is_alloc) {
      throw NonAllocatedBufferError(mod, buffer);
    }
  }

  Array<ObjectRef> LocationsOfInterest() const final { return {}; }
  IRModule mod() const final { return mod_; }

 private:
  IRModule mod_;
  Buffer buffer_;
};

class StorageAlignInvalidFactorError : public ScheduleError {
 public:
  explicit StorageAlignInvalidFactorError(const IRModule& mod, int factor)
      : mod_(std::move(mod)), factor_(factor) {}

  String FastErrorString() const final {
    return "ScheduleError: The input `factor` of storage_align is expected to be a positive "
           "number.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The input `factor` of storage_align is expected to be a positive number. However, the "
          "input `factor` is "
       << factor_ << ", which is out of the expected range.";
    return os.str();
  }

  static void Check(const IRModule& mod, int factor) {
    if (factor <= 0) {
      throw StorageAlignInvalidFactorError(mod, factor);
    }
  }

  Array<ObjectRef> LocationsOfInterest() const final { return {}; }
  IRModule mod() const final { return mod_; }

 private:
  IRModule mod_;
  int factor_;
};

void StorageAlign(ScheduleState self, const StmtSRef& block_sref, int buffer_index, int axis,
                  int factor, int offset) {
  const BlockNode* block_ptr = TVM_SREF_TO_BLOCK(block_ptr, block_sref);
  Buffer buffer = WriteBufferIndexOutOfRangeError::CheckAndGetBuffer(
      self->mod, GetRef<Block>(block_ptr), buffer_index);
  StorageAlignInvalidFactorError::Check(self->mod, factor);
  axis = StorageAlignAxisOutOfRangeError::CheckAndUpdate(self->mod, buffer, axis);
  NonAllocatedBufferError::CheckBufferAllocated(self->mod, block_sref, buffer);

  // Step 1: Get existing or create new annotation value.
  auto it = block_ptr->annotations.find(attr::buffer_dim_align);

  // Use an array to store the storage alignement information for each output tensor.
  // For each output tensor, we use an array of tuples (axis, factor, offset) to specify storage
  // alignment for each dimension.
  Array<Array<Array<Integer>>> storage_align_annotation;

  if (it != block_ptr->annotations.end()) {
    storage_align_annotation = Downcast<Array<Array<Array<Integer>>>>((*it).second);
    ICHECK(storage_align_annotation.size() == block_ptr->writes.size());
  } else {
    storage_align_annotation.resize(block_ptr->writes.size());
  }

  // Step 2: Update the annotation value
  Array<Array<Integer>> buffer_storage_align = storage_align_annotation[buffer_index];
  bool found = false;
  for (size_t j = 0; j < buffer_storage_align.size(); ++j) {
    ICHECK(buffer_storage_align[j].size() == 3);
    if (buffer_storage_align[j][0] == axis) {
      buffer_storage_align.Set(j, {Integer(axis), Integer(factor), Integer(offset)});
      found = true;
      break;
    }
  }
  if (!found) {
    buffer_storage_align.push_back({Integer(axis), Integer(factor), Integer(offset)});
  }
  storage_align_annotation.Set(buffer_index, std::move(buffer_storage_align));

  // Step 3: Replace the block with the new annotation
  Block new_block = WithAnnotation(block_ptr, attr::buffer_dim_align, storage_align_annotation);
  self->Replace(block_sref, new_block, {{GetRef<Block>(block_ptr), new_block}});
}

}  // namespace tir
}  // namespace tvm
