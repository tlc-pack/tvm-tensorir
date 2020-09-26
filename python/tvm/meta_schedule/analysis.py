# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Meta schedule analysis API """
from typing import List, Optional, Tuple

from tvm.ir import Op
from tvm.tir import BufferLoad, IterVar, Var

from . import _ffi_api_analysis
from .random_variable import BlockRV
from .schedule import Schedule


def is_trivial_binding(sch: Schedule, block: BlockRV) -> bool:
    """Checks if
    1) number of blocks vars equals to number of loop vars
    2) each block var is bound to a loop var directly
    3) the order is preserved, i.e. the i-th block var is the i-th loop var

    Parameters
    ----------
    sch: Schedule
        The meta schedule class
    block: BlockRV
        The block random variable to be analyzed

    Returns
    -------
    result : bool
        A boolean indicating if the block binding is trivial
    """
    return bool(
        _ffi_api_analysis.IsTrivialBinding(sch, block)  # pylint: disable=no-member
    )


def get_block_var_types(sch: Schedule, block: BlockRV) -> List[str]:
    """Returns the IterVarType of each block var

    Parameters
    ----------
    sch: Schedule
        The meta schedule class
    block: BlockRV
        The block random variable to be analyzed

    Returns
    -------
    types : List[str]
        An array of integers, the IterVarTypes corresponding to each block var in order
    """
    table = {
        IterVar.DataPar: "spatial",
        IterVar.CommReduce: "reduce",
    }
    types = _ffi_api_analysis.GetBlockVarTypes(sch, block)  # pylint: disable=no-member
    return [table.get(iter_type, "opaque") for iter_type in types]


def is_leaf_block(sch: Schedule, block: BlockRV) -> bool:
    """Checks if the specific block is a leaf block

    Parameters
    ----------
    sch: Schedule
        The meta schedule class
    block: BlockRV
        The block random variable to be analyzed

    Returns
    -------
    result : bool
        A boolean indiciating if the block is a leaf block
    """
    return bool(_ffi_api_analysis.IsLeafBlock(sch, block))  # pylint: disable=no-member


def is_leaf_block_with_single_stmt(sch: Schedule, block: BlockRV) -> bool:
    """Checks if the specific block is a leaf block and its body is a single statement

    Parameters
    ----------
    sch: Schedule
        The meta schedule class
    block: BlockRV
        The block random variable to be analyzed

    Returns
    -------
    result : bool
        A boolean indiciating if the block is a leaf block and its body is a single statement
    """
    return bool(
        _ffi_api_analysis.IsLeafBlockWithSingleStmt(  # pylint: disable=no-member
            sch,
            block,
        )
    )


def get_buffer_store(sch: Schedule, block: BlockRV) -> BufferLoad:
    """Get the buffer written in the single statement of a leaf statement

    Parameters
    ----------
    sch: Schedule
        The meta schedule class
    block: BlockRV
        The block random variable to be analyzed

    Returns
    -------
    buffer_store : BufferLoad
        A BufferLoad indicating the buffer and its indices to be written
        It is intended to return type BufferLoad, because it has included all necessary info
    """
    return _ffi_api_analysis.GetBufferStore(sch, block)  # pylint: disable=no-member


def get_buffer_load(sch: Schedule, block: BlockRV) -> List[BufferLoad]:
    """Get all the buffers read in the single statement of a leaf statement

    Parameters
    ----------
    sch: Schedule
        The meta schedule class
    block: BlockRV
        The block random variable to be analyzed

    Returns
    -------
    buffer_store : List[BufferLoad]
        An array of BufferLoad indicating the buffers and their indices to be read
    """
    return _ffi_api_analysis.GetBufferLoad(sch, block)  # pylint: disable=no-member


def count_op(sch: Schedule, block: BlockRV, op: Op) -> int:
    """Count the number of occurrence of an operator, i.e. tir.exp

    Parameters
    ----------
    sch: Schedule
        The meta schedule class
    block: BlockRV
        The block random variable to be analyzed
    op: Op
        The operator to be counted

    Returns
    -------
    counts : int
        An integer indicating the number of its occurrence
    """
    return _ffi_api_analysis.CountOp(sch, block, op)  # pylint: disable=no-member


def has_branch(sch: Schedule, block: BlockRV) -> bool:
    """Check if there is any branch in the given block, which includes
    1) block predicate
    2) if-then-else statement
    3) select expression
    4) if-then-else operator

    Parameters
    ----------
    sch: Schedule
        The meta schedule class
    block: BlockRV
        The block random variable to be analyzed

    Returns
    -------
    result : bool
        A boolean indicating there is at least a branch in the given block
    """
    return bool(_ffi_api_analysis.HasBranch(sch, block))  # pylint: disable=no-member


def block_vars_used_in_store(sch: Schedule, block: BlockRV) -> Optional[List[Var]]:
    """Check if the specifc block satisfies
    1) it is a leaf block with a single statement as its body
    2) indices in BufferStore are either constants, or block vars +/- constants
    If condition is satisfied, return an array of block vars
    that are used in BufferStore indices in the same order as they appears in indices

    Parameters
    ----------
    sch: Schedule
        The meta schedule class
    block: BlockRV
        The block random variable to be analyzed

    Returns
    -------
    block_vars : Optional[List[Var]]
        An array of block vars, in the same order as they appears in indices,
        if the condition is satisfied; None otherwise
    """
    return _ffi_api_analysis.BlockVarsUsedInStore(  # pylint: disable=no-member
        sch, block
    )


def count_missing_block_vars(load: BufferLoad, block_vars: List[Var]) -> int:
    """Count the number of block vars that are not used in the BufferLoad

    Parameters
    ----------
    load: BufferLoad
        The BufferLoad to be examined
    block_vars: List[Var]
        The list of block vars

    Returns
    -------
    n_missing : int
        An integer indicating number of block vars that are not used
    """
    return _ffi_api_analysis.CountMissingBlockVars(  # pylint: disable=no-member
        load, block_vars
    )


def inspect_load_indices(
    sch: Schedule, block: BlockRV
) -> Optional[Tuple[bool, bool, bool]]:
    """Inspect the mapping between indices in all BufferLoads and block vars used in BufferStore.
    First, call `BlockVarsUsedInStore` to get block vars.
    Second, for each BufferLoad and its indices, check
        1) exists: the mapping from load -> block vars exists
        2) surjective: every block var is mapped to at least once
        3) injective: every block var is mapped to at most once
        4) order: the mapping is kept in order
    If the mapping doesn't exist, then return NullOpt;
    Otherwise, return (surjective, injective, order)

    Parameters
    ----------
    sch: Schedule
        The meta schedule class
    block: BlockRV
        The block random variable to be analyzed

    Returns
    -------
    (surjective, injective, order) : Optional[Tuple[bool, bool, bool]]
        None: the mapping does not exist
        surjective: every block var is mapped to at least once
        injective: every block var is mapped to at most once
        order: the mapping is kept in order
    """
    result = _ffi_api_analysis.InspectLoadIndices(  # pylint: disable=no-member
        sch, block
    )
    if result is None:
        return result
    surjective, injective, ordered = result
    return bool(surjective), bool(injective), bool(ordered)
