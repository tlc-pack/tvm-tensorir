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
from typing import List, Optional

from tvm import tir
from tvm.ir import Op

from . import _ffi_api_analysis
from .instruction import BlockRV
from .schedule import Schedule


def is_trivial_binding(sch: tir.Schedule, block: tir.StmtSRef) -> bool:
    """Checks if
    1) number of blocks vars equals to number of loop vars
    2) each block var is bound to a loop var directly
    3) the order is preserved, i.e. the i-th block var is the i-th loop var

    Parameters
    ----------
    sch: tir.Schedule
        The TIR schedule class
    block: tir.StmtSRef
        The block to be analyzed

    Returns
    -------
    result : bool
        A boolean indicating if the block binding is trivial
    """
    return bool(_ffi_api_analysis.IsTrivialBinding(sch, block))  # pylint: disable=no-member


def is_subroot_block(sch: tir.Schedule, block: tir.StmtSRef) -> bool:
    """Check if a block is the direct children of the root block

    Parameters
    ----------
    sch: tir.Schedule
        The TIR schedule class
    block: tir.StmtSRef
        The block to be analyzed

    Returns
    -------
    result : bool
        A boolean flag indicating if the block is the subroot block
    """
    return bool(_ffi_api_analysis.IsSubrootBlock(sch, block))  # pylint: disable=no-member


def is_leaf_block(sch: tir.Schedule, block: tir.StmtSRef) -> bool:
    """Check if a block has no child block

    Parameters
    ----------
    sch: tir.Schedule
        The TIR schedule class
    block: tir.StmtSRef
        The block to be analyzed

    Returns
    -------
    result : bool
        A boolean flag indicating if the block is a leaf block
    """
    return bool(_ffi_api_analysis.IsLeafBlock(sch, block))  # pylint: disable=no-member


def annotate_loop_type(sch: tir.Schedule, loops: List[tir.StmtSRef], annotation: str) -> None:
    """Annotate the specific loops with the given loop type

    Parameters
    ----------
    sch : tir.Schedule
        The TIR schedule class
    loops : List[tir.StmtSRef]
        The loops to be annotated
    annotation : str
        The loop annotations
    """
    _ffi_api_analysis.AnnotateLoopType(sch, loops, annotation)  # pylint: disable=no-member


def annotate_block_type(sch: tir.Schedule, block: tir.StmtSRef, annotation: str) -> None:
    """Annotate the specific block with the given block type

    Parameters
    ----------
    sch : tir.Schedule
        The TIR schedule class
    block : tir.StmtSRef
        The block to be annotated
    annotation : str
        The block annotations
    """
    _ffi_api_analysis.AnnotateBlockType(sch, block, annotation)  # pylint: disable=no-member


def collect_annotated_loops(sch: tir.Schedule, annotation: str) -> List[List[tir.StmtSRef]]:
    """Collect the loops annotated with each sub-tree

    Parameters
    ----------
    sch : tir.Schedule
        The TIR schedule class
    annotation : str
        The loop annotations

    Returns
    ----------
    result : List[List[tir.StmtSRef]]
        An array containing each chain of annotated loops
    """
    return _ffi_api_analysis.CollectAnnotatedLoops(sch, annotation)  # pylint: disable=no-member


def get_loop_type(sch: tir.Schedule, block: tir.StmtSRef, loops: List[tir.StmtSRef]) -> List[int]:
    """For each loop var by examing its related block var, find its type in one of the following
    IterVarType::kDataPar    = 0
    IterVarType::kCommReduce = 2
    IterVarType::kOpaque     = 4

    Parameters
    ----------
    sch: tir.Schedule
        The TIR schedule class
    block: tir.StmtSRef
        The block to be analyzed
    loops: List[tir.StmtSRef]
        The loops to be analyzed

    Returns
    -------
    result : bool
        An array of the same length as loops_sref
    """
    return _ffi_api_analysis.GetLoopType(sch, block, loops)  # pylint: disable=no-member


def get_block_var_types(sch: tir.Schedule, block: tir.StmtSRef) -> List[str]:
    """Returns the IterVarType of each block var

    Parameters
    ----------
    sch: tir.Schedule
        The TIR schedule class
    block: tir.StmtSRef
        The block to be analyzed

    Returns
    -------
    types : List[str]
        An array of integers, the IterVarTypes corresponding to each block var in order
    """
    table = {
        tir.IterVar.DataPar: "spatial",
        tir.IterVar.CommReduce: "reduce",
    }
    types = _ffi_api_analysis.GetBlockVarTypes(sch, block)  # pylint: disable=no-member
    return [table.get(iter_type, "opaque") for iter_type in types]


def is_spatial(sch: tir.Schedule, block: tir.StmtSRef) -> bool:
    """Check if the iter types of all block vars are data parallel

    Parameters
    ----------
    sch: tir.Schedule
        The TIR schedule class
    block: tir.StmtSRef
        The block to be analyzed

    Returns
    -------
    result : bool
        A boolean indicating if the block is spatial
    """
    return bool(
        _ffi_api_analysis.IsSpatial(  # pylint: disable=no-member
            sch,
            block,
        )
    )


def is_single_stmt_leaf(sch: tir.Schedule, block: tir.StmtSRef) -> bool:
    """Checks if the specific block is a leaf block and its body is a single statement

    Parameters
    ----------
    sch: tir.Schedule
        The TIR schedule class
    block: tir.StmtSRef
        The block to be analyzed

    Returns
    -------
    result : bool
        A boolean indiciating if the block is a leaf block and its body is a single statement
    """
    return bool(
        _ffi_api_analysis.IsSingleStmtLeaf(  # pylint: disable=no-member
            sch,
            block,
        )
    )


def is_output_block(sch: tir.Schedule, block: tir.StmtSRef) -> bool:
    """Checks if a block is output block

    Parameters
    ----------
    sch: tir.Schedule
        The TIR schedule class
    block: tir.StmtSRef
        The block to be analyzed

    Returns
    -------
    result : bool
        A boolean flag indicating if it is an output block
    """
    return bool(_ffi_api_analysis.IsOutputBlock(sch, block))  # pylint: disable=no-member


def count_op(sch: tir.Schedule, block: tir.StmtSRef, op: Op) -> int:
    """Count the number of occurrence of an operator, i.e. tir.exp

    Parameters
    ----------
    sch: tir.Schedule
        The TIR schedule class
    block: tir.StmtSRef
        The block to be analyzed
    op: Op
        The operator to be counted

    Returns
    -------
    counts : int
        An integer indicating the number of its occurrence
    """
    return _ffi_api_analysis.CountOp(sch, block, op)  # pylint: disable=no-member


def has_branch(sch: tir.Schedule, block: tir.StmtSRef) -> int:
    """Check if there is any branch in the given block, which includes
    1) block predicate
    2) if-then-else statement
    3) select expression
    4) if-then-else operator

    Parameters
    ----------
    sch: tir.Schedule
        The TIR schedule class
    block: tir.StmtSRef
        The block to be analyzed

    Returns
    -------
    result : bool
        A boolean indicating there is at least a branch in the given block
    """
    return bool(_ffi_api_analysis.HasBranch(sch, block))  # pylint: disable=no-member


def is_elementwise_match(
    sch: tir.Schedule,
    producer: tir.StmtSRef,
    consumer: tir.StmtSRef,
) -> bool:
    """Checks whether the producer and consumer matches in elementwise way.
    Assuming `consumer` is the only consumer of `producer`.

    Parameters
    ----------
    sch: tir.Schedule
        The TIR class
    producer: tir.StmtSRef
        The producer block
    consumer: tir.StmtSRef
        The consumer block

    Returns
    -------
    result : bool
        A boolean flag indicating if they match
    """
    return bool(
        _ffi_api_analysis.IsElementWiseMatch(sch, producer, consumer)  # pylint: disable=no-member
    )


def needs_multi_level_tiling(sch: tir.Schedule, block: tir.StmtSRef) -> bool:
    """Checks if a block needs multi-level tiling

    Parameters
    ----------
    sch: tir.Schedule
        The TIR class
    block: tir.StmtSRef
        The block to be analyzed

    Returns
    -------
    result : bool
        A boolean flag indicating if the block needs multi-level tiling
    """
    return bool(_ffi_api_analysis.NeedsMultiLevelTiling(sch, block))  # pylint: disable=no-member


def is_strictly_inlineable(sch: tir.Schedule, block: tir.StmtSRef) -> bool:
    """Checks if a block needs multi-level tiling

    Parameters
    ----------
    sch: tir.Schedule
        The TIR class
    block: tir.StmtSRef
        The block to be analyzed

    Returns
    -------
    result : bool
        A boolean flag indicating if the block needs multi-level tiling
    """
    return bool(_ffi_api_analysis.IsStrictlyInlineable(sch, block))  # pylint: disable=no-member


def get_tensorize_loop_mapping(
    sch: tir.Schedule,
    block: tir.StmtSRef,
    desc_func: tir.PrimFunc,
) -> Optional[object]:
    """Check if the given block can be tensorized, and in the meantime gather the necessary
    information for tensorization

    Parameters
    ----------
    sch: Schedule
        The TIR class
    block: tir.StmtSRef
        The block to be analyzed
    desc_func: PrimFunc
        The description function of TensorIntrin we want to match

    Returns
    -------
    result: Optional[TensorizeInfo]
        The necessary information used for tensorization, or None if the block cannot be tensorized
    """
    return _ffi_api_analysis.GetTensorizeLoopMapping(  # pylint: disable=no-member
        sch, block, desc_func
    )


def count_flop(func: tir.PrimFunc) -> float:
    """Count the floating point operations of a PrimFunc

    Parameters
    ----------
    func: PrimFunc
        The PrimFunc to be counted

    Returns
    -------
    flop: float
        The number of floating point operations
    """
    return _ffi_api_analysis.CountFlop(func)  # pylint: disable=no-member
