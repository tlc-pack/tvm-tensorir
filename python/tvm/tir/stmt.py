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
"""Statement AST Node in TVM.

Each statement node have subfields that can be visited from python side.

.. code-block:: python

    x = tvm.tir.Var("n", "int32")
    a = tvm.tir.Var("array", "handle")
    st = tvm.tir.stmt.Store(a, x + 1, 1)
    assert isinstance(st, tvm.tir.stmt.Store)
    assert(st.buffer_var == a)
"""
from enum import IntEnum
import tvm._ffi

from tvm.runtime import Object
from . import _ffi_api


class Stmt(Object):
    """Base class of all the statements."""


@tvm._ffi.register_object("tir.LetStmt")
class LetStmt(Stmt):
    """LetStmt node.

    Parameters
    ----------
    var : Var
        The variable in the binding.

    value : PrimExpr
        The value in to be binded.

    body : Stmt
        The body statement.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, var, value, body, span=None):
        self.__init_handle_by_constructor__(_ffi_api.LetStmt, var, value, body, span)


@tvm._ffi.register_object("tir.AssertStmt")
class AssertStmt(Stmt):
    """AssertStmt node.

    Parameters
    ----------
    condition : PrimExpr
        The assert condition.

    message : PrimExpr
        The error message.

    body : Stmt
        The body statement.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, condition, message, body, span=None):
        self.__init_handle_by_constructor__(_ffi_api.AssertStmt, condition, message, body, span)


class ForKind(IntEnum):
    """The kind of the for loop.

    note
    ----
    ForKind can change the control flow semantics
    of the loop and need to be considered in all TIR passes.
    """

    SERIAL = 0
    PARALLEL = 1
    VECTORIZED = 2
    UNROLLED = 3
    THREAD_BINDING = 4


@tvm._ffi.register_object("tir.For")
class For(Stmt):
    """For node.

    Parameters
    ----------
    loop_var : Var
        The loop variable.

    min_val : PrimExpr
        The beginning value.

    extent : PrimExpr
        The length of the loop.

    kind : ForKind
        The type of the for.

    body : Stmt
        The body statement.

    thread_binding: Optional[tir.IterVar]
        The thread this loop binds to. Only valid
        if kind is ThreadBinding

    annotations: tvm.ir.Map
        Additional annotation hints.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(
        self,
        loop_var,
        min_val,
        extent,
        kind,
        body,
        thread_binding=None,
        annotations=None,
        span=None,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.For,
            loop_var,
            min_val,
            extent,
            kind,
            body,
            thread_binding,
            annotations,
            span,
        )


@tvm._ffi.register_object("tir.Store")
class Store(Stmt):
    """Store node.

    Parameters
    ----------
    buffer_var : Var
        The buffer Variable.

    value : PrimExpr
        The value we want to store.

    index : PrimExpr
        The index in the store expression.

    predicate : PrimExpr
        The store predicate.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, buffer_var, value, index, predicate=None, span=None):
        if predicate is None:
            predicate = _ffi_api.const_true(value.dtype, span)
        self.__init_handle_by_constructor__(
            _ffi_api.Store, buffer_var, value, index, predicate, span
        )


@tvm._ffi.register_object("tir.BufferStore")
class BufferStore(Stmt):
    """Buffer store node.

    Parameters
    ----------
    buffer : Buffer
        The buffer.

    value : PrimExpr
        The value we to be stored.

    indices : List[PrimExpr]
        The indices location to be stored.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, buffer, value, indices, span=None):
        self.__init_handle_by_constructor__(_ffi_api.BufferStore, buffer, value, indices, span)


@tvm._ffi.register_object("tir.BufferRealize")
class BufferRealize(Stmt):
    """Buffer realize node.

    Parameters
    ----------
    buffer : Buffer
        The buffer.

    bounds : List[Range]
        The value we to be stored.

    condition : PrimExpr
        The realize condition.

    body : Stmt
        The body of the statement.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, buffer, bounds, condition, body, span=None):
        self.__init_handle_by_constructor__(
            _ffi_api.BufferRealize, buffer, bounds, condition, body, span
        )


@tvm._ffi.register_object("tir.ProducerStore")
class ProducerStore(Stmt):
    """ProducerStore node.

    Parameters
    ----------
    producer : DataProducer
        The data producer.

    value : PrimExpr
        The value to be stored.

    indices : list of Expr
        The index arguments of the store.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, producer, value, indices, span=None):
        self.__init_handle_by_constructor__(_ffi_api.ProducerStore, producer, value, indices, span)


@tvm._ffi.register_object("tir.Allocate")
class Allocate(Stmt):
    """Allocate node.

    Parameters
    ----------
    buffer_var : Var
        The buffer variable.

    dtype : str
        The data type of the buffer.

    extents : list of Expr
        The extents of the allocate

    condition : PrimExpr
        The condition.

    body : Stmt
        The body statement.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, buffer_var, dtype, extents, condition, body, span=None):
        self.__init_handle_by_constructor__(
            _ffi_api.Allocate, buffer_var, dtype, extents, condition, body, span
        )


@tvm._ffi.register_object("tir.AttrStmt")
class AttrStmt(Stmt):
    """AttrStmt node.

    Parameters
    ----------
    node : Node
        The node to annotate the attribute

    attr_key : str
        Attribute type key.

    value : PrimExpr
        The value of the attribute

    body : Stmt
        The body statement.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, node, attr_key, value, body, span=None):
        self.__init_handle_by_constructor__(_ffi_api.AttrStmt, node, attr_key, value, body, span)


@tvm._ffi.register_object("tir.ProducerRealize")
class ProducerRealize(Stmt):
    """ProducerRealize node.

    Parameters
    ----------
    producer : DataProducer
        The data producer.

    bounds : list of range
        The bound of realize

    condition : PrimExpr
        The realize condition.

    body : Stmt
        The realize body

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, producer, bounds, condition, body, span=None):
        self.__init_handle_by_constructor__(
            _ffi_api.ProducerRealize, producer, bounds, condition, body, span
        )


@tvm._ffi.register_object("tir.SeqStmt")
class SeqStmt(Stmt):
    """Sequence of statements.

    Parameters
    ----------
    seq : List[Stmt]
        The statements

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, seq, span=None):
        self.__init_handle_by_constructor__(_ffi_api.SeqStmt, seq, span)

    def __getitem__(self, i):
        return self.seq[i]

    def __len__(self):
        return len(self.seq)


@tvm._ffi.register_object("tir.IfThenElse")
class IfThenElse(Stmt):
    """IfThenElse node.

    Parameters
    ----------
    condition : PrimExpr
        The expression

    then_case : Stmt
        The statement to execute if condition is true.

    else_case : Stmt
        The statement to execute if condition is false.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, condition, then_case, else_case, span=None):
        self.__init_handle_by_constructor__(
            _ffi_api.IfThenElse, condition, then_case, else_case, span
        )


@tvm._ffi.register_object("tir.Evaluate")
class Evaluate(Stmt):
    """Evaluate node.

    Parameters
    ----------
    value : PrimExpr
        The expression to be evalued.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, value, span=None):
        self.__init_handle_by_constructor__(_ffi_api.Evaluate, value, span)


@tvm._ffi.register_object("tir.Prefetch")
class Prefetch(Stmt):
    """Prefetch node.

    Parameters
    ----------
    buffer : Buffer
        The buffer to be prefetched.

    bounds : list of Range
        The bounds to be prefetched.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, buffer, bounds, span=None):
        self.__init_handle_by_constructor__(_ffi_api.Prefetch, buffer, bounds, span)


@tvm._ffi.register_object
class Block(Stmt):
    """Block node.

    Parameters
    ----------
    iter_vars : list of IterVar
        The block Variable.

    reads : list of BufferRegion
        The read tensor region of the block.

    writes: list of BufferRegion
        The write tensor region of the block.

    name_hint: str
        the tag of the block.

    body: Stmt
        The body of the block.

    alloc_buffers: Optional[list of Buffer]
        The buffer allocations

    annotations: Optional[tvm.ir.Map]
        Additional annotation hints.

    match_buffers: Optional[list of MatchBufferRegion]
        The subregion buffer match

    exec_scope: Optional[str]
        the execution scope.

    init: Optional[Stmt]
        The init block of the reduction block

    span: span : Optional[Span]
        The location of this block in the source code.
    """

    def __init__(
        self,
        iter_vars,
        reads,
        writes,
        name_hint,
        body,
        alloc_buffers=None,
        annotations=None,
        match_buffers=None,
        exec_scope="",
        init=None,
        span=None,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.Block,
            iter_vars,
            reads,
            writes,
            name_hint,
            body,
            alloc_buffers,
            annotations,
            match_buffers,
            exec_scope,
            init,
            span,
        )


@tvm._ffi.register_object
class BlockRealize(Stmt):
    """BlockRealize node.

    Parameters
    ----------
    values : list of Expr
        The binding value of the block var.

    predicate : Expr
        The predicates of the block.

    block : Block
        The block to realize

    """

    def __init__(self, values, predicate, block, span=None):
        self.__init_handle_by_constructor__(_ffi_api.BlockRealize, values, predicate, block, span)


@tvm._ffi.register_object
class BufferRegion(Object):
    """BufferRegion Node

    Parameters
    ----------
    buffer : Buffer
        The tensor of the buffer region

    region : list of Range
        The region array of the buffer region
    """

    def __init__(self, buffer, region):
        self.__init_handle_by_constructor__(_ffi_api.BufferRegion, buffer, region)


@tvm._ffi.register_object
class MatchBufferRegion(Object):
    """MatchBufferRegion Node

    Parameters
    ----------
    buffer : Buffer
        The target buffer

    source : BufferRegion
        The region of source buffer
    """

    def __init__(self, buffer, source):
        self.__init_handle_by_constructor__(_ffi_api.MatchBufferRegion, buffer, source)


def stmt_seq(*args):
    """Make sequence of statements

    Parameters
    ----------
    args : list of Expr or Var
        List of statements to be combined as sequence.

    Returns
    -------
    stmt : Stmt
        The combined statement.
    """
    ret = []
    for value in args:
        if not isinstance(value, Stmt):
            value = Evaluate(value)
        ret.append(value)
    if len(ret) == 1:
        return ret[0]
    return SeqStmt(ret)


def stmt_list(stmt):
    """Make list of stmt from blocks.

    Parameters
    ----------
    stmt : A block statement

    Returns
    -------
    stmt_list : list of Stmt
         The unpacked list of statements
    """
    if isinstance(stmt, SeqStmt):
        res = []
        for x in stmt:
            res += stmt_list(x)
        return res
    return [stmt]
