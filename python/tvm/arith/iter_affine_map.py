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
""" Iterator (quasi)affine mapping patterns."""
import tvm._ffi
from tvm.runtime import Object
from tvm.ir import PrimExpr
from . import _ffi_api


class IterMapExpr(PrimExpr):
    """Base class of all IterMap expressions."""


@tvm._ffi.register_object("arith.IterMark")
class IterMark(Object):
    """Mark the source as an iterator in [0, extent).

    Parameters
    ----------
    source : PrimExpr.
        The source expression.

    extent : PrimExpr
        The extent of the iterator.
    """

    def __init__(self, source, extent):
        self.__init_handle_by_constructor__(_ffi_api.IterMark, source, extent)


@tvm._ffi.register_object("arith.IterSplitExpr")
class IterSplitExpr(IterMapExpr):
    """Split of an iterator.

    result = floormod(floordiv(source, lower_factor), extent) * scale

    Parameters
    ----------
    source : IterMark
        The source marked iterator.

    lower_factor : PrimExpr
        The lower factor to split the domain.

    extent : PrimExpr
        The extent of the split.

    scale : PrimExpr
        Additional scale to the split.
    """

    def __init__(self, source, lower_factor, extent, scale):
        self.__init_handle_by_constructor__(
            _ffi_api.IterSplitExpr, source, lower_factor, extent, scale
        )


@tvm._ffi.register_object("arith.IterSumExpr")
class IterSumExpr(IterMapExpr):
    """Fuse multiple iterators by summing them with scaling.

    result = sum(args) + base

    Parameters
    ----------
    args : List[IterSplitExpr]
        The input to the sum expression.

    base : PrimExpr
        The base offset.
    """

    def __init__(self, args, base):
        self.__init_handle_by_constructor__(_ffi_api.IterSumExpr, args, base)


def detect_iter_map(bindings, root_iters, predicate=True, require_bijective=False):
    """Detect if bindings can be written mapped iters from root iters

    Parameters
    ----------
    bindings : List[PrimExpr]
        The bindings of leaf iterators

    root_iters : Map[Var, Range]
        The domain of each root iterators.

    predicate : PrimExpr
        The predicate tht input iterators follow

    require_bijective : bool
        A boolean flag that indicates whether the bindings should be bijective

    Returns
    -------
    results : List[IterSumExpr]
        The iter map matching result.
        Empty array if no match can be found.
    """
    return _ffi_api.DetectIterMap(bindings, root_iters, predicate, require_bijective)


def subspace_division(bindings, root_iters, sub_iters, predicate=True, require_bijective=False):
    """Detect if bindings can be written as
    [a_0*e_0 + b_0 + c_0, a_1*e_1 + b_1, ..., a_n*e_n + b_n]
    where a = some-quasi-affine-iter-map(root_iters set_minus sub_iters)
          b = some-quasi-affine-iter-map(sub_iters)
          c is constant symbols
          e is the extent of b
    For example, z*12 + y*3 + x + c = (z*4+y)*3 + x, if sub_iters={x}

    Parameters
    ----------
    bindings : List[PrimExpr]
        The bindings to detect pattern for.

    root_iters : Map[Var, Range]
        The domain of each input iterators, which is the basis of the whole space

    sub_iters : Array[Var]
        The subset of input_iters, which is the basis of the subspace

    predicate : PrimExpr
        The predicate for input_iters

    require_bijective : bool
        A boolean flag that indicates whether the bindings should be bijective

    Returns
    -------
    results : List[List[PrimExpr]]
        The iter map matching result. The inner list is of length 2.
        The first expr is the basis of the quotient space.
        The second expr is the basis of the subspace.
        Empty array if no match can be found.
    """
    return _ffi_api.SubspaceDivision(bindings, root_iters, sub_iters, predicate, require_bijective)


def normalize_iter_map_to_expr(expr):
    """Given an IterMapExpr, transform it to normal PrimExpr

    Parameters
    ----------
    expr : IterMapExpr
        the input IterMapExpr

    Returns
    -------
    result : PrimExpr
        the corresponding normal PrimExpr
    """
    return _ffi_api.IterVarMapConvert(expr)
