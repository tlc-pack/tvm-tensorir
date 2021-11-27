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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
from typing import Callable, List

from numpy.testing import assert_allclose
import tvm
from tvm import meta_schedule as ms, te, tir
from tvm.meta_schedule.testing import te_workload

N_FEATURES = 164


def _make_context(target) -> ms.TuneContext:
    return ms.TuneContext(
        target=target,
        num_threads=1,
    )


def _make_candidate(f_sch: Callable[[], tir.Schedule]) -> ms.MeasureCandidate:
    return ms.MeasureCandidate(sch=f_sch(), args_info=[])


def _feature_names(  # pylint: disable=invalid-name
    buffers_per_store: int = 5,
    arith_intensity_curve_num_samples: int = 10,
) -> List[str]:
    result = [
        "float_mad",
        "float_addsub",
        "float_mul",
        "float_divmod",
        "float_cmp",
        "float_mathfunc",
        "float_otherfunc",
        "int_mad",
        "int_addsub",
        "int_mul",
        "int_divmod",
        "int_cmp",
        "int_mathfunc",
        "int_otherfunc",
        "bool_op",
        "select_op",
        "vec_num",
        "vec_prod",
        "vec_len",
        "vec_type.kPosNone",
        "vec_type.kPosInnerSpatial",
        "vec_type.kPosMiddleSpatial",
        "vec_type.kPosOuterSpatial",
        "vec_type.kPosInnerReduce",
        "vec_type.kPosMiddleReduce",
        "vec_type.kPosOuterReduce",
        "vec_type.kPosMixed",
        "unroll_num",
        "unroll_prod",
        "unroll_len",
        "unroll_type.kPosNone",
        "unroll_type.kPosInnerSpatial",
        "unroll_type.kPosMiddleSpatial",
        "unroll_type.kPosOuterSpatial",
        "unroll_type.kPosInnerReduce",
        "unroll_type.kPosMiddleReduce",
        "unroll_type.kPosOuterReduce",
        "unroll_type.kPosMixed",
        "parallel_num",
        "parallel_prod",
        "parallel_len",
        "parallel_type.kPosNone",
        "parallel_type.kPosInnerSpatial",
        "parallel_type.kPosMiddleSpatial",
        "parallel_type.kPosOuterSpatial",
        "parallel_type.kPosInnerReduce",
        "parallel_type.kPosMiddleReduce",
        "parallel_type.kPosOuterReduce",
        "parallel_type.kPosMixed",
        "is_gpu",
        "blockIdx_x_len",
        "blockIdx_y_len",
        "blockIdx_z_len",
        "threadIdx_x_len",
        "threadIdx_y_len",
        "threadIdx_z_len",
        "vthread_len",
    ]
    for i in range(buffers_per_store):
        result.extend(
            f"B{i}.{s}"
            for s in [
                "acc_type.kRead",
                "acc_type.kWrite",
                "acc_type.kReadWrite",
                "bytes",
                "unique_bytes",
                "lines",
                "unique_lines",
                "reuse_type.kLoopMultipleRead",
                "reuse_type.kSerialMultipleReadWrite",
                "reuse_type.kNoReuse",
                "reuse_dis_iter",
                "reuse_dis_bytes",
                "reuse_ct",
                "bytes_d_reuse_ct",
                "unique_bytes_d_reuse_ct",
                "lines_d_reuse_ct",
                "unique_lines_d_reuse_ct",
                "stride",
            ]
        )
    result.extend(f"arith_intensity_curve_{i}" for i in range(arith_intensity_curve_num_samples))
    result.extend(
        [
            "alloc_size",
            "alloc_prod",
            "alloc_outer_prod",
            "alloc_inner_prod",
            "outer_prod",
            "num_loops",
            "auto_unroll_max_step",
        ]
    )
    # 57 + 18 * 5 + 10 + 4 + 3
    assert len(result) == N_FEATURES
    return result


def _zip_feature(feature, names):
    assert feature.ndim == 1
    assert feature.shape[0] == N_FEATURES
    assert len(names) == N_FEATURES
    return list(zip(names, feature))


def _print_feature(feature, st, ed):  # pylint: disable=invalid-name
    named_feature = _zip_feature(feature, _feature_names())
    for k, v in named_feature[st:ed]:
        print("\t", k, v)


def test_cpu_matmul():
    def _create_schedule():
        func = te.create_prim_func(te_workload.matmul(n=512, m=512, k=512))
        sch = tir.Schedule(func, debug_mask="all")
        block = sch.get_block("C")
        i, j, k = sch.get_loops(block)
        i_o, i_i = sch.split(i, factors=[None, 16])  # outer: 32
        j_o, j_i = sch.split(j, factors=[None, 8])  # outer: 64
        sch.reorder(i_o, j_o, k, j_i, i_i)
        sch.vectorize(j_i)
        sch.parallel(i_o)
        sch.parallel(j_o)
        sch.unroll(k)
        return sch

    extractor = ms.feature_extractor.PerStoreFeature()
    (feature,) = extractor.extract_from(
        _make_context(tvm.target.Target("llvm")),
        candidates=[_make_candidate(_create_schedule)],
    )
    feature = feature.numpy()
    assert feature.shape == (1, N_FEATURES)
    f = feature[0]
    # Group 1.1: arith
    assert_allclose(
        actual=f[0:16],
        # fmt: off
        desired=[
            # float math ops
            0, 27, 27, 0, 0, 0, 0,
            # int math ops
            0, 29, 29, 0, 0, 0, 0,
            # bool/select ops
            0, 0,
        ],
        # fmt: on
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.2: vectorize
    assert_allclose(
        actual=f[16:27],
        desired=[1.0, 3.169924, 3.169924, 0, 0, 0, 0, 0, 0, 0, 1],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.3: unroll
    assert_allclose(
        actual=f[27:38],
        desired=[1.0, 9.002815, 9.002815, 0, 0, 0, 0, 0, 0, 0, 1],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.4: parallel
    assert_allclose(
        actual=f[38:49],
        desired=[1.58496, 11.0007, 6.022368, 0, 0, 0, 0, 0, 0, 0, 1],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 1.5: is_gpu, blockIdx.x/y/z, threadIdx.x/y/z, vthread
    assert_allclose(
        actual=f[49:57],
        desired=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.1: Buffer A
    assert_allclose(
        actual=f[57:75],
        desired=[
            1,
            0,
            0,
            29,
            20,
            27,
            14,
            1,
            0,
            0,
            4.087463,
            7.0552826,
            3.169925,
            26,
            17,
            24,
            11.0007038,
            9.002815,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.2: Buffer C
    assert_allclose(
        actual=f[75:93],
        desired=[
            0.0,
            0.0,
            1.0,
            29.0,
            20.000001907348633,
            27.0,
            14.00008773803711,
            1.0,
            0.0,
            0.0,
            7.011227130889893,
            9.250298500061035,
            9.002815246582031,
            20.000001907348633,
            11.000703811645508,
            18.0000057220459,
            5.044394016265869,
            9.002815246582031,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.3: Buffer B
    assert_allclose(
        actual=f[93:111],
        desired=[
            1.0,
            0.0,
            0.0,
            29.0,
            20.000001907348633,
            19.000001907348633,
            14.00008773803711,
            1.0,
            0.0,
            0.0,
            1.0,
            3.700439691543579,
            4.087462902069092,
            25.0,
            16.000022888183594,
            15.000043869018555,
            10.001408576965332,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.4: Dummy padding
    assert_allclose(
        actual=f[111:129],
        desired=[0.0] * 18,
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.5: Dummy padding
    assert_allclose(
        actual=f[129:147],
        desired=[0.0] * 18,
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 3: Arithmetic intensity
    assert_allclose(
        actual=f[147:157],
        desired=[
            0.7097842693328857,
            0.7408391237258911,
            0.8750449419021606,
            0.9449487924575806,
            1.0148526430130005,
            1.0847564935684204,
            1.113688349723816,
            1.1394684314727783,
            1.2119636535644531,
            1.2971993684768677,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 4 & 5
    assert_allclose(
        actual=f[157:164],
        desired=[
            20.000001907348633,
            18.0000057220459,
            1.0,
            27.0,
            27.0,
            2.5849626064300537,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )


if __name__ == "__main__":
    test_cpu_matmul()
    # test_cpu_fusion()
    # test_gpu()
