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
# pylint: disable=missing-function-docstring
"""Test for feature extraction"""
import math
import pytest

import te_workload
from numpy.testing import assert_allclose
from tvm import meta_schedule as ms
from tvm import te, tir


def _float_equal(a: float, b: float) -> bool:
    return math.fabs(a - b) < 1e-6


@pytest.mark.xfail
def test_meta_schedule_per_block_feature_cpu_matmul():
    def _create_schedule(n, m, k):
        func = te.create_prim_func(te_workload.matmul(n=n, m=m, k=k))
        sch = tir.Schedule(func, traced=True)
        block = sch.get_block("C")
        i, j, k = sch.get_loops(block)
        i_o, i_i = sch.split(i, factors=[-1, 16])  # outer: 32
        j_o, j_i = sch.split(j, factors=[-1, 8])  # outer: 64
        sch.reorder(i_o, j_o, k, j_i, i_i)
        sch.vectorize(j_i)
        sch.parallel(i_o)
        sch.parallel(j_o)
        sch.unroll(k)
        return sch

    names = list(ms.feature.per_bloc_feature_names())
    n_features = len(names)
    # Create schedule
    sch = _create_schedule(n=512, m=512, k=512)
    # Extract features
    feature = ms.feature.per_block_feature(sch)
    assert feature.shape == (1, n_features)
    feature = feature[0]
    # correspond the features with their names
    feature_dict = {
        name: value
        for name, value in zip(names, feature)  # pylint: disable=unnecessary-comprehension
    }
    a_name = None
    b_name = None
    c_name = None
    for name in ["B0", "B1", "B2"]:
        if _float_equal(feature_dict[name + ".acc_type.kReadWrite"], 1.0):
            c_name = name
            continue
        if not _float_equal(feature_dict[name + ".acc_type.kRead"], 1.0):
            continue
        if _float_equal(feature_dict[name + ".stride"], 0):
            b_name = name
        else:
            a_name = name

    assert_allclose(
        actual=feature[0:16],
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
    # vectorize
    assert_allclose(
        actual=feature[16:27],
        desired=[1.0, 3.169924, 3.169924, 0, 0, 0, 0, 0, 0, 0, 1],
        rtol=1e-5,
        atol=1e-5,
    )
    # unroll
    assert_allclose(
        actual=feature[27:38],
        desired=[1.0, 9.002815, 9.002815, 0, 0, 0, 0, 0, 0, 0, 1],
        rtol=1e-5,
        atol=1e-5,
    )
    # parallel
    assert_allclose(
        actual=feature[38:49],
        desired=[1.58496, 11.0007, 6.0224, 0, 0, 0, 0, 0, 0, 0, 1],
        rtol=1e-5,
        atol=1e-5,
    )
    # blockIdx / threadIdx / vthread
    assert_allclose(
        actual=feature[49:56],
        desired=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        rtol=1e-5,
        atol=1e-5,
    )
    buffer_feature_dict = {
        "B0": feature[56:74],
        "B1": feature[74:92],
        "B2": feature[92:110],
        "B3": feature[110:128],
        "B4": feature[128:146],
    }
    # features for buffer 'A'
    assert_allclose(
        actual=buffer_feature_dict[a_name],
        desired=[
            # fmt: off
            1, 0, 0, 29, 20, 27, 14, 1, 0, 0,
            4.08746, 7.05528, 3.16992, 26, 17, 24, 11.0007, 9.0028
            # fmt: on
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # features for buffer 'B'
    assert_allclose(
        actual=buffer_feature_dict[b_name],
        desired=[
            # fmt: off
            1, 0, 0, 29, 20, 19, 14, 1, 0, 0,
            1, 3.7004397, 4.0874629, 25, 16, 15, 10.001409, 0,
            # fmt: on
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # features for buffer 'C'
    assert_allclose(
        actual=buffer_feature_dict[c_name],
        desired=[
            # fmt: off
            0, 0, 1, 29, 20.000001907348633, 27, 14.00008773803711, 1, 0, 0,
            7.011227130889893, 9.250298500061035, 9.002815246582031,
            20.000001907348633, 11.000703811645508, 18.0000057220459,
            5.044394016265869, 9.002815246582031,
            # fmt: on
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # an empty buffer "B3"
    assert_allclose(
        actual=buffer_feature_dict["B3"],
        desired=[0] * 18,
        rtol=1e-5,
        atol=1e-5,
    )
    # an empty buffer "B4"
    assert_allclose(
        actual=buffer_feature_dict["B4"],
        desired=[0] * 18,
        rtol=1e-5,
        atol=1e-5,
    )
    # arith intensity curve
    assert_allclose(
        actual=feature[146:156],
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
    # misc
    assert_allclose(
        actual=feature[156:159],
        desired=[
            27,  # outer_prod
            2.58496,  # num_loops
            0,  # auto_unroll_max_step
        ],
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.xfail
def test_meta_schedule_per_block_feature_cpu_fusion():
    def _create_schedule(n, m):
        a = te.placeholder((n, m), name="A")
        b = te.compute((n, m), lambda i, j: a[i][j], name="B")
        c = te.compute((n, m), lambda i, j: b[i][j], name="C")
        func = te.create_prim_func([a, b, c])
        sch = tir.Schedule(func, traced=True)
        block_b = sch.get_block("B")
        block_c = sch.get_block("C")
        _, j = sch.get_loops(block_c)
        sch.compute_at(block_b, j)
        return sch

    names = list(ms.feature.per_bloc_feature_names())
    n_features = len(names)
    # Create schedule
    sch = _create_schedule(n=64, m=32)
    # Extract features
    feature = ms.feature.per_block_feature(sch)
    assert feature.shape == (2, n_features)

    def _check_feature(feature, read_is_serial_reuse):
        # float/int/bool/select ops
        assert_allclose(
            actual=feature[0:16],
            desired=[0] * 16,
            rtol=1e-5,
            atol=1e-5,
        )
        # vectorize
        assert_allclose(
            actual=feature[16:27],
            desired=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            rtol=1e-5,
            atol=1e-5,
        )
        # unroll
        assert_allclose(
            actual=feature[27:38],
            desired=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            rtol=1e-5,
            atol=1e-5,
        )
        # parallel
        assert_allclose(
            actual=feature[38:49],
            desired=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            rtol=1e-5,
            atol=1e-5,
        )
        # blockIdx / threadIdx / vthread
        assert_allclose(
            actual=feature[49:56],
            desired=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            rtol=1e-5,
            atol=1e-5,
        )
        buffer_feature_dict = {
            "B0": feature[56:74],
            "B1": feature[74:92],
            "B2": feature[92:110],
            "B3": feature[110:128],
            "B4": feature[128:146],
        }
        r_name = None
        w_name = None
        for name in ["B0", "B1"]:
            k_read = feature[names.index(name + ".acc_type.kRead")]
            k_write = feature[names.index(name + ".acc_type.kWrite")]
            if _float_equal(k_read, 1.0) and _float_equal(k_write, 0.0):
                r_name = name
            elif _float_equal(k_read, 0.0) and _float_equal(k_write, 1.0):
                w_name = name
        # features for the read buffer
        assert_allclose(
            actual=buffer_feature_dict[w_name],
            # fmt: off
            desired=[
                0.0, 1.0, 0.0, 13.000176, 13.000176, 7.011227, 7.011227,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                14.0000877, 14.0000877, 8.0056, 8.0056, 1.0,
            ],
            # fmt: on
            rtol=1e-5,
            atol=1e-5,
        )
        if read_is_serial_reuse:
            assert_allclose(
                actual=buffer_feature_dict[r_name],
                # fmt: off
                desired=[
                    1, 0, 0, 13.000176, 13.000176, 7.0112271, 7.0112271,
                    0, 1, 0, 1, 4.087463, 1,
                    13.000176, 13.000176, 7.0112271, 7.0112271, 1,
                ],
                # fmt: on
                rtol=1e-5,
                atol=1e-5,
            )
        else:
            assert_allclose(
                actual=buffer_feature_dict[r_name],
                # fmt: off
                desired=[
                    1, 0, 0, 13.000176, 13.000176, 7.011227, 7.011227,
                    0, 0, 1, 0, 0, 0,
                    14.0000877, 14.0000877, 8.0056, 8.0056, 1,
                ],
                # fmt: on
                rtol=1e-5,
                atol=1e-5,
            )

        # an empty buffer "B2"
        assert_allclose(
            actual=buffer_feature_dict["B2"],
            desired=[0] * 18,
            rtol=1e-5,
            atol=1e-5,
        )
        # an empty buffer "B3"
        assert_allclose(
            actual=buffer_feature_dict["B3"],
            desired=[0] * 18,
            rtol=1e-5,
            atol=1e-5,
        )
        # an empty buffer "B4"
        assert_allclose(
            actual=buffer_feature_dict["B4"],
            desired=[0] * 18,
            rtol=1e-5,
            atol=1e-5,
        )
        # arith intensity curve
        assert_allclose(
            actual=feature[146:156],
            desired=[0] * 10,
            rtol=1e-5,
            atol=1e-5,
        )
        # misc
        assert_allclose(
            actual=feature[156:159],
            desired=[
                11.000703811645508,  # outer_prod
                1.5849624872207642,  # num_loops
                0,  # auto_unroll_max_step
            ],
            rtol=1e-5,
            atol=1e-5,
        )

    _check_feature(feature[0], read_is_serial_reuse=False)
    _check_feature(feature[1], read_is_serial_reuse=True)


@pytest.mark.xfail
def test_meta_schedule_per_block_feature_gpu():
    def _create_schedule(n, m, k):
        func = te.create_prim_func(te_workload.matmul(n=n, m=m, k=k))
        sch = tir.Schedule(func, traced=True)
        c = sch.get_block("C")
        c_local = sch.cache_write(c, 0, "local")
        i, j, k = sch.get_loops(c_local)
        # pylint: disable=invalid-name
        i0, i1, i2, i3, i4 = sch.split(i, factors=[-1, 1, 16, 32, 1])  # outer: 1
        j0, j1, j2, j3, j4 = sch.split(j, factors=[-1, 4, 1, 1, 16])  # outer: 8
        k0, k1, k2 = sch.split(k, factors=[-1, 1, 2])  # outer: 256
        # pylint: enable=invalid-name
        # fmt: off
        sch.reorder(
            i0, j0,  # S
            i1, j1,  # S
            i2, j2,  # S
            k0,      # R
            k1,      # R
            i3, j3,  # S
            k2,      # R
            i4, j4,  # S
        )
        # fmt: on
        # thread binding
        i0_j0 = sch.fuse(i0, j0)
        i1_j1 = sch.fuse(i1, j1)
        i2_j2 = sch.fuse(i2, j2)
        sch.bind(i0_j0, "blockIdx.x")
        sch.bind(i1_j1, "vthread")
        sch.bind(i2_j2, "threadIdx.x")
        # fusion
        sch.reverse_compute_at(c, i2_j2)
        # cache read 'B'
        b_shared = sch.cache_read(c_local, 2, "shared")
        sch.compute_at(b_shared, k0)
        _, _, _, _, b_i, b_j = sch.get_loops(b_shared)
        b_ij = sch.fuse(b_i, b_j)
        _, b_j = sch.split(b_ij, factors=[-1, 16])  # outer: 8
        sch.bind(b_j, "threadIdx.x")
        # cache read 'A'
        a_shared = sch.cache_read(c_local, 1, "shared")
        sch.compute_at(a_shared, k0)
        _, _, _, _, a_i, a_j = sch.get_loops(a_shared)
        a_ij = sch.fuse(a_i, a_j)
        _, a_j = sch.split(a_ij, factors=[-1, 16])  # outer: 64
        sch.bind(a_j, "threadIdx.x")
        # auto unroll
        sch.mark_loop(i0_j0, "pragma_auto_unroll_max_step", tir.IntImm("int32", 1024))
        sch.mark_loop(i0_j0, "pragma_unroll_explicit", tir.IntImm("int32", 1))
        return sch

    names = list(ms.feature.per_bloc_feature_names())
    n_features = len(names)
    # Create schedule
    sch = _create_schedule(n=512, m=512, k=512)
    # Extract features
    feature = ms.feature.per_block_feature(sch)
    assert feature.shape == (4, n_features)

    def _check_gpu_threads(feature):
        # blockIdx / threadIdx / vthread
        assert_allclose(
            actual=feature[49:56],
            desired=[3.169925001442312, 1.0, 1.0, 4.087462841250339, 1.0, 1.0, 2.321928094887362],
            rtol=1e-5,
            atol=1e-5,
        )

    def _is_read(feature, buffer_name):
        result = feature[names.index(buffer_name + ".acc_type.kRead")]
        return _float_equal(result, 1.0)

    def _is_write(feature, buffer_name):
        result = feature[names.index(buffer_name + ".acc_type.kWrite")]
        return _float_equal(result, 1.0)

    def _get_read_write_buffer(feature):  # pylint: disable=inconsistent-return-statements
        b_0 = feature[56:74]
        b_1 = feature[74:92]
        if _is_read(feature, "B0") and _is_write(feature, "B1"):
            return b_0, b_1
        if _is_read(feature, "B1") and _is_write(feature, "B0"):
            return b_1, b_0
        assert False

    def _check_empty_buffer(feature, buffer_names):
        buffer_feature_dict = {
            "B0": feature[56:74],
            "B1": feature[74:92],
            "B2": feature[92:110],
            "B3": feature[110:128],
            "B4": feature[128:146],
        }
        for name in buffer_names:
            assert_allclose(
                actual=buffer_feature_dict[name],
                desired=[0] * 18,
                rtol=1e-5,
                atol=1e-5,
            )

    def _check_shared_read(feature, outer_prod, read_feature, write_feature):
        # float/int/bool/select ops, vectorize/unroll/parallel
        assert_allclose(
            actual=feature[16:49],
            desired=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] * 3,
            rtol=1e-5,
            atol=1e-5,
        )
        # blockIdx / threadIdx / vthread
        _check_gpu_threads(feature)
        # arith intensity curve
        assert_allclose(
            actual=feature[146:156],
            desired=[0] * 10,
            rtol=1e-5,
            atol=1e-5,
        )
        # check write buffer
        read, write = _get_read_write_buffer(feature)
        assert_allclose(
            actual=read,
            desired=read_feature,
            rtol=1e-5,
            atol=1e-5,
        )
        assert_allclose(
            actual=write,
            desired=write_feature,
            rtol=1e-5,
            atol=1e-5,
        )
        # check empty buffers
        _check_empty_buffer(feature, ["B2", "B3", "B4"])
        # misc
        assert_allclose(
            actual=feature[156:159],
            desired=[
                outer_prod,  # outer_prod
                2.8073549,  # num_loops
                10.001409,  # auto_unroll_max_step
            ],
            rtol=1e-5,
            atol=1e-5,
        )

    def _check_local_write(feature):
        # float/int/bool/select ops, vectorize/unroll/parallel
        assert_allclose(
            actual=feature[0:49],
            desired=[0] * 16 + [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] * 3,
            rtol=1e-5,
            atol=1e-5,
        )
        # blockIdx / threadIdx / vthread
        _check_gpu_threads(feature)
        # arith intensity curve
        assert_allclose(
            actual=feature[146:156],
            desired=[0] * 10,
            rtol=1e-5,
            atol=1e-5,
        )
        # check write buffer
        read, write = _get_read_write_buffer(feature)
        assert_allclose(
            actual=read,
            # fmt: off
            desired=[1.      ,  0.      ,  0.      , 20.000001, 11.000704, 14.000088,
                     5.044394,  1.      ,  0.      ,  0.      ,  9.002815, 12.000352,
                     4.087463, 16.000022,  7.011227, 10.001408,  1.584963,  1.],
            # fmt: on
            rtol=1e-5,
            atol=1e-5,
        )
        assert_allclose(
            actual=write,
            # fmt: off
            desired=[0, 1, 0, 20, 20, 14, 14, 0, 0, 1, 0, 0, 0, 21, 21, 15, 15, 1],
            # fmt: on
            rtol=1e-5,
            atol=1e-5,
        )
        # check empty buffers
        _check_empty_buffer(feature, ["B2", "B3", "B4"])
        # misc
        # TODO(@junrushao1994): investigate into auto_unroll
        assert_allclose(
            actual=feature[156:159],
            desired=[
                18,  # outer_prod
                2.5849626,  # num_loops
                10.001409,  # auto_unroll_max_step
            ],
            rtol=1e-5,
            atol=1e-5,
        )

    def _check_compute(feature):
        # float/int/bool/select ops, vectorize/unroll/parallel
        assert_allclose(
            actual=feature[0:49],
            desired=[0, 27, 27]
            + [0] * 4
            + [0, 28, 28]
            + [0] * 6
            + [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] * 3,
            rtol=1e-5,
            atol=1e-5,
        )
        # blockIdx / threadIdx / vthread
        _check_gpu_threads(feature)
        # arith intensity curve
        assert_allclose(
            actual=feature[146:156],
            desired=[
                0.709784,
                0.75488,
                0.877591,
                0.995739,
                1.244674,
                1.493608,
                1.70931,
                1.803158,
                1.984183,
                2.204648,
            ],
            rtol=1e-5,
            atol=1e-5,
        )
        # check A, B, C
        buffer_feature_dict = {
            "B0": feature[56:74],
            "B1": feature[74:92],
            "B2": feature[92:110],
            "B3": feature[110:128],
            "B4": feature[128:146],
        }
        a_name = b_name = c_name = None
        for buffer_name in ["B0", "B1", "B2"]:
            if _is_read(feature, buffer_name):
                stride = feature[names.index(buffer_name + ".stride")]
                if _float_equal(stride, 1.0):
                    b_name = buffer_name
                elif _float_equal(stride, 0.0):
                    a_name = buffer_name
                else:
                    assert False
            else:
                k_read_write = feature[names.index(buffer_name + ".acc_type.kReadWrite")]
                assert _float_equal(k_read_write, 1.0)
                c_name = buffer_name
        assert a_name is not None
        assert b_name is not None
        assert c_name is not None
        assert_allclose(
            actual=buffer_feature_dict[a_name],
            # fmt: off
            desired=[
                1, 0, 0, 29, 12.000352, 19, 9.002815, 1, 0, 0, 1,
                3.70044, 4.0874629, 25, 8.005625, 15, 5.044394, 0.0,
            ],
            # fmt: on
            rtol=1e-5,
            atol=1e-5,
        )
        assert_allclose(
            actual=buffer_feature_dict[b_name],
            # fmt: off
            desired=[
                1, 0, 0, 29, 9.002815, 23, 3.169925, 1, 0, 0, 5.044394,
                7.6510515, 5.044394, 24, 4.087463, 18, 0.321928, 1,
            ],
            # fmt: on
            rtol=1e-5,
            atol=1e-5,
        )
        assert_allclose(
            actual=buffer_feature_dict[c_name],
            # fmt: off
            desired=[
                0, 0, 1, 29, 11.000704, 23, 5.044394, 1, 0, 0, 4.0874629,
                7.0552821, 1.5849625, 28, 10.001408, 22, 4.087463, 1,
            ],
            # fmt: on
            rtol=1e-5,
            atol=1e-5,
        )
        # check empty buffers
        _check_empty_buffer(feature, ["B3", "B4"])
        # misc
        assert_allclose(
            actual=feature[156:159],
            desired=[
                27,  # outer_prod
                3,  # num_loops
                10.001409,  # auto_unroll_max_step
            ],
            rtol=1e-5,
            atol=1e-5,
        )

    _check_shared_read(
        feature[0],
        outer_prod=27.0,
        # fmt: off
        read_feature=[
            1, 0, 0, 29, 20, 23, 14, 1, 0, 0, 18, 20.005626, 4.0874629, 25, 16, 19, 10.0014086, 1,
        ],
        write_feature=[
            0, 1, 0, 29, 12.000352, 23, 9.002815, 1, 0, 0, 10.001408, 13.000176, 8.005625, 21, 4.087463, 15, #pylint: disable=line-too-long
            1.584963, 1,
        ],
        # fmt: on
    )
    _check_shared_read(
        feature[1],
        outer_prod=24.0,
        # fmt: off
        read_feature=[
            1, 0, 0, 26, 20, 20, 14, 1, 0, 0, 15,
            20.175551, 4.0874629, 22, 16, 16, 10.0014086, 1,
        ],
        write_feature=[
            0, 1, 0, 26, 9.002815, 20, 3.169925, 1, 0, 0, 7.011227,
            10.001408, 8.005625, 18, 1.584963, 12.000352, 0.044394, 1,
        ],
        # fmt: on
    )
    _check_compute(feature[2])
    _check_local_write(feature[3])


if __name__ == "__main__":
    test_meta_schedule_per_block_feature_cpu_matmul()
    test_meta_schedule_per_block_feature_cpu_fusion()
    test_meta_schedule_per_block_feature_gpu()
