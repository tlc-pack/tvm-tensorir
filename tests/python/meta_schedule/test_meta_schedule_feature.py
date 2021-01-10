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

import te_workload
import tvm
from numpy.testing import assert_allclose
from tvm import meta_schedule as ms
from tvm import te


def _float_equal(a: float, b: float) -> bool:
    return math.fabs(a - b) < 1e-6


def test_meta_schedule_per_block_feature_cpu_matmul():
    def _create_schedule(n, m, k):
        func = te.create_func(te_workload.matmul(n=n, m=m, k=k))
        sch = ms.Schedule(func)
        block = sch.get_block("C")
        i, j, k = sch.get_axes(block)
        i_o, i_i = sch.split(i, factors=[32, 16])
        j_o, j_i = sch.split(j, factors=[64, 8])
        sch.reorder(after_axes=[i_o, j_o, k, j_i, i_i])
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
    feature = ms.feature.calc_per_block_feature(sch)
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
            0, 0, 0, 0, 0, 0, 0,
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


def test_meta_schedule_per_block_feature_cpu_fusion():
    def _create_schedule(n, m):
        a = te.placeholder((n, m), name="A")
        b = te.compute((n, m), lambda i, j: a[i][j], name="B")
        c = te.compute((n, m), lambda i, j: b[i][j], name="C")
        func = te.create_func([a, b, c])
        sch = ms.Schedule(func)
        block_b = sch.get_block("B")
        block_c = sch.get_block("C")
        _, j = sch.get_axes(block_c)
        sch.compute_at(block_b, j)
        return sch

    names = list(ms.feature.per_bloc_feature_names())
    n_features = len(names)
    # Create schedule
    sch = _create_schedule(n=64, m=32)
    # Extract features
    feature = ms.feature.calc_per_block_feature(sch)
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


def test_meta_schedule_per_block_feature_gpu():
    pass


if __name__ == "__main__":
    test_meta_schedule_per_block_feature_cpu_matmul()
    test_meta_schedule_per_block_feature_cpu_fusion()
    # test_meta_schedule_per_block_feature_gpu()
