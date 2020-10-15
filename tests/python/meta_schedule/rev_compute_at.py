import tvm
from tvm import tir
from tvm.script import ty

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,unexpected-keyword-arg
# fmt: off


@tvm.script.tir
def func(var_X: ty.handle, var_W: ty.handle, var_B: ty.handle, var_bn_scale: ty.handle, var_bn_offset: ty.handle, var_compute: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    bn_scale = tir.match_buffer(var_bn_scale, [512, 1, 1], elem_offset=0, align=128, offset_factor=1)
    W = tir.match_buffer(var_W, [512, 512, 3, 3], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(var_B, [512, 1, 1], elem_offset=0, align=128, offset_factor=1)
    X = tir.match_buffer(var_X, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    compute = tir.match_buffer(var_compute, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    bn_offset = tir.match_buffer(var_bn_offset, [512, 1, 1], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: (x + y), tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        pad_temp = tir.buffer_allocate([1, 512, 58, 58], elem_offset=0, align=128, offset_factor=1)
        compute_1 = tir.buffer_allocate([1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
        for i0 in range(0, 1):
            for i1 in range(0, 512):
                for i2 in range(0, 58):
                    for i3 in range(0, 58):
                        with tir.block([1, 512, 58, 58], "pad_temp") as [i0_1, i1_1, i2_1, i3_1]:
                            tir.bind(i0_1, i0)
                            tir.bind(i1_1, i1)
                            tir.bind(i2_1, i2)
                            tir.bind(i3_1, i3)
                            tir.reads([X[i0_1:(i0_1 + 1), i1_1:(i1_1 + 1), (i2_1 - 1):((i2_1 - 1) + 1), (i3_1 - 1):((i3_1 - 1) + 1)]])
                            tir.writes([pad_temp[i0_1:(i0_1 + 1), i1_1:(i1_1 + 1), i2_1:(i2_1 + 1), i3_1:(i3_1 + 1)]])
                            pad_temp[i0_1, i1_1, i2_1, i3_1] = tir.if_then_else(((((i2_1 >= 1) and (i2_1 < 57)) and (i3_1 >= 1)) and (i3_1 < 57)), X[i0_1, i1_1, (i2_1 - 1), (i3_1 - 1)], tir.float32(0), dtype="float32")
        for i0_2_outer_outer_outer, i1_2_outer_outer_outer, i2_2_outer_outer_outer, i3_2_outer_outer_outer in tir.grid(1, 8, 1, 7):
            for i0_2_outer_outer_inner, i1_2_outer_outer_inner, i2_2_outer_outer_inner, i3_2_outer_outer_inner in tir.grid(1, 1, 7, 1):
                for i4_outer, i5_outer, i6_outer in tir.grid(64, 3, 3):
                    for i0_2_outer_inner, i1_2_outer_inner, i2_2_outer_inner, i3_2_outer_inner in tir.grid(1, 16, 1, 1):
                        for i4_inner, i5_inner, i6_inner in tir.grid(8, 1, 1):
                            for i0_2_inner, i1_2_inner, i2_2_inner, i3_2_inner in tir.grid(1, 4, 8, 8):
                                with tir.block([1, 512, 56, 56, tir.reduce_axis(0, 512), tir.reduce_axis(0, 3), tir.reduce_axis(0, 3)], "compute") as [nn, ff, yy, xx, rc, ry, rx]:
                                    tir.bind(nn, (((i0_2_outer_outer_outer + i0_2_outer_outer_inner) + i0_2_outer_inner) + i0_2_inner))
                                    tir.bind(ff, (((((i1_2_outer_outer_outer + i1_2_outer_outer_inner)*16) + i1_2_outer_inner)*4) + i1_2_inner))
                                    tir.bind(yy, (((((i2_2_outer_outer_outer*7) + i2_2_outer_outer_inner) + i2_2_outer_inner)*8) + i2_2_inner))
                                    tir.bind(xx, ((((i3_2_outer_outer_outer + i3_2_outer_outer_inner) + i3_2_outer_inner)*8) + i3_2_inner))
                                    tir.bind(rc, ((i4_outer*8) + i4_inner))
                                    tir.bind(ry, (i5_outer + i5_inner))
                                    tir.bind(rx, (i6_outer + i6_inner))
                                    tir.reads([compute_1[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)], pad_temp[nn:(nn + 1), rc:(rc + 1), (yy + ry):((yy + ry) + 1), (xx + rx):((xx + rx) + 1)], W[ff:(ff + 1), rc:(rc + 1), ry:(ry + 1), rx:(rx + 1)]])
                                    tir.writes([compute_1[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)]])
                                    reducer.step(compute_1[nn, ff, yy, xx], (pad_temp[nn, rc, (yy + ry), (xx + rx)]*W[ff, rc, ry, rx]))
        for i0_6 in range(0, 1):
            for i1_6 in range(0, 512):
                for i2_6 in range(0, 56):
                    for i3_6 in range(0, 56):
                        with tir.block([1, 512, 56, 56], "compute_2") as [i0_7, i1_7, i2_7, i3_7]:
                            tir.bind(i0_7, i0_6)
                            tir.bind(i1_7, i1_6)
                            tir.bind(i2_7, i2_6)
                            tir.bind(i3_7, i3_6)
                            tir.reads([compute_1[i0_7:(i0_7 + 1), i1_7:(i1_7 + 1), i2_7:(i2_7 + 1), i3_7:(i3_7 + 1)], B[i1_7:(i1_7 + 1), 0:1, 0:1], bn_scale[i1_7:(i1_7 + 1), 0:1, 0:1], bn_offset[i1_7:(i1_7 + 1), 0:1, 0:1]])
                            tir.writes([compute[i0_7:(i0_7 + 1), i1_7:(i1_7 + 1), i2_7:(i2_7 + 1), i3_7:(i3_7 + 1)]])
                            compute[i0_7, i1_7, i2_7, i3_7] = tir.max((((compute_1[i0_7, i1_7, i2_7, i3_7] + B[i1_7, 0, 0])*bn_scale[i1_7, 0, 0]) + bn_offset[i1_7, 0, 0]), tir.float32(0))


# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,unexpected-keyword-arg


def main():
    sch = tir.create_schedule(func=func)
    consumer = sch.get_block("compute_2")
    producer = sch.get_block("compute")
    loop = sch.get_axes(producer)[3]
    sch.reverse_compute_at(block=consumer, loop=loop)


if __name__ == "__main__":
    main()
