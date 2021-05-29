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
"""End to end resnet-18 CPU test"""
# pylint: disable=missing-function-docstring
import os

import numpy as np
import pytest
import tvm
import tvm.relay.testing
from tvm import meta_schedule as ms
from tvm import relay, te, auto_scheduler
from tvm.contrib import graph_runtime as runtime
from tvm.contrib.utils import tempdir


# import logging
# logging.basicConfig(level=logging.DEBUG)  # to dump TVM IR after fusion


def get_network(name, batch_size, layout="NHWC", dtype="float32"):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)

    return mod, params, input_shape, output_shape


RPC_KEY = "rtx-3080"
network = "resnet-50"
batch_size = 1
layout = "NHWC"
target = tvm.target.Target("nvidia/geforce-rtx-3080")
dtype = "float32"
TARGET_HOST = tvm.target.Target("llvm")
SPACE = ms.space.PostOrderApply(
    stages=[
        ms.rule.simplify_compute_with_const_tensor(),
        ms.rule.multi_level_tiling(
            structure="SSSRRSRS",
            must_cache_read=True,
            cache_read_scope="shared",
            can_cache_write=True,
            must_cache_write=True,
            cache_write_scope="local",
            consumer_inline_strict=False,
            fusion_levels=[3],
            vector_load_max_len=4,
            tile_binds=["blockIdx.x", "vthread", "threadIdx.x"],
        ),
        ms.rule.special_compute_location_gpu(),
        ms.rule.inline_pure_spatial(strict_mode=False),
        # ms.rule.thread_bind(64),
        ms.rule.parallelize_vectorize_unroll(
            max_jobs_per_core=-1,  # disable parallelize
            max_vectorize_extent=-1,  # disable vectorize
            unroll_max_steps=[0, 16, 64, 512, 1024],
            unroll_explicit=True,
        ),
    ],
    postprocs=[
        ms.postproc.rewrite_cooperative_fetch(),
        ms.postproc.rewrite_unbound_blocks(),
        ms.postproc.rewrite_parallel_vectorize_unroll(),
        ms.postproc.rewrite_reduction_block(),
        ms.postproc.disallow_dynamic_loops(),
        ms.postproc.verify_gpu_code(),
    ],
)


@pytest.mark.skip(reason="needs RPC")
def test_end_to_end_resnet(log):
    os.environ["TVM_TRACKER_KEY"] = RPC_KEY
    mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)

    data = np.random.uniform(-1, 1, size=input_shape).astype("float32")

    # lib_std = relay.build_module.build(mod, target, params=params)
    print("std build over")
    tir_funcs = ms.extract_tasks(mod["main"], params, target)
    print("func num:", len(tir_funcs))
    func = tir_funcs["func2872655767091923505"]
    task = ms.SearchTask(workload=func, task_name="tmp", target=target)

    # sch=ms.Schedule(func=func)
    # b1 = sch.get_block(name="inverse")
    # l2, l3, l4, l5, l6, l7 = sch.get_axes(block=b1)
    # sch.unroll(loop=l2)
    # sch.unroll(loop=l3)
    # sch.unroll(loop=l6)
    # sch.unroll(loop=l7)
    # v8, v9 = sch.sample_perfect_tile(n=2, loop=l4, max_innermost_factor=64, decision=[14, 14])
    # l10, l11 = sch.split(loop=l4, factors=[v8, v9])
    # v12, v13 = sch.sample_perfect_tile(n=2, loop=l5, max_innermost_factor=64, decision=[16, 8])
    # l14, l15 = sch.split(loop=l5, factors=[v12, v13])
    # sch.reorder(*[l10, l14, l11, l15, l2, l3, l6, l7])
    # b16 = sch.get_block(name="data_pack")
    # l17, l18, l19, l20, l21, l22 = sch.get_axes(block=b16)
    # sch.unroll(loop=l17)
    # sch.unroll(loop=l18)
    # sch.unroll(loop=l21)
    # sch.unroll(loop=l22)
    # v23, v24 = sch.sample_perfect_tile(n=2, loop=l19, max_innermost_factor=64, decision=[196, 1])
    # l25, l26 = sch.split(loop=l19, factors=[v23, v24])
    # v27, v28 = sch.sample_perfect_tile(n=2, loop=l20, max_innermost_factor=64, decision=[4, 32])
    # l29, l30 = sch.split(loop=l20, factors=[v27, v28])
    # sch.reorder(*[l25, l29, l26, l30, l17, l18, l21, l22])
    # b31 = sch.get_block(name="bgemm")
    # b32 = sch.cache_write(block=b31, i=0, storage_scope="local")
    # l33, l34, l35, l36, l37 = sch.get_axes(block=b32)
    # v38, v39, v40, v41, v42 = sch.sample_perfect_tile(n=5, loop=l33, max_innermost_factor=64, decision=[4, 1, 1, 1, 1])
    # l43, l44, l45, l46, l47 = sch.split(loop=l33, factors=[v38, v39, v40, v41, v42])
    # v48, v49, v50, v51, v52 = sch.sample_perfect_tile(n=5, loop=l34, max_innermost_factor=64, decision=[4, 1, 1, 1, 1])
    # l53, l54, l55, l56, l57 = sch.split(loop=l34, factors=[v48, v49, v50, v51, v52])
    # v58, v59, v60, v61, v62 = sch.sample_perfect_tile(n=5, loop=l35, max_innermost_factor=64, decision=[1, 1, 28, 7, 1])
    # l63, l64, l65, l66, l67 = sch.split(loop=l35, factors=[v58, v59, v60, v61, v62])
    # v68, v69, v70, v71, v72 = sch.sample_perfect_tile(n=5, loop=l36, max_innermost_factor=64, decision=[8, 4, 4, 1, 1])
    # l73, l74, l75, l76, l77 = sch.split(loop=l36, factors=[v68, v69, v70, v71, v72])
    # v78, v79, v80 = sch.sample_perfect_tile(n=3, loop=l37, max_innermost_factor=64, decision=[8, 4, 4])
    # l81, l82, l83 = sch.split(loop=l37, factors=[v78, v79, v80])
    # sch.reorder(*[l43, l53, l63, l73, l44, l54, l64, l74, l45, l55, l65, l75, l81, l82, l46, l56, l66, l76, l83, l47,
    #               l57, l67, l77])
    # l84 = sch.fuse(*[l43, l53, l63, l73])
    # sch.bind(loop=l84, thread="blockIdx.x")
    # l85 = sch.fuse(*[l44, l54, l64, l74])
    # sch.bind(loop=l85, thread="vthread")
    # l86 = sch.fuse(*[l45, l55, l65, l75])
    # sch.bind(loop=l86, thread="threadIdx.x")
    # b87 = sch.cache_read(block=b32, i=2, storage_scope="shared")
    # sch.compute_at(block=b87, loop=l81, preserve_unit_loop=1)
    # l88, l89, l90, l91, l92, l93, l94, l95 = sch.get_axes(block=b87)
    # l96 = sch.fuse(*[l92, l93, l94, l95])
    # v97, v98 = sch.sample_perfect_tile(n=2, loop=l96, max_innermost_factor=4, decision=[2048, 1])
    # l99, l100 = sch.split(loop=l96, factors=[v97, v98])
    # sch.vectorize(loop=l100)
    # # sch.mark_loop(loop=l99, ann_key="loop_type", ann_val="lazy_cooperative_fetch")
    # b101 = sch.cache_read(block=b32, i=1, storage_scope="shared")
    # sch.compute_at(block=b101, loop=l81, preserve_unit_loop=1)
    # l102, l103, l104, l105, l106, l107, l108, l109 = sch.get_axes(block=b101)
    # l110 = sch.fuse(*[l106, l107, l108, l109])
    # v111, v112 = sch.sample_perfect_tile(n=2, loop=l110, max_innermost_factor=4, decision=[28672, 4])
    # l113, l114 = sch.split(loop=l110, factors=[v111, v112])
    # sch.vectorize(loop=l114)
    # # sch.mark_loop(loop=l113, ann_key="loop_type", ann_val="lazy_cooperative_fetch")
    # sch.reverse_compute_at(block=b31, loop=l86, preserve_unit_loop=1)
    # b115 = sch.get_block(name="input_tile")
    # b116, = sch.get_consumers(block=b115)
    # l117, l118, l119, l120, l121, l122, l123, l124 = sch.get_axes(block=b116)
    # sch.compute_at(block=b115, loop=l120, preserve_unit_loop=1)
    # b125 = sch.get_block(name="T_add")
    # sch.compute_inline(block=b125)
    # b126 = sch.get_block(name="conv2d_winograd")
    # sch.compute_inline(block=b126)
    # b127 = sch.get_block(name="A")
    # sch.compute_inline(block=b127)
    # b128 = sch.get_block(name="B")
    # sch.compute_inline(block=b128)
    # b129 = sch.get_block(name="data_pad")
    # sch.compute_inline(block=b129)
    # b130 = sch.get_block(name="root")
    # v131 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.2, 0.2, 0.2, 0.2, 0.2], decision=4)
    # sch.mark_block(block=b130, ann_key="auto_unroll_explicit", ann_val=v131)
    # b132 = sch.get_block(name="data_pack_shared")
    # l133, l134, l135, l136, l137, l138 = sch.get_axes(block=b132)
    # l139, l140 = sch.split(loop=l137, factors=[None, 112])
    # sch.bind(loop=l140, thread="threadIdx.x")
    # b141 = sch.get_block(name="placeholder_1_shared")
    # l142, l143, l144, l145, l146, l147 = sch.get_axes(block=b141)
    # l148, l149 = sch.split(loop=l146, factors=[None, 112])
    # sch.bind(loop=l149, thread="threadIdx.x")
    # b150 = sch.get_block(name="input_tile")
    # l151, l152, l153, l154, l155, l156, l157, l158 = sch.get_axes(block=b150)
    # l159 = sch.fuse(l151, l152, l153, l154)
    # l160, l161 = sch.split(loop=l159, factors=[None, 64])
    # sch.bind(loop=l160, thread="blockIdx.x")
    # sch.bind(loop=l161, thread="threadIdx.x")
    # b162 = sch.get_block(name="inverse")
    # l163, l164, l165, l166, l167, l168, l169, l170 = sch.get_axes(block=b162)
    # l171 = sch.fuse(l163, l164, l165, l166)
    # l172, l173 = sch.split(loop=l171, factors=[None, 64])
    # sch.bind(loop=l172, thread="blockIdx.x")
    # sch.bind(loop=l173, thread="threadIdx.x")
    # b174 = sch.get_block(name="T_relu")
    # l175, l176, l177, l178 = sch.get_axes(block=b174)
    # l179 = sch.fuse(l175, l176, l177, l178)
    # l180, l181 = sch.split(loop=l179, factors=[None, 64])
    # sch.bind(loop=l180, thread="blockIdx.x")
    # sch.bind(loop=l181, thread="threadIdx.x")
    # b182 = sch.get_block(name="root")
    # b183 = sch.get_block(name="input_tile")
    # l184, l185, l186, l187, l188, l189 = sch.get_axes(block=b183)
    # sch.mark_loop(loop=l184, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    # sch.mark_loop(loop=l184, ann_key="pragma_unroll_explicit", ann_val=1)
    # b190 = sch.get_block(name="data_pack")
    # l191, l192, l193, l194, l195, l196 = sch.get_axes(block=b190)
    # sch.mark_loop(loop=l191, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    # sch.mark_loop(loop=l191, ann_key="pragma_unroll_explicit", ann_val=1)
    # b197 = sch.get_block(name="data_pack_shared")
    # l198, l199, l200, l201, l202, l203, l204 = sch.get_axes(block=b197)
    # sch.mark_loop(loop=l198, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    # sch.mark_loop(loop=l198, ann_key="pragma_unroll_explicit", ann_val=1)
    # b205 = sch.get_block(name="placeholder_1_shared")
    # l206, l207, l208, l209, l210, l211, l212 = sch.get_axes(block=b205)
    # sch.mark_loop(loop=l206, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    # sch.mark_loop(loop=l206, ann_key="pragma_unroll_explicit", ann_val=1)
    # b213 = sch.get_block(name="bgemm")
    # l214, l215, l216, l217, l218, l219, l220, l221, l222, l223, l224, l225, l226, l227 = sch.get_axes(block=b213)
    # sch.mark_loop(loop=l214, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    # sch.mark_loop(loop=l214, ann_key="pragma_unroll_explicit", ann_val=1)
    # b228 = sch.get_block(name="bgemm_local")
    # l229, l230, l231, l232, l233, l234, l235 = sch.get_axes(block=b228)
    # sch.mark_loop(loop=l229, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    # sch.mark_loop(loop=l229, ann_key="pragma_unroll_explicit", ann_val=1)
    # # b236 = sch.get_block(name="inverse")
    # # l237, l238, l239, l240, l241, l242 = sch.get_axes(block=b236)
    # # sch.mark_loop(loop=l237, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    # # sch.mark_loop(loop=l237, ann_key="pragma_unroll_explicit", ann_val=1)
    # # b243 = sch.get_block(name="T_relu")
    # # l244, l245 = sch.get_axes(block=b243)
    # # sch.mark_loop(loop=l244, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    # # sch.mark_loop(loop=l244, ann_key="pragma_unroll_explicit", ann_val=1)
    # b246 = sch.get_block(name="data_pack")
    # l247, l248, l249, l250, l251, l252 = sch.get_axes(block=b246)
    # b253 = sch.decompose_reduction(block=b246, loop=l251)
    # b254 = sch.get_block(name="bgemm")
    # l255, l256, l257, l258, l259, l260, l261, l262, l263, l264, l265, l266, l267, l268 = sch.get_axes(block=b254)
    # b269 = sch.decompose_reduction(block=b254, loop=l258)
    # b270 = sch.get_block(name="inverse")
    # l271, l272, l273, l274, l275, l276 = sch.get_axes(block=b270)
    # b277 = sch.decompose_reduction(block=b270, loop=l275)
    # # # print(SPACE.postprocess(task=task,sch=sch))
    # # # # for trace in sch.trace.as_python():
    # # # #     print(trace)
    # print(tvm.script.asscript(sch.mod["main"]))
    # feature = ms.feature.per_store_feature_batched([sch])[0]
    # np.savetxt('feature_ms.csv',feature,delimiter=',',fmt='%.6f')

    #
    #
    # supports=SPACE.get_support(task)
    # for sch in supports:
    #     print("trace:")
    #     for trace in sch.trace.as_python():
    #         print(trace)
    #     print(tvm.script.asscript(sch.mod))
    #     print(SPACE.postprocess(task=task,sch=sch))

    measures = [512, 704, 512, 512, 2000, 1654, 768, 512, 512, 2500, 2240, 1472, 512, 512, 1920, 1088, 832, 512, 512,
                1344, 640, 640, 512,
                512, 512, 512, 576, 704, 1088]
    ct = 0
    for func in tir_funcs.values():
        if ct == 14:
            sch = ms.autotune(
                task=ms.SearchTask(
                    workload=func,
                    target=target,
                    target_host=TARGET_HOST,
                    log_file=log,
                ),
                space=SPACE,
                strategy=ms.strategy.Evolutionary(
                    total_measures=measures[ct],
                    num_measures_per_iter=64,
                    population=2048,
                    init_measured_ratio=0.2,
                    genetic_algo_iters=4,
                    p_mutate=0.85,
                    mutator_probs={
                        ms.mutator.mutate_tile_size(): 0.90,
                        ms.mutator.mutate_auto_unroll(): 0.10,
                    },
                    cost_model=ms.XGBModel(xgb_eta=0.2),
                    eps_greedy=0.25,
                ),
                measurer=ms.ProgramMeasurer(
                    measure_callbacks=[
                        ms.RecordToFile(),
                    ]
                )
            )
            task = ms.SearchTask(workload=func, task_name="conv2d", target=target)
        # print(SPACE.postprocess(task=task,sch=sch))
        # print(tvm.script.asscript(sch.mod["main"]))
        ct += 1

    # lower=tvm.lower(sch.mod["main"],None)
    # print(lower)
    # func = tvm.driver.build(sch.mod["main"], target=target,target_host=tvm.target.Target(
    #     "llvm"))
    # # print(func.imported_modules[0].get_source())
    #
    # tmp=tempdir()
    # func.export_library(tmp.relpath("net.tar"))
    # # for _ in range(4):
    # remote = auto_scheduler.utils.request_remote(RPC_KEY, "172.16.2.241", 4445, timeout=10000)
    # remote.upload(tmp.relpath("net.tar"))
    # rlib = remote.load_module("net.tar")
    # ctx = remote.gpu()
    # # module = graph_runtime.GraphModule(rlib[rlib.entry_name](ctx))
    # ftimer = rlib.time_evaluator(rlib.entry_name, ctx, repeat=1, min_repeat_ms=300)
    # #
    # data_np = np.random.uniform(size=([1, 28, 28, 128])).astype(np.float32)
    # placeholder1_np = np.random.uniform(size=(4, 4, 128, 128)).astype(np.float32)
    # placeholder2_np=np.random.uniform(size=(1, 1, 1, 128)).astype(np.float32)
    # # placeholder3_np=np.random.uniform(size=(1, 1, 1, 2048)).astype(np.float32)
    # # placeholder4_np=np.random.uniform(size=(1, 1, 1, 2048)).astype(np.float32)
    #
    #
    # #
    # data_tvm = tvm.nd.array(data_np, ctx=ctx)
    # weight_tvm = tvm.nd.array(placeholder1_np, ctx=ctx)
    # placeholder2_tvm=tvm.nd.array(placeholder2_np,ctx=ctx)
    # # placeholder3_tvm=tvm.nd.array(placeholder3_np,ctx=ctx)
    # # placeholder4_tvm=tvm.nd.array(placeholder4_np,ctx=ctx)
    # out_tvm = tvm.nd.empty( (1, 28, 28, 128), ctx=ctx)
    # # func(data_tvm, weight_tvm, out_tvm)
    # # np.testing.assert_allclose(out_np, out_tvm.asnumpy(), rtol=1e-3)
    # print("tir:")
    # perf=ftimer(data_tvm, weight_tvm, placeholder2_tvm,out_tvm).results
    # print(perf)
    # print(
    #     "Execution time of this operator: %.3f ms"
    #     % (np.median(perf) * 1000)
    # )

    # with ms.ApplyHistoryBest(log, SPACE):
    #     with tvm.transform.PassContext(opt_level=3, config={"relay.with_tir_schedule": True,
    #                                                         "relay.backend.use_meta_schedule": True}):
    #         lib = relay.build_module.build(mod, target, params=params)
    #
    # def run_module(lib):
    #     ctx = tvm.context(str(target), 0)
    #     module = runtime.GraphModule(lib["default"](ctx))
    #     data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    #     module.set_input("data", data_tvm)
    #
    #     # Evaluate
    #     print("Evaluate inference time cost...")
    #     ftimer = module.module.time_evaluator("run", ctx, repeat=3, min_repeat_ms=500)
    #     prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
    #     print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))
    #
    #     module.run()
    #     return module.get_output(0)
    #
    # std = run_module(lib_std).asnumpy()
    # out = run_module(lib).asnumpy()
    # np.testing.assert_allclose(out, std, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_end_to_end_resnet("winograd14_ms.json")
