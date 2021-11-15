def build_with_tensorrt(mod, target, params):
    from tvm.ir.transform import PassContext
    from tvm.relay.op.contrib import tensorrt
    from tvm.relay.build_module import (  # pylint: disable=import-outside-toplevel
        _build_module_no_factory as relay_build,
    )

    mod, config = tensorrt.partition_for_tensorrt(mod, params)
    with PassContext(
        opt_level=3,
        config={"relay.ext.tensorrt.options": config},
    ):
        return relay_build(mod, target=target, target_host=None, params=params)


def build_without_tensorrt(mod, target, params):
    from tvm.relay.build_module import (  # pylint: disable=import-outside-toplevel
        _build_module_no_factory as relay_build,
    )

    return relay_build(mod, target=target, target_host=None, params=params)


def run_with_graph_executor(rt_mod, device, evaluator_config, repeated_args):
    import itertools
    from tvm.contrib.graph_executor import GraphModule

    rt_mod = GraphModule(rt_mod["default"](device))

    evaluator = rt_mod.module.time_evaluator(
        func_name="run",
        dev=device,
        number=evaluator_config.number,
        repeat=evaluator_config.repeat,
        min_repeat_ms=evaluator_config.min_repeat_ms,
        f_preproc="cache_flush_cpu_non_first_arg"
        if evaluator_config.enable_cpu_cache_flush
        else "",
    )
    repeated_costs = []
    for args in repeated_args:
        profile_result = evaluator(*args)
        repeated_costs.append(profile_result.results)

    costs = [float(cost) for cost in itertools.chain.from_iterable(repeated_costs)]
    return costs
