# pylint: disable=missing-docstring
import tvm
from tvm import auto_scheduler, te

from tvm.meta_schedule.testing.te_workload import CONFIGS


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matmul_add(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)
    C = te.placeholder((N, M), name="C", dtype=dtype)
    k = te.reduce_axis((0, L), name="k")
    matmul = te.compute(
        (N, M),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="matmul",
        attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
    )
    out = te.compute((N, M), lambda i, j: matmul[i, j] + C[i, j], name="out")
    return [A, B, C, out]


def main():
    workload = "GMM"
    log_file = f"{workload}.json"
    workload_func, params = CONFIGS[workload]
    params = params[0]
    workload_func = auto_scheduler.register_workload(workload_func)
    target = tvm.target.Target("llvm")
    task = auto_scheduler.SearchTask(
        func=workload_func,
        args=params,
        target=target,
    )
    runner = auto_scheduler.RPCRunner(
        key=None,
        host=None,
        port=None,
        n_parallel=1,
    )

    # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=10,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
        runner=None,
    )
    print("Running AutoTuning:")
    task.tune(tune_option)
    print(task.print_best(log_file))
    sch, args = task.apply_best(log_file)

    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))


if __name__ == "__main__":
    main()
