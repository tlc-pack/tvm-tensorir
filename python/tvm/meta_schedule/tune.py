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
"""User-facing Tuning API"""

from contextlib import contextmanager
import logging
import os.path
import tempfile
from typing import Callable, Generator, List, Optional, Union

from tvm.ir.module import IRModule
from tvm.target.target import Target
from tvm.te import Tensor, create_prim_func
from tvm.tir import PrimFunc, Schedule

from . import schedule_rule
from . import measure_callback
from . import postproc
from .builder import Builder, LocalBuilder
from .database import Database, JSONDatabase, TuningRecord
from .measure_callback import MeasureCallback
from .runner import LocalRunner, Runner
from .search_strategy import ReplayFuncConfig, ReplayTraceConfig
from .space_generator import PostOrderApply
from .task_scheduler import RoundRobin, TaskScheduler
from .tune_context import TuneContext


logger = logging.getLogger(__name__)


SearchStrategyConfig = Union[
    ReplayFuncConfig,
    ReplayTraceConfig,
]

TYPE_F_TUNE_CONTEXT = Callable[  # pylint: disable=invalid-name
    [
        IRModule,
        Target,
        SearchStrategyConfig,
        str,
    ],
    TuneContext,
]

TYPE_F_TASK_SCHEDULER = Callable[  # pylint: disable=invalid-name
    [
        List[TuneContext],
        Builder,
        Runner,
        Database,
        List[MeasureCallback],
    ],
    TaskScheduler,
]


def _parse_mod(mod: Union[PrimFunc, IRModule]) -> IRModule:
    if isinstance(mod, PrimFunc):
        mod = mod.with_attr("global_symbol", "main")
        mod = mod.with_attr("tir.noalias", True)
        mod = IRModule({"main": mod})
    if not isinstance(mod, IRModule):
        raise TypeError(f"Expected `mod` to be PrimFunc or IRModule, but gets: {mod}")
    return mod


def _parse_target(target: Union[str, Target]) -> Target:
    if isinstance(target, str):
        target = Target(target)
    if not isinstance(target, Target):
        raise TypeError(f"Expected `target` to be str or Target, but gets: {target}")
    return target


@contextmanager
def _work_dir_context(work_dir: Optional[str]) -> Generator[str, None, None]:
    if work_dir is not None and not os.path.isdir(work_dir):
        raise ValueError(f"`work_dir` must be a directory, but gets: {work_dir}")
    temp_dir = None
    try:
        if work_dir is not None:
            yield work_dir
        else:
            temp_dir = tempfile.TemporaryDirectory()
            yield temp_dir.name
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def _parse_builder(builder: Optional[Builder]) -> Builder:
    if builder is None:
        builder = LocalBuilder()
    if not isinstance(builder, Builder):
        raise TypeError(f"Expected `builder` to be Builder, but gets: {builder}")
    return builder


def _parse_runner(runner: Optional[Runner]) -> Runner:
    if runner is None:
        runner = LocalRunner()
    if not isinstance(runner, Runner):
        raise TypeError(f"Expected `runner` to be Runner, but gets: {runner}")
    return runner


def _parse_database(database: Optional[Database], path: str) -> Database:
    if database is None:
        database = JSONDatabase(
            path_workload=os.path.join(path, "workload.json"),
            path_tuning_record=os.path.join(path, "tuning_record.json"),
        )
    if not isinstance(database, Database):
        raise TypeError(f"Expected `database` to be Database, but gets: {database}")
    return database


def _parse_measure_callbacks(
    measure_callbacks: Optional[List[MeasureCallback]],
) -> List[MeasureCallback]:
    if measure_callbacks is None:
        measure_callbacks = [
            measure_callback.AddToDatabase(),
            measure_callback.RemoveBuildArtifact(),
            measure_callback.EchoStatistics(),
        ]
    if not isinstance(measure_callbacks, (list, tuple)):
        raise TypeError(
            f"Expected `measure_callbacks` to be List[MeasureCallback], "
            f"but gets: {measure_callbacks}"
        )
    measure_callbacks = list(measure_callbacks)
    for i, callback in enumerate(measure_callbacks):
        if not isinstance(callback, MeasureCallback):
            raise TypeError(
                f"Expected `measure_callbacks` to be List[MeasureCallback], "
                f"but measure_callbacks[{i}] is: {callback}"
            )
    return measure_callbacks


def _parse_f_tune_context(f_tune_context: Optional[TYPE_F_TUNE_CONTEXT]) -> TYPE_F_TUNE_CONTEXT:
    def default_llvm(
        mod: IRModule,
        target: Target,
        config: SearchStrategyConfig,
        task_name: str,
    ) -> TuneContext:
        return TuneContext(
            mod=mod,
            target=target,
            space_generator=PostOrderApply(),
            search_strategy=config.create_strategy(),
            sch_rules=[
                schedule_rule.AutoInline(
                    into_producer=False,
                    into_consumer=True,
                    into_cache_only=False,
                    inline_const_tensor=True,
                    disallow_if_then_else=True,
                    require_injective=True,
                    require_ordered=True,
                    disallow_op=["tir.exp"],
                ),
                schedule_rule.MultiLevelTiling(
                    structure="SSRSRS",
                    tile_binds=None,
                    max_innermost_factor=64,
                    vector_load_max_len=None,
                    reuse_read=None,
                    reuse_write=schedule_rule.ReuseType(
                        req="may",
                        levels=[1, 2],
                        scope="global",
                    ),
                ),
                schedule_rule.ParallelizeVectorizeUnroll(
                    max_jobs_per_core=16,
                    max_vectorize_extent=32,
                    unroll_max_steps=[0, 16, 64, 512],
                    unroll_explicit=True,
                ),
            ],
            postprocs=[
                postproc.RewriteParallelVectorizeUnroll(),
                postproc.RewriteReductionBlock(),
            ],
            mutator_probs=None,
            task_name=task_name,
            rand_state=-1,
            num_threads=None,
        )

    def default_cuda(
        mod: IRModule,
        target: Target,
        config: SearchStrategyConfig,
        task_name: str,
    ) -> TuneContext:
        return TuneContext(
            mod=mod,
            target=target,
            space_generator=PostOrderApply(),
            search_strategy=config.create_strategy(),
            sch_rules=[
                schedule_rule.AutoInline(
                    into_producer=False,
                    into_consumer=True,
                    into_cache_only=False,
                    inline_const_tensor=True,
                    disallow_if_then_else=False,
                    require_injective=False,
                    require_ordered=False,
                    disallow_op=None,
                ),
                schedule_rule.MultiLevelTiling(
                    structure="SSSRRSRS",
                    tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
                    max_innermost_factor=64,
                    vector_load_max_len=4,
                    reuse_read=schedule_rule.ReuseType(
                        req="must",
                        levels=[4],
                        scope="shared",
                    ),
                    reuse_write=schedule_rule.ReuseType(
                        req="must",
                        levels=[3],
                        scope="local",
                    ),
                ),
                schedule_rule.AutoInline(
                    into_producer=True,
                    into_consumer=True,
                    into_cache_only=True,
                    inline_const_tensor=True,
                    disallow_if_then_else=False,
                    require_injective=False,
                    require_ordered=False,
                    disallow_op=None,
                ),
                schedule_rule.ParallelizeVectorizeUnroll(
                    max_jobs_per_core=-1,  # disable parallelize
                    max_vectorize_extent=-1,  # disable vectorize
                    unroll_max_steps=[0, 16, 64, 512, 1024],
                    unroll_explicit=True,
                ),
            ],
            postprocs=[
                postproc.RewriteCooperativeFetch(),
                postproc.RewriteUnboundBlock(),
                postproc.RewriteParallelVectorizeUnroll(),
                postproc.RewriteReductionBlock(),
                postproc.VerifyGPUCode(),
            ],
            mutator_probs=None,
            task_name=task_name,
            rand_state=-1,
            num_threads=None,
        )

    def default(
        mod: IRModule,
        target: Target,
        config: SearchStrategyConfig,
        task_name: str,
    ) -> TuneContext:
        if target.kind.name == "llvm":
            return default_llvm(mod, target, config, task_name)
        if target.kind.name == "cuda":
            return default_cuda(mod, target, config, task_name)
        raise NotImplementedError(f"Unsupported target: {target.kind.name}")

    if f_tune_context is None:
        return default
    return f_tune_context


def _parse_f_task_scheduler(
    f_task_scheduler: Optional[TYPE_F_TASK_SCHEDULER],
) -> TYPE_F_TASK_SCHEDULER:
    def default(
        tasks: List[TuneContext],
        builder: Builder,
        runner: Runner,
        database: Database,
        measure_callbacks: List[MeasureCallback],
    ) -> TaskScheduler:
        return RoundRobin(
            tasks=tasks,
            builder=builder,
            runner=runner,
            database=database,
            measure_callbacks=measure_callbacks,
        )

    if f_task_scheduler is None:
        return default
    return f_task_scheduler


def tune_tir(
    mod: Union[IRModule, PrimFunc],
    target: Union[str, Target],
    config: SearchStrategyConfig,
    *,
    task_name: str = "main",
    work_dir: Optional[str] = None,
    builder: Optional[Builder] = None,
    runner: Optional[Runner] = None,
    database: Optional[Database] = None,
    measure_callbacks: Optional[List[MeasureCallback]] = None,
    f_tune_context: Optional[TYPE_F_TUNE_CONTEXT] = None,
    f_task_scheduler: Optional[TYPE_F_TASK_SCHEDULER] = None,
) -> Optional[Schedule]:
    """Tune a TIR IRModule with a given target.

    Parameters
    ----------
    mod : Union[IRModule, PrimFunc]
        The module to tune.
    target : Union[str, Target]
        The target to tune for.
    config : SearchStrategyConfig
        The search strategy config.
    task_name : str
        The name of the task.
    work_dir : Optional[str]
        The working directory to save intermediate results.
    builder : Optional[Builder]
        The builder to use.
    runner : Optional[Runner]
        The runner to use.
    database : Optional[Database]
        The database to use.
    measure_callbacks : Optional[List[MeasureCallback]]
        The callbacks used during tuning.
    f_tune_context : Optional[TYPE_F_TUNE_CONTEXT]
        The function to create TuneContext.
    f_task_scheduler : Optional[TYPE_F_TASK_SCHEDULER]
        The function to create TaskScheduler.

    Returns
    -------
    sch : Optional[Schedule]
        The tuned schedule.
    """

    with _work_dir_context(work_dir) as path:
        logger.info("Working directory: %s", path)
        mod = _parse_mod(mod)
        target = _parse_target(target)
        builder = _parse_builder(builder)
        runner = _parse_runner(runner)
        database = _parse_database(database, path)
        measure_callbacks = _parse_measure_callbacks(measure_callbacks)
        tune_context = _parse_f_tune_context(f_tune_context)(mod, target, config, task_name)
        task_scheduler = _parse_f_task_scheduler(f_task_scheduler)(
            [tune_context],
            builder,
            runner,
            database,
            measure_callbacks,
        )
        task_scheduler.tune()
        workload = database.commit_workload(mod)
        bests: List[TuningRecord] = database.get_top_k(workload, top_k=1)
        if not bests:
            return None
        assert len(bests) == 1
        sch = Schedule(mod)
        bests[0].trace.apply_to_schedule(sch, remove_postproc=False)
        return sch


def tune_te(
    tensors: List[Tensor],
    target: Union[str, Target],
    config: SearchStrategyConfig,
    *,
    task_name: str = "main",
    work_dir: Optional[str] = None,
    builder: Optional[Builder] = None,
    runner: Optional[Runner] = None,
    database: Optional[Database] = None,
    measure_callbacks: Optional[List[MeasureCallback]] = None,
    f_tune_context: Optional[TYPE_F_TUNE_CONTEXT] = None,
    f_task_scheduler: Optional[TYPE_F_TASK_SCHEDULER] = None,
) -> Optional[Schedule]:
    """Tune a TE compute DAG with a given target.

    Parameters
    ----------
    tensor : List[Tensor]
        The list of input/output tensors of the TE compute DAG.
    target : Union[str, Target]
        The target to tune for.
    config : SearchStrategyConfig
        The search strategy config.
    task_name : str
        The name of the task.
    work_dir : Optional[str]
        The working directory to save intermediate results.
    builder : Optional[Builder]
        The builder to use.
    runner : Optional[Runner]
        The runner to use.
    database : Optional[Database]
        The database to use.
    measure_callbacks : Optional[List[MeasureCallback]]
        The callbacks used during tuning.
    f_tune_context : Optional[TYPE_F_TUNE_CONTEXT]
        The function to create TuneContext.
    f_task_scheduler : Optional[TYPE_F_TASK_SCHEDULER]
        The function to create TaskScheduler.

    Returns
    -------
    sch : Optional[Schedule]
        The tuned schedule.
    """
    return tune_tir(
        mod=create_prim_func(tensors),
        target=target,
        config=config,
        task_name=task_name,
        work_dir=work_dir,
        builder=builder,
        runner=runner,
        database=database,
        measure_callbacks=measure_callbacks,
        f_tune_context=f_tune_context,
        f_task_scheduler=f_task_scheduler,
    )
