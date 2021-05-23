import tvm
from tvm import tir

import logging
import numpy as np
import os
logger = logging.getLogger(__name__)

from ..shared import ansor, utils, get_time_evaluator_results, CUDATarget
from ..shared.logger import TemplateLogger, TFLOPSLogger

from .wkl_def import Dense
from .fixture import cuBLASDenseFixture, cuTLASSDenseFixture
from .ansor_saved_schedules import *


def dense_kernel_name(B, T, I, H):
    return 'dense_{}x{}x{}x{}'.format(B, T, I, H)


def test_static_codegen(pytestconfig):
    """
    Kernel Template Generator
    """
    B = pytestconfig.getoption('B')
    T = pytestconfig.getoption('T')
    I = pytestconfig.getoption('I')
    H = pytestconfig.getoption('H')
    FLOPs = 2.0 * B * T * I * H
    template_pysched_fname = pytestconfig.getoption('filename') + '.py.log'
    force_overwrite = pytestconfig.getoption('force_overwrite')
    kernel_name = dense_kernel_name(B, T, I, H)

    template_logger = TemplateLogger(template_pysched_fname, force_overwrite)

    try:
        exec(kernel_name)
    except NameError:
        pass
    else:
        logger.warn("Kernel {} has already been auto-scheduled before".format(kernel_name))
        return

    cublas_fixture = cuBLASDenseFixture(B * T, I, H)
    (sched, in_args), pysched = ansor.auto_schedule(func=Dense, args=(B * T, I, H))
    cuda_kernel = tvm.build(sched, in_args, target=CUDATarget)
    module_data = cublas_fixture.module_data()
    cuda_kernel(*module_data)
    # correctness checking
    np.testing.assert_allclose(module_data[-1].asnumpy(),
                               cublas_fixture.Y_np_expected,
                               rtol=1e-3, atol=1e-3)
    # schedule logging
    logger.info("{}".format(tvm.lower(sched, in_args, simple_mode=True)))
    logger.info("{}".format(cuda_kernel.imported_modules[0].get_source()))
    template_logger.write_template(kernel_name, 'X, W, T_dense, s', pysched)
    # performance comparison
    cublas_avg = np.average(
            get_time_evaluator_results(cublas_fixture.cublas_kernel, module_data))
    ansor_avg = np.average(get_time_evaluator_results(cuda_kernel, module_data))
    logger.info("Runtime Measurements: {} TFLOPS vs. cuBLAS {} TFLOPS"
                .format(FLOPs * 1e-12 / ansor_avg, FLOPs * 1e-12 / cublas_avg))


def test_dynamic_codegen(pytestconfig):
    B = 16
    T = np.arange(1, 129)
    IH = [(768, 2304), (768, 768), (768, 3072), (3072, 768)]
    (sched, in_args), pysched = ansor.auto_schedule(
            func=Dense, args=utils.cross_product(list(B * T), IH))


def test_dynamic_codegen_any(pytestconfig):
    B = 16
    T = tir.Any()
    IH = [(768, 2304), (768, 768), (768, 3072), (3072, 768)]
    (sched, in_args), pysched = ansor.auto_schedule(
            func=Dense, args=utils.cross_product(list(B * T), IH))


def test_perf(pytestconfig):
    B = pytestconfig.getoption('B')
    T = pytestconfig.getoption('T')
    I = pytestconfig.getoption('I')
    H = pytestconfig.getoption('H')
    logger.info("dense_{}x{}x{}x{}".format(B, T, I, H))
    FLOPs = 2 * B * T * I * H

    enable_nvprof = pytestconfig.getoption('nvprof')
    tflops_logger = TFLOPSLogger(pytestconfig.getoption('filename'),
                                 pytestconfig.getoption('force_overwrite'))
    template = pytestconfig.getoption('template')

    dyT = pytestconfig.getoption('dyT')
    dyI = pytestconfig.getoption('dyI')
    dyH = pytestconfig.getoption('dyH')

    if dyT:
        shape_tuple = "{}".format(T)
    elif dyI and dyH:
        shape_tuple = "({}, {})".format(I, H)
    else:
        shape_tuple = "({}, {}, {})".format(T, I, H)

    cublas_fixture = cuBLASDenseFixture(B=B*T, I=I, H=H)
    cutlass_fixture = cuTLASSDenseFixture(cublas_fixture)
    logger.info("Created the cuBLAS and CUTLASS fixture")

    module_data = cublas_fixture.module_data()
    if not enable_nvprof:
        [X, W, Y] = Dense(B * T, I, H)
        sched = tvm.te.create_schedule(Y.op)
        template_not_defined = False
        try:
            eval('{}(X, W, Y, s=sched)'.format(dense_kernel_name(B, T, I, H)))
            cuda_kernel = tvm.build(sched, [X, W, Y], target=CUDATarget)
        except NameError:
            template_not_defined = True

        cublas_results = \
                get_time_evaluator_results(cublas_fixture.cublas_kernel, module_data)
        tflops_logger.write('cuBLAS', shape_tuple, cublas_results, FLOPs)
        cutlass_results = \
                get_time_evaluator_results(cutlass_fixture.cutlass_kernel,
                                           module_data)
        tflops_logger.write('CUTLASS', shape_tuple, cutlass_results, FLOPs)
        if not template_not_defined:
            ansor_baseline_results = get_time_evaluator_results(cuda_kernel, module_data)
            tflops_logger.write('Ansor', shape_tuple, ansor_baseline_results, FLOPs)
        else:
            tflops_logger.write('Ansor', shape_tuple, None, FLOPs)

    os.environ["DIETCODE_SCHED_OPT"] = '1'

    for mode in ['JIT', ]:
        if mode == 'JIT':
            [X, W, Y] = Dense(B=B*T, I=I, H=H)
            args_ext = [X, W, Y]
            module_data_ext = module_data
        else:
            DyT = tir.DynamicAxis('T', [1013]) if dyT else T
            DyI = tir.DynamicAxis('I', [1013]) if dyI else I
            DyH = tir.DynamicAxis('H', [1013]) if dyH else H
            [X, W, Y] = Dense(B=B*DyT, I=DyI, H=DyH)
            args_ext, module_data_ext = [], []
            for axis in ['T', 'I', 'H']:
                if eval('dy{}'.format(axis)):
                    args_ext.append(eval('Dy{}'.format(axis)))
                    module_data_ext.append(eval('{}'.format(axis)))
            args_ext = args_ext + [X, W, Y]
            module_data_ext = module_data_ext + module_data

        sched = tvm.te.create_schedule(Y.op)
        eval('{}(X, W, Y, s=sched)'.format('dense_' + template))
        cuda_kernel = tvm.build(sched, args_ext, target=CUDATarget)

        cuda_kernel_src = cuda_kernel.imported_modules[0].get_source()
        if os.getenv("VERBOSE", "0") == "1":
            logger.info("{}".format(tvm.lower(sched, args_ext, simple_mode=True)))
            logger.info("{}".format(cuda_kernel_src))
        with open("scratchpad_{}.cu".format(mode), 'w') as fout:
            fout.write("{}".format(cuda_kernel_src))

        cuda_kernel(*module_data_ext)
        np.testing.assert_allclose(module_data_ext[-1].asnumpy(),
                                   cublas_fixture.Y_np_expected,
                                   rtol=1e-3, atol=1e-3)

        if not enable_nvprof:
            dietcode_results = get_time_evaluator_results(cuda_kernel, module_data_ext)
            tflops_logger.write('DietCode_{}'.format(mode), shape_tuple,
                                dietcode_results, FLOPs)
    # for mode in ['JIT', 'AOT']
