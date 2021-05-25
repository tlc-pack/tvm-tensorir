import logging
import numpy as np

logger = logging.getLogger(__name__)


class TemplateLogger:
    __slots__ = ['fname']

    def __init__(self, fname, force_overwrite):
        self.fname = fname
        if force_overwrite:
            self.write_header()

    def write_header(self):
        with open(self.fname, 'w') as fout:
            fout.write("""\
import tvm
from tvm import te
""")

    def write_template(self, kernel_name, kernel_args, pysched):
        with open(self.fname, 'a') as fout:
            fout.write('def {}({}):\n'.format(kernel_name, kernel_args))
            fout.write('\n'.join(['    ' + line for line in pysched.split('\n')[:-1]]))
            fout.write('\n\n\n')


class AvgStdMedianLogger:
    __slots__ = ['filename']

    def __init__(self, filename, force_overwrite):
        self.filename = filename
        if force_overwrite:
            self.write_header()

    def write_header(self):
        with open(self.filename, 'w') as fout:
            fout.write('Kernel,ShapeTuple,Avg,STD,Median\n')

    def write(self, kernel, attr, avg, std, median):
        with open(self.filename, 'a') as fout:
            fout.write('\"{}\",\"{}\",{},{},{}\n'
                       .format(kernel, attr, avg, std, median))

    def write_null(self, kernel, attr):
        with open(self.filename, 'a') as fout:
            fout.write('\"{}\",\"{}\",-,-,-\n'.format(kernel, attr))


class TFLOPSLogger:
    __slots__ = ['tflops_logger']

    def __init__(self, filename, force_overwrite):
        if filename == "":
            self.tflops_logger = None
        self.tflops_logger = AvgStdMedianLogger(filename + '.csv', force_overwrite)

    def write(self, kernel, attr, results, FLOPs):
        if self.tflops_logger is None:
            return
        if results is None:
            self.tflops_logger.write_null(kernel, attr)
            logger.info("{} : (-)((-) in TFLOPS)".format(kernel))
            return
        TFLOPS = FLOPs * 1e-12 / np.array(results)
        avg, std, median = np.average(TFLOPS), np.std(TFLOPS), np.median(TFLOPS)
        self.tflops_logger.write(kernel, attr, avg, std, median)
        logger.info("{} : {}+{} (M={}) in TFLOPS)".format(kernel, avg, std, median))
