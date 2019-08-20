"""Common utilities for test"""

import numpy as np
import tvm
import topi
import time

def check_correctness(s, args, inserted_pass, target='llvm', func2=None, return_stmt=False, print_time=False):
    """Check correctness by building the function with and without inserted_pass

    s: schedule
    args: List[Tensor]
    target : string or list of string
    return_stmt : bool
        whether return the lowered Stmt
    func2: Module
        If is not None, will use this instead of compiling the schedule
    """

    if isinstance(target, tuple) or isinstance(target, list):
        target1, target2 = target
    else:
        target1 = target2 = target

    with tvm.build_config(add_lower_pass=[(0, inserted_pass)]):
        func1 = build_time(s, args, target1, "tensorir", print_time)

    func2 = func2 or build_time(s, args, target2, "origin", print_time)

    ctx1 = tvm.context(target1)
    ctx2 = tvm.context(target2)

    bufs1 = [tvm.nd.array(np.array(np.random.randn(*topi.util.get_const_tuple(x.shape)),
                                   dtype=x.dtype), ctx=ctx1) for x in args]
    bufs2 = [tvm.nd.array(x, ctx=ctx2) for x in bufs1]

    func1(*bufs1)
    func2(*bufs2)

    bufs1_np = [x.asnumpy() for x in bufs1]
    bufs2_np = [x.asnumpy() for x in bufs2]

    for x, y in zip(bufs1_np, bufs2_np):
        np.testing.assert_allclose(x, y, atol=1e-4)

    if return_stmt:
        with tvm.build_config(add_lower_pass=[(0, inserted_pass)]):
            return tvm.lower(s, args, target1, simple_mode=True)


def build_time(s, args, target, name, print_time=True):
    start_time = time.time()
    ret = tvm.build(s, args, target)
    end_time = time.time()
    if print_time:
        print('%s: %.2fs' % (name, end_time - start_time))
    return ret
