import tvm
import numpy as np
import topi

def check_correctness(s, args, inserted_pass, target='llvm', std_func = None):
    """Check correctness by building the function with and without inserted_pass"""

    if isinstance(s, tuple) or isinstance(s, list):
        s1, s2 = s
    else:
        s1 = s2 = s

    if isinstance(target, tuple) or isinstance(target, list):
        target1, target2 = target
    else:
        target1 = target2 = target

    with tvm.build_config(add_lower_pass=[(0, inserted_pass)]):
        func1 = tvm.build(s1, args, target1)

    func2 = std_func if std_func else tvm.build(s2, args, target2)

    ctx1 = tvm.context(target1)
    ctx2 = tvm.context(target2)

    bufs1 = [tvm.nd.array(np.random.randn(*topi.util.get_const_tuple(x.shape)).astype(x.dtype), ctx=ctx1)
             for x in args]
    bufs2 = [tvm.nd.array(x, ctx=ctx2) for x in bufs1]

    func1(*bufs1)
    func2(*bufs2)

    bufs1_np = [x.asnumpy() for x in bufs1]
    bufs2_np = [x.asnumpy() for x in bufs2]

    for x, y in zip(bufs1_np, bufs2_np):
        np.testing.assert_allclose(x, y)