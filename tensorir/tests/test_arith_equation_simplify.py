import tvm

def _get_vars(names, dtype='int64'):
    return [tvm.var(x, dtype=dtype) for x in names]

def _is_equal(lhs, rhs):
    diff = tvm.ir_pass.CanonicalSimplify(lhs - rhs)
    return isinstance(diff, tvm.expr.IntImm) and diff.value == 0

def test_linear_split():
    """
    eqs:
    p = io * 8 + ii
    q = jo * 8 + ji
    
    to simplify:
    io * 16 + ii * 2 -> p * 2
    jo * 8 + ji + io * 8 + ii -> q + p
    """
    analyzer = tvm.arith.Analyzer()

    p, q, io, ii, jo, ji = _get_vars(['p', 'q', 'io', 'ii', 'jo', 'ji'])

    analyzer.bind(p, io * 8 + ii)
    analyzer.bind(q, jo * 8 + ji)

    expr = analyzer.equation_simplify(io * 16 + ii * 2)
    assert _is_equal(expr, p * 2)

    expr = analyzer.equation_simplify(jo * 8 + ji + io * 8 + ii)
    assert _is_equal(expr, p + q)

def test_reduction():
    """
    eqs:
    p = io * 8 + ii
    q = jo * 8 + ji

    to simplify:
    io * 8 + ii -> p
    jo * 8 + ji -> q
    ko * 8 + ki -> ko * 8 + ki
    """
    analyzer = tvm.arith.Analyzer()

    p, q, io, ii, jo, ji, ko, ki = _get_vars(['p', 'q', 'io', 'ii', 'jo', 'ji', 'ko', 'ki'])

    analyzer.bind(p, io * 8 + ii)
    analyzer.bind(q, jo * 8 + ji)

    expr = analyzer.equation_simplify(io * 8 + ii)
    assert _is_equal(expr, p)

    expr = analyzer.equation_simplify(jo * 8 + ji)
    assert _is_equal(expr, q)

    expr = analyzer.equation_simplify(ko * 8 + ki)
    assert _is_equal(expr, ko * 8 + ki)

def test_matching():
    """
    eqs:
    p = a + 1
    q = b + 1

    to simplify:
    (a + 1) * (b + 1) -> p * q
    (a + 1) % 8 + (b + 1) / 8 * 8 -> p % 8 + q/8 * 8
    """
    analyzer = tvm.arith.Analyzer()

    p, q, a, b = _get_vars(['p', 'q', 'a', 'b'])
    analyzer.bind(p, a + 1)
    analyzer.bind(q, b + 1)

    expr = analyzer.equation_simplify((a + 1) * (b + 1))
    assert _is_equal(expr, p * q)

def test_fuse():
    """
    eqs:
    p = joii_fused % 8 + io * 8
    p = ji + joii_fused/8 * 8

    to simplify:
    joii_fused % 8 + io * 8 -> p
    (joii_fused % 8) * 2 + io * 16 -> 2 * p
    ((joii_fused * 2) % 16) + io * 16 -> 2 * p
    """
    analyzer = tvm.arith.Analyzer()

    p, q, joii_fused, io, ji = _get_vars(['p', 'q', 'joii_fused', 'io', 'ji'])

    analyzer.bind(p, joii_fused % 8 + io * 8)
    analyzer.bind(q, ji + joii_fused / 8 * 8)

    expr = analyzer.equation_simplify(joii_fused % 8 + io * 8)
    assert _is_equal(expr, p)

    expr = analyzer.equation_simplify(joii_fused % 8 * 2 + io * 16)
    assert _is_equal(expr, 2 * p)

    expr = analyzer.equation_simplify((joii_fused * 2) % 16 + io * 16)
    assert _is_equal(expr, 2 * p)


def test_equation_solver():
    """
    eqs:
    p = 2 * a + c + 4
    q = 4 * b + c + 6

    to simplify:
    4 * a + 8 * b + 4 * c + 20 -> 2 * p + 2 * q
    a + 2 * b + 5 + c -> p / 2 + q / 2
    """

    analyzer = tvm.arith.Analyzer()

    p, q, a, b, c = _get_vars(['p', 'q', 'a', 'b', 'c'])
    analyzer.bind(p, 2 * a + 1)
    analyzer.bind(q, 4 * b + 3)

    expr = analyzer.equation_simplify(4 * a + 8 * b + 4 * c + 20)
    assert _is_equal(expr, 2 * p + 2 * q)

    expr = analyzer.equation_simplify(a + 2 * b + c + 4)
    assert _is_equal(expr, p / 2 + q / 2)


def test_non_linear():
    """
    eqs:
    p = a + 1
    q = b + 1

    to simplify:
    a * b + a + b + 1 -> p * q
    """
    analyzer = tvm.arith.Analyzer()

    p, q, a, b = _get_vars(['p', 'q', 'a', 'b'])
    analyzer.bind(p, a + 1)
    analyzer.bind(q, b + 1)

    expr = analyzer.equation_simplify(a * b + a + b + 1)
    assert _is_equal(expr, p * q)

if __name__ == "__main__":
    test_linear_split()
    test_reduction()
    test_matching()
    test_fuse()
    #test_equation_solver()
    #test_non_linear()

