# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba

# from numba_dpcomp import njit
import math
from numpy.testing import assert_equal  # for nans comparison
from numba_dpcomp.mlir.passes import print_pass_ir, get_print_buffer

import pytest
import itertools

from .utils import parametrize_function_variants
from .utils import njit_cached as njit


def _skip_python_errors(func):
    def _wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ZeroDivisionError, TypeError, ValueError):
            pytest.skip()

    return _wrapper


# TODO: nans and infs not tested yet, we are not sure if want exactly follow
# interpreted python rules
_test_values_no_complex = [
    True,
    False,
    -3,
    -2,
    -1,
    0,
    1,
    2,
    3,
    -2.5,
    -1.0,
    -0.5,
    -0.0,
    0.0,
    0.5,
    1.0,
    2.5,
]

_test_values = _test_values_no_complex + [7j, 11 + 0j, 3 + 5j, -3 + 5j, 3 - 5j, -3 - 5j]


@pytest.mark.parametrize("val", _test_values)
def test_ret(val):
    def py_func(a):
        return a

    jit_func = njit(py_func)
    assert_equal(py_func(val), jit_func(val))


@pytest.mark.parametrize("val", _test_values)
def test_ret_literal(val):
    def py_func():
        return val

    jit_func = njit(py_func)
    assert_equal(py_func(), jit_func())


@parametrize_function_variants(
    "py_func",
    [
        "lambda a, b: a + b",
        "lambda a, b: a - b",
        "lambda a, b: a * b",
        "lambda a, b: a / b",
        "lambda a, b: a // b",
        "lambda a, b: a % b",
        "lambda a, b: a & b",
        "lambda a, b: a | b",
        "lambda a, b: a ^ b",
        "lambda a, b: a >> b",
        "lambda a, b: a << b",
    ],
)
@pytest.mark.parametrize("a, b", itertools.product(_test_values, _test_values))
def test_ops(py_func, a, b):
    jit_func = njit(py_func)
    assert_equal(_skip_python_errors(py_func)(a, b), jit_func(a, b))


@pytest.mark.parametrize("a, b", itertools.product(_test_values, _test_values))
def test_inplace_op(a, b):
    def py_func(a, b):
        a += b
        return a

    jit_func = njit(py_func)
    assert_equal(py_func(a, b), jit_func(a, b))


@pytest.mark.smoke
@parametrize_function_variants(
    "py_func",
    [
        "lambda a: +a",
        "lambda a: -a",
        "lambda a: ~a",
        "lambda a: abs(a)",
    ],
)
@pytest.mark.parametrize("val", _test_values)
def test_unary_ops(py_func, val, request):
    if isinstance(val, bool) and "~a" in str(request.node.callspec.id):
        # TODO: issue in upstream numba
        pytest.xfail()

    jit_func = njit(py_func)
    assert_equal(_skip_python_errors(py_func)(val), jit_func(val))


@parametrize_function_variants(
    "py_func",
    [
        "lambda a, b: a if a > b else b",
        "lambda a, b: a if a < b else b",
        "lambda a, b: a if a >= b else b",
        "lambda a, b: a if a <= b else b",
        "lambda a, b: a if a == b else b",
        "lambda a, b: a if a != b else b",
    ],
)
@pytest.mark.parametrize(
    "a, b", itertools.product(_test_values_no_complex, _test_values_no_complex)
)
def test_cmp_ops(py_func, a, b):
    jit_func = njit(py_func)
    assert_equal(py_func(a, b), jit_func(a, b))


@parametrize_function_variants(
    "py_func",
    [
        "lambda a: a + 42",
        "lambda a: 43 + a",
        "lambda a: a + 42.5",
        "lambda a: 43.5 + a",
    ],
)
@pytest.mark.parametrize("val", _test_values)
def test_const_ops(py_func, val):
    jit_func = njit(py_func)
    assert_equal(py_func(val), jit_func(val))


@parametrize_function_variants(
    "py_func",
    [
        "lambda a: a ** 1",
        "lambda a: a ** 2",
    ],
)
@pytest.mark.parametrize("val", [1, 2.5])
def test_pow_folding(py_func, val):
    jit_func = njit(py_func)

    with print_pass_ir([], ["PostLinalgOptPass"]):
        jit_func = njit(py_func)

        assert_equal(py_func(val), jit_func(val))
        ir = get_print_buffer()
        assert ir.count(f"powf") == 0, ir


@pytest.mark.parametrize("val", _test_values)
def test_var(val):
    def py_func(a):
        c = 1
        c = c + a
        return c

    jit_func = njit(py_func)
    assert_equal(py_func(val), jit_func(val))


@parametrize_function_variants(
    "py_func",
    [
        "lambda a : bool(a)",
        "lambda a : int(a)",
        # 'lambda a : float(a)', TODO: numba can't into float(bool)
        # TODO: str
    ],
)
@pytest.mark.parametrize("val", _test_values_no_complex)
def test_cast(py_func, val):
    jit_func = njit(py_func)
    assert_equal(py_func(val), jit_func(val))


@pytest.mark.parametrize("val", [1, 5, 5.5])
@pytest.mark.parametrize(
    "name", ["sqrt", "log", "exp", "sin", "cos", "erf", "tanh", "floor"]
)
def test_math_uplifting1(val, name):
    py_func = eval(f"lambda a: math.{name}(a)")

    with print_pass_ir([], ["UpliftMathPass"]):
        jit_func = njit(py_func)

        assert_equal(py_func(val), jit_func(val))
        ir = get_print_buffer()
        assert ir.count(f"math.{name}") == 1, ir


@pytest.mark.parametrize("val", [5, 5.5])
@pytest.mark.parametrize(
    "name",
    [
        "atan2",
    ],
)
def test_math_uplifting2(val, name):
    py_func = eval(f"lambda a, b: math.{name}(a, b)")

    with print_pass_ir([], ["UpliftMathPass"]):
        jit_func = njit(py_func)

        assert_equal(py_func(val, val), jit_func(val, val))
        ir = get_print_buffer()
        assert ir.count(f"math.{name}") == 1, ir


@pytest.mark.parametrize("a", [-2, 0, 1, -2.5, 0.0, 3.5])
@pytest.mark.parametrize("b", [-2, 0, 1, -2.5, 0.0, 3.5])
@pytest.mark.parametrize(
    "body, op", [("a if a < b else b", "arith.min"), ("a if a > b else b", "arith.max")]
)
def test_math_uplifting_minmax(a, b, body, op):
    py_func = eval(f"lambda a, b: {body}")
    jit_func = njit(py_func, fastmath=True)

    with print_pass_ir([], ["UpliftMathPass"]):

        assert_equal(py_func(a, b), jit_func(a, b))
        ir = get_print_buffer()
        assert ir.count(op) == 1, ir


@parametrize_function_variants(
    "py_func",
    [
        "lambda x, y, z: x * y + z",
        "lambda x, y, z: z + x * y",
    ],
)
def test_math_uplifting_fma(py_func):
    x = 2.0
    y = 3.0
    z = 4.0

    with print_pass_ir([], ["UpliftFMAPass"]):
        jit_func = njit(py_func, fastmath=False)

        assert_equal(py_func(x, y, z), jit_func(x, y, z))
        ir = get_print_buffer()
        assert ir.count(f"math.fma") == 0, ir

    with print_pass_ir([], ["UpliftFMAPass"]):
        jit_func = njit(py_func, fastmath=True)

        assert_equal(py_func(x, y, z), jit_func(x, y, z))
        ir = get_print_buffer()
        assert ir.count(f"math.fma") == 1, ir


@parametrize_function_variants(
    "py_func",
    [
        "lambda: math.pi",
        "lambda: math.e",
    ],
)
def test_math_const(py_func):
    jit_func = njit(py_func)
    assert_equal(py_func(), jit_func())


def _while_py_func_simple(a, b):
    while a < b:
        a = a * 2
    return a


def _while_py_func_multiple_conds1(a, b):
    while a < 44 and a < b:
        a = a * 2
    return a


def _while_py_func_multiple_conds2(a, b):
    while not a >= 44 and a < b:
        a = a * 2
    return a


def _while_py_func_multiple_conds3(a, b):
    while a < 44 and not a >= b:
        a = a * 2
    return a


def _while_py_func_multiple_conds4(a, b):
    while not a >= 44 and not a >= b:
        a = a * 2
    return a


def _while_py_func_break_middle(a, b):
    while a < b:
        a = a * 2
        if a == 3:
            break
        a = a + 1
    return a


def _while_py_func_nested_break(a, b):
    while a < b:
        a = a * 2
        if a == 3 or a == 7:
            a = a + 7
        else:
            break
        a = a + 1
    return a


@parametrize_function_variants(
    "py_func",
    [
        "_while_py_func_simple",
        "_while_py_func_multiple_conds1",
        "_while_py_func_multiple_conds2",
        "_while_py_func_multiple_conds3",
        "_while_py_func_multiple_conds4",
        "_while_py_func_break_middle",
        # '_while_py_func_nested_break',
    ],
)
def test_while(py_func):
    jit_func = njit(py_func)
    assert_equal(py_func(1, 66), jit_func(1, 66))


def test_indirect_call1():
    def inner_func(a):
        return a + 1

    def func(func, arg):
        return func(arg)

    jit_inner_func = njit(inner_func)
    jit_func = njit(func)

    assert_equal(func(inner_func, 5), jit_func(jit_inner_func, 5))


def test_indirect_call2():
    def inner_func(a):
        return a + 1

    def func(func, *args):
        return func(*args)

    jit_inner_func = njit(inner_func)
    jit_func = njit(func)

    assert_equal(func(inner_func, 5), jit_func(jit_inner_func, 5))


def test_indirect_call_inline():
    def inner_func(a):
        return a + 1

    def func(func, *args):
        return func(*args)

    with print_pass_ir([], ["PostLinalgOptPass"]):
        jit_inner_func = njit(inner_func, inline="always")
        jit_func = njit(func)

        assert_equal(func(inner_func, 5), jit_func(jit_inner_func, 5))
        ir = get_print_buffer()
        assert ir.count("call @") == 0, ir


def test_none_args():
    def py_func(a, b, c, d):
        return b + d

    jit_func = njit(py_func)
    assert_equal(py_func(None, 7, None, 30), jit_func(None, 7, None, 30))


def test_ret_none():
    def py_func1():
        return None

    def py_func2():
        pass

    jit_func1 = njit(py_func1)
    jit_func2 = njit(py_func2)
    assert_equal(py_func1(), jit_func1())
    assert_equal(py_func2(), jit_func2())


@pytest.mark.parametrize(
    "a, b", itertools.product(_test_values_no_complex, _test_values_no_complex)
)
def test_if1(a, b):
    def py_func(a, b):
        c = 3
        if a > 5:
            c = c + a
        c = c + b
        return c

    jit_func = njit(py_func)
    assert_equal(py_func(a, b), jit_func(a, b))


@pytest.mark.parametrize(
    "a, b", itertools.product(_test_values_no_complex, _test_values_no_complex)
)
def test_if2(a, b):
    def py_func(a, b):
        if a > b:
            return a + b
        else:
            return a - b

    jit_func = njit(py_func)
    assert_equal(py_func(a, b), jit_func(a, b))


_tuple_test_values = [True, 2, 3.5]


@pytest.mark.parametrize(
    "a, b, c",
    itertools.product(_tuple_test_values, _tuple_test_values, _tuple_test_values),
)
def test_tuple1(a, b, c):
    def py_func(a, b, c):
        t = (a, b, c)
        return t[0] + t[1] + t[2]

    jit_func = njit(py_func)
    assert_equal(py_func(a, b, c), jit_func(a, b, c))


@pytest.mark.parametrize(
    "a, b, c",
    itertools.product(_tuple_test_values, _tuple_test_values, _tuple_test_values),
)
def test_tuple2(a, b, c):
    def py_func(a, b, c):
        t = (a, b, c)
        x, y, z = t
        return x + y + z

    jit_func = njit(py_func)
    assert_equal(py_func(a, b, c), jit_func(a, b, c))


@parametrize_function_variants(
    "py_func",
    [
        "lambda a, b, c: (a, b) + (c,)",
        "lambda a, b, c: (a,) + (b, c)",
        "lambda a, b, c: (a, b, c) + ()",
        "lambda a, b, c: () + (a, b, c)",
    ],
)
@pytest.mark.parametrize(
    "a, b, c",
    itertools.product(_tuple_test_values, _tuple_test_values, _tuple_test_values),
)
def test_tuple_concat(py_func, a, b, c):
    jit_func = njit(py_func)
    assert_equal(py_func(a, b, c), jit_func(a, b, c))


@pytest.mark.parametrize(
    "a, b, c",
    itertools.product(_tuple_test_values, _tuple_test_values, _tuple_test_values),
)
def test_tuple_len(a, b, c):
    def py_func(a, b, c):
        t = (a, b, c)
        return len(t)

    jit_func = njit(py_func)
    assert_equal(py_func(a, b, c), jit_func(a, b, c))


def test_range1():
    def py_func(a):
        res = 0
        for i in range(a):
            res = res + i
        return res

    with print_pass_ir([], ["BuiltinCallsLoweringPass"]):
        jit_func = njit(py_func)
        assert_equal(py_func(10), jit_func(10))
        ir = get_print_buffer()
        assert ir.count("scf.for") > 0, ir


def test_range2():
    def py_func(a, b):
        res = 0
        for i in range(a, b):
            res = res + i
        return res

    with print_pass_ir([], ["BuiltinCallsLoweringPass"]):
        jit_func = njit(py_func)
        assert_equal(py_func(10, 20), jit_func(10, 20))
        ir = get_print_buffer()
        assert ir.count("scf.for") > 0, ir


def test_range3():
    def py_func(a, b, c):
        res = 0
        for i in range(a, b, c):
            res = res + i
        return res

    with print_pass_ir([], ["BuiltinCallsLoweringPass"]):
        jit_func = njit(py_func)
        assert_equal(py_func(10, 20, 2), jit_func(10, 20, 2))
        ir = get_print_buffer()
        assert ir.count("scf.for") > 0, ir


def test_range_negative_step():
    def py_func(a, b, c):
        res = 0
        for i in range(a, b, c):
            res = res + i
        return res

    with print_pass_ir([], ["BuiltinCallsLoweringPass"]):
        jit_func = njit(py_func)
        assert_equal(py_func(5, -8, -2), jit_func(5, -8, -2))
        ir = get_print_buffer()
        assert ir.count("scf.for") > 0, ir


def test_range_const_step1():
    def py_func(a, b):
        res = 0
        for i in range(a, b, -2):
            res = res + i
        return res

    with print_pass_ir([], ["BuiltinCallsLoweringPass"]):
        jit_func = njit(py_func)
        assert_equal(py_func(5, -8), jit_func(5, -8))
        ir = get_print_buffer()
        assert ir.count("scf.for") > 0, ir


def test_range_const_step2():
    def py_func(a, b):
        res = 0
        for i in range(a, b, 2):
            res = res + i
        return res

    with print_pass_ir([], ["BuiltinCallsLoweringPass"]):
        jit_func = njit(py_func)
        assert_equal(py_func(-5, 8), jit_func(-5, 8))
        ir = get_print_buffer()
        assert ir.count("scf.for") > 0, ir


def test_range_use_index_after():
    def py_func(n):
        res = 0
        for i in range(0, n, 2):
            res = res + i
        return res + i

    with print_pass_ir([], ["BuiltinCallsLoweringPass"]):
        jit_func = njit(py_func)
        assert_equal(py_func(9), jit_func(9))
        ir = get_print_buffer()
        assert ir.count("scf.for") > 0, ir


def test_range_if():
    def py_func(n):
        res = 0
        res1 = 2
        for i in range(n):
            if i > 5:
                res = res + i
            else:
                res1 = res1 + i * 2
        return res + res1

    with print_pass_ir([], ["BuiltinCallsLoweringPass"]):
        jit_func = njit(py_func)
        assert_equal(py_func(10), jit_func(10))
        ir = get_print_buffer()
        assert ir.count("scf.for") > 0, ir


def test_range_ifs():
    def py_func(n):
        res = 0
        for i in range(n):
            if i == 2:
                res = res + 2
            elif i == 7:
                res = res + 5
            elif i == 99:
                res = res + 99
            else:
                res = res + i
        return res

    with print_pass_ir([], ["BuiltinCallsLoweringPass"]):
        jit_func = njit(py_func)
        assert_equal(py_func(10), jit_func(10))
        ir = get_print_buffer()
        assert ir.count("scf.for") > 0, ir


def test_range_continue():
    def py_func(n):
        res = 0
        res1 = 2
        for i in range(n):
            res = res + i
            if i < 5:
                continue
            res1 = res1 + i * 2
        return res + res1

    with print_pass_ir([], ["BuiltinCallsLoweringPass"]):
        jit_func = njit(py_func)
        assert_equal(py_func(10), jit_func(10))
        ir = get_print_buffer()
        assert ir.count("scf.for") > 0, ir


def test_range_nested1():
    def py_func(a, b, c):
        res = 0
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    res = res + i
        return res

    with print_pass_ir([], ["BuiltinCallsLoweringPass"]):
        jit_func = njit(py_func)
        assert_equal(py_func(10, 20, 2), jit_func(10, 20, 2))
        ir = get_print_buffer()
        assert ir.count("scf.for") > 0, ir


def test_range_nested2():
    def py_func(a, b, c):
        res = 0
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    res = res + i + j * 10 + k * 100
        return res

    with print_pass_ir([], ["BuiltinCallsLoweringPass"]):
        jit_func = njit(py_func)
        assert_equal(py_func(10, 20, 2), jit_func(10, 20, 2))
        ir = get_print_buffer()
        assert ir.count("scf.for") > 0, ir


def test_prange1():
    def py_func(a):
        res = 0
        for i in numba.prange(a):
            res = res + i
        return res

    jit_func = njit(py_func, parallel=True)
    assert_equal(py_func(10), jit_func(10))


def test_prange2():
    def py_func(a, b):
        res = 0
        for i in numba.prange(a, b):
            res = res + i
        return res

    jit_func = njit(py_func, parallel=True)
    assert_equal(py_func(10, 20), jit_func(10, 20))


def test_prange_reduce1():
    def py_func(a):
        res = 0
        for i in numba.prange(1, a):
            res = res + i
        return res

    jit_func = njit(py_func, parallel=True)
    assert_equal(py_func(10), jit_func(10))


def test_prange_reduce2():
    def py_func(a):
        res = 1
        for i in numba.prange(1, a):
            res = res * i
        return res

    jit_func = njit(py_func, parallel=True)
    assert_equal(py_func(10), jit_func(10))


def test_prange_reduce3():
    def py_func(a):
        res1 = 0
        res2 = 1
        for i in numba.prange(1, a):
            res1 = res1 + i
            res2 = res2 * i
        return res1 + res2

    jit_func = njit(py_func, parallel=True)
    assert_equal(py_func(10), jit_func(10))


def test_func_call1():
    def py_func1(b):
        return b + 3

    jit_func1 = njit(py_func1)

    def py_func2(a):
        return jit_func1(a) * 4

    jit_func2 = njit(py_func2)

    assert_equal(py_func2(10), jit_func2(10))


def test_func_call2():
    def py_func1(b):
        return b + 3

    jit_func1 = njit(py_func1)

    def py_func2(a):
        return jit_func1(a) * jit_func1(a + 1)

    jit_func2 = njit(py_func2)

    assert_equal(py_func2(10), jit_func2(10))


def test_omitted_args_int():
    def py_func(a=3, b=7):
        return a + b

    jit_func = njit(py_func)
    assert_equal(py_func(), jit_func())


def test_omitted_args_float():
    def py_func(a=3.5, b=7.7):
        return a + b

    jit_func = njit(py_func)
    assert_equal(py_func(), jit_func())


def test_omitted_args_bool():
    def py_func(a=True, b=False):
        res = 1
        if a:
            res = res + 1
        if b:
            res = res * 2
        return res

    jit_func = njit(py_func)
    assert_equal(py_func(), jit_func())


def test_omitted_args_none():
    def py_func1(a=None):
        return a

    jit_func1 = njit(py_func1)

    def py_func2(a=None):
        return jit_func1(a)

    jit_func2 = njit(py_func2)

    assert_equal(py_func2(), jit_func2())
    assert_equal(py_func2(1), jit_func2(1))


@pytest.mark.parametrize("func", [min, max])
@pytest.mark.parametrize(
    "a, b", itertools.product(_test_values_no_complex, _test_values_no_complex)
)
def test_builtin_funcs2(func, a, b):
    # TODO: Skip bools and negative zeros, need investigation
    if (
        isinstance(a, bool)
        or isinstance(b, bool)
        or math.copysign(1, a) < 0
        or math.copysign(1, b) < 0
    ):
        pytest.xfail()

    def py_func(a, b):
        return func(a, b)

    jit_func = njit(py_func)

    assert_equal(py_func(a, b), jit_func(a, b))


@pytest.mark.parametrize("func", [min, max])
@pytest.mark.parametrize(
    "a, b, c",
    itertools.product(_tuple_test_values, _tuple_test_values, _tuple_test_values),
)
def test_builtin_funcs3(func, a, b, c):
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(c, bool):
        pytest.xfail()

    def py_func(a, b, c):
        return func(a, b, c)

    jit_func = njit(py_func)

    assert_equal(py_func(a, b, c), jit_func(a, b, c))


@pytest.mark.skip()
def test_fastmath_indirect():
    def py_func1(a, b, c):
        return a + b * c

    jit_func1 = njit(py_func1, fastmath=True)

    def py_func2(a, b, c):
        return jit_func1(a, b, c)

    jit_func2 = njit(py_func2)

    a, b, c = (2.0, 3.5, 4.7)
    with print_pass_ir([], ["UpliftFMAPass"]):
        assert_equal(py_func1(a, b, c), jit_func1(a, b, c))
        ir = get_print_buffer()
        assert ir.count("math.fma") > 0, ir

    with print_pass_ir([], ["UpliftFMAPass"]):
        assert_equal(py_func2(a, b, c), jit_func2(a, b, c))
        ir = get_print_buffer()
        assert ir.count("math.fma") > 0, ir


def test_replace_parfor():
    def py_func(c):
        res = 0
        for i in numba.prange(len(c)):
            ind = 2 if i == 4 else i
            res = res + c[ind]
        return res

    import numpy as np

    a = np.arange(10)

    jit_func = njit(py_func, parallel=True, replace_parfors=True)
    assert_equal(py_func(a), jit_func(a))
