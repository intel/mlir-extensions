# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba

# from numba_dpcomp import njit
from numba_dpcomp import vectorize
from numba_dpcomp.mlir.passes import print_pass_ir, get_print_buffer
from numpy.testing import assert_equal, assert_allclose  # for nans comparison
import numpy as np
import itertools
import math
from functools import partial
import pytest
from sklearn.datasets import make_regression

from .utils import parametrize_function_variants
from .utils import njit_cached as njit

np.seterr(all="ignore")


def _vectorize_reference(func, arg1):
    ret = np.empty(arg1.shape, arg1.dtype)
    for ind, val in np.ndenumerate(arg1):
        ret[ind] = func(val)
    return ret


_arr_1d_bool = np.array([True, False, True, True, False, True, True, True])
_arr_1d_int32 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
_arr_1d_int64 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
_arr_1d_float32 = np.array([1.0, 2.1, 3.2, 4.3, 5.4, 6.5, 7.6, 8.7], dtype=np.float32)
_arr_1d_float64 = np.array([1.0, 2.1, 3.2, 4.3, 5.4, 6.5, 7.6, 8.7], dtype=np.float64)
_arr_2d_int = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
_arr_2d_float = np.array([[1.0, 2.1, 3.2, 4.3], [5.4, 6.5, 7.6, 8.7]])
_test_arrays = [
    # _arr_1d_bool,
    _arr_1d_int32,
    _arr_1d_int64,
    _arr_1d_float32,
    _arr_1d_float64,
    _arr_2d_int,
    _arr_2d_float,
    _arr_2d_int.T,
    _arr_2d_float.T,
]
_test_arrays_ids = [
    # '_arr_1d_bool',
    "_arr_1d_int32",
    "_arr_1d_int64",
    "_arr_1d_float32",
    "_arr_1d_float64",
    "_arr_2d_int",
    "_arr_2d_float",
    "_arr_2d_int.T",
    "_arr_2d_float.T",
]


@parametrize_function_variants(
    "py_func",
    [
        "lambda a: a.sum()",
        "lambda a: a.min()",
        "lambda a: a.max()",
        "lambda a: a.mean()",
        "lambda a: np.sum(a)",
        "lambda a: np.amax(a)",
        "lambda a: np.amin(a)",
        "lambda a: np.mean(a)",
        "lambda a: np.sqrt(a)",
        "lambda a: np.square(a)",
        "lambda a: np.log(a)",
        "lambda a: np.sin(a)",
        "lambda a: np.cos(a)",
        "lambda a: np.exp(a)",
        "lambda a: np.tanh(a)",
        "lambda a: np.abs(a)",
        "lambda a: np.absolute(a)",
        "lambda a: np.negative(a)",
        "lambda a: a.size",
        "lambda a: a.T",
        "lambda a: a.T.T",
        "lambda a: a.copy()",
    ],
)
@pytest.mark.parametrize("arr", _test_arrays, ids=_test_arrays_ids)
def test_unary(py_func, arr, request):
    jit_func = njit(py_func)
    assert_allclose(py_func(arr), jit_func(arr), rtol=1e-4, atol=1e-7)


_test_binary_test_arrays = [
    # True,
    1,
    2.5,
    # np.array([True, False, True]),
    np.array([1, 2, 3], dtype=np.int32),
    np.array([1, 2, 3], dtype=np.int64),
    np.array([4.4, 5.5, 6.6], dtype=np.float32),
    np.array([4.4, 5.5, 6.6], dtype=np.float64),
]
_test_binary_test_arrays_ids = [
    # 'True',
    "1",
    "2.5",
    # 'np.array([True, False, True])',
    "np.array([1,2,3], dtype=np.int32)",
    "np.array([1,2,3], dtype=np.int64)",
    "np.array([4.4,5.5,6.6], dtype=np.float32)",
    "np.array([4.4,5.5,6.6], dtype=np.float64)",
]


@parametrize_function_variants(
    "py_func",
    [
        "lambda a, b: np.add(a, b)",
        "lambda a, b: a + b",
        "lambda a, b: np.subtract(a, b)",
        "lambda a, b: a - b",
        "lambda a, b: np.multiply(a, b)",
        "lambda a, b: a * b",
        "lambda a, b: np.power(a, b)",
        "lambda a, b: a ** b",
        "lambda a, b: np.true_divide(a, b)",
        "lambda a, b: a / b",
        "lambda a, b: np.arctan2(a, b)",
        "lambda a, b: np.minimum(a, b)",
        "lambda a, b: np.maximum(a, b)",
        "lambda a, b: a < b",
        "lambda a, b: a <= b",
        "lambda a, b: a > b",
        "lambda a, b: a >= b",
        "lambda a, b: a == b",
        "lambda a, b: a != b",
        "lambda a, b: np.where(a < b, a, b)",
        "lambda a, b: np.outer(a, b)",
    ],
)
@pytest.mark.parametrize(
    "a", _test_binary_test_arrays, ids=_test_binary_test_arrays_ids
)
@pytest.mark.parametrize(
    "b", _test_binary_test_arrays, ids=_test_binary_test_arrays_ids
)
def test_binary(py_func, a, b):
    jit_func = njit(py_func)
    # assert_equal(py_func(a,b), jit_func(a,b))
    assert_allclose(py_func(a, b), jit_func(a, b), rtol=1e-7, atol=1e-7)


_test_logical_arrays = [
    True,
    False,
    np.array([True, False]),
    np.array([[False, True], [True, False]]),
]


@parametrize_function_variants(
    "py_func",
    [
        "lambda a: np.logical_not(a)",
    ],
)
@pytest.mark.parametrize("a", _test_logical_arrays)
def test_logical1(py_func, a):
    jit_func = njit(py_func)
    assert_equal(py_func(a), jit_func(a))


@parametrize_function_variants(
    "py_func",
    [
        "lambda a, b: np.logical_and(a, b)",
        "lambda a, b: a & b",
        "lambda a, b: np.logical_or(a, b)",
        "lambda a, b: a | b",
        "lambda a, b: np.logical_xor(a, b)",
        "lambda a, b: a ^ b",
    ],
)
@pytest.mark.parametrize("a", _test_logical_arrays)
@pytest.mark.parametrize("b", _test_logical_arrays)
def test_logical2(py_func, a, b):
    jit_func = njit(py_func)
    assert_equal(py_func(a, b), jit_func(a, b))


_test_broadcast_test_arrays = [
    1,
    np.array([1]),
    np.array([[1]]),
    np.array([[1, 2], [3, 4]]),
    np.array([5, 6]),
    np.array([[5], [6]]),
    np.array([[5, 6]]),
]
_test_broadcast_test_arrays_ids = [
    "1",
    "np.array([1])",
    "np.array([[1]])",
    "np.array([[1,2],[3,4]])",
    "np.array([5,6])",
    "np.array([[5],[6]])",
    "np.array([[5,6]])",
]


@pytest.mark.parametrize(
    "a", _test_broadcast_test_arrays, ids=_test_broadcast_test_arrays_ids
)
@pytest.mark.parametrize(
    "b", _test_broadcast_test_arrays, ids=_test_broadcast_test_arrays_ids
)
def test_broadcast(a, b):
    def py_func(a, b):
        return np.add(a, b)

    jit_func = njit(py_func)
    assert_equal(py_func(a, b), jit_func(a, b))


def test_staticgetitem():
    def py_func(a):
        return a[1]

    jit_func = njit(py_func)
    arr = np.asarray([5, 6, 7])
    assert_equal(py_func(arr), jit_func(arr))


@pytest.mark.parametrize("i", list(range(-2, 3)))
def test_getitem1(i):
    def py_func(a, b):
        return a[b]

    jit_func = njit(py_func)
    arr = np.asarray([5, 6, 7])
    assert_equal(py_func(arr, i), jit_func(arr, i))


def test_getitem2():
    def py_func(a, b):
        return a[b]

    jit_func = njit(py_func)
    arr = np.asarray([[[1, 2, 3], [5, 6, 7]]])
    assert_equal(py_func(arr, 0), jit_func(arr, 0))


def test_getitem3():
    def py_func(a, b, c):
        return a[b, c]

    jit_func = njit(py_func)
    arr = np.asarray([[[1, 2, 3], [5, 6, 7]]])
    assert_equal(py_func(arr, 0, 0), jit_func(arr, 0, 0))


def test_unituple_getitem1():
    def py_func(a, b, c, i):
        t = (a, b, c)
        return t[i]

    jit_func = njit(py_func)
    assert_equal(py_func(1, 2, 3, 1), jit_func(1, 2, 3, 1))


def test_unituple_getitem2():
    def py_func(t, i):
        return t[i]

    jit_func = njit(py_func)
    t = (1, 2, 3)
    assert_equal(py_func(t, 1), jit_func(t, 1))


@pytest.mark.parametrize("arr", _test_arrays, ids=_test_arrays_ids)
@pytest.mark.parametrize("mask", [[True], [False], [True, False], [False, True]])
def test_getitem_mask(arr, mask):
    if arr.ndim > 1:
        pytest.xfail()  # TODO: not supprted by numba

    def py_func(a, m):
        return a[m]

    mask = np.resize(mask, arr.size).reshape(arr.shape)

    jit_func = njit(py_func)
    assert_equal(py_func(arr, mask), jit_func(arr, mask))


def test_array_len():
    def py_func(a):
        return len(a)

    jit_func = njit(py_func)
    arr = np.asarray([5, 6, 7])
    assert_equal(py_func(arr), jit_func(arr))


@parametrize_function_variants(
    "py_func",
    [
        "lambda a: np.sum(a, axis=0)",
        "lambda a: np.sum(a, axis=1)",
        "lambda a: np.sum(a, axis=-1)",
        "lambda a: np.sum(a, axis=-2)",
        # 'lambda a: np.amax(a, axis=0)', # TODO: Not supported by numba
        # 'lambda a: np.amax(a, axis=1)',
        # 'lambda a: np.amin(a, axis=0)',
        # 'lambda a: np.amin(a, axis=1)',
    ],
)
@pytest.mark.parametrize(
    "arr",
    [
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
    ],
)
def test_reduce_axis(py_func, arr):
    jit_func = njit(py_func)
    assert_equal(py_func(arr), jit_func(arr))


@parametrize_function_variants(
    "py_func",
    [
        "lambda a: np.flip(a)",
    ],
)
@pytest.mark.parametrize(
    "arr",
    [
        np.array([1, 2, 3, 4, 5, 6], dtype=np.int32),
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.int32),
    ],
)
def test_flip1(py_func, arr):
    jit_func = njit(py_func)
    assert_equal(py_func(arr), jit_func(arr))


@parametrize_function_variants(
    "py_func",
    [
        "lambda a: np.flip(a, axis=0)",
        "lambda a: np.flip(a, axis=1)",
        "lambda a: np.flip(a, axis=-1)",
        "lambda a: np.flip(a, axis=-2)",
    ],
)
@pytest.mark.parametrize(
    "arr",
    [
        np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.int32),
        np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32),
    ],
)
@pytest.mark.xfail
def test_flip2(py_func, arr):
    jit_func = njit(py_func)
    assert_equal(py_func(arr), jit_func(arr))


def test_sum_add():
    def py_func(a, b):
        return np.add(a, b).sum()

    jit_func = njit(py_func)
    arr1 = np.asarray([1, 2, 3])
    arr2 = np.asarray([4, 5, 6])
    assert_equal(py_func(arr1, arr2), jit_func(arr1, arr2))


def test_sum_add2():
    def py_func(a, b, c):
        t = np.add(a, b)
        return np.add(t, c).sum()

    jit_func = njit(py_func)
    arr1 = np.asarray([1, 2, 3])
    arr2 = np.asarray([4, 5, 6])
    arr3 = np.asarray([7, 8, 9])
    assert_equal(py_func(arr1, arr2, arr3), jit_func(arr1, arr2, arr3))


@pytest.mark.parametrize(
    "a,b",
    [
        (np.array([1, 2, 3], np.float32), np.array([4, 5, 6], np.float32)),
        (
            np.array([[1, 2, 3], [4, 5, 6]], np.float32),
            np.array([[1, 2], [3, 4], [5, 6]], np.float32),
        ),
    ],
)
@pytest.mark.parametrize("parallel", [False, True])
def test_dot(a, b, parallel):
    def py_func(a, b):
        return np.dot(a, b)

    jit_func = njit(py_func, parallel=parallel)
    assert_equal(py_func(a, b), jit_func(a, b))


@pytest.mark.parametrize(
    "a,b",
    [
        (
            np.array([[1, 2, 3], [4, 5, 6]], np.float32),
            np.array([[1, 2], [3, 4], [5, 6]], np.float32),
        ),
    ],
)
@parametrize_function_variants(
    "py_func",
    [
        "lambda a, b, c: np.dot(a, b, c)",
        "lambda a, b, c: np.dot(a, b, out=c)",
    ],
)
def test_dot_out(a, b, py_func):
    jit_func = njit(py_func)

    tmp = np.dot(a, b)
    res_py = np.zeros_like(tmp)
    res_jit = np.zeros_like(tmp)

    py_func(a, b, res_py)
    jit_func(a, b, res_jit)
    assert_equal(res_py, res_jit)


def test_prange_lowering():
    def py_func(arr):
        res = 0
        for i in numba.prange(len(arr)):
            res += arr[i]

        return res

    with print_pass_ir([], ["ParallelToTbbPass"]):
        jit_func = njit(py_func, parallel=True)
        arr = np.arange(10000, dtype=np.float32)
        assert_equal(py_func(arr), jit_func(arr))
        ir = get_print_buffer()
        assert ir.count("imex_util.parallel") == 1, ir


def test_loop_fusion1():
    def py_func(arr):
        l = len(arr)
        res1 = 0
        for i in numba.prange(l):
            res1 += arr[i]

        res2 = 1.0
        for i in numba.prange(l):
            res2 *= arr[i]

        return res1, res2

    with print_pass_ir([], ["PostLinalgOptPass"]):
        jit_func = njit(py_func)
        arr = np.arange(1, 15, dtype=np.float32)
        assert_equal(py_func(arr), jit_func(arr))
        ir = get_print_buffer()
        assert ir.count("scf.parallel") == 1, ir
        assert ir.count("memref.load") == 1, ir


def test_loop_fusion2():
    def py_func(arr):
        l = len(arr)
        res1 = 0
        for i in numba.prange(l):
            res1 += arr[i]

        res1 += 10

        res2 = 0.0
        for i in numba.prange(l):
            res2 *= arr[i]

        return res1, res2

    with print_pass_ir([], ["PostLinalgOptPass"]):
        jit_func = njit(py_func)
        arr = np.arange(1, 15, dtype=np.float32)
        assert_equal(py_func(arr), jit_func(arr))
        ir = get_print_buffer()
        assert ir.count("scf.parallel") == 1, ir
        assert ir.count("memref.load") == 1, ir


def test_loop_fusion3():
    def py_func(arr):
        l = len(arr)
        res1 = 0
        for i in numba.prange(l):
            res1 += arr[i]

        res2 = 1.0
        for i in numba.prange(l):
            res2 *= arr[i] * res1

        return res1, res2

    with print_pass_ir([], ["PostLinalgOptPass"]):
        jit_func = njit(py_func)
        arr = np.arange(1, 15, dtype=np.float32)
        assert_equal(py_func(arr), jit_func(arr))
        ir = get_print_buffer()
        assert ir.count("scf.parallel") == 2, ir
        assert ir.count("memref.load") == 2, ir


def test_copy_fusion():
    def py_func(a, b):
        a = a + 1
        b[:] = a

    jit_func = njit(py_func)
    a = np.arange(13)

    res_py = np.zeros_like(a)
    res_jit = np.zeros_like(a)

    with print_pass_ir([], ["PostLinalgOptPass"]):
        py_func(a, res_py)
        jit_func(a, res_jit)

        assert_equal(res_py, res_jit)
        ir = get_print_buffer()
        assert ir.count("scf.parallel") == 1, ir


def test_broadcast_fusion():
    def py_func(a):
        return a + a * a

    jit_func = njit(py_func)
    a = np.arange(13)

    with print_pass_ir([], ["PostLinalgOptPass"]):
        assert_equal(py_func(a), jit_func(a))
        ir = get_print_buffer()
        assert ir.count("scf.parallel") == 1, ir


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32])
def test_np_reduce(dtype):
    def py_func(arr):
        return arr.sum()

    with print_pass_ir([], ["PostLinalgOptPass"]):
        jit_func = njit(py_func)
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        assert_equal(py_func(arr), jit_func(arr))
        ir = get_print_buffer()
        assert ir.count("scf.parallel") == 1, ir
        assert ir.count("memref.load") == 1, ir


def test_indirect_call_array():
    def inner_func(a):
        return a + 3

    def func(func, *args):
        return func(*args)

    jit_inner_func = njit(inner_func)
    jit_func = njit(func)

    arr = np.array([[1, 2, 3], [4, 5, 6]])
    # arr = 5
    assert_equal(func(inner_func, arr), jit_func(jit_inner_func, arr))


def test_loop_if():
    def py_func(arr):
        for i in range(len(arr)):
            if arr[i] == 5:
                arr[i] = 6
        return arr

    jit_func = njit(py_func)
    arr1 = np.arange(100)
    arr2 = np.arange(100)
    assert_equal(py_func(arr1), jit_func(arr2))


def test_static_setitem1():
    def py_func(a):
        a[1] = 42
        return a

    jit_func = njit(py_func)
    arr = np.asarray([1, 2, 3])
    assert_equal(py_func(arr.copy()), jit_func(arr.copy()))


def test_static_setitem2():
    def py_func(a):
        a[:] = 42
        return a

    jit_func = njit(py_func)
    arr = np.asarray([1, 2, 3])
    assert_equal(py_func(arr.copy()), jit_func(arr.copy()))


def test_static_setitem3():
    def py_func(a):
        a[(0, 1)] = 42
        return a

    jit_func = njit(py_func)
    arr = np.asarray([[1, 2], [3, 4]])
    assert_equal(py_func(arr.copy()), jit_func(arr.copy()))


@pytest.mark.parametrize("i", list(range(-2, 3)))
def test_setitem1(i):
    def py_func(a, b):
        a[b] = 42
        return a[b]

    jit_func = njit(py_func)
    arr = np.asarray([1, 2, 3])
    assert_equal(py_func(arr, i), jit_func(arr, i))


def test_setitem2():
    def py_func(a, b, c):
        a[b, c] = 42
        return a[b, c]

    jit_func = njit(py_func)
    arr = np.asarray([[1, 2, 3], [4, 5, 6]])
    assert_equal(py_func(arr, 1, 2), jit_func(arr, 1, 2))


@pytest.mark.parametrize("d", [np.array([5, 6]), 7])
def test_setitem_slice1(d):
    def py_func(a, b, c, d):
        a[b:c] = d
        return a

    jit_func = njit(py_func)
    arr = np.asarray([1, 2, 3, 4])
    assert_equal(py_func(arr.copy(), 1, 3, d), jit_func(arr.copy(), 1, 3, d))


@pytest.mark.parametrize("d", [np.array([5, 6, 7]), 7])
def test_setitem_slice2(d):
    def py_func(a, c, d):
        a[:c] = d
        return a

    jit_func = njit(py_func)
    arr = np.asarray([1, 2, 3, 4])
    assert_equal(py_func(arr.copy(), 3, d), jit_func(arr.copy(), 3, d))


@pytest.mark.parametrize("d", [np.array([5, 6, 7]), 7])
def test_setitem_slice3(d):
    def py_func(a, b, d):
        a[b:] = d
        return a

    jit_func = njit(py_func)
    arr = np.asarray([1, 2, 3, 4])
    assert_equal(py_func(arr.copy(), 1, d), jit_func(arr.copy(), 1, d))


def test_setitem_loop():
    def py_func(a):
        for i in range(len(a)):
            a[i] = a[i] + i
        return a.sum()

    jit_func = njit(py_func)
    arr = np.asarray([3, 2, 1])
    assert_equal(py_func(arr.copy()), jit_func(arr.copy()))


def test_array_bounds1():
    def py_func(a):
        res = 0
        for i in range(len(a)):
            if i >= len(a) or i < 0:
                res = res + 1
            else:
                res = res + a[i]
        return res

    arr = np.asarray([3, 2, 1])

    with print_pass_ir([], ["PostLinalgOptPass"]):
        jit_func = njit(py_func)
        assert_equal(py_func(arr.copy()), jit_func(arr.copy()))
        ir = get_print_buffer()
        assert ir.count("cmpi") == 0, ir


def test_array_bounds2():
    def py_func(a):
        res = 0
        for i in range(len(a)):
            if i < len(a) and i >= 0:
                res = res + a[i]
            else:
                res = res + 1
        return res

    arr = np.asarray([3, 2, 1])

    with print_pass_ir([], ["PostLinalgOptPass"]):
        jit_func = njit(py_func)
        assert_equal(py_func(arr.copy()), jit_func(arr.copy()))
        ir = get_print_buffer()
        assert ir.count("cmpi") == 0, ir


def test_array_bounds3():
    def py_func(a):
        res = 0
        for i in range(len(a)):
            if 0 <= i < len(a):
                res = res + a[i]
            else:
                res = res + 1
        return res

    arr = np.asarray([3, 2, 1])

    with print_pass_ir([], ["PostLinalgOptPass"]):
        jit_func = njit(py_func)
        assert_equal(py_func(arr.copy()), jit_func(arr.copy()))
        ir = get_print_buffer()
        assert ir.count("cmpi") == 0, ir


@pytest.mark.xfail(reason="Need to improve CmpLoopBoundsSimplify")
def test_array_bounds4():
    def py_func(a):
        res = 0
        for i in range(len(a) - 1):
            if 0 <= i < (len(a) - 1):
                res = res + a[i]
            else:
                res = res + 1
        return res

    arr = np.asarray([3, 2, 1])

    with print_pass_ir([], ["PostLinalgOptPass"]):
        jit_func = njit(py_func)
        assert_equal(py_func(arr.copy()), jit_func(arr.copy()))
        ir = get_print_buffer()
        assert ir.count("cmpi") == 0, ir


def test_array_shape():
    def py_func(a):
        shape = a.shape
        return shape[0] + shape[1] * 10

    jit_func = njit(py_func)
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    assert_equal(py_func(arr), jit_func(arr))


def test_array_return():
    def py_func(a):
        return a

    jit_func = njit(py_func)
    arr = np.array([1, 2, 3])
    assert_equal(py_func(arr), jit_func(arr))


def test_array_prange_const():
    def py_func(a, b):
        a[0] = 42
        for i in numba.prange(b):
            a[0] = 1
        return a[0]

    jit_func = njit(py_func, parallel=True)
    arr = np.array([0.0])
    assert_equal(py_func(arr, 5), jit_func(arr, 5))


def test_empty1():
    def py_func(d):
        a = np.empty(d)
        for i in range(d):
            a[i] = i
        return a

    jit_func = njit(py_func)
    assert_equal(py_func(5), jit_func(5))


def test_empty2():
    def py_func(d1, d2):
        a = np.empty((d1, d2))
        for i in range(d1):
            for j in range(d2):
                a[i, j] = i + j * 10
        return a

    jit_func = njit(py_func)
    assert_equal(py_func(5, 7), jit_func(5, 7))


@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_empty3(dtype):
    def py_func(a):
        return np.empty(a.shape, a.dtype)

    jit_func = njit(py_func)
    arr = np.array([1, 2, 3], dtype=dtype)
    assert_equal(py_func(arr).shape, jit_func(arr).shape)
    assert_equal(py_func(arr).dtype, jit_func(arr).dtype)


@pytest.mark.parametrize("shape", [1, (2,), (2, 3), (4, 5, 6)])
@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_empty_like(shape, dtype):
    def py_func(a):
        return np.empty_like(a)

    jit_func = njit(py_func)
    arr = np.empty(shape=shape, dtype=dtype)
    assert_equal(py_func(arr).shape, jit_func(arr).shape)
    assert_equal(py_func(arr).dtype, jit_func(arr).dtype)


@pytest.mark.parametrize("func", [np.zeros, np.ones], ids=["zeros", "ones"])
def test_init1(func):
    def py_func(d):
        return func(d)

    jit_func = njit(py_func)
    assert_equal(py_func(5), jit_func(5))


@pytest.mark.parametrize("func", [np.zeros, np.ones], ids=["zeros", "ones"])
@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_init2(func, dtype):
    def py_func(a):
        return func(a.shape, a.dtype)

    jit_func = njit(py_func)
    arr = np.array([1, 2, 3], dtype=dtype)
    assert_equal(py_func(arr).shape, jit_func(arr).shape)
    assert_equal(py_func(arr).dtype, jit_func(arr).dtype)


@pytest.mark.parametrize("func", [np.zeros, np.ones], ids=["zeros", "ones"])
@pytest.mark.xfail
def test_init3(func):
    def py_func(d):
        return func(d, dtype=np.dtype("int64"))

    jit_func = njit(py_func)
    assert_equal(py_func(5), jit_func(5))


@pytest.mark.parametrize("func", [np.zeros, np.ones], ids=["zeros", "ones"])
def test_init4(func):
    def py_func(d):
        return func(d)

    jit_func = njit(py_func)
    assert_equal(py_func((2, 1)), jit_func((2, 1)))


@pytest.mark.parametrize("shape", [2, (3, 4), (5, 6, 7)])
@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
@pytest.mark.parametrize(
    "func", [np.zeros_like, np.ones_like], ids=["zeros_like", "ones_like"]
)
def test_init_like(shape, dtype, func):
    def py_func(d):
        return func(d)

    a = np.empty(shape=shape, dtype=dtype)
    jit_func = njit(py_func)
    assert_equal(py_func(a), jit_func(a))


@parametrize_function_variants(
    "py_func",
    [
        "lambda : np.arange(0)",
        "lambda : np.arange(1)",
        "lambda : np.arange(7)",
        "lambda : np.arange(-1)",
        "lambda : np.arange(-1,6)",
        "lambda : np.arange(-1,6,1)",
        "lambda : np.arange(-1,6,2)",
        "lambda : np.arange(-1,6,3)",
        "lambda : np.arange(6,-1,-1)",
        "lambda : np.arange(6,-1,-2)",
        "lambda : np.arange(6,-1,-3)",
        "lambda : np.arange(5,dtype=np.int32)",
        "lambda : np.arange(5,dtype=np.float32)",
    ],
)
def test_arange(py_func):
    jit_func = njit(py_func)
    assert_equal(py_func(), jit_func())


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_dtype_param(dtype):
    def py_func(dt):
        return np.zeros((1,), dtype=dt)

    jit_func = njit(py_func)

    jit_func = njit(py_func)
    assert_equal(py_func(dtype).shape, jit_func(dtype).shape)
    assert_equal(py_func(dtype).dtype, jit_func(dtype).dtype)


def test_parallel():
    def py_func(a, b):
        return np.add(a, b)

    jit_func = njit(py_func, parallel=True)
    arr = np.asarray([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    assert_equal(py_func(arr, arr), jit_func(arr, arr))


def test_parallel_reduce():
    def py_func(a):
        shape = a.shape
        res = 0
        for i in range(shape[0]):
            for j in numba.prange(shape[1]):
                for k in numba.prange(shape[2]):
                    res = res + a[i, j, k]
        return res

    jit_func = njit(py_func, parallel=True)
    arr = np.asarray([[[1, 2, 3], [4, 5, 6]]]).repeat(10000, 0)
    assert_equal(py_func(arr), jit_func(arr))


@parametrize_function_variants(
    "func",
    [
        "lambda a : a + 1",
        "lambda a : math.erf(a)",
        # 'lambda a : 5 if a == 1 else a', TODO: investigate
    ],
)
@pytest.mark.parametrize("arr", _test_arrays, ids=_test_arrays_ids)
def test_vectorize(func, arr):
    arr = np.array(arr)
    vec_func = vectorize(func)
    # assert_equal(_vectorize_reference(func, arr), vec_func(arr))
    assert_allclose(
        _vectorize_reference(func, arr), vec_func(arr), rtol=1e-7, atol=1e-7
    )


@pytest.mark.parametrize("arr", _test_arrays, ids=_test_arrays_ids)
def test_vectorize_indirect(arr):
    def func(a):
        return a + 1

    vec_func = vectorize(func)

    def py_func(a):
        return vec_func(a)

    jit_func = njit(py_func, parallel=True)

    arr = np.array(arr)
    assert_equal(_vectorize_reference(func, arr), jit_func(arr))


@pytest.mark.parametrize(
    "arr",
    [
        np.array([[1, 2], [3, 4]]),
        # np.array([[1,2],[3,4]]).T,
    ],
)
def test_fortran_layout(arr):
    def py_func(a):
        return a.T

    jit_func = njit(py_func)

    assert_equal(py_func(arr), jit_func(arr))


def test_contigious_layout_opt():
    def py_func(a):
        return a[0, 1]

    jit_func = njit(py_func)

    a = np.array([[1, 2], [3, 4]])
    b = a.T

    layoutStr = "strided<[?, ?], offset: ?>"
    with print_pass_ir([], ["MakeStridedLayoutPass"]):
        assert_equal(py_func(a), jit_func(a))
        ir = get_print_buffer()
        assert ir.count(layoutStr) == 0, ir

    with print_pass_ir([], ["MakeStridedLayoutPass"]):
        assert_equal(py_func(b), jit_func(b))
        ir = get_print_buffer()
        assert ir.count(layoutStr) != 0, ir


@pytest.mark.skip(reason="Layout type inference need rework")
def test_contigious_layout_return():
    def py_func1():
        return np.ones((2, 3), np.float32).T

    jit_func1 = njit(py_func1)

    def py_func2(a):
        return a

    jit_func2 = njit(py_func2)

    def py_func3():
        a = jit_func1()
        return jit_func2(a)

    jit_func3 = njit(py_func3)

    assert_equal(py_func3(), jit_func3())


@parametrize_function_variants(
    "a",
    [
        # 'np.array(1)', TODO zero rank arrays
        # 'np.array(2.5)',
        "np.array([])",
        "np.array([1,2,3])",
        "np.array([[1,2,3]])",
        "np.array([[1,2],[3,4],[5,6]])",
    ],
)
def test_atleast2d(a):
    def py_func(a):
        return np.atleast_2d(a)

    jit_func = njit(py_func)
    assert_equal(py_func(a), jit_func(a))


_test_reshape_test_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
_test_reshape_test_arrays = [
    _test_reshape_test_array,
    _test_reshape_test_array.reshape((2, 6)),
    _test_reshape_test_array.reshape((2, 3, 2)),
]


@parametrize_function_variants(
    "py_func",
    [
        "lambda a: a.reshape(a.size)",
        "lambda a: a.reshape((a.size,))",
        "lambda a: a.reshape((a.size,1))",
        "lambda a: a.reshape((1, a.size))",
        "lambda a: a.reshape(1, a.size)",
        "lambda a: a.reshape((1, a.size, 1))",
        "lambda a: a.reshape((-1, a.size, 1))",
        "lambda a: a.reshape((1, -1, 1))",
        "lambda a: a.reshape((1, a.size, -1))",
        "lambda a: a.reshape(1, a.size, 1)",
        "lambda a: a.reshape(-1, a.size, 1)",
        "lambda a: a.reshape(1, -1, 1)",
        "lambda a: a.reshape(1, a.size, -1)",
        "lambda a: np.reshape(a, a.size)",
        "lambda a: np.reshape(a, (a.size,))",
        "lambda a: np.reshape(a, (a.size,1))",
        "lambda a: np.reshape(a, (1, a.size))",
        "lambda a: np.reshape(a, (1, a.size, 1))",
    ],
)
@pytest.mark.parametrize("array", _test_reshape_test_arrays)
def test_reshape(py_func, array):
    jit_func = njit(py_func)
    assert_equal(py_func(array), jit_func(array))


@pytest.mark.xfail(reason="numba: reshape() supports contiguous array only")
def test_reshape_non_contiguous():
    def py_func(a):
        return a.reshape(4)

    jit_func = njit(py_func)
    array = np.arange(16).reshape((4, 4))[1:3, 1:3]
    assert_equal(py_func(array), jit_func(array))


@parametrize_function_variants(
    "py_func",
    [
        # 'lambda a: a.flat', TODO: flat support
        "lambda a: a.flatten()",
    ],
)
@pytest.mark.parametrize("array", _test_reshape_test_arrays)
def test_flatten(py_func, array):
    jit_func = njit(py_func)
    assert_equal(py_func(array), jit_func(array))


@parametrize_function_variants(
    "py_func",
    [
        "lambda a, b: ()",
        "lambda a, b: (a,b)",
        "lambda a, b: ((a,b),(a,a),(b,b),())",
    ],
)
@pytest.mark.parametrize(
    "a,b",
    itertools.product(
        *(([1, 2.5, np.array([1, 2, 3]), np.array([4.5, 6.7, 8.9])],) * 2)
    ),
)
def test_tuple_ret(py_func, a, b):
    jit_func = njit(py_func)
    assert_equal(py_func(a, b), jit_func(a, b))


@pytest.mark.parametrize(
    "arrays",
    [
        ([1, 2, 3], [4, 5, 6]),
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
        ([[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]),
        ([1, 2, 3], [4, 5, 6], [7, 8, 9]),
        ([1, 2], [3, 4], [5, 6], [7, 8]),
    ],
)
@pytest.mark.parametrize("axis", [0, 1, 2])  # TODO: None
def test_concat(arrays, axis):
    arr = tuple(np.array(a) for a in arrays)
    num_dims = len(arr[0].shape)
    if axis >= num_dims:
        pytest.skip()  # TODO: unselect
    num_arrays = len(arrays)
    if num_arrays == 2:

        def py_func(arr1, arr2):
            return np.concatenate((arr1, arr2), axis=axis)

    elif num_arrays == 3:

        def py_func(arr1, arr2, arr3):
            return np.concatenate((arr1, arr2, arr3), axis=axis)

    elif num_arrays == 4:

        def py_func(arr1, arr2, arr3, arr4):
            return np.concatenate((arr1, arr2, arr3, arr4), axis=axis)

    else:
        assert False
    jit_func = njit(py_func)
    assert_equal(py_func(*arr), jit_func(*arr))


@pytest.mark.parametrize(
    "arr",
    [
        np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32),
        np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32),
        np.array([True, False, True, True, False, True, True, True]),
    ],
)
@parametrize_function_variants(
    "py_func",
    [
        "lambda a, b, c, d: a[b:c]",
        "lambda a, b, c, d: a[3:c]",
        "lambda a, b, c, d: a[b:4]",
        "lambda a, b, c, d: a[3:4]",
        "lambda a, b, c, d: a[1:-2]",
        "lambda a, b, c, d: a[b:c:d]",
        "lambda a, b, c, d: a[b:c:1]",
        "lambda a, b, c, d: a[b:c:2]",
        "lambda a, b, c, d: a[3:4:2]",
    ],
)
def test_slice1(arr, py_func):
    jit_func = njit(py_func)
    assert_equal(py_func(arr, 3, 4, 2), jit_func(arr, 3, 4, 2))


def test_slice2():
    def py_func(a, i, j, k):
        a1 = a[1]
        a2 = a1[2]
        return a2[3]

    arr = np.arange(3 * 4 * 5).reshape((3, 4, 5))
    jit_func = njit(py_func)
    assert_equal(py_func(arr, 1, 2, 3), jit_func(arr, 1, 2, 3))


@pytest.mark.parametrize("count", [0, 1, 5, 7, 8, 16, 17, 32])
def test_slice3(count):
    def py_func(A):
        B = A[::3]
        for i in range(len(B)):
            B[i] = i
        return B

    arr = np.zeros(count)
    jit_func = njit(py_func)
    assert_equal(py_func(arr.copy()[::2]), jit_func(arr.copy()[::2]))


def test_multidim_slice():
    def py_func(a, b):
        return a[1, b, :]

    jit_func = njit(py_func)

    a = np.array([[[1], [2], [3]], [[4], [5], [6]]])
    assert_equal(py_func(a, 0), jit_func(a, 0))


def test_size_ret():
    def py_func(a, b):
        return a.size / b

    jit_func = njit(py_func)

    a = np.array([[[1], [2], [3]], [[4], [5], [6]]])
    assert_equal(py_func(a, 3), jit_func(a, 3))


def test_alias1():
    def py_func():
        a = np.zeros(7)
        b = a[2:4]
        b[1] = 5
        return a

    jit_func = njit(py_func)

    a = np.ones(1)

    assert_equal(py_func(), jit_func())


def test_alias2():
    def py_func(n):
        b = np.zeros((n, n))
        a = b[0]
        for j in range(n):
            a[j] = j + 1
        return b.sum()

    jit_func = njit(py_func)

    assert_equal(py_func(4), jit_func(4))


@pytest.mark.xfail
def test_inplace_alias():
    def py_func(a):
        a += 1
        a[:] = 3

    jit_func = njit(py_func)

    a = np.ones(1)

    py_arg = a.copy()
    jit_arg = a.copy()
    py_func(py_arg)
    jit_func(jit_arg)
    assert_equal(py_arg, jit_arg)


@pytest.mark.parametrize("a", [np.array([[1, 2], [4, 5]])])
@pytest.mark.parametrize("b", [True, False])
def test_tensor_if(a, b):
    def py_func(m, rowvar):
        m_arr = np.atleast_2d(m)
        if not rowvar:
            m_arr = m_arr.T
        return m_arr

    jit_func = njit(py_func)

    assert_equal(py_func(a, b), jit_func(a, b))


def _cov(m, y=None, rowvar=True, bias=False, ddof=None):
    return np.cov(m, y, rowvar, bias, ddof)


_rnd = np.random.RandomState(42)


@parametrize_function_variants(
    "m",
    [
        "np.array([[0, 2], [1, 1], [2, 0]]).T",
        "_rnd.randn(100).reshape(5, 20)",
        "np.asfortranarray(np.array([[0, 2], [1, 1], [2, 0]]).T)",
        "_rnd.randn(100).reshape(5, 20)[:, ::2]",
        "np.array([0.3942, 0.5969, 0.7730, 0.9918, 0.7964])",
        # 'np.full((4, 5), fill_value=True)', TODO
        "np.array([np.nan, 0.5969, -np.inf, 0.9918, 0.7964])",
        "np.linspace(-3, 3, 33).reshape(33, 1)",
        # non-array inputs
        "((0.1, 0.2), (0.11, 0.19), (0.09, 0.21))",  # UniTuple
        "((0.1, 0.2), (0.11, 0.19), (0.09j, 0.21j))",  # Tuple
        "(-2.1, -1, 4.3)",
        "(1, 2, 3)",
        "[4, 5, 6]",
        "((0.1, 0.2, 0.3), (0.1, 0.2, 0.3))",
        "[(1, 2, 3), (1, 3, 2)]",
        "3.142",
        # '((1.1, 2.2, 1.5),)',
        # empty data structures
        "np.array([])",
        "np.array([]).reshape(0, 2)",
        "np.array([]).reshape(2, 0)",
        "()",
    ],
)
def test_cov_basic(m):
    if isinstance(m, (list, float)) or len(m) == 0 or np.iscomplexobj(m):
        pytest.xfail()
    py_func = _cov
    jit_func = njit(py_func)
    assert_allclose(py_func(m), jit_func(m), rtol=1e-15, atol=1e-15)


_cov_inputs_m = _rnd.randn(105).reshape(15, 7)


@pytest.mark.parametrize("m", [_cov_inputs_m])
@pytest.mark.parametrize("y", [None, _cov_inputs_m[::-1]])
@pytest.mark.parametrize("rowvar", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("ddof", [None, -1, 0, 1, 3.0, True])
def test_cov_explicit_arguments(m, y, rowvar, bias, ddof):
    py_func = _cov
    jit_func = njit(py_func)
    assert_allclose(
        py_func(m=m, y=y, rowvar=rowvar, bias=bias, ddof=ddof),
        jit_func(m=m, y=y, rowvar=rowvar, bias=bias, ddof=ddof),
        rtol=1e-14,
        atol=1e-14,
    )


@parametrize_function_variants(
    "m, y, rowvar",
    [
        "(np.array([-2.1, -1, 4.3]), np.array([3, 1.1, 0.12]), True)",
        "(np.array([1, 2, 3]), np.array([1j, 2j, 3j]), True)",
        "(np.array([1j, 2j, 3j]), np.array([1, 2, 3]), True)",
        "(np.array([1, 2, 3]), np.array([1j, 2j, 3]), True)",
        "(np.array([1j, 2j, 3]), np.array([1, 2, 3]), True)",
        "(np.array([]), np.array([]), True)",
        "(1.1, 2.2, True)",
        "(_rnd.randn(10, 3), np.array([-2.1, -1, 4.3]).reshape(1, 3) / 10, True)",
        "(np.array([-2.1, -1, 4.3]), np.array([[3, 1.1, 0.12], [3, 1.1, 0.12]]), True)",
        # '(np.array([-2.1, -1, 4.3]), np.array([[3, 1.1, 0.12], [3, 1.1, 0.12]]), False)',
        "(np.array([[3, 1.1, 0.12], [3, 1.1, 0.12]]), np.array([-2.1, -1, 4.3]), True)",
        # '(np.array([[3, 1.1, 0.12], [3, 1.1, 0.12]]), np.array([-2.1, -1, 4.3]), False)',
    ],
)
def test_cov_edge_cases(m, y, rowvar):
    if (
        not isinstance(m, np.ndarray)
        or not isinstance(y, np.ndarray)
        or np.iscomplexobj(m)
        or np.iscomplexobj(y)
    ):
        pytest.xfail()
    py_func = _cov
    jit_func = njit(py_func)
    assert_allclose(
        py_func(m=m, y=y, rowvar=rowvar),
        jit_func(m=m, y=y, rowvar=rowvar),
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "arr",
    [
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32).reshape((3, 3)),
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32).reshape((3, 3)),
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=np.int32).reshape((5, 2)),
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=np.float32).reshape((5, 2)),
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=np.int32).reshape((5, 2)).T,
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=np.float32).reshape((5, 2)).T,
    ],
)
@pytest.mark.parametrize("parallel", [False, True])
def test_mean_loop(arr, parallel):
    def py_func(data):
        tdata = data.T
        m = np.empty(tdata.shape[0])
        for i in numba.prange(tdata.shape[0]):
            m[i] = np.mean(tdata[i])
        return m

    jit_func = njit(py_func, parallel=parallel)
    assert_equal(py_func(arr), jit_func(arr))


@pytest.mark.parametrize(
    "arr",
    [
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32).reshape((3, 3)),
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32).reshape((3, 3)),
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=np.int32).reshape((5, 2)),
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=np.float32).reshape((5, 2)),
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=np.int32).reshape((5, 2)).T,
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=np.float32).reshape((5, 2)).T,
        make_regression(n_samples=2**10, n_features=2**7, random_state=0)[0],
    ],
)
@pytest.mark.parametrize("parallel", [False, True])
def test_mean_loop_cov(arr, parallel):
    def py_func(data):
        tdata = data.T
        m = np.empty(tdata.shape[0])
        for i in numba.prange(tdata.shape[0]):
            m[i] = np.mean(tdata[i])
        c = data - m
        v = np.cov(c.T)
        return c, v

    jit_func = njit(py_func, parallel=parallel)
    c1, v1 = py_func(arr)
    c2, v2 = jit_func(arr)
    assert_allclose(c1, c2, rtol=1e-15, atol=1e-11)
    assert_allclose(v1, v2, rtol=1e-15, atol=1e-11)


@pytest.mark.parametrize(
    "N,k",
    [
        (1, 0),
        (2, -1),
        (2, 0),
        (2, 1),
        (3, -2),
        (3, -1),
        (3, 0),
        (3, 1),
        (3, 2),
    ],
)
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_eye1(N, k, dtype):
    def py_func(N, k):
        return np.eye(N=N, k=k, dtype=dtype)

    jit_func = njit(py_func)
    assert_equal(py_func(N, k), jit_func(N, k))


@pytest.mark.parametrize(
    "N,M,k",
    [
        (2, 3, -1),
        (2, 3, 0),
        (2, 3, 1),
        (3, 2, -1),
        (3, 2, 0),
        (3, 2, 1),
    ],
)
def test_eye2(N, M, k):
    def py_func(N, M, k):
        return np.eye(N, M, k)

    jit_func = njit(py_func)
    assert_equal(py_func(N, M, k), jit_func(N, M, k))


_matmul_inputs_vars = [
    ([2], [3]),
    ([2, 3], [4, 5]),
    ([2, 3], [[2, 3], [4, 5]]),
    ([1, 2, 3], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    ([[2, 3], [4, 5]], [2, 3]),
    ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1, 2, 3]),
    ([[2, 3], [4, 5]], [[2, 3], [4, 5]]),
    (np.arange(4 * 5).reshape(4, 5), np.arange(5)),
    (np.arange(40 * 50).reshape(40, 50), np.arange(50)),
]


@parametrize_function_variants(
    "py_func",
    [
        # 'lambda a, b: np.matmul(a, b)',
        "lambda a, b: a @ b",
    ],
)
@pytest.mark.parametrize(
    "a,b", _matmul_inputs_vars
)  # ids=list(map(str, _matmul_inputs_vars))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_matmul1(py_func, a, b, dtype):
    a = np.array(a, dtype=dtype)
    b = np.array(b, dtype=dtype)
    jit_func = njit(py_func)
    assert_allclose(py_func(a, b), jit_func(a, b), rtol=1e-4, atol=1e-7)


@parametrize_function_variants(
    "py_func",
    [
        "lambda a, b: (a @ b) @ a",
    ],
)
@pytest.mark.parametrize(
    "a,b",
    [
        (np.arange(4 * 5).reshape(4, 5), np.arange(5)),
        (np.arange(20 * 25).reshape(20, 25), np.arange(25)),
        (np.arange(4000 * 5000).reshape(4000, 5000), np.arange(5000)),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_matmul2(py_func, a, b, dtype):
    a = np.array(a, dtype=dtype)
    b = np.array(b, dtype=dtype)
    jit_func = njit(py_func)
    assert_allclose(py_func(a, b), jit_func(a, b), rtol=1e-4, atol=1e-7)
