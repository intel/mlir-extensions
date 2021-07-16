# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numba
# from numba_dpcomp import njit
from numba_dpcomp import vectorize
from numpy.testing import assert_equal, assert_allclose # for nans comparison
import numpy as np
from numba.tests.support import TestCase
import unittest
import itertools
from functools import partial
import pytest
from sklearn.datasets import make_regression

from .utils import parametrize_function_variants
from .utils import njit_cached as njit

np.seterr(all='ignore')

def _vectorize_reference(func, arg1):
    ret = np.empty(arg1.shape, arg1.dtype)
    for ind, val in np.ndenumerate(arg1):
        ret[ind] = func(val)
    return ret

_arr_1d_int = np.array([1,2,3,4,5,6,7,8])
_arr_1d_float = np.array([1.0,2.1,3.2,4.3,5.4,6.5,7.6,8.7])
_arr_2d_int = np.array([[1,2,3,4],[5,6,7,8]])
_arr_2d_float = np.array([[1.0,2.1,3.2,4.3],[5.4,6.5,7.6,8.7]])
_test_arrays = [
    _arr_1d_int,
    _arr_1d_float,
    _arr_2d_int,
    _arr_2d_float,
    _arr_2d_int.T,
    _arr_2d_float.T,
]
_test_arrays_ids = [
    '1d_int',
    '1d_float',
    '2d_int',
    '2d_float',
    '2d_int.T',
    '2d_float.T',
]

@parametrize_function_variants("py_func", [
    'lambda a: a.sum()',
    'lambda a: np.sum(a)',
    'lambda a: np.mean(a)',
    'lambda a: np.sqrt(a)',
    'lambda a: np.square(a)',
    'lambda a: np.log(a)',
    'lambda a: np.sin(a)',
    'lambda a: np.cos(a)',
    'lambda a: a.size',
    'lambda a: a.T',
    'lambda a: a.T.T',
])
@pytest.mark.parametrize("arr",
                         _test_arrays,
                         ids=_test_arrays_ids)
def test_unary(py_func, arr, request):
    jit_func = njit(py_func)
    assert_allclose(py_func(arr), jit_func(arr), rtol=1e-15, atol=1e-15)

_test_binary_test_arrays = [1, 2.5, np.array([1,2,3]), np.array([4.4,5.5,6.6])]
_test_binary_test_arrays_ids = ['1', '2.5', 'np.array([1,2,3])', 'np.array([4.4,5.5,6.6])']
@parametrize_function_variants("py_func", [
    'lambda a, b: np.add(a, b)',
    'lambda a, b: a + b',
    'lambda a, b: np.subtract(a, b)',
    'lambda a, b: a - b',
    'lambda a, b: np.multiply(a, b)',
    'lambda a, b: a * b',
    'lambda a, b: np.power(a, b)',
    'lambda a, b: a ** b',
    'lambda a, b: np.true_divide(a, b)',
    'lambda a, b: a / b',
])
@pytest.mark.parametrize("a",
                         _test_binary_test_arrays,
                         ids=_test_binary_test_arrays_ids)
@pytest.mark.parametrize("b",
                         _test_binary_test_arrays,
                         ids=_test_binary_test_arrays_ids)
def test_binary(py_func, a, b):
    jit_func = njit(py_func)
    assert_equal(py_func(a,b), jit_func(a,b))

_test_broadcast_test_arrays = [
    1,
    np.array([1]),
    np.array([[1]]),
    np.array([[1,2],[3,4]]),
    np.array([5,6]),
    np.array([[5],[6]]),
    np.array([[5,6]]),
]
_test_broadcast_test_arrays_ids = [
    '1',
    'np.array([1])',
    'np.array([[1]])',
    'np.array([[1,2],[3,4]])',
    'np.array([5,6])',
    'np.array([[5],[6]])',
    'np.array([[5,6]])',
]
@pytest.mark.parametrize("a",
                         _test_broadcast_test_arrays,
                         ids=_test_broadcast_test_arrays_ids)
@pytest.mark.parametrize("b",
                         _test_broadcast_test_arrays,
                         ids=_test_broadcast_test_arrays_ids)
def test_broadcast(a, b):
    def py_func(a, b):
        return np.add(a, b)

    jit_func = njit(py_func)
    assert_equal(py_func(a,b), jit_func(a,b))

def test_staticgetitem():
    def py_func(a):
        return a[1]

    jit_func = njit(py_func)
    arr = np.asarray([5,6,7])
    assert_equal(py_func(arr), jit_func(arr))

@pytest.mark.parametrize("i",
                         list(range(3)))
def test_getitem1(i):
    def py_func(a, b):
        return a[b]

    jit_func = njit(py_func)
    arr = np.asarray([5,6,7])
    assert_equal(py_func(arr, i), jit_func(arr, i))

def test_getitem2():
    def py_func(a, b):
        return a[b]

    jit_func = njit(py_func)
    arr = np.asarray([[[1,2,3],[5,6,7]]])
    assert_equal(py_func(arr, 0), jit_func(arr, 0))

def test_getitem3():
    def py_func(a, b, c):
        return a[b, c]

    jit_func = njit(py_func)
    arr = np.asarray([[[1,2,3],[5,6,7]]])
    assert_equal(py_func(arr, 0, 0), jit_func(arr, 0, 0))

def test_array_len():
    def py_func(a):
        return len(a)

    jit_func = njit(py_func)
    arr = np.asarray([5,6,7])
    assert_equal(py_func(arr), jit_func(arr))

@parametrize_function_variants("py_func", [
    'lambda a: np.sum(a, axis=0)',
    'lambda a: np.sum(a, axis=1)',
    ])
@pytest.mark.parametrize("arr", [
    np.array([[1,2,3],[4,5,6]], dtype=np.int32),
    np.array([[1,2,3],[4,5,6]], dtype=np.float32),
    ])
def test_sum_axis(py_func, arr):
    jit_func = njit(py_func)
    assert_equal(py_func(arr), jit_func(arr))

def test_sum_add():
    def py_func(a, b):
        return np.add(a, b).sum()

    jit_func = njit(py_func)
    arr1 = np.asarray([1,2,3])
    arr2 = np.asarray([4,5,6])
    assert_equal(py_func(arr1, arr2), jit_func(arr1, arr2))

def test_sum_add2():
    def py_func(a, b, c):
        t = np.add(a, b)
        return np.add(t, c).sum()

    jit_func = njit(py_func)
    arr1 = np.asarray([1,2,3])
    arr2 = np.asarray([4,5,6])
    arr3 = np.asarray([7,8,9])
    assert_equal(py_func(arr1, arr2, arr3), jit_func(arr1, arr2, arr3))

@pytest.mark.parametrize("a,b", [
    (np.array([1,2,3], np.float32), np.array([4,5,6], np.float32)),
    (np.array([[1,2,3],[4,5,6]], np.float32), np.array([[1,2],[3,4],[5,6]], np.float32)),
    ])
@pytest.mark.parametrize("parallel", [False, True])
def test_dot(a, b, parallel):
    def py_func(a, b):
        return np.dot(a, b)

    jit_func = njit(py_func, parallel=parallel)
    assert_equal(py_func(a, b), jit_func(a, b))

class TestMlirBasic(TestCase):
    def test_static_setitem(self):
        def py_func(a):
            a[1] = 42
            return a[1]

        jit_func = njit(py_func)
        arr = np.asarray([1,2,3])
        assert_equal(py_func(arr), jit_func(arr))

    def test_setitem1(self):
        def py_func(a, b):
            a[b] = 42
            return a[b]

        jit_func = njit(py_func)
        arr = np.asarray([1,2,3])
        assert_equal(py_func(arr, 1), jit_func(arr, 1))

    def test_setitem2(self):
        def py_func(a, b, c):
            a[b, c] = 42
            return a[b, c]

        jit_func = njit(py_func)
        arr = np.asarray([[1,2,3],[4,5,6]])
        assert_equal(py_func(arr, 1, 2), jit_func(arr, 1, 2))

    def test_setitem_loop(self):
        def py_func(a):
            for i in range(len(a)):
                a[i] = a[i] + i
            return a.sum()

        jit_func = njit(py_func)
        arr = np.asarray([3,2,1])
        assert_equal(py_func(arr.copy()), jit_func(arr.copy()))

    def test_array_bounds1(self):
        def py_func(a):
            res = 0
            for i in range(len(a)):
                if i >= len(a) or i < 0:
                    res = res + 1
                else:
                    res = res + a[i]
            return res

        jit_func = njit(py_func)
        arr = np.asarray([3,2,1])
        assert_equal(py_func(arr.copy()), jit_func(arr.copy()))

    def test_array_bounds2(self):
        def py_func(a):
            res = 0
            for i in range(len(a)):
                if i < len(a) and i >= 0:
                    res = res + a[i]
                else:
                    res = res + 1
            return res

        jit_func = njit(py_func)
        arr = np.asarray([3,2,1])
        assert_equal(py_func(arr.copy()), jit_func(arr.copy()))

    def test_array_bounds3(self):
        def py_func(a):
            res = 0
            for i in range(len(a)):
                if 0 <= i < len(a):
                    res = res + a[i]
                else:
                    res = res + 1
            return res

        jit_func = njit(py_func)
        arr = np.asarray([3,2,1])
        assert_equal(py_func(arr.copy()), jit_func(arr.copy()))

    def test_array_bounds4(self):
        def py_func(a):
            res = 0
            for i in range(len(a) - 1):
                if 0 <= i < (len(a) - 1):
                    res = res + a[i]
                else:
                    res = res + 1
            return res

        jit_func = njit(py_func)
        arr = np.asarray([3,2,1])
        assert_equal(py_func(arr.copy()), jit_func(arr.copy()))

    def test_array_shape(self):
        def py_func(a):
            shape = a.shape
            return shape[0] + shape[1] * 10

        jit_func = njit(py_func)
        arr = np.array([[1,2,3],[4,5,6]])
        assert_equal(py_func(arr), jit_func(arr))

    def test_array_return(self):
        def py_func(a):
            return a

        jit_func = njit(py_func)
        arr = np.array([1,2,3])
        assert_equal(py_func(arr), jit_func(arr))

    def test_array_prange_const(self):
        def py_func(a, b):
            a[0] = 42
            for i in numba.prange(b):
                a[0] = 1
            return a[0]

        jit_func = njit(py_func, parallel=True)
        arr = np.array([0.0])
        assert_equal(py_func(arr, 5), jit_func(arr, 5))

    def test_empty1(self):
        def py_func(d):
            a = np.empty(d)
            for i in range(d):
                a[i] = i
            return a

        jit_func = njit(py_func)
        assert_equal(py_func(5), jit_func(5))

    def test_empty2(self):
        def py_func(d1, d2):
            a = np.empty((d1, d2))
            for i in range(d1):
                for j in range(d2):
                    a[i, j] = i + j * 10
            return a

        jit_func = njit(py_func)
        assert_equal(py_func(5, 7), jit_func(5, 7))

    def test_empty3(self):
        def py_func(a):
            return np.empty(a.shape, a.dtype)

        jit_func = njit(py_func)
        arr = np.array([1,2,3])
        for t in ['int32','int64','float32','float64']:
            a = arr.astype(t)
            assert_equal(py_func(a).shape, jit_func(a).shape)
            assert_equal(py_func(a).dtype, jit_func(a).dtype)

    def test_zeros1(self):
        def py_func(d):
            return np.zeros(d)

        jit_func = njit(py_func)
        assert_equal(py_func(5), jit_func(5))

    def test_zeros2(self):
        def py_func(a):
            return np.zeros(a.shape, a.dtype)

        jit_func = njit(py_func)
        arr = np.array([1, 2, 3])
        for t in ['int32', 'int64', 'float32', 'float64']:
            a = arr.astype(t)
            assert_equal(py_func(a).shape, jit_func(a).shape)
            assert_equal(py_func(a).dtype, jit_func(a).dtype)

    @unittest.expectedFailure
    def test_zeros3(self):
        def py_func(d):
            return np.zeros(d, dtype=np.dtype('int64'))

        jit_func = njit(py_func)
        assert_equal(py_func(5), jit_func(5))

    def test_zeros4(self):
        def py_func(d):
            return np.zeros(d)

        jit_func = njit(py_func)
        assert_equal(py_func((2, 1)), jit_func((2, 1)))

    def test_parallel(self):
        def py_func(a, b):
            return np.add(a, b)

        jit_func = njit(py_func, parallel=True)
        arr = np.asarray([[[1,2,3],[4,5,6]],
                          [[1,2,3],[4,5,6]]])
        assert_equal(py_func(arr,arr), jit_func(arr,arr))

    def test_parallel_reduce(self):
        def py_func(a):
            shape = a.shape
            res = 0
            for i in range(shape[0]):
                for j in numba.prange(shape[1]):
                    for k in numba.prange(shape[2]):
                        res = res + a[i,j,k]
            return res

        jit_func = njit(py_func, parallel=True)
        arr = np.asarray([[[1,2,3],[4,5,6]]]).repeat(10000,0)
        assert_equal(py_func(arr), jit_func(arr))

    def test_vectorize(self):
        import math
        funcs = [
            lambda a : a + 1,
            lambda a : math.erf(a),
            # lambda a : 5 if a == 1 else a, # TODO: investigate
        ]

        for func in funcs:
            vec_func = vectorize(func)

            for a in _test_arrays:
                arr = np.array(a)
                assert_equal(_vectorize_reference(func, arr), vec_func(arr))

    def test_vectorize_indirect(self):
        def func(a):
            return a + 1

        vec_func = vectorize(func)

        def py_func(a):
            return vec_func(a)

        jit_func = njit(py_func, parallel=True)

        for a in _test_arrays:
            arr = np.array(a)
            assert_equal(_vectorize_reference(func, arr), jit_func(arr))

    def test_fortran_layout(self):
        def py_func(a):
            return a.T

        jit_func = njit(py_func)

        arr = np.array([[1,2],[3,4]])
        for a in [arr]: # TODO: arr.T
            assert_equal(py_func(a), jit_func(a))

@parametrize_function_variants("a", [
    # 'np.array(1)', TODO zero rank arrays
    # 'np.array(2.5)',
    'np.array([])',
    'np.array([1,2,3])',
    'np.array([[1,2,3]])',
    'np.array([[1,2],[3,4],[5,6]])',
    ])
def test_atleast2d(a):
    def py_func(a):
        return np.atleast_2d(a)

    jit_func = njit(py_func)
    assert_equal(py_func(a), jit_func(a))

_test_reshape_test_array = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
_test_reshape_test_arrays = [
    _test_reshape_test_array,
    _test_reshape_test_array.reshape((2,6)),
    _test_reshape_test_array.reshape((2,3,2)),
]
@parametrize_function_variants("py_func", [
    'lambda a: a.reshape(a.size)',
    'lambda a: a.reshape((a.size,))',
    'lambda a: a.reshape((a.size,1))',
    'lambda a: a.reshape((1, a.size))',
    'lambda a: a.reshape((1, a.size, 1))',
    ])
@pytest.mark.parametrize("array", _test_reshape_test_arrays)
def test_reshape(py_func, array):
    jit_func = njit(py_func)
    assert_equal(py_func(array), jit_func(array))

@parametrize_function_variants("py_func", [
    # 'lambda a: a.flat', TODO: flat support
    'lambda a: a.flatten()',
    ])
@pytest.mark.parametrize("array", _test_reshape_test_arrays)
def test_flatten(py_func, array):
    jit_func = njit(py_func)
    assert_equal(py_func(array), jit_func(array))

@parametrize_function_variants("py_func", [
    'lambda a, b: ()',
    'lambda a, b: (a,b)',
    'lambda a, b: ((a,b),(a,a),(b,b),())',
    ])
@pytest.mark.parametrize("a,b",
        itertools.product(*(([1,2.5,np.array([1,2,3]), np.array([4.5,6.7,8.9])],)*2))
    )
def test_tuple_ret(py_func, a, b):
    jit_func = njit(py_func)
    assert_equal(py_func(a, b), jit_func(a, b))

@pytest.mark.parametrize("arrays",
                         [([1,2,3],[4,5,6]),
                          ([[1,2],[3,4]],[[5,6],[7,8]]),
                          ([[[1],[2]],[[3],[4]]],[[[5],[6]],[[7],[8]]]),
                          ([1,2,3],[4,5,6],[7,8,9]),
                          ([1,2],[3,4],[5,6],[7,8]),
                         ])
@pytest.mark.parametrize("axis",
                         [0,1,2]) # TODO: None
def test_concat(arrays, axis):
    arr = tuple(np.array(a) for a in arrays)
    num_dims = len(arr[0].shape);
    if axis >= num_dims:
        pytest.skip() # TODO: unselect
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

@parametrize_function_variants("py_func", [
    'lambda a, b, c, d: a[b:c]',
    'lambda a, b, c, d: a[3:c]',
    'lambda a, b, c, d: a[b:4]',
    'lambda a, b, c, d: a[3:4]',
    'lambda a, b, c, d: a[b:c:d]',
    'lambda a, b, c, d: a[b:c:1]',
    'lambda a, b, c, d: a[b:c:2]',
    'lambda a, b, c, d: a[3:4:2]',
    ])
def test_slice(py_func):
    arr = np.array([1,2,3,4,5,6,7,8])
    jit_func = njit(py_func)
    assert_equal(py_func(arr, 3, 4, 2), jit_func(arr, 3, 4, 2))

def test_multidim_slice():
    def py_func(a, b):
        return a[1, b,:]
    jit_func = njit(py_func)

    a = np.array([[[1],[2],[3]],[[4],[5],[6]]])
    assert_equal(py_func(a, 0), jit_func(a, 0))

def test_size_ret():
    def py_func(a, b):
        return a.size / b
    jit_func = njit(py_func)

    a = np.array([[[1],[2],[3]],[[4],[5],[6]]])
    assert_equal(py_func(a, 3), jit_func(a, 3))

@pytest.mark.parametrize("a", [
    np.array([[1,2],[4,5]])
    ])
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

@parametrize_function_variants("m", [
    'np.array([[0, 2], [1, 1], [2, 0]]).T',
    '_rnd.randn(100).reshape(5, 20)',
    'np.asfortranarray(np.array([[0, 2], [1, 1], [2, 0]]).T)',
    '_rnd.randn(100).reshape(5, 20)[:, ::2]',
    'np.array([0.3942, 0.5969, 0.7730, 0.9918, 0.7964])',
    # 'np.full((4, 5), fill_value=True)', TODO
    'np.array([np.nan, 0.5969, -np.inf, 0.9918, 0.7964])',
    'np.linspace(-3, 3, 33).reshape(33, 1)',

    # non-array inputs
    '((0.1, 0.2), (0.11, 0.19), (0.09, 0.21))',  # UniTuple
    '((0.1, 0.2), (0.11, 0.19), (0.09j, 0.21j))',  # Tuple
    '(-2.1, -1, 4.3)',
    '(1, 2, 3)',
    '[4, 5, 6]',
    '((0.1, 0.2, 0.3), (0.1, 0.2, 0.3))',
    '[(1, 2, 3), (1, 3, 2)]',
    '3.142',
    # '((1.1, 2.2, 1.5),)',

    # empty data structures
    'np.array([])',
    'np.array([]).reshape(0, 2)',
    'np.array([]).reshape(2, 0)',
    '()',
    ])
def test_cov_basic(m):
    if isinstance(m, (list, float)) or len(m) == 0 or np.iscomplexobj(m):
        pytest.xfail()
    py_func = _cov
    jit_func = njit(py_func)
    assert_allclose(py_func(m), jit_func(m), rtol=1e-15, atol=1e-15)

_cov_inputs_m = _rnd.randn(105).reshape(15, 7)
@pytest.mark.parametrize("m",
                         [_cov_inputs_m])
@pytest.mark.parametrize("y",
                         [None, _cov_inputs_m[::-1]])
@pytest.mark.parametrize("rowvar",
                         [False, True])
@pytest.mark.parametrize("bias",
                         [False, True])
@pytest.mark.parametrize("ddof",
                         [None, -1, 0, 1, 3.0, True])
def test_cov_explicit_arguments(m, y, rowvar, bias, ddof):
    py_func = _cov
    jit_func = njit(py_func)
    assert_allclose(py_func(m=m, y=y, rowvar=rowvar, bias=bias, ddof=ddof), jit_func(m=m, y=y, rowvar=rowvar, bias=bias, ddof=ddof), rtol=1e-14, atol=1e-14)

@parametrize_function_variants("m, y, rowvar", [
    '(np.array([-2.1, -1, 4.3]), np.array([3, 1.1, 0.12]), True)',
    '(np.array([1, 2, 3]), np.array([1j, 2j, 3j]), True)',
    '(np.array([1j, 2j, 3j]), np.array([1, 2, 3]), True)',
    '(np.array([1, 2, 3]), np.array([1j, 2j, 3]), True)',
    '(np.array([1j, 2j, 3]), np.array([1, 2, 3]), True)',
    '(np.array([]), np.array([]), True)',
    '(1.1, 2.2, True)',
    '(_rnd.randn(10, 3), np.array([-2.1, -1, 4.3]).reshape(1, 3) / 10, True)',
    '(np.array([-2.1, -1, 4.3]), np.array([[3, 1.1, 0.12], [3, 1.1, 0.12]]), True)',
    # '(np.array([-2.1, -1, 4.3]), np.array([[3, 1.1, 0.12], [3, 1.1, 0.12]]), False)',
    '(np.array([[3, 1.1, 0.12], [3, 1.1, 0.12]]), np.array([-2.1, -1, 4.3]), True)',
    # '(np.array([[3, 1.1, 0.12], [3, 1.1, 0.12]]), np.array([-2.1, -1, 4.3]), False)',
    ])
def test_cov_edge_cases(m, y, rowvar):
    if not isinstance(m, np.ndarray) or not isinstance(y, np.ndarray) or np.iscomplexobj(m) or np.iscomplexobj(y):
        pytest.xfail()
    py_func = _cov
    jit_func = njit(py_func)
    assert_allclose(py_func(m=m, y=y, rowvar=rowvar), jit_func(m=m, y=y, rowvar=rowvar), rtol=1e-14, atol=1e-14)

@pytest.mark.parametrize("arr", [
    np.array([1,2,3,4,5,6,7,8,9], dtype=np.int32).reshape((3,3)),
    np.array([1,2,3,4,5,6,7,8,9], dtype=np.float32).reshape((3,3)),
    np.array([1,2,3,4,5,6,7,8,9,0], dtype=np.int32).reshape((5,2)),
    np.array([1,2,3,4,5,6,7,8,9,0], dtype=np.float32).reshape((5,2)),
    np.array([1,2,3,4,5,6,7,8,9,0], dtype=np.int32).reshape((5,2)).T,
    np.array([1,2,3,4,5,6,7,8,9,0], dtype=np.float32).reshape((5,2)).T,
    ])
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

@pytest.mark.parametrize("arr", [
    np.array([1,2,3,4,5,6,7,8,9], dtype=np.int32).reshape((3,3)),
    np.array([1,2,3,4,5,6,7,8,9], dtype=np.float32).reshape((3,3)),
    np.array([1,2,3,4,5,6,7,8,9,0], dtype=np.int32).reshape((5,2)),
    np.array([1,2,3,4,5,6,7,8,9,0], dtype=np.float32).reshape((5,2)),
    np.array([1,2,3,4,5,6,7,8,9,0], dtype=np.int32).reshape((5,2)).T,
    np.array([1,2,3,4,5,6,7,8,9,0], dtype=np.float32).reshape((5,2)).T,
    make_regression(n_samples=2**10, n_features=2**7, random_state=0)[0],
    ])
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


if __name__ == '__main__':
    unittest.main()
