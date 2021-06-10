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
from numba_dpcomp import njit, vectorize
from numpy.testing import assert_equal, assert_allclose # for nans comparison
import numpy as np
from numba.tests.support import TestCase
import unittest
import itertools
import pytest

def _vectorize_reference(func, arg1):
    ret = np.empty(arg1.shape, arg1.dtype)
    for ind, val in np.ndenumerate(arg1):
        ret[ind] = func(val)
    return ret

_arr_1d_int = [1,2,3,4,5,6,7,8]
_arr_1d_float = [1.0,2.1,3.2,4.3,5.4,6.5,7.6,8.7]
_arr_2d_int = [[1,2,3,4],[5,6,7,8]]
_arr_2d_float = [[1.0,2.1,3.2,4.3],[5.4,6.5,7.6,8.7]]
_test_arrays = [_arr_1d_int, _arr_1d_float, _arr_2d_int, _arr_2d_float]
_test_arrays_ids = ["1d_int", "1d_float", "2d_int", "2d_float"]

@pytest.mark.parametrize("py_func",
                         [lambda a: a.sum(),
                          lambda a: np.sum(a),
                          lambda a: np.sqrt(a),
                          lambda a: np.square(a),
                          lambda a: np.log(a),
                          lambda a: np.sin(a),
                          lambda a: np.cos(a),
                          lambda a: a.size,
                          # lambda a: a.T, TODO: need fortran layout support
                          lambda a: a.T.T,
                         ],
                         ids=["a.sum", "sum", "sqrt", "square", "log", "sin",
                              "cos", "a.size", "a.T.T"])
@pytest.mark.parametrize("arr_list",
                         _test_arrays,
                         ids=_test_arrays_ids)
def test_unary(py_func, arr_list, request):
    jit_func = njit(py_func)
    arr = np.array(arr_list)
    assert_allclose(py_func(arr), jit_func(arr), rtol=1e-15, atol=1e-15)

class TestMlirBasic(TestCase):

    def test_staticgetitem(self):
        def py_func(a):
            return a[1]

        jit_func = njit(py_func)
        arr = np.asarray([5,6,7])
        assert_equal(py_func(arr), jit_func(arr))

    def test_getitem(self):
        def py_func(a, b):
            return a[b]

        jit_func = njit(py_func)
        arr = np.asarray([5,6,7])
        for i in range(3):
            assert_equal(py_func(arr, i), jit_func(arr, i))

    def test_array_len(self):
        def py_func(a):
            return len(a)

        jit_func = njit(py_func)
        arr = np.asarray([5,6,7])
        assert_equal(py_func(arr), jit_func(arr))

    def test_binary(self):
        funcs = [
            lambda a, b: np.add(a, b),
            lambda a, b: a + b,
            lambda a, b: np.subtract(a, b),
            lambda a, b: a - b,
            lambda a, b: np.multiply(a, b),
            lambda a, b: a * b,
            lambda a, b: np.power(a, b),
            lambda a, b: a ** b,
            lambda a, b: np.true_divide(a, b),
            lambda a, b: a / b,
        ]

        test_data = [1, 2.5, np.array([1,2,3]), np.array([4.4,5.5,6.6])]
        for py_func in funcs:
            jit_func = njit(py_func)
            for a1, a2 in itertools.product(test_data, test_data):
                assert_equal(py_func(a1,a2), jit_func(a1,a2))

    def test_sum_axis(self):
        funcs = [
            lambda a: np.sum(a, axis=0),
            lambda a: np.sum(a, axis=1),
        ]

        for py_func in funcs:
            jit_func = njit(py_func)
            arr = np.array([[1,2,3],[4,5,6]])
            for a in [arr, arr.astype(np.float32)]:
                assert_equal(py_func(a), jit_func(a))

    def test_sum_add(self):
        def py_func(a, b):
            return np.add(a, b).sum()

        jit_func = njit(py_func)
        arr1 = np.asarray([1,2,3])
        arr2 = np.asarray([4,5,6])
        assert_equal(py_func(arr1, arr2), jit_func(arr1, arr2))

    def test_sum_add2(self):
        def py_func(a, b, c):
            t = np.add(a, b)
            return np.add(t, c).sum()

        jit_func = njit(py_func)
        arr1 = np.asarray([1,2,3])
        arr2 = np.asarray([4,5,6])
        arr3 = np.asarray([7,8,9])
        assert_equal(py_func(arr1, arr2, arr3), jit_func(arr1, arr2, arr3))

    def test_dot(self):
        def py_func(a, b):
            return np.dot(a, b)

        jit_func = njit(py_func)
        arr1 = np.asarray([1,2,3], np.float32)
        arr2 = np.asarray([4,5,6], np.float32)
        arr3 = np.asarray([[1,2,3],[4,5,6]], np.float32)
        arr4 = np.asarray([[1,2],[3,4],[5,6]], np.float32)

        for a, b in [(arr1,arr2), (arr3,arr4)]:
            assert_equal(py_func(a, b), jit_func(a, b))

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

    @unittest.expectedFailure
    def test_zeros4(self):
        def py_func(d):
            return np.zeros(d)

        jit_func = njit(py_func)
        assert_equal(py_func((2, 1)), jit_func((2, 1)))

    def test_reshape(self):
        funcs = [
            lambda a: a.reshape(a.size),
            lambda a: a.reshape((a.size,)),
            lambda a: a.reshape((a.size,1)),
            lambda a: a.reshape((1, a.size)),
            lambda a: a.reshape((1, a.size, 1)),
        ]

        arr1 = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
        # arr2 = arr1.reshape((2,6))
        # arr3 = arr1.reshape((2,3,2))
        for py_func in funcs:
            jit_func = njit(py_func)
            # for a in [arr1,arr2,arr3]: TODO: flatten support
            for a in [arr1]:
                assert_equal(py_func(a), jit_func(a))

    def test_broadcast(self):
        def py_func(a, b):
            return np.add(a, b)

        jit_func = njit(py_func)

        test_data = [
            1,
            np.array([1]),
            np.array([[1]]),
            np.array([[1,2],[3,4]]),
            np.array([5,6]),
            np.array([[5],[6]]),
            np.array([[5,6]]),
        ]

        for a, b in itertools.product(test_data, test_data):
            assert_equal(py_func(a,b), jit_func(a,b))

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

    def test_slice1(self):
        funcs = [
            lambda a, b, c, d: a[b:c],
            lambda a, b, c, d: a[b:c:d],
        ]

        arr = np.array([1,2,3,4,5,6,7,8])
        for py_func in funcs:
            jit_func = njit(py_func)

            assert_equal(py_func(arr, 3, 4,2), jit_func(arr, 3, 4,2))

    def test_atleast2d(self):
        def py_func(a):
            return np.atleast_2d(a)

        jit_func = njit(py_func)

        # for val in (1, 2.5, [], [1,2,3], [[1,2],[3,4],[5,6]]): // TODO: unranked array support
        for val in ([], [1,2,3], [[1,2],[3,4],[5,6]]):
            a = np.array(val)
            assert_equal(py_func(a), jit_func(a))

if __name__ == '__main__':
    unittest.main()
