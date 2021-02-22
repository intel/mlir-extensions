import numba
from numba import njit
from numpy.testing import assert_equal # for nans comparison
import numpy as np
from numba.tests.support import TestCase
import unittest

_arr_1d_int = [1,2,3,4,5,6,7,8]
_arr_1d_float = [1.0,2.1,3.2,4.3,5.4,6.5,7.6,8.7]
_arr_2d_int = [[1,2,3,4],[5,6,7,8]]
_arr_2d_float = [[1.0,2.1,3.2,4.3],[5.4,6.5,7.6,8.7]]
_test_arrays = [_arr_1d_int, _arr_1d_float, _arr_2d_int, _arr_2d_float]
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

    def test_unary(self):
        funcs = [
            lambda a: a.sum(),
            lambda a: np.sum(a),
            lambda a: np.sqrt(a),
            lambda a: np.square(a),
            lambda a: a.size,
            # lambda a: a.T, TODO: need fortran layout support
            lambda a: a.T.T,
        ]

        for py_func in funcs:
            jit_func = njit(py_func)
            for a in _test_arrays:
                arr = np.array(a)
                assert_equal(py_func(arr), jit_func(arr))

    def test_add(self):
        def py_func(a, b):
            return np.add(a, b)

        jit_func = njit(py_func)
        arr1 = np.array([1,2,3])
        arr2 = np.array([4,5,6])
        assert_equal(py_func(arr1,arr2), jit_func(arr1,arr2))

    def test_add_scalar(self):
        def py_func(a, b):
            return np.add(a, b)

        jit_func = njit(py_func)
        arr1 = 1
        arr2 = 2
        assert_equal(py_func(arr1, arr2), jit_func(arr1, arr2))

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
            return shape[0] + shape[1]

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

if __name__ == '__main__':
    unittest.main()
