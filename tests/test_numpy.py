import numba
from numba import njit
from numpy.testing import assert_equal # for nans comparison
import numpy as np
from numba.tests.support import TestCase
import unittest

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

    def test_sum(self):
        def py_func(a):
            return a.sum()

        jit_func = njit(py_func)
        arr = np.asarray([1,2,3])
        assert_equal(py_func(arr), jit_func(arr))

    def test_add_scalar(self):
        def py_func(a, b):
            return np.add(a, b)

        jit_func = njit(py_func)
        arr1 = 1
        arr2 = 2
        assert_equal(py_func(arr1, arr2), jit_func(arr1, arr2))

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

    def test_setitem(self):
        def py_func(a, b):
            a[b] = 42
            return a[b]

        jit_func = njit(py_func)
        arr = np.asarray([1,2,3])
        assert_equal(py_func(arr, 1), jit_func(arr, 1))

    def test_setitem_loop(self):
        def py_func(a):
            for i in range(len(a)):
                a[i] = a[i] + i
            return a.sum()

        jit_func = njit(py_func)
        arr = np.asarray([3,2,1])
        assert_equal(py_func(arr.copy()), jit_func(arr.copy()))

    def test_array_bounds(self):
        def py_func(a):
            res = 0
            for i in range(len(a)):
                if i >= len(a):
                    res = res + 1
                else:
                    res = res + a[i]
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

if __name__ == '__main__':
    unittest.main()
