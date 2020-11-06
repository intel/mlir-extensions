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

    def test_sum(self):
        def py_func(a):
            return a.sum()

        jit_func = njit(py_func)
        arr = np.asarray([1,2,3])
        assert_equal(py_func(arr), jit_func(arr))

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

if __name__ == '__main__':
    unittest.main()
