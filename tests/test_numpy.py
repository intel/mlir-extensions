import numba
from numba import njit
from numpy.testing import assert_equal # for nans comparison
import numpy as np
from numba.tests.support import TestCase
import unittest

class TestMlirBasic(TestCase):

    def test_getitem(self):
        def py_func(a, b):
            return a[b]

        jit_func = njit(py_func)
        arr = np.asarray([5,6,7])
        for i in range(3):
            assert_equal(py_func(arr, i), jit_func(arr, i))

    @unittest.skip
    def test_sum(self):
        def py_func(a):
            return a.sum()

        jit_func = njit(py_func)
        arr = np.asarray([1,2,3])
        assert_equal(py_func(arr), jit_func(arr))

if __name__ == '__main__':
    unittest.main()
