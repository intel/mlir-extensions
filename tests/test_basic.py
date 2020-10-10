import numba
from numba import njit

from numba.tests.support import TestCase
import unittest

import itertools

_test_values = [-3,-2,-1,0,1,2,3]
class TestMlirBasic(TestCase):

    def test_ret(self):
        def py_func(a):
            return a

        jit_func = njit(py_func)
        for val in _test_values:
            self.assertEqual(py_func(val), jit_func(val))

    def test_ops(self):
        py_funcs = [
            lambda a, b: a + b,
            #lambda a, b: a - b,
            ]

        for py_func in py_funcs:
            jit_func = njit(py_func)
            for a, b in itertools.product(_test_values, _test_values):
                self.assertEqual(py_func(a, b), jit_func(a, b))


if __name__ == '__main__':
    unittest.main()
