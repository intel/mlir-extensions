import numba
from numba import njit
from math import nan, inf
from numpy.testing import assert_equal # for nans comparison

from numba.tests.support import TestCase
import unittest

import itertools

_test_values = [-3,-2,-1,0,1,2,3,-2.5,-1.0,-0.5 -0.0, 0.0, 0.5, 1.0, 2.5, -inf, inf] # TODO: nans
class TestMlirBasic(TestCase):

    def test_ret(self):
        def py_func(a):
            return a

        jit_func = njit(py_func)
        for val in _test_values:
            assert_equal(py_func(val), jit_func(val))

    def test_ops(self):
        py_funcs = [
            lambda a, b: a + b,
            lambda a, b: a - b,
            lambda a, b: a * b,
            lambda a, b: a / b,
            # TODO: floordiv
            ]

        for py_func in py_funcs:
            jit_func = njit(py_func)
            for a, b in itertools.product(_test_values, _test_values):
                try:
                    assert_equal(py_func(a, b), jit_func(a, b))
                except ZeroDivisionError:
                    pass

    def test_unary_ops(self):
        py_funcs = [
            lambda a: +a,
            lambda a: -a,
            ]

        for py_func in py_funcs:
            jit_func = njit(py_func)
            for a in _test_values:
                assert_equal(py_func(a), jit_func(a))

    def test_cmp_ops(self):
        py_funcs = [
            lambda a, b: a if a > b else b,
            lambda a, b: a if a < b else b,
            lambda a, b: a if a >= b else b,
            lambda a, b: a if a <= b else b,
            lambda a, b: a if a == b else b,
            lambda a, b: a if a != b else b,
            ]

        for py_func in py_funcs:
            jit_func = njit(py_func)
            for a, b in itertools.product(_test_values, _test_values):
                assert_equal(py_func(a, b), jit_func(a, b))

    def test_const_ops(self):
        py_funcs = [
            lambda a: a + 42,
            lambda a: 43 + a,
            lambda a: a + 42.5,
            lambda a: 43.5 + a,
            ]

        for py_func in py_funcs:
            jit_func = njit(py_func)
            for val in _test_values:
                assert_equal(py_func(val), jit_func(val))

    def test_var(self):
        def py_func(a):
            c = 1
            c = c + a
            return c

        jit_func = njit(py_func)
        for val in _test_values:
            assert_equal(py_func(val), jit_func(val))

    def test_ret_none(self):
        def py_func1():
            return None

        def py_func2():
            pass

        jit_func1 = njit(py_func1)
        jit_func2 = njit(py_func2)
        assert_equal(py_func1(), jit_func1())
        assert_equal(py_func2(), jit_func2())

    def test_jump(self):
        def py_func(a, b):
            c = 3
            if a > 5:
                c = c + a
            c = c + b
            return c

        jit_func = njit(py_func)
        for a, b in itertools.product(_test_values, _test_values):
            assert_equal(py_func(a, b), jit_func(a, b))

    @unittest.skip
    def test_tuple(self):
        def py_func(a, b, c):
            t = (a,b,c)
            return t[0] + t[1] + t[2]

        jit_func = njit(py_func)
        for a, b, c in itertools.product(_test_values, _test_values, _test_values):
            assert_equal(py_func(a, b, c), jit_func(a, b, c))

    def test_range1(self):
        def py_func(a):
            res = 0
            for i in range(a):
                res = res + i
            return res

        jit_func = njit(py_func)
        assert_equal(py_func(10), jit_func(10))

    def test_range2(self):
        def py_func(a, b):
            res = 0
            for i in range(a, b):
                res = res + i
            return res

        jit_func = njit(py_func)
        assert_equal(py_func(10, 20), jit_func(10, 20))

    def test_range3(self):
        def py_func(a, b, c):
            res = 0
            for i in range(a, b, c):
                res = res + i
            return res

        jit_func = njit(py_func)
        assert_equal(py_func(10, 20, 2), jit_func(10, 20, 2))

    def test_range_use_index_after(self):
        def py_func(n):
            res = 0
            for i in range(0, n, 2):
                res = res + i
            return res + i

        jit_func = njit(py_func)
        assert_equal(py_func(9), jit_func(9))

    def test_range_if(self):
        def py_func(n):
            res = 0
            res1 = 2
            for i in range(n):
                if i > 5:
                    res = res + i
                else:
                    res1 = res1 + i * 2
            return res + res1

        jit_func = njit(py_func)
        assert_equal(py_func(10), jit_func(10))

    def test_range_ifs(self):
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

        jit_func = njit(py_func)
        assert_equal(py_func(10), jit_func(10))

    def test_range_continue(self):
        def py_func(n):
            res = 0
            res1 = 2
            for i in range(n):
                res = res + i
                if i < 5:
                    continue
                res1 = res1 + i * 2
            return res + res1

        jit_func = njit(py_func)
        assert_equal(py_func(10), jit_func(10))

    def test_range_nested(self):
        def py_func(a, b, c):
            res = 0
            for i in range(a):
                for j in range(b):
                    for k in range(c):
                        res = res + i + j * 10 + k * 100
            return res

        jit_func = njit(py_func)
        assert_equal(py_func(10, 20, 2), jit_func(10, 20, 2))


if __name__ == '__main__':
    unittest.main()
