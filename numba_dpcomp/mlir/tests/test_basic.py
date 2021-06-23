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
from numba_dpcomp import njit
from math import nan, inf, isnan
from numpy.testing import assert_equal # for nans comparison

from numba.tests.support import TestCase
import unittest
import pytest
import sys

import itertools

# TODO: nans and infs not tested yet, we are not sure if want exactly follow
# interpreted python rules
_test_values = [-3,-2,-1,0,1,2,3,-2.5,-1.0,-0.5 -0.0, 0.0, 0.5, 1.0, 2.5]
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
            lambda a, b: a % b,
            # TODO: floordiv
            ]

        for py_func in py_funcs:
            jit_func = njit(py_func)
            for a, b in itertools.product(_test_values, _test_values):
                try:
                    assert_equal(py_func(a, b), jit_func(a, b))
                except ZeroDivisionError:
                    pass

    def test_inplace_op(self):
        def py_func(a,b):
            a += b
            return a

        jit_func = njit(py_func)
        for a, b in itertools.product(_test_values, _test_values):
            assert_equal(py_func(a, b), jit_func(a, b))

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

    def test_none_args(self):
        def py_func(a, b, c, d):
            return b + d

        jit_func = njit(py_func)
        assert_equal(py_func(None, 7, None, 30), jit_func(None, 7, None, 30))

    def test_ret_none(self):
        def py_func1():
            return None

        def py_func2():
            pass

        jit_func1 = njit(py_func1)
        jit_func2 = njit(py_func2)
        assert_equal(py_func1(), jit_func1())
        assert_equal(py_func2(), jit_func2())

    def test_if1(self):
        def py_func(a, b):
            c = 3
            if a > 5:
                c = c + a
            c = c + b
            return c

        jit_func = njit(py_func)
        for a, b in itertools.product(_test_values, _test_values):
            assert_equal(py_func(a, b), jit_func(a, b))

    def test_if2(self):
        def py_func(a, b):
            if a > b:
                return a + b
            else:
                return a - b

        jit_func = njit(py_func)
        for a, b in itertools.product(_test_values, _test_values):
            assert_equal(py_func(a, b), jit_func(a, b))

    def test_tuple1(self):
        def py_func(a, b, c):
            t = (a,b,c)
            return t[0] + t[1] + t[2]

        jit_func = njit(py_func)
        for a, b, c in itertools.product(_test_values, _test_values, _test_values):
            assert_equal(py_func(a, b, c), jit_func(a, b, c))

    def test_tuple2(self):
        def py_func(a, b, c):
            t = (a,b,c)
            x, y, z = t
            return x + y + y

        jit_func = njit(py_func)
        for a, b, c in itertools.product(_test_values, _test_values, _test_values):
            assert_equal(py_func(a, b, c), jit_func(a, b, c))

    def test_tuple_len(self):
        def py_func(a, b, c):
            t = (a,b,c)
            return len(t)

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

    def test_range_negative_step(self):
        def py_func(a, b, c):
            res = 0
            for i in range(a, b, c):
                res = res + i
            return res

        jit_func = njit(py_func)
        assert_equal(py_func(5, -8, -2), jit_func(5, -8, -2))

    def test_range_const_step1(self):
        def py_func(a, b):
            res = 0
            for i in range(a, b, -2):
                res = res + i
            return res

        jit_func = njit(py_func)
        assert_equal(py_func(5, -8), jit_func(5, -8))

    def test_range_const_step2(self):
        def py_func(a, b):
            res = 0
            for i in range(a, b, 2):
                res = res + i
            return res

        jit_func = njit(py_func)
        assert_equal(py_func(-5, 8), jit_func(-5, 8))

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

    def test_range_nested1(self):
        def py_func(a, b, c):
            res = 0
            for i in range(a):
                for j in range(b):
                    for k in range(c):
                        res = res + i
            return res

        jit_func = njit(py_func)
        assert_equal(py_func(10, 20, 2), jit_func(10, 20, 2))

    def test_range_nested2(self):
        def py_func(a, b, c):
            res = 0
            for i in range(a):
                for j in range(b):
                    for k in range(c):
                        res = res + i + j * 10 + k * 100
            return res

        jit_func = njit(py_func)
        assert_equal(py_func(10, 20, 2), jit_func(10, 20, 2))

    def test_prange1(self):
        def py_func(a):
            res = 0
            for i in numba.prange(a):
                res = res + i
            return res

        jit_func = njit(py_func, parallel=True)
        assert_equal(py_func(10), jit_func(10))

    def test_prange2(self):
        def py_func(a, b):
            res = 0
            for i in numba.prange(a, b):
                res = res + i
            return res

        jit_func = njit(py_func, parallel=True)
        assert_equal(py_func(10, 20), jit_func(10, 20))

    def test_prange_reduce1(self):
        def py_func(a):
            res = 0
            for i in numba.prange(1, a):
                res = res + i
            return res

        jit_func = njit(py_func, parallel=True)
        assert_equal(py_func(10), jit_func(10))

    def test_prange_reduce2(self):
        def py_func(a):
            res = 1
            for i in numba.prange(1, a):
                res = res * i
            return res

        jit_func = njit(py_func, parallel=True)
        assert_equal(py_func(10), jit_func(10))

    def test_prange_reduce3(self):
        def py_func(a):
            res1 = 0
            res2 = 1
            for i in numba.prange(1, a):
                res1 = res1 + i
                res2 = res2 * i
            return res1 + res2

        jit_func = njit(py_func, parallel=True)
        assert_equal(py_func(10), jit_func(10))


    def test_func_call1(self):
        def py_func1(b):
            return b + 3

        jit_func1 = njit(py_func1)

        def py_func2(a):
            return jit_func1(a) * 4

        jit_func2 = njit(py_func2)

        assert_equal(py_func2(10), jit_func2(10))

    def test_func_call2(self):
        def py_func1(b):
            return b + 3

        jit_func1 = njit(py_func1)

        def py_func2(a):
            return jit_func1(a) * jit_func1(a + 1)

        jit_func2 = njit(py_func2)

        assert_equal(py_func2(10), jit_func2(10))

    def test_while(self):
        def py_func_simple(a, b):
            while a < b:
                a = a * 2
            return a

        def py_func_multiple_conds1(a, b):
            while a < 44 and a < b:
                a = a * 2
            return a

        def py_func_multiple_conds2(a, b):
            while not a >= 44 and a < b:
                a = a * 2
            return a

        def py_func_multiple_conds3(a, b):
            while a < 44 and not a >= b:
                a = a * 2
            return a

        def py_func_multiple_conds4(a, b):
            while not a >= 44 and not a >= b:
                a = a * 2
            return a

        def py_func_break_middle(a, b):
            while a < b:
                a = a * 2
                if a == 3: break
                a = a + 1
            return a

        def py_func_nested_break(a, b):
            while a < b:
                a = a * 2
                if a == 3 or a == 7:
                    a = a + 7
                else:
                    break
                a = a + 1
            return a

        funcs = [
            py_func_simple,
            py_func_multiple_conds1,
            py_func_multiple_conds2,
            py_func_multiple_conds3,
            py_func_multiple_conds4,
            py_func_break_middle,
            # py_func_nested_break,
        ]

        for py_func in funcs:
            jit_func = njit(py_func)
            assert_equal(py_func(1,66), jit_func(1,66))


    def test_omitted_args1(self):
        def py_func(a = 3, b = 7):
            return a + b

        jit_func = njit(py_func)
        assert_equal(py_func(), jit_func())

    # DPNP is available only on Linux and changes versions of dependencies
    # Looks like it makes effect and test fails:
    # RuntimeError: Failed in nopython mode pipeline (step: <class 'numba_dpcomp.mlir.passes.MlirBackend'>)
    # Cannot generate LLVM module
    # cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface` registration for dialect for op: plier.arg
    @pytest.mark.skipif(sys.platform in ['linux'], reason="Unexpected behaviour in DPNP/Linux environment")
    def test_omitted_args2(self):
        def py_func(a = True, b = False):
            res = 1
            if a:
                res = res + 1
            if b:
                res = res * 2
            return res

        jit_func = njit(py_func)
        assert_equal(py_func(), jit_func())

    def test_omitted_args3(self):
        def py_func1(a = None):
            return a

        jit_func1 = njit(py_func1)

        def py_func2(a = None):
            return jit_func1(a)

        jit_func2 = njit(py_func2)

        assert_equal(py_func2(), jit_func2())
        assert_equal(py_func2(1), jit_func2(1))

if __name__ == '__main__':
    unittest.main()
