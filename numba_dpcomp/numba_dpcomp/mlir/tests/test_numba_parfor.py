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

import sys
import copy
import numbers
import pytest
import numpy as np
import types as pytypes

from numba_dpcomp import njit, jit, vectorize

from numba_dpcomp.mlir.passes import (
    print_pass_ir,
    get_print_buffer,
    is_print_buffer_empty,
)

import numba.tests.test_parfors


def _gen_tests():
    testcases = [
        numba.tests.test_parfors.TestPrangeBasic,
        numba.tests.test_parfors.TestPrangeSpecific,
        numba.tests.test_parfors.TestParforsVectorizer,
    ]

    xfail_tests = {
        "test_prange03mul",
        "test_prange09",
        "test_prange03sub",
        "test_prange10",
        "test_prange03",
        "test_prange03div",
        "test_prange07",
        "test_prange06",
        "test_prange16",
        "test_prange12",
        "test_prange04",
        "test_prange13",
        "test_prange25",
        "test_prange21",
        "test_prange14",
        "test_prange18",
        "test_prange_nested_reduction1",
        "test_list_setitem_hoisting",
        "test_prange23",
        "test_prange24",
        "test_list_comprehension_prange",
        "test_prange22",
        "test_prange_raises_invalid_step_size",
        "test_kde_example",
        "test_issue7501",
        "test_parfor_race_1",
        "test_check_alias_analysis",
        "test_nested_parfor_push_call_vars",
        "test_record_array_setitem_yield_array",
        "test_record_array_setitem",
        "test_multiple_call_getattr_object",
        "test_prange_two_instances_same_reduction_var",
        "test_prange_conflicting_reduction_ops",
        "test_ssa_false_reduction",
        "test_prange26",
        "test_prange_two_conditional_reductions",
        "test_argument_alias_recarray_field",
        "test_mutable_list_param",
        "test_signed_vs_unsigned_vec_asm",
        "test_unsigned_refusal_to_vectorize",
        "test_vectorizer_fastmath_asm",
    }

    skip_tests = {
        "test_prange27",
        "test_copy_global_for_parfor",
    }

    def _wrap_test_class(test_base):
        class _Wrapper(test_base):
            def _gen_normal(self, func):
                return njit()(func)

            def _gen_parallel(self, func):
                def wrapper(*args, **kwargs):
                    with print_pass_ir([], ["ParallelToTbbPass"]):
                        res = njit(parallel=True)(func)(*args, **kwargs)
                        ir = get_print_buffer()
                        # Check some parallel loops were actually generated
                        assert ir.count("plier_util.parallel") > 0, ir
                    return res

                return wrapper

            def _gen_parallel_fastmath(self, func):
                def wrapper(*args, **kwargs):
                    with print_pass_ir([], ["PostLLVMLowering"]):
                        res = njit(parallel=True, fastmath=True)(func)(*args, **kwargs)
                        ir = get_print_buffer()
                        # Check some fastmath llvm flags were generated
                        count = 0
                        for line in ir.splitlines():
                            for op in ("fadd", "fsub", "fmul", "fdiv", "frem", "fcmp"):
                                if line.count("llvm." + op) and line.count(
                                    "llvm.fastmath<fast>"
                                ):
                                    count += 1
                        assert count > 0, ir
                    return res

                return wrapper

            def get_gufunc_asm(self, func, schedule_type, *args, **kwargs):
                assert False

            def prange_tester(self, pyfunc, *args, **kwargs):
                patch_instance = kwargs.pop("patch_instance", None)
                scheduler_type = kwargs.pop("scheduler_type", None)
                check_fastmath = kwargs.pop("check_fastmath", False)
                check_fastmath_result = kwargs.pop("check_fastmath_result", False)
                check_scheduling = kwargs.pop("check_scheduling", True)
                check_args_for_equality = kwargs.pop("check_arg_equality", None)
                assert not kwargs, "Unhandled kwargs: " + str(kwargs)

                pyfunc = self.generate_prange_func(pyfunc, patch_instance)

                cfunc = self._gen_normal(pyfunc)
                cpfunc = self._gen_parallel(pyfunc)

                if check_fastmath or check_fastmath_result:
                    fastmath_pcres = self._gen_parallel_fastmath(pyfunc)

                def copy_args(*args):
                    if not args:
                        return tuple()
                    new_args = []
                    for x in args:
                        if isinstance(x, np.ndarray):
                            new_args.append(x.copy("k"))
                        elif isinstance(x, np.number):
                            new_args.append(x.copy())
                        elif isinstance(x, numbers.Number):
                            new_args.append(x)
                        elif isinstance(x, tuple):
                            new_args.append(copy.deepcopy(x))
                        elif isinstance(x, list):
                            new_args.append(x[:])
                        else:
                            raise ValueError("Unsupported argument type encountered")
                    return tuple(new_args)

                # python result
                py_args = copy_args(*args)
                py_expected = pyfunc(*py_args)

                # njit result
                njit_args = copy_args(*args)
                njit_output = cfunc(*njit_args)

                # parfor result
                parfor_args = copy_args(*args)
                parfor_output = cpfunc(*parfor_args)

                if check_args_for_equality is None:
                    np.testing.assert_almost_equal(njit_output, py_expected, **kwargs)
                    np.testing.assert_almost_equal(parfor_output, py_expected, **kwargs)
                    self.assertEqual(type(njit_output), type(parfor_output))
                else:
                    assert len(py_args) == len(check_args_for_equality)
                    for pyarg, njitarg, parforarg, argcomp in zip(
                        py_args, njit_args, parfor_args, check_args_for_equality
                    ):
                        argcomp(njitarg, pyarg, **kwargs)
                        argcomp(parforarg, pyarg, **kwargs)

                # Ignore check_scheduling
                # if check_scheduling:
                #     self.check_scheduling(cpfunc, scheduler_type)

                # if requested check fastmath variant
                if check_fastmath or check_fastmath_result:
                    parfor_fastmath_output = fastmath_pcres(*copy_args(*args))
                    if check_fastmath_result:
                        np.testing.assert_almost_equal(
                            parfor_fastmath_output, py_expected, **kwargs
                        )

        return _Wrapper

    def _replace_global(func, name, newval):
        if name in func.__globals__:
            func.__globals__[name] = newval

    def _gen_test_func(func):
        _replace_global(func, "jit", jit)
        _replace_global(func, "njit", njit)
        _replace_global(func, "vectorize", vectorize)

        def wrapper():
            return func()

        return wrapper

    this_module = sys.modules[__name__]
    for tc in testcases:
        inst = _wrap_test_class(tc)()
        for func_name in dir(tc):
            if func_name.startswith("test"):
                func = getattr(inst, func_name)
                if callable(func):
                    func = _gen_test_func(func)
                    if func_name in xfail_tests:
                        func = pytest.mark.xfail(func)
                    elif func_name in skip_tests:
                        func = pytest.mark.skip(func)

                    setattr(this_module, func_name, func)


_gen_tests()
del _gen_tests
