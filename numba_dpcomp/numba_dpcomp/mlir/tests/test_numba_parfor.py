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

from numba_dpcomp.mlir.settings import _readenv
from numba_dpcomp import njit, jit, vectorize

from numba.core.registry import CPUDispatcher
from numba_dpcomp.mlir.passes import (
    print_pass_ir,
    get_print_buffer,
    is_print_buffer_empty,
)

import numba.tests.test_parfors


def _gen_tests():
    if not _readenv("DPCOMP_ENABLE_PARFOR_TESTS", int, 0):
        return

    testcases = [
        numba.tests.test_parfors.TestPrangeBasic,
        numba.tests.test_parfors.TestPrangeSpecific,
        numba.tests.test_parfors.TestParforsVectorizer,
        numba.tests.test_parfors.TestParforBasic,
        numba.tests.test_parfors.TestParforNumericalMisc,
        numba.tests.test_parfors.TestParforNumPy,
        numba.tests.test_parfors.TestParfors,
        numba.tests.test_parfors.TestParforsLeaks,
        numba.tests.test_parfors.TestParforsSlice,
        # numba.tests.test_parfors.TestParforsOptions,
        numba.tests.test_parfors.TestParforsBitMask,
        numba.tests.test_parfors.TestParforsMisc,
        # numba.tests.test_parfors.TestParforsDiagnostics,
    ]

    xfail_tests = {
        "test_prange03sub", # sub reduction
        "test_prange03div", # div reduction
        "test_prange25", # list support
        "test_prange18", # memssa failure
        "test_list_setitem_hoisting", # list support
        "test_list_comprehension_prange", # list comprehension support
        "test_prange_raises_invalid_step_size", # we actually support arbirary step in prange
        "test_issue7501", # invalid tensor<->memref canonicalization
        "test_parfor_race_1", # cfg->scf conversion failure
        "test_nested_parfor_push_call_vars", # Can't resolve function 'negative'
        "test_record_array_setitem_yield_array", # Record and string support
        "test_record_array_setitem", # Record and string support
        "test_multiple_call_getattr_object", # Can't resolve function 'negative'
        "test_prange_two_instances_same_reduction_var", # Non-trivial reduction
        "test_prange_conflicting_reduction_ops",  # Conflicting reduction reduction check
        "test_ssa_false_reduction", # Frontend: object has no attribute 'name'
        "test_argument_alias_recarray_field", # Record support
        "test_mutable_list_param", # List support
        "test_signed_vs_unsigned_vec_asm", # Need to hook asm checks
        "test_unsigned_refusal_to_vectorize", # Need to hook asm checks
        "test_vectorizer_fastmath_asm", # Need to hook asm checks
        "test_kde_example", # List suport
        "test_prange27", # Literal return issue
        "test_simple01", # Empty shape not failed
        "test_kmeans", # List suport
        "test_simple14", # Slice shape mismatch
        "test_ndarray_fill", # array.fill
        "test_fuse_argmin_argmax_max_min", # numpy argmin, argmax
        "test_max", # max reduction
        "test_min", # min reduction
        "test_arange", # numpy.arange
        "test_pi", # np.random.ranf
        "test_simple20", # AssertionError not raised
        "test_simple24", # numpy.arange
        "test_0d_array", # numpy prod
        "test_argmin", # numpy.argmin
        "test_argmax", # numpy.argmax
        "test_simple07", # complex128 support
        "test_ndarray_fill2d", # array.fill
        "test_simple18", # np.linalg.svd
        "test_linspace", # np.linspace
        "test_std", # array.std
        "test_reshape_with_neg_one", # unsupported reshape
        "test_mvdot", # np.dot unsupported args
        "test_array_tuple_concat", # tuple concat
        "test_namedtuple1", # namedtuple support
        "test_0d_broadcast", # np.array
        "test_var", # array.var
        "test_reshape_with_too_many_neg_one", # unsupported reshape
        "test_namedtuple2", # namedtuple support
        "test_simple19", # np.dot unsupported args
        "test_no_hoisting_with_member_function_call", # set support
        "test_parfor_dtype_type", # dtype cast
        "test_tuple3", # numpy.arange
        "test_parfor_array_access3", # TypeError: unsupported operand type(s) for -: 'NoneType' and 'NoneType'
        "test_preparfor_canonicalize_kws", # array.argsort
        "test_parfor_array_access4", # np.dot unsupported args
        "test_tuple_concat_with_reverse_slice", # enumerate
        "test_reduce", # functools.reduce
        "test_two_d_array_reduction", # np.arange
        "test_tuple_concat", # tuple concat
        "test_two_d_array_reduction_with_float_sizes", # np.array
        "test_two_d_array_reduction_reuse", # np.arange
        "test_parfor_slice21", # unsupported reshape
        "test_parfor_array_access_lower_slice", # np.arange
        "test_size_assertion", # AssertionError not raised
        "test_parfor_slice18", # np.arange
        "test_simple12", # complex128
        "test_parfor_slice2", # AssertionError not raised
        "test_parfor_slice6", # array.transpose
        "test_parfor_slice22", # slice using array
        "test_simple13", # complex128
        "test_parfor_bitmask1", # setitem with mask
        "test_parfor_bitmask2", # setitem with mask
        "test_parfor_bitmask3", # setitem with mask
        "test_parfor_bitmask4", # setitem with mask
        "test_parfor_bitmask5", # setitem with mask
        "test_parfor_bitmask6", # setitem with mask
        "test_issue3169", # list support
        "test_issue3748", # unituple of literal dynamic getitem
        "test_issue5001", # list suport
        "test_issue5167", # np.full
        "test_issue6095_numpy_max", # operand #1 does not dominate this use
        "test_issue5065", # tuple unpack
        "test_no_state_change_in_gufunc_lowering_on_error", # custom pipeline
        "test_namedtuple3", # namedtuple
        "test_issue6102", # list support
        "test_oversized_tuple_as_arg_to_kernel", # UnsupportedParforsError not raised
        "test_issue5942_2", # invalid result
        "test_reshape_with_large_neg", # unsupported reshape
        "test_parfor_ufunc_typing", # np.isinf
        "test_issue_5098", # list support and more
        "test_parfor_slice27", # Literal return issue
        "test_ufunc_expr", # np.bitwise_and(
        "test_parfor_generate_fuse", # operand #0 does not dominate this use
        "test_parfor_slice7", # array.transpose
        "test_one_d_array_reduction", # np.array
    }

    skip_tests = {}

    def countParfors(test_func, args, **kws):
        pytest.xfail()

    def countArrays(test_func, args, **kws):
        pytest.xfail()

    def countArrayAllocs(test_func, args, **kws):
        pytest.xfail()

    def countNonParforArrayAccesses(test_func, args, **kws):
        pytest.xfail()

    def get_optimized_numba_ir(test_func, args, **kws):
        pytest.xfail()

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
                        if ir.count("plier_util.parallel") == 0:
                            # In some cases we can canonicalize all loops away
                            # Make sure no loops are present
                            assert ir.count("scf.for") == 0, ir
                            assert ir.count("scf.parallel") == 0, ir
                    return res

                return wrapper

            def _gen_parallel_fastmath(self, func):
                ops = (
                    "fadd",
                    "fsub",
                    "fmul",
                    "fdiv",
                    "frem",
                    "fcmp",
                )

                def wrapper(*args, **kwargs):
                    with print_pass_ir([], ["PostLLVMLowering"]):
                        res = njit(parallel=True, fastmath=True)(func)(*args, **kwargs)
                        ir = get_print_buffer()
                        # Check some fastmath llvm flags were generated
                        opCount = 0
                        fastCount = 0
                        for line in ir.splitlines():
                            for op in ops:
                                if line.count("llvm." + op) > 0:
                                    opCount += 1
                                    if line.count("llvm.fastmath<fast>") > 0:
                                        fastCount += 1
                                    break
                        if opCount > 0:
                            assert fastCount > 0, it
                    return res

                return wrapper

            def get_gufunc_asm(self, func, schedule_type, *args, **kwargs):
                assert False

            def prange_tester(self, pyfunc, *args, **kwargs):
                patch_instance = kwargs.pop("patch_instance", None)

                pyfunc = self.generate_prange_func(pyfunc, patch_instance)
                return self._check_impl(pyfunc, *args, **kwargs)

            def check(self, pyfunc, *args, **kwargs):
                if isinstance(pyfunc, CPUDispatcher):
                    pyfunc = pyfunc.py_func

                return self._check_impl(pyfunc, *args, **kwargs)

            def _check_impl(self, pyfunc, *args, **kwargs):
                scheduler_type = kwargs.pop("scheduler_type", None)
                check_fastmath = kwargs.pop("check_fastmath", False)
                check_fastmath_result = kwargs.pop("check_fastmath_result", False)
                check_scheduling = kwargs.pop("check_scheduling", True)
                check_args_for_equality = kwargs.pop("check_arg_equality", None)
                # assert not kwargs, "Unhandled kwargs: " + str(kwargs)

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

        _replace_global(func, "countParfors", countParfors)
        _replace_global(func, "countArrays", countArrays)
        _replace_global(func, "countArrayAllocs", countArrayAllocs)
        _replace_global(
            func, "countNonParforArrayAccesses", countNonParforArrayAccesses
        )
        _replace_global(func, "get_optimized_numba_ir", get_optimized_numba_ir)

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
