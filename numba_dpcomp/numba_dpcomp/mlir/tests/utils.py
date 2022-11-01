# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from numba_dpcomp import njit
import inspect
import pytest


def parametrize_function_variants(name, strings):
    caller_frame = inspect.stack()[1]
    caller_module = inspect.getmodule(caller_frame[0])
    g = vars(caller_module)
    funcs = [eval(f, g) for f in strings]
    return pytest.mark.parametrize(name, funcs, ids=strings)


class JitfuncCache:
    def __init__(self, decorator):
        self._cached_funcs = {}
        self._decorator = decorator

    def cached_decorator(self, func, *args, **kwargs):
        if args or kwargs:
            return self._decorator(func, *args, **kwargs)
        cached = self._cached_funcs.get(func)
        if cached is not None:
            return cached

        jitted = self._decorator(func)
        self._cached_funcs[func] = jitted
        return jitted


njit_cache = JitfuncCache(njit)
njit_cached = njit_cache.cached_decorator
