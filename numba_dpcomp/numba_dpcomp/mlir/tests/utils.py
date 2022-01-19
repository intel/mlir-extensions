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
