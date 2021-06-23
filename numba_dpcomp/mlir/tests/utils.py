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

_cached_funcs = {}

def njit_cached(func, *args, **kwargs):
    if args or kwargs:
        return njit(func, *args, **kwargs)
    global _cached_funcs
    if func in _cached_funcs:
        return _cached_funcs[func]
    jitted = njit(func)
    _cached_funcs[func] = jitted
    return jitted
