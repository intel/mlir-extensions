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

import ctypes
import atexit
import llvmlite.binding as ll
from .utils import load_lib, mlir_func_name

runtime_lib = load_lib('dpcomp-math-runtime')
assert not runtime_lib is None

_init_func = runtime_lib.dpcomp_math_runtime_init
_init_func()


def load_function_variants(func_name, suffixes):
    for s in suffixes:
        name = func_name + s
        mlir_name = mlir_func_name(name)
        func = getattr(runtime_lib, name)
        ll.add_symbol(mlir_name, ctypes.cast(func, ctypes.c_void_p).value)

load_function_variants('dpcomp_linalg_eig_', ['float32','float64'])

_finalize_func = runtime_lib.dpcomp_math_runtime_finalize

@atexit.register
def _cleanup():
    _finalize_func()
