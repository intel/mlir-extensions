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
from .utils import load_lib, mlir_func_name, register_cfunc

from numba.core.runtime import _nrt_python as _nrt

try:
    runtime_lib = load_lib("dpcomp-gpu-runtime")
    IS_GPU_RUNTIME_AVAILABLE = True
except:
    IS_GPU_RUNTIME_AVAILABLE = False


if IS_GPU_RUNTIME_AVAILABLE:
    from .python_rt import get_alloc_func

    def _register_funcs():
        _funcs = [
            "dpcompGpuStreamCreate",
            "dpcompGpuStreamDestroy",
            "dpcompGpuModuleLoad",
            "dpcompGpuModuleDestroy",
            "dpcompGpuKernelGet",
            "dpcompGpuKernelDestroy",
            "dpcompGpuLaunchKernel",
            "dpcompGpuSuggestBlockSize",
            "dpcompGpuWait",
            "dpcompGpuAlloc",
            mlir_func_name("get_global_id"),
            mlir_func_name("get_local_id"),
            mlir_func_name("get_global_size"),
            mlir_func_name("get_local_size"),
        ]

        _atomic_ops = ["add", "sub"]
        _atomic_ops_types = ["int32", "int64", "float32", "float64"]

        from itertools import product

        for o, t in product(_atomic_ops, _atomic_ops_types):
            _funcs.append(mlir_func_name("atomic_" + o + "_" + t))

        for name in _funcs:
            func = getattr(runtime_lib, name)
            register_cfunc(ll, name, func)

        _alloc_func = runtime_lib.dpcompGpuSetMemInfoAllocFunc
        _alloc_func.argtypes = [ctypes.c_void_p]
        _alloc_func(get_alloc_func())

    _register_funcs()
    del _register_funcs
