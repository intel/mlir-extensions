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
    runtime_lib = load_lib('dpcomp-gpu-runtime')
    IS_GPU_RUNTIME_AVAILABLE = True
except:
    IS_GPU_RUNTIME_AVAILABLE = False

if IS_GPU_RUNTIME_AVAILABLE:
    _funcs = [
        'dpcompGpuStreamCreate',
        'dpcompGpuStreamDestroy',
        'dpcompGpuModuleLoad',
        'dpcompGpuModuleDestroy',
        'dpcompGpuKernelGet',
        'dpcompGpuKernelDestroy',
        'dpcompGpuLaunchKernel',
        'dpcompGpuWait',
        'dpcompGpuAlloc',
    ]

    for name in _funcs:
        func = getattr(runtime_lib, name)
        register_cfunc(ll, name, func)


    _alloc_func = runtime_lib.dpcompGpuSetMemInfoAllocFunc
    _alloc_func.argtypes = [ctypes.c_void_p]
    _numba_alloc_ptr = ctypes.cast(_nrt.c_helpers['Allocate'], ctypes.c_void_p)
    _alloc_func(_numba_alloc_ptr.value)
