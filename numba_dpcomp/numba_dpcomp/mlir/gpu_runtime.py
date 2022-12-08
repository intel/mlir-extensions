# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import atexit
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
            mlir_func_name("get_group_id"),
            mlir_func_name("get_global_size"),
            mlir_func_name("get_local_size"),
            mlir_func_name("kernel_barrier"),
            mlir_func_name("kernel_mem_fence"),
        ]

        from itertools import product

        _types = ["int32", "int64", "float32", "float64"]

        _atomic_ops = ["add", "sub"]
        for o, t in product(_atomic_ops, _types):
            _funcs.append(mlir_func_name(f"atomic_{o}_{t}"))

        for n, t in product(range(8), _types):
            _funcs.append(mlir_func_name(f"local_array_{t}_{n}"))
            _funcs.append(mlir_func_name(f"private_array_{t}_{n}"))

        _group_ops = ["reduce_add"]
        for o, t in product(_group_ops, _types):
            _funcs.append(mlir_func_name(f"group_{o}_{t}"))

        for name in _funcs:
            if hasattr(runtime_lib, name):
                func = getattr(runtime_lib, name)
            else:
                func = 1
            register_cfunc(name, func)

        _alloc_func = runtime_lib.dpcompGpuSetMemInfoAllocFunc
        _alloc_func.argtypes = [ctypes.c_void_p]
        _alloc_func(get_alloc_func())

    _register_funcs()
    del _register_funcs

    get_device_caps_addr = int(
        ctypes.cast(runtime_lib.dpcompGetDeviceCapabilities, ctypes.c_void_p).value
    )
