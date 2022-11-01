# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
from numba.core.runtime import _nrt_python as _nrt
from .utils import load_lib, register_cfunc

runtime_lib = load_lib("dpcomp-python-runtime")


def _register_funcs():
    _funcs = ["dpcompAllocMemInfo", "dpcompUnboxSyclInterface"]

    for name in _funcs:
        func = getattr(runtime_lib, name)
        register_cfunc(name, func)

    _alloc_func = runtime_lib.dpcompSetMemInfoAllocFunc
    _alloc_func.argtypes = [ctypes.c_void_p]
    _numba_alloc_ptr = ctypes.cast(_nrt.c_helpers["Allocate"], ctypes.c_void_p)
    _alloc_func(_numba_alloc_ptr.value)


_register_funcs()
del _register_funcs

_alloc_func = runtime_lib.dpcompAllocMemInfo


def get_alloc_func():
    return ctypes.cast(_alloc_func, ctypes.c_void_p).value
