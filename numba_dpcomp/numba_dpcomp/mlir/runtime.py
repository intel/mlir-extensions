# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import atexit
from numba.np.ufunc.parallel import get_thread_count
from .utils import load_lib, register_cfunc

runtime_lib = load_lib("dpcomp-runtime")

_init_func = runtime_lib.dpcompParallelInit
_init_func.argtypes = [ctypes.c_int]
_init_func(get_thread_count())

_finalize_func = runtime_lib.dpcompParallelFinalize

_funcs = [
    "dpcompParallelFor",
    "memrefCopy",
    "dpcompTakeContext",
    "dpcompReleaseContext",
    "dpcompPurgeContext",
]

for name in _funcs:
    func = getattr(runtime_lib, name)
    register_cfunc(name, func)


@atexit.register
def _cleanup():
    _finalize_func()
