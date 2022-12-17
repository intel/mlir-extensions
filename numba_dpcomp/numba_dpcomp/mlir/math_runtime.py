# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import atexit
from .utils import load_lib, mlir_func_name, register_cfunc
from .settings import MKL_AVAILABLE

runtime_lib = load_lib("dpcomp-math-runtime")

_init_func = runtime_lib.dpcompMathRuntimeInit
_init_func()


def load_function_variants(func_name, suffixes):
    for s in suffixes:
        name = func_name % s
        mlir_name = mlir_func_name(name)
        func = getattr(runtime_lib, name)
        register_cfunc(mlir_name, func)


load_function_variants("dpcompLinalgEig_%s", ["float32", "float64"])
if MKL_AVAILABLE:
    load_function_variants("mkl_gemm_%s", ["float32", "float64"])
    load_function_variants("mkl_gemm_%s_device", ["float32", "float64"])

_finalize_func = runtime_lib.dpcompMathRuntimeFinalize


@atexit.register
def _cleanup():
    _finalize_func()
