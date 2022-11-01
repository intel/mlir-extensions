# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .compiler_context import global_compiler_context
from .. import mlir_compiler

from numba.core.runtime import _nrt_python as _nrt


def _register_symbols():
    names = [
        "MemInfo_alloc_safe_aligned",
        "MemInfo_call_dtor",
    ]

    helpers = _nrt.c_helpers

    for name in names:
        func_name = "NRT_" + name
        sym = helpers[name]
        mlir_compiler.register_symbol(global_compiler_context, func_name, sym)


_register_symbols()
del _register_symbols
