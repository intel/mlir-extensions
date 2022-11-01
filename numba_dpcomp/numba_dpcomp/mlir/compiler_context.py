# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .settings import DEBUG_TYPE, DUMP_LLVM, DUMP_OPTIMIZED, DUMP_ASSEMBLY
from .. import mlir_compiler


def _init_compiler():
    def _print(s):
        print(s, end="")

    def _get_printer(enabled):
        return _print if enabled else None

    settings = {}
    settings["debug_type"] = DEBUG_TYPE
    settings["llvm_printer"] = _get_printer(DUMP_LLVM)
    settings["optimized_printer"] = _get_printer(DUMP_OPTIMIZED)
    settings["asm_printer"] = _get_printer(DUMP_ASSEMBLY)
    return mlir_compiler.init_compiler(settings)


global_compiler_context = _init_compiler()
del _init_compiler
