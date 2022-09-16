# Copyright 2022 Intel Corporation
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

from .settings import DEBUG_TYPE, DUMP_LLVM, DUMP_OPTIMIZED, DUMP_ASSEMBLY
from .. import mlir_compiler


def _init_compiler():
    def _print(s):
        print(s, end='')

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
