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

from .compiler_context import global_compiler_context
from .. import mlir_compiler

from numba.core.runtime import _nrt_python as _nrt

def _register_symbols():
    names = [
        'MemInfo_alloc_safe_aligned',
    ]

    helpers = _nrt.c_helpers

    for name in names:
        func_name = 'NRT_' + name
        sym = helpers[name]
        mlir_compiler.register_symbol(global_compiler_context, func_name, sym)

_register_symbols()
del _register_symbols
