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
import os
import atexit
import sys
import numba_dpcomp
import llvmlite.binding as ll
from .compiler_context import global_compiler_context
from .. import mlir_compiler

def load_lib(name):
    runtime_search_paths = [os.path.dirname(numba_dpcomp.__file__)]

    try:
        runtime_search_paths += os.environ["PYTHONPATH"].split(os.pathsep)
    except KeyError:
        pass

    if sys.platform.startswith("linux"):
        lib_name = f"lib{name}.so"
    elif sys.platform.startswith("darwin"):
        lib_name = f"lib{name}.dylib"
    elif sys.platform.startswith("win"):
        lib_name = f"{name}.dll"
    else:
        assert False, "unsupported platform"

    saved_errors = []
    for path in runtime_search_paths:
        lib_path = lib_name if len(path) == 0 else os.path.join(path, lib_name)
        try:
            return ctypes.CDLL(lib_path)
        except Exception as e:
            saved_errors.append(f'CDLL("{lib_path}"): {str(e)}')

    raise ValueError(f'load_lib("{name}") failed:\n' + "\n".join(saved_errors))


def mlir_func_name(name):
    return "_mlir_ciface_" + name


_registered_cfuncs = []


def register_cfunc(name, cfunc):
    global _registered_cfuncs
    ptr = ctypes.cast(cfunc, ctypes.c_void_p)
    _registered_cfuncs.append(ptr)
    ll.add_symbol(name, ptr.value)
    mlir_compiler.register_symbol(global_compiler_context, name, ptr.value)
