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
from numba.np.ufunc.parallel import get_thread_count
import llvmlite.binding as ll
import numba_dpcomp

def load_runtume_lib():
    runtime_search_paths = [os.path.dirname(numba_dpcomp.__file__)]

    try:
        runtime_search_paths += os.environ['PYTHONPATH'].split(os.pathsep)
    except KeyError:
        pass

    if sys.platform.startswith('linux'):
        lib_name = 'libdpcomp-runtime.so'
    elif sys.platform.startswith('darwin'):
        lib_name = 'libdpcomp-runtime.dylib'
    elif sys.platform.startswith('win'):
        lib_name = 'dpcomp-runtime.dll'
    else:
        return None

    for path in runtime_search_paths:
        lib_path = lib_name if len(path) == 0 else os.path.join(path, lib_name)
        try:
            return ctypes.CDLL(lib_path)
        except:
            pass

    return None

runtime_lib = load_runtume_lib();
assert not runtime_lib is None

_init_func = runtime_lib.dpcomp_parallel_init
_init_func.argtypes = [ctypes.c_int]
_init_func(get_thread_count())

_finalize_func = runtime_lib.dpcomp_parallel_finalize

_parallel_for_func = runtime_lib.dpcomp_parallel_for
ll.add_symbol('dpcomp_parallel_for', ctypes.cast(_parallel_for_func, ctypes.c_void_p).value)

@atexit.register
def _cleanup():
    _finalize_func()
