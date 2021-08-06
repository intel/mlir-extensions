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
import atexit
import llvmlite.binding as ll
from .utils import load_lib, mlir_func_name

runtime_lib = load_lib('dpcomp-gpu-runtime')
assert not runtime_lib is None

_funcs = [
    'dpcompGpuStreamCreate',
    'dpcompGpuStreamDestroy',
    'dpcompGpuModuleLoad',
    'dpcompGpuModuleDestroy',
    'dpcompGpuKernelGet',
    'dpcompGpuKernelDestroy',
    'dpcompGpuLaunchKernel',
    'dpcompGpuWait',
]

for name in _funcs:
    func = getattr(runtime_lib, name)
    ll.add_symbol(name, ctypes.cast(func, ctypes.c_void_p).value)
