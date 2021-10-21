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

from collections import namedtuple
from itertools import product

from .kernel_impl import Kernel as OrigKernel

_ExecutionState = namedtuple('_ExecutionState', [
    'global_size',
    'local_size',
    'indices',
    ])

_execution_state = None

def get_exec_state():
    global _execution_state
    assert _execution_state is not None
    return _execution_state

def get_global_id_proxy(index):
    return get_exec_state().indices[index]

def get_global_size_proxy(index):
    return get_exec_state().global_size[index]


def _setup_execution_state(global_size, local_size):
    import numba_dpcomp.mlir.kernel_impl
    global _execution_state
    assert _execution_state is None
    _execution_state =_ExecutionState(
        global_size=tuple(reversed(global_size)),
        local_size=tuple(reversed(local_size)),
        indices=[0]*len(global_size))
    return _execution_state


def _destroy_execution_state():
    global _execution_state
    _execution_state = None

_globals_to_replace = [
    ('get_global_id', get_global_id_proxy),
    ('get_global_size', get_global_size_proxy),
]

def _replace_globals(src):
    old_globals = [src.get(name, None) for name, _ in _globals_to_replace]
    for name, new_val in _globals_to_replace:
        src[name] = new_val
    return old_globals

def _restore_globals(src, old_globals):
    for i, (name, _) in enumerate(_globals_to_replace):
        old_val = old_globals[i]
        if old_val is not None:
            src[name] = old_val

def _execute_kernel(global_size, local_size, func, *args):
    saved_globals = _replace_globals(func.__globals__)
    state = _setup_execution_state(global_size, local_size)
    try:
        for indices in product(*(range(d) for d in global_size)):
            state.indices[:] = indices
            func(*args)
    finally:
        _restore_globals(func.__globals__, saved_globals)
        _destroy_execution_state()


class Kernel(OrigKernel):
    def __init__(self, func):
        super().__init__(func)

    def __call__(self, *args, **kwargs):
        self.check_call_args(args, kwargs)

        _execute_kernel(self.global_size, self.local_size, self.py_func, *args)


def kernel(func):
    return Kernel(func)
