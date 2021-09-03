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

import copy

from numba import prange
from numba.core import types
from numba.core.typing.templates import ConcreteTemplate, signature, infer_global

from .linalg_builder import register_func, is_int
from numba_dpcomp.mlir.func_registry import add_func

from ..decorators import njit

def _raise_error(desc):
    raise ValueError(desc)

def _process_dims(dims):
    if isinstance(dims, int):
        return (dims,)
    elif isinstance(dims, (list, tuple)):
        n = len(dims)
        if n > 3:
            _raise_error(f'Invalid dimentions count: {n}')
        return tuple(dims)
    else:
        _raise_error(f'Invalid dimentions type: {type(dims)}')

class _gpu_range(object):
    def __new__(cls, *args):
        return range(*args)

add_func(_gpu_range, '_gpu_range')

@infer_global(_gpu_range, typing_key=_gpu_range)
class Range(ConcreteTemplate):
    cases = [
        signature(types.range_state32_type, types.int32),
        signature(types.range_state32_type, types.int32, types.int32),
        signature(types.range_state32_type, types.int32, types.int32,
                  types.int32),
        signature(types.range_state64_type, types.int64),
        signature(types.range_state64_type, types.int64, types.int64),
        signature(types.range_state64_type, types.int64, types.int64,
                  types.int64),
        signature(types.unsigned_range_state64_type, types.uint64),
        signature(types.unsigned_range_state64_type, types.uint64, types.uint64),
        signature(types.unsigned_range_state64_type, types.uint64, types.uint64,
                  types.uint64),
    ]

def _kernel_body0(global_size, local_size, body, *args):
    body(*args)

def _kernel_body1(global_size, local_size, body, *args):
    for i in _gpu_range(global_size[0]):
        body(*args)

def _kernel_body2(global_size, local_size, body, *args):
    for i in _gpu_range(global_size[0]):
        for j in _gpu_range(global_size[1]):
            body(*args)

def _kernel_body3(global_size, local_size, body, *args):
    for i in _gpu_range(global_size[0]):
        for j in _gpu_range(global_size[1]):
            for k in _gpu_range(global_size[2]):
                body(*args)

_kernel_body_selector = [
    _kernel_body0,
    _kernel_body1,
    _kernel_body2,
    _kernel_body3,
]

class Kernel:
    def __init__(self, func):
        self.global_size = ()
        self.local_size = ()
        self.py_func = func

    def copy(self):
        return copy.copy(self)

    def configure(self, global_size, local_size):
        ret = self.copy()
        ret.global_size = global_size
        ret.local_size = local_size
        return ret

    def __getitem__(self, args):
        nargs = len(args)
        if nargs < 1 or nargs > 2:
            _raise_error(f'Invalid kernel arguments count: {nargs}')

        gs = _process_dims(args[0])
        ls = _process_dims(args[1]) if nargs > 1 else ()
        return self.configure(gs, ls)

    def __call__(self, *args, **kwargs):
        if kwargs:
            _raise_error('kwargs not supported')

        jit_func = njit(parallel=True, inline='always')(self.py_func)
        jit_kern = njit(parallel=True)(_kernel_body_selector[len(self.global_size)])
        jit_kern(self.global_size, self.local_size, jit_func, *args)


def kernel(func):
    return Kernel(func)

def _stub_error():
    raise NotImplementedError('This is a stub')

def get_global_id(axis):
    _stub_error()


@infer_global(get_global_id)
class GetGlobalId(ConcreteTemplate):
    cases = [signature(types.uint64, types.uint64)]

@register_func('get_global_id', get_global_id)
def get_global_id_impl(builder, axis):
    if isinstance(axis, int) or is_int(axis):
        res = 0
        return builder.external_call('get_global_id', axis, res)
