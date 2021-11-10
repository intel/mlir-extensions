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
import sys

from numba import prange
from numba.core import types
from numba.core.typing.templates import AbstractTemplate, ConcreteTemplate, signature, infer_global

from .linalg_builder import is_int, dtype_str, FuncRegistry
from .numpy.funcs import register_func
from .func_registry import add_func

from ..decorators import njit

registry = FuncRegistry()

def _raise_error(desc):
    raise ValueError(desc)

def _stub_error():
    raise NotImplementedError('This is a stub')

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
class _RangeId(ConcreteTemplate):
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

def _set_local_size(*args):
    _stub_error()

@registry.register_func('_set_local_size', _set_local_size)
def _set_local_size_impl(builder, *args):
    nargs = len(args)
    if nargs > 0 and nargs <= 3:
        res = 0 # TODO: dummy ret, remove
        return builder.external_call('set_local_size', inputs=args, outputs=res)

@infer_global(_set_local_size)
class _SetLocalSizeId(ConcreteTemplate):
    cases = [
        signature(types.void, types.int64),
        signature(types.void, types.int64, types.int64),
        signature(types.void, types.int64, types.int64, types.int64),
    ]


def _kernel_body0(global_size, local_size, body, *args):
    body(*args)

def _kernel_body1(global_size, local_size, body, *args):
    _set_local_size(local_size[0])
    for i in _gpu_range(global_size[0]):
        body(*args)

def _kernel_body2(global_size, local_size, body, *args):
    _set_local_size(local_size[1], local_size[0])
    x, y = global_size
    for i in _gpu_range(y):
        for j in _gpu_range(x):
            body(*args)

def _kernel_body3(global_size, local_size, body, *args):
    _set_local_size(local_size[2], local_size[1], local_size[0])
    x, y, z = global_size
    for i in _gpu_range(z):
        for j in _gpu_range(y):
            for k in _gpu_range(x):
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
        global_dim_count = len(global_size)
        local_dim_count = len(local_size)
        assert(local_dim_count <= global_dim_count)
        if local_dim_count < global_dim_count:
            local_size = tuple(local_size[i] if i < local_dim_count else 1 for i in range(global_dim_count))
        ret = self.copy()
        ret.global_size = global_size
        ret.local_size = local_size
        return ret

    def check_call_args(self, args, kwargs):
        if kwargs:
            _raise_error('kwargs not supported')

    def __getitem__(self, args):
        nargs = len(args)
        if nargs < 1 or nargs > 2:
            _raise_error(f'Invalid kernel arguments count: {nargs}')

        gs = _process_dims(args[0])
        ls = _process_dims(args[1]) if nargs > 1 else ()
        return self.configure(gs, ls)

    def __call__(self, *args, **kwargs):
        self.check_call_args(args, kwargs)

        jit_func = njit(inline='always',enable_gpu_pipeline=True)(self.py_func)
        jit_kern = njit(enable_gpu_pipeline=True)(_kernel_body_selector[len(self.global_size)])
        jit_kern(self.global_size, self.local_size, jit_func, *args)


def kernel(func):
    return Kernel(func)

def _define_api_funcs():
    kernel_api_funcs = [
        'get_global_id',
        'get_global_size',
        'get_local_size',
    ]

    def get_func(func_name):
        def api_func_impl(builder, axis):
            if isinstance(axis, int) or is_int(axis):
                res = 0
                return builder.external_call(func_name, axis, res)
        return api_func_impl

    def get_stub_func(func_name):
        exec(f'def {func_name}(axis): _stub_error()')
        return eval(func_name)

    class ApiFuncId(ConcreteTemplate):
        cases = [signature(types.uint64, types.uint64)]

    this_module = sys.modules[__name__]

    for func_name in kernel_api_funcs:
        func = get_stub_func(func_name)
        setattr(this_module, func_name, func)

        infer_global(func)(ApiFuncId)
        registry.register_func(func_name, func)(get_func(func_name))

_define_api_funcs()
del _define_api_funcs

class Stub(object):
    """A stub object to represent special objects which is meaningless
    outside the context of DPPY compilation context.
    """

    __slots__ = ()  # don't allocate __dict__

    def __new__(cls):
        raise NotImplementedError("%s is not instantiable" % cls)


class atomic(Stub):
    pass

def _define_atomic_funcs():
    funcs = ['add', 'sub']

    def get_func(func_name):
        def api_func_impl(builder, arr, idx, val):
            if not (isinstance(idx, int) and idx == 0):
                arr = builder.subview(arr, idx)
            return builder.external_call(f'{func_name}_{dtype_str(builder, arr.dtype)}', (arr, val), val)
        return api_func_impl

    def get_stub_func(func_name):
        exec(f'def {func_name}(arr, idx, val): _stub_error()')
        return eval(func_name)

    class _AtomicId(AbstractTemplate):
        def generic(self, args, kws):
            assert not kws
            ary, idx, val = args

            if ary.ndim == 1:
                return signature(ary.dtype, ary, types.intp, ary.dtype)
            elif ary.ndim > 1:
                return signature(ary.dtype, ary, idx, ary.dtype)

    this_module = sys.modules[__name__]

    for name in funcs:
        func_name = f'atomic_{name}'
        func = get_stub_func(func_name)
        setattr(this_module, func_name, func)

        infer_global(func)(_AtomicId)
        registry.register_func(func_name, func)(get_func(func_name))
        setattr(atomic, name, func)

_define_atomic_funcs()
del _define_atomic_funcs
