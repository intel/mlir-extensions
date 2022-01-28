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

from ..decorators import mlir_njit

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

def _kernel_marker(*args):
    _stub_error()

@registry.register_func('_kernel_marker', _kernel_marker)
def _kernel_marker_impl(builder, *args):
    if (len(args) == 6):
        res = 0 #TODO: remove
        return builder.external_call('kernel_marker', inputs=args, outputs=res)

@infer_global(_kernel_marker)
class _KernelMarkerId(ConcreteTemplate):
    cases = [
        signature(types.void, types.int64, types.int64, types.int64, types.int64, types.int64, types.int64),
    ]

def _get_default_local_size():
    _stub_error()

@registry.register_func('_get_default_local_size', _get_default_local_size)
def _get_default_local_size_impl(builder, *args):
    res = (0,0,0)
    return builder.external_call('get_default_local_size', inputs=args, outputs=res)

@infer_global(_get_default_local_size)
class _GetDefaultLocalSizeId(ConcreteTemplate):
    cases = [
        signature(types.UniTuple(types.int64, 3), types.int64, types.int64, types.int64),
    ]

@mlir_njit(enable_gpu_pipeline=True)
def _kernel_body(global_size, local_size, body, *args):
    x, y, z = global_size
    lx, ly, lz = local_size
    _kernel_marker(x, y, z, lx, ly, lz)
    gx = (x + lx - 1) // lx
    gy = (y + ly - 1) // ly
    gz = (z + lz - 1) // lz
    for gi in _gpu_range(gx):
        for gj in _gpu_range(gy):
            for gk in _gpu_range(gz):
                for li in _gpu_range(lx):
                    for lj in _gpu_range(ly):
                        for lk in _gpu_range(lz):
                            ibx = (gi * lx + li) < x
                            iby = (gj * ly + lj) < y
                            ibz = (gk * lz + lk) < z
                            in_bounds = ibx and iby and ibz
                            if (in_bounds):
                                body(*args)

@mlir_njit(enable_gpu_pipeline=True)
def _kernel_body_def_size(global_size, body, *args):
    x, y, z = global_size
    lx, ly, lz = _get_default_local_size(x, y, z)
    _kernel_marker(x, y, z, lx, ly, lz)
    gx = (x + lx - 1) // lx
    gy = (y + ly - 1) // ly
    gz = (z + lz - 1) // lz
    for gi in _gpu_range(gx):
        for gj in _gpu_range(gy):
            for gk in _gpu_range(gz):
                for li in _gpu_range(lx):
                    for lj in _gpu_range(ly):
                        for lk in _gpu_range(lz):
                            ibx = (gi * lx + li) < x
                            iby = (gj * ly + lj) < y
                            ibz = (gk * lz + lk) < z
                            in_bounds = ibx and iby and ibz
                            if (in_bounds):
                                body(*args)

def _extend_dims(dims):
    l = len(dims)
    if (l < 3):
        return tuple(dims + (1,) * (3 - l))
    return dims


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
        if local_dim_count != 0 and local_dim_count < global_dim_count:
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

        jit_func = mlir_njit(inline='always',enable_gpu_pipeline=True)(self.py_func)
        local_size = self.local_size
        if (len(local_size) != 0):
            _kernel_body(_extend_dims(self.global_size), _extend_dims(self.local_size), jit_func, *args)
        else:
            _kernel_body_def_size(_extend_dims(self.global_size), jit_func, *args)


def kernel(func):
    return Kernel(func)

DEFAULT_LOCAL_SIZE = ()

kernel_func = mlir_njit(inline='always')

def _define_api_funcs():
    kernel_api_funcs = [
        'get_global_id',
        'get_local_id',
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

            dtype = arr.dtype
            val = builder.cast(val, dtype)
            return builder.external_call(f'{func_name}_{dtype_str(builder, dtype)}', (arr, val), val)
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
