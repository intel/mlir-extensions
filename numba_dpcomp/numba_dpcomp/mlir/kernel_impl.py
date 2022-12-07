# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

from numba import prange
from numba.core import types
from numba.core.typing.npydecl import parse_dtype, parse_shape
from numba.core.types.npytypes import Array
from numba.core.typing.templates import (
    AbstractTemplate,
    ConcreteTemplate,
    signature,
    infer_global,
)

from .linalg_builder import is_int, dtype_str, FuncRegistry
from .numpy.funcs import register_func
from .func_registry import add_func

from ..decorators import mlir_njit
from .kernel_base import KernelBase
from .dpctl_interop import check_usm_ndarray_args

registry = FuncRegistry()


def _stub_error():
    raise NotImplementedError("This is a stub")


class _gpu_range(object):
    def __new__(cls, *args):
        return range(*args)


add_func(_gpu_range, "_gpu_range")


@infer_global(_gpu_range, typing_key=_gpu_range)
class _RangeId(ConcreteTemplate):
    cases = [
        signature(types.range_state32_type, types.int32),
        signature(types.range_state32_type, types.int32, types.int32),
        signature(types.range_state32_type, types.int32, types.int32, types.int32),
        signature(types.range_state64_type, types.int64),
        signature(types.range_state64_type, types.int64, types.int64),
        signature(types.range_state64_type, types.int64, types.int64, types.int64),
        signature(types.unsigned_range_state64_type, types.uint64),
        signature(types.unsigned_range_state64_type, types.uint64, types.uint64),
        signature(
            types.unsigned_range_state64_type, types.uint64, types.uint64, types.uint64
        ),
    ]


def _set_default_local_size():
    _stub_error()


@registry.register_func("_set_default_local_size", _set_default_local_size)
def _set_default_local_size_impl(builder, *args):
    index_type = builder.index
    i64 = builder.int64
    zero = builder.cast(0, index_type)
    res = (zero, zero, zero)
    res = builder.external_call("set_default_local_size", inputs=args, outputs=res)
    return tuple(builder.cast(r, i64) for r in res)


@infer_global(_set_default_local_size)
class _SetDefaultLocalSizeId(ConcreteTemplate):
    cases = [
        signature(
            types.UniTuple(types.int64, 3), types.int64, types.int64, types.int64
        ),
    ]


def _kernel_body(global_size, local_size, body, *args):
    x, y, z = global_size
    lx, ly, lz = local_size
    _set_default_local_size(lx, ly, lz)
    for gi in _gpu_range(x):
        for gj in _gpu_range(y):
            for gk in _gpu_range(z):
                body(*args)


def _kernel_body_def_size(global_size, body, *args):
    x, y, z = global_size
    for gi in _gpu_range(x):
        for gj in _gpu_range(y):
            for gk in _gpu_range(z):
                body(*args)


def _extend_dims(dims):
    l = len(dims)
    if l < 3:
        return tuple(dims + (1,) * (3 - l))
    return dims


class Kernel(KernelBase):
    def __init__(self, func, kwargs):
        super().__init__(func)
        self._jit_func = mlir_njit(inline="always", enable_gpu_pipeline=True)(func)
        self._kern_body = mlir_njit(enable_gpu_pipeline=True, **kwargs)(_kernel_body)
        self._kern_body_def_size = mlir_njit(enable_gpu_pipeline=True, **kwargs)(
            _kernel_body_def_size
        )

    def __call__(self, *args, **kwargs):
        self.check_call_args(args, kwargs)

        # kwargs is not supported
        check_usm_ndarray_args(args)

        local_size = self.local_size
        if len(local_size) != 0:
            self._kern_body(
                _extend_dims(self.global_size),
                _extend_dims(self.local_size),
                self._jit_func,
                *args,
            )
        else:
            self._kern_body_def_size(
                _extend_dims(self.global_size), self._jit_func, *args
            )


def kernel(func=None, **kwargs):
    if func is None:

        def wrapper(f):
            return Kernel(f, kwargs)

        return wrapper
    return Kernel(func, kwargs)


DEFAULT_LOCAL_SIZE = ()

kernel_func = mlir_njit(inline="always")


def _define_api_funcs():
    kernel_api_funcs = [
        "get_global_id",
        "get_local_id",
        "get_group_id",
        "get_global_size",
        "get_local_size",
    ]

    def get_func(func_name):
        def api_func_impl(builder, axis):
            if isinstance(axis, int) or is_int(axis.type, builder):
                res = builder.cast(0, builder.int64)
                return builder.external_call(func_name, axis, res)

        return api_func_impl

    def get_stub_func(func_name):
        exec(f"def {func_name}(axis): _stub_error()")
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
    funcs = ["add", "sub"]

    def get_func(func_name):
        def api_func_impl(builder, arr, idx, val):
            if not (isinstance(idx, int) and idx == 0):
                arr = builder.subview(arr, idx)

            dtype = arr.dtype
            val = builder.cast(val, dtype)
            return builder.external_call(
                f"{func_name}_{dtype_str(builder, dtype)}", (arr, val), val
            )

        return api_func_impl

    def get_stub_func(func_name):
        exec(f"def {func_name}(arr, idx, val): _stub_error()")
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
        func_name = f"atomic_{name}"
        func = get_stub_func(func_name)
        setattr(this_module, func_name, func)

        infer_global(func)(_AtomicId)
        registry.register_func(func_name, func)(get_func(func_name))
        setattr(atomic, name, func)


_define_atomic_funcs()
del _define_atomic_funcs


# mem fence
CLK_LOCAL_MEM_FENCE = 0x1
CLK_GLOBAL_MEM_FENCE = 0x2


def barrier(flags=None):
    _stub_error()


@registry.register_func("barrier", barrier)
def _barrier_impl(builder, flags=None):
    if flags is None:
        flags = CLK_GLOBAL_MEM_FENCE

    res = 0  # TODO: remove
    return builder.external_call("kernel_barrier", inputs=flags, outputs=res)


@infer_global(barrier)
class _BarrierId(ConcreteTemplate):
    cases = [signature(types.void, types.int64), signature(types.void)]


def mem_fence(flags=None):
    _stub_error()


@registry.register_func("mem_fence", mem_fence)
def _memf_fence_impl(builder, flags=None):
    if flags is None:
        flags = CLK_GLOBAL_MEM_FENCE

    res = 0  # TODO: remove
    return builder.external_call("kernel_mem_fence", inputs=flags, outputs=res)


@infer_global(mem_fence)
class _MemFenceId(ConcreteTemplate):
    cases = [signature(types.void, types.int64), signature(types.void)]


class local(Stub):
    pass


def local_array(shape, dtype):
    _stub_error()


setattr(local, "array", local_array)


@infer_global(local_array)
class _LocalId(AbstractTemplate):
    def generic(self, args, kws):
        shape = kws["shape"] if "shape" in kws else args[0]
        dtype = kws["dtype"] if "dtype" in kws else args[1]

        ndim = parse_shape(shape)
        dtype = parse_dtype(dtype)
        arr_type = Array(dtype=dtype, ndim=ndim, layout="C")
        return signature(arr_type, shape, dtype)


@registry.register_func("local_array", local_array)
def _local_array_impl(builder, shape, dtype):
    try:
        len(shape)  # will raise if not available
    except:
        shape = (shape,)

    func_name = f"local_array_{dtype_str(builder, dtype)}_{len(shape)}"
    res = builder.init_tensor(shape, dtype)
    return builder.external_call(
        func_name, inputs=shape, outputs=res, return_tensor=True
    )


class private(Stub):
    pass


def private_array(shape, dtype):
    _stub_error()


setattr(private, "array", private_array)


@infer_global(private_array)
class _PrivateId(AbstractTemplate):
    def generic(self, args, kws):
        shape = kws["shape"] if "shape" in kws else args[0]
        dtype = kws["dtype"] if "dtype" in kws else args[1]

        ndim = parse_shape(shape)
        dtype = parse_dtype(dtype)
        arr_type = Array(dtype=dtype, ndim=ndim, layout="C")
        return signature(arr_type, shape, dtype)


@registry.register_func("private_array", private_array)
def _private_array_impl(builder, shape, dtype):
    try:
        len(shape)  # will raise if not available
    except:
        shape = (shape,)

    func_name = f"private_array_{dtype_str(builder, dtype)}_{len(shape)}"
    res = builder.init_tensor(shape, dtype)
    return builder.external_call(
        func_name, inputs=shape, outputs=res, return_tensor=True
    )


class group(Stub):
    pass


def group_reduce_add(shape, dtype):
    _stub_error()


setattr(group, "reduce_add", group_reduce_add)


@infer_global(group_reduce_add)
class _GroupId(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        elem_type = args[0]

        return signature(elem_type, elem_type)


@registry.register_func("group_reduce_add", group_reduce_add)
def _group_add_impl(builder, value):
    elem_type = value.type
    func_name = f"group_reduce_add_{dtype_str(builder, elem_type)}"
    res = builder.cast(0, elem_type)
    return builder.external_call(func_name, inputs=value, outputs=res)
