# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .func_registry import add_func
from .inner_compiler import compile_func
import numpy


class Var:
    def __init__(self, context, ssa_val):
        self._context = context
        self._ssa_val = ssa_val

    @property
    def shape(self):
        return self._shape(self._context, self._ssa_val)

    @property
    def strides(self):
        return self._strides(self._context, self._ssa_val)

    @property
    def dtype(self):
        return self._dtype(self._context, self._ssa_val)

    @property
    def type(self):
        return self._type(self._context, self._ssa_val)

    def __len__(self):
        res = self._len(self._context, self._ssa_val)
        if res is None:
            raise ValueError("No len")

        return res

    def __getitem__(self, index):
        return self._getitem(self._context, self._ssa_val, index)

    def __add__(self, o):
        return self._binop(self._context, self._ssa_val, o, "+")

    def __radd__(self, o):
        return self._binop(self._context, self._ssa_val, o, "+")

    def __sub__(self, o):
        return self._binop(self._context, self._ssa_val, o, "-")

    def __rsub__(self, o):
        return self._binop(self._context, self._ssa_val, o, "r-")

    def __mul__(self, o):
        return self._binop(self._context, self._ssa_val, o, "*")

    def __rmul__(self, o):
        return self._binop(self._context, self._ssa_val, o, "*")

    def __truediv__(self, o):
        return self._binop(self._context, self._ssa_val, o, "/")

    def __floordiv__(self, o):
        return self._binop(self._context, self._ssa_val, o, "//")

    def __lt__(self, o):
        return self._binop(self._context, self._ssa_val, o, "lt")

    def __le__(self, o):
        return self._binop(self._context, self._ssa_val, o, "le")

    def __gt__(self, o):
        return self._binop(self._context, self._ssa_val, o, "gt")

    def __ge__(self, o):
        return self._binop(self._context, self._ssa_val, o, "ge")

    def __eq__(self, o):
        return self._binop(self._context, self._ssa_val, o, "eq")

    def __ne__(self, o):
        return self._binop(self._context, self._ssa_val, o, "ne")

    def __neg__(self):
        return self._unop(self._context, self._ssa_val, "-")

    def __pos__(self):
        return self._unop(self._context, self._ssa_val, "+")

    def __str__(self):
        return self._str(self._context, self._ssa_val)

    def __repr__(self):
        return self._str(self._context, self._ssa_val)

    def literal(self):
        return self._literal(self._context, self._ssa_val)


def literal(obj):
    if isinstance(obj, Var):
        return obj.literal()

    return obj


class Type:
    def __init__(self, mlir_type, eq, printer):
        self._mlir_type = mlir_type
        self._eq = eq
        self._str = printer

    def __eq__(self, other):
        return self._eq(self._mlir_type, other._mlir_type)

    def __str__(self):
        return self._str(self._mlir_type)

    def __repr__(self):
        return self._str(self._mlir_type)


def is_literal(val):
    return not isinstance(val, Var)


DYNAMIC_DIM = numpy.iinfo(numpy.int64).min


class Builder:
    def __init__(self, context):
        self._context = context

    def broadcast(self, *args, result_type):
        return self._broadcast(self._context, args, result_type)

    def init_tensor(self, shape, dtype, init_val=None):
        return self._init_tensor(self._context, shape, dtype, init_val)

    def fill_tensor(self, tensor, value):
        return self._fill_tensor(self._context, tensor, value)

    def linalg_generic(self, inputs, outputs, iterators, maps, body):
        return self._linalg_generic(
            self._context, inputs, outputs, iterators, maps, body
        )

    def linalg_index(self, dim):
        return self._linalg_index(self._context, dim)

    def from_elements(self, values, dtype=None):
        return self._from_elements(self._context, values, dtype)

    def extract(self, value, indices):
        return self._extract(self._context, value, indices)

    def reshape(self, src, dims):
        return self._reshape(self._context, src, dims)

    def external_call(
        self, name, inputs, outputs, decorate=True, return_tensor=False, attrs=None
    ):
        return self._external_call(
            self._context, name, inputs, outputs, decorate, return_tensor, attrs
        )

    def insert(self, src, dst, offsets, strides):
        return self._insert(self._context, src, dst, offsets, strides)

    def inline_func(self, func, res_type, *args):  # TODO: kwargs
        return self._inline_func(self._context, func, res_type, args)

    def cast(self, arg, dtype):
        return self._cast(self._context, arg, dtype)

    def undef(self, dtype):
        return self._undef(self._context, dtype)

    def subview(self, src, offset, size=None, strides=None, result_rank=None):
        return self._subview(self._context, src, offset, size, strides, result_rank)

    def select(self, cond, true_val, false_val):
        return self._select(self._context, cond, true_val, false_val)

    def force_copy(self, arr):
        return self._force_copy(self._context, arr)

    def array_type(self, dims, dtype):
        return self._array_type(self._context, dims, dtype)


class FuncRegistry:
    def __init__(self):
        self.funcs = {}

    def register_func(self, name, orig_func=None):
        def _decorator(func):
            mangled_name = name + "()"
            assert not mangled_name in self.funcs
            self.funcs[mangled_name] = func
            if not orig_func is None:
                add_func(orig_func, name)
            return func

        return _decorator

    def register_attr(self, name):
        def _decorator(func):
            assert not name in self.funcs
            self.funcs[name] = func
            return func

        return _decorator

    def lookup_func(self, name):
        return self.funcs.get(name)


def _get_numpy_types(builder):
    return [
        (builder.bool, numpy.bool_),
        (builder.int8, numpy.int8),
        (builder.uint8, numpy.uint8),
        (builder.int16, numpy.int16),
        (builder.uint16, numpy.uint16),
        (builder.int32, numpy.int32),
        (builder.uint32, numpy.uint32),
        (builder.int64, numpy.int64),
        (builder.uint64, numpy.uint64),
        (builder.float32, numpy.float32),
        (builder.float64, numpy.float64),
        (builder.complex64, numpy.complex64),
        (builder.complex128, numpy.complex128),
    ]


def type_to_numpy(builder, t):
    for src, dst in _get_numpy_types(builder):
        if t == src:
            return dst

    assert False, f"Cannot convert type: {str(t)}"


def type_from_numpy(builder, t):
    for dst, src in _get_numpy_types(builder):
        if t == src:
            return dst

    assert False, f"Cannot convert type: {str(t)}"


def broadcast_type(builder, args):
    l = len(args)
    assert l > 0
    lhs = args[0]
    if l == 1:
        return lhs
    elif l == 2:
        rhs = args[1]
    else:
        rhs = broadcast_type(builder, args[1:])

    lhs = type_to_numpy(builder, lhs)
    rhs = type_to_numpy(builder, rhs)
    return type_from_numpy(builder, numpy.promote_types(lhs, rhs))


def get_val_type(builder, a):
    if isinstance(a, float):
        return builder.float64
    elif isinstance(a, int):
        return builder.int64
    return a.type


def get_array_type(builder, a):
    if isinstance(a, float):
        return builder.float64
    elif isinstance(a, int):
        return builder.int64
    return a.dtype


def broadcast_type_arrays(builder, args):
    return broadcast_type(builder, tuple(get_array_type(builder, a) for a in args))


def eltwise(builder, args, body, res_type=None):
    if isinstance(args, tuple):
        args = builder.broadcast(
            *args, result_type=broadcast_type_arrays(builder, args)
        )
    else:
        args = (args,)

    if res_type is None:
        res_type = args[0].dtype

    shape = args[0].shape

    try:
        num_dims = len(shape)
    except:
        num_dims = 0

    if num_dims == 0:
        dummy = builder.cast(0, res_type)
        return builder.inline_func(body, res_type, *(args + (dummy,)))
    else:
        iterators = ["parallel" for _ in range(num_dims)]
        dims = ",".join(["d%s" % i for i in range(num_dims)])
        expr = f"({dims}) -> ({dims})"
        maps = [expr for _ in range(len(args) + 1)]
        init = builder.init_tensor(shape, res_type)

        return builder.linalg_generic(args, init, iterators, maps, body)


def convert_array(builder, arr, dtype):
    if arr.dtype == dtype:
        return arr

    return eltwise(builder, arr, lambda a, b: a, dtype)


def _flatten_tuple(src):
    try:
        l = len(src)
    except:
        l = 0

    if l != 0:
        shape, elements = _flatten_tuple(src[0])
        for i in range(1, l):
            shape1, elements1 = _flatten_tuple(src[i])
            assert shape == shape1
            elements += elements1

        if shape is None:
            shape = [l]
        else:
            shape = [l] + shape
        return (shape, elements)
    return (None, [src])


def asarray(builder, src, dtype=None):
    shape, elements = _flatten_tuple(src)

    if shape is None:
        return src

    if dtype is None:
        dtype = broadcast_type_arrays(builder, elements)

    arr = builder.from_elements(elements, dtype)

    if len(shape) > 1:
        arr = builder.reshape(arr, shape)

    return arr


def is_int(t, b):
    types = [
        b.bool,
        b.int8,
        b.uint8,
        b.int16,
        b.uint16,
        b.int32,
        b.uint32,
        b.int64,
        b.uint64,
    ]
    return t in types


def is_float(t, b):
    return t == b.float16 or t == b.float32 or t == b.float64


def is_complex(t, b):
    return t == b.complex64 or t == b.complex128


def dtype_str(builder, dtype):
    names = [
        (builder.bool, "bool"),
        (builder.int8, "int8"),
        (builder.int16, "int16"),
        (builder.int32, "int32"),
        (builder.int64, "int64"),
        (builder.uint8, "uint8"),
        (builder.uint16, "uint16"),
        (builder.uint32, "uint32"),
        (builder.uint64, "uint64"),
        (builder.int8_signless, "int8"),
        (builder.int16_signless, "int16"),
        (builder.int32_signless, "int32"),
        (builder.int64_signless, "int64"),
        (builder.float32, "float32"),
        (builder.float64, "float64"),
    ]
    for t, name in names:
        if t == dtype:
            return name

    assert False, f"dtype_str unhandled type: {dtype}"


def dtype_size(builder, dtype):
    sizes = [
        (builder.bool, 1),
        (builder.int8, 1),
        (builder.int16, 2),
        (builder.int32, 4),
        (builder.int64, 8),
        (builder.uint8, 1),
        (builder.uint16, 2),
        (builder.uint32, 4),
        (builder.uint64, 8),
        (builder.int8_signless, 1),
        (builder.int16_signless, 2),
        (builder.int32_signless, 4),
        (builder.int64_signless, 8),
        (builder.float32, 4),
        (builder.float64, 8),
    ]
    for t, size in sizes:
        if t == dtype:
            return size

    assert False, f"dtype_size unhandled type: {dtype}"
