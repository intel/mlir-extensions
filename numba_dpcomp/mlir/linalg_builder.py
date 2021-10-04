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

from .func_registry import add_func
from .inner_compiler import compile_func

class Var:
    def __init__(self, context, ssa_val):
        self._context = context
        self._ssa_val = ssa_val

    @property
    def shape(self):
        return self._shape(self._context, self._ssa_val)

    @property
    def dtype(self):
        return self._dtype(self._context, self._ssa_val)

    @property
    def type(self):
        return self._type(self._context, self._ssa_val)

    def __len__(self):
        return self._len(self._context, self._ssa_val)

    def __getitem__(self, index):
        return self._getitem(self._context, self._ssa_val, index)

    def __add__(self, o): return self._binop(self._context, self._ssa_val, o, '+')
    def __radd__(self, o): return self._binop(self._context, self._ssa_val, o, '+')
    def __mul__(self, o): return self._binop(self._context, self._ssa_val, o, '*')
    def __rmul__(self, o): return self._binop(self._context, self._ssa_val, o, '*')
    def __truediv__(self, o): return self._binop(self._context, self._ssa_val, o, '/')

    def __str__(self): return self._str(self._context, self._ssa_val)
    def __repr__(self): return self._str(self._context, self._ssa_val)

class Type:
    def __init__(self, mlir_type, eq):
        self._mlir_type = mlir_type
        self._eq = eq

    def __eq__(self, other):
        return self._eq(self._mlir_type, other._mlir_type)

def is_literal(val):
    return not isinstance(val, Var)

DYNAMIC_DIM = -1

class Builder:
    def __init__(self, context):
        self._context = context

    def broadcast(self, *args):
        return self._broadcast(self._context, args)

    def init_tensor(self, shape, dtype, init_val=None):
        return self._init_tensor(self._context, shape, dtype, init_val)

    def fill_tensor(self, tensor, value):
        return self._fill_tensor(self._context, tensor, value)

    def generic(self, inputs, outputs, iterators, maps, body):
        return self._generic(self._context, inputs, outputs, iterators, maps, body)

    def from_elements(self, values, dtype):
        return self._from_elements(self._context, values, dtype)

    def extract(self, value, indices):
        return self._extract(self._context, value, indices)

    def reshape(self, src, dims):
        return self._reshape(self._context, src, dims)

    def external_call(self, name, inputs, outputs, decorate=True):
        return self._external_call(self._context, name, inputs, outputs, decorate)

    def insert(self, src, dst, offsets, sizes, strides):
        return self._insert(self._context, src, dst, offsets, sizes, strides)

    def inline_func(self, func, res_type, *args): # TODO: kwargs
        return self._inline_func(self._context, func, res_type, args)

    def cast(self, arg, dtype):
        return self._cast(self._context, arg, dtype)

    def undef(self, dtype):
        return self._undef(self._context, dtype)

    def array_type(self, dims, dtype):
        return self._array_type(self._context, dims, dtype)

class FuncRegistry:
    def __init__(self):
        self.funcs = {}

    def register_func(self, name, orig_func = None):
        def _decorator(func):
            mangled_name = name + '()'
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

def broadcast_type(builder, args):
    return args[0].dtype # TODO

def eltwise(builder, args, body, res_type = None):
    if isinstance(args, tuple):
        args = builder.broadcast(*args)
    else:
        args = (args,)

    if res_type is None:
        res_type = args[0].dtype

    shape = args[0].shape

    num_dims = len(shape)
    if num_dims == 0:
        dummy = builder.cast(0, res_type)
        return builder.inline_func(body, res_type, *(args + (dummy,)))
    else:
        iterators = ['parallel' for _ in range(num_dims)]
        dims = ','.join(['d%s' % i for i in range(num_dims)])
        expr = f'({dims}) -> ({dims})'
        maps = [expr for _ in range(len(args) + 1)]
        init = builder.init_tensor(shape, res_type)

        return builder.generic(args, init, iterators, maps, body)

def convert_array(builder, arr, dtype):
    if arr.dtype == dtype:
        return arr

    return eltwise(builder, arr, lambda a, b: a, dtype)

def _flatten_tuple(src):
    l = len(src)
    if isinstance(l, int) and l != 0:
        shape, elements = _flatten_tuple(src[0])
        for i in range(1, l):
            shape1, elements1 = _flatten_tuple(src[i])
            assert(shape == shape1)
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
        dtype = broadcast_type(builder, elements)

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
