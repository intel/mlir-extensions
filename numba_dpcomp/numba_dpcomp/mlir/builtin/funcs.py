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

from ..linalg_builder import FuncRegistry, is_int, is_float, broadcast_type
from ..func_registry import add_func

import math

add_func(slice, "slice")
add_func(range, "range")

registry = FuncRegistry()


def register_func(name, orig_func=None):
    global registry
    return registry.register_func(name, orig_func)


@register_func("bool", bool)
def bool_cast_impl(builder, arg):
    return builder.cast(arg, builder.bool)


@register_func("int", int)
def int_cast_impl(builder, arg):
    return builder.cast(arg, builder.int64)


@register_func("float", float)
def float_cast_impl(builder, arg):
    return builder.cast(arg, builder.float64)


@register_func("len", len)
def len_impl(builder, arg):
    return builder.cast(len(arg), builder.int64)


def _get_type(builder, v):
    if isinstance(v, float):
        return builder.float64
    elif isinstance(v, int):
        return builder.int64
    return v.type


@register_func("min", min)
def min_impl(builder, *args):
    if len(args) > 2:
        rhs = min_impl(builder, *args[1:])
    else:
        rhs = args[1]

    lhs = args[0]
    res_type = broadcast_type(
        builder, (_get_type(builder, lhs), _get_type(builder, rhs))
    )
    lhs = builder.cast(lhs, res_type)
    rhs = builder.cast(rhs, res_type)
    cond = lhs < rhs
    return builder.select(cond, lhs, rhs)


@register_func("max", max)
def max_impl(builder, *args):
    if len(args) > 2:
        rhs = max_impl(builder, *args[1:])
    else:
        rhs = args[1]

    lhs = args[0]
    res_type = broadcast_type(
        builder, (_get_type(builder, lhs), _get_type(builder, rhs))
    )
    lhs = builder.cast(lhs, res_type)
    rhs = builder.cast(rhs, res_type)
    cond = lhs > rhs
    return builder.select(cond, lhs, rhs)


def _gen_math_funcs():
    def get_func(name, N):
        def func(builder, *args):
            if len(args) != N:
                return None

            t = args[0].type
            if not is_int(t, builder) and not is_float(t, builder):
                return None

            for a in args[1:]:
                if a.type != t:
                    return None

            fname = name
            if t == builder.float32:
                fname = "f" + fname
            elif t != builder.float64:
                t = builder.float64
                args = tuple(builder.cast(arg, builder.float64) for arg in args)

            res = builder.cast(0, t)
            return builder.external_call(fname, args, res, decorate=False)

        return func

    math_funcs = [
        ("log", 1),
        ("sqrt", 1),
        ("exp", 1),
        ("erf", 1),
        ("sin", 1),
        ("cos", 1),
        ("tanh", 1),
        ("atan2", 2),
    ]

    for func, N in math_funcs:
        fname = "math." + func
        py_func = eval(fname)
        register_func(fname, py_func)(get_func(func, N))


_gen_math_funcs()
del _gen_math_funcs
