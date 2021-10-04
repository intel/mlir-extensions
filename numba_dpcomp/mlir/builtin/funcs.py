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

from ..linalg_builder import FuncRegistry, is_int, is_float

import math

registry = FuncRegistry()

def register_func(name, orig_func=None):
    global registry
    return registry.register_func(name, orig_func)

@register_func('bool', bool)
def bool_cast_impl(builder, arg):
    return builder.cast(arg, builder.bool)

@register_func('int', int)
def int_cast_impl(builder, arg):
    return builder.cast(arg, builder.int64)

@register_func('float', float)
def float_cast_impl(builder, arg):
    return builder.cast(arg, builder.float64)

@register_func('len', len)
def len_impl(builder, arg):
    l = len(arg)
    if l is not None:
        return builder.cast(l, builder.int64)

def _gen_math_funcs():
    def get_func(name):
        def func(builder, arg):
            t = arg.type
            if not is_int(t, builder) and not is_float(t, builder):
                return None

            fname = name
            if t == builder.float32:
                fname = 'f' + fname
            elif t != builder.float64:
                t = builder.float64
                arg = builder.cast(arg, builder.float64)

            res = builder.cast(0, t)
            return builder.external_call(fname, arg, res, decorate=False)

        return func

    math_funcs = ['log', 'sqrt', 'exp', 'erf', 'sin', 'cos']

    for func in math_funcs:
        fname = 'math.' + func
        py_func = eval(fname)
        register_func(fname, py_func)(get_func(func))

_gen_math_funcs()
del _gen_math_funcs
