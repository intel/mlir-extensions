# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from .linalg_builder import eltwise
from .numpy.funcs import register_func
from numba.core.typing.templates import infer_global, CallableTemplate
from numba.core import types
import sys


def vectorize(arg_or_function=(), **kws):
    if inspect.isfunction(arg_or_function):
        return _gen_vectorize(arg_or_function)

    return _gen_vectorize


class _VecFuncTyper(CallableTemplate):
    def generic(self):
        def typer(a):
            if isinstance(a, types.Array):
                return a

        return typer


def _gen_vectorized_func_name(func, mod):
    func_name = f"_{func.__module__}_{func.__qualname__}_vectorized"
    for c in ["<", ">", "."]:
        func_name = func_name.replace(c, "_")

    i = 0
    while True:
        new_name = func_name if i == 0 else f"{func_name}{i}"
        if not hasattr(mod, new_name):
            return new_name
        i += 1


def _gen_vectorize(func):
    num_args = len(inspect.signature(func).parameters)
    if num_args == 1:
        mod = sys.modules[__name__]
        func_name = _gen_vectorized_func_name(func, mod)

        exec(f"def {func_name}(arg): pass")
        vec_func_inner = eval(func_name)

        setattr(mod, func_name, vec_func_inner)
        infer_global(vec_func_inner)(_VecFuncTyper)

        from ..decorators import mlir_njit

        jit_func = mlir_njit(func, inline="always")

        @register_func(func_name, vec_func_inner)
        def impl(builder, arg):
            return eltwise(builder, arg, lambda a, b: jit_func(a))

        def vec_func(arg):
            return vec_func_inner(arg)

        return mlir_njit(vec_func, inline="always")
    else:
        assert False
