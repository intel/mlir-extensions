import inspect
from .linalg_builder import register_func, eltwise
from numba.core.typing.templates import infer_global, CallableTemplate
from numba.core import types
import sys

def vectorize(arg_or_function=(), **kws):
    print(arg_or_function, kws)
    if inspect.isfunction(arg_or_function):
        return _gen_vectorize(arg_or_function)

    return _gen_vectorize

# def _dummy_vec_func(*args): pass

# @infer_global(_dummy_vec_func)
class VecFuncTyper(CallableTemplate):
    def generic(self):
        def typer(a):
            if isinstance(a, types.Array):
                return a
        return typer

def _gen_vectorize(func):
    num_args = len(inspect.signature(func).parameters)
    if num_args == 1:
        func_name =  f'_{func.__module__}_{func.__qualname__}_vectorized'.replace('<', '_').replace('>', '_').replace('.', '_')

        exec(f'def {func_name}(arg): pass')
        vec_func_inner = eval(func_name)
        print(vec_func_inner)
        mod = sys.modules[__name__]
        setattr(mod, func_name, vec_func_inner)
        print(dir(mod))
        infer_global(vec_func_inner)(VecFuncTyper)

        from ..decorators import njit
        jit_func = njit(func)

        @register_func(func_name, vec_func_inner)
        def impl(builder, arg):
            return eltwise(builder, arg, lambda a, b: jit_func(a))

        def vec_func(arg):
            return vec_func_inner(arg)

        return njit(vec_func)
    else:
        assert(False)
