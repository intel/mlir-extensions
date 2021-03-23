from numba import runtests

from . import builtin_funcs
from . import math_funcs

from .numpy import funcs

def test(*args, **kwargs):
    return runtests.main("numba.mlir.tests", *args, **kwargs)
