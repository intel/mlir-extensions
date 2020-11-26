from numba import runtests

import numba.mlir.builtin_funcs
import numba.mlir.numpy_funcs

def test(*args, **kwargs):
    return runtests.main("numba.mlir.tests", *args, **kwargs)
