from numba.mlir.func_registry import add_func

from numba import prange

add_func(range, 'range')
add_func(len, 'len')
add_func(bool, 'bool')

add_func(prange, 'numba.prange')
