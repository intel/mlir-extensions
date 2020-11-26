from numba.mlir.func_registry import add_func

add_func(range, 'range')
add_func(len, 'len')
add_func(bool, 'bool')
