from numba.mlir.func_registry import add_func

import math

_funcs = ['log', 'sqrt', 'exp', 'erf']

for f in _funcs:
    fname = 'math.' + f
    add_func(eval(fname), fname)


