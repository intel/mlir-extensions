from numba.mlir.func_registry import add_func

import numpy

add_func(numpy.add, 'numpy.add')
