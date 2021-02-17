from  ..linalg_builder import register_func

import numpy
import math

def eltwise(builder, args, body, res_type = None):
    if isinstance(args, tuple):
        args = builder.broadcast(*args)
    else:
        args = (args,)

    if res_type is None:
        res_type = args[0].dtype

    shape = args[0].shape

    num_dims = len(shape)
    iterators = ['parallel' for _ in range(num_dims)]
    dims = ','.join(['d%s' % i for i in range(num_dims)])
    expr = f'({dims}) -> ({dims})'
    maps = [expr for _ in range(len(args) + 1)]
    init = builder.init_tensor(shape, res_type)

    return builder.generic(args, init, iterators, maps, body)

@register_func('numpy.add', numpy.add)
def add_impl(builder, arg1, arg2):
    def body(a, b, c):
        return a + b

    return eltwise(builder, (arg1, arg2), body)

@register_func('array.sum')
def sum_impl(builder, arg):
    shape = arg.shape

    num_dims = len(shape)
    iterators = ['reduction' for _ in range(num_dims)]
    dims = ','.join(['d%s' % i for i in range(num_dims)])
    expr1 = f'({dims}) -> ({dims})'
    expr2 = f'({dims}) -> (0)'
    maps = [expr1,expr2]
    init = builder.from_elements(0, arg.dtype)

    def body(a, b):
        return a + b

    res = builder.generic(arg, init, iterators, maps, body)
    return builder.extract(res, 0)

@register_func('numpy.sqrt', numpy.sqrt)
def sqrt_impl(builder, arg):

    def body(a, b):
        return math.sqrt(a)

    return eltwise(builder, arg, body, builder.float64)

@register_func('numpy.square', numpy.square)
def quare_impl(builder, arg):

    def body(a, b):
        return a * a

    return eltwise(builder, arg, body)
