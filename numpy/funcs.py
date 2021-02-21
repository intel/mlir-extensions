from  ..linalg_builder import register_func, is_literal

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
@register_func('numpy.sum', numpy.sum)
def sum_impl(builder, arg, axis=None):
    if axis is None:
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
    elif  isinstance(axis, int):
        shape = arg.shape
        num_dims = len(shape)
        iterators = [('reduction' if i == axis else 'parallel') for i in range(num_dims)]
        dims1 = ','.join(['d%s' % i for i in range(num_dims)])
        dims2 = ','.join(['d%s' % i for i in range(num_dims) if i != axis])
        expr1 = f'({dims1}) -> ({dims1})'
        expr2 = f'({dims1}) -> ({dims2})'
        maps = [expr1,expr2]
        res_shape = tuple(shape[i] for i in range(len(shape)) if i != axis)

        init = builder.init_tensor(res_shape, builder.int64, 0) #TODO: type
        # val = builder.fill_tensor(init, 0)

        def body(a, b):
            return a + b

        return builder.generic(arg, init, iterators, maps, body)


@register_func('numpy.sqrt', numpy.sqrt)
def sqrt_impl(builder, arg):

    def body(a, b):
        return math.sqrt(a)

    return eltwise(builder, arg, body, builder.float64)

@register_func('numpy.square', numpy.square)
def square_impl(builder, arg):

    def body(a, b):
        return a * a

    return eltwise(builder, arg, body)

@register_func('numpy.empty', numpy.empty)
def empty_impl(builder, shape):
    # TODO: dtype
    return builder.init_tensor(shape, builder.float64)
