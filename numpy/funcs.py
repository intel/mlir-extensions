from  ..linalg_builder import register_func, register_attr, is_literal

import numpy
import math

def is_int(t, b):
    return t == b.int8 or t == b.int16 or t == b.int32 or t == b.int64

def is_float(t, b):
    return t == b.float16 or t == b.float32 or t == b.float64

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
@register_func('operator.add')
def add_impl(builder, arg1, arg2):
    def body(a, b, c):
        return a + b

    return eltwise(builder, (arg1, arg2), body)

@register_func('numpy.subtract', numpy.subtract)
@register_func('operator.sub')
def sub_impl(builder, arg1, arg2):
    def body(a, b, c):
        return a - b

    return eltwise(builder, (arg1, arg2), body)

@register_func('numpy.multiply', numpy.multiply)
@register_func('operator.mul')
def mul_impl(builder, arg1, arg2):
    def body(a, b, c):
        return a * b

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

        orig_type = arg.dtype
        if is_int(orig_type, builder):
            res_type = builder.int64
        else:
            res_type = orig_type
        init = builder.init_tensor(res_shape, res_type, 0)

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

@register_func('numpy.dot', numpy.dot)
def dot_impl(builder, a, b):
    shape1 = a.shape
    shape2 = b.shape
    if len(shape1) == 1 and len(shape2) == 1:
        iterators = ['reduction']
        expr1 = '(d0) -> (d0)'
        expr2 = '(d0) -> (0)'
        maps = [expr1,expr1,expr2]
        init = builder.from_elements(0, a.dtype)

        def body(a, b, c):
            return a * b + c

        res = builder.generic((a,b), init, iterators, maps, body)
        return builder.extract(res, 0)
    if len(shape1) == 2 and len(shape2) == 2:
        iterators = ['parallel','parallel','reduction']
        expr1 = '(d0,d1,d2) -> (d0,d2)'
        expr2 = '(d0,d1,d2) -> (d2,d1)'
        expr3 = '(d0,d1,d2) -> (d0,d1)'
        maps = [expr1,expr2,expr3]
        res_shape = (shape1[0], shape2[1])
        init = builder.init_tensor(res_shape, a.dtype, 0)

        def body(a, b, c):
            return a * b + c

        return builder.generic((a,b), init, iterators, maps, body)

@register_attr('array.size')
def size_impl(builder, arg):
    shape = arg.shape
    res = builder.init_tensor([], builder.index, 1)
    for i in range(len(shape)):
        res = res * shape[i]
    return res

@register_attr('array.T')
def transpose_impl(builder, arg):
    shape = arg.shape
    dims = len(shape)
    if dims == 1:
        return arg
    if dims == 2:
        iterators = ['parallel','parallel']
        expr1 = '(d0,d1) -> (d0,d1)'
        expr2 = '(d0,d1) -> (d1,d0)'
        maps = [expr1,expr2]
        res_shape = (shape[1], shape[0])
        init = builder.init_tensor(res_shape, arg.dtype)

        def body(a, b):
            return a

        return builder.generic(arg, init, iterators, maps, body)

def flatten(builder, arg, src_dims_count):
    if 1 == src_dims_count:
        return arg
    dims = ','.join(['d%s' % i for i in range(src_dims_count)])
    expr = f'({dims}) -> ({dims})'
    maps = [
        expr
    ]
    return builder.reshape(arg, 1, maps)

def find_size_index(shape):
    size_index = -1
    for i in range(len(shape)):
            d = shape[i]
            if is_literal(d):
                if 1 != d:
                    return -1
            else:
                if size_index != -1:
                    return -1
                size_index = i
    return size_index

@register_func('array.reshape')
def reshape_impl(builder, arg, new_shape):
    shape = arg.shape
    src_count = len(shape)
    count = len(new_shape)
    if count == 1:
        return flatten(builder, arg, src_count)
    else:
        size_index = find_size_index(new_shape)
        if size_index < 0:
            return

        flat = flatten(builder, arg, src_count)
        init = builder.init_tensor(new_shape, arg.dtype)

        iterators = ['parallel' for _ in range(count)]
        dims1 = ','.join(['d%s' % i for i in range(count)])
        dims3 = ','.join(['d%s' % i if i == size_index else '0' for i in range(count)])
        expr1 = f'({dims1}) -> (d{size_index})'
        expr2 = f'({dims1}) -> ({dims1})'
        maps = [expr1, expr2]

        def body(a, b):
            return a

        return builder.generic(flat, init, iterators, maps, body)

