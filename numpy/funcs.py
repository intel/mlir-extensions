from  ..linalg_builder import register_func

@register_func('numpy.add')
def add_impl(builder, arg1, arg2):
    a1, a2 = builder.broadcast(arg1, arg2)
    shape = a1.shape

    num_dims = len(shape)
    iterators = ['parallel' for _ in range(num_dims)]
    dims = ','.join(['d%s' % i for i in range(num_dims)])
    expr = f'({dims}) -> ({dims})'
    maps = [expr,expr,expr]
    init = builder.init_tensor(shape, a1.dtype)

    def body(a, b, c):
        return a + b

    return builder.generic((a1,a2), init, iterators, maps, body)

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
