
_mlir_func_names = {}
    #     id(range) : 'range',
    #     id(len) : 'len',
    #     id(bool) : 'bool',
    #     id(numpy.add) : 'numpy.add'
    # }

def add_func(func, name):
    key = id(func)
    assert not key in _mlir_func_names
    _mlir_func_names[key] = name

def get_func_name(func):
    return _mlir_func_names.get(id(func), None)
