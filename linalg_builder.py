from .func_registry import add_func

class Var:
    def __init__(self, ssa_val, shape, dtype):
        self._ssa_val = ssa_val
        self._shape = shape
        self._dtype = dtype

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype



class Val:
    def __init__(self, const_val, ssa_val):
        self._const_val = const_val
        self._ssa_val = ssa_val

    def is_const(self):
        return not _const_val is None

class Builder:
    def __init__(self, context):
        self._context = context

    def broadcast(self, *args):
        return self._broadcast(self._context, args)

    def init_tensor(self, shape, dtype):
        return self._init_tensor(self._context, shape, dtype)

    def generic(self, inputs, outputs, iterators, maps, body):
        return self._generic(self._context, inputs, outputs, iterators, maps, body)

    def from_elements(self, values, dtype):
        return self._from_elements(self._context, values, dtype)

    def extract(self, value, indices):
        return self._extract(self._context, value, indices)

def compile_func(*args, **kwargs):
    import numba.mlir.inner_compiler
    return numba.mlir.inner_compiler.compile_func(*args, **kwargs)

_func_registry = {}

def register_func(name, orig_func = None):
    def _decorator(func):
        global _func_registry
        assert not name in _func_registry
        _func_registry[name] = func
        if not orig_func is None:
            add_func(orig_func, name)
        return func
    return _decorator

def lookup_func(name):
    global _func_registry
    return _func_registry.get(name)
