
_mlir_func_names = {}
_active_funcs_stack = []

def add_func(func, name):
    key = id(func)
    assert not key in _mlir_func_names
    _mlir_func_names[key] = name

def get_func_name(func):
    return _mlir_func_names.get(id(func), None)

def push_active_funcs_stack():
    global _active_funcs_stack
    _active_funcs_stack.append({})

def pop_active_funcs_stack():
    global _active_funcs_stack
    assert(len(_active_funcs_stack) > 0)
    _active_funcs_stack.pop()

def add_active_funcs(name, func):
    global _active_funcs_stack
    assert(len(_active_funcs_stack) > 0)
    top = _active_funcs_stack[-1]
    top[name] = func

def find_active_func(name):
    global _active_funcs_stack
    assert(len(_active_funcs_stack) > 0)
    for elem in reversed(_active_funcs_stack):
        res = elem.get(name)
        if not res is None:
            return res
    return None
