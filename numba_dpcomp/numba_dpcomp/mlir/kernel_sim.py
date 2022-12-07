# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import namedtuple
from itertools import product
from functools import reduce
import copy
import numpy as np

try:
    from greenlet import greenlet

    _greenlet_found = True
except ImportError:
    _greenlet_found = False

from .kernel_base import KernelBase
from .kernel_impl import (
    get_global_id,
    get_local_id,
    get_group_id,
    get_global_size,
    get_local_size,
    atomic,
    atomic_add,
    atomic_sub,
    barrier,
    mem_fence,
    local,
    group,
)

_ExecutionState = namedtuple(
    "_ExecutionState",
    [
        "global_size",
        "local_size",
        "indices",
        "wg_size",
        "tasks",
        "current_task",
        "local_arrays",
        "current_local_array",
        "reduce_val",
    ],
)

_execution_state = None


def get_exec_state():
    global _execution_state
    assert _execution_state is not None
    return _execution_state


def get_global_id_proxy(index):
    return get_exec_state().indices[index]


def get_local_id_proxy(index):
    state = get_exec_state()
    return state.indices[index] % state.local_size[index]


def get_group_id_proxy(index):
    state = get_exec_state()
    return state.indices[index] // state.local_size[index]


def get_global_size_proxy(index):
    return get_exec_state().global_size[index]


def get_local_size_proxy(index):
    return get_exec_state().local_size[index]


class atomic_proxy:
    @staticmethod
    def add(arr, ind, val):
        new_val = arr[ind] + val
        arr[ind] = new_val
        return new_val

    @staticmethod
    def sub(arr, ind, val):
        new_val = arr[ind] - val
        arr[ind] = new_val
        return new_val


def _save_local_state(state):
    indices = copy.deepcopy(state.indices)
    current_local_array = state.current_local_array[0]
    state.current_local_array[0] = 0
    return (indices, current_local_array)


def _restore_local_state(state, saved_state):
    state.indices[:] = saved_state[0]
    state.current_local_array[0] = saved_state[1]


def _reset_local_state(state, wg_size):
    state.wg_size[0] = wg_size
    state.current_task[0] = 0
    state.local_arrays.clear()
    state.current_local_array[0] = 0


def _barrier_impl(state):
    wg_size = state.wg_size[0]
    assert wg_size > 0
    if wg_size > 1:
        assert len(state.tasks) > 0
        saved_state = _save_local_state(state)
        next_task = state.current_task[0] + 1
        if next_task >= wg_size:
            next_task = 0
        state.current_task[0] = next_task
        state.tasks[next_task].switch()
        _restore_local_state(state, saved_state)


def barrier_proxy(flags):
    state = get_exec_state()
    _barrier_impl(state)


def mem_fence_proxy(flags):
    pass  # Nothing


class local_proxy:
    @staticmethod
    def array(shape, dtype):
        state = get_exec_state()
        current = state.current_local_array[0]
        if state.current_task[0] == 0:
            arr = np.zeros(shape, dtype)
            state.local_arrays.append(arr)
        else:
            arr = state.local_arrays[current]
        state.current_local_array[0] = current + 1
        return arr


def _reduce_impl(state, value, op):
    if state.current_task[0] == 0:
        state.reduce_val[0] = value
    else:
        state.reduce_val[0] = op(state.reduce_val[0], value)
    _barrier_impl(state)
    return state.reduce_val[0]


class group_proxy:
    @staticmethod
    def reduce_add(value):
        state = get_exec_state()
        return _reduce_impl(state, value, lambda a, b: a + b)


def _setup_execution_state(global_size, local_size):
    import numba_dpcomp.mlir.kernel_impl

    global _execution_state
    assert _execution_state is None

    _execution_state = _ExecutionState(
        global_size=global_size,
        local_size=local_size,
        indices=[0] * len(global_size),
        wg_size=[None],
        tasks=[],
        current_task=[None],
        local_arrays=[],
        current_local_array=[0],
        reduce_val=[None],
    )
    return _execution_state


def _destroy_execution_state():
    global _execution_state
    _execution_state = None


_globals_to_replace = [
    ("get_global_id", get_global_id, get_global_id_proxy),
    ("get_local_id", get_local_id, get_local_id_proxy),
    ("get_group_id", get_local_id, get_group_id_proxy),
    ("get_global_size", get_global_size, get_global_size_proxy),
    ("get_local_size", get_local_size, get_local_size_proxy),
    ("atomic", atomic, atomic_proxy),
    ("atomic_add", atomic_add, atomic_proxy.add),
    ("atomic_sub", atomic_sub, atomic_proxy.sub),
    ("barrier", barrier, barrier_proxy),
    ("mem_fence", mem_fence, mem_fence_proxy),
    ("local", local, local_proxy),
    ("local_array", local.array, local_proxy.array),
    ("group", group, group_proxy),
    ("group_reduce_add", group.reduce_add, group_proxy.reduce_add),
]


def _replace_globals(src):
    old_globals = [src.get(name, None) for name, _, _ in _globals_to_replace]
    for name, _, new_val in _globals_to_replace:
        src[name] = new_val
    return old_globals


def _restore_globals(src, old_globals):
    for i, (name, _, _) in enumerate(_globals_to_replace):
        old_val = old_globals[i]
        if old_val is not None:
            src[name] = old_val


def _replace_closure(src):
    if src is None:
        return None

    old_vals = [e.cell_contents for e in src]
    for e in src:
        old_val = e.cell_contents
        for _, obj, new_val in _globals_to_replace:
            if old_val is obj:
                e.cell_contents = new_val
                break
    return old_vals


def _restore_closure(src, old_closure):
    if old_closure is None:
        return

    for i in range(len(src)):
        src[i].cell_contents = old_closure[i]


def _capture_func(func, indices, args):
    def wrapper():
        get_exec_state().indices[:] = indices
        func(*args)

    return wrapper


_barrier_ops = ["barrier"]


def _have_barrier_ops(func):
    g = func.__globals__
    return any(n in g for n in _barrier_ops)


def _execute_kernel(global_size, local_size, func, *args):
    if len(local_size) == 0:
        local_size = (1,) * len(global_size)

    saved_globals = _replace_globals(func.__globals__)
    saved_closure = _replace_closure(func.__closure__)
    state = _setup_execution_state(global_size, local_size)
    try:
        groups = tuple((g + l - 1) // l for g, l in zip(global_size, local_size))
        need_barrier = max(local_size) > 1 and _have_barrier_ops(func)
        for gid in product(*(range(g) for g in groups)):
            offset = tuple(g * l for g, l in zip(gid, local_size))
            size = tuple(
                min(g - o, l) for o, g, l in zip(offset, global_size, local_size)
            )
            count = reduce(lambda a, b: a * b, size)
            _reset_local_state(state, count)

            indices_range = (range(o, o + s) for o, s in zip(offset, size))

            if need_barrier:
                global _greenlet_found
                assert _greenlet_found, "greenlet package not installed"
                tasks = state.tasks
                assert len(tasks) == 0
                for indices in product(*indices_range):
                    tasks.append(greenlet(_capture_func(func, indices, args)))

                for t in tasks:
                    t.switch()

                tasks.clear()
            else:
                for indices in product(*indices_range):
                    state.indices[:] = indices
                    func(*args)

    finally:
        _restore_closure(func.__closure__, saved_closure)
        _restore_globals(func.__globals__, saved_globals)
        _destroy_execution_state()


class Kernel(KernelBase):
    def __init__(self, func):
        super().__init__(func)

    def __call__(self, *args, **kwargs):
        self.check_call_args(args, kwargs)

        _execute_kernel(self.global_size, self.local_size, self.py_func, *args)


def kernel(func):
    return Kernel(func)
