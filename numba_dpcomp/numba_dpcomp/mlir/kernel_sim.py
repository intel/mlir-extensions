# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    get_global_size,
    get_local_size,
    atomic,
    atomic_add,
    atomic_sub,
    barrier,
    mem_fence,
    local,
    local_array,
)

_ExecutionState = namedtuple(
    "_ExecutionState",
    ["global_size", "local_size", "indices", "wg_size", "tasks", "current_task"],
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


def barrier_proxy(flags):
    state = get_exec_state()
    wg_size = state.wg_size[0]
    assert wg_size > 0
    if wg_size > 1:
        assert len(state.tasks) > 0
        indices = copy.deepcopy(state.indices)
        next_task = state.current_task[0] + 1
        if next_task >= wg_size:
            next_task = 0
        state.current_task[0] = next_task
        state.tasks[next_task].switch()
        state.indices[:] = indices


def mem_fence_proxy(flags):
    pass  # Nothing


class local_proxy:
    @staticmethod
    def array(shape, dtype):
        arr = np.zeros(shape, dtype)
        return arr


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
    )
    return _execution_state


def _destroy_execution_state():
    global _execution_state
    _execution_state = None


_globals_to_replace = [
    ("get_global_id", get_global_id, get_global_id_proxy),
    ("get_local_id", get_local_id, get_local_id_proxy),
    ("get_global_size", get_global_size, get_global_size_proxy),
    ("get_local_size", get_local_size, get_local_size_proxy),
    ("atomic", atomic, atomic_proxy),
    ("atomic_add", atomic_add, atomic_proxy.add),
    ("atomic_sub", atomic_sub, atomic_proxy.sub),
    ("barrier", barrier, barrier_proxy),
    ("mem_fence", mem_fence, mem_fence_proxy),
    ("local", local, local_proxy),
    ("local_array", local_array, local_proxy.array),
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
            state.wg_size[0] = count
            state.current_task[0] = 0

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
