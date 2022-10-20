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

_mlir_func_names = {}
_active_funcs_stack = []


def add_func(func, name):
    global _mlir_func_names
    key = id(func)
    assert _mlir_func_names.get(key, name) == name
    _mlir_func_names[key] = name


def get_func_name(func):
    return _mlir_func_names.get(id(func), None)


def push_active_funcs_stack():
    global _active_funcs_stack
    _active_funcs_stack.append({})


def pop_active_funcs_stack():
    global _active_funcs_stack
    assert len(_active_funcs_stack) > 0
    _active_funcs_stack.pop()


def add_active_funcs(name, func, flags):
    global _active_funcs_stack
    assert len(_active_funcs_stack) > 0
    top = _active_funcs_stack[-1]
    top[name] = (func, flags)


def find_active_func(name):
    global _active_funcs_stack
    assert len(_active_funcs_stack) > 0
    for elem in reversed(_active_funcs_stack):
        res = elem.get(name)
        if not res is None:
            return res
    return None
