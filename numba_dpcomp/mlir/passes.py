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

from numba.core.compiler_machinery import (FunctionPass, register_pass)
from numba.core import (types)
import numba.core.types.functions

from .settings import PRINT_IR, DEBUG_TYPE
from . import func_registry
from .. import mlir_compiler

_mlir_last_compiled_func = None
_mlir_active_module = None

def _init_compiler():
    settings = {}
    settings['debug_type'] = DEBUG_TYPE
    mlir_compiler.init_compiler(settings)

_init_compiler()

class MlirBackendBase(FunctionPass):

    def __init__(self, push_func_stack):
        self._push_func_stack = push_func_stack
        self._get_func_name = func_registry.get_func_name
        FunctionPass.__init__(self)

    def run_pass(self, state):
        if self._push_func_stack:
            func_registry.push_active_funcs_stack()
            try:
                res = self.run_pass_impl(state)
            finally:
                func_registry.pop_active_funcs_stack()
            return res
        else:
            return self.run_pass_impl(state)

    def _resolve_func_name(self, obj):
        name, func = self._resolve_func_name_impl(obj)
        if not (name is None or func is None):
            func_registry.add_active_funcs(name, func)
        return name

    def _resolve_func_name_impl(self, obj):
        if isinstance(obj, types.Function):
            func = obj.typing_key
            return (self._get_func_name(func), None)
        if isinstance(obj, types.BoundFunction):
            return (str(obj.typing_key), None)
        if isinstance(obj, numba.core.types.functions.Dispatcher):
            func = obj.dispatcher.py_func
            return (func.__module__ + "." + func.__qualname__, func)
        return (None, None)

    def _get_func_context(self, state):
        mangler = state.targetctx.mangler
        mangler = default_mangler if mangler is None else mangler
        unique_name = state.func_ir.func_id.unique_name
        modname = state.func_ir.func_id.func.__module__
        from numba.core.funcdesc import qualifying_prefix
        qualprefix = qualifying_prefix(modname, unique_name)
        fn_name = mangler(qualprefix, state.args)

        from numba.np.ufunc.parallel import get_thread_count

        ctx = {}
        ctx['compiler_settings'] = {'verify': True, 'pass_statistics': False, 'pass_timings': False, 'ir_printing': PRINT_IR}
        ctx['typemap'] = lambda op: state.typemap[op.name]
        ctx['fnargs'] = lambda: state.args
        ctx['restype'] = lambda: state.return_type
        ctx['fnname'] = lambda: fn_name
        ctx['resolve_func'] = self._resolve_func_name
        ctx['fastmath'] = lambda: state.targetctx.fastmath
        ctx['max_concurrency'] = lambda: get_thread_count() if state.flags.auto_parallel.enabled else 0
        return ctx

@register_pass(mutates_CFG=True, analysis_only=False)
class MlirDumpPlier(MlirBackendBase):

    _name = "mlir_dump_plier"

    def __init__(self):
        MlirBackendBase.__init__(self, push_func_stack=True)

    def run_pass(self, state):
        module = mlir_compiler.create_module()
        ctx = self._get_func_context(state)
        mlir_compiler.lower_function(ctx, module, state.func_ir)
        print(mlir_compiler.module_str(module))
        return True

def get_mlir_func():
    global _mlir_last_compiled_func
    return _mlir_last_compiled_func

@register_pass(mutates_CFG=True, analysis_only=False)
class MlirBackend(MlirBackendBase):

    _name = "mlir_backend"

    def __init__(self):
        MlirBackendBase.__init__(self, push_func_stack=True)

    def run_pass_impl(self, state):
        import numba_dpcomp.mlir_compiler as mlir_compiler
        global _mlir_active_module
        old_module = _mlir_active_module

        try:
            module = mlir_compiler.create_module()
            _mlir_active_module = module
            global _mlir_last_compiled_func
            ctx = self._get_func_context(state)
            _mlir_last_compiled_func = mlir_compiler.lower_function(ctx, module, state.func_ir)
            mod_ir = mlir_compiler.compile_module(ctx, module)
        finally:
            _mlir_active_module = old_module
        state.metadata['mlir_blob'] = mod_ir
        return True

@register_pass(mutates_CFG=True, analysis_only=False)
class MlirBackendInner(MlirBackendBase):

    _name = "mlir_backend_inner"

    def __init__(self):
        MlirBackendBase.__init__(self, push_func_stack=False)

    def run_pass_impl(self, state):
        global _mlir_active_module
        module = _mlir_active_module
        assert not module is None
        global _mlir_last_compiled_func
        ctx = self._get_func_context(state)
        _mlir_last_compiled_func = mlir_compiler.lower_function(ctx, module, state.func_ir)
        from numba.core.compiler import compile_result
        state.cr = compile_result()
        return True
