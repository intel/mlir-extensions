# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from numba.core import types
from numba.core.compiler import DEFAULT_FLAGS, compile_result
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.funcdesc import qualifying_prefix
from numba.np.ufunc.parallel import get_thread_count
import numba.core.types.functions
from contextlib import contextmanager

from .settings import DUMP_IR, OPT_LEVEL, DUMP_DIAGNOSTICS
from . import func_registry
from .. import mlir_compiler
from .compiler_context import global_compiler_context


_print_before = []
_print_after = []
_print_buffer = ""


def write_print_buffer(text):
    global _print_buffer
    _print_buffer += text


def get_print_buffer():
    global _print_buffer
    if len(_print_buffer) == 0:
        raise ValueError("Pass print buffer is empty")

    return _print_buffer


def is_print_buffer_empty():
    global _print_buffer
    return len(_print_buffer) == 0


@contextmanager
def print_pass_ir(print_before, print_after):
    global _print_before
    global _print_after
    global _print_buffer
    old_before = _print_before
    old_after = _print_after
    old_buffer = _print_buffer
    _print_before = print_before
    _print_after = print_after
    _print_buffer = ""
    try:
        yield (print_before, print_after)
    finally:
        _print_before = old_before
        _print_after = old_after
        _print_buffer = old_buffer


_mlir_last_compiled_func = None
_mlir_active_module = None


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
        name, func, flags = self._resolve_func_impl(obj)
        if not (name is None or func is None):
            func_registry.add_active_funcs(name, func, flags)
        return name

    def _resolve_func_impl(self, obj):
        if isinstance(obj, types.Function):
            func = obj.typing_key
            return (self._get_func_name(func), None, DEFAULT_FLAGS)
        if isinstance(obj, types.BoundFunction):
            return (str(obj.typing_key), None, DEFAULT_FLAGS)
        if isinstance(obj, numba.core.types.functions.Dispatcher):
            flags = DEFAULT_FLAGS
            func = obj.dispatcher.py_func
            inline_type = obj.dispatcher.targetoptions.get("inline", None)
            if inline_type is not None:
                flags.inline._inline = inline_type

            parallel_type = obj.dispatcher.targetoptions.get("parallel", None)
            if parallel_type is not None:
                flags.auto_parallel = parallel_type

            fastmath_type = obj.dispatcher.targetoptions.get("fastmath", None)
            if fastmath_type is not None:
                flags.fastmath = fastmath_type

            return (func.__module__ + "." + func.__qualname__, func, flags)
        return (None, None, None)

    def _get_func_context(self, state):
        mangler = state.targetctx.mangler
        mangler = default_mangler if mangler is None else mangler
        unique_name = state.func_ir.func_id.unique_name
        modname = state.func_ir.func_id.func.__module__
        qualprefix = qualifying_prefix(modname, unique_name)
        abi_tags = [state.flags.get_mangle_string()]
        fn_name = mangler(qualprefix, state.args, abi_tags=abi_tags)

        ctx = {}
        ctx["compiler_settings"] = {
            "verify": True,
            "pass_statistics": False,
            "pass_timings": False,
            "ir_printing": DUMP_IR,
            "diag_printing": DUMP_DIAGNOSTICS,
            "print_before": _print_before,
            "print_after": _print_after,
            "print_callback": write_print_buffer,
        }
        ctx["typemap"] = lambda op: state.typemap[op.name]
        ctx["fnargs"] = lambda: state.args
        ctx["restype"] = lambda: state.return_type
        ctx["fnname"] = lambda: fn_name
        ctx["resolve_func"] = self._resolve_func_name
        ctx["fastmath"] = lambda: state.targetctx.fastmath
        ctx["force_inline"] = lambda: state.flags.inline.is_always_inline
        ctx["max_concurrency"] = (
            lambda: get_thread_count() if state.flags.auto_parallel.enabled else 0
        )
        ctx["opt_level"] = lambda: OPT_LEVEL
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
        self.enable_gpu_pipeline = False

    def run_pass_impl(self, state):
        global _mlir_active_module
        old_module = _mlir_active_module

        try:
            mod_settings = {"enable_gpu_pipeline": self.enable_gpu_pipeline}
            module = mlir_compiler.create_module(mod_settings)
            _mlir_active_module = module
            global _mlir_last_compiled_func
            ctx = self._get_func_context(state)
            _mlir_last_compiled_func = mlir_compiler.lower_function(
                ctx, module, state.func_ir
            )

            # TODO: properly handle returned module ownership
            compiled_mod = mlir_compiler.compile_module(
                global_compiler_context, ctx, module
            )
            func_name = ctx["fnname"]()
            func_ptr = mlir_compiler.get_function_pointer(
                global_compiler_context, compiled_mod, func_name
            )
        finally:
            _mlir_active_module = old_module
        state.metadata["mlir_func_ptr"] = func_ptr
        state.metadata["mlir_func_name"] = func_name
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class MlirBackendGPU(MlirBackend):
    def __init__(self):
        MlirBackend.__init__(self)
        self.enable_gpu_pipeline = True


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
        _mlir_last_compiled_func = mlir_compiler.lower_function(
            ctx, module, state.func_ir
        )
        state.cr = compile_result()
        return True
