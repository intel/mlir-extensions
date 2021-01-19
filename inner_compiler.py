from numba.core.typed_passes import get_mlir_func, NopythonTypeInference, AnnotateTypes, MlirBackendInner
from numba.core.compiler import CompilerBase, DefaultPassBuilder, DEFAULT_FLAGS, compile_extra
from numba.core.compiler_machinery import PassManager
from numba.core import typing, cpu
# from numba import njit

class MlirTempCompiler(CompilerBase): # custom compiler extends from CompilerBase

    def define_pipelines(self):
        dpb = DefaultPassBuilder
        pm = PassManager('MlirTempCompiler')
        untyped_passes = dpb.define_untyped_pipeline(self.state)
        pm.passes.extend(untyped_passes.passes)

        pm.add_pass(NopythonTypeInference, "nopython frontend")
        pm.add_pass(AnnotateTypes, "annotate types")
        pm.add_pass(MlirBackendInner, "mlir backend")

        pm.finalize()
        return [pm]

def _compile_isolated(func, args, return_type=None, flags=DEFAULT_FLAGS,
                     locals={}):
    from numba.core.registry import cpu_target
    typingctx = typing.Context()
    targetctx = cpu.CPUContext(typingctx)
    # Register the contexts in case for nested @jit or @overload calls
    with cpu_target.nested_context(typingctx, targetctx):
        return compile_extra(typingctx, targetctx, func, args, return_type,
                             flags, locals, pipeline_class=MlirTempCompiler)

def compile_func(func, args):
    _compile_isolated(func, args)
    return get_mlir_func()
