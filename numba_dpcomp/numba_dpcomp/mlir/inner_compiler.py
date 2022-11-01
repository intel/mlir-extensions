# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from numba.core.untyped_passes import ReconstructSSA
from numba.core.typed_passes import NopythonTypeInference, AnnotateTypes
from numba.core.compiler import (
    CompilerBase,
    DefaultPassBuilder,
    DEFAULT_FLAGS,
    compile_extra,
)
from numba.core.compiler_machinery import PassManager
from numba.core.registry import cpu_target
from numba.core import typing, cpu

from numba_dpcomp.mlir.passes import MlirBackendInner, get_mlir_func


class MlirTempCompiler(CompilerBase):  # custom compiler extends from CompilerBase
    def define_pipelines(self):
        dpb = DefaultPassBuilder
        pm = PassManager("MlirTempCompiler")
        untyped_passes = dpb.define_untyped_pipeline(self.state)
        pm.passes.extend(untyped_passes.passes)

        pm.add_pass(ReconstructSSA, "ssa")
        pm.add_pass(NopythonTypeInference, "nopython frontend")
        pm.add_pass(AnnotateTypes, "annotate types")
        pm.add_pass(MlirBackendInner, "mlir backend")

        pm.finalize()
        return [pm]


def _compile_isolated(func, args, return_type=None, flags=DEFAULT_FLAGS, locals={}):
    typingctx = cpu_target.typing_context
    targetctx = cpu_target.target_context
    # typingctx = typing.Context()
    # targetctx = cpu.CPUContext(typingctx)
    # with cpu_target.nested_context(typingctx, targetctx):
    return compile_extra(
        typingctx,
        targetctx,
        func,
        args,
        return_type,
        flags,
        locals,
        pipeline_class=MlirTempCompiler,
    )


def compile_func(func, args, flags=DEFAULT_FLAGS):
    _compile_isolated(func, args, flags=flags)
    return get_mlir_func()
