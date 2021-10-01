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

from numba.core.untyped_passes import ReconstructSSA
from numba.core.typed_passes import NopythonTypeInference, AnnotateTypes
from numba.core.compiler import CompilerBase, DefaultPassBuilder, DEFAULT_FLAGS, compile_extra
from numba.core.compiler_machinery import PassManager
from numba.core.registry import cpu_target
from numba.core import typing, cpu

from numba_dpcomp.mlir.passes import MlirBackendInner, get_mlir_func

class MlirTempCompiler(CompilerBase): # custom compiler extends from CompilerBase

    def define_pipelines(self):
        dpb = DefaultPassBuilder
        pm = PassManager('MlirTempCompiler')
        untyped_passes = dpb.define_untyped_pipeline(self.state)
        pm.passes.extend(untyped_passes.passes)

        pm.add_pass(ReconstructSSA, "ssa")
        pm.add_pass(NopythonTypeInference, "nopython frontend")
        pm.add_pass(AnnotateTypes, "annotate types")
        pm.add_pass(MlirBackendInner, "mlir backend")

        pm.finalize()
        return [pm]

def _compile_isolated(func, args, return_type=None, flags=DEFAULT_FLAGS,
                     locals={}):
    typingctx = cpu_target.typing_context
    targetctx = cpu_target.target_context
    # typingctx = typing.Context()
    # targetctx = cpu.CPUContext(typingctx)
    # with cpu_target.nested_context(typingctx, targetctx):
    return compile_extra(typingctx, targetctx, func, args, return_type,
                         flags, locals, pipeline_class=MlirTempCompiler)

def compile_func(func, args, flags=DEFAULT_FLAGS):
    _compile_isolated(func, args, flags=flags)
    return get_mlir_func()
