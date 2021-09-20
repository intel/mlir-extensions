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

"""
Define compiler pipelines.
"""

from .lowering import mlir_NoPythonBackend


from numba.core.typed_passes import (AnnotateTypes, IRLegalization)

from numba_dpcomp.mlir.passes import MlirDumpPlier, MlirBackend, MlirBackendGPU
from numba.core.compiler_machinery import PassManager
from numba.core.compiler import CompilerBase as orig_CompilerBase
from numba.core.compiler import DefaultPassBuilder as orig_DefaultPassBuilder
from numba.core.typed_passes import NoPythonBackend as orig_NoPythonBackend

def _replace_pass(passes, old_pass, new_pass):
    count = 0;
    ret = []
    for p, n in passes:
        if p == old_pass:
            count += 1
            ret.append((new_pass, str(new_pass)))
        else:
            ret.append((p, n))
    return ret, count

class mlir_PassBuilder(orig_DefaultPassBuilder):
    @staticmethod
    def define_nopython_pipeline(state, enable_gpu_pipeline=False, name='nopython'):
        pm = orig_DefaultPassBuilder.define_nopython_pipeline(state, name)

        import numba_dpcomp.mlir.settings
        if numba_dpcomp.mlir.settings.USE_MLIR:
            if enable_gpu_pipeline:
                pm.add_pass_after(MlirBackendGPU, AnnotateTypes)
            else:
                pm.add_pass_after(MlirBackend, AnnotateTypes)
            pm.passes, replaced = _replace_pass(pm.passes, orig_NoPythonBackend, mlir_NoPythonBackend)
            assert replaced == 1

        if numba_dpcomp.mlir.settings.DUMP_PLIER:
            pm.add_pass_after(MlirDumpPlier, AnnotateTypes)

        pm.finalize()
        return pm

class mlir_compiler_pipeline(orig_CompilerBase):
    def define_pipelines(self):
        # this maintains the objmode fallback behaviour
        pms = []
        if not self.state.flags.force_pyobject:
            pms.append(mlir_PassBuilder.define_nopython_pipeline(self.state))
        if self.state.status.can_fallback or self.state.flags.force_pyobject:
            pms.append(
                mlir_PassBuilder.define_objectmode_pipeline(self.state)
            )
        return pms

class mlir_compiler_gpu_pipeline(orig_CompilerBase):
    def define_pipelines(self):
        # this maintains the objmode fallback behaviour
        pms = []
        if not self.state.flags.force_pyobject:
            pms.append(mlir_PassBuilder.define_nopython_pipeline(self.state, enable_gpu_pipeline=True))
        if self.state.status.can_fallback or self.state.flags.force_pyobject:
            pms.append(
                mlir_PassBuilder.define_objectmode_pipeline(self.state)
            )
        return pms
