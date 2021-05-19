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

from numba_dpcomp.mlir.passes import MlirDumpPlier, MlirBackend
from numba.core.compiler_machinery import PassManager
from numba.core.compiler import CompilerBase as orig_CompilerBase
from numba.core.compiler import DefaultPassBuilder as orig_DefaultPassBuilder


class mlir_PassBuilder(orig_DefaultPassBuilder):
    @staticmethod
    def define_nopython_pipeline(state, name='nopython'):
        """Returns an nopython mode pipeline based PassManager
        """
        # compose pipeline from untyped, typed and lowering parts
        dpb = mlir_PassBuilder
        pm = PassManager(name)
        untyped_passes = dpb.define_untyped_pipeline(state)
        pm.passes.extend(untyped_passes.passes)

        typed_passes = dpb.define_typed_pipeline(state)
        pm.passes.extend(typed_passes.passes)

        lowering_passes = dpb.define_nopython_lowering_pipeline(state)
        pm.passes.extend(lowering_passes.passes)

        pm.finalize()
        return pm

    @staticmethod
    def define_nopython_lowering_pipeline(state, name='nopython_lowering'):
        pm = PassManager(name)
        # legalise
        pm.add_pass(IRLegalization,
                    "ensure IR is legal prior to lowering")

        # lower
        pm.add_pass(mlir_NoPythonBackend, "nopython mode backend")
        pm.finalize()
        return pm

    @staticmethod
    def define_typed_pipeline(state, name="typed"):
        pm = orig_DefaultPassBuilder.define_typed_pipeline(state, name)
        import numba_dpcomp.mlir.settings
        if numba_dpcomp.mlir.settings.USE_MLIR:
            pm.add_pass_after(MlirBackend, AnnotateTypes)

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
