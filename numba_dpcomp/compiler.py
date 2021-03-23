"""
Define compiler pipelines.
"""

from .lowering import mlir_NoPythonBackend


from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
                                     NopythonRewrites, PreParforPass,
                                     ParforPass, DumpParforDiagnostics,
                                     IRLegalization,
                                     InlineOverloads, PreLowerStripPhis)

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
        pm.add_pass(DumpParforDiagnostics, "dump parfor diagnostics")
        pm.finalize()
        return pm

    @staticmethod
    def define_typed_pipeline(state, name="typed"):
        """Returns the typed part of the nopython pipeline"""
        pm = PassManager(name)
        # typing
        pm.add_pass(NopythonTypeInference, "nopython frontend")
        pm.add_pass(AnnotateTypes, "annotate types")

        import numba_dpcomp.mlir.settings

        if numba_dpcomp.mlir.settings.DUMP_PLIER:
            pm.add_pass(MlirDumpPlier, "mlir dump plier")

        if numba_dpcomp.mlir.settings.USE_MLIR:
            pm.add_pass(MlirBackend, "mlir backend")

        # strip phis
        pm.add_pass(PreLowerStripPhis, "remove phis nodes")

        # optimisation
        pm.add_pass(InlineOverloads, "inline overloaded functions")
        if state.flags.auto_parallel.enabled:
            pm.add_pass(PreParforPass, "Preprocessing for parfors")
        if not state.flags.no_rewrites:
            pm.add_pass(NopythonRewrites, "nopython rewrites")
        if state.flags.auto_parallel.enabled:
            pm.add_pass(ParforPass, "convert to parfors")

        pm.finalize()
        return pm


class mlir_compiler(orig_CompilerBase):
    def define_pipelines(self):
        # this maintains the objmode fallback behaviour
        pms = []
        if not self.state.flags.force_pyobject:
            pms.append(mlir_PassBuilder.define_nopython_pipeline(self.state))
        if self.state.status.can_fallback or self.state.flags.force_pyobject:
            pms.append(
                mlir_PassBuilder.define_objectmode_pipeline(self.state)
            )
        # if self.state.status.can_giveup:
            # pms.append(
                # mlir_PassBuilder.define_interpreted_pipeline(self.state)
            # )
        return pms
