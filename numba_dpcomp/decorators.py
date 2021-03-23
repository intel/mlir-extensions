"""
Define @jit and related decorators.
"""

# TODO: refactor, split the code to several files

from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
                                     NopythonRewrites, PreParforPass,
                                     ParforPass, InlineOverloads,
                                     PreLowerStripPhis,
                                     IRLegalization,
                                     DumpParforDiagnostics,
                                     fallback_context)
from numba_dpcomp.mlir.passes import MlirDumpPlier, MlirBackend
from numba.core.compiler_machinery import PassManager
from numba.core.compiler_machinery import register_pass

from numba.core.compiler import CompilerBase as orig_CompilerBase
from numba.core.compiler import DefaultPassBuilder as orig_DefaultPassBuilder
from numba.core.decorators import jit as orig_jit


import numba_dpcomp.mlir.settings
_use_mlir = numba_dpcomp.mlir.settings.USE_MLIR

from numba.core.lowering import Lower as orig_Lower

# looks like that we don't need it but it is inherited from BaseLower too
# from numba.core.pylowering import PyLower as orig_PyLower

from numba.core import typing, types, funcdesc, config, removerefctpass


class mlir_lower(orig_Lower):
    def lower(self):
        # Emit the Env into the module
        self.emit_environment_object()
        if self.generator_info is None:
            self.genlower = None
            self.lower_normal_function(self.fndesc)
        else:
            self.genlower = self.GeneratorLower(self)
            self.gentype = self.genlower.gentype

            self.genlower.lower_init_func(self)
            self.genlower.lower_next_func(self)
            if self.gentype.has_finalizer:
                self.genlower.lower_finalize_func(self)

        if config.DUMP_LLVM:
            print(("LLVM DUMP %s" % self.fndesc).center(80, '-'))
            if config.HIGHLIGHT_DUMPS:
                try:
                    from pygments import highlight
                    from pygments.lexers import LlvmLexer as lexer
                    from pygments.formatters import Terminal256Formatter
                    from numba.misc.dump_style import by_colorscheme
                    print(highlight(self.module.__repr__(), lexer(),
                                    Terminal256Formatter(
                                        style=by_colorscheme())))
                except ImportError:
                    msg = "Please install pygments to see highlighted dumps"
                    raise ValueError(msg)
            else:
                print(self.module)
            print('=' * 80)

        # Special optimization to remove NRT on functions that do not need it.
        if self.context.enable_nrt and self.generator_info is None:
            removerefctpass.remove_unnecessary_nrt_usage(self.function,
                                                         context=self.context,
                                                         fndesc=self.fndesc)

        # Run target specific post lowering transformation
        self.context.post_lowering(self.module, self.library)

        if not _use_mlir:
            # Materialize LLVM Module
            self.library.add_ir_module(self.module)

    def lower_normal_function(self, fndesc):
        """
        Lower non-generator *fndesc*.
        """
        if _use_mlir:
            mod_ir = self.mlir_blob
            import llvmlite.binding as llvm
            mod = llvm.parse_bitcode(mod_ir)

        self.setup_function(fndesc)

        if _use_mlir:
            self.library.add_llvm_module(mod);
        else:
            # Init argument values
            self.extract_function_arguments()
            entry_block_tail = self.lower_function_body()

            # Close tail of entry block
            self.builder.position_at_end(entry_block_tail)
            self.builder.branch(self.blkmap[self.firstblk])


from numba.core.typed_passes import NativeLowering as orig_NativeLowering

@register_pass(mutates_CFG=True, analysis_only=False)
class mlir_NativeLowering(orig_NativeLowering):
    def __init__(self):
        orig_NativeLowering.__init__(self)

    def run_pass(self, state):
        targetctx = state.targetctx
        library = state.library
        interp = state.func_ir  # why is it called this?!
        typemap = state.typemap
        restype = state.return_type
        calltypes = state.calltypes
        flags = state.flags
        metadata = state.metadata

        msg = ("Function %s failed at nopython "
               "mode lowering" % (state.func_id.func_name,))
        with fallback_context(state, msg):
            # Lowering
            fndesc = \
                funcdesc.PythonFunctionDescriptor.from_specialized_function(
                    interp, typemap, restype, calltypes,
                    mangler=targetctx.mangler, inline=flags.forceinline,
                    noalias=flags.noalias)

            with targetctx.push_code_library(library):
                lower = mlir_lower(targetctx, library, fndesc, interp,
                                   metadata=metadata)
                if _use_mlir:
                    setattr(lower, 'mlir_blob', state.mlir_blob)

                lower.lower()
                if not flags.no_cpython_wrapper:
                    lower.create_cpython_wrapper(flags.release_gil)

                if not flags.no_cfunc_wrapper:
                    # skip cfunc wrapper generation if unsupported
                    # argument or return types are used
                    for t in state.args:
                        if isinstance(t, (types.Omitted, types.Generator)):
                            break
                    else:
                        if isinstance(restype,
                                      (types.Optional, types.Generator)):
                            pass
                        else:
                            lower.create_cfunc_wrapper()

                env = lower.env
                call_helper = lower.call_helper
                del lower

            from numba.core.compiler import _LowerResult  # TODO: move this
            if flags.no_compile:
                state['cr'] = _LowerResult(fndesc, call_helper,
                                           cfunc=None, env=env)
            else:
                # Prepare for execution
                cfunc = targetctx.get_executable(library, fndesc, env)
                # Insert native function for use by other jitted-functions.
                # We also register its library to allow for inlining.
                targetctx.insert_user_function(cfunc, fndesc, [library])
                state['cr'] = _LowerResult(fndesc, call_helper,
                                           cfunc=cfunc, env=env)
        return True


from numba.core.typed_passes import NoPythonBackend as orig_NoPythonBackend

@register_pass(mutates_CFG=True, analysis_only=False)
class mlir_NoPythonBackend(orig_NoPythonBackend):
    def __init__(self):
        orig_NoPythonBackend.__init__(self)

    def run_pass(self, state):
        """
        Back-end: Generate LLVM IR from Numba IR, compile to machine code
        """
        if state.library is None:
            codegen = state.targetctx.codegen()
            state.library = codegen.create_library(state.func_id.func_qualname)
            # Enable object caching upfront, so that the library can
            # be later serialized.
            state.library.enable_object_caching()

        # TODO: Pull this out into the pipeline
        mlir_NativeLowering().run_pass(state)
        lowered = state['cr']
        signature = typing.signature(state.return_type, *state.args)

        from numba.core.compiler import compile_result
        state.cr = compile_result(
            typing_context=state.typingctx,
            target_context=state.targetctx,
            entry_point=lowered.cfunc,
            typing_error=state.status.fail_reason,
            type_annotation=state.type_annotation,
            library=state.library,
            call_helper=lowered.call_helper,
            signature=signature,
            objectmode=False,
            interpmode=False,
            lifted=state.lifted,
            fndesc=lowered.fndesc,
            environment=lowered.env,
            metadata=state.metadata,
            reload_init=state.reload_init,
        )
        return True



#########################################################################

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


def jit(signature_or_function=None, locals={}, cache=False,
        pipeline_class=None, boundscheck=False, **options):
    return orig_jit(signature_or_function=signature_or_function,
                    locals=locals,
                    cache=cache,
                    pipeline_class=mlir_compiler,
                    boundscheck=boundscheck,
                    **options)


def njit(*args, **kws):
    """
    Equivalent to jit(nopython=True)

    See documentation for jit function/decorator for full description.
    """
    if 'nopython' in kws:
        warnings.warn('nopython is set for njit and is ignored', RuntimeWarning)
    if 'forceobj' in kws:
        warnings.warn('forceobj is set for njit and is ignored', RuntimeWarning)
        del kws['forceobj']
    kws.update({'nopython': True})
    return jit(*args, **kws)
