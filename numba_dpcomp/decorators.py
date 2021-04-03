"""
Define @jit and related decorators.
"""

from .mlir.compiler import mlir_compiler_pipeline
from .mlir.vectorize import vectorize as mlir_vectorize
from .mlir.settings import USE_MLIR

from numba.core.decorators import jit as orig_jit
from numba.core.decorators import njit as orig_njit
from numba.np.ufunc import vectorize as orig_vectorize

if USE_MLIR:
    def jit(signature_or_function=None, locals={}, cache=False,
            pipeline_class=None, boundscheck=False, **options):
        return orig_jit(signature_or_function=signature_or_function,
                        locals=locals,
                        cache=cache,
                        pipeline_class=mlir_compiler_pipeline,
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
    vectorize = mlir_vectorize
else:
    jit = orig_jit
    njit = orig_njit
    vectorize = orig_vectorize
