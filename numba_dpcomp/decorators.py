"""
Define @jit and related decorators.
"""

from .compiler import mlir_compiler


from numba.core.decorators import jit as orig_jit


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
