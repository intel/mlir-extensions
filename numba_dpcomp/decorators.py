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
Define @jit and related decorators.
"""

from .mlir.compiler import mlir_compiler_pipeline, mlir_compiler_gpu_pipeline
from .mlir.vectorize import vectorize as mlir_vectorize
from .mlir.settings import USE_MLIR

from numba.core.decorators import jit as orig_jit
from numba.core.decorators import njit as orig_njit
from numba.np.ufunc import vectorize as orig_vectorize

if USE_MLIR:
    def jit(signature_or_function=None, locals={}, cache=False,
            pipeline_class=None, boundscheck=False, **options):
        if not options.get('nopython', False):
            return orig_jit(signature_or_function=signature_or_function,
                            locals=locals,
                            cache=cache,
                            boundscheck=boundscheck,
                            **options)

        pipeline = mlir_compiler_gpu_pipeline if options.get('enable_gpu_pipeline') else mlir_compiler_pipeline
        options.pop('enable_gpu_pipeline', None)
        return orig_jit(signature_or_function=signature_or_function,
                        locals=locals,
                        cache=cache,
                        pipeline_class=pipeline,
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


def override_numba_decorators():
    if USE_MLIR:
        import numba
        numba.jit = jit
        numba.njit = njit
        numba.vectorize = vectorize
