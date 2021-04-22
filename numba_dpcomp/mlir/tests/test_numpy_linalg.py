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

import pytest
import numpy
from numba_dpcomp import njit

def vvsort(val, vec, size):
    for i in range(size):
        imax = i
        for j in range(i + 1, size):
            if numpy.abs(val[imax]) < numpy.abs(val[j]):
                imax = j

        temp = val[i]
        val[i] = val[imax]
        val[imax] = temp

        for k in range(size):
            temp = vec[k, i]
            vec[k, i] = vec[k, imax]
            vec[k, imax] = temp


@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
@pytest.mark.parametrize("size",
                         [2, 4, 8, 16, 300])
def test_eig_arange(type, size):
    a = numpy.arange(size * size, dtype=type).reshape((size, size))
    symm_orig = numpy.tril(a) + numpy.tril(a, -1).T + numpy.diag(numpy.full((size,), size * size, dtype=type))
    symm = symm_orig.copy()
    dpnp_symm_orig = symm_orig.copy()
    dpnp_symm = symm_orig.copy()

    def py_func_val(s):
        return numpy.linalg.eig(s)[0]

    def py_func_vec(s):
        return numpy.linalg.eig(s)[1]

    jit_func_val = njit(py_func_val)
    jit_func_vec = njit(py_func_vec)

    dpnp_val, dpnp_vec = (jit_func_val(dpnp_symm), jit_func_vec(dpnp_symm))
    np_val, np_vec = (py_func_val(symm), py_func_vec(symm))

    # DPNP sort val/vec by abs value
    vvsort(dpnp_val, dpnp_vec, size)

    # NP sort val/vec by abs value
    vvsort(np_val, np_vec, size)

    # NP change sign of vectors
    for i in range(np_vec.shape[1]):
        if np_vec[0, i] * dpnp_vec[0, i] < 0:
            np_vec[:, i] = -np_vec[:, i]

    numpy.testing.assert_array_equal(symm_orig, symm)
    numpy.testing.assert_array_equal(dpnp_symm_orig, dpnp_symm)

    assert (dpnp_val.dtype == np_val.dtype)
    assert (dpnp_vec.dtype == np_vec.dtype)
    assert (dpnp_val.shape == np_val.shape)
    assert (dpnp_vec.shape == np_vec.shape)

    numpy.testing.assert_allclose(dpnp_val, np_val, rtol=1e-05, atol=1e-05)
    numpy.testing.assert_allclose(dpnp_vec, np_vec, rtol=1e-05, atol=1e-05)

