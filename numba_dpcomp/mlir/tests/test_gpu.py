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

from numpy.testing import assert_equal
import numpy as np

from numba_dpcomp.mlir.settings import _readenv
from numba_dpcomp.mlir.kernel_impl import kernel, get_global_id
from numba_dpcomp.mlir.kernel_sim import kernel as kernel_sim
from numba_dpcomp.mlir.passes import print_pass_ir, get_print_buffer

GPU_TESTS_ENABLED = _readenv('DPCOMP_ENABLE_GPU_TESTS', int, 0)

def require_gpu(func):
    return pytest.mark.skipif(not GPU_TESTS_ENABLED, reason='GPU tests disabled')(func)

@require_gpu
def test_simple():
    def func(a, b, c):
        i = get_global_id(0)
        j = get_global_id(1)
        k = get_global_id(2)
        c[i, j, k] = a[i, j, k] + b[i, j, k]

    sim_func = kernel_sim(func)
    gpu_func = kernel(func)

    a = np.array([[[1,2,3],[4,5,6]]], np.float32)
    b = np.array([[[7,8,9],[10,11,12]]], np.float32)

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape](a, b, sim_res)

    gpu_res = np.zeros(a.shape, a.dtype)

    with print_pass_ir([],['ConvertParallelLoopToGpu']):
        gpu_func[a.shape](a, b, gpu_res)
        ir = get_print_buffer()
        assert ir.count('gpu.launch blocks') == 1, ir

    assert_equal(gpu_res, sim_res)

@require_gpu
def test_inner_loop():
    def func(a, b, c):
        i = get_global_id(0)
        res = 0.0
        for j in range(a[i]):
            res = res + b[j]
        c[i] = res

    sim_func = kernel_sim(func)
    gpu_func = kernel(func)

    a = np.array([1,2,3,4], np.int32)
    b = np.array([5,6,7,8,9], np.float32)

    sim_res = np.zeros(a.shape, b.dtype)
    sim_func[a.shape](a, b, sim_res)

    gpu_res = np.zeros(a.shape, b.dtype)

    with print_pass_ir([],['ConvertParallelLoopToGpu']):
        gpu_func[a.shape](a, b, gpu_res)
        ir = get_print_buffer()
        assert ir.count('gpu.launch blocks') == 1, ir

    assert_equal(gpu_res, sim_res)
