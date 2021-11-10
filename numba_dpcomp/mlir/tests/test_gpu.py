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

from numpy.testing import assert_equal, assert_allclose
import numpy as np
import math

from numba_dpcomp.mlir.settings import _readenv
from numba_dpcomp.mlir.kernel_impl import kernel, get_global_id, get_global_size, get_local_size, atomic
from numba_dpcomp.mlir.kernel_sim import kernel as kernel_sim
from numba_dpcomp.mlir.passes import print_pass_ir, get_print_buffer

GPU_TESTS_ENABLED = _readenv('DPCOMP_ENABLE_GPU_TESTS', int, 0)

def require_gpu(func):
    return pytest.mark.skipif(not GPU_TESTS_ENABLED, reason='GPU tests disabled')(func)

@require_gpu
def test_simple1():
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
    sim_func[a.shape, ()](a, b, sim_res)

    gpu_res = np.zeros(a.shape, a.dtype)

    with print_pass_ir([],['ConvertParallelLoopToGpu']):
        gpu_func[a.shape, ()](a, b, gpu_res)
        ir = get_print_buffer()
        assert ir.count('gpu.launch blocks') == 1, ir

    assert_equal(gpu_res, sim_res)

@require_gpu
def test_simple2():
    get_id = get_global_id
    def func(a, b, c):
        i = get_id(0)
        j = get_id(1)
        k = get_id(2)
        c[i, j, k] = a[i, j, k] + b[i, j, k]

    sim_func = kernel_sim(func)
    gpu_func = kernel(func)

    a = np.array([[[1,2,3],[4,5,6]]], np.float32)
    b = np.array([[[7,8,9],[10,11,12]]], np.float32)

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape, ()](a, b, sim_res)

    gpu_res = np.zeros(a.shape, a.dtype)

    with print_pass_ir([],['ConvertParallelLoopToGpu']):
        gpu_func[a.shape, ()](a, b, gpu_res)
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
    sim_func[a.shape, ()](a, b, sim_res)

    gpu_res = np.zeros(a.shape, b.dtype)

    with print_pass_ir([],['ConvertParallelLoopToGpu']):
        gpu_func[a.shape, ()](a, b, gpu_res)
        ir = get_print_buffer()
        assert ir.count('gpu.launch blocks') == 1, ir

    assert_equal(gpu_res, sim_res)

@require_gpu
@pytest.mark.parametrize("op", ['sqrt', 'log', 'sin', 'cos'])
def test_math_funcs(op):
    f = eval(f'math.{op}')
    def func(a, b):
        i = get_global_id(0)
        b[i] = f(a[i])

    sim_func = kernel_sim(func)
    gpu_func = kernel(func)

    a = np.array([1,2,3,4,5,6,7,8,9], np.float32)

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape, ()](a, sim_res)

    gpu_res = np.zeros(a.shape, a.dtype)

    with print_pass_ir([],['GPUToSpirvPass']):
        gpu_func[a.shape, ()](a, gpu_res)
        ir = get_print_buffer()
        assert ir.count(f'OCL.{op}') == 1, ir

    assert_allclose(gpu_res, sim_res, rtol=1e-5)

_test_shapes = [
    (1,),
    (7,),
    (1,1),
    (7,13),
    (1,1,1),
    (7,13,23),
]

@require_gpu
@pytest.mark.parametrize("shape", _test_shapes)
def test_get_global_id(shape):
    def func1(c):
        i = get_global_id(0)
        c[i] = i

    def func2(c):
        i = get_global_id(0)
        j = get_global_id(1)
        c[i, j] = i + j * 100

    def func3(c):
        i = get_global_id(0)
        j = get_global_id(1)
        k = get_global_id(2)
        c[i, j, k] = i + j * 100 + k * 10000

    func = [func1, func2, func3][len(shape) - 1]

    sim_func = kernel_sim(func)
    gpu_func = kernel(func)

    dtype = np.int32

    sim_res = np.zeros(shape, dtype)
    sim_func[shape, ()](sim_res)

    gpu_res = np.zeros(shape, dtype)

    with print_pass_ir([],['ConvertParallelLoopToGpu']):
        gpu_func[shape, ()](gpu_res)
        ir = get_print_buffer()
        assert ir.count('gpu.launch blocks') == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("shape", _test_shapes)
def test_get_global_size(shape):
    def func1(c):
        i = get_global_id(0)
        w = get_global_size(0)
        c[i] = w

    def func2(c):
        i = get_global_id(0)
        j = get_global_id(1)
        w = get_global_size(0)
        h = get_global_size(1)
        c[i, j] = w + h * 100

    def func3(c):
        i = get_global_id(0)
        j = get_global_id(1)
        k = get_global_id(2)
        w = get_global_size(0)
        h = get_global_size(1)
        d = get_global_size(2)
        c[i, j, k] = w + h * 100 + d * 10000

    func = [func1, func2, func3][len(shape) - 1]

    sim_func = kernel_sim(func)
    gpu_func = kernel(func)

    dtype = np.int32

    sim_res = np.zeros(shape, dtype)
    sim_func[shape, ()](sim_res)

    gpu_res = np.zeros(shape, dtype)

    with print_pass_ir([],['ConvertParallelLoopToGpu']):
        gpu_func[shape, ()](gpu_res)
        ir = get_print_buffer()
        assert ir.count('gpu.launch blocks') == 1, ir

    assert_equal(gpu_res, sim_res)

@require_gpu
@pytest.mark.parametrize("shape", _test_shapes)
@pytest.mark.parametrize("lsize", [(), (1,1,1), (2,4,8)])
def test_get_local_size(shape, lsize):
    def func1(c):
        i = get_global_id(0)
        w = get_local_size(0)
        c[i] = w

    def func2(c):
        i = get_global_id(0)
        j = get_global_id(1)
        w = get_local_size(0)
        h = get_local_size(1)
        c[i, j] = w + h * 100

    def func3(c):
        i = get_global_id(0)
        j = get_global_id(1)
        k = get_global_id(2)
        w = get_local_size(0)
        h = get_local_size(1)
        d = get_local_size(2)
        c[i, j, k] = w + h * 100 + d * 10000

    func = [func1, func2, func3][len(shape) - 1]

    sim_func = kernel_sim(func)
    gpu_func = kernel(func)

    dtype = np.int32

    if (len(lsize) > len(shape)):
        lsize =tuple(lsize[:len(shape)])

    sim_res = np.zeros(shape, dtype)
    sim_func[shape, lsize](sim_res)

    gpu_res = np.zeros(shape, dtype)

    with print_pass_ir([],['ConvertParallelLoopToGpu']):
        gpu_func[shape, lsize](gpu_res)
        ir = get_print_buffer()
        assert ir.count('gpu.launch blocks') == 1, ir

    assert_equal(gpu_res, sim_res)

@require_gpu
@pytest.mark.parametrize("dtype", ['int32', 'int64']) # TODO: float
@pytest.mark.parametrize("atomic_op", [atomic.add])
def test_atomics(dtype, atomic_op):
    def func(a, b):
        i = get_global_id(0)
        atomic_op(b, 0, a[i])

    sim_func = kernel_sim(func)
    gpu_func = kernel(func)

    a = np.array([1,2,3,4,5,6,7,8,9], dtype)

    sim_res = np.zeros([1], dtype)
    sim_func[a.shape, ()](a, sim_res)

    gpu_res = np.zeros([1], dtype)

    with print_pass_ir([],['ConvertParallelLoopToGpu']):
        gpu_func[a.shape, ()](a, gpu_res)
        ir = get_print_buffer()
        assert ir.count('gpu.launch blocks') == 1, ir

    assert_equal(gpu_res, sim_res)
