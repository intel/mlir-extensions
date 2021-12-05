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
from numba_dpcomp.mlir.kernel_impl import kernel, get_global_id, get_global_size, get_local_size, atomic, kernel_func
from numba_dpcomp.mlir.kernel_sim import kernel as kernel_sim
from numba_dpcomp.mlir.passes import print_pass_ir, get_print_buffer

from .utils import JitfuncCache

kernel_cache = JitfuncCache(kernel)
kernel_cached = kernel_cache.cached_decorator

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
    gpu_func = kernel_cached(func)

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
    gpu_func = kernel_cached(func)

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
def test_simple3():
    def func(a, b):
        i = get_global_id(0)
        b[i, 0] = a[i, 0]
        b[i, 1] = a[i, 1]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([[1,2],[3,4],[5,6]], np.float32)

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape[0], ()](a, sim_res)

    gpu_res = np.zeros(a.shape, a.dtype)

    with print_pass_ir([],['ConvertParallelLoopToGpu']):
        gpu_func[a.shape[0], ()](a, gpu_res)
        ir = get_print_buffer()
        assert ir.count('gpu.launch blocks') == 1, ir

    assert_equal(gpu_res, sim_res)

@require_gpu
def test_slice():
    def func(a, b):
        i = get_global_id(0)
        b1 = b[i]
        j = get_global_id(1)
        b2 = b1[j]
        k = get_global_id(2)
        b2[k] = a[i, j, k]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.arange(3*4*5).reshape((3,4,5))

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape, ()](a, sim_res)

    gpu_res = np.zeros(a.shape, a.dtype)

    with print_pass_ir([],['ConvertParallelLoopToGpu']):
        gpu_func[a.shape, ()](a, gpu_res)
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
    gpu_func = kernel_cached(func)

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

def _test_unary(func, dtype, ir_pass, ir_check):
    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([1,2,3,4,5,6,7,8,9], dtype)

    sim_res = np.zeros(a.shape, dtype)
    sim_func[a.shape, ()](a, sim_res)

    gpu_res = np.zeros(a.shape, dtype)

    with print_pass_ir([],[ir_pass]):
        gpu_func[a.shape, ()](a, gpu_res)
        ir = get_print_buffer()
        assert ir_check(ir), ir

    assert_allclose(gpu_res, sim_res, rtol=1e-5)

def _test_binary(func, dtype, ir_pass, ir_check):
    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([1,2,3,4,5,6,7,8,9], dtype)
    b = np.array([11,12,13,14,15,16,17,18,19], dtype)

    sim_res = np.zeros(a.shape, dtype)
    sim_func[a.shape, ()](a, b, sim_res)

    gpu_res = np.zeros(a.shape, dtype)

    with print_pass_ir([],[ir_pass]):
        gpu_func[a.shape, ()](a, b, gpu_res)
        ir = get_print_buffer()
        assert ir_check(ir), ir

    assert_allclose(gpu_res, sim_res, rtol=1e-5)

@require_gpu
@pytest.mark.parametrize("op", ['sqrt', 'log', 'sin', 'cos'])
def test_math_funcs_unary(op):
    f = eval(f'math.{op}')
    def func(a, b):
        i = get_global_id(0)
        b[i] = f(a[i])

    _test_unary(func, np.float32, 'GPUToSpirvPass', lambda ir: ir.count(f'OCL.{op}') == 1)

@require_gpu
@pytest.mark.parametrize("op", ['+', '-', '*', '/', '%', '**'])
def test_gpu_ops_binary(op):
    f = eval(f'lambda a, b: a {op} b')
    inner = kernel_func(f)
    def func(a, b, c):
        i = get_global_id(0)
        c[i] = inner(a[i], b[i])

    _test_binary(func, np.float32, 'ConvertParallelLoopToGpu', lambda ir: ir.count(f'gpu.launch blocks') == 1)

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
    gpu_func = kernel_cached(func)

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
    gpu_func = kernel_cached(func)

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
    gpu_func = kernel_cached(func)

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

_atomic_dtypes = ['int32', 'int64', 'float32']
_atomic_funcs = [atomic.add, atomic.sub]

def _check_atomic_ir(ir):
    return ir.count('spv.AtomicIAdd') == 1 or ir.count('spv.AtomicISub') == 1 or ir.count('spv.AtomicFAddEXT') == 1

def _test_atomic(func, dtype, ret_size):
    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([1,2,3,4,5,6,7,8,9], dtype)

    sim_res = np.zeros([ret_size], dtype)
    sim_func[a.shape, ()](a, sim_res)

    gpu_res = np.zeros([ret_size], dtype)

    with print_pass_ir([],['GPUToSpirvPass']):
        gpu_func[a.shape, ()](a, gpu_res)
        ir = get_print_buffer()
        assert _check_atomic_ir(ir), ir

    assert_equal(gpu_res, sim_res)

@require_gpu
@pytest.mark.parametrize("dtype", _atomic_dtypes)
@pytest.mark.parametrize("atomic_op", _atomic_funcs)
def test_atomics(dtype, atomic_op):
    def func(a, b):
        i = get_global_id(0)
        atomic_op(b, 0, a[i])

    _test_atomic(func, dtype, 1)

@require_gpu
@pytest.mark.xfail(reason='Only direct func calls work for now')
def test_atomics_modname():
    def func(a, b):
        i = get_global_id(0)
        atomic.add(b, 0, a[i])

    _test_atomic(func, 'int32', 1)

@require_gpu
@pytest.mark.parametrize("dtype", _atomic_dtypes)
@pytest.mark.parametrize("atomic_op", _atomic_funcs)
def test_atomics_offset(dtype, atomic_op):
    def func(a, b):
        i = get_global_id(0)
        atomic_op(b, i % 2, a[i])

    _test_atomic(func, dtype, 2)

@require_gpu
@pytest.mark.parametrize("atomic_op", _atomic_funcs)
def test_atomics_different_types1(atomic_op):
    dtype = 'int32'
    def func(a, b):
        i = get_global_id(0)
        atomic_op(b, 0, a[i] + 1)

    _test_atomic(func, dtype, 1)

@require_gpu
@pytest.mark.parametrize("atomic_op", _atomic_funcs)
def test_atomics_different_types2(atomic_op):
    dtype = 'int32'
    def func(a, b):
        i = get_global_id(0)
        atomic_op(b, 0, 1)

    _test_atomic(func, dtype, 1)

@require_gpu
@pytest.mark.parametrize("funci", [1,2])
def test_atomics_multidim(funci):
    atomic_op = atomic.add
    dtype = 'int32'
    def func1(a, b):
        i = get_global_id(0)
        j = get_global_id(1)
        atomic_op(b, (i % 2,0), a[i, j])

    def func2(a, b):
        i = get_global_id(0)
        j = get_global_id(1)
        atomic_op(b, (i % 2,j % 2), a[i, j])

    func = func1 if funci == 1 else func2

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype)

    sim_res = np.zeros((2,2), dtype)
    sim_func[a.shape, ()](a, sim_res)

    gpu_res = np.zeros((2,2), dtype)

    with print_pass_ir([],['GPUToSpirvPass']):
        gpu_func[a.shape, ()](a, gpu_res)
        ir = get_print_buffer()
        assert _check_atomic_ir(ir), ir

    assert_equal(gpu_res, sim_res)
