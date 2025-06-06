// RUN: imex-opt --split-input-file --convert-ndarray-to-linalg %s -verify-diagnostics -o -| FileCheck %s

// -----
func.func @test_linspace(%arg0: i64, %arg1: i64, %arg2: index) -> tensor<?xindex> {
    %0 = ndarray.linspace %arg0 %arg1 %arg2 false : (i64, i64, index) -> tensor<?xindex>
    return %0 : tensor<?xindex>
}
// CHECK-LABEL: @test_linspace
// CHECK: arith.sitofp
// CHECK: arith.sitofp
// CHECK: arith.index_cast
// CHECK: arith.subf
// CHECK: arith.divf
// CHECK: tensor.empty
// CHECK: [[V0:%.*]] = linalg.generic{{.*}}["parallel"]
// CHECK: [[v8:%.*]] = linalg.index 0 : index
// CHECK-NEXT: [[v9:%.*]] = arith.index_cast [[v8]] : index to i64
// CHECK-NEXT: [[v10:%.*]] = arith.sitofp [[v9]] : i64 to f64
// CHECK-NEXT: [[v11:%.*]] = arith.mulf
// CHECK-NEXT: [[v12:%.*]] = arith.addf [[v11]],
// CHECK-NEXT: [[v13:%.*]] = arith.fptosi [[v12]] : f64 to i64
// CHECK-NEXT: [[v14:%.*]] = arith.index_cast [[v13]] : i64 to index
// CHECK-NEXT: linalg.yield [[v14]] : index
// CHECK: return [[V0]] : tensor<?xindex>

// -----
func.func @test_reshape(%arg0: index) -> tensor<?x?xi64> {
    %0 = tensor.empty(%arg0) : tensor<?xi64>
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %1 = "ndarray.reshape"(%0, %c0, %c3) : (tensor<?xi64>, index, index) -> tensor<?x?xi64>
    return %1 : tensor<?x?xi64>
}
// CHECK-LABEL: @test_reshape
// CHECK: tensor.empty
// CHECK: tensor.from_elements
// CHECK: tensor.reshape
// CHECK-SAME: -> tensor<?x?xi64>

// -----
func.func @test_reshape2(%arg0: index) -> tensor<?x?xi64> {
    %0 = tensor.empty(%arg0) : tensor<?xi64>
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %1 = "ndarray.reshape"(%0, %c0, %c3) {copy = 1 : i1} : (tensor<?xi64>, index, index) -> tensor<?x?xi64>
    return %1 : tensor<?x?xi64>
}
// CHECK-LABEL: @test_reshape2
// CHECK: tensor.empty
// CHECK: tensor.dim
// CHECK: memref.alloc
// CHECK: bufferization.to_buffer
// CHECK: region.env_region "protect_copy_op"
// CHECK: memref.copy
// CHECK: tensor.from_elements
// CHECK: tensor.reshape
// CHECK-SAME: -> tensor<?x?xi64>

// -----
#GPUENV = #ndarray.envs<#region.gpu_env<device = "g">>
func.func @test_env() -> (tensor<16x16xf32, #GPUENV>, tensor<256xf32, #GPUENV>) attributes {llvm.emit_c_interface} {
    %0 = region.env_region #GPUENV -> tensor<16x16xf32, #GPUENV> {
        %2 = tensor.empty() : tensor<16x16xf32, #GPUENV>
        region.env_region_yield %2 : tensor<16x16xf32, #GPUENV>
    }
    %1 = region.env_region #GPUENV -> tensor<256xf32, #GPUENV> {
        %c256 = arith.constant 32 : index
        %2 = ndarray.reshape %0 %c256 : tensor<16x16xf32, #GPUENV> -> tensor<256xf32, #GPUENV>
        region.env_region_yield %2 : tensor<256xf32, #GPUENV>
    }
    return %0, %1 : tensor<16x16xf32, #GPUENV>, tensor<256xf32, #GPUENV>
}
// CHECK-LABEL: func.func @test_env
// CHECK: region.env_region #ndarray.envs<#region.gpu_env<device = "g">> -> tensor<16x16xf32, #ndarray.envs<#region.gpu_env<device = "g">>> {
// CHECK: tensor.empty
// CHECK: region.env_region_yield
// CHECK: region.env_region #ndarray.envs<#region.gpu_env<device = "g">> -> tensor<256xf32, #ndarray.envs<#region.gpu_env<device = "g">>> {
// CHECK: tensor.reshape
// CHECK: region.env_region_yield
// CHECK: return
// CHECK-SAME: tensor<16x16xf32, #ndarray.envs<#region.gpu_env<device = "g">>>, tensor<256xf32, #ndarray.envs<#region.gpu_env<device = "g">>>

// COM: FIXME
// COM: func.func @test_copy(%a: tensor<?xi64>) -> tensor<?xi64> {
// COM:     %0 = ndarray.copy %a: tensor<?xi64> -> tensor<?xi64>
// COM:     %1 = ndarray.copy %0: tensor<?xi64> -> tensor<?xi64, #region.gpu_env<device = "XeGPU">>
// COM:     %2 = ndarray.copy %1: tensor<?xi64, #region.gpu_env<device = "XeGPU">> -> tensor<?xi64>
// COM:     return %0 : tensor<?xi64>
// COM: }
// COM: CHECK-LABEL: func.func @test_copy
// COM: CHECK-NEXT: bufferization.to_tensor
// COM: CHECK-NEXT: arith.constant 0 : index
// COM: CHECK-NEXT: tensor.dim
// COM: CHECK-NEXT: memref.alloc
// COM: CHECK-NEXT: bufferization.to_buffer
// COM: CHECK-NEXT: region.env_region "protect_copy_op"
// COM: CHECK-NEXT: memref.copy
// COM: CHECK-NEXT: }
// COM: CHECK-NEXT: bufferization.to_tensor
// COM: CHECK-NEXT: bufferization.to_buffer
// COM: CHECK-NEXT: arith.constant 0 : index
// COM: CHECK-NEXT: tensor.dim
// COM: CHECK-NEXT: memref.alloc
// COM: CHECK-NEXT: bufferization.to_buffer
// COM: CHECK-NEXT: region.env_region "protect_copy_op"
// COM: CHECK-NEXT: memref.copy
// COM: CHECK-NEXT: }
// COM: CHECK-NEXT: bufferization.to_tensor
// COM: CHECK-NEXT: arith.constant 0 : index
// COM: CHECK-NEXT: tensor.dim
// COM: CHECK-NEXT: memref.alloc
// COM: CHECK-NEXT: bufferization.to_buffer
// COM: CHECK-NEXT: region.env_region "protect_copy_op"
// COM: CHECK-NEXT: memref.copy
// COM: CHECK-NEXT: }
// COM: CHECK-NEXT: bufferization.to_tensor
// COM: CHECK-NEXT: return
// COM: CHECK-SAME: memref<?xi64, strided<[?], offset: ?>>

// -----
func.func @test_copy(%a: tensor<?xi64>) -> tensor<?xi64> {
    %0 = ndarray.copy %a: tensor<?xi64> -> tensor<?xi64>
    %1 = ndarray.copy %0: tensor<?xi64> -> tensor<?xi64, #region.gpu_env<device = "XeGPU">>
    %2 = ndarray.copy %1: tensor<?xi64, #region.gpu_env<device = "XeGPU">> -> tensor<?xi64>
    return %0 : tensor<?xi64>
}
// CHECK-LABEL: func.func @test_copy(
// CHECK-SAME: [[varg0:%.*]]: tensor<?xi64>) -> tensor<?xi64> {
// CHECK-NEXT: [[vc0:%.*]] = arith.constant 0 : index
// CHECK-NEXT: [[vdim:%.*]] = tensor.dim [[varg0]], [[vc0]] : tensor<?xi64>
// CHECK-NEXT: [[valloc:%.*]] = memref.alloc([[vdim]]) {alignment = 8 : i64} : memref<?xi64>
// CHECK-NEXT: [[v0:%.*]] = bufferization.to_buffer [[varg0]] : tensor<?xi64> to memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: region.env_region "protect_copy_op" {
// CHECK-NEXT: memref.copy [[v0]], [[valloc]] : memref<?xi64, strided<[?], offset: ?>> to memref<?xi64>
// CHECK: [[v1:%.*]] = bufferization.to_tensor [[valloc]] restrict writable : memref<?xi64> to tensor<?xi64>
// CHECK-NEXT: [[vc0_0:%.*]] = arith.constant 0 : index
// CHECK-NEXT: [[vdim_1:%.*]] = tensor.dim [[v1]], [[vc0_0]] : tensor<?xi64>
// CHECK-NEXT: [[valloc_2:%.*]] = memref.alloc([[vdim_1]]) {alignment = 8 : i64} : memref<?xi64>
// CHECK-NEXT: [[v2:%.*]] = bufferization.to_buffer [[v1]] : tensor<?xi64> to memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: region.env_region "protect_copy_op" {
// CHECK-NEXT: memref.copy [[v2]], [[valloc_2]] : memref<?xi64, strided<[?], offset: ?>> to memref<?xi64>
// CHECK: [[v3:%.*]] = bufferization.to_tensor [[valloc_2]] restrict writable : memref<?xi64> to tensor<?xi64, #region.gpu_env<device = "XeGPU">>
// CHECK-NEXT: [[vc0_3:%.*]] = arith.constant 0 : index
// CHECK-NEXT: [[vdim_4:%.*]] = tensor.dim [[v3]], [[vc0_3]] : tensor<?xi64, #region.gpu_env<device = "XeGPU">>
// CHECK-NEXT: [[valloc_5:%.*]] = memref.alloc([[vdim_4]]) {alignment = 8 : i64} : memref<?xi64>
// CHECK-NEXT: [[v4:%.*]] = bufferization.to_buffer [[v3]] : tensor<?xi64, #region.gpu_env<device = "XeGPU">> to memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: region.env_region "protect_copy_op" {
// CHECK-NEXT: memref.copy [[v4]], [[valloc_5]] : memref<?xi64, strided<[?], offset: ?>> to memref<?xi64>
// CHECK: [[v5:%.*]] = bufferization.to_tensor [[valloc_5]] restrict writable : memref<?xi64> to tensor<?xi64>
// CHECK-NEXT: return [[v1]] : tensor<?xi64>

// -----
func.func @test_delete(%arg0: tensor<?xi64>) {
    ndarray.delete %arg0 : tensor<?xi64>
    return
}
// CHECK-LABEL: @test_delete
// CHECK: memref.dealloc
// CHECK-SAME: : memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: return

// -----
func.func @test_cast_elemtype_f64f32(%arg0: tensor<16xf64>) -> tensor<16xf32> {
    %0 = ndarray.cast_elemtype %arg0 : tensor<16xf64> to tensor<16xf32>
    return %0 : tensor<16xf32>
  }
// CHECK-LABEL: @test_cast_elemtype_f64f32
// CHECK-SAME: ([[V0:%.*]]: tensor<16xf64>) -> tensor<16xf32> {
// CHECK-NEXT: tensor.empty
// CHECK-NEXT: [[V1:%.*]] = linalg.generic
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.truncf
// CHECK-NEXT: linalg.yield
// CHECK-NEXT: } -> tensor<16xf32>

// -----
func.func @test_cast_elemtype_f32f64(%arg0: tensor<16xf32>) -> tensor<16xf64> {
    %0 = ndarray.cast_elemtype %arg0 : tensor<16xf32> to tensor<16xf64>
    return %0 : tensor<16xf64>
  }
// CHECK-LABEL: @test_cast_elemtype_f32f64
// CHECK: arith.extf
// CHECK: } -> tensor<16xf64>

// -----
func.func @test_cast_elemtype_i32f32(%arg0: tensor<16xi32>) -> tensor<16xf32> {
    %0 = ndarray.cast_elemtype %arg0 : tensor<16xi32> to tensor<16xf32>
    return %0 : tensor<16xf32>
  }
// CHECK-LABEL: @test_cast_elemtype_i32f32
// CHECK: arith.sitofp
// CHECK: } -> tensor<16xf32>

// -----
func.func @test_cast_elemtype_f32i32(%arg0: tensor<16xf32>) -> tensor<16xi32> {
    %0 = ndarray.cast_elemtype %arg0 : tensor<16xf32> to tensor<16xi32>
    return %0 : tensor<16xi32>
  }
// CHECK-LABEL: @test_cast_elemtype_f32i32
// CHECK: arith.fptosi
// CHECK: } -> tensor<16xi32>

// -----
func.func @test_cast_elemtype_ui32f32(%arg0: tensor<16xui32>) -> tensor<16xf32> {
    %0 = ndarray.cast_elemtype %arg0 : tensor<16xui32> to tensor<16xf32>
    return %0 : tensor<16xf32>
  }
// CHECK-LABEL: @test_cast_elemtype_ui32f32
// CHECK: arith.uitofp
// CHECK: } -> tensor<16xf32>

// -----
func.func @test_cast_elemtype_noop(%arg0: tensor<16xi32>) -> tensor<16xi32> {
    %0 = ndarray.cast_elemtype %arg0 : tensor<16xi32> to tensor<16xi32>
    return %0 : tensor<16xi32>
  }
// CHECK-LABEL: @test_cast_elemtype_noop
// CHECK: return %arg0

// -----
func.func @test_cast_elemtype_copy(%arg0: tensor<16xi32>) -> tensor<16xi32> {
    %0 = ndarray.cast_elemtype %arg0 {copy = 1 : i1} : tensor<16xi32> to tensor<16xi32>
    return %0 : tensor<16xi32>
  }
// CHECK-LABEL: @test_cast_elemtype_copy
// CHECK: bufferization.to_buffer
// CHECK: region.env_region "protect_copy_op"
// CHECK-NEXT: memref.copy
// CHECK-NEXT: }
