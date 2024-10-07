// RUN: imex-opt --split-input-file --convert-ndarray-to-linalg %s -verify-diagnostics -o -| FileCheck %s

// -----
func.func @test_subview(%arg0: !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.subview %arg0[%c0][%c3][%c3] : !ndarray.ndarray<?xi64> to !ndarray.ndarray<?xi64>
    return %0 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: @test_subview
// CHECK-NEXT: [[V:%.*]] = bufferization.to_tensor %arg0 restrict : memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[C1:%.*]] = arith.constant
// CHECK-NEXT: [[V0:%.*]] = bufferization.to_memref [[V]] : memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: [[S0:%.*]] = memref.subview [[V0]][[[C0]]] [[[C1]]] [[[C1]]] : memref<?xi64, strided<[?], offset: ?>> to memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: [[V1:%.*]] = bufferization.to_tensor [[S0]] restrict writable : memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: [[V2:%.*]] = bufferization.to_memref
// CHECK-NEXT: return [[V2]] : memref<?xi64, strided<[?], offset: ?>>

// -----
func.func @test_static_mr_2_tnsr_2_static_mr(%arg0: memref<55xi32, strided<[1], offset: 2>>) -> !ndarray.ndarray<3xi32> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %nda = ndarray.from_memref %arg0 : memref<55xi32, strided<[1], offset: 2>> -> !ndarray.ndarray<55xi32>
    %0 = ndarray.subview %nda[%c0][%c3][%c3] : !ndarray.ndarray<55xi32> to !ndarray.ndarray<3xi32>
    return %0 : !ndarray.ndarray<3xi32>
}
// CHECK-LABEL: @test_static_mr_2_tnsr_2_static_mr
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[C1:%.*]] = arith.constant
// CHECK-NEXT: [[V:%.*]] = bufferization.to_tensor %arg0 restrict : memref<55xi32, strided<[1], offset: 2>>
// CHECK-NEXT: [[V0:%.*]] = bufferization.to_memref [[V]] : memref<55xi32, strided<[1], offset: 2>>
// CHECK-NEXT: [[S0:%.*]] = memref.subview [[V0]][[[C0]]] [[[C1]]] [[[C1]]] : memref<55xi32, strided<[1], offset: 2>> to memref<?xi32, strided<[?], offset: ?>>
// CHECK-NEXT: [[V1:%.*]] = bufferization.to_tensor [[S0]] restrict writable : memref<?xi32, strided<[?], offset: ?>>
// CHECK-NEXT: [[V2:%.*]] = bufferization.to_memref
// CHECK-NEXT: [[V3:%.*]] = memref.cast [[V2]]
// CHECK-NEXT: return [[V3]] : memref<3xi32, strided<[?], offset: ?>>

// -----
func.func @test_extract_slice(%arg0: !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.extract_slice %arg0[%c0][%c3][%c3] : !ndarray.ndarray<?xi64> to !ndarray.ndarray<?xi64>
    return %0 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: @test_extract_slice
// CHECK-NEXT: [[V:%.*]] = bufferization.to_tensor %arg0 restrict : memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[C1:%.*]] = arith.constant
// CHECK-NEXT: [[V0:%.*]] = tensor.extract_slice [[V]][%c0] [%c3] [%c3] : tensor<?xi64> to tensor<?xi64>
// CHECK-NEXT: [[V1:%.*]] = bufferization.to_memref [[V0:%.*]]
// CHECK-NEXT: return [[V1]] : memref<?xi64, strided<[?], offset: ?>>

// -----
func.func @test_linspace(%arg0: i64, %arg1: i64, %arg2: index) -> !ndarray.ndarray<?xindex> {
    %0 = ndarray.linspace %arg0 %arg1 %arg2 false : (i64, i64, index) -> !ndarray.ndarray<?xindex>
    return %0 : !ndarray.ndarray<?xindex>
}
// CHECK-LABEL: @test_linspace
// CHECK: arith.sitofp
// CHECK: arith.sitofp
// CHECK: arith.index_cast
// CHECK: arith.subf
// CHECK: arith.divf
// CHECK: tensor.empty
// CHECK: [[V0:%.*]] = linalg.generic{{.*}}["parallel"]
// CHECK: [[V2:%.*]] = bufferization.to_memref [[V0]]
// CHECK-NEXT: return [[V2]] : memref<?xindex, strided<[?], offset: ?>>

func.func @test_create(%arg0: index, %arg1: index, %arg2: i64) -> !ndarray.ndarray<?x?xi64> {
    %0 = ndarray.create %arg0, %arg1 value %arg2 {dtype = 2 : i8} : (index, index, i64) -> !ndarray.ndarray<?x?xi64>
    return %0 : !ndarray.ndarray<?x?xi64>
}
// CHECK-LABEL: @test_create
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-NEXT: bb
// CHECK-NEXT: linalg.yield
// CHECK-NEXT: } -> tensor<?x?xi64>
// CHECK: [[V3:%.*]] = bufferization.to_memref
// CHECK-NEXT: return [[V3]] : memref<?x?xi64, strided<[?, ?], offset: ?>>

// -----
func.func @test_reshape(%arg0: index) -> !ndarray.ndarray<?x?xi64> {
    %0 = ndarray.create %arg0 {dtype = 2 : i8} : (index) -> !ndarray.ndarray<?xi64>
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %1 = "ndarray.reshape"(%0, %c0, %c3) : (!ndarray.ndarray<?xi64>, index, index) -> !ndarray.ndarray<?x?xi64>
    return %1 : !ndarray.ndarray<?x?xi64>
}
// CHECK-LABEL: @test_reshape
// CHECK: tensor.empty
// CHECK: tensor.from_elements
// CHECK: tensor.reshape
// CHECK-SAME: -> tensor<?x?xi64>

// -----
func.func @test_reshape2(%arg0: index) -> !ndarray.ndarray<?x?xi64> {
    %0 = ndarray.create %arg0 {dtype = 2 : i8} : (index) -> !ndarray.ndarray<?xi64>
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %1 = "ndarray.reshape"(%0, %c0, %c3) {copy = 1 : i1} : (!ndarray.ndarray<?xi64>, index, index) -> !ndarray.ndarray<?x?xi64>
    return %1 : !ndarray.ndarray<?x?xi64>
}
// CHECK-LABEL: @test_reshape2
// CHECK: tensor.empty
// CHECK: tensor.dim
// CHECK: memref.alloc
// CHECK: bufferization.to_memref
// CHECK: region.env_region "protect_copy_op"
// CHECK: memref.copy
// CHECK: tensor.from_elements
// CHECK: tensor.reshape
// CHECK-SAME: -> tensor<?x?xi64>

// -----
func.func @test_ewbin(%arg0: !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64> {
    %0 = ndarray.ewbin %arg0, %arg0 {op = 21 : i32} : (!ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64>
    return %0 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: #map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @test_ewbin(
// CHECK: arith.constant
// CHECK: tensor.dim
// CHECK: tensor.empty
// CHECK: linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
// CHECK: arith.muli
// CHECK: [[V3:%.*]] = bufferization.to_memref
// CHECK-NEXT: return [[V3]] : memref<?xi64, strided<[?], offset: ?>>

// -----
func.func @test_ewbin_bcast(%arg0: !ndarray.ndarray<?x?xi64>, %arg1: !ndarray.ndarray<i64>) -> !ndarray.ndarray<?x?xi64> {
    %0 = ndarray.ewbin %arg0, %arg1 {op = 0 : i32} : (!ndarray.ndarray<?x?xi64>, !ndarray.ndarray<i64>) -> !ndarray.ndarray<?x?xi64>
    return %0 : !ndarray.ndarray<?x?xi64>
}
// CHECK-LABEL: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #map1 = affine_map<(d0, d1) -> ()>
// CHECK-LABEL: @test_ewbin_bcast
// CHECK: arith.constant
// CHECK: tensor.dim
// CHECK: arith.constant
// CHECK: tensor.dim
// CHECK: tensor.empty
// CHECK: linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]}
// CHECK: arith.addi
// CHECK: [[V3:%.*]] = bufferization.to_memref
// CHECK: return [[V3]] : memref<?x?xi64, strided<[?, ?], offset: ?>>

// -----
func.func @test_ewbin_3d(%arg0: !ndarray.ndarray<?x?x?xi64>) -> !ndarray.ndarray<?x?x?xi64> {
    %0 = ndarray.ewbin %arg0, %arg0 {op = 0 : i32} : (!ndarray.ndarray<?x?x?xi64>, !ndarray.ndarray<?x?x?xi64>) -> !ndarray.ndarray<?x?x?xi64>
    return %0 : !ndarray.ndarray<?x?x?xi64>
}
// CHECK-LABEL: #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: @test_ewbin_3d
// CHECK: arith.constant
// CHECK: tensor.dim
// CHECK: arith.constant
// CHECK: tensor.dim
// CHECK: tensor.empty
// CHECK: linalg.generic{{.*}}["parallel", "parallel", "parallel"]
// CHECK: arith.addi
// CHECK: return %{{.+}} : memref<?x?x?xi64, strided<[?, ?, ?], offset: ?>>

// -----
func.func @test_reduction(%arg0: !ndarray.ndarray<?xi64>) -> i64 {
    %0 = ndarray.reduction %arg0 {op = 4 : i32} : !ndarray.ndarray<?xi64> -> !ndarray.ndarray<i64>
    %1 = builtin.unrealized_conversion_cast %0 : !ndarray.ndarray<i64> to i64
    return %1 : i64
}
// CHECK-LABEL: @test_reduction
// CHECK: [[C0:%.*]] = linalg.fill
// CHECK: linalg.generic{{.*}}["reduction"]}{{.*}}outs([[C0]]
// CHECK: return %{{.}} : i64

// -----
func.func @test_reduction_3d(%arg0: !ndarray.ndarray<?x?x?xi64>) -> i64 {
    %0 = ndarray.reduction %arg0 {op = 4 : i32} : !ndarray.ndarray<?x?x?xi64> -> !ndarray.ndarray<i64>
    %1 = builtin.unrealized_conversion_cast %0 : !ndarray.ndarray<i64> to i64
    return %1 : i64
}
// CHECK-LABEL: @test_reduction_3d
// CHECK: [[C0:%.*]] = linalg.fill
// CHECK: linalg.generic{{.*}}["reduction", "reduction", "reduction"]}{{.*}}outs([[C0]]
// CHECK: return %{{.}} : i64

// -----
func.func @test_insert_slice(%arg0: !ndarray.ndarray<?xi64>, %arg1: !ndarray.ndarray<?xi64>) {
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i3 = arith.constant 3 : index
    ndarray.insert_slice %arg1 into %arg0[%i0] [%i3] [%i1] : !ndarray.ndarray<?xi64> into !ndarray.ndarray<?xi64>
    return
}
// CHECK-LABEL: @test_insert_slice
// CHECK-NEXT: [[V:%.*]] = bufferization.to_tensor %arg0 restrict : memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: [[VV:%.*]] = bufferization.to_tensor %arg1 restrict : memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[C1:%.*]] = arith.constant
// CHECK-NEXT: [[C3:%.*]] = arith.constant
// CHECK-NEXT: [[V0:%.*]] = bufferization.to_memref [[VV]]
// CHECK-NEXT: [[V1:%.*]] = bufferization.to_memref [[V]]
// CHECK-NEXT: [[SV:%.*]] = memref.subview [[V1]][[[C0]]] [[[C3]]] [[[C1]]] : memref<?xi64, strided<[?], offset: ?>> to memref<?xi64, strided<[?], offset: ?>>
// CHECK: memref.copy [[V0]], [[SV]] : memref<?xi64, strided<[?], offset: ?>> to memref<?xi64, strided<[?], offset: ?>>

// -----
func.func @test_insert_slice_scalar(%arg0: !ndarray.ndarray<?xi64>, %arg1: !ndarray.ndarray<i64>) {
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i3 = arith.constant 3 : index
    ndarray.insert_slice %arg1 into %arg0[%i0] [%i3] [%i1] : !ndarray.ndarray<i64> into !ndarray.ndarray<?xi64>
    return
}
// CHECK-LABEL: @test_insert_slice_scalar
// CHECK-NEXT: [[V:%.*]] = bufferization.to_tensor %arg0 restrict : memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: [[VV:%.*]] = bufferization.to_tensor %arg1 restrict : memref<i64, strided<[], offset: ?>>
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[C1:%.*]] = arith.constant
// CHECK-NEXT: [[C3:%.*]] = arith.constant
// CHECK-NEXT: [[V0:%.*]] = bufferization.to_memref [[VV]]
// CHECK-NEXT: [[V1:%.*]] = bufferization.to_memref [[V]]
// CHECK-NEXT: [[SV:%.*]] = memref.subview [[V1]][[[C0]]] [[[C3]]] [[[C1]]] : memref<?xi64, strided<[?], offset: ?>> to memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins([[V0]] : memref<i64, strided<[], offset: ?>>) outs([[SV]] : memref<?xi64, strided<[?], offset: ?>>)

// -----
func.func @test_immutable_insert_slice(%arg0: !ndarray.ndarray<?xi64>, %arg1: !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64> {
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i3 = arith.constant 3 : index
    %0 = ndarray.immutable_insert_slice %arg1 into %arg0[%i0] [%i3] [%i1] : !ndarray.ndarray<?xi64> into !ndarray.ndarray<?xi64>
    return %0 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: @test_immutable_insert_slice
// CHECK-NEXT: [[A0:%.*]] = bufferization.to_tensor %arg0 restrict : memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: [[A1:%.*]] = bufferization.to_tensor %arg1 restrict : memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[C1:%.*]] = arith.constant
// CHECK-NEXT: [[C3:%.*]] = arith.constant
// CHECK-NEXT: [[V0:%.*]] = tensor.insert_slice [[A1]] into [[A0]][%c0] [%c3] [%c1] : tensor<?xi64> into tensor<?xi64>
// CHECK-NEXT: [[V1:%.*]] = bufferization.to_memref [[V0]]

// -----
func.func @test_dim(%arg0: !ndarray.ndarray<?xi64>) -> index {
    %c0 = arith.constant 0 : index
    %1 = ndarray.dim %arg0 %c0 : !ndarray.ndarray<?xi64> -> index
    return %1 : index
}
// CHECK-LABEL: func.func @test_dim
// CHECK: [[V0:%.*]] = tensor.dim
// CHECK-NEXT: return [[V0]] : index

// -----
func.func @test_load(%arg0: !ndarray.ndarray<?xi64>) -> i64 {
    %i3 = arith.constant 3 : index
    %1 = ndarray.load %arg0 [%i3]  : !ndarray.ndarray<?xi64>
    return %1 : i64
}
// CHECK-LABEL: @test_load
// CHECK-NEXT: [[V:%.*]] = bufferization.to_tensor %arg0 restrict : memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[V0:%.*]] = tensor.extract [[V]][[[C0]]] : tensor<?xi64>
// CHECK-NEXT: return [[V0]] : i64

// -----
func.func @test_to_tensor(%arg0: !ndarray.ndarray<?xi64>) -> tensor<?xi64> {
    %0 = ndarray.to_tensor %arg0 : !ndarray.ndarray<?xi64> -> tensor<?xi64>
    return %0 : tensor<?xi64>
}
// CHECK-LABEL: @test_to_tensor
// CHECK-NEXT: [[V:%.*]] = bufferization.to_tensor %arg0 restrict : memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: return [[V]] : tensor<?xi64>

// -----
func.func @test_ewbin_type_cast1(%arg0: !ndarray.ndarray<16xi1>, %arg1: !ndarray.ndarray<16xi64>) -> !ndarray.ndarray<16xi64> {
    %0 = ndarray.ewbin %arg0, %arg1 {op = 0 : i32} : (!ndarray.ndarray<16xi1>, !ndarray.ndarray<16xi64>) -> !ndarray.ndarray<16xi64>
    return %0 : !ndarray.ndarray<16xi64>
  }
// CHECK-LABEL: @test_ewbin_type_cast1
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.extsi
// CHECK-NEXT: arith.addi

// -----
func.func @test_ewbin_type_cast2(%arg0: !ndarray.ndarray<16xi64>, %arg1: !ndarray.ndarray<16xi1>) -> !ndarray.ndarray<16xi64> {
    %0 = ndarray.ewbin %arg0, %arg1 {op = 0 : i32} : (!ndarray.ndarray<16xi64>, !ndarray.ndarray<16xi1>) -> !ndarray.ndarray<16xi64>
    return %0 : !ndarray.ndarray<16xi64>
  }
// CHECK-LABEL: @test_ewbin_type_cast2
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.extsi
// CHECK-NEXT: arith.addi

// -----
func.func @test_ewbin_type_cast3(%arg0: !ndarray.ndarray<16xi1>, %arg1: !ndarray.ndarray<16xf64>) -> !ndarray.ndarray<16xf64> {
    %0 = ndarray.ewbin %arg0, %arg1 {op = 0 : i32} : (!ndarray.ndarray<16xi1>, !ndarray.ndarray<16xf64>) -> !ndarray.ndarray<16xf64>
    return %0 : !ndarray.ndarray<16xf64>
  }
// CHECK-LABEL: @test_ewbin_type_cast3
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.sitofp
// CHECK-NEXT: arith.addf

// -----
func.func @test_ewbin_type_cast4(%arg0: !ndarray.ndarray<16xf64>, %arg1: !ndarray.ndarray<16xf32>) -> !ndarray.ndarray<16xf64> {
    %0 = ndarray.ewbin %arg0, %arg1 {op = 0 : i32} : (!ndarray.ndarray<16xf64>, !ndarray.ndarray<16xf32>) -> !ndarray.ndarray<16xf64>
    return %0 : !ndarray.ndarray<16xf64>
  }
// CHECK-LABEL: @test_ewbin_type_cast4
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.extf
// CHECK-NEXT: arith.addf

// -----
func.func @test() -> (!ndarray.ndarray<16x16xf32, #region.gpu_env<device = "g">>, !ndarray.ndarray<16x16xf32, #region.gpu_env<device = "g">>) attributes {llvm.emit_c_interface} {
    %c16 = arith.constant 16 : index
    %cst = arith.constant 1.000000e+00 : f32
    %0 = region.env_region #region.gpu_env<device = "g"> -> !ndarray.ndarray<16x16xf32, #region.gpu_env<device = "g">> {
        %2 = ndarray.create %c16, %c16 value %cst : (index, index, f32) -> !ndarray.ndarray<16x16xf32, #region.gpu_env<device = "g">>
        region.env_region_yield %2 : !ndarray.ndarray<16x16xf32, #region.gpu_env<device = "g">>
    }
    %1 = region.env_region #region.gpu_env<device = "g"> -> !ndarray.ndarray<16x16xf32, #region.gpu_env<device = "g">> {
        %2 = ndarray.ewbin %0, %0 {op = 0 : i32} : (!ndarray.ndarray<16x16xf32, #region.gpu_env<device = "g">>, !ndarray.ndarray<16x16xf32, #region.gpu_env<device = "g">>) -> !ndarray.ndarray<16x16xf32, #region.gpu_env<device = "g">>
        region.env_region_yield %2 : !ndarray.ndarray<16x16xf32, #region.gpu_env<device = "g">>
    }
    return %0, %1 : !ndarray.ndarray<16x16xf32, #region.gpu_env<device = "g">>, !ndarray.ndarray<16x16xf32, #region.gpu_env<device = "g">>
}
// CHECK-LABEL: func.func @test
// CHECK: region.env_region #region.gpu_env<device = "g"> -> tensor<16x16xf32> {
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK: linalg.yield
// CHECK: region.env_region_yield
// CHECK: region.env_region #region.gpu_env<device = "g"> -> tensor<16x16xf32> {
// CHECK: tensor.empty()
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: linalg.yield
// CHECK: region.env_region_yield
// CHECK: return
// CHECK-SAME: memref<16x16xf32, strided<[?, ?], offset: ?>>, memref<16x16xf32, strided<[?, ?], offset: ?>>

// -----
func.func @test_copy(%a: !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64> {
    %0 = ndarray.copy %a: !ndarray.ndarray<?xi64> -> !ndarray.ndarray<?xi64>
    %1 = ndarray.copy %0: !ndarray.ndarray<?xi64> -> !ndarray.ndarray<?xi64, #region.gpu_env<device = "XeGPU">>
    %2 = ndarray.copy %1: !ndarray.ndarray<?xi64, #region.gpu_env<device = "XeGPU">> -> !ndarray.ndarray<?xi64>
    return %0 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: func.func @test_copy
// CHECK-NEXT: bufferization.to_tensor
// CHECK-NEXT: arith.constant 0 : index
// CHECK-NEXT: tensor.dim
// CHECK-NEXT: memref.alloc
// CHECK-NEXT: bufferization.to_memref
// CHECK-NEXT: region.env_region "protect_copy_op"
// CHECK-NEXT: memref.copy
// CHECK-NEXT: }
// CHECK-NEXT: bufferization.to_tensor
// CHECK-NEXT: bufferization.to_memref
// CHECK-NEXT: arith.constant 0 : index
// CHECK-NEXT: tensor.dim
// CHECK-NEXT: memref.alloc
// CHECK-NEXT: bufferization.to_memref
// CHECK-NEXT: region.env_region "protect_copy_op"
// CHECK-NEXT: memref.copy
// CHECK-NEXT: }
// CHECK-NEXT: bufferization.to_tensor
// CHECK-NEXT: arith.constant 0 : index
// CHECK-NEXT: tensor.dim
// CHECK-NEXT: memref.alloc
// CHECK-NEXT: bufferization.to_memref
// CHECK-NEXT: region.env_region "protect_copy_op"
// CHECK-NEXT: memref.copy
// CHECK-NEXT: }
// CHECK-NEXT: bufferization.to_tensor
// CHECK-NEXT: return
// CHECK-SAME: memref<?xi64, strided<[?], offset: ?>>

// -----
func.func @test_delete(%arg0: !ndarray.ndarray<?xi64>) {
    ndarray.delete %arg0 : !ndarray.ndarray<?xi64>
    return
}
// CHECK-LABEL: @test_delete
// CHECK: memref.dealloc
// CHECK-SAME: : memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: return

// -----
func.func @test_cast_elemtype_f64f32(%arg0: !ndarray.ndarray<16xf64>) -> !ndarray.ndarray<16xf32> {
    %0 = ndarray.cast_elemtype %arg0 : !ndarray.ndarray<16xf64> to !ndarray.ndarray<16xf32>
    return %0 : !ndarray.ndarray<16xf32>
  }
// CHECK-LABEL: @test_cast_elemtype_f64f32
// CHECK-NEXT: [[V0:%.*]] = bufferization.to_tensor %arg0
// CHECK-NEXT: tensor.empty
// CHECK-NEXT: [[V1:%.*]] = linalg.generic
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.truncf
// CHECK-NEXT: linalg.yield
// CHECK-NEXT: } -> tensor<16xf32>

// -----
func.func @test_cast_elemtype_f32f64(%arg0: !ndarray.ndarray<16xf32>) -> !ndarray.ndarray<16xf64> {
    %0 = ndarray.cast_elemtype %arg0 : !ndarray.ndarray<16xf32> to !ndarray.ndarray<16xf64>
    return %0 : !ndarray.ndarray<16xf64>
  }
// CHECK-LABEL: @test_cast_elemtype_f32f64
// CHECK: arith.extf
// CHECK: } -> tensor<16xf64>

// -----
func.func @test_cast_elemtype_i32f32(%arg0: !ndarray.ndarray<16xi32>) -> !ndarray.ndarray<16xf32> {
    %0 = ndarray.cast_elemtype %arg0 : !ndarray.ndarray<16xi32> to !ndarray.ndarray<16xf32>
    return %0 : !ndarray.ndarray<16xf32>
  }
// CHECK-LABEL: @test_cast_elemtype_i32f32
// CHECK: arith.sitofp
// CHECK: } -> tensor<16xf32>

// -----
func.func @test_cast_elemtype_f32i32(%arg0: !ndarray.ndarray<16xf32>) -> !ndarray.ndarray<16xi32> {
    %0 = ndarray.cast_elemtype %arg0 : !ndarray.ndarray<16xf32> to !ndarray.ndarray<16xi32>
    return %0 : !ndarray.ndarray<16xi32>
  }
// CHECK-LABEL: @test_cast_elemtype_f32i32
// CHECK: arith.fptosi
// CHECK: } -> tensor<16xi32>

// -----
func.func @test_cast_elemtype_ui32f32(%arg0: !ndarray.ndarray<16xui32>) -> !ndarray.ndarray<16xf32> {
    %0 = ndarray.cast_elemtype %arg0 : !ndarray.ndarray<16xui32> to !ndarray.ndarray<16xf32>
    return %0 : !ndarray.ndarray<16xf32>
  }
// CHECK-LABEL: @test_cast_elemtype_ui32f32
// CHECK: arith.uitofp
// CHECK: } -> tensor<16xf32>

// -----
func.func @test_cast_elemtype_noop(%arg0: !ndarray.ndarray<16xi32>) -> !ndarray.ndarray<16xi32> {
    %0 = ndarray.cast_elemtype %arg0 : !ndarray.ndarray<16xi32> to !ndarray.ndarray<16xi32>
    return %0 : !ndarray.ndarray<16xi32>
  }
// CHECK-LABEL: @test_cast_elemtype_noop
// CHECK: return %arg0

// -----
func.func @test_cast_elemtype_copy(%arg0: !ndarray.ndarray<16xi32>) -> !ndarray.ndarray<16xi32> {
    %0 = ndarray.cast_elemtype %arg0 {copy = 1 : i1} : !ndarray.ndarray<16xi32> to !ndarray.ndarray<16xi32>
    return %0 : !ndarray.ndarray<16xi32>
  }
// CHECK-LABEL: @test_cast_elemtype_copy
// CHECK: bufferization.to_memref
// CHECK: region.env_region "protect_copy_op"
// CHECK-NEXT: memref.copy
// CHECK-NEXT: }

// -----
func.func @test_from_memref(%arg0: memref<5xi32, strided<[?], offset: ?>>) -> !ndarray.ndarray<5xi32> {
    %0 = ndarray.from_memref %arg0 : memref<5xi32, strided<[?], offset: ?>> -> !ndarray.ndarray<5xi32>
    return %0 : !ndarray.ndarray<5xi32>
}
// CHECK-LABEL: @test_from_memref
// CHECK: [[V0:%.*]] = bufferization.to_tensor
// CHECK: [[V1:%.*]] = bufferization.to_memref [[V0]]
// CHECK-NEXT: return [[V1]] : memref<5xi32, strided<[?], offset: ?>>

// -----
func.func @test_permute_dims(%arg0: !ndarray.ndarray<5x3x2xi32>) -> !ndarray.ndarray<2x3x5xi32> {
    %0 = ndarray.permute_dims %arg0 [2, 1, 0] : !ndarray.ndarray<5x3x2xi32> -> !ndarray.ndarray<2x3x5xi32>
    return %0 : !ndarray.ndarray<2x3x5xi32>
}
// CHECK-LABEL: @test_permute_dims
// CHECK: [[V0:%.*]] = bufferization.to_tensor
// CHECK: [[V1:%.*]] = bufferization.to_memref [[V0]]
// CHECK: [[V2:%.*]] = memref.transpose [[V1]] (d0, d1, d2) -> (d2, d1, d0)
// CHECK-NEXT: return [[V2]] : memref<2x3x5xi32, strided<[?, ?, ?], offset: ?>>
