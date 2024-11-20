// RUN: imex-opt --split-input-file -lower-distruntime-to-idtr %s -verify-diagnostics -o -| FileCheck %s

// -----
module {
  func.func @test_copy_reshape(%arg0: tensor<?x?xi64>) -> tensor<?xi64> {
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c9 = arith.constant 9 : index
    %handle, %nlArray = distruntime.copy_reshape %arg0 g_shape %c3, %c3 l_offs %c1, %c1 to n_g_shape %c9 n_offs %c3 n_shape %c3 {team = 22 : i64} : (tensor<?x?xi64>, index, index, index, index, index, index, index) -> (!distruntime.asynchandle, tensor<?xi64>)
    "distruntime.wait"(%handle) : (!distruntime.asynchandle) -> ()
    return %nlArray : tensor<?xi64>
  }
}
// CHECK-LABEL: func.func private @_idtr_nprocs(i64) -> index
// CHECK-NEXT: func.func private @_idtr_prank(i64) -> index
// CHECK-NEXT: func.func private @_idtr_reduce_all_f64(memref<*xf64>, i32)
// CHECK-NEXT: func.func private @_idtr_reduce_all_f32(memref<*xf32>, i32)
// CHECK-NEXT: func.func private @_idtr_reduce_all_i64(memref<*xi64>, i32)
// CHECK-NEXT: func.func private @_idtr_reduce_all_i32(memref<*xi32>, i32)
// CHECK-NEXT: func.func private @_idtr_reduce_all_i16(memref<*xi16>, i32)
// CHECK-NEXT: func.func private @_idtr_reduce_all_i8(memref<*xi8>, i32)
// CHECK-NEXT: func.func private @_idtr_reduce_all_i1(memref<*xi1>, i32)
// CHECK-NEXT: func.func private @_idtr_copy_reshape_f64(i64, memref<*xindex>, memref<*xindex>, memref<*xf64>, memref<*xindex>, memref<*xindex>, memref<*xf64>) -> i64
// CHECK-NEXT: func.func private @_idtr_copy_reshape_f32(i64, memref<*xindex>, memref<*xindex>, memref<*xf32>, memref<*xindex>, memref<*xindex>, memref<*xf32>) -> i64
// CHECK-NEXT: func.func private @_idtr_copy_reshape_i64(i64, memref<*xindex>, memref<*xindex>, memref<*xi64>, memref<*xindex>, memref<*xindex>, memref<*xi64>) -> i64
// CHECK-NEXT: func.func private @_idtr_copy_reshape_i32(i64, memref<*xindex>, memref<*xindex>, memref<*xi32>, memref<*xindex>, memref<*xindex>, memref<*xi32>) -> i64
// CHECK-NEXT: func.func private @_idtr_copy_reshape_i16(i64, memref<*xindex>, memref<*xindex>, memref<*xi16>, memref<*xindex>, memref<*xindex>, memref<*xi16>) -> i64
// CHECK-NEXT: func.func private @_idtr_copy_reshape_i8(i64, memref<*xindex>, memref<*xindex>, memref<*xi8>, memref<*xindex>, memref<*xindex>, memref<*xi8>) -> i64
// CHECK-NEXT: func.func private @_idtr_copy_reshape_i1(i64, memref<*xindex>, memref<*xindex>, memref<*xi1>, memref<*xindex>, memref<*xindex>, memref<*xi1>) -> i64
// CHECK-NEXT: func.func private @_idtr_update_halo_f64(i64, memref<*xindex>, memref<*xindex>, memref<*xf64>, memref<*xindex>, memref<*xindex>, memref<*xf64>, memref<*xf64>, i64)
// CHECK-NEXT: func.func private @_idtr_update_halo_f32(i64, memref<*xindex>, memref<*xindex>, memref<*xf32>, memref<*xindex>, memref<*xindex>, memref<*xf32>, memref<*xf32>, i64)
// CHECK-NEXT: func.func private @_idtr_update_halo_i64(i64, memref<*xindex>, memref<*xindex>, memref<*xi64>, memref<*xindex>, memref<*xindex>, memref<*xi64>, memref<*xi64>, i64)
// CHECK-NEXT: func.func private @_idtr_update_halo_i32(i64, memref<*xindex>, memref<*xindex>, memref<*xi32>, memref<*xindex>, memref<*xindex>, memref<*xi32>, memref<*xi32>, i64)
// CHECK-NEXT: func.func private @_idtr_update_halo_i16(i64, memref<*xindex>, memref<*xindex>, memref<*xi16>, memref<*xindex>, memref<*xindex>, memref<*xi16>, memref<*xi16>, i64)
// CHECK-NEXT: func.func private @_idtr_update_halo_i8(i64, memref<*xindex>, memref<*xindex>, memref<*xi8>, memref<*xindex>, memref<*xindex>, memref<*xi8>, memref<*xi8>, i64)
// CHECK-NEXT: func.func private @_idtr_update_halo_i1(i64, memref<*xindex>, memref<*xindex>, memref<*xi1>, memref<*xindex>, memref<*xindex>, memref<*xi1>, memref<*xi1>, i64)
// CHECK-LABEL: func.func @test_copy_reshape
// CHECK: [[V0:%.*]] = tensor.empty(%c3) : tensor<?xi64>
// CHECK: [[handle:%.*]] = call @_idtr_copy_reshape_i64
// CHECK: call @_idtr_wait([[handle]]) : (i64) -> ()
// CHECK: return [[V0]] : tensor<?xi64>

// -----
module {
  func.func @test_copy_permute(%arg0: tensor<5x2xi64>) -> tensor<?x?xi64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %h, %a = distruntime.copy_permute %arg0 g_shape %c5, %c2 l_offs %c0, %c0 to n_offs %c0, %c0 n_shape %c2, %c5 axes [1, 0] {team=22 : i64} : (tensor<5x2xi64>, index, index, index, index, index, index, index, index) -> (!distruntime.asynchandle, tensor<?x?xi64>)
    "distruntime.wait"(%h) : (!distruntime.asynchandle) -> ()
    return %a : tensor<?x?xi64>
  }
}
// CHECK-LABEL: func.func @test_copy_permute
// CHECK: [[V0:%.*]] = tensor.empty(%c2, %c5) : tensor<?x?xi64>
// CHECK: [[handle:%.*]] = call @_idtr_copy_permute_i64
// CHECK: call @_idtr_wait([[handle]]) : (i64) -> ()
// CHECK: return [[V0]] : tensor<?x?xi64>
