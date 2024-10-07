// RUN: imex-opt --split-input-file -lower-distruntime-to-idtr %s -verify-diagnostics -o -| FileCheck %s

// -----
module {
    func.func @test_nprocs() -> index {
        %0 = "distruntime.team_size"() {team=22 : i64} : () -> index
        return %0 : index
    }
}
// CHECK-LABEL: func.func private @_idtr_nprocs(i64) -> index
// CHECK-LABEL: func.func private @_idtr_prank(i64) -> index
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
// CHECK-NEXT: func.func private @_idtr_wait(i64)
// CHECK-LABEL: func.func @test_nprocs() -> index {
// CHECK: [[C0:%.*]] = arith.constant
// CHECK: call @_idtr_nprocs([[C0]])

// -----
module {
    func.func @test_prank() -> index {
        %1 = "distruntime.team_member"() {team=22 : i64} : () -> index
        return %1 : index
    }
}
// CHECK-LABEL: func.func @test_prank() -> index {
// CHECK: [[C0:%.*]] = arith.constant
// CHECK: call @_idtr_prank([[C0]])

// -----
module {
    func.func @test_allreduce(%arg0: memref<i64, strided<[], offset: ?>>) {
        "distruntime.allreduce"(%arg0) {op = 4 : i32} : (memref<i64, strided<[], offset: ?>>) -> ()
        return
    }
}
// CHECK-LABEL: func.func @test_allreduce(%arg0: memref<i64, strided<[], offset: ?>>) {
// CHECK: memref.cast
// CHECK: call @_idtr_reduce_all_i64

// -----
module {
    func.func @test_wait(%arg0: !ndarray.ndarray<?xi64>) {
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 4 : index
        %c12 = arith.constant 12 : index
        %handle, %lHalo, %rHalo = "distruntime.get_halo"(%arg0, %c12, %c4, %c4, %c5) {team = 22, key = 1 : i64}: (!ndarray.ndarray<?xi64>, index, index, index, index) -> (!distruntime.asynchandle, !ndarray.ndarray<0xi64>, !ndarray.ndarray<0xi64>)
        "distruntime.wait"(%handle) : (!distruntime.asynchandle) -> ()
        return
    }
}
// CHECK-LABEL: func.func @test_wait(%arg0: !ndarray.ndarray<?xi64>) {
// CHECK: memref.alloc()
// CHECK: memref.alloc()
// CHECK: ndarray.to_tensor
// CHECK: memref.alloc()
// CHECK: memref.alloc()
// CHECK: ndarray.create
// CHECK: ndarray.to_tensor
// CHECK: ndarray.create
// CHECK: ndarray.to_tensor
// CHECK: [[handle:%.*]] = call @_idtr_update_halo_i64(
// CHECK-SAME: : (i64, memref<*xindex>, memref<*xindex>, memref<*xi64>, memref<*xindex>, memref<*xindex>, memref<*xi64>, memref<*xi64>, i64) -> i64
// CHECK: call @_idtr_wait([[handle]]) : (i64) -> ()
// CHECK: return

// -----
module {
  func.func @test_copy_reshape(%arg0: !ndarray.ndarray<?x?xi64>) -> !ndarray.ndarray<3xi64> {
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c9 = arith.constant 9 : index
    %handle, %nlArray = distruntime.copy_reshape %arg0 g_shape %c3, %c3 l_offs %c1, %c1 to n_g_shape %c9 n_offs %c3 n_shape %c3 {team = 22 : i64} : (!ndarray.ndarray<?x?xi64>, index, index, index, index, index, index, index) -> (!distruntime.asynchandle, !ndarray.ndarray<3xi64>)
    "distruntime.wait"(%handle) : (!distruntime.asynchandle) -> ()
    return %nlArray : !ndarray.ndarray<3xi64>
  }
}
// CHECK-LABEL: func.func @test_copy_reshape
// CHECK: [[V0:%.*]] = ndarray.create %c3 : (index) -> !ndarray.ndarray<3xi64>
// CHECK: [[handle:%.*]] = call @_idtr_copy_reshape_i64
// CHECK: call @_idtr_wait([[handle]]) : (i64) -> ()
// CHECK: return [[V0]] : !ndarray.ndarray<3xi64>

// -----
module {
  func.func @test_copy_permute(%arg0: !ndarray.ndarray<5x2xi64>) -> !ndarray.ndarray<2x5xi64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %h, %a = distruntime.copy_permute %arg0 g_shape %c5, %c2 l_offs %c0, %c0 to n_offs %c0, %c0 n_shape %c2, %c5 axes [1, 0] {team=22 : i64} : (!ndarray.ndarray<5x2xi64>, index, index, index, index, index, index, index, index) -> (!distruntime.asynchandle, !ndarray.ndarray<2x5xi64>)
    "distruntime.wait"(%h) : (!distruntime.asynchandle) -> ()
    return %a : !ndarray.ndarray<2x5xi64>
  }
}
// CHECK-LABEL: func.func @test_copy_permute
// CHECK: [[V0:%.*]] = ndarray.create %c2, %c5 : (index, index) -> !ndarray.ndarray<2x5xi64>
// CHECK: [[handle:%.*]] = call @_idtr_copy_permute_i64
// CHECK: call @_idtr_wait([[handle]]) : (i64) -> ()
// CHECK: return [[V0]] : !ndarray.ndarray<2x5xi64>
