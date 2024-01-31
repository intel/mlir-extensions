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
// CHECK-NEXT: func.func private @_idtr_reshape_f64(i64, memref<*xindex>, memref<*xindex>, memref<*xf64>, memref<*xindex>)
// CHECK-NEXT: func.func private @_idtr_reshape_f32(i64, memref<*xindex>, memref<*xindex>, memref<*xf32>, memref<*xindex>)
// CHECK-NEXT: func.func private @_idtr_reshape_i64(i64, memref<*xindex>, memref<*xindex>, memref<*xi64>, memref<*xindex>)
// CHECK-NEXT: func.func private @_idtr_reshape_i32(i64, memref<*xindex>, memref<*xindex>, memref<*xi32>, memref<*xindex>)
// CHECK-NEXT: func.func private @_idtr_reshape_i16(i64, memref<*xindex>, memref<*xindex>, memref<*xi16>, memref<*xindex>)
// CHECK-NEXT: func.func private @_idtr_reshape_i8(i64, memref<*xindex>, memref<*xindex>, memref<*xi8>, memref<*xindex>)
// CHECK-NEXT: func.func private @_idtr_reshape_i1(i64, memref<*xindex>, memref<*xindex>, memref<*xi1>, memref<*xindex>)
// CHECK-NEXT: func.func private @_idtr_update_halo_f64(i64, memref<*xindex>, memref<*xindex>, memref<*xf64>, memref<*xindex>, memref<*xindex>, memref<*xf64>, memref<*xf64>, i64)
// CHECK-NEXT: func.func private @_idtr_update_halo_f32(i64, memref<*xindex>, memref<*xindex>, memref<*xf32>, memref<*xindex>, memref<*xindex>, memref<*xf32>, memref<*xf32>, i64)
// CHECK-NEXT: func.func private @_idtr_update_halo_i64(i64, memref<*xindex>, memref<*xindex>, memref<*xi64>, memref<*xindex>, memref<*xindex>, memref<*xi64>, memref<*xi64>, i64)
// CHECK-NEXT: func.func private @_idtr_update_halo_i32(i64, memref<*xindex>, memref<*xindex>, memref<*xi32>, memref<*xindex>, memref<*xindex>, memref<*xi32>, memref<*xi32>, i64)
// CHECK-NEXT: func.func private @_idtr_update_halo_i16(i64, memref<*xindex>, memref<*xindex>, memref<*xi16>, memref<*xindex>, memref<*xindex>, memref<*xi16>, memref<*xi16>, i64)
// CHECK-NEXT: func.func private @_idtr_update_halo_i8(i64, memref<*xindex>, memref<*xindex>, memref<*xi8>, memref<*xindex>, memref<*xindex>, memref<*xi8>, memref<*xi8>, i64)
// CHECK-NEXT: func.func private @_idtr_update_halo_i1(i64, memref<*xindex>, memref<*xindex>, memref<*xi1>, memref<*xindex>, memref<*xindex>, memref<*xi1>, memref<*xi1>, i64)
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
// CHECK: [[lHalo:%.*]] = memref.cast
// CHECK: [[rHalo:%.*]] = memref.cast
// CHECK: call @_idtr_wait_i64([[handle]], [[lHalo]], [[rHalo]]) : (i64, memref<*xi64>, memref<*xi64>) -> ()
// CHECK: return
