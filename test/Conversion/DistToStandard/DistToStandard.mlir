// RUN: imex-opt --split-input-file --convert-dist-to-standard %s -verify-diagnostics -o -| FileCheck %s

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_nprocs(%arg0: index) -> index {
        %1 = "dist.nprocs"(%arg0) : (index) -> index
        return %1 : index
    }
}
// CHECK-LABEL: func.func private @_idtr_nprocs(index) -> index
// CHECK-LABEL: func.func private @_idtr_prank(index) -> index
// CHECK-LABEL: func.func private @_idtr_reduce_all(memref<i64, strided<[], offset: ?>>, i32, i32)
// CHECK-LABEL: func.func @test_nprocs(%arg0: index) -> index {
// CHECK: @_idtr_nprocs(%arg0)

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_prank(%arg0: index) -> index {
        %1 = "dist.prank"(%arg0) : (index) -> index
        return %1 : index
    }
}
// CHECK-LABEL: func.func private @_idtr_nprocs(index) -> index
// CHECK-LABEL: func.func private @_idtr_prank(index) -> index
// CHECK-LABEL: func.func private @_idtr_reduce_all(memref<i64, strided<[], offset: ?>>, i32, i32)
// CHECK-LABEL: func.func @test_prank(%arg0: index) -> index {
// CHECK: call @_idtr_prank(%arg0)

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_init_dist_tensor(%gshape: memref<1xindex>, %pt: !ptensor.ptensor<1 x i64>, %loffs: memref<1xindex>, %team: index) -> !dist.dtensor<<1 x i64>> {
        %1 = "dist.init_dist_tensor"(%gshape, %pt, %loffs, %team) : (memref<1xindex>, !ptensor.ptensor<1 x i64>, memref<1xindex>, index) -> !dist.dtensor<<1 x i64>>
        return %1 : !dist.dtensor<<1 x i64>>
    }
}
// CHECK-LABEL: func.func private @_idtr_nprocs(index) -> index
// CHECK-LABEL: func.func private @_idtr_prank(index) -> index
// CHECK-LABEL: func.func private @_idtr_reduce_all(memref<i64, strided<[], offset: ?>>, i32, i32)
// CHECK-LABEL: func.func @test_init_dist_tensor
// CHECK: builtin.unrealized_conversion_cast

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_local_offsets(%np : index, %prank: index, %shape: memref<1xindex>) -> memref<1xindex> {
        %0 = "dist.local_offsets"(%np, %prank, %shape) {rank = 1 : i64} : (index, index, memref<1xindex>) -> memref<1xindex>
        return %0 : memref<1xindex>
    }
}
// CHECK-LABEL: func.func private @_idtr_nprocs(index) -> index
// CHECK-LABEL: func.func private @_idtr_prank(index) -> index
// CHECK-LABEL: func.func private @_idtr_reduce_all(memref<i64, strided<[], offset: ?>>, i32, i32)
// CHECK-LABEL: func.func @test_local_offsets(%arg0: index, %arg1: index, %arg2: memref<1xindex>) -> memref<1xindex> {
// CHECK: memref.load
// CHECK: arith.subi
// CHECK: arith.muli

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_local_shape(%np : index, %prank: index, %shape: memref<1xindex>) -> memref<1xindex> {
        %0 = "dist.local_shape"(%np, %prank, %shape) {rank = 1 : i64} : (index, index, memref<1xindex>) -> memref<1xindex>
        return %0 : memref<1xindex>
    }
}
// CHECK-LABEL: func.func private @_idtr_nprocs(index) -> index
// CHECK-LABEL: func.func private @_idtr_prank(index) -> index
// CHECK-LABEL: func.func private @_idtr_reduce_all(memref<i64, strided<[], offset: ?>>, i32, i32)
// CHECK-LABEL: func.func @test_local_shape(%arg0: index, %arg1: index, %arg2: memref<1xindex>) -> memref<1xindex> {
// CHECK: memref.load
// CHECK: arith.subi

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_allreduce(%arg0: memref<i64, strided<[], offset: ?>>) -> memref<i64, strided<[], offset: ?>> {
        %0 = "dist.allreduce"(%arg0) {op = 4 : i32} : (memref<i64, strided<[], offset: ?>>) -> memref<i64, strided<[], offset: ?>>
        return %0 : memref<i64, strided<[], offset: ?>>
    }
}
// CHECK-LABEL: func.func private @_idtr_nprocs(index) -> index
// CHECK-LABEL: func.func private @_idtr_prank(index) -> index
// CHECK-LABEL: func.func private @_idtr_reduce_all(memref<i64, strided<[], offset: ?>>, i32, i32)
// CHECK-LABEL: func.func @test_allreduce(%arg0: memref<i64, strided<[], offset: ?>>) -> memref<i64, strided<[], offset: ?>> {
// CHECK: @_idtr_reduce_all

// -----
module {
    func.func @test_local_of_slice(%arg0: !dist.dtensor<<1 x i64>>) -> (index, index, index) {
        %c0 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %l_offsets, %l_sizes, %g_offsets = dist.local_of_slice %arg0[%c0] [%c3] [%c3] : !dist.dtensor<<1 x i64>> to (index, index, index)
        return %l_offsets, %l_sizes, %g_offsets : index, index, index
    }
}
// CHECK-LABEL: func.func @test_local_of_slice(%arg0: memref<?xindex>, %arg1: !ptensor.ptensor<1 x i64>, %arg2: memref<?xindex>, %arg3: index) -> (index, index, index) {
// CHECK: "ptensor.extract_tensor"(%arg1) : (!ptensor.ptensor<1 x i64>) -> memref<?xi64, strided<[?], offset: ?>>
// CHECK: memref.dim %0, %c0_0 : memref<?xi64, strided<[?], offset: ?>>
// CHECK: memref.load %arg2[%c0_0] : memref<?xindex>
// CHECK: arith.cmpi ult
// CHECK: arith.cmpi ule
// CHECK: arith.select
// CHECK: arith.select
// CHECK: arith.select
// CHECK: [[V1:%.*]] = arith.select
// CHECK: arith.select
// CHECK: [[V2:%.*]] = arith.select
// CHECK: [[V3:%.*]] = arith.addi
// CHECK: return [[V1]], [[V2]], [[V3]] : index, index, index
