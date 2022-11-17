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
// CHECK-LABEL: func.func private @_idtr_reduce_all(tensor<i64>, i32, i32)
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
// CHECK-LABEL: func.func private @_idtr_reduce_all(tensor<i64>, i32, i32)
// CHECK-LABEL: func.func @test_prank(%arg0: index) -> index {
// CHECK: call @_idtr_prank(%arg0)

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_init_dist_tensor(%gshape: tensor<1xindex>, %pt: !ptensor.ptensor<1 x i64>, %loffs: tensor<1xindex>, %team: index) -> !dist.dtensor<<1 x i64>> {
        %1 = "dist.init_dist_tensor"(%gshape, %pt, %loffs, %team) : (tensor<1xindex>, !ptensor.ptensor<1 x i64>, tensor<1xindex>, index) -> !dist.dtensor<<1 x i64>>
        return %1 : !dist.dtensor<<1 x i64>>
    }
}
// CHECK-LABEL: func.func private @_idtr_nprocs(index) -> index
// CHECK-LABEL: func.func private @_idtr_prank(index) -> index
// CHECK-LABEL: func.func private @_idtr_reduce_all(tensor<i64>, i32, i32)
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
// CHECK-LABEL: func.func private @_idtr_reduce_all(tensor<i64>, i32, i32)
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
// CHECK-LABEL: func.func private @_idtr_reduce_all(tensor<i64>, i32, i32)
// CHECK-LABEL: func.func @test_local_shape(%arg0: index, %arg1: index, %arg2: memref<1xindex>) -> memref<1xindex> {
// CHECK: memref.load
// CHECK: arith.subi

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_allreduce(%arg0: tensor<i64>) -> tensor<i64> {
        %0 = "dist.allreduce"(%arg0) {op = 4 : i32} : (tensor<i64>) -> tensor<i64>
        return %0 : tensor<i64>
    }
}
// CHECK-LABEL: func.func private @_idtr_nprocs(index) -> index
// CHECK-LABEL: func.func private @_idtr_prank(index) -> index
// CHECK-LABEL: func.func private @_idtr_reduce_all(tensor<i64>, i32, i32)
// CHECK-LABEL: func.func @test_allreduce(%arg0: tensor<i64>) -> tensor<i64> {
// CHECK: @_idtr_reduce_all
