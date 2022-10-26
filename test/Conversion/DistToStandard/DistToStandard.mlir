// RUN: imex-opt --split-input-file --convert-dist-to-standard %s -verify-diagnostics -o -| FileCheck %s

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_nprocs(%arg0: i64) -> i64 {
        %1 = "dist.nprocs"(%arg0) : (i64) -> i64
        return %1 : i64
    }
}
// CHECK-LABEL: func.func private @_idtr_nprocs(i64) -> i64
// CHECK-LABEL: func.func private @_idtr_prank() -> i64
// CHECK-LABEL: func.func private @_idtr_reduce_all(tensor<i64>, i32, i32)
// CHECK-LABEL: func.func @test_nprocs(%arg0: i64) -> i64 {
// CHECK: @_idtr_nprocs(%arg0)

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_prank(%arg0: i64) -> i64 {
        %1 = "dist.prank"(%arg0) : (i64) -> i64
        return %1 : i64
    }
}
// CHECK-LABEL: func.func private @_idtr_nprocs(i64) -> i64
// CHECK-LABEL: func.func private @_idtr_prank() -> i64
// CHECK-LABEL: func.func private @_idtr_reduce_all(tensor<i64>, i32, i32)
// CHECK-LABEL: func.func @test_prank(%arg0: i64) -> i64 {
// CHECK: call @_idtr_prank()

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_distinfo(%shape: tensor<1xindex>, %team: i64) -> !dist.info<1> {
        %1 = "dist.distinfo"(%shape, %team) {rank = 1 : i64} : (tensor<1xindex>, i64) -> !dist.info<1>
        return %1 : !dist.info<1>
    }
}
// CHECK-LABEL: func.func private @_idtr_nprocs(i64) -> i64
// CHECK-LABEL: func.func private @_idtr_prank() -> i64
// CHECK-LABEL: func.func private @_idtr_reduce_all(tensor<i64>, i32, i32)
// CHECK-LABEL: func.func @test_distinfo
// CHECK: @_idtr_nprocs
// CHECK: call @_idtr_prank
// CHECK: builtin.unrealized_conversion_cast

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_init_dist_tensor(%arg0: !ptensor.ptensor<tensor<?xi64>>, %arg1: !dist.info<1>) -> !dist.dtensor<<tensor<?xi64>>> {
        %1 = "dist.init_dist_tensor"(%arg0, %arg1) : (!ptensor.ptensor<tensor<?xi64>>, !dist.info<1>) -> !dist.dtensor<<tensor<?xi64>>>
        return %1 : !dist.dtensor<<tensor<?xi64>>>
    }
}
// CHECK-LABEL: func.func private @_idtr_nprocs(i64) -> i64
// CHECK-LABEL: func.func private @_idtr_prank() -> i64
// CHECK-LABEL: func.func private @_idtr_reduce_all(tensor<i64>, i32, i32)
// CHECK-LABEL: func.func @test_init_dist_tensor
// CHECK: builtin.unrealized_conversion_cast

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_local_offsets(%np : i64, %prank: i64, %shape: tensor<1xindex>) -> tensor<1xi64> {
        %0 = "dist.local_offsets"(%np, %prank, %shape) {rank = 1 : i64} : (i64, i64, tensor<1xindex>) -> tensor<1xi64>
        return %0 : tensor<1xi64>
    }
}
// CHECK-LABEL: func.func private @_idtr_nprocs(i64) -> i64
// CHECK-LABEL: func.func private @_idtr_prank() -> i64
// CHECK-LABEL: func.func private @_idtr_reduce_all(tensor<i64>, i32, i32)
// CHECK-LABEL: func.func @test_local_offsets(%arg0: i64, %arg1: i64, %arg2: tensor<1xindex>) -> tensor<1xi64> {
// CHECK: shape.get_extent
// CHECK: arith.subi
// CHECK: arith.muli

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_local_shape(%np : i64, %prank: i64, %shape: tensor<1xindex>) -> tensor<1xi64> {
        %0 = "dist.local_shape"(%np, %prank, %shape) {rank = 1 : i64} : (i64, i64, tensor<1xindex>) -> tensor<1xi64>
        return %0 : tensor<1xi64>
    }
}
// CHECK-LABEL: func.func private @_idtr_nprocs(i64) -> i64
// CHECK-LABEL: func.func private @_idtr_prank() -> i64
// CHECK-LABEL: func.func private @_idtr_reduce_all(tensor<i64>, i32, i32)
// CHECK-LABEL: func.func @test_local_shape(%arg0: i64, %arg1: i64, %arg2: tensor<1xindex>) -> tensor<1xi64> {
// CHECK: shape.get_extent
// CHECK: arith.subi

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_allreduce(%arg0: tensor<i64>) -> tensor<i64> {
        %0 = "dist.allreduce"(%arg0) {op = 4 : i32} : (tensor<i64>) -> tensor<i64>
        return %0 : tensor<i64>
    }
}
// CHECK-LABEL: func.func private @_idtr_nprocs(i64) -> i64
// CHECK-LABEL: func.func private @_idtr_prank() -> i64
// CHECK-LABEL: func.func private @_idtr_reduce_all(tensor<i64>, i32, i32)
// CHECK-LABEL: func.func @test_allreduce(%arg0: tensor<i64>) -> tensor<i64> {
// CHECK: @_idtr_reduce_all
