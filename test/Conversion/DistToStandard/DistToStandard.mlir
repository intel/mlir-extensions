// RUN: imex-opt --split-input-file --convert-dist-to-standard %s -verify-diagnostics -o -| FileCheck %s

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_register_ptensor() -> i64 {
        %0 = shape.const_shape [-1] : tensor<1xindex>
        %1 = "dist.register_ptensor"(%0) : (tensor<1xindex>) -> i64
        return %1 : i64
    }
}
// CHECK-LABEL: func.func private @_idtr_init_dtensor(tensor<?xi64>, i64) -> i64
// CHECK-NEXT: func.func private @_idtr_local_shape(i64, tensor<?xi64>, i64)
// CHECK-NEXT: func.func private @_idtr_local_offsets(i64, tensor<?xi64>, i64)
// CHECK-NEXT: func.func private @_idtr_reduce_all(tensor<i64>, i32, i32)
// CHECK-NEXT: func.func @test_register_ptensor() -> i64 {
// CHECK-NEXT: arith.constant
// CHECK-NEXT: shape.const_shape [-1] : tensor<1xindex>
// CHECK-NEXT: tensor.cast
// CHECK-NEXT: arith.index_cast
// CHECK-NEXT: [[C0:%.*]] = call @_idtr_init_dtensor({{.*}}) -> i64
// CHECK-NEXT: return [[C0]] : i64

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_local_shape(%arg0: i64) -> tensor<?xi64> {
        %0 = "dist.local_shape"(%arg0) {rank = 1 : i64}: (i64) -> tensor<?xi64>
        return %0 : tensor<?xi64>
    }
}
// CHECK-LABEL: func.func private @_idtr_init_dtensor(tensor<?xi64>, i64) -> i64
// CHECK-NEXT: func.func private @_idtr_local_shape(i64, tensor<?xi64>, i64)
// CHECK-NEXT: func.func private @_idtr_local_offsets(i64, tensor<?xi64>, i64)
// CHECK-NEXT: func.func private @_idtr_reduce_all(tensor<i64>, i32, i32)
// CHECK: func.func @test_local_shape(%arg0: i64) -> tensor<?xi64> {
// CHECK: [[C0:%.*]] = tensor.empty({{.*}}) : tensor<?xi64>
// CHECK: call @_idtr_local_shape(%arg0, [[C0]], {{.*}}) : (i64, tensor<?xi64>, i64) -> ()
// CHECK: return [[C0]] : tensor<?xi64>

// // -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_local_offsets(%arg0: i64) -> tensor<?xi64> {
        %0 = "dist.local_offsets"(%arg0) {rank = 1 : i64} : (i64) -> tensor<?xi64>
        return %0 : tensor<?xi64>
    }
}
// CHECK-LABEL: func.func private @_idtr_init_dtensor(tensor<?xi64>, i64) -> i64
// CHECK-NEXT: func.func private @_idtr_local_shape(i64, tensor<?xi64>, i64)
// CHECK-NEXT: func.func private @_idtr_local_offsets(i64, tensor<?xi64>, i64)
// CHECK-NEXT: func.func private @_idtr_reduce_all(tensor<i64>, i32, i32)
// CHECK: func.func @test_local_offsets(%arg0: i64) -> tensor<?xi64> {
// CHECK: [[C0:%.*]] = tensor.empty({{.*}}) : tensor<?xi64>
// CHECK: call @_idtr_local_offsets(%arg0, [[C0]], {{.*}}) : (i64, tensor<?xi64>, i64) -> ()
// CHECK: return [[C0]] : tensor<?xi64>

// // -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_allreduce(%arg0: tensor<i64>) -> tensor<i64> {
        %0 = "dist.allreduce"(%arg0) {op = 4 : i32} : (tensor<i64>) -> tensor<i64>
        return %0 : tensor<i64>
    }
}
// CHECK-LABEL: func.func private @_idtr_init_dtensor(tensor<?xi64>, i64) -> i64
// CHECK-NEXT: func.func private @_idtr_local_shape(i64, tensor<?xi64>, i64)
// CHECK-NEXT: func.func private @_idtr_local_offsets(i64, tensor<?xi64>, i64)
// CHECK-NEXT: func.func private @_idtr_reduce_all(tensor<i64>, i32, i32)
// CHECK: func.func @test_allreduce(%arg0: tensor<i64>) -> tensor<i64> {
// CHECK: call @_idtr_reduce_all(%arg0, {{.*}}) : (tensor<i64>, i32, i32) -> ()
// CHECK: return %arg0 : tensor<i64>
