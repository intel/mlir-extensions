// RUN: DEBUG_MESH_INDEX=2 imex-opt %s --pass-pipeline="builtin.module(func.func(mesh-spmdization),canonicalize)" | FileCheck %s

mesh.mesh @mesh4(shape = 4)

// CHECK-LABEL: @test_linspace
func.func @test_linspace() -> tensor<?xi64> {
    %c0 = arith.constant 0 : i64
    %c10 = arith.constant 10 : i64
    // CHECK: [[vcst_0:%.*]] = arith.constant 4.000000e+00 : f64
    // CHECK-NEXT: [[vcst:%.*]] = arith.constant 3.000000e+00 : f64
    // CHECK-NEXT: [[vcst_1:%.*]] = arith.constant 7.000000e+00 : f64
    // CHECK-NEXT: [[v0:%.*]] = ndarray.linspace [[vcst_0]] [[vcst_1]] [[vcst]] false : (f64, f64, f64) -> tensor<?xi64>
    %0 = ndarray.linspace %c0 %c10 %c10 false : (i64, i64, i64) -> tensor<?xi64>
    %s = mesh.sharding @mesh4 split_axes = [[0]] : !mesh.sharding
    %1 = mesh.shard %0 to %s : tensor<?xi64>
    // CHECK-NEXT: return [[v0]] : tensor<?xi64>
    return %1 : tensor<?xi64>
}

// CHECK-LABEL: @test_linspace_halos
func.func @test_linspace_halos() -> tensor<?xi64> {
    %c0 = arith.constant 0 : i64
    %c10 = arith.constant 10 : i64
    // CHECK: [[vcst:%.*]] = arith.constant 3.000000e+00 : f64
    // CHECK-NEXT: [[vcst_0:%.*]] = arith.constant 7.000000e+00 : f64
    // CHECK-NEXT: [[vcst_1:%.*]] = arith.constant 1.000000e+01 : f64
    // CHECK-NEXT: [[v0:%.*]] = ndarray.linspace [[vcst]] [[vcst_1]] [[vcst_0]] false : (f64, f64, f64) -> tensor<?xi64>
    %0 = ndarray.linspace %c0 %c10 %c10 false : (i64, i64, i64) -> tensor<?xi64>
    %s = mesh.sharding @mesh4 split_axes = [[0]] halo_sizes = [1, 3]: !mesh.sharding
    %1 = mesh.shard %0 to %s : tensor<?xi64>
    // CHECK-NEXT: return [[v0]] : tensor<?xi64>
    return %1 : tensor<?xi64>
}

// CHECK-LABEL: @test_linspace_offsets
func.func @test_linspace_offsets() -> tensor<?xi64> {
    %c0 = arith.constant 0 : i64
    %c10 = arith.constant 10 : i64
    // CHECK: [[vcst:%.*]] = arith.constant 1.000000e+00 : f64
    // CHECK-NEXT: [[vcst_0:%.*]] = arith.constant 5.000000e+00 : f64
    // CHECK-NEXT: [[vcst_1:%.*]] = arith.constant 6.000000e+00 : f64
    // CHECK-NEXT: [[v0:%.*]] = ndarray.linspace [[vcst_0]] [[vcst_1]] [[vcst]] false : (f64, f64, f64) -> tensor<?xi64>
    %0 = ndarray.linspace %c0 %c10 %c10 false : (i64, i64, i64) -> tensor<?xi64>
    %s = mesh.sharding @mesh4 split_axes = [[0]] sharded_dims_offsets = [0, 0, 5, 6, 10]: !mesh.sharding
    %1 = mesh.shard %0 to %s : tensor<?xi64>
    // CHECK-NEXT: return [[v0]] : tensor<?xi64>
    return %1 : tensor<?xi64>
}
