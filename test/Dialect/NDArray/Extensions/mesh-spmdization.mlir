// RUN: DEBUG_MESH_INDEX=2,1 imex-opt %s --pass-pipeline="builtin.module(func.func(mesh-spmdization),canonicalize)" | FileCheck %s

mesh.mesh @mesh4(shape = 4)

// CHECK-LABEL: @test_copyop
// CHECK-SAME: [[varg0:%.*]]: tensor<256x1024xi64>) -> tensor<256x1024xi64> {
func.func @test_copyop(%arg0: tensor<1024x1024xi64>) -> tensor<1024x1024xi64> {
    %sharding = mesh.sharding @mesh4 split_axes = [[0]] : !mesh.sharding
    %sharding_annotated = mesh.shard %arg0 to %sharding : tensor<1024x1024xi64>
    %sharding_0 = mesh.sharding @mesh4 split_axes = [[0]] : !mesh.sharding
    %sharding_annotated_1 = mesh.shard %sharding_annotated to %sharding_0 annotate_for_users : tensor<1024x1024xi64>
    // CHECK-NEXT: [[v0:%.*]] = ndarray.copy [[varg0]] : tensor<256x1024xi64> -> tensor<256x1024xi64>
    %0 = ndarray.copy %sharding_annotated_1 : tensor<1024x1024xi64> -> tensor<1024x1024xi64>
    %sharding_2 = mesh.sharding @mesh4 split_axes = [[0]] : !mesh.sharding
    %sharding_annotated_3 = mesh.shard %0 to %sharding_2 : tensor<1024x1024xi64>
    // CHECK-NEXT: return [[v0]] : tensor<256x1024xi64>
    return %sharding_annotated_3 : tensor<1024x1024xi64>
}

// CHECK-LABEL: @test_deleteop
// CHECK-SAME: [[varg0:%.*]]: tensor<256x1024xi64>) {
func.func @test_deleteop(%arg0: tensor<1024x1024xi64>) {
    %sharding = mesh.sharding @mesh4 split_axes = [[0]] : !mesh.sharding
    %sharding_annotated = mesh.shard %arg0 to %sharding : tensor<1024x1024xi64>
    %sharding_0 = mesh.sharding @mesh4 split_axes = [[0]] : !mesh.sharding
    %sharding_annotated_1 = mesh.shard %sharding_annotated to %sharding_0 annotate_for_users : tensor<1024x1024xi64>
    // CHECK-NEXT: ndarray.delete [[varg0]] : tensor<256x1024xi64>
    ndarray.delete %sharding_annotated_1 : tensor<1024x1024xi64>
    // CHECK-NEXT: return
    return
}

// CHECK-LABEL: @test_cast_elemtypeop
// CHECK-SAME: [[varg0:%.*]]: tensor<256x1024xi64>) -> tensor<256x1024xf64> {
func.func @test_cast_elemtypeop(%arg0: tensor<1024x1024xi64>) -> tensor<1024x1024xf64> {
    %sharding = mesh.sharding @mesh4 split_axes = [[0]] : !mesh.sharding
    %sharding_annotated = mesh.shard %arg0 to %sharding : tensor<1024x1024xi64>
    %sharding_0 = mesh.sharding @mesh4 split_axes = [[0]] : !mesh.sharding
    %sharding_annotated_1 = mesh.shard %sharding_annotated to %sharding_0 annotate_for_users : tensor<1024x1024xi64>
    // CHECK-NEXT: [[v0:%.*]] = ndarray.cast_elemtype [[varg0]] : tensor<256x1024xi64> to tensor<256x1024xf64>
    %0 = ndarray.cast_elemtype %sharding_annotated_1 : tensor<1024x1024xi64> to tensor<1024x1024xf64>
    %sharding_2 = mesh.sharding @mesh4 split_axes = [[0]] : !mesh.sharding
    %sharding_annotated_3 = mesh.shard %0 to %sharding_2 : tensor<1024x1024xf64>
    // CHECK-NEXT: return [[v0]] : tensor<256x1024xf64>
    return %sharding_annotated_3 : tensor<1024x1024xf64>
}

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

mesh.mesh @mesh4x4(shape = 4x4)

// CHECK-LABEL: func.func @test_subview_insert_slice_2d(
// CHECK-SAME: [[varg0:%.*]]: tensor<300x300xi64>) -> tensor<302x302xi64> {
func.func @test_subview_insert_slice_2d(%arg0: tensor<1200x1200xi64>) -> tensor<1200x1200xi64> {
    %sharding = mesh.sharding @mesh4x4 split_axes = [[0], [1]] : !mesh.sharding
    %sharding_annotated = mesh.shard %arg0 to %sharding : tensor<1200x1200xi64>
    %sharding_0 = mesh.sharding @mesh4x4 split_axes = [[0], [1]] halo_sizes = [0, 2, 0, 2] : !mesh.sharding
    %sharding_annotated_1 = mesh.shard %sharding_annotated to %sharding_0 : tensor<1200x1200xi64>
    %sharding_2 = mesh.sharding @mesh4x4 split_axes = [[0], [1]] sharded_dims_offsets = [0, 298, 598, 898, 1000, 0, 298, 598, 898, 1000] : !mesh.sharding
    %sharding_annotated_3 = mesh.shard %sharding_annotated_1 to %sharding_0 annotate_for_users : tensor<1200x1200xi64>
    // CHECK: [[v0:%.*]] = tensor.empty() : tensor<302x302xi64>
    // CHECK-NEXT: [[vinserted_slice:%.*]] = tensor.insert_slice [[varg0]] into [[v0]][0, 0] [300, 300] [1, 1] : tensor<300x300xi64> into tensor<302x302xi64>
    // CHECK-NEXT: [[v1:%.*]] = mesh.update_halo [[vinserted_slice]] on @mesh4x4 split_axes = {{\[\[}}0], [1]] halo_sizes = [0, 2, 0, 2] : tensor<302x302xi64>
    // CHECK-NEXT: [[v2:%.*]] = ndarray.subview [[v1]][2, 2] [300, 300] [1, 1] : tensor<302x302xi64> to tensor<300x300xi64>
    %0 = ndarray.subview %sharding_annotated_3[4, 4] [1000, 1000] [1, 1] : tensor<1200x1200xi64> to tensor<1000x1000xi64>
    %sharding_annotated_4 = mesh.shard %0 to %sharding_2 : tensor<1000x1000xi64>
    %sharding_annotated_5 = mesh.shard %sharding_annotated_4 to %sharding_2 annotate_for_users : tensor<1000x1000xi64>
    %sharding_annotated_6 = mesh.shard %sharding_annotated_1 to %sharding_0 annotate_for_users : tensor<1200x1200xi64>
    // CHECK-NEXT: [[v3:%.*]] = ndarray.insert_slice [[v2]] into [[v1]][0, 0] [300, 300] [1, 1] : tensor<300x300xi64> into tensor<302x302xi64>
    // CHECK-NEXT: [[v4:%.*]] = mesh.update_halo [[v3]] on @mesh4x4 split_axes = {{\[\[}}0], [1]] halo_sizes = [0, 2, 0, 2] : tensor<302x302xi64>
    %1 = ndarray.insert_slice %sharding_annotated_5 into %sharding_annotated_6[2, 2] [1000, 1000] [1, 1] : tensor<1000x1000xi64> into tensor<1200x1200xi64>
    %sharding_annotated_7 = mesh.shard %1 to %sharding_0 annotate_for_users : tensor<1200x1200xi64>
    // CHECK: return [[v4]] : tensor<302x302xi64>
    return %sharding_annotated_7 : tensor<1200x1200xi64>
}
