// RUN: imex-opt %s --pass-pipeline="builtin.module(func.func(sharding-propagation))" | FileCheck %s

builtin.module {
    
mesh.mesh @mesh4(shape = 4)
// CHECK-LABEL: @test_shard_propagate_subview_balanced
func.func @test_shard_propagate_subview_balanced(%arg0: tensor<1024x1024xi64>) -> tensor<4x3xi64> {
    // CHECK: [[S:%.*]] = mesh.sharding @mesh4 split_axes = {{\[\[}}0]] : !mesh.sharding
    %s = mesh.sharding @mesh4 split_axes = [[0]] : !mesh.sharding
    // CHECK: mesh.shard %arg0 to [[S]] : tensor<1024x1024xi64>
    %0 = mesh.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK: mesh.sharding @mesh4 split_axes = {{\[\[}}0]] sharded_dims_offsets = [1, 1, 1, 1] : !mesh.sharding
    %1 = ndarray.subview %0[1, 0][4, 3][256, 1] : tensor<1024x1024xi64> to tensor<4x3xi64>
    return %1 : tensor<4x3xi64>
}

// CHECK-LABEL: @test_shard_propagate_subview_leading
func.func @test_shard_propagate_subview_leading(%arg0: tensor<1024x1024xi64>) -> tensor<3x3xi64> {
    // CHECK: [[S:%.*]] = mesh.sharding @mesh4 split_axes = {{\[\[}}0]] : !mesh.sharding
    %s = mesh.sharding @mesh4 split_axes = [[0]] : !mesh.sharding
    // CHECK: mesh.shard %arg0 to [[S]] : tensor<1024x1024xi64>
    %0 = mesh.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK: mesh.sharding @mesh4 split_axes = {{\[\[}}0]] sharded_dims_offsets = [3, 0, 0, 0] : !mesh.sharding
    %1 = ndarray.subview %0[0, 0][3, 3][3, 1] : tensor<1024x1024xi64> to tensor<3x3xi64>
    return %1 : tensor<3x3xi64>
}

// CHECK-LABEL: @test_shard_propagate_subview_mid
func.func @test_shard_propagate_subview_mid(%arg0: tensor<1024x1024xi64>) -> tensor<3x3xi64> {
    // CHECK: [[S:%.*]] = mesh.sharding @mesh4 split_axes = {{\[\[}}0]] : !mesh.sharding
    %s = mesh.sharding @mesh4 split_axes = [[0]] : !mesh.sharding
    // CHECK: mesh.shard %arg0 to [[S]] : tensor<1024x1024xi64>
    %0 = mesh.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK: mesh.sharding @mesh4 split_axes = {{\[\[}}0]] sharded_dims_offsets = [0, 1, 2, 0] : !mesh.sharding
    %1 = ndarray.subview %0[511, 0][3, 3][1, 1] : tensor<1024x1024xi64> to tensor<3x3xi64>
    return %1 : tensor<3x3xi64>
}

// CHECK-LABEL: @test_shard_propagate_subview_trailing
func.func @test_shard_propagate_subview_trailing(%arg0: tensor<1024x1024xi64>) -> tensor<3x3xi64> {
    // CHECK: [[S:%.*]] = mesh.sharding @mesh4 split_axes = {{\[\[}}0]] : !mesh.sharding
    %s = mesh.sharding @mesh4 split_axes = [[0]] : !mesh.sharding
    // CHECK: mesh.shard %arg0 to [[S]] : tensor<1024x1024xi64>
    %0 = mesh.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK: mesh.sharding @mesh4 split_axes = {{\[\[}}0]] sharded_dims_offsets = [0, 0, 0, 3] : !mesh.sharding
    %1 = ndarray.subview %0[1000, 0][3, 3][1, 1] : tensor<1024x1024xi64> to tensor<3x3xi64>
    return %1 : tensor<3x3xi64>
}

// CHECK-LABEL: @test_shard_propagate_subview_gap
func.func @test_shard_propagate_subview_gap(%arg0: tensor<1024x1024xi64>) -> tensor<3x3xi64> {
    // CHECK: [[S:%.*]] = mesh.sharding @mesh4 split_axes = {{\[\[}}0]] : !mesh.sharding
    %s = mesh.sharding @mesh4 split_axes = [[0]] : !mesh.sharding
    // CHECK: mesh.shard %arg0 to [[S]] : tensor<1024x1024xi64>
    %0 = mesh.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK: mesh.sharding @mesh4 split_axes = {{\[\[}}0]] sharded_dims_offsets = [1, 0, 1, 1] : !mesh.sharding
    %1 = ndarray.subview %0[255, 0][3, 3][257, 1] : tensor<1024x1024xi64> to tensor<3x3xi64>
    return %1 : tensor<3x3xi64>
}

// CHECK-LABEL: @test_shard_propagate_insert_slice
func.func @test_shard_propagate_insert_slice(%arg0: tensor<1024x1024xi64>, %arg1: tensor<3x3xi64>) {
    %s = mesh.sharding @mesh4 split_axes = [[0]] : !mesh.sharding
    %0 = mesh.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK: %[[sharding_2:.*]] = mesh.sharding @mesh4 split_axes = {{\[\[}}0]] sharded_dims_offsets = [3, 0, 0, 0] : !mesh.sharding
    // CHECK: %[[sharding_annotated_1:.*]] = mesh.shard %arg1 to %[[sharding_2]] annotate_for_users : tensor<3x3xi64>
    // CHECK-NEXT: ndarray.insert_slice %[[sharding_annotated_1]] into
    ndarray.insert_slice %arg1 into %0[0, 0][3, 3][1, 1] : tensor<3x3xi64> into tensor<1024x1024xi64>
    return
}

mesh.mesh @mesh4x4(shape = 4x4)

// CHECK-LABEL: @test_shard_propagate_insert_slice_2d
func.func @test_shard_propagate_insert_slice_2d(%arg0: tensor<1024x1024xi64>, %arg1: tensor<3x3xi64>) {
    %s = mesh.sharding @mesh4x4 split_axes = [[0], [1]] : !mesh.sharding
    %0 = mesh.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK: %[[sharding_2:.*]] = mesh.sharding @mesh4x4 split_axes = {{\[\[}}0], [1]] sharded_dims_offsets = [3, 0, 0, 0, 1, 0, 1, 1] : !mesh.sharding
    // CHECK: %[[sharding_annotated_1:.*]] = mesh.shard %arg1 to %[[sharding_2]] annotate_for_users : tensor<3x3xi64>
    // CHECK-NEXT: ndarray.insert_slice %[[sharding_annotated_1]] into
    ndarray.insert_slice %arg1 into %0[0, 255][3, 3][1, 257] : tensor<3x3xi64> into tensor<1024x1024xi64>
    return
}

// CHECK-LABEL: @test_shard_propagate_insert_slice_2d_2
func.func @test_shard_propagate_insert_slice_2d_2(%arg0: tensor<1024x1024xi64>, %arg1: tensor<600x3xi64>) {
    %s = mesh.sharding @mesh4x4 split_axes = [[0], [1]] : !mesh.sharding
    %0 = mesh.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK: %[[sharding_2:.*]] = mesh.sharding @mesh4x4 split_axes = {{\[\[}}0], [1]] sharded_dims_offsets = [156, 256, 188, 0, 1, 0, 1, 1] : !mesh.sharding
    // CHECK: %[[sharding_annotated_1:.*]] = mesh.shard %arg1 to %[[sharding_2]] annotate_for_users : tensor<600x3xi64>
    // CHECK-NEXT: ndarray.insert_slice %[[sharding_annotated_1]] into
    ndarray.insert_slice %arg1 into %0[100, 255][600, 3][1, 257] : tensor<600x3xi64> into tensor<1024x1024xi64>
    return
}

}
