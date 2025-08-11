// RUN: imex-opt %s --pass-pipeline="builtin.module(func.func(sharding-propagation))" | FileCheck %s

builtin.module {
shard.grid @mesh4(shape = 4)

// CHECK-LABEL: @test_copyop
// CHECK-SAME: [[varg0:%.*]]: tensor<1024x1024xi64>) -> tensor<1024x1024xi64> {
func.func @test_copyop(%arg0: tensor<1024x1024xi64>) -> tensor<1024x1024xi64> {
    // CHECK-NEXT: [[vsharding:%.*]] = shard.sharding @mesh4 split_axes = {{\[\[}}0]] : !shard.sharding
    %s = shard.sharding @mesh4 split_axes = [[0]] : !shard.sharding
    // CHECK-NEXT: [[vsharding_annotated:%.*]] = shard.shard [[varg0]] to [[vsharding]] : tensor<1024x1024xi64>
    %0 = shard.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK-NEXT: [[vsharding_0:%.*]] = shard.sharding @mesh4 split_axes = {{\[\[}}0]] : !shard.sharding
    // CHECK-NEXT: [[vsharding_annotated_1:%.*]] = shard.shard [[vsharding_annotated]] to [[vsharding_0]] annotate_for_users : tensor<1024x1024xi64>
    // CHECK-NEXT: [[v0:%.*]] = ndarray.copy [[vsharding_annotated_1]] : tensor<1024x1024xi64> -> tensor<1024x1024xi64>
    // CHECK-NEXT: [[vsharding_2:%.*]] = shard.sharding @mesh4 split_axes = {{\[\[}}0]] : !shard.sharding
    // CHECK-NEXT: [[vsharding_annotated_3:%.*]] = shard.shard [[v0]] to [[vsharding_2]] : tensor<1024x1024xi64>
    %1 = ndarray.copy %0 : tensor<1024x1024xi64> -> tensor<1024x1024xi64>
    return %1 : tensor<1024x1024xi64>
}

// CHECK-LABEL: @test_deleteop
// CHECK-SAME: [[varg0:%.*]]: tensor<1024x1024xi64>) {
func.func @test_deleteop(%arg0: tensor<1024x1024xi64>) {
    // CHECK-NEXT: [[vsharding:%.*]] = shard.sharding @mesh4 split_axes = {{\[\[}}0]] : !shard.sharding
    %s = shard.sharding @mesh4 split_axes = [[0]] : !shard.sharding
    // CHECK-NEXT: [[vsharding_annotated:%.*]] = shard.shard [[varg0]] to [[vsharding]] : tensor<1024x1024xi64>
    %0 = shard.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK-NEXT: [[vsharding_0:%.*]] = shard.sharding @mesh4 split_axes = {{\[\[}}0]] : !shard.sharding
    // CHECK-NEXT: [[vsharding_annotated_1:%.*]] = shard.shard [[vsharding_annotated]] to [[vsharding_0]] annotate_for_users : tensor<1024x1024xi64>
    // CHECK-NEXT: ndarray.delete [[vsharding_annotated_1]] : tensor<1024x1024xi64>
    ndarray.delete %0 : tensor<1024x1024xi64>
    return
}

// CHECK-LABEL: @test_cast_elemtypeop
// CHECK-SAME: [[varg0:%.*]]: tensor<1024x1024xi64>) -> tensor<1024x1024xf64> {
func.func @test_cast_elemtypeop(%arg0: tensor<1024x1024xi64>) -> tensor<1024x1024xf64> {
    // CHECK-NEXT: [[vsharding:%.*]] = shard.sharding @mesh4 split_axes = {{\[\[}}0]] : !shard.sharding
    %s = shard.sharding @mesh4 split_axes = [[0]] : !shard.sharding
    // CHECK-NEXT: [[vsharding_annotated:%.*]] = shard.shard [[varg0]] to [[vsharding]] : tensor<1024x1024xi64>
    %0 = shard.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK-NEXT: [[vsharding_0:%.*]] = shard.sharding @mesh4 split_axes = {{\[\[}}0]] : !shard.sharding
    // CHECK-NEXT: [[vsharding_annotated_1:%.*]] = shard.shard [[vsharding_annotated]] to [[vsharding_0]] annotate_for_users : tensor<1024x1024xi64>
    // CHECK-NEXT: [[v0:%.*]] = ndarray.cast_elemtype [[vsharding_annotated_1]] : tensor<1024x1024xi64> to tensor<1024x1024xf64>
    // CHECK-NEXT: [[vsharding_2:%.*]] = shard.sharding @mesh4 split_axes = {{\[\[}}0]] : !shard.sharding
    // CHECK-NEXT: [[vsharding_annotated_3:%.*]] = shard.shard [[v0]] to [[vsharding_2]] : tensor<1024x1024xf64>
    %1 = ndarray.cast_elemtype %0 : tensor<1024x1024xi64> to tensor<1024x1024xf64>
    return %1 : tensor<1024x1024xf64>
}

// CHECK-LABEL: @test_shard_propagate_subview_balanced
func.func @test_shard_propagate_subview_balanced(%arg0: tensor<1024x1024xi64>) -> tensor<4x3xi64> {
    // CHECK: [[S:%.*]] = shard.sharding @mesh4 split_axes = {{\[\[}}0]] : !shard.sharding
    %s = shard.sharding @mesh4 split_axes = [[0]] : !shard.sharding
    // CHECK: shard.shard %arg0 to [[S]] : tensor<1024x1024xi64>
    %0 = shard.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK: shard.sharding @mesh4 split_axes = {{\[\[}}0]] sharded_dims_offsets = [0, 1, 2, 3, 4] : !shard.sharding
    %1 = ndarray.subview %0[1, 0][4, 3][256, 1] : tensor<1024x1024xi64> to tensor<4x3xi64>
    return %1 : tensor<4x3xi64>
}

// CHECK-LABEL: @test_shard_propagate_subview_leading
func.func @test_shard_propagate_subview_leading(%arg0: tensor<1024x1024xi64>) -> tensor<3x3xi64> {
    // CHECK: [[S:%.*]] = shard.sharding @mesh4 split_axes = {{\[\[}}0]] : !shard.sharding
    %s = shard.sharding @mesh4 split_axes = [[0]] : !shard.sharding
    // CHECK: shard.shard %arg0 to [[S]] : tensor<1024x1024xi64>
    %0 = shard.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK: shard.sharding @mesh4 split_axes = {{\[\[}}0]] sharded_dims_offsets = [0, 3, 3, 3, 3] : !shard.sharding
    %1 = ndarray.subview %0[0, 0][3, 3][3, 1] : tensor<1024x1024xi64> to tensor<3x3xi64>
    return %1 : tensor<3x3xi64>
}

// CHECK-LABEL: @test_shard_propagate_subview_mid
func.func @test_shard_propagate_subview_mid(%arg0: tensor<1024x1024xi64>) -> tensor<3x3xi64> {
    // CHECK: [[S:%.*]] = shard.sharding @mesh4 split_axes = {{\[\[}}0]] : !shard.sharding
    %s = shard.sharding @mesh4 split_axes = [[0]] : !shard.sharding
    // CHECK: shard.shard %arg0 to [[S]] : tensor<1024x1024xi64>
    %0 = shard.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK: shard.sharding @mesh4 split_axes = {{\[\[}}0]] sharded_dims_offsets = [0, 0, 1, 3, 3] : !shard.sharding
    %1 = ndarray.subview %0[511, 0][3, 3][1, 1] : tensor<1024x1024xi64> to tensor<3x3xi64>
    return %1 : tensor<3x3xi64>
}

// CHECK-LABEL: @test_shard_propagate_subview_trailing
func.func @test_shard_propagate_subview_trailing(%arg0: tensor<1024x1024xi64>) -> tensor<3x3xi64> {
    // CHECK: [[S:%.*]] = shard.sharding @mesh4 split_axes = {{\[\[}}0]] : !shard.sharding
    %s = shard.sharding @mesh4 split_axes = [[0]] : !shard.sharding
    // CHECK: shard.shard %arg0 to [[S]] : tensor<1024x1024xi64>
    %0 = shard.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK: shard.sharding @mesh4 split_axes = {{\[\[}}0]] sharded_dims_offsets = [0, 0, 0, 0, 3] : !shard.sharding
    %1 = ndarray.subview %0[1000, 0][3, 3][1, 1] : tensor<1024x1024xi64> to tensor<3x3xi64>
    return %1 : tensor<3x3xi64>
}

// CHECK-LABEL: @test_shard_propagate_subview_gap
func.func @test_shard_propagate_subview_gap(%arg0: tensor<1024x1024xi64>) -> tensor<3x3xi64> {
    // CHECK: [[S:%.*]] = shard.sharding @mesh4 split_axes = {{\[\[}}0]] : !shard.sharding
    %s = shard.sharding @mesh4 split_axes = [[0]] : !shard.sharding
    // CHECK: shard.shard %arg0 to [[S]] : tensor<1024x1024xi64>
    %0 = shard.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK: shard.sharding @mesh4 split_axes = {{\[\[}}0]] sharded_dims_offsets = [0, 1, 1, 2, 3] : !shard.sharding
    %1 = ndarray.subview %0[255, 0][3, 3][257, 1] : tensor<1024x1024xi64> to tensor<3x3xi64>
    return %1 : tensor<3x3xi64>
}

// CHECK-LABEL: @test_shard_propagate_insert_slice
func.func @test_shard_propagate_insert_slice(%arg0: tensor<1024x1024xi64>, %arg1: tensor<3x3xi64>) {
    %s = shard.sharding @mesh4 split_axes = [[0]] : !shard.sharding
    %0 = shard.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK: %[[sharding_2:.*]] = shard.sharding @mesh4 split_axes = {{\[\[}}0]] sharded_dims_offsets = [0, 3, 3, 3, 3] : !shard.sharding
    // CHECK: %[[sharding_annotated_1:.*]] = shard.shard %arg1 to %[[sharding_2]] annotate_for_users : tensor<3x3xi64>
    // CHECK: ndarray.insert_slice %[[sharding_annotated_1]] into
    ndarray.insert_slice %arg1 into %0[0, 0][3, 3][1, 1] : tensor<3x3xi64> into tensor<1024x1024xi64>
    return
}

shard.grid @mesh4x4(shape = 4x4)

// CHECK-LABEL: @test_shard_propagate_insert_slice_2d
func.func @test_shard_propagate_insert_slice_2d(%arg0: tensor<1024x1024xi64>, %arg1: tensor<3x3xi64>) {
    %s = shard.sharding @mesh4x4 split_axes = [[0], [1]] : !shard.sharding
    %0 = shard.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK: %[[sharding_2:.*]] = shard.sharding @mesh4x4 split_axes = {{\[\[}}0], [1]] sharded_dims_offsets = [0, 3, 3, 3, 3, 0, 1, 1, 2, 3] : !shard.sharding
    // CHECK: %[[sharding_annotated_1:.*]] = shard.shard %arg1 to %[[sharding_2]] annotate_for_users : tensor<3x3xi64>
    // CHECK: ndarray.insert_slice %[[sharding_annotated_1]] into
    ndarray.insert_slice %arg1 into %0[0, 255][3, 3][1, 257] : tensor<3x3xi64> into tensor<1024x1024xi64>
    return
}

// CHECK-LABEL: @test_shard_propagate_insert_slice_2d_2
func.func @test_shard_propagate_insert_slice_2d_2(%arg0: tensor<1024x1024xi64>, %arg1: tensor<600x3xi64>) {
    %s = shard.sharding @mesh4x4 split_axes = [[0], [1]] : !shard.sharding
    %0 = shard.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK: %[[sharding_2:.*]] = shard.sharding @mesh4x4 split_axes = {{\[\[}}0], [1]] sharded_dims_offsets = [0, 156, 412, 600, 600, 0, 1, 1, 2, 3] : !shard.sharding
    // CHECK: %[[sharding_annotated_1:.*]] = shard.shard %arg1 to %[[sharding_2]] annotate_for_users : tensor<600x3xi64>
    // CHECK: ndarray.insert_slice %[[sharding_annotated_1]] into
    ndarray.insert_slice %arg1 into %0[100, 255][600, 3][1, 257] : tensor<600x3xi64> into tensor<1024x1024xi64>
    return
}

}
