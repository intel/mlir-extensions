// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s

// -----
func.func @test_copy_reshape(%arg0: tensor<?x?xi64>) {
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c9 = arith.constant 9 : index
    %h, %a = distruntime.copy_reshape %arg0 g_shape %c3, %c3 l_offs %c1, %c1 to n_g_shape %c9 n_offs %c3 n_shape %c3 {team=22 : i64} : (tensor<?x?xi64>, index, index, index, index, index, index, index) -> (!distruntime.asynchandle, tensor<?xi64>)
    return
}
// CHECK-LABEL: func.func @test_copy_reshape(%arg0: tensor<?x?xi64>) {
// CHECK: distruntime.copy_reshape %arg0 g_shape %c3, %c3 l_offs %c1, %c1 to n_g_shape %c9 n_offs %c3 n_shape %c3 {team = 22 : i64} : (tensor<?x?xi64>, index, index, index, index, index, index, index) -> (!distruntime.asynchandle, tensor<?xi64>)

// -----
func.func @test_copy_permute(%arg0: tensor<5x2xi64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %h, %a = distruntime.copy_permute %arg0 g_shape %c5, %c2 l_offs %c0, %c0 to n_offs %c0, %c0 n_shape %c2, %c5 axes [1, 0] {team=22 : i64} : (tensor<5x2xi64>, index, index, index, index, index, index, index, index) -> (!distruntime.asynchandle, tensor<2x5xi64>)
    return
}
// CHECK-LABEL: func.func @test_copy_permute(%arg0: tensor<5x2xi64>) {
// CHECK: distruntime.copy_permute %arg0 g_shape %c5, %c2 l_offs %c0, %c0 to n_offs %c0, %c0 n_shape %c2, %c5 axes [1, 0] {team = 22 : i64} : (tensor<5x2xi64>, index, index, index, index, index, index, index, index) -> (!distruntime.asynchandle, tensor<2x5xi64>)
