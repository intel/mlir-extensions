// RUN: imex-opt --split-input-file --ptensor-dist %s -verify-diagnostics -o -| FileCheck %s

// -----
func.func @test_arange(%arg0: i64, %arg1: i64, %arg2: i64) -> i64 {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %0 = "ptensor.arange"(%arg0, %arg1, %arg2, %c0, %c1) : (i64, i64, i64, i64, i64) -> !ptensor.ptensor<tensor<?xi64>>
    %1 ="ptensor.ewbin"(%0, %0) {op = 0 : i32} : (!ptensor.ptensor<tensor<?xi64>>, !ptensor.ptensor<tensor<?xi64>>) -> !ptensor.ptensor<tensor<?xi64>>
    %2 = builtin.unrealized_conversion_cast %1 : !ptensor.ptensor<tensor<?xi64>> to i64
    return %2 : i64
}
// CHECK: [[Vc0_i64:%.+]] = arith.constant 0 : i64
// CHECK: [[Vc1_i64:%.+]] = arith.constant 1 : i64
// CHECK: [[Vcm1_i64:%.+]] = arith.constant -1 : i64
// CHECK: [[Vc0:%.+]] = arith.constant 0 : index
// CHECK: [[V0:%.+]] = shape.const_shape [-1] : tensor<1xindex>
// CHECK: [[Vfalse:%.+]] = arith.constant false
// CHECK: [[V1:%.+]] = arith.cmpi ult, %arg2, [[Vc0_i64]] : i64
// CHECK: [[V2:%.+]] = arith.select [[V1]], [[Vc1_i64]], [[Vcm1_i64]] : i64
// CHECK: [[V3:%.+]] = arith.addi %arg1, %arg2 : i64
// CHECK: [[V4:%.+]] = arith.addi [[V3]], [[V2]] : i64
// CHECK: [[V5:%.+]] = arith.subi [[V4]], %arg0 : i64
// CHECK: [[V6:%.+]] = arith.divui [[V5]], %arg2 : i64
// CHECK: [[V7:%.+]] = arith.index_cast [[V6]] : i64 to index
// CHECK: [[V8:%.+]] = tensor.empty([[V7]]) : tensor<?xi64>
// CHECK: [[V9:%.+]] = shape.shape_of [[V8]] : tensor<?xi64> -> tensor<1xindex>
// CHECK: [[V10:%.+]] = "dist.distinfo"([[V9]], [[Vc1_i64]]) {rank = 1 : i64} : (tensor<1xindex>, i64) -> !dist.info<1>
// CHECK: [[V11:%.+]] = "dist.extract_from_info"([[V10]]) {what = 1 : i32} : (!dist.info<1>) -> tensor<1xi64>
// CHECK: [[Vextracted:%.+]] = tensor.extract [[V11]][[[Vc0]]] : tensor<1xi64>
// CHECK: [[V12:%.+]] = "dist.extract_from_info"([[V10]]) {what = 2 : i32} : (!dist.info<1>) -> tensor<1xi64>
// CHECK: [[Vextracted_0:%.+]] = tensor.extract [[V12]][[[Vc0]]] : tensor<1xi64>
// CHECK: [[V13:%.+]] = arith.muli [[Vextracted_0]], %arg2 : i64
// CHECK: [[V14:%.+]] = arith.addi %arg0, [[V13]] : i64
// CHECK: [[V15:%.+]] = arith.muli [[Vextracted]], %arg2 : i64
// CHECK: [[V16:%.+]] = arith.addi [[V14]], [[V15]] : i64
// CHECK: [[V17:%.+]] = "ptensor.arange"([[V14]], [[V16]], %arg2, [[Vc0_i64]]) : (i64, i64, i64, i64) -> !ptensor.ptensor<tensor<?xi64>>
// CHECK: [[V18:%.+]] = "dist.init_dist_tensor"([[V17]], [[V10]]) : (!ptensor.ptensor<tensor<?xi64>>, !dist.info<1>) -> !dist.dtensor<<tensor<?xi64>>>
// CHECK: [[V19:%.+]] = "dist.get_ptensor"([[V18]]) : (!dist.dtensor<<tensor<?xi64>>>) -> !ptensor.ptensor<tensor<?xi64>>
// CHECK: [[V20:%.+]] = "ptensor.extract_rtensor"([[V19]]) : (!ptensor.ptensor<tensor<?xi64>>) -> tensor<?xi64>
// CHECK: [[V21:%.+]] = "ptensor.init_ptensor"([[V20]], [[Vfalse]]) : (tensor<?xi64>, i1) -> !ptensor.ptensor<tensor<?xi64>>
// CHECK: [[V22:%.+]] = "dist.get_ptensor"([[V18]]) : (!dist.dtensor<<tensor<?xi64>>>) -> !ptensor.ptensor<tensor<?xi64>>
// CHECK: [[V23:%.+]] = "ptensor.extract_rtensor"([[V22]]) : (!ptensor.ptensor<tensor<?xi64>>) -> tensor<?xi64>
// CHECK: [[V24:%.+]] = "ptensor.init_ptensor"([[V23]], [[Vfalse]]) : (tensor<?xi64>, i1) -> !ptensor.ptensor<tensor<?xi64>>
// CHECK: [[V25:%.+]] = "ptensor.ewbin"([[V21]], [[V24]]) {op = 0 : i32} : (!ptensor.ptensor<tensor<?xi64>>, !ptensor.ptensor<tensor<?xi64>>) -> !ptensor.ptensor<tensor<?xi64>>
// CHECK: [[V26:%.+]] = "dist.get_info"([[V18]]) : (!dist.dtensor<<tensor<?xi64>>>) -> !dist.info<1>
// CHECK: [[V27:%.+]] = "dist.extract_from_info"([[V26]]) {what = 3 : i32} : (!dist.info<1>) -> i64
// CHECK: [[V28:%.+]] = "dist.distinfo"([[V0]], [[V27]]) {rank = 1 : i64} : (tensor<1xindex>, i64) -> !dist.info<1>
// CHECK: [[V29:%.+]] = "dist.init_dist_tensor"([[V25]], [[V28]]) : (!ptensor.ptensor<tensor<?xi64>>, !dist.info<1>) -> !dist.dtensor<<tensor<?xi64>>>
// CHECK: [[V30:%.+]] = builtin.unrealized_conversion_cast [[V29]] : !dist.dtensor<<tensor<?xi64>>> to i64
