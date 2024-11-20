// RUN: imex-opt --split-input-file --drop-regions %s -verify-diagnostics -o -| FileCheck %s

// -----
func.func @test_region() -> tensor<?x?x?xf64> {
    %r = region.env_region #region.gpu_env<device = "test"> -> tensor<?x?x?xf64> {
        %c3 = arith.constant 3 : index
        %0 = tensor.empty(%c3, %c3, %c3) : tensor<?x?x?xf64>
        region.env_region_yield %0 : tensor<?x?x?xf64>
    }
    return %r : tensor<?x?x?xf64>
}
// CHECK-LABEL: func.func @test_region
// CHECK-NEXT: [[C3:%.*]] = arith.constant 3
// CHECK-NEXT: [[RV:%.*]] = tensor.empty([[C3]], [[C3]], [[C3]]) : tensor<?x?x?xf64>
// CHECK-NEXT: return [[RV]]
