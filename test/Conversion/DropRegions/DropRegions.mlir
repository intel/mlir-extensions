// RUN: imex-opt --split-input-file --drop-regions %s -verify-diagnostics -o -| FileCheck %s

// -----
func.func @test_region() -> !ndarray.ndarray<?x?x?xf64> {
    %r = region.env_region #region.gpu_env<device = "test"> -> !ndarray.ndarray<?x?x?xf64> {
        %c3 = arith.constant 3 : index
        %0 = ndarray.create %c3, %c3, %c3 {dtype = 0 : i8} : (index, index, index) -> !ndarray.ndarray<?x?x?xf64>
        region.env_region_yield %0 : !ndarray.ndarray<?x?x?xf64>
    }
    return %r : !ndarray.ndarray<?x?x?xf64>
}
// CHECK-LABEL: func.func @test_region
// CHECK-NEXT: [[C3:%.*]] = arith.constant 3
// CHECK-NEXT: [[RV:%.*]] = ndarray.create [[C3]], [[C3]], [[C3]]
// CHECK-NEXT: return [[RV]]
