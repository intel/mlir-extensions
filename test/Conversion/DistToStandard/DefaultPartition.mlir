// RUN: imex-opt --split-input-file --convert-dist-to-standard -canonicalize %s -verify-diagnostics -o -| FileCheck %s

func.func @test_def_part() -> (index, index, index, index, index, index, index, index, index, index, index, index) {
    %NP = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c7 = arith.constant 7 : index
    %c32 = arith.constant 32 : index
    %c33 = arith.constant 33 : index
    %c31 = arith.constant 31 : index
    %o0, %s0 = "dist.default_partition"(%NP, %c7, %c31) : (index, index, index) -> (index, index)
    %o1, %s1 = "dist.default_partition"(%NP, %c7, %c32) : (index, index, index) -> (index, index)
    %o2, %s2 = "dist.default_partition"(%NP, %c7, %c33) : (index, index, index) -> (index, index)
    %o3, %s3 = "dist.default_partition"(%NP, %c2, %c31) : (index, index, index) -> (index, index)
    %o4, %s4 = "dist.default_partition"(%NP, %c1, %c31) : (index, index, index) -> (index, index)
    %o5, %s5 = "dist.default_partition"(%NP, %c0, %c31) : (index, index, index) -> (index, index)
    return %o0, %s0, %o1, %s1, %o2, %s2, %o3, %s3, %o4, %s4, %o5, %s5 : index, index, index, index, index, index, index, index, index, index, index, index
}
// CHECK-LABEL: func.func @test_def_part()
// CHECK: return %c27, %c4, %c28, %c4, %c28, %c5, %c7, %c4, %c3, %c4, %c0, %c3

// -----
func.func @test_def_part2() -> (index, index, index, index, index, index, index, index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %o0:2, %s0:2 = "dist.default_partition"(%c2, %c1, %c8, %c8) : (index, index, index, index) -> (index, index, index, index)
    %o1:2, %s1:2 = "dist.default_partition"(%c2, %c0, %c1, %c7) : (index, index, index, index) -> (index, index, index, index)
    return %o0#0, %o0#1, %s0#0, %s0#1, %o1#0, %o1#1, %s1#0, %s1#1 : index, index, index, index, index, index, index, index
}
// CHECK-LABEL: func.func @test_def_part2()
// CHECK: return %c4, %c0, %c4, %c8, %c0, %c0, %c0, %c7
