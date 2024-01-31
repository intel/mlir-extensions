// RUN: imex-opt --split-input-file --convert-dist-to-standard -canonicalize %s -verify-diagnostics -o -| FileCheck %s

func.func @test_bb() -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index

    %o1, %s1 = dist.local_bounding_box false[%c0] [%c8] [%c1] [%c4] [%c4] : index, index
    %o2, %s2 = dist.local_bounding_box false[%c1] [%c7] [%c1] [%c4] [%c4] : index, index
    %o3, %s3 = dist.local_bounding_box false[%c0] [%c8] [%c1] [%c4] [%c0] : index, index
    %o4, %s4 = dist.local_bounding_box false[%c1] [%c7] [%c1] [%c0] [%c0] : index, index
    %o5, %s5 = dist.local_bounding_box false[%c1] [%c7] [%c1] [%c2] [%c2] : index, index
    %o6, %s6 = dist.local_bounding_box false[%c0] [%c0] [%c1] [%c4] [%c4] : index, index
    %o7, %s7 = dist.local_bounding_box false[%c1] [%c3] [%c1] [%c0] [%c4] : index, index

    return %o1, %s1, %o2, %s2, %o3, %s3, %o4, %s4, %o5, %s5, %o6, %s6, %o7, %s7 : index, index, index, index, index, index, index, index, index, index, index, index, index, index
}
// CHECK-LABEL: func.func @test_bb
// CHECK: return %c4, %c4, %c5, %c3, %c4, %c0, %c1, %c0, %c3, %c2, %c4, %c0, %c1, %c3

// -----
func.func @test_bb2() -> (index, index, index, index, index, index, index, index, index, index, index, index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index

    %o1, %s1 = dist.local_bounding_box false[%c1] [%c7] [%c1] [%c4] [%c4] bboffs %c4 bb_sizes %c4 : index, index
    %o2, %s2 = dist.local_bounding_box false[%c0] [%c8] [%c1] [%c4] [%c0] bboffs %o1 bb_sizes %s1 : index, index
    %o3, %s3 = dist.local_bounding_box false[%c1] [%c3] [%c1] [%c0] [%c4] bboffs %c0 bb_sizes %c5 : index, index
    %o4, %s4 = dist.local_bounding_box false[%c1] [%c7] [%c1] [%c2] [%c2] bboffs %c1 bb_sizes %c0 : index, index
    %o5, %s5 = dist.local_bounding_box false[%c0] [%c1] [%c1] [%c0] [%c1] bboffs %o4 bb_sizes %s4 : index, index
    %o6, %s6 = dist.local_bounding_box false[%c7] [%c8] [%c1] [%c0] [%c1] bboffs %o5 bb_sizes %s5 : index, index

    return %o1, %s1, %o2, %s2, %o3, %s3, %o4, %s4, %o5, %s5, %o6, %s6 : index, index, index, index, index, index, index, index, index, index, index, index
}
// CHECK-LABEL: func.func @test_bb2
// CHECK: return %c4, %c4, %c4, %c4, %c0, %c5, %c3, %c2, %c0, %c5, %c0, %c8
