// RUN: imex-opt --split-input-file --dist-infer-elementwise-cores -canonicalize %s -verify-diagnostics -o -| FileCheck %s

module {
  func.func @test_infer() -> !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>> attributes {llvm.emit_c_interface} {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1_i64 = arith.constant 1 : i64
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c2_i64 = arith.constant 2 : i64
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c0_i64 = arith.constant 0 : i64
  %0 = "distruntime.team_size"() {team = 94098061490592 : i64} : () -> index
  %1 = "distruntime.team_member"() {team = 94098061490592 : i64} : () -> index
  %l_offsets, %l_shape = "dist.default_partition"(%0, %1, %c16) : (index, index, index) -> (index, index)
  %2 = ndarray.create %l_shape value %c0_i64 {dtype = 2 : i8} : (index, i64) -> !ndarray.ndarray<?xi64>
  %3 = ndarray.create %c0 {dtype = 2 : i8} : (index) -> !ndarray.ndarray<0xi64>
  %4 = ndarray.cast %3 : !ndarray.ndarray<0xi64> to !ndarray.ndarray<?xi64>
  %5 = dist.init_dist_array l_offset %l_offsets parts %4, %2, %4 : index, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64> to !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>
  %t_offsets, %t_sizes = dist.local_target_of_slice %5[%c3] [%c10] [%c1] : !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>> to index, index
  %result_offsets, %result_sizes = dist.local_bounding_box false [%c0] [%c10] [%c1] [%t_offsets] [%t_sizes] : index, index
  %result_offsets_0, %result_sizes_1 = dist.local_bounding_box false [%c1] [%c10] [%c1] [%t_offsets] [%t_sizes] bboffs %result_offsets bb_sizes %result_sizes : index, index
  %result_offsets_2, %result_sizes_3 = dist.local_bounding_box false [%c2] [%c10] [%c1] [%t_offsets] [%t_sizes] bboffs %result_offsets_0 bb_sizes %result_sizes_1 : index, index
  %result_offsets_4, %result_sizes_5 = dist.local_bounding_box false [%c3] [%c10] [%c1] [%t_offsets] [%t_sizes] bboffs %result_offsets_2 bb_sizes %result_sizes_3 : index, index
  %6 = dist.repartition %5 loffs %result_offsets_4 lsizes %result_sizes_5 : !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>, index, index to !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>
  %7 = dist.subview %6[%c0] [10] [%c1] toffs %t_offsets tsizes %t_sizes : !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>> to !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>
  %8 = dist.subview %6[%c1] [10] [%c1] toffs %t_offsets tsizes %t_sizes : !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>> to !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>
  %9 = dist.subview %6[%c2] [10] [%c1] toffs %t_offsets tsizes %t_sizes : !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>> to !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>
  %10 = dist.subview %6[%c3] [10] [%c1] toffs %t_offsets tsizes %t_sizes : !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>> to !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>
  %11 = ndarray.create  value %c2_i64 {dtype = 2 : i8} : (i64) -> !ndarray.ndarray<i64>
  %12 = dist.init_dist_array parts %11 : !ndarray.ndarray<i64> to !ndarray.ndarray<i64, #dist.dist_env<team = 0>>
  %13 = "dist.ewbin"(%7, %12) {op = 21 : i32} : (!ndarray.ndarray<10xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>, !ndarray.ndarray<i64, #dist.dist_env<team = 0>>) -> !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>
  %14 = "dist.ewbin"(%13, %8) {op = 0 : i32} : (!ndarray.ndarray<10xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>, !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>) -> !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>
  %15 = "dist.ewbin"(%14, %9) {op = 0 : i32} : (!ndarray.ndarray<10xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>, !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>) -> !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>
  %16 = ndarray.create  value %c1_i64 {dtype = 2 : i8} : (i64) -> !ndarray.ndarray<i64>
  %17 = dist.init_dist_array parts %16 : !ndarray.ndarray<i64> to !ndarray.ndarray<i64, #dist.dist_env<team = 0>>
  %18 = "dist.ewbin"(%10, %17) {op = 21 : i32} : (!ndarray.ndarray<10xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>, !ndarray.ndarray<i64, #dist.dist_env<team = 0>>) -> !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>
  %19 = "dist.ewbin"(%15, %18) {op = 0 : i32} : (!ndarray.ndarray<10xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>, !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>) -> !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>
  ndarray.insert_slice %19 into %5[%c3] [%c10] [%c1] : !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>> into !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>
  %20 = "ndarray.cast"(%5) : (!ndarray.ndarray<16xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>) -> !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>
  return %20 : !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 loffs = ? lparts = ?,?,?>>
  }
}
// CHECK-LABEL:  func.func @test_infer()
// CHECK: [[vc10:%.*]] = arith.constant 10 : index
// CHECK: [[vc0:%.*]] = arith.constant 0 : index
// CHECK: [[vc1_i64:%.*]] = arith.constant 1 : i64
// CHECK: [[vc3:%.*]] = arith.constant 3 : index
// CHECK: [[vc2:%.*]] = arith.constant 2 : index
// CHECK: [[vc2_i64:%.*]] = arith.constant 2 : i64
// CHECK: [[vc1:%.*]] = arith.constant 1 : index
// CHECK: [[vc16:%.*]] = arith.constant 16 : index
// CHECK: [[vc0_i64:%.*]] = arith.constant 0 : i64
// CHECK: [[v0:%.*]] = "distruntime.team_size"() <{team = 94098061490592 : i64}> : () -> index
// CHECK: [[v1:%.*]] = "distruntime.team_member"() <{team = 94098061490592 : i64}> : () -> index
// CHECK: [[vl_offsets:%.*]], [[vl_shape:%.*]] = "dist.default_partition"([[v0]], [[v1]], [[vc16]]) : (index, index, index) -> (index, index)
// CHECK: [[v2:%.*]] = ndarray.create [[vl_shape]] value [[vc0_i64]] {dtype = 2 : i8} : (index, i64) -> !ndarray.ndarray<?xi64>
// CHECK: [[v3:%.*]] = ndarray.create [[vc0]] {dtype = 2 : i8} : (index) -> !ndarray.ndarray<0xi64>
// CHECK: [[v4:%.*]] = ndarray.cast [[v3]] : !ndarray.ndarray<0xi64> to !ndarray.ndarray<?xi64>
// CHECK: [[v5:%.*]] = dist.init_dist_array l_offset [[vl_offsets]] parts [[v4]], [[v2]], [[v4]] : index, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64> to !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>>
// CHECK: [[vt_offsets:%.*]], [[vt_sizes:%.*]] = dist.local_target_of_slice [[v5]][[[vc3]]] [[[vc10]]] [[[vc1]]] : !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>> to index, index
// CHECK: [[vresultOffsets:%.*]], [[vresultSizes:%.*]] = dist.local_core [[v5]] toffs [[vt_offsets]] tsizes [[vt_sizes]] soffs %c{{[0-9]}} ssizes [[vc10]] sstrides [[vc1]] : !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>> to index, index
// CHECK: [[vresultOffsets_0:%.*]], [[vresultSizes_1:%.*]] = dist.local_core [[v5]] toffs [[vt_offsets]] tsizes [[vt_sizes]] soffs %c{{[0-9]}} ssizes [[vc10]] sstrides [[vc1]] coffs [[vresultOffsets]] csizes [[vresultSizes]] : !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>> to index, index
// CHECK: [[vresultOffsets_2:%.*]], [[vresultSizes_3:%.*]] = dist.local_core [[v5]] toffs [[vt_offsets]] tsizes [[vt_sizes]] soffs %c{{[0-9]}} ssizes [[vc10]] sstrides [[vc1]] coffs [[vresultOffsets_0]] csizes [[vresultSizes_1]] : !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>> to index, index
// CHECK: [[vresultOffsets_4:%.*]], [[vresultSizes_5:%.*]] = dist.local_core [[v5]] toffs [[vt_offsets]] tsizes [[vt_sizes]] soffs %c{{[0-9]}} ssizes [[vc10]] sstrides [[vc1]] coffs [[vresultOffsets_2]] csizes [[vresultSizes_3]] : !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>> to index, index
// CHECK: [[vresult_offsets:%.*]], [[vresult_sizes:%.*]] = dist.local_bounding_box false[[[vc0]]] [[[vc10]]] [[[vc1]]] [[[vt_offsets]]] [[[vt_sizes]]] : index, index
// CHECK: [[vresult_offsets_6:%.*]], [[vresult_sizes_7:%.*]] = dist.local_bounding_box false[[[vc1]]] [[[vc10]]] [[[vc1]]] [[[vt_offsets]]] [[[vt_sizes]]] bboffs [[vresult_offsets]] bb_sizes [[vresult_sizes]] : index, index
// CHECK: [[vresult_offsets_8:%.*]], [[vresult_sizes_9:%.*]] = dist.local_bounding_box false[[[vc2]]] [[[vc10]]] [[[vc1]]] [[[vt_offsets]]] [[[vt_sizes]]] bboffs [[vresult_offsets_6]] bb_sizes [[vresult_sizes_7]] : index, index
// CHECK: [[vresult_offsets_10:%.*]], [[vresult_sizes_11:%.*]] = dist.local_bounding_box false[[[vc3]]] [[[vc10]]] [[[vc1]]] [[[vt_offsets]]] [[[vt_sizes]]] bboffs [[vresult_offsets_8]] bb_sizes [[vresult_sizes_9]] : index, index
// CHECK: [[v6:%.*]] = dist.repartition [[v5]] loffs [[vresult_offsets_10]] lsizes [[vresult_sizes_11]] : !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>>, index, index to !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>>
// CHECK: [[v7:%.*]] = dist.subview [[v6]][[[vc0]]] [10] [[[vc1]]] toffs [[vt_offsets]] tsizes [[vt_sizes]] : !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>> to !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>>
// CHECK: [[v8:%.*]] = dist.subview [[v6]][[[vc1]]] [10] [[[vc1]]] toffs [[vt_offsets]] tsizes [[vt_sizes]] : !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>> to !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>>
// CHECK: [[v9:%.*]] = dist.subview [[v6]][[[vc2]]] [10] [[[vc1]]] toffs [[vt_offsets]] tsizes [[vt_sizes]] : !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>> to !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>>
// CHECK: [[v10:%.*]] = dist.subview [[v6]][[[vc3]]] [10] [[[vc1]]] toffs [[vt_offsets]] tsizes [[vt_sizes]] : !ndarray.ndarray<16xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>> to !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>>
// CHECK: [[v11:%.*]] = ndarray.create value [[vc2_i64]] {dtype = 2 : i8} : (i64) -> !ndarray.ndarray<i64>
// CHECK: [[v12:%.*]] = dist.init_dist_array parts [[v11]] : !ndarray.ndarray<i64> to !ndarray.ndarray<i64, #dist.dist_env<team = 0 : i64>>
// CHECK: [[v13:%.*]] = "dist.ewbin"([[v7]], [[v12]], [[vresultOffsets_4]], [[vresultSizes_5]], [[vt_offsets]]) <{op = 21 : i32}> : (!ndarray.ndarray<10xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>>, !ndarray.ndarray<i64, #dist.dist_env<team = 0 : i64>>, index, index, index) -> !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>>
// CHECK: [[v14:%.*]] = "dist.ewbin"([[v13]], [[v8]], [[vresultOffsets_4]], [[vresultSizes_5]], [[vt_offsets]]) <{op = 0 : i32}> : (!ndarray.ndarray<10xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>>, !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>>, index, index, index) -> !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>>
// CHECK: [[v15:%.*]] = "dist.ewbin"([[v14]], [[v9]], [[vresultOffsets_4]], [[vresultSizes_5]], [[vt_offsets]]) <{op = 0 : i32}> : (!ndarray.ndarray<10xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>>, !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>>, index, index, index) -> !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>>
// CHECK: [[v16:%.*]] = ndarray.create value [[vc1_i64]] {dtype = 2 : i8} : (i64) -> !ndarray.ndarray<i64>
// CHECK: [[v17:%.*]] = dist.init_dist_array parts [[v16]] : !ndarray.ndarray<i64> to !ndarray.ndarray<i64, #dist.dist_env<team = 0 : i64>>
// CHECK: [[v18:%.*]] = "dist.ewbin"([[v10]], [[v17]], [[vresultOffsets_4]], [[vresultSizes_5]], [[vt_offsets]]) <{op = 21 : i32}> : (!ndarray.ndarray<10xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>>, !ndarray.ndarray<i64, #dist.dist_env<team = 0 : i64>>, index, index, index) -> !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>>
// CHECK: [[v19:%.*]] = "dist.ewbin"([[v15]], [[v18]], [[vresultOffsets_4]], [[vresultSizes_5]], [[vt_offsets]]) <{op = 0 : i32}> : (!ndarray.ndarray<10xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>>, !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>>, index, index, index) -> !ndarray.ndarray<10xi64, #dist.dist_env<team = 0 : i64 loffs = ? lparts = ?,?,?>>
