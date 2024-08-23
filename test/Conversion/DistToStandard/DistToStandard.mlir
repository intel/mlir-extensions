// RUN: imex-opt --split-input-file --convert-dist-to-standard %s -verify-diagnostics -o -| FileCheck %s

func.func @test_copy(%arg0: !ndarray.ndarray<2xi64>, %arg1: !ndarray.ndarray<6xi64>, %arg2: !ndarray.ndarray<0xi64>) -> (!ndarray.ndarray<2xi64>, !ndarray.ndarray<6xi64>, !ndarray.ndarray<0xi64>) {
  %c1 = arith.constant 1 : index
  %a = dist.init_dist_array l_offset %c1 parts %arg0, %arg1, %arg2 : index, !ndarray.ndarray<2xi64>, !ndarray.ndarray<6xi64>, !ndarray.ndarray<0xi64> to !ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>>
  %1 = ndarray.copy %a : !ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>> -> !ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>>
  %2 = ndarray.copy %1 : !ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>> -> !ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>, #region.gpu_env<device = "XeGPU">>
  %4 = ndarray.copy %2 : !ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>, #region.gpu_env<device = "XeGPU">> -> !ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>>
  %20, %21, %22 = "dist.parts_of"(%4) : (!ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>>) -> (!ndarray.ndarray<2xi64>, !ndarray.ndarray<6xi64>, !ndarray.ndarray<0xi64>)
  return %20, %21, %22 : !ndarray.ndarray<2xi64>, !ndarray.ndarray<6xi64>, !ndarray.ndarray<0xi64>
}
// CHECK-LABEL: func.func @test_copy
// CHECK-SAME: [[arg0:%.*]]: !ndarray.ndarray<2xi64>, [[arg1:%.*]]: !ndarray.ndarray<6xi64>, [[arg2:%.*]]: !ndarray.ndarray<0xi64>
// CHECK: [[v1:%.*]] = ndarray.copy [[arg0]] : !ndarray.ndarray<2xi64> -> !ndarray.ndarray<2xi64>
// CHECK-NEXT: [[v2:%.*]] = ndarray.copy [[arg1]] : !ndarray.ndarray<6xi64> -> !ndarray.ndarray<6xi64>
// CHECK-NEXT: [[v3:%.*]] = ndarray.copy [[arg2]] : !ndarray.ndarray<0xi64> -> !ndarray.ndarray<0xi64>
// CHECK-NEXT: [[v4:%.*]] = ndarray.copy [[v1]] : !ndarray.ndarray<2xi64> -> !ndarray.ndarray<2xi64, #region.gpu_env<device = "XeGPU">>
// CHECK-NEXT: [[v5:%.*]] = ndarray.copy [[v2]] : !ndarray.ndarray<6xi64> -> !ndarray.ndarray<6xi64, #region.gpu_env<device = "XeGPU">>
// CHECK-NEXT: [[v6:%.*]] = ndarray.copy [[v3]] : !ndarray.ndarray<0xi64> -> !ndarray.ndarray<0xi64, #region.gpu_env<device = "XeGPU">>
// CHECK-NEXT: [[v7:%.*]] = ndarray.copy [[v4]] : !ndarray.ndarray<2xi64, #region.gpu_env<device = "XeGPU">> -> !ndarray.ndarray<2xi64>
// CHECK-NEXT: [[v8:%.*]] = ndarray.copy [[v5]] : !ndarray.ndarray<6xi64, #region.gpu_env<device = "XeGPU">> -> !ndarray.ndarray<6xi64>
// CHECK-NEXT: [[v9:%.*]] = ndarray.copy [[v6]] : !ndarray.ndarray<0xi64, #region.gpu_env<device = "XeGPU">> -> !ndarray.ndarray<0xi64>

// -----
func.func @test_delete(%arg0: !ndarray.ndarray<2xi64>, %arg1: !ndarray.ndarray<6xi64>, %arg2: !ndarray.ndarray<0xi64>) {
  %c1 = arith.constant 1 : index
  %a = dist.init_dist_array l_offset %c1 parts %arg0, %arg1, %arg2 : index, !ndarray.ndarray<2xi64>, !ndarray.ndarray<6xi64>, !ndarray.ndarray<0xi64> to !ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>>
  ndarray.delete %a : !ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>>
  return
}
// CHECK-LABEL: func.func @test_delete
// CHECK: ndarray.delete
// CHECK-SAME: : !ndarray.ndarray<2xi64>
// CHECK-NEXT: ndarray.delete
// CHECK-SAME: : !ndarray.ndarray<6xi64>
// CHECK-NEXT: ndarray.delete
// CHECK-SAME: : !ndarray.ndarray<0xi64>

// -----
func.func @test_init_dist_array(%arg0: !ndarray.ndarray<2xi64>, %arg1: !ndarray.ndarray<6xi64>, %arg2: !ndarray.ndarray<0xi64>) -> (!ndarray.ndarray<2xi64>, !ndarray.ndarray<6xi64>, !ndarray.ndarray<0xi64>) {
  %c1 = arith.constant 1 : index
  %a = dist.init_dist_array l_offset %c1 parts %arg0, %arg1, %arg2 : index, !ndarray.ndarray<2xi64>, !ndarray.ndarray<6xi64>, !ndarray.ndarray<0xi64> to !ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>>
  %20, %21, %22 = "dist.parts_of"(%a) : (!ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>>) -> (!ndarray.ndarray<2xi64>, !ndarray.ndarray<6xi64>, !ndarray.ndarray<0xi64>)
  return %20, %21, %22 : !ndarray.ndarray<2xi64>, !ndarray.ndarray<6xi64>, !ndarray.ndarray<0xi64>
}
// CHECK-LABEL: func.func @test_init_dist_array
// CHECK-SAME: [[arg0:%.*]]: !ndarray.ndarray<2xi64>, [[arg1:%.*]]: !ndarray.ndarray<6xi64>, [[arg2:%.*]]: !ndarray.ndarray<0xi64>
// CHECK: return [[arg0]], [[arg1]], [[arg2]] : !ndarray.ndarray<2xi64>, !ndarray.ndarray<6xi64>, !ndarray.ndarray<0xi64>

// -----
func.func @test_local_partition(%np : index, %prank: index, %shape: index) -> (index, index) {
  %0, %1 = "dist.default_partition"(%np, %prank, %shape) {rank = 1 : i64} : (index, index, index) -> (index, index)
  return %0, %1 : index, index
}
// CHECK-LABEL: func.func @test_local_partition(%arg0: index, %arg1: index, %arg2: index) -> (index, index) {
// CHECK: arith.remsi
// CHECK: arith.divsi
// CHECK-DAG: arith.addi
// CHECK-DAG: arith.cmpi
// CHECK-DAG: arith.select
// CHECK-DAG: arith.addi
// CHECK-DAG: arith.subi
// CHECK-DAG: arith.subi
// CHECK-DAG: arith.maxsi
// CHECK-DAG: arith.muli
// CHECK: arith.addi
// CHECK: arith.maxsi


// -----
func.func @test_local_target_of_slice(%arg0: !ndarray.ndarray<?xi64>, %arg1: !ndarray.ndarray<?xi64>, %arg2: !ndarray.ndarray<?xi64>, %c0 : index, %c3 : index) -> (index, index) {
  %c1 = arith.constant 1 : index
  %a = dist.init_dist_array l_offset %c1 parts %arg0, %arg1, %arg2 : index, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64> to !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 1 lparts = ?,?,?>>
  %l_offsets, %l_sizes = dist.local_target_of_slice %a[%c0] [%c3] [%c3] : !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 1 lparts = ?,?,?>> to index, index
  return %l_offsets, %l_sizes : index, index
}
// CHECK-LABEL: func.func @test_local_target_of_slice
// CHECK arith.constant
// CHECK memref.load
// CHECK arith.constant
// CHECK arith.constant
// CHECK ndarray.dim
// CHECK arith.constant
// CHECK ndarray.dim
// CHECK arith.addi
// CHECK arith.constant
// CHECK ndarray.dim
// CHECK arith.addi
// CHECK arith.constant
// CHECK arith.constant
// CHECK arith.muli
// CHECK arith.addi
// CHECK arith.addi
// CHECK arith.maxsi
// CHECK arith.subi
// CHECK arith.addi
// CHECK arith.subi
// CHECK arith.divsi
// CHECK arith.muli
// CHECK arith.addi
// CHECK arith.minsi
// CHECK arith.addi
// CHECK arith.subi
// CHECK arith.maxsi
// CHECK arith.divsi
// CHECK arith.subi
// CHECK arith.divsi
// CHECK arith.minsi
// CHECK arith.constant
// CHECK: return

// -----
func.func @test_copy_reshape() -> () {
  %i1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index

  %0 = ndarray.create %c3, %c4 value %i1 {dtype = 4 : i8} : (index, index, i32) -> !ndarray.ndarray<3x4xi32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>>
  %1 = ndarray.reshape %0 %c2, %c3, %c2 {copy = true}
       : !ndarray.ndarray<3x4xi32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>>
       -> !ndarray.ndarray<2x3x2xi32, #dist.dist_env<team = 22 loffs = ?,?,? lparts = ?x?x?,?x?x?,?x?x?>>
  return
}
// CHECK-LABEL: @test_copy_reshape
// CHECK: "distruntime.team_size"() <{team = 22 : i64}> : () -> index
// CHECK: "distruntime.team_member"() <{team = 22 : i64}> : () -> index
// CHECK: [[handle:%.*]], [[nlArray:%.*]] = distruntime.copy_reshape
// CHECK: "distruntime.wait"([[handle]]) : (!distruntime.asynchandle) -> ()

// -----
func.func @test_repartition(%arg0: !ndarray.ndarray<?x?xi64>, %arg1: !ndarray.ndarray<?x?xi64>, %arg2: !ndarray.ndarray<?x?xi64>) -> (!ndarray.ndarray<?x?xi64>, !ndarray.ndarray<?x?xi64>, !ndarray.ndarray<?x?xi64>) {
  %c0 = arith.constant 0 : index
  %a = dist.init_dist_array l_offset %c0, %c0 parts %arg0, %arg1, %arg2 : index, index, !ndarray.ndarray<?x?xi64>, !ndarray.ndarray<?x?xi64>, !ndarray.ndarray<?x?xi64> to !ndarray.ndarray<10x12xi64, #dist.dist_env<team = 22 loffs = 0,0 lparts = ?x?,?x?,?x?>>
  %4 = dist.repartition %a : !ndarray.ndarray<10x12xi64, #dist.dist_env<team = 22 loffs = 0,0 lparts = ?x?,?x?,?x?>> to !ndarray.ndarray<10x12xi64, #dist.dist_env<team = 22 loffs = 0,0 lparts = ?x?,?x?,?x?>>
  %20, %21, %22 = "dist.parts_of"(%4) : (!ndarray.ndarray<10x12xi64, #dist.dist_env<team = 22 loffs = 0,0 lparts = ?x?,?x?,?x?>>) -> (!ndarray.ndarray<?x?xi64>, !ndarray.ndarray<?x?xi64>, !ndarray.ndarray<?x?xi64>)
  return %20, %21, %22 : !ndarray.ndarray<?x?xi64>, !ndarray.ndarray<?x?xi64>, !ndarray.ndarray<?x?xi64>
}
// CHECK-LABEL: @test_repartition
// CHECK: ndarray.dim
// CHECK: ndarray.dim
// CHECK: ndarray.dim
// CHECK: ndarray.dim
// CHECK: distruntime.get_halo

// -----
func.func @test_local_core(%arg0: !ndarray.ndarray<?xi64>, %arg1: !ndarray.ndarray<?xi64>, %arg2: !ndarray.ndarray<?xi64>, %arg3: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c8 = arith.constant 8 : index
  %a = dist.init_dist_array l_offset %arg3 parts %arg0, %arg1, %arg2 : index, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64> to !ndarray.ndarray<16xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>

  %resultOffsets, %resultSizes = dist.local_core %a toffs %c0 tsizes %c6 soffs %c0 ssizes %c8 sstrides %c1 : !ndarray.ndarray<16xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>> to index, index
  %resultOffsets_6, %resultSizes_7 = dist.local_core %a toffs %c0 tsizes %c6 soffs %c1 ssizes %c8 sstrides %c1 coffs %resultOffsets csizes %resultSizes : !ndarray.ndarray<16xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>> to index, index
  return
}
// CHECK-LABEL: func.func @test_local_core
// CHECK-SAME: [[arg0:%.*]]: !ndarray.ndarray<?xi64>, [[arg1:%.*]]: !ndarray.ndarray<?xi64>, [[arg2:%.*]]: !ndarray.ndarray<?xi64>, [[arg3:%.*]]: index
// CHECK: [[vc0:%.*]] = arith.constant 0 : index
// CHECK: [[vc0_0:%.*]] = arith.constant 0 : index
// CHECK: [[vc6:%.*]] = arith.constant 6 : index
// CHECK: [[vc8:%.*]] = arith.constant 8 : index
// CHECK: [[vc0_1:%.*]] = arith.constant 0 : index
// CHECK: [[vdim:%.*]] = ndarray.dim [[arg1]] [[vc0_1]] : !ndarray.ndarray<?xi64> -> index
// CHECK: [[vc0_2:%.*]] = arith.constant 0 : index
// CHECK: [[vdim_3:%.*]] = ndarray.dim [[arg0]] [[vc0_2]] : !ndarray.ndarray<?xi64> -> index
// CHECK: [[v0:%.*]] = arith.addi [[arg3]], [[vdim_3]] : index
// CHECK: [[vc0_4:%.*]] = arith.constant 0 : index
// CHECK: [[vc1:%.*]] = arith.constant 1 : index
// CHECK: [[v1:%.*]] = arith.addi [[arg3]], [[vdim]] : index
// CHECK: [[v2:%.*]] = arith.maxsi [[arg3]], [[vc0]] : index
// CHECK: [[vcm1:%.*]] = arith.constant -1 : index
// CHECK: [[v3:%.*]] = arith.addi [[v2]], [[vcm1]] : index
// CHECK: [[v4:%.*]] = arith.divsi [[v3]], [[vc0_0]] : index
// CHECK: [[v5:%.*]] = arith.minsi [[v1]], [[vc0]] : index
// CHECK: [[v6:%.*]] = arith.addi [[v5]], [[vcm1]] : index
// CHECK: [[v7:%.*]] = arith.maxsi [[v6]], [[vc0_4]] : index
// CHECK: [[v8:%.*]] = arith.divsi [[v7]], [[vc0_0]] : index
// CHECK: [[vc0_5:%.*]] = arith.constant 0 : index
// CHECK: [[v9:%.*]] = arith.divsi [[vc0_5]], [[vc0_0]] : index
// CHECK: [[v10:%.*]] = arith.minsi [[v9]], [[vc8]] : index
// CHECK: [[vc0_6:%.*]] = arith.constant 0 : index
// CHECK: [[v11:%.*]] = arith.subi [[vc0]], [[v10]] : index
// CHECK: [[v12:%.*]] = arith.subi [[vc0_6]], [[v11]] : index
// CHECK: [[v13:%.*]] = arith.maxsi [[v12]], [[vc0_6]] : index
// CHECK: [[v14:%.*]] = arith.subi [[v8]], [[v11]] : index
// CHECK: [[v15:%.*]] = arith.subi [[v14]], [[v13]] : index
// CHECK: [[v16:%.*]] = arith.subi [[vc6]], [[v13]] : index
// CHECK: [[vc6_7:%.*]] = arith.constant 6 : index
// CHECK: [[v17:%.*]] = arith.subi [[vc6_7]], [[v13]] : index
// CHECK: [[v18:%.*]] = arith.minsi [[v15]], [[v16]] : index
// CHECK: [[v19:%.*]] = arith.minsi [[v17]], [[v18]] : index
// CHECK: [[vc0_8:%.*]] = arith.constant 0 : index
// CHECK: [[vdim_9:%.*]] = ndarray.dim [[arg1]] [[vc0_8]] : !ndarray.ndarray<?xi64> -> index
// CHECK: [[vc0_10:%.*]] = arith.constant 0 : index
// CHECK: [[vdim_11:%.*]] = ndarray.dim [[arg0]] [[vc0_10]] : !ndarray.ndarray<?xi64> -> index
// CHECK: [[v20:%.*]] = arith.addi [[arg3]], [[vdim_11]] : index
// CHECK: [[vc0_12:%.*]] = arith.constant 0 : index
// CHECK: [[vc1_13:%.*]] = arith.constant 1 : index
// CHECK: [[v21:%.*]] = arith.addi [[arg3]], [[vdim_9]] : index
// CHECK: [[v22:%.*]] = arith.maxsi [[arg3]], [[vc0_0]] : index
// CHECK: [[vcm1_14:%.*]] = arith.constant -1 : index
// CHECK: [[v23:%.*]] = arith.addi [[v22]], [[vcm1_14]] : index
// CHECK: [[v24:%.*]] = arith.divsi [[v23]], [[vc0_0]] : index
// CHECK: [[v25:%.*]] = arith.minsi [[v21]], [[vc0_0]] : index
// CHECK: [[v26:%.*]] = arith.addi [[v25]], [[vcm1_14]] : index
// CHECK: [[v27:%.*]] = arith.maxsi [[v26]], [[vc0_12]] : index
// CHECK: [[v28:%.*]] = arith.divsi [[v27]], [[vc0_0]] : index
// CHECK: [[vc0_15:%.*]] = arith.constant 0 : index
// CHECK: [[v29:%.*]] = arith.divsi [[vc0_15]], [[vc0_0]] : index
// CHECK: [[v30:%.*]] = arith.minsi [[v29]], [[vc8]] : index
// CHECK: [[vc0_16:%.*]] = arith.constant 0 : index
// CHECK: [[v31:%.*]] = arith.subi [[vc0]], [[v30]] : index
// CHECK: [[v32:%.*]] = arith.subi [[vc0_16]], [[v31]] : index
// CHECK: [[v33:%.*]] = arith.maxsi [[v13]], [[v32]] : index
// CHECK: [[v34:%.*]] = arith.subi [[v28]], [[v31]] : index
// CHECK: [[v35:%.*]] = arith.subi [[v34]], [[v33]] : index
// CHECK: [[v36:%.*]] = arith.subi [[vc6]], [[v33]] : index
// CHECK: [[v37:%.*]] = arith.addi [[v13]], [[v19]] : index
// CHECK: [[v38:%.*]] = arith.subi [[v37]], [[v33]] : index
// CHECK: [[v39:%.*]] = arith.minsi [[v35]], [[v36]] : index
// CHECK: [[v40:%.*]] = arith.minsi [[v38]], [[v39]] : index
// CHECK: return

// -----
func.func @test_cast_elemtype(%arg0: !ndarray.ndarray<?xi64>, %arg1: !ndarray.ndarray<?xi64>, %arg2: !ndarray.ndarray<?xi64>, %arg3: index) -> (!ndarray.ndarray<?xi32>, !ndarray.ndarray<?xi32>, !ndarray.ndarray<?xi32>) {
  %a = dist.init_dist_array l_offset %arg3 parts %arg0, %arg1, %arg2 : index, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64> to !ndarray.ndarray<16xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
  %4 = ndarray.cast_elemtype %a : !ndarray.ndarray<16xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>> to !ndarray.ndarray<16xi32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
  %20, %21, %22 = "dist.parts_of"(%4) : (!ndarray.ndarray<16xi32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) -> (!ndarray.ndarray<?xi32>, !ndarray.ndarray<?xi32>, !ndarray.ndarray<?xi32>)
  return %20, %21, %22 : !ndarray.ndarray<?xi32>, !ndarray.ndarray<?xi32>, !ndarray.ndarray<?xi32>
}
// CHECK-LABEL: @test_cast_elemtype
// CHECK: [[V1:%.*]] = ndarray.cast_elemtype %arg0
// CHECK-NEXT: [[V2:%.*]] = ndarray.cast_elemtype %arg1
// CHECK-NEXT: [[V3:%.*]] = ndarray.cast_elemtype %arg2
// CHECK: return [[V1]], [[V2]], [[V3]]

// -----
func.func @test_copy_permute() -> () {
  %i1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %src = ndarray.create %c3, %c4 value %i1 {dtype = 4 : i8} : (index, index, i32) -> !ndarray.ndarray<3x4xi32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>>
  %dist = ndarray.permute_dims %src [1, 0]
       : !ndarray.ndarray<3x4xi32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>>
       -> !ndarray.ndarray<4x3xi32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>>
  return
}
// CHECK-LABEL: @test_copy_permute
// CHECK: "distruntime.team_size"() <{team = 22 : i64}> : () -> index
// CHECK: "distruntime.team_member"() <{team = 22 : i64}> : () -> index
// CHECK: [[handle:%.*]], [[nlArray:%.*]] = distruntime.copy_permute
// CHECK: "distruntime.wait"([[handle]]) : (!distruntime.asynchandle) -> ()
