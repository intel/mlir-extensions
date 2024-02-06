// RUN: imex-opt --split-input-file --convert-dist-to-standard %s -verify-diagnostics -o -| FileCheck %s

func.func @test_copy(%a : !ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>>) -> !ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>> {
  %1 = ndarray.copy %a : !ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>> -> !ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>>
  %2 = ndarray.copy %1 : !ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>> -> !ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>, #region.gpu_env<device = "XeGPU">>
  %4 = ndarray.copy %2 : !ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>, #region.gpu_env<device = "XeGPU">> -> !ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>>
  return %4 : !ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>>
}
// CHECK-LABEL: func.func @test_copy
// CHECK-SAME: ([[arg0:%.*]]: !ndarray.ndarray<2xi64>, [[arg1:%.*]]: !ndarray.ndarray<6xi64>, [[arg2:%.*]]: !ndarray.ndarray<0xi64>, [[arg3:%.*]]: memref<1xindex>) -> (!ndarray.ndarray<2xi64>, !ndarray.ndarray<6xi64>, !ndarray.ndarray<0xi64>, memref<1xindex>) {
// CHECK: [[v1:%.*]] = ndarray.copy [[arg0]] : !ndarray.ndarray<2xi64> -> !ndarray.ndarray<2xi64>
// CHECK-NEXT: [[v2:%.*]] = ndarray.copy [[arg1]] : !ndarray.ndarray<6xi64> -> !ndarray.ndarray<6xi64>
// CHECK-NEXT: [[v3:%.*]] = ndarray.copy [[arg2]] : !ndarray.ndarray<0xi64> -> !ndarray.ndarray<0xi64>
// CHECK-NEXT: [[v4:%.*]] = ndarray.copy [[v1]] : !ndarray.ndarray<2xi64> -> !ndarray.ndarray<2xi64, #region.gpu_env<device = "XeGPU">>
// CHECK-NEXT: [[v5:%.*]] = ndarray.copy [[v2]] : !ndarray.ndarray<6xi64> -> !ndarray.ndarray<6xi64, #region.gpu_env<device = "XeGPU">>
// CHECK-NEXT: [[v6:%.*]] = ndarray.copy [[v3]] : !ndarray.ndarray<0xi64> -> !ndarray.ndarray<0xi64, #region.gpu_env<device = "XeGPU">>
// CHECK-NEXT: [[v7:%.*]] = ndarray.copy [[v4]] : !ndarray.ndarray<2xi64, #region.gpu_env<device = "XeGPU">> -> !ndarray.ndarray<2xi64>
// CHECK-NEXT: [[v8:%.*]] = ndarray.copy [[v5]] : !ndarray.ndarray<6xi64, #region.gpu_env<device = "XeGPU">> -> !ndarray.ndarray<6xi64>
// CHECK-NEXT: [[v9:%.*]] = ndarray.copy [[v6]] : !ndarray.ndarray<0xi64, #region.gpu_env<device = "XeGPU">> -> !ndarray.ndarray<0xi64>
// CHECK-NEXT: [[valloc:%.*]] = memref.alloc() {alignment = 8 : i64} : memref<1xindex>
// CHECK: return [[v7]], [[v8]], [[v9]], [[valloc]] : !ndarray.ndarray<2xi64>, !ndarray.ndarray<6xi64>, !ndarray.ndarray<0xi64>, memref<1xindex>

// -----
func.func @test_delete(%a : !ndarray.ndarray<33xi64, #dist.dist_env<team = 22 : i64 loffs = 1 lparts = 2,6,0>>) {
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
module {
    func.func @test_init_dist_array(%pt: !ndarray.ndarray<?xi64>, %gshape: index, %loffs: index) -> !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>> {
        %1 = dist.init_dist_array l_offset %loffs parts %pt, %pt, %pt : index, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64> to !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
        return %1 : !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    }
}
// CHECK-LABEL: func.func @test_init_dist_array
// CHECK: memref.store
// CHECK: return
// CHECK-SAME: !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>, memref<1xindex>

// -----
module {
    func.func @test_local_partition(%np : index, %prank: index, %shape: index) -> (index, index) {
        %0, %1 = "dist.default_partition"(%np, %prank, %shape) {rank = 1 : i64} : (index, index, index) -> (index, index)
        return %0, %1 : index, index
    }
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
module {
    func.func @test_local_target_of_slice(%arg0: !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 1 lparts = ?,?,?>>, %c0 : index, %c3 : index) -> (index, index) {
        %l_offsets, %l_sizes = dist.local_target_of_slice %arg0[%c0] [%c3] [%c3] : !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 1 lparts = ?,?,?>> to index, index
        return %l_offsets, %l_sizes : index, index
    }
}
// CHECK-LABEL: func.func @test_local_target_of_slice(%arg0: !ndarray.ndarray<?xi64>, %arg1: !ndarray.ndarray<?xi64>, %arg2: !ndarray.ndarray<?xi64>, %arg3: memref<1xindex>, %arg4: index, %arg5: index) -> (index, index) {
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
func.func @test_0d_inout(%arg0: !ndarray.ndarray<i64, #dist.dist_env<team = 22>>, %arg1: !ndarray.ndarray<i64, #dist.dist_env<team = 22>>) -> !ndarray.ndarray<i64, #dist.dist_env<team = 22>> {
  %2 = "dist.ewbin"(%arg0, %arg1) {op = 23 : i32} : (!ndarray.ndarray<i64, #dist.dist_env<team = 22>>, !ndarray.ndarray<i64, #dist.dist_env<team = 22>>) -> !ndarray.ndarray<i64, #dist.dist_env<team = 22>>
  return %2 : !ndarray.ndarray<i64, #dist.dist_env<team = 22>>
}
// CHECK-LABEL: func.func @test_0d_inout(%arg0: !ndarray.ndarray<i64>, %arg1: !ndarray.ndarray<i64>) -> !ndarray.ndarray<i64> {
// CHECK-NEXT: %0 = ndarray.ewbin %arg0, %arg1 {op = 23 : i32} : (!ndarray.ndarray<i64>, !ndarray.ndarray<i64>) -> !ndarray.ndarray<i64>
// CHECK-NEXT: return %0 : !ndarray.ndarray<i64>

// -----
module {
    func.func @test_repartition(%arg0: !ndarray.ndarray<10x12xi64, #dist.dist_env<team = 22 loffs = 0,0 lparts = ?x?,?x?,?x?>>) -> !ndarray.ndarray<10x12xi64, #dist.dist_env<team = 22 loffs = 0,0 lparts = ?x?,?x?,?x?>> {
        %0 = dist.repartition %arg0 : !ndarray.ndarray<10x12xi64, #dist.dist_env<team = 22 loffs = 0,0 lparts = ?x?,?x?,?x?>> to !ndarray.ndarray<10x12xi64, #dist.dist_env<team = 22 loffs = 0,0 lparts = ?x?,?x?,?x?>>
        return %0 : !ndarray.ndarray<10x12xi64, #dist.dist_env<team = 22 loffs = 0,0 lparts = ?x?,?x?,?x?>>
    }
}
// CHECK-LABEL: @test_repartition
// CHECK: ndarray.dim
// CHECK: ndarray.dim
// CHECK: ndarray.dim
// CHECK: ndarray.dim
// CHECK: distruntime.get_halo
// CHECK: memref.store
// CHECK: memref.store
// CHECK: return

// -----
func.func @test_local_core(%arg0: !ndarray.ndarray<16xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 0 : index
    %c6 = arith.constant 6 : index
    %c8 = arith.constant 8 : index
    %resultOffsets, %resultSizes = dist.local_core %arg0 toffs %c0 tsizes %c6 soffs %c0 ssizes %c8 sstrides %c1 : !ndarray.ndarray<16xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>> to index, index
    %resultOffsets_6, %resultSizes_7 = dist.local_core %arg0 toffs %c0 tsizes %c6 soffs %c1 ssizes %c8 sstrides %c1 coffs %resultOffsets csizes %resultSizes : !ndarray.ndarray<16xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>> to index, index
    return
}
// CHECK: [[vc0:%.*]] = arith.constant 0 : index
// CHECK: [[v0:%.*]] = memref.load %arg3[[[vc0]]] : memref<1xindex>
// CHECK: [[vc0_0:%.*]] = arith.constant 0 : index
// CHECK: [[vc0_1:%.*]] = arith.constant 0 : index
// CHECK: [[vc6:%.*]] = arith.constant 6 : index
// CHECK: [[vc8:%.*]] = arith.constant 8 : index
// CHECK: [[vc0_2:%.*]] = arith.constant 0 : index
// CHECK: [[vdim:%.*]] = ndarray.dim %arg1 [[vc0_2]] : !ndarray.ndarray<?xi64> -> index
// CHECK: [[vc0_3:%.*]] = arith.constant 0 : index
// CHECK: [[vdim_4:%.*]] = ndarray.dim %arg0 [[vc0_3]] : !ndarray.ndarray<?xi64> -> index
// CHECK: [[v1:%.*]] = arith.addi [[v0]], [[vdim_4]] : index
// CHECK: [[vc0_5:%.*]] = arith.constant 0 : index
// CHECK: [[vc1:%.*]] = arith.constant 1 : index
// CHECK: [[v2:%.*]] = arith.addi [[v0]], [[vdim]] : index
// CHECK: [[v3:%.*]] = arith.maxsi [[v0]], [[vc0_0]] : index
// CHECK: [[vcm1:%.*]] = arith.constant -1 : index
// CHECK: [[v4:%.*]] = arith.addi [[v3]], [[vcm1]] : index
// CHECK: [[v5:%.*]] = arith.divsi [[v4]], [[vc0_1]] : index
// CHECK: [[v6:%.*]] = arith.minsi [[v2]], [[vc0_0]] : index
// CHECK: [[v7:%.*]] = arith.addi [[v6]], [[vcm1]] : index
// CHECK: [[v8:%.*]] = arith.maxsi [[v7]], [[vc0_5]] : index
// CHECK: [[v9:%.*]] = arith.divsi [[v8]], [[vc0_1]] : index
// CHECK: [[vc0_6:%.*]] = arith.constant 0 : index
// CHECK: [[v10:%.*]] = arith.divsi [[vc0_6]], [[vc0_1]] : index
// CHECK: [[v11:%.*]] = arith.minsi [[v10]], [[vc8]] : index
// CHECK: [[vc0_7:%.*]] = arith.constant 0 : index
// CHECK: [[v12:%.*]] = arith.subi [[vc0_0]], [[v11]] : index
// CHECK: [[v13:%.*]] = arith.subi [[vc0_7]], [[v12]] : index
// CHECK: [[v14:%.*]] = arith.maxsi [[v13]], [[vc0_7]] : index
// CHECK: [[v15:%.*]] = arith.subi [[v9]], [[v12]] : index
// CHECK: [[v16:%.*]] = arith.subi [[v15]], [[v14]] : index
// CHECK: [[v17:%.*]] = arith.subi [[vc6]], [[v14]] : index
// CHECK: [[vc6_8:%.*]] = arith.constant 6 : index
// CHECK: [[v18:%.*]] = arith.subi [[vc6_8]], [[v14]] : index
// CHECK: [[v19:%.*]] = arith.minsi [[v16]], [[v17]] : index
// CHECK: [[v20:%.*]] = arith.minsi [[v18]], [[v19]] : index
// CHECK: [[vc0_9:%.*]] = arith.constant 0 : index
// CHECK: [[vdim_10:%.*]] = ndarray.dim %arg1 [[vc0_9]] : !ndarray.ndarray<?xi64> -> index
// CHECK: [[vc0_11:%.*]] = arith.constant 0 : index
// CHECK: [[vdim_12:%.*]] = ndarray.dim %arg0 [[vc0_11]] : !ndarray.ndarray<?xi64> -> index
// CHECK: [[v21:%.*]] = arith.addi [[v0]], [[vdim_12]] : index
// CHECK: [[vc0_13:%.*]] = arith.constant 0 : index
// CHECK: [[vc1_14:%.*]] = arith.constant 1 : index
// CHECK: [[v22:%.*]] = arith.addi [[v0]], [[vdim_10]] : index
// CHECK: [[v23:%.*]] = arith.maxsi [[v0]], [[vc0_1]] : index
// CHECK: [[vcm1_15:%.*]] = arith.constant -1 : index
// CHECK: [[v24:%.*]] = arith.addi [[v23]], [[vcm1_15]] : index
// CHECK: [[v25:%.*]] = arith.divsi [[v24]], [[vc0_1]] : index
// CHECK: [[v26:%.*]] = arith.minsi [[v22]], [[vc0_1]] : index
// CHECK: [[v27:%.*]] = arith.addi [[v26]], [[vcm1_15]] : index
// CHECK: [[v28:%.*]] = arith.maxsi [[v27]], [[vc0_13]] : index
// CHECK: [[v29:%.*]] = arith.divsi [[v28]], [[vc0_1]] : index
// CHECK: [[vc0_16:%.*]] = arith.constant 0 : index
// CHECK: [[v30:%.*]] = arith.divsi [[vc0_16]], [[vc0_1]] : index
// CHECK: [[v31:%.*]] = arith.minsi [[v30]], [[vc8]] : index
// CHECK: [[vc0_17:%.*]] = arith.constant 0 : index
// CHECK: [[v32:%.*]] = arith.subi [[vc0_0]], [[v31]] : index
// CHECK: [[v33:%.*]] = arith.subi [[vc0_17]], [[v32]] : index
// CHECK: [[v34:%.*]] = arith.maxsi [[v14]], [[v33]] : index
// CHECK: [[v35:%.*]] = arith.subi [[v29]], [[v32]] : index
// CHECK: [[v36:%.*]] = arith.subi [[v35]], [[v34]] : index
// CHECK: [[v37:%.*]] = arith.subi [[vc6]], [[v34]] : index
// CHECK: [[v38:%.*]] = arith.addi [[v14]], [[v20]] : index
// CHECK: [[v39:%.*]] = arith.subi [[v38]], [[v34]] : index
// CHECK: [[v40:%.*]] = arith.minsi [[v36]], [[v37]] : index
// CHECK: [[v41:%.*]] = arith.minsi [[v39]], [[v40]] : index

// -----
func.func @test_cast_elemtype(%arg0: !ndarray.ndarray<16xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) -> !ndarray.ndarray<16xi32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>> {
    %0 = ndarray.cast_elemtype %arg0 : !ndarray.ndarray<16xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>> to !ndarray.ndarray<16xi32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    return %0 : !ndarray.ndarray<16xi32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
  }
// CHECK-LABEL: @test_cast_elemtype
// CHECK: [[V1:%.*]] = ndarray.cast_elemtype %arg0
// CHECK-NEXT: [[V2:%.*]] = ndarray.cast_elemtype %arg1
// CHECK-NEXT: [[V3:%.*]] = ndarray.cast_elemtype %arg2
// CHECK: return [[V1]], [[V2]], [[V3]],
