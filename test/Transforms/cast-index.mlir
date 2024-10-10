// RUN: imex-opt -cast-index %s | FileCheck %s

module @castindex attributes {gpu.container_module} {
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    // CHECK: gpu.func @test_kernel
    gpu.func @test_kernel(%arg0: memref<4x5xf16>, %arg1: memref<4x5xf16>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 4, 5, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // CHECK: %[[BID_X:.*]] = gpu.block_id  x
      %0 = gpu.block_id  x
      // CHECK: %[[BID_Y:.*]] = gpu.block_id  y
      %1 = gpu.block_id  y
      // CHECK: %[[BID_Z:.*]] = gpu.block_id  z
      %2 = gpu.block_id  z
      // CHECK: %[[TID_X:.*]] = gpu.thread_id  x
      %3 = gpu.thread_id  x
      // CHECK: %[[TID_Y:.*]] = gpu.thread_id  y
      %4 = gpu.thread_id  y
      // CHECK: %[[TID_Z:.*]] = gpu.thread_id  z
      %5 = gpu.thread_id  z
      // CHECK: %[[VAR_0_1:.*]] = index.casts %[[BID_X]] : index to i32
      // CHECK: %[[VAR_3_1:.*]] = index.casts %[[TID_X]] : index to i32
      // CHECK: %[[VAR_6:.*]] = arith.divui %[[VAR_0_1]], %[[VAR_3_1]] : i32
      // CHECK: %[[RE_6:.*]] = index.casts %[[VAR_6]] : i32 to index
      %6 = arith.divui %0, %3 : index
      // CHECK: %[[VAR_1_1:.*]] = index.casts %[[BID_Y]] : index to i32
      // CHECK: %[[VAR_3_2:.*]] = index.casts %[[TID_X]] : index to i32
      // CHECK: %[[VAR_7:.*]] = arith.remui %[[VAR_1_1]], %[[VAR_3_2]] : i32
      // CHECK: %[[RE_7:.*]] = index.casts %[[VAR_7]] : i32 to index
      %7 = arith.remui %1, %3 : index
      // CHECK: %[[VAR_2_1:.*]] = index.casts %[[BID_Z]] : index to i32
      // CHECK: %[[VAR_5_1:.*]] = index.casts %[[TID_Z]] : index to i32
      // CHECK: %[[VAR_8:.*]] = arith.muli %[[VAR_2_1]], %[[VAR_5_1]] : i32
      // CHECK: %[[RE_8:.*]] = index.casts %[[VAR_8]] : i32 to index
      %8 = arith.muli %2, %5 : index
      // CHECK: %[[VAR_2_2:.*]] = index.casts %[[BID_Z]] : index to i32
      // CHECK: %[[VAR_4_1:.*]] = index.casts %[[TID_Y]] : index to i32
      // CHECK: %[[VAR_9:.*]], %[[VAR_19:.*]] = arith.mulsi_extended %[[VAR_2_2]], %[[VAR_4_1]] : i32
      // CHECK: %[[RE_9:.*]] = index.casts %[[VAR_9]] : i32 to index
      %9, %19 = arith.mulsi_extended %2, %4 : index
      // CHECK: %[[VAR_0_2:.*]] = index.casts %[[BID_X]] : index to i32
      // CHECK: %[[VAR_4_2:.*]] = index.casts %[[TID_Y]] : index to i32
      // CHECK: %[[VAR_10:.*]] = arith.divsi %[[VAR_0_2]], %[[VAR_4_2]] : i32
      // CHECK: %[[RE_10:.*]] = index.casts %[[VAR_10]] : i32 to index
      %10 = arith.divsi %0, %4 : index
      // CHECK: %[[VAR_1_2:.*]] = index.casts %[[BID_Y]] : index to i32
      // CHECK: %[[VAR_4_3:.*]] = index.casts %[[TID_Y]] : index to i32
      // CHECK: %[[VAR_11:.*]] = arith.remsi %[[VAR_1_2]], %[[VAR_4_3]] : i32
      // CHECK: %[[RE_11:.*]] = index.casts %[[VAR_11]] : i32 to index
      %11 = arith.remsi %1, %4 : index
      // CHECK: %[[VAR_12:.*]] = memref.load %[[ARG_0:.*]][%[[RE_6]], %[[RE_7]]] : memref<4x5xf16>
      %12 = memref.load %arg0[%6, %7] : memref<4x5xf16>
      // CHECK: %[[VAR_13:.*]] = memref.load %[[ARG_0]][%[[RE_8]], %[[RE_9]]] : memref<4x5xf16>
      %13 = memref.load %arg0[%8, %9] : memref<4x5xf16>
      // CHECK: %[[VAR_14:.*]] = arith.addf %[[VAR_12]], %[[VAR_13]] : f16
      %14 = arith.addf %12, %13 : f16
      // CHECK: memref.store %[[VAR_14]], %[[ARG_1:.*]][%[[RE_10]], %[[RE_11]]] : memref<4x5xf16>
      memref.store %14, %arg1[%10, %11] : memref<4x5xf16>
      gpu.return
    }
  }
}
