// RUN: imex-opt -allow-unregistered-dialect -split-input-file -imex-convert-gpu-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, #spirv.resource_limits<>>
} {
  func.func @load_store(%arg0: memref<2x5xf32>, %arg1: memref<2x5xf32>, %arg2: memref<2x5xf32>) {
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index

    gpu.launch_func @kernels::@load_store_kernel
        blocks in (%c2, %c5, %c1) threads in (%c1, %c1, %c1)
        args(%arg0: memref<2x5xf32>, %arg1: memref<2x5xf32>, %arg2: memref<2x5xf32>,
             %c0 : index, %c0 : index, %c1 : index, %c1 : index)
    return
  }

  // CHECK-LABEL: spirv.module @{{.*}} Physical64 OpenCL
  gpu.module @kernels {
    // CHECK-DAG: spirv.GlobalVariable @[[$LOCALINVOCATIONIDVAR:.*]] built_in("LocalInvocationId") : !spirv.ptr<vector<3xi64>, Input>
    // CHECK-DAG:spirv.GlobalVariable @[[$WORKGROUPIDVAR:.*]] built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    // CHECK-LABEL:    spirv.func @load_store_kernel(
    // CHECK-SAME: %[[ARG0:.*]]: !spirv.ptr<!spirv.array<10 x f32>, CrossWorkgroup>{{.*}}, %[[ARG1:.*]]: !spirv.ptr<!spirv.array<10 x f32>, CrossWorkgroup>{{.*}}, %[[ARG2:.*]]: !spirv.ptr<!spirv.array<10 x f32>, CrossWorkgroup>,
    // CHECK-SAME: %[[ARG3:.*]]: i64, %[[ARG4:.*]]: i64, %[[ARG5:.*]]: i64, %[[ARG6:.*]]: i64
    gpu.func @load_store_kernel(%arg0: memref<2x5xf32>, %arg1: memref<2x5xf32>, %arg2: memref<2x5xf32>, %arg3: index, %arg4: index, %arg5: index, %arg6: index) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // CHECK: %[[ADDRESSWORKGROUPID:.*]] = spirv.mlir.addressof @[[$WORKGROUPIDVAR]]
      // CHECK: %[[WORKGROUPID:.*]] = spirv.Load "Input" %[[ADDRESSWORKGROUPID]]
      // CHECK: %[[WORKGROUPIDX:.*]] = spirv.CompositeExtract %[[WORKGROUPID]]{{\[}}0 : i32{{\]}}
      // CHECK: %[[ADDRESSLOCALINVOCATIONID:.*]] = spirv.mlir.addressof @[[$LOCALINVOCATIONIDVAR]]
      // CHECK: %[[LOCALINVOCATIONID:.*]] = spirv.Load "Input" %[[ADDRESSLOCALINVOCATIONID]]
      // CHECK: %[[LOCALINVOCATIONIDX:.*]] = spirv.CompositeExtract %[[LOCALINVOCATIONID]]{{\[}}0 : i32{{\]}}
      %0 = gpu.block_id x
      %1 = gpu.block_id y
      %2 = gpu.thread_id x

      // CHECK: %[[INDEX1:.*]] = spirv.IAdd %[[ARG3]], %[[WORKGROUPIDX]]
      %12 = arith.addi %arg3, %0 : index
      // CHECK: %[[INDEX2:.*]] = spirv.IAdd %[[ARG4]], %[[LOCALINVOCATIONIDX]]
      %13 = arith.addi %arg4, %2 : index
      // CHECK: %[[OFFSET1_0:.*]] = spirv.Constant 0 : i64
      // CHECK: %[[STRIDE1_1:.*]] = spirv.Constant 5 : i64
      // CHECK: %[[UPDATE1_1:.*]] = spirv.IMul %[[INDEX1]], %[[STRIDE1_1]] : i64
      // CHECK: %[[STRIDE1_2:.*]] = spirv.Constant 1 : i64
      // CHECK: %[[OFFSET1_2:.*]] = spirv.IAdd %[[INDEX2]], %[[UPDATE1_1]] : i64
      // CHECK: %[[PTR1:.*]] = spirv.AccessChain %[[ARG0]]{{\[}}
      // CHECK-NEXT: %[[VAL1:.*]] = spirv.Load "CrossWorkgroup" %[[PTR1]]
      %14 = memref.load %arg0[%12, %13] : memref<2x5xf32>
      // CHECK: %[[PTR2:.*]] = spirv.AccessChain %[[ARG1]]{{\[}}
      // CHECK-NEXT: %[[VAL2:.*]] = spirv.Load "CrossWorkgroup" %[[PTR2]]
      %15 = memref.load %arg1[%12, %13] : memref<2x5xf32>
      // CHECK: %[[VAL3:.*]] = spirv.FAdd %[[VAL1]], %[[VAL2]]
      %16 = arith.addf %14, %15 : f32
      // CHECK: %[[PTR3:.*]] = spirv.AccessChain %[[ARG2]]{{\[}}
      // CHECK-NEXT: spirv.Store "CrossWorkgroup" %[[PTR3]], %[[VAL3]]
      memref.store %16, %arg2[%12, %13] : memref<2x5xf32>
  %17 = math.rsqrt %14 : f32
      gpu.return
    }
  }
}
