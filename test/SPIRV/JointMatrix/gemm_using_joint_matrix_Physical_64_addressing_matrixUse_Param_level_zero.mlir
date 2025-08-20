// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
// CHECK-COUNT-4194304: 4970.23
module @gemm_using_jointmatrix_module attributes {gpu.container_module} {
  memref.global "private" constant @__constant_A_2048x2048xbf16 : memref<2048x2048xbf16> = dense<1.100000e+00>
  memref.global "private" constant @__constant_B_1024x2048x2xbf16 : memref<1024x2048x2xbf16> = dense<2.200000e+00>
  memref.global "private" constant @__constant_C_2048x2048xf32 : memref<2048x2048xf32> = dense<0.000000e+00>

  // memref.global "private" constant @__constant_test_store : memref<1xf32> = dense<0.000000e+00>
  // M = 8
  // K = 16
  // N = 16
  // SG_SIZE = 16
  func.func @test(%arg_A: memref<2048x2048xbf16>, %arg_B: memref<1024x2048x2xbf16>, %arg_C: memref<2048x2048xf32>) -> memref<2048x2048xf32> attributes {llvm.emit_c_interface} {
    %c2048 = arith.constant 2048 : index
    %c1024 = arith.constant 1024 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index

    %memref_arg_A_i8 = gpu.alloc host_shared () : memref<8388608xi8>
    %memref_arg_A_bf16 = memref.view %memref_arg_A_i8[%c0][] : memref<8388608xi8> to memref<2048x2048xbf16>
    %memref_arg_A_i16 = memref.view %memref_arg_A_i8[%c0][] : memref<8388608xi8> to memref<2048x2048xi16>
    memref.copy %arg_A, %memref_arg_A_bf16 : memref<2048x2048xbf16> to memref<2048x2048xbf16>
    %memref_arg_A_i16_flat = memref.cast %memref_arg_A_i16 : memref<2048x2048xi16> to memref<*xi16>

    %memref_arg_B_i8 = gpu.alloc  host_shared () : memref<8388608xi8>
    %memref_arg_B_bf16 = memref.view %memref_arg_B_i8[%c0][] : memref<8388608xi8> to memref<1024x2048x2xbf16> //VNNI transformed
    %memref_arg_B_i16 = memref.view %memref_arg_B_i8[%c0][] : memref<8388608xi8> to memref<1024x2048x2xi16> //VNNI transformed
    memref.copy %arg_B, %memref_arg_B_bf16 : memref<1024x2048x2xbf16> to memref<1024x2048x2xbf16>
    %memref_arg_B_i16_flat = memref.cast %memref_arg_B_i16 : memref<1024x2048x2xi16> to memref<*xi16>

    %memref_arg_C_i8 = gpu.alloc  host_shared () : memref<16777216xi8>
    %memref_arg_C_f32 = memref.view %memref_arg_C_i8[%c0][] : memref<16777216xi8> to memref<2048x2048xf32>
    memref.copy %arg_C, %memref_arg_C_f32 : memref<2048x2048xf32> to memref<2048x2048xf32>
    %memref_arg_C_f32_flat = memref.cast %memref_arg_C_f32 : memref<2048x2048xf32> to memref<*xf32>

    // To use sycl runtime, the blocks and threads needs to be passed in a slightly different way to match the Global and local size
    //


    // Calling convetion in MLIR:
    // ===========================
    // blocks in (gridX, gridY, gridZ) threads in (blockX, blockY, blockZ)

    // Calling convetion in SYCL/DPC++:
    // =================================
    // nd_range<dimensions>({global_size.x, global_size.y, global_size.z}, {local_size.x, local_size.y, local_size.z})

    // Conversion between MLIR and SYCL/DPC++ convetion:
    // ===================================================
    // Change of dimensions (X and Z dimensions are interchanged):
    // =============================================================
    // Gobal Range/Size:
    // ===================
    // global_size.x = blockZ * gridZ,
    // global_size.y = blockY * gridY,
    // global_size.y = blockX * gridX

    // Local Range/Size:
    // ====================
    // local_size.x = blockZ
    // local_size.y = blockY
    // local_size.z = blockX

    // For details see: mlir-extensions/lib/ExecutionEngine/SYCLRUNTIME/SyclRuntimeWrappers.cpp

    gpu.launch_func   @gemm_using_jointmatrix_module::@gemm_using_jointmatrix blocks in (%c256, %c128, %c1) threads in (%c1, %c16, %c1) args(%memref_arg_A_i16 : memref<2048x2048xi16>, %memref_arg_B_i16 : memref<1024x2048x2xi16>, %memref_arg_C_f32 : memref<2048x2048xf32>)

    gpu.dealloc  %memref_arg_A_i8 : memref<8388608xi8>
    gpu.dealloc  %memref_arg_B_i8 : memref<8388608xi8>
    return %memref_arg_C_f32 : memref<2048x2048xf32>

  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.get_global @__constant_A_2048x2048xbf16 : memref<2048x2048xbf16>
    %B = memref.get_global @__constant_B_1024x2048x2xbf16 : memref<1024x2048x2xbf16>
    %C = memref.get_global @__constant_C_2048x2048xf32 : memref<2048x2048xf32>

    %result = call @test(%A, %B, %C) : (memref<2048x2048xbf16>, memref<1024x2048x2xbf16>, memref<2048x2048xf32>) -> memref<2048x2048xf32>
    %cast = memref.cast %result : memref<2048x2048xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}

  spirv.module @__spv__gemm_using_jointmatrix_module Physical64 OpenCL requires #spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    // workgroup_size = [1, 16, 1], subgroup_size = 16
    spirv.GlobalVariable @__builtin_var_NumWorkgroups__ built_in("NumWorkgroups") : !spirv.ptr<vector<3xi64>, Input>
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.GlobalVariable @__builtin_var_SubgroupId__ built_in("SubgroupId") : !spirv.ptr<i64, Input>
    spirv.GlobalVariable @__builtin_var_SubgroupSize__ built_in("SubgroupSize") : !spirv.ptr<i64, Input>
    spirv.GlobalVariable @__builtin_var_GlobalInvocationId__ built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.GlobalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spirv.ptr<vector<3xi64>, Input>
    // gpu.known_block_size = array<i32: 1, 16, 1>, gpu.known_grid_size = array<i32: 256, 128, 1>,
    spirv.func @gemm_using_jointmatrix(%arg0 : !spirv.ptr<!spirv.array<4194304 x i16>, CrossWorkgroup>, %arg1 : !spirv.ptr<!spirv.array<4194304 x i16>, CrossWorkgroup>, %arg2 : !spirv.ptr<!spirv.array<4194304 x f32>, CrossWorkgroup>) "None" attributes { workgroup_attributions = 0 : i64} {
      %c0 = spirv.Constant 0 : i64
      %c1 = spirv.Constant 1 : i64

      %subgroup_size = spirv.Constant 16 : i64
      %c_M = spirv.Constant 2048 : i64
      %c_N = spirv.Constant 2048 : i64
      %c_K = spirv.Constant 2048 : i64

      %c_tM = spirv.Constant 8 : i64
      %c_tK = spirv.Constant 16 : i64
      %c_tN = spirv.Constant 16 : i64

      %c_vnni_factor = spirv.Constant 2 : i64

      // Get workgroup IDs
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %workgroupId_x = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %workgroupId_y = spirv.CompositeExtract %0[1 : i32] : vector<3xi64>
      %workgroupId_z = spirv.CompositeExtract %0[2 : i32] : vector<3xi64>

      // Get the subgroup ID
      %__builtin_var_SubgroupId___addr = spirv.mlir.addressof @__builtin_var_SubgroupId__ : !spirv.ptr<i64, Input>
      %subgroupId = spirv.Load "Input" %__builtin_var_SubgroupId___addr : i64

      // Get the numworkgroups
      %__builtin_var_NumWorkgroups___addr = spirv.mlir.addressof @__builtin_var_NumWorkgroups__ : !spirv.ptr<vector<3xi64>, Input>
      %1 = spirv.Load "Input" %__builtin_var_NumWorkgroups___addr : vector<3xi64>
      %numworkgroups_x = spirv.CompositeExtract %1[0 : i32] : vector<3xi64>
      %numworkgroups_y = spirv.CompositeExtract %1[1 : i32] : vector<3xi64>
      %numworkgroups_z = spirv.CompositeExtract %1[2 : i32] : vector<3xi64>

      // Get the global invocation ID (global ID)
      %__builtin_var_GlobalInvocationId___addr = spirv.mlir.addressof @__builtin_var_GlobalInvocationId__ : !spirv.ptr<vector<3xi64>, Input>
      %2 = spirv.Load "Input" %__builtin_var_GlobalInvocationId___addr : vector<3xi64>
      %globalId_x = spirv.CompositeExtract %2[0 : i32] : vector<3xi64>
      %globalId_y = spirv.CompositeExtract %2[1 : i32] : vector<3xi64>
      %globalId_z = spirv.CompositeExtract %2[2 : i32] : vector<3xi64>

      // Get the local ID (thred ID)

      %__builtin_var_LocalInvocationId___addr = spirv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spirv.ptr<vector<3xi64>, Input>
      %3 = spirv.Load "Input" %__builtin_var_LocalInvocationId___addr : vector<3xi64>
      %localId_x = spirv.CompositeExtract %3[0 : i32] : vector<3xi64>
      %localId_y = spirv.CompositeExtract %3[1 : i32] : vector<3xi64>
      %localId_z = spirv.CompositeExtract %3[2 : i32] : vector<3xi64>

      %sg_start_x = spirv.ISub %globalId_x, %localId_x : i64
      %sg_start_y = spirv.ISub %globalId_y, %localId_y : i64
      // Load C
      // %load_offset_C = sg_start_x * tM * colsB + sg_start_y / SG_SIZE * tN

      %mul_c_1 = spirv.IMul %sg_start_x, %c_tM : i64
      %mul_c_2 = spirv.IMul %mul_c_1, %c_N : i64

      %div_c_1 = spirv.UDiv %sg_start_y, %subgroup_size : i64
      %mul_c_3 = spirv.IMul %div_c_1, %c_tN : i64

      %load_offset_C = spirv.IAdd %mul_c_2, %mul_c_3 : i64
      %load_address_C = spirv.AccessChain %arg2[%load_offset_C] : !spirv.ptr<!spirv.array<4194304xf32>, CrossWorkgroup>, i64

      %joint_matrix_C = spirv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> %load_address_C, %c_N  : (!spirv.ptr<f32, CrossWorkgroup>, i64) -> !spirv.jointmatrix<8x16xf32, RowMajor, Subgroup, Accumulator>

      // Loop through (k = 0; k < colsA / tK; k++)
      %loop_cnt = spirv.UDiv %c_K, %c_tK : i64
      spirv.mlir.loop {
        spirv.Branch ^bb1(%c0, %joint_matrix_C: i64, !spirv.jointmatrix<8x16xf32, RowMajor, Subgroup, Accumulator>)
        ^bb1(%k: i64, %matrixC1: !spirv.jointmatrix<8x16xf32, RowMajor, Subgroup, Accumulator>):  // 2 preds: ^bb0, ^bb2
          %5 = spirv.ULessThan %k, %loop_cnt : i64
          spirv.BranchConditional %5, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          // Loading A
          // %load_offset_A = sg_start_x * tM * colsA + k * tK
          // %load_offset_A = (%globalId_x - %localId_x) * %c_K + (%k * %c_tK)
          %mul_1 = spirv.IMul %sg_start_x, %c_tM : i64
          %mul_2 = spirv.IMul %mul_1, %c_K : i64  // sg_start_x * tM * colsA

          %mul_3 = spirv.IMul %k, %c_tK : i64     // k * tK
          %load_offset_A = spirv.IAdd %mul_2, %mul_3 : i64
          %load_address_A = spirv.AccessChain %arg0[%load_offset_A] : !spirv.ptr<!spirv.array<4194304xi16>, CrossWorkgroup>, i64

          %joint_matrix_A = spirv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> %load_address_A, %c_K  : (!spirv.ptr<i16, CrossWorkgroup>, i64) -> !spirv.jointmatrix<8x16xi16, RowMajor, Subgroup, MatrixA>

          // Loading B
          // %load_offset_B = (k * tK / vnniFactor) * (colsB * vnniFactor) + sg_start_y / SG_SIZE * tN * vnniFactor
          %div_1 = spirv.UDiv %c_tK, %c_vnni_factor : i64   // k * tK
          %mul_4 = spirv.IMul %k, %div_1 : i64            // (k * tK / vnniFactor)

          %mul_5 = spirv.IMul %c_N, %c_vnni_factor : i64  // (colsB * vnniFactor)

          %mul_6 = spirv.IMul %mul_4, %mul_5 : i64        // (k * tK / vnniFactor) * (colsB * vnniFactor)

          %div_2 = spirv.UDiv %sg_start_y, %subgroup_size : i64 // sg_start_y / SG_SIZE
          %mul_7 = spirv.IMul %div_2, %c_tN : i64               // sg_start_y / SG_SIZE * tN
          %mul_8 = spirv.IMul %mul_7, %c_vnni_factor : i64      // sg_start_y / SG_SIZE * tN * vnniFactor

          %load_offset_B = spirv.IAdd %mul_6, %mul_8 : i64

          %load_address_B = spirv.AccessChain %arg1[%load_offset_B] : !spirv.ptr<!spirv.array<4194304xi16>, CrossWorkgroup>, i64
          %stride_B = spirv.IMul %c_N, %c_vnni_factor : i64


          %joint_matrix_B = spirv.INTEL.JointMatrixLoad <Subgroup> <Packed> %load_address_B, %stride_B  : (!spirv.ptr<i16, CrossWorkgroup>, i64) -> !spirv.jointmatrix<16x16xi16, Packed, Subgroup, MatrixB>

          %r = spirv.INTEL.JointMatrixMad <Subgroup> %joint_matrix_A, %joint_matrix_B, %matrixC1 : !spirv.jointmatrix<8x16xi16, RowMajor, Subgroup, MatrixA>, !spirv.jointmatrix<16x16xi16, Packed, Subgroup, MatrixB> -> !spirv.jointmatrix<8x16xf32,  RowMajor, Subgroup, Accumulator>

          %incr_k = spirv.IAdd %k, %c1 : i64
          %6 = spirv.ULessThan %incr_k, %loop_cnt : i64
          spirv.BranchConditional %6, ^continue(%incr_k, %r : i64, !spirv.jointmatrix<8x16xf32,  RowMajor, Subgroup, Accumulator>), ^store(%incr_k, %r : i64, !spirv.jointmatrix<8x16xf32, RowMajor, Subgroup, Accumulator>)

        ^store(%k2: i64, %matrixC2: !spirv.jointmatrix<8x16xf32, RowMajor, Subgroup, Accumulator>):
          spirv.INTEL.JointMatrixStore <Subgroup> <RowMajor> %load_address_C, %matrixC2, %c_N  : (!spirv.ptr<f32, CrossWorkgroup>, !spirv.jointmatrix<8x16xf32, RowMajor, Subgroup, Accumulator>, i64)
          spirv.Branch ^continue(%k2, %matrixC2: i64, !spirv.jointmatrix<8x16xf32, RowMajor, Subgroup, Accumulator>)

        ^continue(%k3: i64, %matrixC3: !spirv.jointmatrix<8x16xf32, RowMajor, Subgroup, Accumulator>):
          spirv.Branch ^bb1(%k3, %matrixC3: i64, !spirv.jointmatrix<8x16xf32, RowMajor, Subgroup, Accumulator>)
        ^bb3:  // pred: ^bb1
          spirv.mlir.merge
      }
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @gemm_using_jointmatrix, @__builtin_var_NumWorkgroups__, @__builtin_var_WorkgroupId__, @__builtin_var_SubgroupId__, @__builtin_var_SubgroupSize__, @__builtin_var_GlobalInvocationId__, @__builtin_var_LocalInvocationId__
    // Setting up workgroup size (@details in mlir/test/Dialect/SPIRV/Transforms/abi-interface-opencl.mlir)
    // spirv.ExecutionMode @gemm_using_jointmatrix "LocalSize", 1, 16, 1

    // Setting up subgroup size for the specific kernel
    spirv.ExecutionMode @gemm_using_jointmatrix "SubgroupSize", 16
  }

  gpu.module @gemm_using_jointmatrix_module attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_AMD_shader_ballot]>, api=OpenCL, #spirv.resource_limits<subgroup_size = 16>>} {
    gpu.func @gemm_using_jointmatrix(%arg0 : memref<2048x2048xi16>, %arg1 : memref<1024x2048x2xi16>, %arg2 : memref<2048x2048xf32>) kernel attributes {gpu.known_block_size = array<i32: 16, 1, 1>, gpu.known_grid_size = array<i32: 128, 256, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // skipping the gpu.func body, since we already have spirv.func body, this won't be used
      gpu.return
    }
  }

}
