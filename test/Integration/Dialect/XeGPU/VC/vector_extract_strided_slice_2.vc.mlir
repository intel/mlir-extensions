// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<32x16xf32>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<32x16xf32>
    memref.copy %A, %memref : memref<32x16xf32> to memref<32x16xf32>
    %memref_1 = gpu.alloc  host_shared () : memref<8x16xf32>
    gpu.launch_func  @module0::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<32x16xf32>, %memref_1 : memref<8x16xf32>)
    gpu.dealloc  %memref : memref<32x16xf32>
    return %memref_1 : memref<8x16xf32>
  }

  gpu.module @module0 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%A: memref<32x16xf32>, %Out: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      // load tile
      %tile = xegpu.create_nd_tdesc %A [%c0, %c0] : memref<32x16xf32> -> !xegpu.tensor_desc<32x8xf32, #xegpu.block_tdesc_attr<array_length = 2>>
      %value = xegpu.load_nd %tile : !xegpu.tensor_desc<32x8xf32, #xegpu.block_tdesc_attr<array_length = 2>> -> vector<2x32x8xf32>
      // extract the bottom 8x8 part of first 32x8 block
      %sub_tile0 = vector.extract_strided_slice %value { offsets = [0, 24], strides = [1, 1], sizes = [1, 8] } : vector<2x32x8xf32> to vector<1x8x8xf32>
      // extract the bottom 8x8 part of second 32x8 block
      %sub_tile1 = vector.extract_strided_slice %value { offsets = [1, 24], strides = [1, 1], sizes = [1, 8] } : vector<2x32x8xf32> to vector<1x8x8xf32>
      // combine these two 8x8 tiles into a single 8x16 tile
      %t1 = vector.shape_cast %sub_tile0 : vector<1x8x8xf32> to vector<8x8xf32>
      %t2 = vector.shape_cast %sub_tile1 : vector<1x8x8xf32> to vector<8x8xf32>
      %t3 = vector.shuffle %t1, %t2 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x8xf32>, vector<8x8xf32>
      %t4 = vector.shape_cast %t3 : vector<16x8xf32> to vector<8x16xf32>

      // store the result
      %out_tile = xegpu.create_nd_tdesc %Out [%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %t4, %out_tile : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // init constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c24 = arith.constant 24 : index
    %c1_f32 = arith.constant 1.0 : f32
    %A = memref.alloc() : memref<32x16xf32>
    %Out_cpu = memref.alloc() : memref<8x16xf32>
    // fill A with values form 0, 1, ...., 511
    scf.for %i = %c0 to %c32 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        %t1 = arith.muli %i, %c16 : index
        %val = arith.addi %t1, %j : index
        %val_i32 = arith.index_cast %val : index to i32
        %val_f32 = arith.sitofp %val_i32 : i32 to f32
        %cond = arith.cmpi "sge", %i, %c24 : index
        // only store the bottom 8x16 into Out_cpu
        scf.if %cond {
          %i_cpu = arith.subi %i, %c24 : index
          memref.store %val_f32, %Out_cpu[%i_cpu, %j] : memref<8x16xf32>
        }
        memref.store %val_f32, %A[%i, %j] : memref<32x16xf32>
      }
    }
    // run GPU version
    %Out_gpu = call @test(%A) : (memref<32x16xf32>) -> memref<8x16xf32>
    %Out_gpu_cast = memref.cast %Out_gpu : memref<8x16xf32> to memref<*xf32>
    %A_cast = memref.cast %A : memref<32x16xf32> to memref<*xf32>
    %Out_cpu_cast = memref.cast %Out_cpu : memref<8x16xf32> to memref<*xf32>
    // print GPU and CPU outs
    // call @printMemrefF32(%Out_gpu_cast) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%Out_cpu_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%Out_gpu_cast, %Out_cpu_cast) : (memref<*xf32>, memref<*xf32>) -> ()
    // dealloc
    memref.dealloc %A : memref<32x16xf32>
    // gpu dealloc
    gpu.dealloc %Out_gpu : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
