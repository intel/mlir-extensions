// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

// NOTES :
// This example assumes one subgroup per one workgroup and the kernel specifies the computation
// done by a single subgroup.

module @transpose attributes {gpu.container_module} {
  func.func @transpose_test(%A: memref<1024x1024xf16>) -> memref<1024x1024xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %A_gpu = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %A, %A_gpu : memref<1024x1024xf16> to memref<1024x1024xf16>
    %B_gpu = gpu.alloc  host_shared () : memref<1024x1024xf16>
    gpu.launch_func  @transpose_kernel::@transpose_kernel blocks in (%c64, %c32, %c1) threads in (%c1, %c1, %c1) args(%A_gpu : memref<1024x1024xf16>, %B_gpu : memref<1024x1024xf16>)
    gpu.dealloc  %A_gpu : memref<1024x1024xf16>
    return %B_gpu : memref<1024x1024xf16>
  }
  gpu.module @transpose_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @transpose_kernel(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %m = arith.muli %block_id_x, %c16 : index
      %n = arith.muli %block_id_y, %c32 : index
      // initalize A and B tiles
      %a_tile = xetile.init_tile %A[%m, %n] : memref<1024x1024xf16> -> !xetile.tile<16x32xf16>
      %a_value = xetile.load_tile %a_tile  : !xetile.tile<16x32xf16> -> vector<16x32xf16>
      %b_value = vector.transpose %a_value, [1, 0] : vector<16x32xf16> to vector<32x16xf16>

      %b_tile = xetile.init_tile %B[%n, %m] : memref<1024x1024xf16> -> !xetile.tile<32x16xf16>
      xetile.store_tile %b_value, %b_tile: vector<32x16xf16>, !xetile.tile<32x16xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %cf_0 = arith.constant 0.0 : f16
    %cf_1 = arith.constant 1.0 : f16
    %A = memref.alloc() : memref<1024x1024xf16>
    %B_ref = memref.alloc() : memref<1024x1024xf32>
    // intialize matrix A; A[i, j] = j; B[i, j] = i
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        %t = index.castu %j : index to i16
        %val = arith.uitofp %t : i16 to f16
        memref.store %val, %A[%i, %j] : memref<1024x1024xf16>
        %val_f32 = arith.extf %val : f16 to f32
        memref.store %val_f32, %B_ref[%j, %i] : memref<1024x1024xf32>
      }
    }

    %2 = call @transpose_test(%A) : (memref<1024x1024xf16>) -> memref<1024x1024xf16>
    %cast_B = memref.cast %2 : memref<1024x1024xf16> to memref<*xf16>
    %cast_B_ref = memref.cast %B_ref : memref<1024x1024xf32> to memref<*xf32>
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF16(%cast_B, %cast_B_ref) : (memref<*xf16>, memref<*xf32>) -> ()
    memref.dealloc %A : memref<1024x1024xf16>
    memref.dealloc %B_ref : memref<1024x1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
