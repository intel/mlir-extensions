// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=opencl-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%opencl_runtime --filecheck
module @softmax attributes {gpu.container_module} {
  func.func @reduce_test(%a: memref<16x1024xf32>) -> memref<1x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index

    %a_gpu = gpu.alloc host_shared () : memref<16x1024xf32>
    memref.copy %a, %a_gpu : memref<16x1024xf32> to memref<16x1024xf32>
    %b_gpu = gpu.alloc  host_shared () : memref<1x1024xf32>

    gpu.launch_func @kernel::@reduce_dim_1 blocks in (%c1, %c32, %c1) threads in (%c1, %c1, %c1) args(%a_gpu : memref<16x1024xf32>, %b_gpu : memref<1x1024xf32>)

    gpu.dealloc %a_gpu : memref<16x1024xf32>
    return %b_gpu : memref<1x1024xf32>
  }

  gpu.module @kernel  attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    // the kernel is a 16x32 block reduction. each thread is assigned with a 16x32 block, and do reduction along dim-0 independently.
    gpu.func @reduce_dim_1(%a: memref<16x1024xf32>, %b: memref<1x1024xf32>)  kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index

      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y

      %m = arith.muli %block_id_x, %c16 : index
      %n = arith.muli %block_id_y, %c32 : index
      %1 = xetile.init_tile %a[%m, %n] : memref<16x1024xf32> -> !xetile.tile<16x32xf32>
      %2 = xetile.load_tile %1: !xetile.tile<16x32xf32> -> vector<16x32xf32>
      %4 = xetile.reduce <add>, %2 [0]: vector<16x32xf32> -> vector<1x32xf32>
      %5 = xetile.init_tile %b[0, %n] : memref<1x1024xf32> -> !xetile.tile<1x32xf32>
      xetile.store_tile %4, %5: vector<1x32xf32>, !xetile.tile<1x32xf32>
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %c0_f32 = arith.constant 0.0 : f32
    %c32_f32 = arith.constant 32.0 : f32
    %c100_f32 = arith.constant 100.0 : f32
    %a = memref.alloc() : memref<16x1024xf32>
    %b_ref = memref.alloc() : memref<1024xf32>

    // intialize matrix A ; A[i, j] = j
    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        %t = index.castu %j : index to i16
        %u = arith.uitofp %t : i16 to f32
        %v = arith.divf %u, %c100_f32 : f32
        memref.store %v, %a[%i, %j] : memref<16x1024xf32>
      }
    }

    scf.for %j = %c0 to %c1024 step %c1 {
      %sum = scf.for %i = %c0 to %c16 step %c1 iter_args(%arg = %c0_f32) -> (f32) {
        %val = memref.load %a[%i, %j] : memref<16x1024xf32>
        %2 = arith.addf %arg, %val : f32
        scf.yield %2 : f32
      }
      memref.store %sum, %b_ref[%j] : memref<1024xf32>
    }

    %b = call @reduce_test(%a) : (memref<16x1024xf32>) -> memref<1x1024xf32>
    %cast_b = memref.cast %b : memref<1x1024xf32> to memref<*xf32>
    %cast_b_ref = memref.cast %b_ref : memref<1024xf32> to memref<*xf32>
    // call @printMemrefF32(%cast_b): (memref<*xf32>) -> ()
    // call @printMemrefF32(%cast_b_ref): (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast_b, %cast_b_ref) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %a : memref<16x1024xf32>
    memref.dealloc %b_ref : memref<1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
