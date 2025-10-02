// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<32x32xf16>) -> memref<32x32xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %A_gpu = gpu.alloc  host_shared () : memref<32x32xf16>
    memref.copy %A, %A_gpu : memref<32x32xf16> to memref<32x32xf16>
    %B_gpu = gpu.alloc  host_shared () : memref<32x32xf16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c4, %c2, %c1) args(%A_gpu : memref<32x32xf16>, %B_gpu : memref<32x32xf16>)
    gpu.dealloc  %A_gpu : memref<32x32xf16>
    return %B_gpu : memref<32x32xf16>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%A: memref<32x32xf16>, %B: memref<32x32xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index

      %tid_x = gpu.thread_id x
      %tid_y = gpu.thread_id y
      %m = arith.muli %tid_x, %c8 : index
      %n = arith.muli %tid_y, %c16 : index

      %a_tile = xegpu.create_nd_tdesc %A[%m, %n] : memref<32x32xf16> -> !xegpu.tensor_desc<8x16xf16>
      %a = xegpu.load_nd %a_tile : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>

      %slm = memref.alloc() : memref<2048xi8, 3>

      %mem_desc_store = xegpu.create_mem_desc %slm : memref<2048xi8, 3> -> !xegpu.mem_desc<32x32xf16, #xegpu.mem_layout<stride = [1, 32], block = [16, 16]>>
      xegpu.store_matrix %a, %mem_desc_store[%m, %n] : vector<8x16xf16>, !xegpu.mem_desc<32x32xf16, #xegpu.mem_layout<stride = [1, 32], block = [16, 16]>>, index, index

      gpu.barrier

      %mem_desc_load = xegpu.create_mem_desc %slm : memref<2048xi8, 3> -> !xegpu.mem_desc<32x32xf16, #xegpu.mem_layout<block = [16, 16]>>
      %d = xegpu.load_matrix %mem_desc_load[%m, %n] : !xegpu.mem_desc<32x32xf16, #xegpu.mem_layout<block = [16, 16]>>, index, index -> vector<8x16xf16>

      %b_tile = xegpu.create_nd_tdesc %B[%m, %n] : memref<32x32xf16> -> !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %d, %b_tile: vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %cf_0 = arith.constant 0.0 : f16
    %cf_1 = arith.constant 1.0 : f16
    %A = memref.alloc() : memref<32x32xf16>
    %Ref = memref.alloc() : memref<32x32xf32>

    // intialize matrix A ; A[i, j] = j
    scf.for %i = %c0 to %c32 step %c1 {
      scf.for %j = %c0 to %c32 step %c1 {
        %mul = arith.muli %i, %c32 : index
        %add = arith.addi %mul, %j : index
        %t = index.castu %add : index to i16
        %val = arith.uitofp %t : i16 to f16
        memref.store %val, %A[%i, %j] : memref<32x32xf16>
        %t32 = index.castu %add : index to i32
        %val32 = arith.uitofp %t32 : i32 to f32
        memref.store %val32, %Ref[%j, %i] : memref<32x32xf32>
      }
    }

    %B = call @test(%A) : (memref<32x32xf16>) -> memref<32x32xf16>
    %cast = memref.cast %B : memref<32x32xf16> to memref<*xf16>
    // call @printMemrefF16(%cast) : (memref<*xf16>) -> ()
    %cast_ref = memref.cast %Ref : memref<32x32xf32> to memref<*xf32>
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF16(%cast, %cast_ref) : (memref<*xf16>, memref<*xf32>) -> ()
    memref.dealloc %A : memref<32x32xf16>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
