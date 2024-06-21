// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  memref.global "private"  @__constant_8x16xf16 : memref<8x16xf16> = dense<1.0>
  memref.global "private"  @__constant_16x16xf16 : memref<16x16xf16> = dense<1.0>
  memref.global "private"  @__constant_16x16xf32 : memref<16x16xf32> = dense<0.0>

  func.func @test(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<8x16xf16>
    memref.copy %arg0, %memref : memref<8x16xf16> to memref<8x16xf16>
    %memref_0 = gpu.alloc  host_shared () : memref<16x16xf16>
    memref.copy %arg1, %memref_0 : memref<16x16xf16> to memref<16x16xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<8x16xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<8x16xf16>, %memref_0 : memref<16x16xf16>, %memref_1 : memref<8x16xf32>)
    gpu.dealloc  %memref : memref<8x16xf16>
    gpu.dealloc  %memref_0 : memref<16x16xf16>
    return %memref_1 : memref<8x16xf32>
  }

  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = xegpu.create_nd_tdesc %arg0[0, 0]
      : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      %1 = xegpu.create_nd_tdesc %arg1[0, 0]
      : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
      %2 = xegpu.create_nd_tdesc %arg2[0, 0]
      : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      %3 = xegpu.load_nd %0 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      %4 = xegpu.load_nd %1 {packed} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
      %5 = xegpu.load_nd %2 : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      %6 = xegpu.dpas %3, %4, %5 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      xegpu.store_nd %6,%2 : vector<8x16xf32>,!xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_8x16xf16 : memref<8x16xf16>
    %1 = memref.get_global @__constant_16x16xf16 : memref<16x16xf16>
    %ref = memref.get_global @__constant_16x16xf32 : memref<16x16xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index

    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %int0 = arith.index_cast %arg0 : index to i16
        %int1 = arith.index_cast %arg1 : index to i16
        %c16_i16 = arith.constant 16 : i16
        %idx0 = arith.muli %int0, %c16_i16 : i16
        %idx1 = arith.addi %int1, %idx0 : i16
        %fp = arith.uitofp %idx1 : i16 to f16
        %cst100 = arith.constant 1.0 : f16
        %val0 = arith.divf %fp, %cst100 : f16
        %cst1 = arith.constant 1.0 : f16
        %val1 = arith.addf %val0, %cst1 : f16
        memref.store %val0, %0[%arg0, %arg1] : memref<8x16xf16>
        memref.store %val1, %1[%arg0, %arg1] : memref<16x16xf16>
      }
    }
    // caculate the result C matrix
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %acc = memref.load %ref[%arg0, %arg1] : memref<16x16xf32>
        %res = scf.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %acc) -> f32 {
          %a = memref.load %0[%arg0, %arg2] : memref<8x16xf16>
          %b = memref.load %1[%arg2, %arg1] : memref<16x16xf16>
          %c = arith.mulf %a, %b : f16
          %cc = arith.extf %c : f16 to f32
          %ccc = arith.addf %cc, %arg3 : f32
          scf.yield %ccc : f32
        }
        memref.store %res, %ref[%arg0, %arg1] : memref<16x16xf32>
      }
    }

    %cast_ref = memref.cast %ref : memref<16x16xf32> to memref<*xf32>
    %2 = call @test(%0, %1) : (memref<8x16xf16>, memref<16x16xf16>) -> memref<8x16xf32>
    %cast = memref.cast %2 : memref<8x16xf32> to memref<*xf32>
    // call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    // CHECK:   [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast, %cast_ref) : (memref<*xf32>, memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
