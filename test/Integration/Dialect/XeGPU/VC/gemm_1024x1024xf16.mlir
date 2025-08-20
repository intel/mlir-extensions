// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  memref.global "private" @__constant_1024x1024xf16 : memref<1024x1024xf16> = dense<0.0>
  memref.global "private" @__constant_1024x1024xf16_ : memref<1024x1024xf16> = dense<0.0>
  memref.global "private" @__constant_1024x1024xf32 : memref<1024x1024xf32> = dense<0.0>
  func.func @test(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>) -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %arg0, %memref : memref<1024x1024xf16> to memref<1024x1024xf16>
    %memref_0 = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %arg1, %memref_0 : memref<1024x1024xf16> to memref<1024x1024xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<1024x1024xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c128, %c64, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<1024x1024xf16>, %memref_0 : memref<1024x1024xf16>, %memref_1 : memref<1024x1024xf32>)
    gpu.dealloc  %memref : memref<1024x1024xf16>
    gpu.dealloc  %memref_0 : memref<1024x1024xf16>
    return %memref_1 : memref<1024x1024xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 128, 64, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1024 = arith.constant 1024 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = arith.muli %0, %c8 : index
      %3 = arith.muli %1, %c16 : index
      %4 = xegpu.create_nd_tdesc %arg2[%2, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
      %5 = xegpu.load_nd %4 : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      // each work-group has 1 subgroup. the subgroup caculates a [8x16 = 8x1024 * 1024x16] block
      %6 = scf.for %arg3 = %c0 to %c1024 step %c16 iter_args(%arg4 = %5) -> (vector<8x16xf32>) {
        %7 = xegpu.create_nd_tdesc %arg0[%2, %arg3] : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
        %8 = xegpu.create_nd_tdesc %arg1[%arg3, %3] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
        %9 = xegpu.load_nd %7 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
        %10 = xegpu.load_nd %8 {packed} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %11 = xegpu.dpas %9, %10, %arg4 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        scf.yield %11 : vector<8x16xf32>
      }
      xegpu.store_nd %6, %4 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_1024x1024xf16 : memref<1024x1024xf16>
    %1 = memref.get_global @__constant_1024x1024xf16_ : memref<1024x1024xf16>
    %ref = memref.get_global @__constant_1024x1024xf32 : memref<1024x1024xf32>
    %init = arith.constant 0.0 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    // fill the top-left block 128x128
    // A matrix: row-major, start from 0.0, increase 0.01 per element
    // B matrix: A matrix + 1.0
    scf.for %arg0 = %c0 to %c128 step %c1 {
      scf.for %arg1 = %c0 to %c128 step %c1 {
        %int0 = arith.index_cast %arg0 : index to i16
        %int1 = arith.index_cast %arg1 : index to i16
        %c128_i16 = arith.constant 128 : i16
        %idx0 = arith.muli %int0, %c128_i16 : i16
        %idx1 = arith.addi %int1, %idx0 : i16
        %fp = arith.uitofp %idx1 : i16 to f16
        %cst100 = arith.constant 100.0 : f16
        %val0 = arith.divf %fp, %cst100 : f16
        %cst1 = arith.constant 1.0 : f16
        %val1 = arith.addf %val0, %cst1 : f16
        memref.store %val0, %0[%arg0, %arg1] : memref<1024x1024xf16>
        memref.store %val1, %1[%arg0, %arg1] : memref<1024x1024xf16>
      }
    }
    // caculate the result C matrix
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %acc = memref.load %ref[%arg0, %arg1] : memref<1024x1024xf32>
        %res = scf.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %acc) -> f32 {
          %a = memref.load %0[%arg0, %arg2] : memref<1024x1024xf16>
          %b = memref.load %1[%arg2, %arg1] : memref<1024x1024xf16>
          %c = arith.mulf %a, %b : f16
          %cc = arith.extf %c : f16 to f32
          %ccc = arith.addf %cc, %arg3 : f32
          scf.yield %ccc : f32
        }
        memref.store %res, %ref[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }

    %2 = call @test(%0, %1) : (memref<1024x1024xf16>, memref<1024x1024xf16>) -> memref<1024x1024xf32>
    %cast = memref.cast %2 : memref<1024x1024xf32> to memref<*xf32>
    //call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    %cast_ref = memref.cast %ref : memref<1024x1024xf32> to memref<*xf32>
    //call @printMemrefF32(%cast_ref) : (memref<*xf32>) -> ()
    // CHECK:   [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast, %cast_ref) : (memref<*xf32>, memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
