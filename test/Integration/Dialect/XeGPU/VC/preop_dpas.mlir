// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  memref.global "private" @__constant_32x32xf16 : memref<32x32xf16> = dense<1.0>
  memref.global "private" @__Bconstant_32x32xf16 : memref<32x32xf16> = dense<2.0>
  func.func @test(%arg0: memref<32x32xf16>, %arg1: memref<32x32xf16>) -> memref<32x32xf32> attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index

    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<32x32xf16>
    memref.copy %arg0, %memref : memref<32x32xf16> to memref<32x32xf16>
    %memref_0 = gpu.alloc  host_shared () : memref<32x32xf16>
    memref.copy %arg1, %memref_0 : memref<32x32xf16> to memref<32x32xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<32x32xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c4, %c2, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<32x32xf16>, %memref_0 : memref<32x32xf16>, %memref_1 : memref<32x32xf32>)
    gpu.dealloc  %memref : memref<32x32xf16>
    gpu.dealloc  %memref_0 : memref<32x32xf16>
    return %memref_1 : memref<32x32xf32>
  }

gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%A: memref<32x32xf16>, %B: memref<32x32xf16>, %C: memref<32x32xf32>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 4, 2, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {


      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %cst = arith.constant dense<1.0> : vector<8x16xf16>
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y

       %2 = arith.muli %0, %c8 : index
      %3 = arith.muli %1, %c16 : index

      %4 = xegpu.create_nd_tdesc %C[%2, %3] : memref<32x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      %5 = xegpu.load_nd %4 : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>

      %6 = scf.for %arg3 = %c0 to %c32 step %c16 iter_args(%arg4 = %5) -> (vector<8x16xf32>) {
        %A0 = xegpu.create_nd_tdesc %A[%2, %arg3] : memref<32x32xf16> -> !xegpu.tensor_desc<8x16xf16>
        %A0_val = xegpu.load_nd %A0 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>

        %B0 = xegpu.create_nd_tdesc %B[%arg3, %3] : memref<32x32xf16> -> !xegpu.tensor_desc<16x16xf16>
        %B0_val = xegpu.load_nd %B0 {packed} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>

        %A0_preop = arith.addf %A0_val, %cst : vector<8x16xf16>

        %dpas0 = xegpu.dpas %A0_preop, %B0_val , %arg4: vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        scf.yield %dpas0 : vector<8x16xf32>
      }
      xegpu.store_nd %6, %4 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>

      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index

    %A = memref.get_global @__constant_32x32xf16 : memref<32x32xf16>
    %B = memref.get_global @__Bconstant_32x32xf16 : memref<32x32xf16>
    %C_ref = memref.alloc() : memref<32x32xf32>

    // caculate the result C matrix
    scf.for %i = %c0 to %c32 step %c1 {
      scf.for %j = %c0 to %c32 step %c1 {
        %acc = arith.constant 0.0 : f32
        %res = scf.for %k = %c0 to %c32 step %c1 iter_args(%acc1 = %acc) -> f32 {
          %a = memref.load %A[%i, %k] : memref<32x32xf16>
          %b = memref.load %B[%k, %j] : memref<32x32xf16>
          // adjust for preop in GPU kernel, where we add 1 between load and dpas
          %cst1 = arith.constant 1.0 : f16
          %a_adj = arith.addf %a, %cst1 : f16
          %c = arith.mulf %a_adj, %b : f16
          %cc = arith.extf %c : f16 to f32
          %ccc = arith.addf %cc, %acc1 : f32
          scf.yield %ccc : f32
        }
        memref.store %res, %C_ref[%i, %j] : memref<32x32xf32>
      }
    }

   %2 = call @test(%A, %B) : (memref<32x32xf16>, memref<32x32xf16>) -> memref<32x32xf32>
    %cast = memref.cast %2 : memref<32x32xf32> to memref<*xf32>
    // call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    %cast_ref = memref.cast %C_ref : memref<32x32xf32> to memref<*xf32>
    // call @printMaxErrorF32(%cast, %cast_ref) : (memref<*xf32>, memref<*xf32>) -> ()
    // call @printMemrefF32(%cast_ref) : (memref<*xf32>) -> ()
    // CHECK:   [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast, %cast_ref) : (memref<*xf32>, memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMaxErrorF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
