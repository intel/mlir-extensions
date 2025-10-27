// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<16xf32>) -> memref<16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<16xf32>
    gpu.memcpy  %memref, %arg0 : memref<16xf32>, memref<16xf32>
    %memref_0 = gpu.alloc  () : memref<16xf32>
    gpu.launch_func  @test_kernel::@test_atomic_rmw blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<16xf32>, %memref_0 : memref<16xf32>)
    %alloc = memref.alloc() : memref<16xf32>
    gpu.memcpy  %alloc, %memref : memref<16xf32>, memref<16xf32>
    gpu.dealloc  %memref : memref<16xf32>
    return %alloc : memref<16xf32>
  }
  gpu.module @test_kernel  {
    gpu.func @test_atomic_rmw(%arg0: memref<16xf32>, %arg1: memref<16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %cst = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01]> : vector<16xf32>
      %cst_0 = arith.constant dense<true> : vector<16xi1>
      %cst_1 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>
      %0 = xegpu.create_tdesc %arg0, %cst_1 : memref<16xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
      %1 = xegpu.atomic_rmw addf %0, %cst_0, %cst : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>, vector<16xf32> -> vector<16xf32>
      %2 = xegpu.create_tdesc %arg1, %cst_1 : memref<16xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
      xegpu.store %1, %2, %cst_0  : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 1.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<16xf32>
    scf.for %arg0 = %c0 to %c16 step %c1 {
      memref.store %cst, %alloc[%arg0] : memref<16xf32>
    }
    //CHECK: [1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15, 16]
    %0 = call @test(%alloc) : (memref<16xf32>) -> memref<16xf32>
    %cast = memref.cast %0 : memref<16xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}

