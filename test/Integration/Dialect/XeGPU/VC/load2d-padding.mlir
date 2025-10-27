// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  // memref.global "private" constant @__constant_8x16xf32 : memref<8x16xf32> = dense<1.0>
  memref.global "private" constant @__constant_8x16xf32 : memref<8x16xf32> = dense<1.000000e+00>
  func.func @test(%arg0: memref<8x16xf32>, %arg1: index) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<8x16xf32>
    gpu.memcpy  %memref, %arg0 : memref<8x16xf32>, memref<8x16xf32>
    %memref_0 = gpu.alloc  () : memref<8x16xf32>
    gpu.launch_func  @test_kernel::@test_padding blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<8x16xf32>, %memref_0 : memref<8x16xf32>, %arg1 : index)
    gpu.dealloc  %memref : memref<8x16xf32>
    %alloc = memref.alloc() : memref<8x16xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<8x16xf32>, memref<8x16xf32>
    gpu.dealloc  %memref_0 : memref<8x16xf32>
    return %alloc : memref<8x16xf32>
  }
  gpu.module @test_kernel  {
    gpu.func @test_padding(%arg0: memref<8x16xf32>, %arg1: memref<8x16xf32>, %arg2: index) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = xegpu.create_nd_tdesc %arg0[%arg2, %arg2] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      %1 = xegpu.create_nd_tdesc %arg1[0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      %2 = xegpu.load_nd %0  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      xegpu.store_nd %2, %1  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c7 = arith.constant 7 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_8x16xf32 : memref<8x16xf32>
// CHECK: ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
// CHECK: ( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
    %1 = call @test(%0, %c1) : (memref<8x16xf32>, index) -> memref<8x16xf32>
    %2 = call @test(%0, %c2) : (memref<8x16xf32>, index) -> memref<8x16xf32>
    %3 = vector.load %1[%c7, %c0] : memref<8x16xf32>, vector<16xf32>
    vector.print %3 : vector<16xf32>
    %4 = vector.load %2[%c0, %c0] : memref<8x16xf32>, vector<16xf32>
    vector.print %4 : vector<16xf32>
    return
  }
}

