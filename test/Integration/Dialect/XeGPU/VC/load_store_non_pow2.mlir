// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @loadstore attributes {gpu.container_module} {
  func.func @test(%arg0: memref<8x16xf32>, %arg1: memref<8x16xf32>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<8x16xf32>
    %memref_0 = gpu.alloc  () : memref<8x16xf32>
    gpu.memcpy  %memref, %arg0 : memref<8x16xf32>, memref<8x16xf32>
    gpu.memcpy  %memref_0, %arg1 : memref<8x16xf32>, memref<8x16xf32>
    gpu.launch_func  @module::@test blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<8x16xf32>, %memref_0 : memref<8x16xf32>)
    gpu.dealloc  %memref : memref<8x16xf32>
    %alloc = memref.alloc() : memref<8x16xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<8x16xf32>, memref<8x16xf32>
    gpu.dealloc  %memref_0 : memref<8x16xf32>
    return %alloc : memref<8x16xf32>
  }
  gpu.module @module  {
    gpu.func @test(%arg0: memref<8x16xf32>, %arg1: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      // load A tile
      // store to B tile
      %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      %2 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %1, %2  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // TRY 8x15. While it can encode vector type to 120f32 for intrinsics, the result is wrong.
    // fill A with 2, B with 0
    // Load from A, store to B
    // call @printMemrefF32(%A_nonzero) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%B_filled) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %alloc = memref.alloc() : memref<8x16xf32>
    %alloc_1 = memref.alloc() : memref<8x16xf32>
    %cast = memref.cast %alloc : memref<8x16xf32> to memref<*xf32>
    %cast_2 = memref.cast %alloc_1 : memref<8x16xf32> to memref<*xf32>
    call @fillResource1DF32(%cast, %cst_0) : (memref<*xf32>, f32) -> ()
    call @fillResource1DF32(%cast_2, %cst) : (memref<*xf32>, f32) -> ()
    %0 = call @test(%alloc, %alloc_1) : (memref<8x16xf32>, memref<8x16xf32>) -> memref<8x16xf32>
    %cast_3 = memref.cast %0 : memref<8x16xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_3) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<8x16xf32>
    memref.dealloc %alloc_1 : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF32(memref<*xf32>, f32) attributes {llvm.emit_c_interface}
}
