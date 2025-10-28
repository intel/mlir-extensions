// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<1x32xf16>) -> memref<1x32xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<1x32xf16>
    gpu.memcpy  %memref, %arg0 : memref<1x32xf16>, memref<1x32xf16>
    %memref_0 = gpu.alloc  () : memref<1x32xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<1x32xf16>, %memref_0 : memref<1x32xf32>)
    gpu.dealloc  %memref : memref<1x32xf16>
    %alloc = memref.alloc() : memref<1x32xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<1x32xf32>, memref<1x32xf32>
    gpu.dealloc  %memref_0 : memref<1x32xf32>
    return %alloc : memref<1x32xf32>
  }
  gpu.module @test_kernel  {
    gpu.func @test_kernel(%arg0: memref<1x32xf16>, %arg1: memref<1x32xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<1x32xf16> -> !xegpu.tensor_desc<1x16xf16>
      %1 = xegpu.create_nd_tdesc %arg0[0, 16] : memref<1x32xf16> -> !xegpu.tensor_desc<1x16xf16>
      %2 = xegpu.load_nd %0  : !xegpu.tensor_desc<1x16xf16> -> vector<1x16xf16>
      %3 = xegpu.load_nd %1  : !xegpu.tensor_desc<1x16xf16> -> vector<1x16xf16>
      %4 = arith.extf %2 : vector<1x16xf16> to vector<1x16xf32>
      %5 = arith.extf %3 : vector<1x16xf16> to vector<1x16xf32>
      %6 = xegpu.create_nd_tdesc %arg1[0, 0] : memref<1x32xf32> -> !xegpu.tensor_desc<1x16xf32>
      %7 = xegpu.create_nd_tdesc %arg1[0, 16] : memref<1x32xf32> -> !xegpu.tensor_desc<1x16xf32>
      xegpu.store_nd %4, %6  : vector<1x16xf32>, !xegpu.tensor_desc<1x16xf32>
      xegpu.store_nd %5, %7  : vector<1x16xf32>, !xegpu.tensor_desc<1x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // call @printMemrefF16(%A_cast) : (memref<*xf16>) -> ()
    // call @printMemrefF32(%B_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant -2.000000e+00 : f32
    %true = arith.constant true
    %alloc = memref.alloc() : memref<1x32xf16>
    %cast = memref.cast %alloc : memref<1x32xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%cast, %cst_0, %cst, %true) : (memref<*xf16>, f32, f32, i1) -> ()
    %0 = call @test(%alloc) : (memref<1x32xf16>) -> memref<1x32xf32>
    %cast_1 = memref.cast %alloc : memref<1x32xf16> to memref<*xf16>
    %cast_2 = memref.cast %0 : memref<1x32xf32> to memref<*xf32>
    call @printAllcloseF16(%cast_1, %cast_2) : (memref<*xf16>, memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
