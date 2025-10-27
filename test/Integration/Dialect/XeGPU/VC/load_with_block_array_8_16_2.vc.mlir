// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  memref.global "private" constant @__constant_8x32xf16 : memref<8x32xf16> = dense<5.000000e-01>
  func.func @test(%arg0: memref<8x32xf16>) -> memref<8x32xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<8x32xf16>
    gpu.memcpy  %memref, %arg0 : memref<8x32xf16>, memref<8x32xf16>
    %memref_0 = gpu.alloc  () : memref<8x32xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<8x32xf16>, %memref_0 : memref<8x32xf32>)
    gpu.dealloc  %memref : memref<8x32xf16>
    %alloc = memref.alloc() : memref<8x32xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<8x32xf32>, memref<8x32xf32>
    gpu.dealloc  %memref_0 : memref<8x32xf32>
    return %alloc : memref<8x32xf32>
  }
  gpu.module @test_kernel  {
    gpu.func @test_kernel(%arg0: memref<8x32xf16>, %arg1: memref<8x32xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x32xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
      %1 = xegpu.create_nd_tdesc %arg1[0, 0] : memref<8x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      %2 = xegpu.create_nd_tdesc %arg1[0, 16] : memref<8x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      %3 = xegpu.load_nd %0 <{l1_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x8x16xf16>
      %4 = vector.extract %3[0] : vector<8x16xf16> from vector<2x8x16xf16>
      %5 = vector.extract %3[1] : vector<8x16xf16> from vector<2x8x16xf16>
      %6 = arith.extf %4 : vector<8x16xf16> to vector<8x16xf32>
      %7 = arith.extf %5 : vector<8x16xf16> to vector<8x16xf32>
      xegpu.store_nd %6, %1  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %7, %2  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant -5.000000e-01 : f32
    %false = arith.constant false
    %alloc = memref.alloc() : memref<8x32xf16>
    %cast = memref.cast %alloc : memref<8x32xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%cast, %cst_0, %cst, %false) : (memref<*xf16>, f32, f32, i1) -> ()
    %0 = call @test(%alloc) : (memref<8x32xf16>) -> memref<8x32xf32>
    %cast_1 = memref.cast %alloc : memref<8x32xf16> to memref<*xf16>
    %cast_2 = memref.cast %0 : memref<8x32xf32> to memref<*xf32>
    call @printAllcloseF16(%cast_1, %cast_2) : (memref<*xf16>, memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

