// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  memref.global "private" constant @__constant_16x32xf16 : memref<16x32xf16> = dense<5.000000e-01>
  func.func @test(%arg0: memref<16x32xf16>) -> memref<16x32xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<16x32xf16>
    gpu.memcpy  %memref, %arg0 : memref<16x32xf16>, memref<16x32xf16>
    %memref_0 = gpu.alloc  () : memref<16x32xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<16x32xf16>, %memref_0 : memref<16x32xf32>)
    gpu.dealloc  %memref : memref<16x32xf16>
    %alloc = memref.alloc() : memref<16x32xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<16x32xf32>, memref<16x32xf32>
    gpu.dealloc  %memref_0 : memref<16x32xf32>
    return %alloc : memref<16x32xf32>
  }
  gpu.module @test_kernel  {
      // %16 = vector.extract %4[0, 0]: f32 from vector<16x16xf32>
      // %17 = vector.extract %5[0, 0]: f32 from vector<16x16xf32>
      // gpu.printf "\narray 0: %f, array 1: %f.\n" %16, %17: f32, f32
    gpu.func @test_kernel(%arg0: memref<16x32xf16>, %arg1: memref<16x32xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<16x32xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
      %1 = xegpu.load_nd %0 <{l1_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x16x16xf16>
      %2 = arith.extf %1 : vector<2x16x16xf16> to vector<2x16x16xf32>
      %3 = vector.extract %2[0] : vector<16x16xf32> from vector<2x16x16xf32>
      %4 = vector.extract %2[1] : vector<16x16xf32> from vector<2x16x16xf32>
      %5 = vector.shape_cast %3 : vector<16x16xf32> to vector<2x8x16xf32>
      %6 = vector.shape_cast %4 : vector<16x16xf32> to vector<2x8x16xf32>
      %7 = vector.extract %5[0] : vector<8x16xf32> from vector<2x8x16xf32>
      %8 = vector.extract %5[1] : vector<8x16xf32> from vector<2x8x16xf32>
      %9 = vector.extract %6[0] : vector<8x16xf32> from vector<2x8x16xf32>
      %10 = vector.extract %6[1] : vector<8x16xf32> from vector<2x8x16xf32>
      %11 = xegpu.create_nd_tdesc %arg1[0, 0] : memref<16x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      %12 = xegpu.create_nd_tdesc %arg1[0, 16] : memref<16x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      %13 = xegpu.create_nd_tdesc %arg1[8, 0] : memref<16x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      %14 = xegpu.create_nd_tdesc %arg1[8, 16] : memref<16x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %7, %11  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %9, %12  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %8, %13  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %10, %14  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    //call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant -5.000000e-01 : f32
    %false = arith.constant false
    %alloc = memref.alloc() : memref<16x32xf16>
    %cast = memref.cast %alloc : memref<16x32xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%cast, %cst_0, %cst, %false) : (memref<*xf16>, f32, f32, i1) -> ()
    %0 = call @test(%alloc) : (memref<16x32xf16>) -> memref<16x32xf32>
    %cast_1 = memref.cast %alloc : memref<16x32xf16> to memref<*xf16>
    %cast_2 = memref.cast %0 : memref<16x32xf32> to memref<*xf32>
    call @printAllcloseF16(%cast_1, %cast_2) : (memref<*xf16>, memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

