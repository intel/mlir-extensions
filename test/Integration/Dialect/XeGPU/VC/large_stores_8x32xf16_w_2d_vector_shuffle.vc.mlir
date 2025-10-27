// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<8x32xf16>) -> memref<8x32xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<8x32xf16>
    gpu.memcpy  %memref, %arg0 : memref<8x32xf16>, memref<8x32xf16>
    %memref_0 = gpu.alloc  () : memref<8x32xf16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<8x32xf16>, %memref_0 : memref<8x32xf16>)
    gpu.dealloc  %memref : memref<8x32xf16>
    %alloc = memref.alloc() : memref<8x32xf16>
    gpu.memcpy  %alloc, %memref_0 : memref<8x32xf16>, memref<8x32xf16>
    gpu.dealloc  %memref_0 : memref<8x32xf16>
    return %alloc : memref<8x32xf16>
  }
  gpu.module @test_kernel  {
    gpu.func @test_kernel(%arg0: memref<8x32xf16>, %arg1: memref<8x32xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x32xf16> -> !xegpu.tensor_desc<8x16xf16>
      %1 = xegpu.create_nd_tdesc %arg0[0, 16] : memref<8x32xf16> -> !xegpu.tensor_desc<8x16xf16>
      %2 = xegpu.load_nd %0 <{l1_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      %3 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      %4 = vector.shuffle %2, %3 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %5 = vector.shape_cast %4 : vector<16x16xf16> to vector<256xf16>
      %6 = vector.shape_cast %5 : vector<256xf16> to vector<8x32xf16>
      %7 = xegpu.create_nd_tdesc %arg1[0, 0] : memref<8x32xf16> -> !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %6, %7  : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // call @printMemrefF16(%A_cast): (memref<*xf16>) -> ()
    // call @printMemrefF16(%B_cast): (memref<*xf16>) -> ()
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant -5.000000e-01 : f32
    %false = arith.constant false
    %alloc = memref.alloc() : memref<8x32xf16>
    %cast = memref.cast %alloc : memref<8x32xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%cast, %cst_0, %cst, %false) : (memref<*xf16>, f32, f32, i1) -> ()
    %0 = call @test(%alloc) : (memref<8x32xf16>) -> memref<8x32xf16>
    %cast_1 = memref.cast %alloc : memref<8x32xf16> to memref<*xf16>
    %alloc_2 = memref.alloc() : memref<8x32xf32>
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c32 step %c1 {
        %1 = memref.load %0[%arg0, %arg1] : memref<8x32xf16>
        %2 = arith.extf %1 : f16 to f32
        memref.store %2, %alloc_2[%arg0, %arg1] : memref<8x32xf32>
      }
    }
    // CHECK: [ALLCLOSE: TRUE]
    %cast_3 = memref.cast %alloc_2 : memref<8x32xf32> to memref<*xf32>
    call @printAllcloseF16(%cast_1, %cast_3) : (memref<*xf16>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<8x32xf16>
    memref.dealloc %alloc_2 : memref<8x32xf32>
    memref.dealloc  %0 : memref<8x32xf16>
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF16(memref<*xf16>, f32) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

