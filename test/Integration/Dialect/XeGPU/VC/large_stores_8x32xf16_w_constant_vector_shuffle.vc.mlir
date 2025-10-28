// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test() -> memref<8x32xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<8x32xf16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<8x32xf16>)
    %alloc = memref.alloc() : memref<8x32xf16>
    gpu.memcpy  %alloc, %memref : memref<8x32xf16>, memref<8x32xf16>
    gpu.dealloc  %memref : memref<8x32xf16>
    return %alloc : memref<8x32xf16>
  }
  gpu.module @test_kernel  {
      // do a vector shuffle with two constant vectors that have different number of elements.
    gpu.func @test_kernel(%arg0: memref<8x32xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %cst = arith.constant dense<1.000000e+00> : vector<192xf16>
      %cst_0 = arith.constant dense<2.000000e+00> : vector<64xf16>
      %0 = vector.shape_cast %cst : vector<192xf16> to vector<12x16xf16>
      %1 = vector.shape_cast %cst_0 : vector<64xf16> to vector<4x16xf16>
      %2 = vector.shuffle %0, %1 [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 9, 13, 10, 14, 11, 15] : vector<12x16xf16>, vector<4x16xf16>
      %3 = vector.shape_cast %2 : vector<16x16xf16> to vector<256xf16>
      %4 = vector.shape_cast %3 : vector<256xf16> to vector<8x32xf16>
      %5 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x32xf16> -> !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %4, %5  : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = call @test() : () -> memref<8x32xf16>
    %alloc = memref.alloc() : memref<8x32xf32>
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c32 step %c1 {
        %1 = memref.load %0[%arg0, %arg1] : memref<8x32xf16>
        %2 = arith.extf %1 : f16 to f32
        memref.store %2, %alloc[%arg0, %arg1] : memref<8x32xf32>
      }
    }
    // CHECK: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
    // CHECK-NEXT: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
    // CHECK-NEXT: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
    // CHECK-NEXT: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
    // CHECK-NEXT: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2],
    // CHECK-NEXT: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2],
    // CHECK-NEXT: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2],
    // CHECK-NEXT: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2]
    %cast = memref.cast %alloc : memref<8x32xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<8x32xf32>
    gpu.dealloc  %0 : memref<8x32xf16>
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF16(memref<*xf16>, f32) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
