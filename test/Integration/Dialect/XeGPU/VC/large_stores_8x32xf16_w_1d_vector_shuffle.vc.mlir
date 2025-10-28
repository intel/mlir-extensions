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
      %4 = vector.shape_cast %2 : vector<8x16xf16> to vector<128xf16>
      %5 = vector.shape_cast %3 : vector<8x16xf16> to vector<128xf16>
      %6 = vector.shuffle %4, %5 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<128xf16>, vector<128xf16>
      %7 = vector.shape_cast %6 : vector<256xf16> to vector<8x32xf16>
      %8 = xegpu.create_nd_tdesc %arg1[0, 0] : memref<8x32xf16> -> !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %7, %8  : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
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
