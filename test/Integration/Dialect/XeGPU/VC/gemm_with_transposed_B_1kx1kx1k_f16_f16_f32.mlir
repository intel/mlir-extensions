// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  memref.global "private" @__constant_1024x1024xf16 : memref<1024x1024xf16> = dense<0.000000e+00>
  memref.global "private" @__constant_1024x1024xf16_ : memref<1024x1024xf16> = dense<0.000000e+00>
  memref.global "private" @__constant_1024x1024xf32 : memref<1024x1024xf32> = dense<0.000000e+00>
  func.func @test(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>) -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<1024x1024xf16>
    gpu.memcpy  %memref, %arg0 : memref<1024x1024xf16>, memref<1024x1024xf16>
    %memref_0 = gpu.alloc  () : memref<1024x1024xf16>
    gpu.memcpy  %memref_0, %arg1 : memref<1024x1024xf16>, memref<1024x1024xf16>
    %memref_1 = gpu.alloc  () : memref<1024x1024xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c128, %c64, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<1024x1024xf16>, %memref_0 : memref<1024x1024xf16>, %memref_1 : memref<1024x1024xf32>)
    gpu.dealloc  %memref : memref<1024x1024xf16>
    gpu.dealloc  %memref_0 : memref<1024x1024xf16>
    %alloc = memref.alloc() : memref<1024x1024xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<1024x1024xf32>, memref<1024x1024xf32>
    gpu.dealloc  %memref_1 : memref<1024x1024xf32>
    return %alloc : memref<1024x1024xf32>
  }
  gpu.module @test_kernel  {
    gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 128, 64, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1024 = arith.constant 1024 : index
      // each work-group has 1 subgroup. the subgroup caculates a [8x16 = 8x1024 * 1024x16] block
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c8 : index
      %1 = arith.muli %block_id_y, %c16 : index
      %2 = xegpu.create_nd_tdesc %arg2[%0, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
      %3 = xegpu.load_nd %2  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      %4 = scf.for %arg3 = %c0 to %c1024 step %c16 iter_args(%arg4 = %3) -> (vector<8x16xf32>) {
        %5 = xegpu.create_nd_tdesc %arg0[%0, %arg3] : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
        %6 = xegpu.create_nd_tdesc %arg1[%1, %arg3] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
        %7 = xegpu.load_nd %5  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
        %8 = xegpu.load_nd %6 <{transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %9 = xegpu.dpas %7, %8, %arg4 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        scf.yield %9 : vector<8x16xf32>
      }
      xegpu.store_nd %4, %2  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 1.000000e+00 : f16
    %cst_0 = arith.constant 1.000000e+02 : f16
    %c128_i16 = arith.constant 128 : i16
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_1024x1024xf16 : memref<1024x1024xf16>
    %1 = memref.get_global @__constant_1024x1024xf16_ : memref<1024x1024xf16>
    // fill the top-left block 128x128
    // A matrix: row-major, start from 0.0, increase 0.01 per element
    // B matrix: A matrix + 1.0
    %2 = memref.get_global @__constant_1024x1024xf32 : memref<1024x1024xf32>
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %4 = arith.index_cast %arg0 : index to i16
        %5 = arith.index_cast %arg1 : index to i16
        %6 = arith.muli %4, %c128_i16 : i16
        %7 = arith.addi %5, %6 : i16
        %8 = arith.uitofp %7 : i16 to f16
        %9 = arith.divf %8, %cst_0 : f16
        %10 = arith.addf %9, %cst : f16
        memref.store %9, %0[%arg0, %arg1] : memref<1024x1024xf16>
        memref.store %10, %1[%arg0, %arg1] : memref<1024x1024xf16>
      }
    }
    // caculate the result C matrix
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
          // %cc = arith.extf %c : f16 to f32
        %4 = memref.load %2[%arg0, %arg1] : memref<1024x1024xf32>
        %5 = scf.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %4) -> (f32) {
          %6 = memref.load %0[%arg0, %arg2] : memref<1024x1024xf16>
          %7 = memref.load %1[%arg1, %arg2] : memref<1024x1024xf16>
          %8 = arith.extf %6 : f16 to f32
          %9 = arith.extf %7 : f16 to f32
          %10 = arith.mulf %8, %9 : f32
          %11 = arith.addf %10, %arg3 : f32
          scf.yield %11 : f32
        }
        memref.store %5, %2[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    // call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%cast_ref) : (memref<*xf32>) -> ()
    // CHECK:   [ALLCLOSE: TRUE]
    %3 = call @test(%0, %1) : (memref<1024x1024xf16>, memref<1024x1024xf16>) -> memref<1024x1024xf32>
    %cast = memref.cast %3 : memref<1024x1024xf32> to memref<*xf32>
    %cast_1 = memref.cast %2 : memref<1024x1024xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_1) : (memref<*xf32>, memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

