// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  memref.global "private" @__constant_32x32xf16 : memref<32x32xf16> = dense<1.000000e+00>
  memref.global "private" @__constant_B32x32xf16 : memref<32x32xf16> = dense<2.000000e+00>
  memref.global "private" @__constant_1x32xf16 : memref<1x32xf16> = dense<1.000000e+01>
  func.func @test(%arg0: memref<32x32xf16>, %arg1: memref<32x32xf16>, %arg2: memref<1x32xf16>) -> memref<32x32xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %memref = gpu.alloc  () : memref<32x32xf16>
    %memref_0 = gpu.alloc  () : memref<32x32xf16>
    %memref_1 = gpu.alloc  () : memref<1x32xf16>
    gpu.memcpy  %memref, %arg0 : memref<32x32xf16>, memref<32x32xf16>
    gpu.memcpy  %memref_0, %arg1 : memref<32x32xf16>, memref<32x32xf16>
    gpu.memcpy  %memref_1, %arg2 : memref<1x32xf16>, memref<1x32xf16>
    %memref_2 = gpu.alloc  () : memref<32x32xf32>
    gpu.launch_func  @module0::@test_kernel blocks in (%c4, %c2, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<32x32xf16>, %memref_0 : memref<32x32xf16>, %memref_2 : memref<32x32xf32>, %memref_1 : memref<1x32xf16>)
    gpu.dealloc  %memref : memref<32x32xf16>
    gpu.dealloc  %memref_0 : memref<32x32xf16>
    gpu.dealloc  %memref_1 : memref<1x32xf16>
    %alloc = memref.alloc() : memref<32x32xf32>
    gpu.memcpy  %alloc, %memref_2 : memref<32x32xf32>, memref<32x32xf32>
    gpu.dealloc  %memref_2 : memref<32x32xf32>
    return %alloc : memref<32x32xf32>
  }
  gpu.module @module0  {
    gpu.func @test_kernel(%arg0: memref<32x32xf16>, %arg1: memref<32x32xf16>, %arg2: memref<32x32xf32>, %arg3: memref<1x32xf16>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 4, 2, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
        // load A tile
        // load B tile
        // load B cast
        // extract first 8 elems
        // reshape and broadcast over col dim
        // add to A
        // do DPAS
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c8 : index
      %1 = arith.muli %block_id_y, %c16 : index
      %2 = xegpu.create_nd_tdesc %arg2[%0, %1] : memref<32x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      %3 = xegpu.load_nd %2  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      %4 = scf.for %arg4 = %c0 to %c32 step %c16 iter_args(%arg5 = %3) -> (vector<8x16xf32>) {
        %5 = xegpu.create_nd_tdesc %arg0[%0, %arg4] : memref<32x32xf16> -> !xegpu.tensor_desc<8x16xf16>
        %6 = xegpu.load_nd %5  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
        %7 = xegpu.create_nd_tdesc %arg1[%arg4, %1] : memref<32x32xf16> -> !xegpu.tensor_desc<16x16xf16>
        %8 = xegpu.load_nd %7 <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %9 = xegpu.create_nd_tdesc %arg3[%c0, %c0] : memref<1x32xf16> -> !xegpu.tensor_desc<1x32xf16>
        %10 = xegpu.load_nd %9  : !xegpu.tensor_desc<1x32xf16> -> vector<1x32xf16>
        %11 = vector.extract_strided_slice %10 {offsets = [0, 0], sizes = [1, 8], strides = [1, 1]} : vector<1x32xf16> to vector<1x8xf16>
        %12 = vector.shape_cast %11 : vector<1x8xf16> to vector<8xf16>
        %13 = vector.shape_cast %12 : vector<8xf16> to vector<8x1xf16>
        %14 = vector.broadcast %13 : vector<8x1xf16> to vector<8x16xf16>
        %15 = arith.addf %6, %14 : vector<8x16xf16>
        %16 = xegpu.dpas %15, %8, %arg5 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        scf.yield %16 : vector<8x16xf32>
      }
      // store
      xegpu.store_nd %4, %2  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // init constants
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    // random init
    // run GPU version
    // run CPU version
    %0 = memref.get_global @__constant_32x32xf16 : memref<32x32xf16>
    %1 = memref.get_global @__constant_B32x32xf16 : memref<32x32xf16>
    %2 = memref.get_global @__constant_1x32xf16 : memref<1x32xf16>
    %alloc = memref.alloc() : memref<32x32xf32>
    %3 = call @test(%0, %1, %2) : (memref<32x32xf16>, memref<32x32xf16>, memref<1x32xf16>) -> memref<32x32xf32>
    %cast = memref.cast %3 : memref<32x32xf32> to memref<*xf32>
    scf.for %arg0 = %c0 to %c32 step %c1 {
      scf.for %arg1 = %c0 to %c32 step %c1 {
        %4 = scf.for %arg2 = %c0 to %c32 step %c1 iter_args(%arg3 = %cst) -> (f32) {
          %5 = memref.load %0[%arg0, %arg2] : memref<32x32xf16>
          %6 = memref.load %1[%arg2, %arg1] : memref<32x32xf16>
          %7 = memref.load %2[%c0, %arg0] : memref<1x32xf16>
          %8 = arith.addf %5, %7 : f16
          %9 = arith.extf %8 : f16 to f32
          %10 = arith.extf %6 : f16 to f32
          %11 = arith.mulf %9, %10 : f32
          %12 = arith.addf %arg3, %11 : f32
          scf.yield %12 : f32
        }
        // only update the first 8x8 of the result, next 8x8 is value 1
        memref.store %4, %alloc[%arg0, %arg1] : memref<32x32xf32>
      }
    }
    // print GPU and CPU outs
    // call @printMemrefF32(%Out_cpu_cast) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%Out_gpu_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %cast_0 = memref.cast %alloc : memref<32x32xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_0) : (memref<*xf32>, memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

