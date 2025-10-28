// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  memref.global "private" @__constant_8x32xf16 : memref<8x32xf16> = dense<1.000000e+00>
  memref.global "private" @__constant_16x32xf16 : memref<16x32xf16> = dense<2.000000e+00>
  func.func @test(%arg0: memref<8x32xf16>, %arg1: memref<16x32xf16>) -> memref<8x32xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<8x32xf16>
    %memref_0 = gpu.alloc  () : memref<16x32xf16>
    gpu.memcpy  %memref, %arg0 : memref<8x32xf16>, memref<8x32xf16>
    gpu.memcpy  %memref_0, %arg1 : memref<16x32xf16>, memref<16x32xf16>
    %memref_1 = gpu.alloc  () : memref<8x32xf32>
    gpu.launch_func  @module0::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<8x32xf16>, %memref_0 : memref<16x32xf16>, %memref_1 : memref<8x32xf32>)
    gpu.dealloc  %memref : memref<8x32xf16>
    gpu.dealloc  %memref_0 : memref<16x32xf16>
    %alloc = memref.alloc() : memref<8x32xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<8x32xf32>, memref<8x32xf32>
    gpu.dealloc  %memref_1 : memref<8x32xf32>
    return %alloc : memref<8x32xf32>
  }
  gpu.module @module0  {
    gpu.func @test_kernel(%arg0: memref<8x32xf16>, %arg1: memref<16x32xf16>, %arg2: memref<8x32xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      // load A tile
      // load B tile
      // do DPAS
      // extract second 8x8
      // shift the first half to left and use %cst_8x8 as the second half
      // store
      %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x32xf16> -> !xegpu.tensor_desc<8x16xf16>
      %1 = xegpu.create_nd_tdesc %arg0[%c0, %c16] : memref<8x32xf16> -> !xegpu.tensor_desc<8x16xf16>
      %2 = xegpu.load_nd %0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      %3 = xegpu.load_nd %1  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      %4 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x32xf16> -> !xegpu.tensor_desc<16x16xf16>
      %5 = xegpu.create_nd_tdesc %arg1[%c0, %c16] : memref<16x32xf16> -> !xegpu.tensor_desc<16x16xf16>
      %6 = xegpu.load_nd %4 <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
      %7 = xegpu.load_nd %5 <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
      %8 = xegpu.dpas %2, %6 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
      %9 = xegpu.dpas %3, %7 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
      %10 = vector.extract_strided_slice %8 {offsets = [0, 8], sizes = [8, 8], strides = [1, 1]} : vector<8x16xf32> to vector<8x8xf32>
      %11 = vector.extract_strided_slice %9 {offsets = [0, 8], sizes = [8, 8], strides = [1, 1]} : vector<8x16xf32> to vector<8x8xf32>
      %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
      %12 = vector.shape_cast %cst : vector<64xf32> to vector<8x8xf32>
      %13 = vector.shuffle %10, %12 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x8xf32>, vector<8x8xf32>
      %14 = vector.shuffle %11, %12 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x8xf32>, vector<8x8xf32>
      %15 = vector.shape_cast %13 : vector<16x8xf32> to vector<8x16xf32>
      %16 = vector.shape_cast %14 : vector<16x8xf32> to vector<8x16xf32>
      %17 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      %18 = xegpu.create_nd_tdesc %arg2[%c0, %c16] : memref<8x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %15, %17  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %16, %18  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // init constants
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c24 = arith.constant 24 : index
    %c32 = arith.constant 32 : index
    // random init
    // run GPU version
    // run CPU version
    %0 = memref.get_global @__constant_8x32xf16 : memref<8x32xf16>
    %1 = memref.get_global @__constant_16x32xf16 : memref<16x32xf16>
    %alloc = memref.alloc() : memref<8x32xf32>
    %2 = call @test(%0, %1) : (memref<8x32xf16>, memref<16x32xf16>) -> memref<8x32xf32>
    %cast = memref.cast %2 : memref<8x32xf32> to memref<*xf32>
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c8 to %c16 step %c1 {
        %3 = scf.for %arg2 = %c0 to %c16 step %c1 iter_args(%arg3 = %cst) -> (f32) {
          %5 = memref.load %0[%arg0, %arg2] : memref<8x32xf16>
          %6 = memref.load %1[%arg2, %arg1] : memref<16x32xf16>
          %7 = arith.extf %5 : f16 to f32
          %8 = arith.extf %6 : f16 to f32
          %9 = arith.mulf %7, %8 : f32
          %10 = arith.addf %arg3, %9 : f32
          scf.yield %10 : f32
        }
        // only update the 8x8 of first half of 8x32 of the result, next 8x8 is value 1
        %4 = arith.subi %arg1, %c8 : index
        memref.store %3, %alloc[%arg0, %4] : memref<8x32xf32>
        memref.store %cst_0, %alloc[%arg0, %arg1] : memref<8x32xf32>
      }
    }
    // run CPU version
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c24 to %c32 step %c1 {
        %3 = scf.for %arg2 = %c0 to %c16 step %c1 iter_args(%arg3 = %cst) -> (f32) {
          %5 = memref.load %0[%arg0, %arg2] : memref<8x32xf16>
          %6 = memref.load %1[%arg2, %arg1] : memref<16x32xf16>
          %7 = arith.extf %5 : f16 to f32
          %8 = arith.extf %6 : f16 to f32
          %9 = arith.mulf %7, %8 : f32
          %10 = arith.addf %arg3, %9 : f32
          scf.yield %10 : f32
        }
        // only update the 8x8 of second half of 8x32 of the result, next 8x8 is value 1
        %4 = arith.subi %arg1, %c8 : index
        memref.store %3, %alloc[%arg0, %4] : memref<8x32xf32>
        memref.store %cst_0, %alloc[%arg0, %arg1] : memref<8x32xf32>
      }
    }
    // print GPU and CPU outs
    // call @printMemrefF32(%Out_cpu_cast) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%Out_gpu_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %cast_1 = memref.cast %alloc : memref<8x32xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_1) : (memref<*xf32>, memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
