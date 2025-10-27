// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<8x32xf16>, %arg1: memref<16x32xf16>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<8x32xf16>
    %memref_0 = gpu.alloc  () : memref<16x32xf16>
    gpu.memcpy  %memref, %arg0 : memref<8x32xf16>, memref<8x32xf16>
    gpu.memcpy  %memref_0, %arg1 : memref<16x32xf16>, memref<16x32xf16>
    %memref_1 = gpu.alloc  () : memref<8x16xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<8x32xf16>, %memref_0 : memref<16x32xf16>, %memref_1 : memref<8x16xf32>)
    gpu.dealloc  %memref : memref<8x32xf16>
    gpu.dealloc  %memref_0 : memref<16x32xf16>
    %alloc = memref.alloc() : memref<8x16xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<8x16xf32>, memref<8x16xf32>
    gpu.dealloc  %memref_1 : memref<8x16xf32>
    return %alloc : memref<8x16xf32>
  }
  gpu.module @test_kernel  {
    gpu.func @test_kernel(%arg0: memref<8x32xf16>, %arg1: memref<16x32xf16>, %arg2: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      // load A tiles
      // load B tiles
      // do DPAS
      // take fmax
      // store fmax
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
      %10 = arith.maximumf %8, %9 fastmath<nnan> : vector<8x16xf32>
      %11 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %10, %11  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // init constants
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant -5.000000e-01 : f32
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    // run GPU version
    // run CPU version
    %alloc = memref.alloc() : memref<8x32xf16>
    %alloc_2 = memref.alloc() : memref<16x32xf16>
    %alloc_3 = memref.alloc() : memref<8x16xf32>
    %cast = memref.cast %alloc : memref<8x32xf16> to memref<*xf16>
    %cast_4 = memref.cast %alloc_2 : memref<16x32xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%cast, %cst_1, %cst_0, %false) : (memref<*xf16>, f32, f32, i1) -> ()
    call @fillResource1DRandomF16(%cast_4, %cst_1, %cst_0, %false) : (memref<*xf16>, f32, f32, i1) -> ()
    %0 = call @test(%alloc, %alloc_2) : (memref<8x32xf16>, memref<16x32xf16>) -> memref<8x16xf32>
    %cast_5 = memref.cast %0 : memref<8x16xf32> to memref<*xf32>
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %1:2 = scf.for %arg2 = %c0 to %c16 step %c1 iter_args(%arg3 = %cst, %arg4 = %cst) -> (f32, f32) {
          %3 = arith.addi %arg2, %c16 : index
          %4 = arith.addi %arg1, %c16 : index
          %5 = memref.load %alloc[%arg0, %arg2] : memref<8x32xf16>
          %6 = memref.load %alloc[%arg0, %3] : memref<8x32xf16>
          %7 = memref.load %alloc_2[%arg2, %arg1] : memref<16x32xf16>
          %8 = memref.load %alloc_2[%arg2, %4] : memref<16x32xf16>
          %9 = arith.extf %5 : f16 to f32
          %10 = arith.extf %6 : f16 to f32
          %11 = arith.extf %7 : f16 to f32
          %12 = arith.extf %8 : f16 to f32
          %13 = arith.mulf %9, %11 : f32
          %14 = arith.mulf %10, %12 : f32
          %15 = arith.addf %arg3, %13 : f32
          %16 = arith.addf %arg4, %14 : f32
          scf.yield %15, %16 : f32, f32
        }
        %2 = arith.maximumf %1#0, %1#1 : f32
        memref.store %2, %alloc_3[%arg0, %arg1] : memref<8x16xf32>
      }
    }
    // print GPU and CPU outs
    // CHECK: [ALLCLOSE: TRUE]
    // dealloc
    // gpu dealloc
    %cast_6 = memref.cast %alloc_3 : memref<8x16xf32> to memref<*xf32>
    call @printMemrefF32(%cast_6) : (memref<*xf32>) -> ()
    call @printMemrefF32(%cast_5) : (memref<*xf32>) -> ()
    call @printAllcloseF32(%cast_5, %cast_6) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<8x32xf16>
    memref.dealloc %alloc_2 : memref<16x32xf16>
    memref.dealloc %alloc_3 : memref<8x16xf32>
    memref.dealloc  %0 : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

