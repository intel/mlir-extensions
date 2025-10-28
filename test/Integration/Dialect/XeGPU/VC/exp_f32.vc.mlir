// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<8x16xf32>) -> (memref<8x16xf32>, memref<8x16xf32>) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<8x16xf32>
    gpu.memcpy  %memref, %arg0 : memref<8x16xf32>, memref<8x16xf32>
    %memref_0 = gpu.alloc  () : memref<8x16xf32>
    %memref_1 = gpu.alloc  () : memref<8x16xf32>
    gpu.launch_func  @module0::@test_exp_larger_vec blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<8x16xf32>, %memref_0 : memref<8x16xf32>)
    gpu.launch_func  @module1::@test_exp_generic_vec blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<8x16xf32>, %memref_1 : memref<8x16xf32>)
    %alloc = memref.alloc() : memref<8x16xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<8x16xf32>, memref<8x16xf32>
    %alloc_2 = memref.alloc() : memref<8x16xf32>
    gpu.memcpy  %alloc_2, %memref_1 : memref<8x16xf32>, memref<8x16xf32>
    gpu.dealloc  %memref : memref<8x16xf32>
    gpu.dealloc  %memref_0 : memref<8x16xf32>
    gpu.dealloc  %memref_1 : memref<8x16xf32>
    return %alloc, %alloc_2 : memref<8x16xf32>, memref<8x16xf32>
  }
  gpu.module @module0  {
    gpu.func @test_exp_larger_vec(%arg0: memref<8x16xf32>, %arg1: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      // load A tile
      // take exp
      // store
      %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      %2 = math.exp %1 : vector<8x16xf32>
      %3 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %2, %3  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  gpu.module @module1  {
    gpu.func @test_exp_generic_vec(%arg0: memref<8x16xf32>, %arg1: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      // load A tile
      // extract the loaded vector into 16xf32 vectors
      // do generic size exp
      // construct 4x16xf32 vector from the smaller ones
      // store
      %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      %2 = vector.extract %1[0] : vector<16xf32> from vector<8x16xf32>
      %3 = vector.extract %1[1] : vector<16xf32> from vector<8x16xf32>
      %4 = vector.extract %1[2] : vector<16xf32> from vector<8x16xf32>
      %5 = vector.extract %1[3] : vector<16xf32> from vector<8x16xf32>
      %6 = vector.extract %1[4] : vector<16xf32> from vector<8x16xf32>
      %7 = vector.extract %1[5] : vector<16xf32> from vector<8x16xf32>
      %8 = vector.extract %1[6] : vector<16xf32> from vector<8x16xf32>
      %9 = vector.extract %1[7] : vector<16xf32> from vector<8x16xf32>
      %10 = math.exp %2 : vector<16xf32>
      %11 = math.exp %3 : vector<16xf32>
      %12 = math.exp %4 : vector<16xf32>
      %13 = math.exp %5 : vector<16xf32>
      %14 = math.exp %6 : vector<16xf32>
      %15 = math.exp %7 : vector<16xf32>
      %16 = math.exp %8 : vector<16xf32>
      %17 = math.exp %9 : vector<16xf32>
      %18 = vector.shape_cast %10 : vector<16xf32> to vector<1x16xf32>
      %19 = vector.shape_cast %11 : vector<16xf32> to vector<1x16xf32>
      %20 = vector.shape_cast %12 : vector<16xf32> to vector<1x16xf32>
      %21 = vector.shape_cast %13 : vector<16xf32> to vector<1x16xf32>
      %22 = vector.shape_cast %14 : vector<16xf32> to vector<1x16xf32>
      %23 = vector.shape_cast %15 : vector<16xf32> to vector<1x16xf32>
      %24 = vector.shape_cast %16 : vector<16xf32> to vector<1x16xf32>
      %25 = vector.shape_cast %17 : vector<16xf32> to vector<1x16xf32>
      %26 = vector.shuffle %18, %19 [0, 1] : vector<1x16xf32>, vector<1x16xf32>
      %27 = vector.shuffle %20, %21 [0, 1] : vector<1x16xf32>, vector<1x16xf32>
      %28 = vector.shuffle %22, %23 [0, 1] : vector<1x16xf32>, vector<1x16xf32>
      %29 = vector.shuffle %24, %25 [0, 1] : vector<1x16xf32>, vector<1x16xf32>
      %30 = vector.shuffle %26, %27 [0, 1, 2, 3] : vector<2x16xf32>, vector<2x16xf32>
      %31 = vector.shuffle %28, %29 [0, 1, 2, 3] : vector<2x16xf32>, vector<2x16xf32>
      %32 = vector.shuffle %30, %31 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x16xf32>, vector<4x16xf32>
      %33 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %32, %33  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // init constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    // run GPU version
    // run CPU version
    %cst = arith.constant -1.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %false = arith.constant false
    %alloc = memref.alloc() : memref<8x16xf32>
    %alloc_1 = memref.alloc() : memref<8x16xf32>
    %cast = memref.cast %alloc : memref<8x16xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%cast, %cst, %cst_0, %false) : (memref<*xf32>, f32, f32, i1) -> ()
    %0:2 = call @test(%alloc) : (memref<8x16xf32>) -> (memref<8x16xf32>, memref<8x16xf32>)
    %cast_2 = memref.cast %0#1 : memref<8x16xf32> to memref<*xf32>
    %cast_3 = memref.cast %0#0 : memref<8x16xf32> to memref<*xf32>
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %1 = memref.load %alloc[%arg0, %arg1] : memref<8x16xf32>
        %2 = math.exp %1 : f32
        memref.store %2, %alloc_1[%arg0, %arg1] : memref<8x16xf32>
      }
    }
    // print GPU and CPU outs
    // call @printMemrefF32(%Out_cpu_cast) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%Out_gpu_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    // CHECK: [ALLCLOSE: TRUE]
    // dealloc
    // gpu dealloc
    %cast_4 = memref.cast %alloc_1 : memref<8x16xf32> to memref<*xf32>
    call @printAllcloseF32(%cast_2, %cast_4) : (memref<*xf32>, memref<*xf32>) -> ()
    call @printAllcloseF32(%cast_3, %cast_4) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<8x16xf32>
    memref.dealloc %alloc_1 : memref<8x16xf32>
    memref.dealloc  %0#1 : memref<8x16xf32>
    memref.dealloc  %0#0 : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF32(memref<*xf32>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
