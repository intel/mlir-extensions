// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<8x16xf32>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<8x16xf32>
    gpu.memcpy  %memref, %arg0 : memref<8x16xf32>, memref<8x16xf32>
    %memref_0 = gpu.alloc  () : memref<8x16xf32>
    gpu.launch_func  @module0::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<8x16xf32>, %memref_0 : memref<8x16xf32>)
    gpu.dealloc  %memref : memref<8x16xf32>
    %alloc = memref.alloc() : memref<8x16xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<8x16xf32>, memref<8x16xf32>
    gpu.dealloc  %memref_0 : memref<8x16xf32>
    return %alloc : memref<8x16xf32>
  }
  gpu.module @module0  {
    gpu.func @test_kernel(%arg0: memref<8x16xf32>, %arg1: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      // load tile
      // extract row at pos 2
      // insert row at pos 7
      // store
      %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      %2 = vector.extract %1[2] : vector<16xf32> from vector<8x16xf32>
      %3 = vector.insert %2, %1 [7] : vector<16xf32> into vector<8x16xf32>
      %4 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %3, %4  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // init constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    // random init
    // run GPU version
    // run CPU version
    %cst = arith.constant -3.000000e+00 : f32
    %cst_0 = arith.constant 3.000000e+00 : f32
    %false = arith.constant false
    %alloc = memref.alloc() : memref<8x16xf32>
    %alloc_1 = memref.alloc() : memref<8x16xf32>
    %cast = memref.cast %alloc : memref<8x16xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%cast, %cst, %cst_0, %false) : (memref<*xf32>, f32, f32, i1) -> ()
    %0 = call @test(%alloc) : (memref<8x16xf32>) -> memref<8x16xf32>
    %cast_2 = memref.cast %0 : memref<8x16xf32> to memref<*xf32>
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %1 = memref.load %alloc[%arg0, %arg1] : memref<8x16xf32>
        memref.store %1, %alloc_1[%arg0, %arg1] : memref<8x16xf32>
      }
    }
    scf.for %arg0 = %c0 to %c16 step %c1 {
      %1 = memref.load %alloc[%c2, %arg0] : memref<8x16xf32>
      memref.store %1, %alloc_1[%c7, %arg0] : memref<8x16xf32>
    }
    // print GPU and CPU outs
    // call @printMemrefF32(%Out_cpu_cast) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%Out_gpu_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    // dealloc
    // gpu dealloc
    %cast_3 = memref.cast %alloc_1 : memref<8x16xf32> to memref<*xf32>
    call @printAllcloseF32(%cast_2, %cast_3) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<8x16xf32>
    memref.dealloc %alloc_1 : memref<8x16xf32>
    memref.dealloc  %0 : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF32(memref<*xf32>, f32, f32, i1) attributes {llvm.emit_c_interface}
}

