// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<8x16xf32>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<8x16xf32>
    gpu.memcpy  %memref, %arg0 : memref<8x16xf32>, memref<8x16xf32>
    %memref_0 = gpu.alloc  () : memref<8x16xf32>
    %cast = memref.cast %memref : memref<8x16xf32> to memref<?x?xf32>
    %cast_1 = memref.cast %memref_0 : memref<8x16xf32> to memref<?x?xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%cast : memref<?x?xf32>, %cast_1 : memref<?x?xf32>, %c8 : index, %c16 : index, %c16 : index, %c1 : index, %c0 : index, %c0 : index)
    gpu.dealloc  %memref : memref<8x16xf32>
    %alloc = memref.alloc() : memref<8x16xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<8x16xf32>, memref<8x16xf32>
    gpu.dealloc  %memref_0 : memref<8x16xf32>
    return %alloc : memref<8x16xf32>
  }
  gpu.module @test_kernel  {
    gpu.func @test_kernel(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = xegpu.create_nd_tdesc %arg0[%arg6, %arg7], shape : [%arg2, %arg3], strides : [%arg4, %arg5] : memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
      %1 = xegpu.load_nd %0 <{l1_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      %2 = xegpu.create_nd_tdesc %arg1[%arg6, %arg7], shape : [%arg2, %arg3], strides : [%arg4, %arg5] : memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %1, %2  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // call @printMemrefF32(%B_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant -5.000000e-01 : f32
    %false = arith.constant false
    %alloc = memref.alloc() : memref<8x16xf32>
    %cast = memref.cast %alloc : memref<8x16xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%cast, %cst_0, %cst, %false) : (memref<*xf32>, f32, f32, i1) -> ()
    %0 = call @test(%alloc) : (memref<8x16xf32>) -> memref<8x16xf32>
    %cast_1 = memref.cast %0 : memref<8x16xf32> to memref<*xf32>
    %cast_2 = memref.cast %alloc : memref<8x16xf32> to memref<*xf32>
    call @printAllcloseF32(%cast_2, %cast_1) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF32(memref<*xf32>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
