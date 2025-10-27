// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/gpu-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @eltwise_add attributes {gpu.container_module} {
  memref.global "private" constant @__constant_10x20xbf16 : memref<10x20xbf16> = dense<5.000000e-01>
  func.func @test(%arg0: memref<10x20xbf16>, %arg1: memref<10x20xbf16>) -> memref<10x20xbf16> {
    %c20 = arith.constant 20 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<10x20xbf16>
    gpu.memcpy  %memref, %arg1 : memref<10x20xbf16>, memref<10x20xbf16>
    %memref_0 = gpu.alloc  () : memref<10x20xbf16>
    gpu.memcpy  %memref_0, %arg0 : memref<10x20xbf16>, memref<10x20xbf16>
    %memref_1 = gpu.alloc  () : memref<10x20xbf16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c10, %c20, %c1) threads in (%c1, %c1, %c1)  args(%memref_0 : memref<10x20xbf16>, %memref : memref<10x20xbf16>, %memref_1 : memref<10x20xbf16>)
    %alloc = memref.alloc() : memref<10x20xbf16>
    gpu.memcpy  %alloc, %memref_1 : memref<10x20xbf16>, memref<10x20xbf16>
    gpu.dealloc  %memref_1 : memref<10x20xbf16>
    gpu.dealloc  %memref_0 : memref<10x20xbf16>
    gpu.dealloc  %memref : memref<10x20xbf16>
    return %alloc : memref<10x20xbf16>
  }
  gpu.module @test_kernel {
    gpu.func @test_kernel(%arg0: memref<10x20xbf16>, %arg1: memref<10x20xbf16>, %arg2: memref<10x20xbf16>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 10, 20, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %cst = arith.constant 5.000000e-01 : bf16
      %0 = memref.load %arg0[%block_id_x, %block_id_y] : memref<10x20xbf16>
      %1 = memref.load %arg1[%block_id_x, %block_id_y] : memref<10x20xbf16>
      %2 = arith.addf %0, %1 : bf16
      %3 = arith.addf %2, %cst : bf16
      memref.store %3, %arg2[%block_id_x, %block_id_y] : memref<10x20xbf16>
      gpu.return
    }
  }
  func.func @main() {
    %0 = memref.get_global @__constant_10x20xbf16 : memref<10x20xbf16>
    %1 = memref.get_global @__constant_10x20xbf16 : memref<10x20xbf16>
    %2 = call @test(%0, %1) : (memref<10x20xbf16>, memref<10x20xbf16>) -> memref<10x20xbf16>
    %cast = memref.cast %2 : memref<10x20xbf16> to memref<*xbf16>
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-COUNT-200: 1.5
    call @printMemrefBF16(%cast) : (memref<*xbf16>) -> ()
    return
  }
  func.func private @printMemrefBF16(memref<*xbf16>) attributes {llvm.emit_c_interface}
}

