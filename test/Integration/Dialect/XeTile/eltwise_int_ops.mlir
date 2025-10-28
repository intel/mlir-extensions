// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @eltwise_int attributes {gpu.container_module} {
  memref.global "private" constant @__constant_5_1024x1024xi32 : memref<1024x1024xi32> = dense<5>
  memref.global "private" constant @__constant_2_1024x1024xi32 : memref<1024x1024xi32> = dense<2>
  func.func @eltwise_int_test(%arg0: memref<1024x1024xi32>, %arg1: memref<1024x1024xi32>) -> memref<1024x1024xi32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %memref = gpu.alloc  () : memref<1024x1024xi32>
    gpu.memcpy  %memref, %arg0 : memref<1024x1024xi32>, memref<1024x1024xi32>
    %memref_0 = gpu.alloc  () : memref<1024x1024xi32>
    gpu.memcpy  %memref_0, %arg1 : memref<1024x1024xi32>, memref<1024x1024xi32>
    %memref_1 = gpu.alloc  () : memref<1024x1024xi32>
    gpu.launch_func  @eltwise_int::@eltwise_int blocks in (%c64, %c32, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<1024x1024xi32>, %memref_0 : memref<1024x1024xi32>, %memref_1 : memref<1024x1024xi32>)
    gpu.dealloc  %memref : memref<1024x1024xi32>
    gpu.dealloc  %memref_0 : memref<1024x1024xi32>
    %alloc = memref.alloc() : memref<1024x1024xi32>
    gpu.memcpy  %alloc, %memref_1 : memref<1024x1024xi32>, memref<1024x1024xi32>
    gpu.dealloc  %memref_1 : memref<1024x1024xi32>
    return %alloc : memref<1024x1024xi32>
  }
  gpu.module @eltwise_int {
    gpu.func @eltwise_int(%arg0: memref<1024x1024xi32>, %arg1: memref<1024x1024xi32>, %arg2: memref<1024x1024xi32>) kernel attributes {VectorComputeFunctionINTEL, known_block_size = array<i32: 1, 32, 1>, known_grid_size = array<i32: 1, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c16 : index
      %1 = arith.muli %block_id_y, %c32 : index
      %2 = xetile.init_tile %arg0[%0, %1] : memref<1024x1024xi32> -> !xetile.tile<16x32xi32>
      %3 = xetile.load_tile %2 : !xetile.tile<16x32xi32> -> vector<16x32xi32>
      %4 = xetile.init_tile %arg1[%0, %1] : memref<1024x1024xi32> -> !xetile.tile<16x32xi32>
      %5 = xetile.load_tile %4 : !xetile.tile<16x32xi32> -> vector<16x32xi32>
      %6 = arith.addi %3, %5 : vector<16x32xi32>
      %7 = arith.subi %3, %5 : vector<16x32xi32>
      %8 = arith.muli %6, %7 : vector<16x32xi32>
      %9 = arith.divsi %8, %6 : vector<16x32xi32>
      %10 = arith.divui %8, %6 : vector<16x32xi32>
      %11 = arith.remsi %9, %8 : vector<16x32xi32>
      %12 = arith.remui %10, %11 : vector<16x32xi32>
      %13 = arith.addi %11, %12 : vector<16x32xi32>
      %14 = xetile.init_tile %arg2[%0, %1] : memref<1024x1024xi32> -> !xetile.tile<16x32xi32>
      xetile.store_tile %13,  %14 : vector<16x32xi32>, !xetile.tile<16x32xi32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-COUNT-1048576: 3
    %0 = memref.get_global @__constant_5_1024x1024xi32 : memref<1024x1024xi32>
    %1 = memref.get_global @__constant_2_1024x1024xi32 : memref<1024x1024xi32>
    %2 = call @eltwise_int_test(%0, %1) : (memref<1024x1024xi32>, memref<1024x1024xi32>) -> memref<1024x1024xi32>
    %cast = memref.cast %2 : memref<1024x1024xi32> to memref<*xi32>
    call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
    return
  }
  func.func private @printMemrefI32(memref<*xi32>) attributes {llvm.emit_c_interface}
}
