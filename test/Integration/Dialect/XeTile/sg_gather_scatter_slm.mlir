// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

// NOTES :
// This example assumes one subgroup per one workgroup and the kernel specifies the computation
// done by a single subgroup.
module @gemm attributes {gpu.container_module} {
  // a test case case return the transpose of A, which is viewed as memref<32x32xf16>.
  // it uses one workgroup containing 32 subgroups, organized as (8x4), so each subgroup
  // works on a 4x8 tile of A. It used SLM to do the transpose, to evaluate the functionality
  // of the SLM operations.
  func.func @test(%arg0: memref<32x32xf16>) -> memref<32x32xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %memref = gpu.alloc  () : memref<32x32xf16>
    gpu.memcpy  %memref, %arg0 : memref<32x32xf16>, memref<32x32xf16>
    %memref_0 = gpu.alloc  () : memref<32x32xf16>
    gpu.launch_func  @test_kernel::@trans_kernel blocks in (%c1, %c1, %c1) threads in (%c4, %c8, %c1)  args(%memref : memref<32x32xf16>, %memref_0 : memref<32x32xf16>)
    gpu.dealloc  %memref : memref<32x32xf16>
    %alloc = memref.alloc() : memref<32x32xf16>
    gpu.memcpy  %alloc, %memref_0 : memref<32x32xf16>, memref<32x32xf16>
    gpu.dealloc  %memref_0 : memref<32x32xf16>
    return %alloc : memref<32x32xf16>
  }
  gpu.module @test_kernel {
    gpu.func @trans_kernel(%arg0: memref<32x32xf16>, %arg1: memref<32x32xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      // %tid_y = arith.divui %sgid, %c4 : index
      // %tid_x = arith.remui %sgid, %c4 : index
      // load data from global memory using block load
      // %slm = memref.alloc() : memref<32x32xf16, 3>
      // %cast = memref.reinterpret_cast %slm to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf16, 3> to memref<1024xf16, 3>
      // store data to slm using original layout
      %0 = gpu.subgroup_id : index
      %1 = arith.shrui %0, %c2 : index
      %2 = arith.andi %0, %c3 : index
      %3 = arith.muli %1, %c4 : index
      %4 = arith.muli %2, %c8 : index
      %5 = xetile.init_tile %arg0[%3, %4] : memref<32x32xf16> -> !xetile.tile<4x8xf16>
      %6 = xetile.load_tile %5 : !xetile.tile<4x8xf16> -> vector<4x8xf16>
      %alloc = memref.alloc() : memref<1024xf16, 3>
      %cst = arith.constant dense<true> : vector<4x8xi1>
      %cst_0 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [32, 33, 34, 35, 36, 37, 38, 39], [64, 65, 66, 67, 68, 69, 70, 71], [96, 97, 98, 99, 100, 101, 102, 103]]> : vector<4x8xindex>
      %7 = arith.muli %1, %c128 : index
      %8 = arith.addi %7, %4 : index
      %9 = vector.broadcast %8: index to vector<4x8xindex>
      %10 = arith.addi %cst_0, %9 : vector<4x8xindex>
      %11 = xetile.init_tile %alloc, %10 : memref<1024xf16, 3>, vector<4x8xindex> -> !xetile.tile<4x8xf16, #xetile.tile_attr<memory_space = 3 : i64, scattered = true>>
      xetile.store %6, %11, %cst : vector<4x8xf16>, !xetile.tile<4x8xf16, #xetile.tile_attr<memory_space = 3 : i64, scattered = true>>, vector<4x8xi1>
      gpu.barrier
      // load data from slm using indices with transpose effects
      %cst_1 = arith.constant dense<[[0, 32, 64, 96, 128, 160, 192, 224], [1, 33, 65, 97, 129, 161, 193, 225], [2, 34, 66, 98, 130, 162, 194, 226], [3, 35, 67, 99, 131, 163, 195, 227]]> : vector<4x8xindex>
      %12 = arith.muli %2, %c256 : index
      %13 = arith.muli %1, %c4 : index
      %14 = arith.addi %12, %13 : index
      %15 = vector.broadcast %14: index to vector<4x8xindex>
      %16 = arith.addi %cst_1, %15 : vector<4x8xindex>
      %17 = xetile.init_tile %alloc, %16 : memref<1024xf16, 3>, vector<4x8xindex> -> !xetile.tile<4x8xf16, #xetile.tile_attr<memory_space = 3 : i64, scattered = true>>
      %18 = xetile.load %17, %cst : !xetile.tile<4x8xf16, #xetile.tile_attr<memory_space = 3 : i64, scattered = true>>, vector<4x8xi1> -> vector<4x8xf16>
      %19 = xetile.init_tile %arg1[%3, %4] : memref<32x32xf16> -> !xetile.tile<4x8xf16>
      xetile.store_tile %18,  %19 : vector<4x8xf16>, !xetile.tile<4x8xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    // intialize matrix A ;
    %alloc = memref.alloc() : memref<32x32xf16>
    %alloc_0 = memref.alloc() : memref<32x32xf32>
    scf.for %arg0 = %c0 to %c32 step %c1 {
      scf.for %arg1 = %c0 to %c32 step %c1 {
        %1 = arith.muli %arg0, %c32 : index
        %2 = arith.addi %1, %arg1 : index
        %3 = index.castu %2 : index to i16
        %4 = arith.uitofp %3 : i16 to f16
        memref.store %4, %alloc[%arg0, %arg1] : memref<32x32xf16>
        %5 = index.castu %2 : index to i32
        %6 = arith.uitofp %5 : i32 to f32
        memref.store %6, %alloc_0[%arg1, %arg0] : memref<32x32xf32>
      }
    }
    //CHECK: [ALLCLOSE: TRUE]
    %0 = call @test(%alloc) : (memref<32x32xf16>) -> memref<32x32xf16>
    %cast = memref.cast %0 : memref<32x32xf16> to memref<*xf16>
    %cast_1 = memref.cast %alloc_0 : memref<32x32xf32> to memref<*xf32>
    call @printAllcloseF16(%cast, %cast_1) : (memref<*xf16>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<32x32xf16>
    memref.dealloc %alloc_0 : memref<32x32xf32>
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
