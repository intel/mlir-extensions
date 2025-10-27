// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (100)>
module @gemm attributes {gpu.container_module} {
  func.func @gemm_k_oob_entry(%arg0: memref<128x100xf16>, %arg1: memref<256x100xf16>) -> memref<128x256xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %memref = gpu.alloc  () : memref<128x100xf16>
    gpu.memcpy  %memref, %arg0 : memref<128x100xf16>, memref<128x100xf16>
    %memref_0 = gpu.alloc  () : memref<256x100xf16>
    gpu.memcpy  %memref_0, %arg1 : memref<256x100xf16>, memref<256x100xf16>
    %memref_1 = gpu.alloc  () : memref<128x256xf32>
    gpu.launch_func  @gemm_k_oob::@gemm_k_oob blocks in (%c2, %c1, %c1) threads in (%c4, %c8, %c1)  args(%memref : memref<128x100xf16>, %memref_0 : memref<256x100xf16>, %memref_1 : memref<128x256xf32>)
    gpu.dealloc  %memref : memref<128x100xf16>
    gpu.dealloc  %memref_0 : memref<256x100xf16>
    %alloc = memref.alloc() : memref<128x256xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<128x256xf32>, memref<128x256xf32>
    gpu.dealloc  %memref_1 : memref<128x256xf32>
    return %alloc : memref<128x256xf32>
  }
  gpu.module @gemm_k_oob {
    gpu.func @gemm_k_oob(%arg0: memref<128x100xf16>, %arg1: memref<256x100xf16>, %arg2: memref<128x256xf32>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 4, 8, 1>, gpu.known_grid_size = array<i32: 2, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c0 = arith.constant 0 : index
      %cst = arith.constant dense<0.000000e+00> : vector<32x32xf16>
      %c100 = arith.constant 100 : index
      %c256 = arith.constant 256 : index
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %cst_0 = arith.constant dense<0.000000e+00> : vector<32x32xf32>
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %block_dim_y = gpu.block_dim  y
      %0 = arith.muli %thread_id_x, %block_dim_y : index
      %1 = arith.addi %0, %thread_id_y : index
      %2 = arith.divsi %1, %c8 : index
      %3 = arith.remsi %1, %c8 : index
      %4 = arith.muli %2, %c32 : index
      %5 = arith.remsi %4, %c128 : index
      %6 = arith.muli %3, %c32 : index
      %7 = arith.remsi %6, %c256 : index
      %8 = xetile.init_tile %arg0[%5, %c0] : memref<128x100xf16> -> !xetile.tile<32x32xf16>
      %9 = xetile.init_tile %arg1[%7, %c0] : memref<256x100xf16> -> !xetile.tile<32x32xf16>
      %10:3 = scf.for %arg3 = %c0 to %c100 step %c32 iter_args(%arg4 = %cst_0, %arg5 = %8, %arg6 = %9) -> (vector<32x32xf32>, !xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>) {
        %12 = xetile.update_tile_offset %arg6, [%c0, %c32] : !xetile.tile<32x32xf16>
        %13 = xetile.update_tile_offset %arg5, [%c0, %c32] : !xetile.tile<32x32xf16>
        %14 = xetile.load_tile %arg5 {padding = 0.000000e+00 : f32} : !xetile.tile<32x32xf16> -> vector<32x32xf16>
        %15 = xetile.load_tile %arg6 {padding = 0.000000e+00 : f32} : !xetile.tile<32x32xf16> -> vector<32x32xf16>
        %16 = vector.transpose %15, [1, 0] : vector<32x32xf16> to vector<32x32xf16>
        xegpu.compile_hint
        %17 = math.exp %14 : vector<32x32xf16>
        %18 = arith.subi %c100, %arg3 : index
        %19 = vector.create_mask %c32, %18 : vector<32x32xi1>
        %20 = arith.select %19, %17, %cst : vector<32x32xi1>, vector<32x32xf16>
        %21 = math.exp %16 : vector<32x32xf16>
        xegpu.compile_hint
        %22 = xetile.tile_mma %20, %21, %arg4 : vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
        xegpu.compile_hint
        scf.yield %22, %13, %12 : vector<32x32xf32>, !xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>
      } {lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 3>, step = 32 : index, upperBoundMap = #map1}
      %11 = xetile.init_tile %arg2[%5, %7] : memref<128x256xf32> -> !xetile.tile<32x32xf32>
      xetile.store_tile %10#0,  %11 : vector<32x32xf32>, !xetile.tile<32x32xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 738.90564 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    // intialize matrix A with ones
    %cst_0 = arith.constant 1.000000e+00 : f16
    %alloc = memref.alloc() : memref<128x100xf16>
    %alloc_1 = memref.alloc() : memref<256x100xf16>
    %alloc_2 = memref.alloc() : memref<128x256xf32>
    scf.for %arg0 = %c0 to %c128 step %c1 {
      scf.for %arg1 = %c0 to %c100 step %c1 {
        memref.store %cst_0, %alloc[%arg0, %arg1] : memref<128x100xf16>
      }
    }
    // intialize matrix B with ones
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c100 step %c1 {
        memref.store %cst_0, %alloc_1[%arg0, %arg1] : memref<256x100xf16>
      }
    }
    // intialize matrix GEMM_ref
    scf.for %arg0 = %c0 to %c128 step %c1 {
      scf.for %arg1 = %c0 to %c256 step %c1 {
        memref.store %cst, %alloc_2[%arg0, %arg1] : memref<128x256xf32>
      }
    }
    // CHECK: Max absolute error 0.
    // CHECK: Max relative error 0.00
    %0 = call @gemm_k_oob_entry(%alloc, %alloc_1) : (memref<128x100xf16>, memref<256x100xf16>) -> memref<128x256xf32>
    %cast = memref.cast %0 : memref<128x256xf32> to memref<*xf32>
    %cast_3 = memref.cast %alloc_2 : memref<128x256xf32> to memref<*xf32>
    call @printMaxErrorF32(%cast, %cast_3) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<128x100xf16>
    memref.dealloc %alloc_1 : memref<256x100xf16>
    memref.dealloc %alloc_2 : memref<128x256xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMaxErrorF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

