// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (96)>
module @gemm_output_f16 attributes {gpu.container_module} {
  func.func @gemm_output_f16_entry(%arg0: memref<128x96xf16>, %arg1: memref<256x96xf16>, %arg2: memref<128x256xf16>) -> memref<128x256xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %memref = gpu.alloc  () : memref<128x96xf16>
    gpu.memcpy  %memref, %arg0 : memref<128x96xf16>, memref<128x96xf16>
    %memref_0 = gpu.alloc  () : memref<256x96xf16>
    gpu.memcpy  %memref_0, %arg1 : memref<256x96xf16>, memref<256x96xf16>
    %memref_1 = gpu.alloc  () : memref<128x256xf16>
    gpu.memcpy  %memref_1, %arg2 : memref<128x256xf16>, memref<128x256xf16>
    %memref_2 = gpu.alloc  () : memref<128x256xf16>
    gpu.launch_func  @gemm_output_f16::@gemm_output_f16 blocks in (%c1, %c1, %c1) threads in (%c4, %c8, %c1)  args(%memref : memref<128x96xf16>, %memref_0 : memref<256x96xf16>, %memref_1 : memref<128x256xf16>, %memref_2 : memref<128x256xf16>)
    gpu.dealloc  %memref : memref<128x96xf16>
    gpu.dealloc  %memref_0 : memref<256x96xf16>
    gpu.dealloc  %memref_1 : memref<128x256xf16>
    %alloc = memref.alloc() : memref<128x256xf16>
    gpu.memcpy  %alloc, %memref_2 : memref<128x256xf16>, memref<128x256xf16>
    gpu.dealloc  %memref_2 : memref<128x256xf16>
    return %alloc : memref<128x256xf16>
  }
  gpu.module @gemm_output_f16 {
    gpu.func @gemm_output_f16(%arg0: memref<128x96xf16>, %arg1: memref<256x96xf16>, %arg2: memref<128x256xf16>, %arg3: memref<128x256xf16>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 4, 8, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c96 = arith.constant 96 : index
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %cst = arith.constant dense<0.000000e+00> : vector<32x32xf16>
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
      %8 = xetile.init_tile %arg0[%5, %c0] : memref<128x96xf16> -> !xetile.tile<32x32xf16>
      %9 = xetile.init_tile %arg1[%7, %c0] : memref<256x96xf16> -> !xetile.tile<32x32xf16>
      %10:3 = scf.for %arg4 = %c0 to %c96 step %c32 iter_args(%arg5 = %cst, %arg6 = %8, %arg7 = %9) -> (vector<32x32xf16>, !xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>) {
        %15 = xetile.update_tile_offset %arg7, [%c0, %c32] : !xetile.tile<32x32xf16>
        %16 = xetile.update_tile_offset %arg6, [%c0, %c32] : !xetile.tile<32x32xf16>
        %17 = xetile.load_tile %arg6 {padding = 0.000000e+00 : f32} : !xetile.tile<32x32xf16> -> vector<32x32xf16>
        %18 = xetile.load_tile %arg7 {padding = 0.000000e+00 : f32} : !xetile.tile<32x32xf16> -> vector<32x32xf16>
        %19 = vector.transpose %18, [1, 0] : vector<32x32xf16> to vector<32x32xf16>
        xegpu.compile_hint
        xegpu.compile_hint
        %20 = xetile.tile_mma %17, %19, %arg5 : vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf16> -> vector<32x32xf16>
        xegpu.compile_hint
        scf.yield %20, %16, %15 : vector<32x32xf16>, !xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>
      } {lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 3>, step = 32 : index, upperBoundMap = #map1}
      %11 = xetile.init_tile %arg2[%5, %7] : memref<128x256xf16> -> !xetile.tile<32x32xf16>
      %12 = xetile.load_tile %11 {padding = 0.000000e+00 : f32} : !xetile.tile<32x32xf16> -> vector<32x32xf16>
      %13 = arith.addf %10#0, %12 : vector<32x32xf16>
      %14 = xetile.init_tile %arg3[%5, %7] : memref<128x256xf16> -> !xetile.tile<32x32xf16>
      xetile.store_tile %13,  %14 : vector<32x32xf16>, !xetile.tile<32x32xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 1.000000e+02 : f16
    %cst_0 = arith.constant 4.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c96 = arith.constant 96 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    // intialize matrix A with ones
    %cst_1 = arith.constant 1.000000e+00 : f16
    %alloc = memref.alloc() : memref<128x96xf16>
    %alloc_2 = memref.alloc() : memref<256x96xf16>
    %alloc_3 = memref.alloc() : memref<128x256xf16>
    %alloc_4 = memref.alloc() : memref<128x256xf16>
    scf.for %arg0 = %c0 to %c128 step %c1 {
      scf.for %arg1 = %c0 to %c96 step %c1 {
        memref.store %cst_1, %alloc[%arg0, %arg1] : memref<128x96xf16>
      }
    }
    // intialize matrix B with ones
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c96 step %c1 {
        memref.store %cst_1, %alloc_2[%arg0, %arg1] : memref<256x96xf16>
      }
    }
    // intialize matrix POSTOP (second operand of the postop) and OUTPUT_ref.
    scf.for %arg0 = %c0 to %c128 step %c1 {
      scf.for %arg1 = %c0 to %c256 step %c1 {
        memref.store %cst_0, %alloc_3[%arg0, %arg1] : memref<128x256xf16>
        memref.store %cst, %alloc_4[%arg0, %arg1] : memref<128x256xf16>
      }
    }
    // TODO: investigate why printAllcloseF16 was returning false even when the
    // tensors are identical. It looks like an issue when comparing f16 values.
    // For now using printMaxErrorF16.
    // call @printAllcloseF16(%cast_OUTPUT, %cast_OUTPUT_ref) : (memref<*xf16>, memref<*xf16>) -> ()
    // CHECK: Max absolute error 0
    // CHECK: Max relative error 0
    %0 = call @gemm_output_f16_entry(%alloc, %alloc_2, %alloc_3) : (memref<128x96xf16>, memref<256x96xf16>, memref<128x256xf16>) -> memref<128x256xf16>
    %cast = memref.cast %0 : memref<128x256xf16> to memref<*xf16>
    %cast_5 = memref.cast %alloc_4 : memref<128x256xf16> to memref<*xf16>
    call @printMaxErrorF16(%cast, %cast_5) : (memref<*xf16>, memref<*xf16>) -> ()
    memref.dealloc %alloc : memref<128x96xf16>
    memref.dealloc %alloc_2 : memref<256x96xf16>
    memref.dealloc %alloc_3 : memref<128x256xf16>
    memref.dealloc %alloc_4 : memref<128x256xf16>
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printMaxErrorF16(memref<*xf16>, memref<*xf16>) attributes {llvm.emit_c_interface}
}
