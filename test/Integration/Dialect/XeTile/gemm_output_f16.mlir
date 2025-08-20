// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (96)>
module @gemm_output_f16 attributes {gpu.container_module} {
  func.func @gemm_output_f16_entry(
      %A: memref<128x96xf16>, %B: memref<256x96xf16>, %POSTOP: memref<128x256xf16>) ->
      memref<128x256xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %A_gpu = gpu.alloc  host_shared () : memref<128x96xf16>
    memref.copy %A, %A_gpu : memref<128x96xf16> to memref<128x96xf16>
    %B_gpu = gpu.alloc  host_shared () : memref<256x96xf16>
    memref.copy %B, %B_gpu : memref<256x96xf16> to memref<256x96xf16>
    %POSTOP_gpu = gpu.alloc  host_shared () : memref<128x256xf16>
    memref.copy %POSTOP, %POSTOP_gpu : memref<128x256xf16> to memref<128x256xf16>
    %OUTPUT_gpu = gpu.alloc  host_shared () : memref<128x256xf16>
    gpu.launch_func  @gemm_output_f16::@gemm_output_f16 blocks in (%c1, %c1, %c1) threads in (%c4, %c8, %c1)
                     args(%A_gpu : memref<128x96xf16>,
                          %B_gpu : memref<256x96xf16>,
                          %POSTOP_gpu : memref<128x256xf16>,
                          %OUTPUT_gpu : memref<128x256xf16>)
    gpu.dealloc  %A_gpu : memref<128x96xf16>
    gpu.dealloc  %B_gpu : memref<256x96xf16>
    gpu.dealloc  %POSTOP_gpu : memref<128x256xf16>
    return %OUTPUT_gpu : memref<128x256xf16>
  }

  gpu.module @gemm_output_f16 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Bfloat16ConversionINTEL, BFloat16TypeKHR, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_bfloat16, SPV_KHR_expect_assume, SPV_INTEL_bfloat16_conversion, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
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
        %15 = xetile.update_tile_offset %arg7, [%c0,  %c32] :  !xetile.tile<32x32xf16>
        %16 = xetile.update_tile_offset %arg6, [%c0,  %c32] :  !xetile.tile<32x32xf16>
        %17 = xetile.load_tile %arg6 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xf16> -> vector<32x32xf16>
        %18 = xetile.load_tile %arg7 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xf16> -> vector<32x32xf16>
        %19 = vector.transpose %18, [1, 0] : vector<32x32xf16> to vector<32x32xf16>
        xegpu.compile_hint
        xegpu.compile_hint
        %20 = xetile.tile_mma %17, %19, %arg5 : vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf16> -> vector<32x32xf16>
        xegpu.compile_hint
        scf.yield %20, %16, %15 : vector<32x32xf16>, !xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>
      } {lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 3>, step = 32 : index, upperBoundMap = #map1}
      %11 = xetile.init_tile %arg2[%5, %7] : memref<128x256xf16> -> !xetile.tile<32x32xf16>
      %12 = xetile.load_tile %11 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xf16> -> vector<32x32xf16>
      %13 = arith.addf %10#0, %12 : vector<32x32xf16>
      %14 = xetile.init_tile %arg3[%5, %7] : memref<128x256xf16> -> !xetile.tile<32x32xf16>
      xetile.store_tile %13,  %14 : vector<32x32xf16>, !xetile.tile<32x32xf16>
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c96 = arith.constant 96 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %cf_1 = arith.constant 1.0 : f16
    %A = memref.alloc() : memref<128x96xf16>
    %B = memref.alloc() : memref<256x96xf16>
    %POSTOP = memref.alloc() : memref<128x256xf16>
    %OUTPUT_ref = memref.alloc() : memref<128x256xf16>
    // intialize matrix A with ones
    scf.for %i = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c96 step %c1 {
        memref.store %cf_1, %A[%i, %j] : memref<128x96xf16>
      }
    }
    // intialize matrix B with ones
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c96 step %c1 {
        memref.store %cf_1, %B[%i, %j] : memref<256x96xf16>
      }
    }
    // intialize matrix POSTOP (second operand of the postop) and OUTPUT_ref.
    %cf_4 = arith.constant 4.0 : f16
    %cf_result = arith.constant 100.0 : f16
    scf.for %i = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        memref.store %cf_4, %POSTOP[%i, %j] : memref<128x256xf16>
        memref.store %cf_result, %OUTPUT_ref[%i, %j] : memref<128x256xf16>
      }
    }

    %OUTPUT = call @gemm_output_f16_entry(%A, %B, %POSTOP) : (memref<128x96xf16>, memref<256x96xf16>, memref<128x256xf16>) -> memref<128x256xf16>
    %cast_OUTPUT = memref.cast %OUTPUT : memref<128x256xf16> to memref<*xf16>
    %cast_OUTPUT_ref = memref.cast %OUTPUT_ref : memref<128x256xf16> to memref<*xf16>
    // TODO: investigate why printAllcloseF16 was returning false even when the
    // tensors are identical. It looks like an issue when comparing f16 values.
    // For now using printMaxErrorF16.
    // call @printAllcloseF16(%cast_OUTPUT, %cast_OUTPUT_ref) : (memref<*xf16>, memref<*xf16>) -> ()
    // CHECK: Max absolute error 0
    // CHECK: Max relative error 0
    call @printMaxErrorF16(%cast_OUTPUT, %cast_OUTPUT_ref) : (memref<*xf16>, memref<*xf16>) -> ()
    memref.dealloc %A : memref<128x96xf16>
    memref.dealloc %B : memref<256x96xf16>
    memref.dealloc %POSTOP : memref<128x256xf16>
    memref.dealloc %OUTPUT_ref : memref<128x256xf16>
    return
  }

  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printMaxErrorF16(memref<*xf16>, memref<*xf16>) attributes {llvm.emit_c_interface}
}
