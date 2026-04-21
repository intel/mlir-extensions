// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup igc-cmd-options=-ze-opt-large-register-file" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --shared-libs=%irunner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// Example of pass pipeline usage:
// %python_executable %imex_runner --requires=mlir-levelzero-runtime,spirv-backend -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
//                                        --runner mlir-runner -e main \
//                                        --entry-point-result=void \
//                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

#a = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [8, 16]>
#b = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [16, 16]>
#c = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>
#a_load = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [32, 16]>
#b_load = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [32, 16]>
#a_prefetch = #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 32], inst_data = [8, 16]>
#b_prefetch = #xegpu.layout<sg_layout = [4, 8], sg_data = [8, 32], inst_data = [8, 16]>
module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<4096x4096xf16>, %B: memref<4096x4096xf16>, %C: memref<4096x4096xf32>, %bias: memref<4096xf32>) -> f64 attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c1000_f64 = arith.constant 1000.0 : f64
    %empty_return = arith.constant 0.0 : f64
    %A_gpu = gpu.alloc () : memref<4096x4096xf16>
    gpu.memcpy %A_gpu, %A : memref<4096x4096xf16>, memref<4096x4096xf16>
    %B_gpu = gpu.alloc () : memref<4096x4096xf16>
    gpu.memcpy %B_gpu, %B : memref<4096x4096xf16>, memref<4096x4096xf16>
    %C_gpu = gpu.alloc () : memref<4096x4096xf32>
    gpu.memcpy %C_gpu, %C : memref<4096x4096xf32>, memref<4096x4096xf32>
    %bias_gpu = gpu.alloc () : memref<4096xf32>
    gpu.memcpy %bias_gpu, %bias : memref<4096xf32>, memref<4096xf32>
    // Execute once for correctness test.
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c16, %c16, %c1) threads in (%c512, %c1, %c1) args(%bias_gpu : memref<4096xf32>, %C_gpu : memref<4096x4096xf32>, %A_gpu : memref<4096x4096xf16>, %B_gpu : memref<4096x4096xf16>)
    gpu.wait
    // Copy result back to host.
    gpu.memcpy %C, %C_gpu : memref<4096x4096xf32>, memref<4096x4096xf32>
    // Warmup loop.
    %nwarmup = arith.constant 20 : index
    scf.for %arg5 = %c0 to %nwarmup step %c1 {
      gpu.launch_func  @test_kernel::@test_kernel blocks in (%c16, %c16, %c1) threads in (%c512, %c1, %c1) args(%bias_gpu : memref<4096xf32>, %C_gpu : memref<4096x4096xf32>, %A_gpu : memref<4096x4096xf16>, %B_gpu : memref<4096x4096xf16>)
      gpu.wait
    }
    // Timing loop.
    %nruns = arith.constant 100 : index
    %0 = func.call @rtclock() : () -> f64
    scf.for %arg5 = %c0 to %nruns step %c1 {
      gpu.launch_func  @test_kernel::@test_kernel blocks in (%c16, %c16, %c1) threads in (%c512, %c1, %c1) args(%bias_gpu : memref<4096xf32>, %C_gpu : memref<4096x4096xf32>, %A_gpu : memref<4096x4096xf16>, %B_gpu : memref<4096x4096xf16>)
      gpu.wait
    }
    %1 = func.call @rtclock() : () -> f64
    // Calculate average time in ms.
    %time = arith.subf %1, %0 : f64
    %nruns_i64 = arith.index_cast %nruns : index to i64
    %nruns_f64 = arith.sitofp %nruns_i64 : i64 to f64
    %avg_time = arith.divf %time, %nruns_f64 : f64
    %avg_time_ms = arith.mulf %avg_time, %c1000_f64 : f64

    gpu.dealloc %A_gpu : memref<4096x4096xf16>
    gpu.dealloc %B_gpu : memref<4096x4096xf16>
    gpu.dealloc %C_gpu : memref<4096x4096xf32>
    gpu.dealloc %bias_gpu : memref<4096xf32>

    return %avg_time_ms : f64
  }

  gpu.module @test_kernel {
    gpu.func @test_kernel(%arg0: memref<4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<4096x4096xf16>, %arg3: memref<4096x4096xf16>) kernel attributes {known_block_size = array<i32: 512, 1, 1>, known_grid_size = array<i32: 16, 16, 1>} {
      %cst = arith.constant dense<true> : vector<256xi1>
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c256 overflow<nsw> : index
      %1 = arith.muli %block_id_y, %c256 overflow<nsw> : index
      %2 = vector.step : vector<256xindex>
      %3 = vector.broadcast %1 : index to vector<256xindex>
      %4 = arith.addi %3, %2 : vector<256xindex>
      %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<4096xf32> -> index
      %5 = arith.index_cast %intptr : index to i64
      %6 = xegpu.load %5[%4], %cst <{layout = #xegpu.slice<#c, dims = [0]>}> : i64, vector<256xindex>, vector<256xi1> -> vector<256xf32>
      %7 = vector.broadcast %6 : vector<256xf32> to vector<256x256xf32>

      %c_tdesc = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf32> -> !xegpu.tensor_desc<256x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #c>

      // There are two different ways to get the initial value of C tile:
      //  1. load from global memory (matrix-multiply-accumulate)  or
      //  2. initialize with 0 (matrix-multiply).
      // We choose to initialize with 0 here to save the global memory bandwidth.
      // Also most bencmarking is done with matrix-multiply, which doesn't require the initial value of C tile,
      // so initializing with 0 can better reflect the performance of matrix multiply itself.
      // The load version is kept in comments for reference and potential future use.

      // %9 = xegpu.load_nd %c_tdesc[%0, %1]  : !xegpu.tensor_desc<256x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #c> -> vector<256x256xf32>

      %9 = arith.constant dense<0.0> : vector<256x256xf32>

      %a_tdesc_load = xegpu.create_nd_tdesc %arg2 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #a_load>
      %a_tdesc_prefetch = xegpu.create_nd_tdesc %arg2 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #a_prefetch>

      xegpu.prefetch_nd %a_tdesc_prefetch[%0, %c0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #a_prefetch}> : !xegpu.tensor_desc<256x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #a_prefetch>
      %b_tdesc_load = xegpu.create_nd_tdesc %arg3 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #b_load>
      %b_tdesc_prefetch = xegpu.create_nd_tdesc %arg3 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #b_prefetch>
      xegpu.prefetch_nd %b_tdesc_prefetch[%c0, %1] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #b_prefetch}> : !xegpu.tensor_desc<32x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #b_prefetch>
      %12 = scf.for %arg4 = %c0 to %c4096 step %c32 iter_args(%arg5 = %9) -> (vector<256x256xf32>) {
        %14 = arith.addi %arg4, %c32 : index
        xegpu.prefetch_nd %b_tdesc_prefetch[%14, %1] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #b_prefetch}> : !xegpu.tensor_desc<32x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #b_prefetch>
        xegpu.prefetch_nd %a_tdesc_prefetch[%0, %14] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #a_prefetch}> : !xegpu.tensor_desc<256x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #a_prefetch>
        %15 = xegpu.load_nd %a_tdesc_load[%0, %arg4] <{layout = #a_load}> : !xegpu.tensor_desc<256x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #a_load> -> vector<256x32xf16>
        %16 = xegpu.load_nd %b_tdesc_load[%arg4, %1] <{layout = #b_load}> : !xegpu.tensor_desc<32x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #b_load> -> vector<32x256xf16>
        %17 = xegpu.convert_layout %15 <{input_layout = #a_load, target_layout = #a}> : vector<256x32xf16>
        %18 = xegpu.convert_layout %16 <{input_layout = #b_load, target_layout = #b}> : vector<32x256xf16>
        %19 = xegpu.dpas %17, %18, %arg5 {layout_a = #a, layout_b = #b, layout_cd = #c} : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf32> -> vector<256x256xf32>
        scf.yield %19 : vector<256x256xf32>
      }
      %13 = arith.addf %7, %12 : vector<256x256xf32>
      xegpu.store_nd %13, %c_tdesc[%0, %1] <{layout = #c}> : vector<256x256xf32>, !xegpu.tensor_desc<256x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #c>
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_f16 = arith.constant 1.0 : f16
    %c2_f16 = arith.constant 2.0 : f16
    %c4096 = arith.constant 4096 : index
    %cf_0 = arith.constant 0.0 : f16
    %cf_1 = arith.constant 1.0 : f16
    %A = memref.alloc() : memref<4096x4096xf16>
    %B = memref.alloc() : memref<4096x4096xf16>
    %C = memref.alloc() : memref<4096x4096xf32>
    %bias = memref.alloc() : memref<4096xf32>
    %C_ref = memref.alloc() : memref<4096x4096xf32>
    %c_gen_int = arith.constant 0 : i1
    %cf_lower = arith.constant 0.0 : f32
    %cf_upper = arith.constant 1.0 : f32
    // Initialize matrix A with random values in (0.0, 1.0).
    %A_random = memref.cast %A : memref<4096x4096xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%A_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf16>, f32, f32, i1) -> ()

    // Initialize matrix B with random values.
    %B_random = memref.cast %B : memref<4096x4096xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%B_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf16>, f32, f32, i1) -> ()

    // Initialize bias vector with random values.
    %bias_random = memref.cast %bias : memref<4096xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%bias_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf32>, f32, f32, i1) -> ()

    // Initialize matrix C and C_ref ; C[i, j] = 0
    %c0_f32 = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c4096 step %c1 {
      scf.for %j = %c0 to %c4096 step %c1 {
        memref.store %c0_f32, %C[%i, %j] : memref<4096x4096xf32>
        memref.store %c0_f32, %C_ref[%i, %j] : memref<4096x4096xf32>
      }
    }

    // Run GPU version.
    %time = call @test(%A, %B, %C, %bias) : (memref<4096x4096xf16>, memref<4096x4096xf16>, memref<4096x4096xf32>, memref<4096xf32>) -> f64
    %gpu_result_cast = memref.cast %C : memref<4096x4096xf32> to memref<*xf32>

    // Run CPU version.
    %A_cast = memref.cast %A : memref<4096x4096xf16> to memref<*xf16>
    %B_cast = memref.cast %B : memref<4096x4096xf16> to memref<*xf16>
    %C_cast = memref.cast %C_ref : memref<4096x4096xf32> to memref<*xf32>
    call @gemmF16F16F32(%A_cast, %B_cast, %C_cast) : (memref<*xf16>, memref<*xf16>, memref<*xf32>) -> ()
    // Add bias to C_ref.
    scf.for %i = %c0 to %c4096 step %c1 {
      scf.for %j = %c0 to %c4096 step %c1 {
        %c_val = memref.load %C_ref[%i, %j] : memref<4096x4096xf32>
        %bias_val = memref.load %bias[%j] : memref<4096xf32>
        %c_bias = arith.addf %c_val, %bias_val : f32
        memref.store %c_bias, %C_ref[%i, %j] : memref<4096x4096xf32>
      }
    }

    %C_row_0 = memref.subview %C_ref[0, 0][1, 4096][1, 1] : memref<4096x4096xf32> to memref<1x4096xf32, strided<[4096, 1], offset:0>>
    %C_row_0_cast = memref.cast %C_row_0 : memref<1x4096xf32, strided<[4096, 1], offset: 0>> to memref<*xf32>
    // call @printMemrefF32(%C_row_0_cast) : (memref<*xf32>) -> ()

    %C_row_0_gpu  = memref.subview %C[0, 0][1, 4096][1, 1] : memref<4096x4096xf32> to memref<1x4096xf32, strided<[4096, 1], offset:0>>
    %C_row_0_cast_gpu = memref.cast %C_row_0_gpu : memref<1x4096xf32, strided<[4096, 1], offset: 0>> to memref<*xf32>
    // call @printMemrefF32(%C_row_0_cast_gpu) : (memref<*xf32>) -> ()

    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%gpu_result_cast, %C_cast) : (memref<*xf32>, memref<*xf32>) -> ()

    // Print time.
    vector.print str "Average time (ms): "
    vector.print %time : f64

    memref.dealloc %A : memref<4096x4096xf16>
    memref.dealloc %B : memref<4096x4096xf16>
    memref.dealloc %C : memref<4096x4096xf32>
    memref.dealloc %C_ref : memref<4096x4096xf32>
    memref.dealloc %bias : memref<4096xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMaxErrorF16(memref<*xf16>, memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF32(memref<*xf32>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @gemmF16F16F32(memref<*xf16>, memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @rtclock() -> f64
}
