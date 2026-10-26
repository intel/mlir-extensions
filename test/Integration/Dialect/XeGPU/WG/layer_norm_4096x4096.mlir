// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup" | \
// RUN:   mlir-runner \
// RUN:     --shared-libs=%mlir_levelzero_runtime \
// RUN:     --shared-libs=%mlir_runner_utils \
// RUN:     --shared-libs=%mlir_c_runner_utils \
// RUN:     --shared-libs=%irunner_utils \
// RUN:     --entry-point-result=void \
// RUN: | FileCheck %s

// Row-wise layer norm: (x - mean) * rsqrt(var + eps) * gamma + beta,
// with var = mean((x - mean)^2).

module attributes {gpu.container_module} {
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cM = arith.constant 4096 : index
    %cN = arith.constant 4096 : index
    %c_grid = arith.constant 4096 : index
    %c_block = arith.constant 256 : index
    %c0_f32 = arith.constant 0.0 : f32
    %c_gen_int = arith.constant 0 : i1

    // Off-center inputs expose cancellation in E[x^2] - E[x]^2.
    %in_lo = arith.constant 24.5 : f32
    %in_hi = arith.constant 25.5 : f32
    %g_lo = arith.constant 0.5 : f32
    %g_hi = arith.constant 1.5 : f32
    %b_lo = arith.constant -0.1 : f32
    %b_hi = arith.constant 0.1 : f32

    %input = memref.alloc() : memref<4096x4096xf32>
    %input_cast = memref.cast %input : memref<4096x4096xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%input_cast, %in_lo, %in_hi, %c_gen_int) : (memref<*xf32>, f32, f32, i1) -> ()

    // Vary row offsets in [-512, 511] to reject fixed-shift variance formulas.
    %ro_span = arith.constant 1024 : index
    %ro_half = arith.constant 5.120000e+02 : f32
    scf.for %ro_i = %c0 to %cM step %c1 {
      %ro_m = arith.remsi %ro_i, %ro_span : index
      %ro_mi = arith.index_cast %ro_m : index to i32
      %ro_mf = arith.sitofp %ro_mi : i32 to f32
      %ro_off = arith.subf %ro_mf, %ro_half : f32
      scf.for %ro_j = %c0 to %cN step %c1 {
        %ro_v = memref.load %input[%ro_i, %ro_j] : memref<4096x4096xf32>
        %ro_w = arith.addf %ro_v, %ro_off : f32
        memref.store %ro_w, %input[%ro_i, %ro_j] : memref<4096x4096xf32>
      }
    }

    %gamma = memref.alloc() : memref<4096xf32>
    %gamma_cast = memref.cast %gamma : memref<4096xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%gamma_cast, %g_lo, %g_hi, %c_gen_int) : (memref<*xf32>, f32, f32, i1) -> ()

    %beta = memref.alloc() : memref<4096xf32>
    %beta_cast = memref.cast %beta : memref<4096xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%beta_cast, %b_lo, %b_hi, %c_gen_int) : (memref<*xf32>, f32, f32, i1) -> ()

    %output = memref.alloc() : memref<4096x4096xf32>
    scf.for %i = %c0 to %cM step %c1 {
      scf.for %j = %c0 to %cN step %c1 {
        memref.store %c0_f32, %output[%i, %j] : memref<4096x4096xf32>
      }
    }

    %input_gpu = gpu.alloc () : memref<4096x4096xf32>
    gpu.memcpy %input_gpu, %input : memref<4096x4096xf32>, memref<4096x4096xf32>
    %output_gpu = gpu.alloc () : memref<4096x4096xf32>
    gpu.memcpy %output_gpu, %output : memref<4096x4096xf32>, memref<4096x4096xf32>
    %gamma_gpu = gpu.alloc () : memref<4096xf32>
    gpu.memcpy %gamma_gpu, %gamma : memref<4096xf32>, memref<4096xf32>
    %beta_gpu = gpu.alloc () : memref<4096xf32>
    gpu.memcpy %beta_gpu, %beta : memref<4096xf32>, memref<4096xf32>

    gpu.launch_func  @layer_norm::@layer_norm
      blocks in (%c_grid, %c1, %c1) threads in (%c_block, %c1, %c1)
      args(%input_gpu : memref<4096x4096xf32>, %output_gpu : memref<4096x4096xf32>, %gamma_gpu : memref<4096xf32>, %beta_gpu : memref<4096xf32>)
    gpu.wait

    gpu.memcpy %output, %output_gpu : memref<4096x4096xf32>, memref<4096x4096xf32>
    gpu.dealloc %input_gpu : memref<4096x4096xf32>
    gpu.dealloc %output_gpu : memref<4096x4096xf32>
    gpu.dealloc %gamma_gpu : memref<4096xf32>
    gpu.dealloc %beta_gpu : memref<4096xf32>

    %cpu_out = memref.alloc() : memref<4096x4096xf32>
    call @cpu_reference(%input, %gamma, %beta, %cpu_out) : (memref<4096x4096xf32>, memref<4096xf32>, memref<4096xf32>, memref<4096x4096xf32>) -> ()

    // Check max absolute error against f64; allow f32 drift from row offsets.
    %err_zero = arith.constant 0.000000e+00 : f32
    %maxerr = scf.for %ei = %c0 to %cM step %c1 iter_args(%em0 = %err_zero) -> f32 {
      %em1 = scf.for %ej = %c0 to %cN step %c1 iter_args(%em2 = %em0) -> f32 {
        %eg = memref.load %output[%ei, %ej] : memref<4096x4096xf32>
        %er = memref.load %cpu_out[%ei, %ej] : memref<4096x4096xf32>
        %ed = arith.subf %eg, %er : f32
        %ea = math.absf %ed : f32
        %em = arith.maximumf %em2, %ea : f32
        scf.yield %em : f32
      }
      scf.yield %em1 : f32
    }
    vector.print str "MAX_ABS_ERROR:"
    vector.print %maxerr : f32
    %err_tol = arith.constant 2.000000e-03 : f32
    %err_ok = arith.cmpf olt, %maxerr, %err_tol : f32
    // CHECK: [ALLCLOSE: TRUE]
    scf.if %err_ok {
      vector.print str "[ALLCLOSE: TRUE]"
    } else {
      vector.print str "[ALLCLOSE: FALSE]"
    }

    memref.dealloc %input : memref<4096x4096xf32>
    memref.dealloc %output : memref<4096x4096xf32>
    memref.dealloc %gamma : memref<4096xf32>
    memref.dealloc %beta : memref<4096xf32>
    memref.dealloc %cpu_out : memref<4096x4096xf32>
    return
  }

  // Use f64 statistics to avoid sharing the GPU's f32 rounding error.
  func.func @cpu_reference(%input: memref<4096x4096xf32>, %gamma: memref<4096xf32>, %beta: memref<4096xf32>, %out: memref<4096x4096xf32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cM = arith.constant 4096 : index
    %cN = arith.constant 4096 : index
    %zero64 = arith.constant 0.0 : f64
    %n64 = arith.constant 4096.0 : f64
    %eps64 = arith.constant 1.000000e-05 : f64
    %one64 = arith.constant 1.0 : f64

    scf.for %row = %c0 to %cM step %c1 {
      // Pass 1: mean.
      %sum = scf.for %col = %c0 to %cN step %c1 iter_args(%acc = %zero64) -> f64 {
        %v = memref.load %input[%row, %col] : memref<4096x4096xf32>
        %v64 = arith.extf %v : f32 to f64
        %next = arith.addf %acc, %v64 : f64
        scf.yield %next : f64
      }
      %mean = arith.divf %sum, %n64 : f64

      // Pass 2: variance about that mean.
      %sq = scf.for %col = %c0 to %cN step %c1 iter_args(%acc = %zero64) -> f64 {
        %v = memref.load %input[%row, %col] : memref<4096x4096xf32>
        %v64 = arith.extf %v : f32 to f64
        %d = arith.subf %v64, %mean : f64
        %d2 = arith.mulf %d, %d : f64
        %next = arith.addf %acc, %d2 : f64
        scf.yield %next : f64
      }
      %var = arith.divf %sq, %n64 : f64
      %var_eps = arith.addf %var, %eps64 : f64
      %std = math.sqrt %var_eps : f64
      %inv_std = arith.divf %one64, %std : f64

      // Pass 3: normalize, scale, shift.
      scf.for %col = %c0 to %cN step %c1 {
        %v = memref.load %input[%row, %col] : memref<4096x4096xf32>
        %v64 = arith.extf %v : f32 to f64
        %g = memref.load %gamma[%col] : memref<4096xf32>
        %b = memref.load %beta[%col] : memref<4096xf32>
        %g64 = arith.extf %g : f32 to f64
        %b64 = arith.extf %b : f32 to f64
        %centered = arith.subf %v64, %mean : f64
        %normed = arith.mulf %centered, %inv_std : f64
        %scaled = arith.mulf %normed, %g64 : f64
        %shifted = arith.addf %scaled, %b64 : f64
        %r = arith.truncf %shifted : f64 to f32
        memref.store %r, %out[%row, %col] : memref<4096x4096xf32>
      }
    }
    return
  }

  // One row per work-group, 256 work-items, 1x512 chunks.
  // Keep reductions outside loops: in-loop cross-subgroup reductions currently
  // need inter-iteration barriers for correctness.
  // See ../Performance_report_layer_norm_4096x4096.md.
  gpu.module @layer_norm [#xevm.target<O = 3>] {
    gpu.func @layer_norm(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<4096xf32>, %arg3: memref<4096xf32>) kernel attributes {known_block_size = array<i32: 256, 1, 1>, known_grid_size = array<i32: 4096, 1, 1>} {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 4096 : index
      %cC = arith.constant 512 : index
      %row = gpu.block_id x
      %zero = arith.constant dense<0.000000e+00> : vector<1xf32>
      %zero_tile = arith.constant dense<0.000000e+00> : vector<1x512xf32>
      %inv_n = arith.constant dense<2.44140625E-4> : vector<1xf32>
      %eps = arith.constant dense<9.99999974E-6> : vector<1xf32>

      %input_desc = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf32> -> !xegpu.tensor_desc<1x512xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      %output_desc = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf32> -> !xegpu.tensor_desc<1x512xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      %gamma_2d = memref.expand_shape %arg2 [[0, 1]] output_shape [1, 4096] : memref<4096xf32> into memref<1x4096xf32>
      %beta_2d = memref.expand_shape %arg3 [[0, 1]] output_shape [1, 4096] : memref<4096xf32> into memref<1x4096xf32>
      %gamma_desc = xegpu.create_nd_tdesc %gamma_2d : memref<1x4096xf32> -> !xegpu.tensor_desc<1x512xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      %beta_desc = xegpu.create_nd_tdesc %beta_2d : memref<1x4096xf32> -> !xegpu.tensor_desc<1x512xf32, #xegpu.block_tdesc_attr<boundary_check = false>>

      // Pass 1: mean.
      %psum = scf.for %k1 = %c0 to %cN step %cC iter_args(%acc = %zero_tile) -> vector<1x512xf32> {
        %tile = xegpu.load_nd %input_desc[%row, %k1] <{layout = #xegpu.layout<sg_layout = [1, 16], sg_data = [1, 32]>}> : !xegpu.tensor_desc<1x512xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<1x512xf32>
        %next = arith.addf %acc, %tile fastmath<fast> : vector<1x512xf32>
        scf.yield %next : vector<1x512xf32>
      }
      %sum = vector.multi_reduction <add>, %psum, %zero [1] : vector<1x512xf32> to vector<1xf32>
      %mean = arith.mulf %sum, %inv_n fastmath<fast> : vector<1xf32>
      %mean_bc = vector.broadcast %mean : vector<1xf32> to vector<1x512xf32>

      // Pass 2: variance about that mean.
      %psq = scf.for %k2 = %c0 to %cN step %cC iter_args(%acc2 = %zero_tile) -> vector<1x512xf32> {
        %tile = xegpu.load_nd %input_desc[%row, %k2] <{layout = #xegpu.layout<sg_layout = [1, 16], sg_data = [1, 32]>}> : !xegpu.tensor_desc<1x512xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<1x512xf32>
        %centered = arith.subf %tile, %mean_bc fastmath<fast> : vector<1x512xf32>
        %squared = arith.mulf %centered, %centered fastmath<fast> : vector<1x512xf32>
        %next = arith.addf %acc2, %squared fastmath<fast> : vector<1x512xf32>
        scf.yield %next : vector<1x512xf32>
      }
      %sum_sq = vector.multi_reduction <add>, %psq, %zero [1] : vector<1x512xf32> to vector<1xf32>
      %variance = arith.mulf %sum_sq, %inv_n fastmath<fast> : vector<1xf32>
      %variance_eps = arith.addf %variance, %eps fastmath<fast> : vector<1xf32>
      %inv_std = math.rsqrt %variance_eps fastmath<fast> : vector<1xf32>
      %inv_std_bc = vector.broadcast %inv_std : vector<1xf32> to vector<1x512xf32>

      // Pass 3: normalize, scale, shift and store.
      // Bypass L1 for streamed input; cache gamma/beta reused across work-groups.
      scf.for %k3 = %c0 to %cN step %cC {
        %tile = xegpu.load_nd %input_desc[%row, %k3] <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<cached>, layout = #xegpu.layout<sg_layout = [1, 16], sg_data = [1, 32]>}> : !xegpu.tensor_desc<1x512xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<1x512xf32>
        %gamma = xegpu.load_nd %gamma_desc[%c0, %k3] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #xegpu.layout<sg_layout = [1, 16], sg_data = [1, 32]>}> : !xegpu.tensor_desc<1x512xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<1x512xf32>
        %beta = xegpu.load_nd %beta_desc[%c0, %k3] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #xegpu.layout<sg_layout = [1, 16], sg_data = [1, 32]>}> : !xegpu.tensor_desc<1x512xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<1x512xf32>
        %centered = arith.subf %tile, %mean_bc fastmath<fast> : vector<1x512xf32>
        %normalized = arith.mulf %centered, %inv_std_bc fastmath<fast> : vector<1x512xf32>
        %scaled = arith.mulf %normalized, %gamma fastmath<fast> : vector<1x512xf32>
        %result = arith.addf %scaled, %beta fastmath<fast> : vector<1x512xf32>
        xegpu.store_nd %result, %output_desc[%row, %k3] <{layout = #xegpu.layout<sg_layout = [1, 16], sg_data = [1, 32]>}> : vector<1x512xf32>, !xegpu.tensor_desc<1x512xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      }
      gpu.return
    }
  }

  func.func private @fillResource1DRandomF32(memref<*xf32>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
