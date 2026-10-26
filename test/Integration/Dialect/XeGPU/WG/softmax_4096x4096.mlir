// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup" | \
// RUN:   mlir-runner \
// RUN:     --shared-libs=%mlir_levelzero_runtime \
// RUN:     --shared-libs=%mlir_runner_utils \
// RUN:     --shared-libs=%mlir_c_runner_utils \
// RUN:     --shared-libs=%irunner_utils \
// RUN:     --entry-point-result=void \
// RUN: | FileCheck %s


module attributes {gpu.container_module} {
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    %c4096 = arith.constant 4096 : index
    %c0_f32 = arith.constant 0.0 : f32
    %cf_lower = arith.constant -0.5 : f32
    %cf_upper = arith.constant 0.5 : f32
    %c_gen_int = arith.constant 0 : i1

    %arg0 = memref.alloc() : memref<4096x4096xf32>
    %arg0_cast = memref.cast %arg0 : memref<4096x4096xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%arg0_cast, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf32>, f32, f32, i1) -> ()

    %arg1 = memref.alloc() : memref<4096x4096xf32>
    scf.for %i = %c0 to %c4096 step %c1 {
      scf.for %j = %c0 to %c4096 step %c1 {
        memref.store %c0_f32, %arg1[%i, %j] : memref<4096x4096xf32>
      }
    }

    %arg0_gpu = gpu.alloc () : memref<4096x4096xf32>
    gpu.memcpy %arg0_gpu, %arg0 : memref<4096x4096xf32>, memref<4096x4096xf32>
    %arg1_gpu = gpu.alloc () : memref<4096x4096xf32>
    gpu.memcpy %arg1_gpu, %arg1 : memref<4096x4096xf32>, memref<4096x4096xf32>

    gpu.launch_func  @softmax::@softmax
      blocks in (%c4096, %c1, %c1) threads in (%c512, %c1, %c1)
      args(%arg0_gpu : memref<4096x4096xf32>, %arg1_gpu : memref<4096x4096xf32>)
    gpu.wait

    gpu.memcpy %arg1, %arg1_gpu : memref<4096x4096xf32>, memref<4096x4096xf32>
    gpu.dealloc %arg0_gpu : memref<4096x4096xf32>
    gpu.dealloc %arg1_gpu : memref<4096x4096xf32>

    %cpu_out = memref.alloc() : memref<4096x4096xf32>
    scf.for %i = %c0 to %c4096 step %c1 {
      scf.for %j = %c0 to %c4096 step %c1 {
        memref.store %c0_f32, %cpu_out[%i, %j] : memref<4096x4096xf32>
      }
    }
    call @cpu_reference(%arg0, %cpu_out) : (memref<4096x4096xf32>, memref<4096x4096xf32>) -> ()

    %arg1_star = memref.cast %arg1 : memref<4096x4096xf32> to memref<*xf32>
    %cpu_out_star = memref.cast %cpu_out : memref<4096x4096xf32> to memref<*xf32>
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%arg1_star, %cpu_out_star) : (memref<*xf32>, memref<*xf32>) -> ()

    memref.dealloc %arg0 : memref<4096x4096xf32>
    memref.dealloc %arg1 : memref<4096x4096xf32>
    memref.dealloc %cpu_out : memref<4096x4096xf32>
    return
  }

  // Numerically stable row-wise softmax reference (max-subtract, exp, normalize).
  func.func @cpu_reference(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4096 = arith.constant 4096 : index
    %neg_inf = arith.constant 0xFF800000 : f32
    %zero = arith.constant 0.0 : f32
    scf.for %row = %c0 to %c4096 step %c1 {
      // Step 1: find row max
      %max = scf.for %col = %c0 to %c4096 step %c1 iter_args(%cur_max = %neg_inf) -> f32 {
        %val = memref.load %arg0[%row, %col] : memref<4096x4096xf32>
        %new_max = arith.maximumf %cur_max, %val : f32
        scf.yield %new_max : f32
      }
      // Step 2: compute exp(x - max) and accumulate sum
      %sum = scf.for %col = %c0 to %c4096 step %c1 iter_args(%cur_sum = %zero) -> f32 {
        %val = memref.load %arg0[%row, %col] : memref<4096x4096xf32>
        %shifted = arith.subf %val, %max : f32
        %exp_val = math.exp %shifted : f32
        memref.store %exp_val, %arg1[%row, %col] : memref<4096x4096xf32>
        %new_sum = arith.addf %cur_sum, %exp_val : f32
        scf.yield %new_sum : f32
      }
      // Step 3: divide by sum
      scf.for %col = %c0 to %c4096 step %c1 {
        %exp_val = memref.load %arg1[%row, %col] : memref<4096x4096xf32>
        %result = arith.divf %exp_val, %sum : f32
        memref.store %result, %arg1[%row, %col] : memref<4096x4096xf32>
      }
    }
    return
  }

  // One row per work-group, 512 work-items, 1x1024 chunks.
  // Keep reductions outside loops: in-loop cross-subgroup reductions currently
  // need inter-iteration barriers for correctness.
  // See ../Performance_report_softmax_4096x4096.md.
  gpu.module @softmax [#xevm.target<O = 3>] {
    gpu.func @softmax(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>) kernel attributes {known_block_size = array<i32: 512, 1, 1>, known_grid_size = array<i32: 4096, 1, 1>} {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 4096 : index
      %cC = arith.constant 1024 : index
      %neg_inf = arith.constant dense<0xFF800000> : vector<1xf32>
      %neg_inf_tile = arith.constant dense<0xFF800000> : vector<1x1024xf32>
      %zero = arith.constant dense<0.000000e+00> : vector<1xf32>
      %zero_tile = arith.constant dense<0.000000e+00> : vector<1x1024xf32>
      %one = arith.constant dense<1.000000e+00> : vector<1xf32>
      %row = gpu.block_id x
      %input_desc = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf32> -> !xegpu.tensor_desc<1x1024xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      %output_desc = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf32> -> !xegpu.tensor_desc<1x1024xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      // Pass 1: row maximum.
      %pmax = scf.for %k1 = %c0 to %cN step %cC iter_args(%accm = %neg_inf_tile) -> vector<1x1024xf32> {
        %tile = xegpu.load_nd %input_desc[%row, %k1] <{layout = #xegpu.layout<sg_layout = [1, 32], sg_data = [1, 32]>}> : !xegpu.tensor_desc<1x1024xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<1x1024xf32>
        %next = arith.maximumf %accm, %tile : vector<1x1024xf32>
        scf.yield %next : vector<1x1024xf32>
      }
      %row_max = vector.multi_reduction <maximumf>, %pmax, %neg_inf [1] : vector<1x1024xf32> to vector<1xf32>
      %max_bc = vector.broadcast %row_max : vector<1xf32> to vector<1x1024xf32>
      // Pass 2: sum of exp(x - row_max).
      %psum = scf.for %k2 = %c0 to %cN step %cC iter_args(%accs = %zero_tile) -> vector<1x1024xf32> {
        %tile = xegpu.load_nd %input_desc[%row, %k2] <{layout = #xegpu.layout<sg_layout = [1, 32], sg_data = [1, 32]>}> : !xegpu.tensor_desc<1x1024xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<1x1024xf32>
        %shifted = arith.subf %tile, %max_bc : vector<1x1024xf32>
        %exp_tile = math.exp %shifted : vector<1x1024xf32>
        %next = arith.addf %accs, %exp_tile : vector<1x1024xf32>
        scf.yield %next : vector<1x1024xf32>
      }
      %row_sum = vector.multi_reduction <add>, %psum, %zero [1] : vector<1x1024xf32> to vector<1xf32>
      // Hoist the reciprocal out of the normalization loop.
      %inv_sum = arith.divf %one, %row_sum : vector<1xf32>
      %inv_bc = vector.broadcast %inv_sum : vector<1xf32> to vector<1x1024xf32>
      // Pass 3: normalize and store.
      scf.for %k3 = %c0 to %cN step %cC {
        %tile = xegpu.load_nd %input_desc[%row, %k3] <{layout = #xegpu.layout<sg_layout = [1, 32], sg_data = [1, 32]>}> : !xegpu.tensor_desc<1x1024xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<1x1024xf32>
        %shifted = arith.subf %tile, %max_bc : vector<1x1024xf32>
        %exp_tile = math.exp %shifted : vector<1x1024xf32>
        %normalized = arith.mulf %exp_tile, %inv_bc : vector<1x1024xf32>
        xegpu.store_nd %normalized, %output_desc[%row, %k3] <{layout = #xegpu.layout<sg_layout = [1, 32], sg_data = [1, 32]>}> : vector<1x1024xf32>, !xegpu.tensor_desc<1x1024xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      }
      gpu.return
    }
  }

  func.func private @fillResource1DRandomF32(memref<*xf32>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
