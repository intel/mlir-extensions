// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup" | \
// RUN:   mlir-runner \
// RUN:     --shared-libs=%mlir_levelzero_runtime \
// RUN:     --shared-libs=%mlir_runner_utils \
// RUN:     --shared-libs=%mlir_c_runner_utils \
// RUN:     --shared-libs=%irunner_utils \
// RUN:     --entry-point-result=void
#lo_sg_8x1_data_8x32 = #xegpu.layout<sg_layout = [8, 1], sg_data = [8, 32], order = [1, 0]>
module attributes {gpu.container_module} {
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %c0_f32 = arith.constant 0.0 : f32
    %cf_lower = arith.constant -0.5 : f32
    %cf_upper = arith.constant 0.5 : f32
    %c_gen_int = arith.constant 0 : i1

    // Allocate and randomly initialize input in [-0.5, 0.5]
    %arg0 = memref.alloc() : memref<1024x512xf32>
    %arg0_cast = memref.cast %arg0 : memref<1024x512xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%arg0_cast, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf32>, f32, f32, i1) -> ()

    // Allocate and zero-initialize GPU output buffer (CPU side)
    %arg1 = memref.alloc() : memref<1024x512xf32>
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c512 step %c1 {
        memref.store %c0_f32, %arg1[%i, %j] : memref<1024x512xf32>
      }
    }

    // Allocate GPU buffers and copy input/output to GPU
    %arg0_gpu = gpu.alloc () : memref<1024x512xf32>
    gpu.memcpy %arg0_gpu, %arg0 : memref<1024x512xf32>, memref<1024x512xf32>
    %arg1_gpu = gpu.alloc () : memref<1024x512xf32>
    gpu.memcpy %arg1_gpu, %arg1 : memref<1024x512xf32>, memref<1024x512xf32>

    // Launch kernel and wait for completion
    gpu.launch_func  @main_kernel::@main_kernel
      blocks in (%c16, %c1, %c1) threads in (%c128, %c1, %c1)
      args(%arg0_gpu : memref<1024x512xf32>, %arg1_gpu : memref<1024x512xf32>)
    gpu.wait

    // Copy result back to host
    gpu.memcpy %arg1, %arg1_gpu : memref<1024x512xf32>, memref<1024x512xf32>
    gpu.dealloc %arg0_gpu : memref<1024x512xf32>
    gpu.dealloc %arg1_gpu : memref<1024x512xf32>

    // Compute CPU reference
    %cpu_out = memref.alloc() : memref<1024x512xf32>
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c512 step %c1 {
        memref.store %c0_f32, %cpu_out[%i, %j] : memref<1024x512xf32>
      }
    }
    call @cpu_reference(%arg0, %cpu_out) : (memref<1024x512xf32>, memref<1024x512xf32>) -> ()

    // Compare GPU and CPU results
    %arg1_star = memref.cast %arg1 : memref<1024x512xf32> to memref<*xf32>
    %cpu_out_star = memref.cast %cpu_out : memref<1024x512xf32> to memref<*xf32>
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%arg1_star, %cpu_out_star) : (memref<*xf32>, memref<*xf32>) -> ()
    // Debug print the first row of GPU and CPU output
    // %gpu_row_0 = memref.subview %arg1[%c0, %c0] [%c1, %c512] [%c1, %c1]
    //   : memref<1024x512xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    // %gpu_row_0_star = memref.cast %gpu_row_0
    //   : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<*xf32>
    // call @printMemrefF32(%gpu_row_0_star) : (memref<*xf32>) -> ()

    // %cpu_row_0 = memref.subview %cpu_out[%c0, %c0] [%c1, %c512] [%c1, %c1]
    //   : memref<1024x512xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    // %cpu_row_0_star = memref.cast %cpu_row_0
    //   : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<*xf32>
    // call @printMemrefF32(%cpu_row_0_star) : (memref<*xf32>) -> ()

    memref.dealloc %arg0 : memref<1024x512xf32>
    memref.dealloc %arg1 : memref<1024x512xf32>
    memref.dealloc %cpu_out : memref<1024x512xf32>
    return
  }
  func.func @cpu_reference(%arg0: memref<1024x512xf32>, %arg1: memref<1024x512xf32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c512 = arith.constant 512 : index
    %neg_inf = arith.constant 0xFF800000 : f32
    %zero = arith.constant 0.0 : f32
    // Iterate over each row
    scf.for %row = %c0 to %c1024 step %c1 {
      // Step 1: find row max
      %max = scf.for %col = %c0 to %c512 step %c1 iter_args(%cur_max = %neg_inf) -> f32 {
        %val = memref.load %arg0[%row, %col] : memref<1024x512xf32>
        %new_max = arith.maximumf %cur_max, %val : f32
        scf.yield %new_max : f32
      }
      // Step 2: compute exp(x - max) and accumulate sum
      %sum = scf.for %col = %c0 to %c512 step %c1 iter_args(%cur_sum = %zero) -> f32 {
        %val = memref.load %arg0[%row, %col] : memref<1024x512xf32>
        %shifted = arith.subf %val, %max : f32
        %exp_val = math.exp %shifted : f32
        memref.store %exp_val, %arg1[%row, %col] : memref<1024x512xf32>
        %new_sum = arith.addf %cur_sum, %exp_val : f32
        scf.yield %new_sum : f32
      }
      // Step 3: divide by sum
      scf.for %col = %c0 to %c512 step %c1 {
        %exp_val = memref.load %arg1[%row, %col] : memref<1024x512xf32>
        %result = arith.divf %exp_val, %sum : f32
        memref.store %result, %arg1[%row, %col] : memref<1024x512xf32>
      }
    }
    return
  }
  gpu.module @main_kernel [#xevm.target<chip = "pvc">] {
    gpu.func @main_kernel(%arg0: memref<1024x512xf32>, %arg1: memref<1024x512xf32>) kernel attributes {intel_reqd_sub_group_size = 16 : i32, known_block_size = array<i32: 128, 1, 1>, known_grid_size = array<i32: 16, 1, 1>} {
      %neg_inf_row = arith.constant  dense<0xFF800000> : vector<64xf32>
      %zero_row    = arith.constant  dense<0.000000e+00> : vector<64xf32>
      %c0   = arith.constant 0 : index
      %c32  = arith.constant 32 : index
      %c64  = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %block_id_x = gpu.block_id  x
      %row_base = arith.muli %block_id_x, %c64 overflow<nsw> : index
      %in_desc = xegpu.create_nd_tdesc %arg0 : memref<1024x512xf32>
        -> !xegpu.tensor_desc<64x32xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      %out_desc = xegpu.create_nd_tdesc %arg1 : memref<1024x512xf32>
        -> !xegpu.tensor_desc<64x32xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      // Loop 1: walk 64x32 tiles left-to-right, maintaining global row-max and
      //         global row-sum = sum_k exp(x_k - global_max).
      //         When global_max grows, rescale the running sum by exp(old - new).
      %result:2 = scf.for %k = %c0 to %c512 step %c32
          iter_args(%global_max = %neg_inf_row, %global_sum = %zero_row)
          -> (vector<64xf32>, vector<64xf32>) {
        // Load 64x32 input tile at column offset k
        %tile = xegpu.load_nd %in_desc[%row_base, %k] <{layout = #lo_sg_8x1_data_8x32}> :
          !xegpu.tensor_desc<64x32xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
          -> vector<64x32xf32>
        // Per-row max of current tile
        %tile_max = vector.multi_reduction <maximumf>, %tile, %neg_inf_row

          [1] : vector<64x32xf32> to vector<64xf32>
        // Update global max
        %new_global_max = arith.maximumf %global_max, %tile_max
           : vector<64xf32>
        // Correction factor: exp(old_global_max - new_global_max)
        %diff = arith.subf %global_max, %new_global_max
           : vector<64xf32>
        %alpha = math.exp %diff
           : vector<64xf32>
        // Rescale accumulated sum: global_sum = global_sum * alpha
        %global_sum_rescaled = arith.mulf %global_sum, %alpha
           : vector<64xf32>
        // Broadcast new_global_max across columns of the 64x32 tile
        %max_bc1 = vector.broadcast %new_global_max
           : vector<64xf32> to vector<32x64xf32>
        %max_bc1_t = vector.transpose %max_bc1, [1, 0]
          : vector<32x64xf32> to vector<64x32xf32>
        // exp(tile - new_global_max)
        %shifted1 = arith.subf %tile, %max_bc1_t
          : vector<64x32xf32>
        %exp_tile1 = math.exp %shifted1
          : vector<64x32xf32>
        // Per-row sum of exp values for this tile
        %tile_sum = vector.multi_reduction <add>, %exp_tile1, %zero_row

          [1] : vector<64x32xf32> to vector<64xf32>
        // Update global sum: rescaled previous + new tile contribution
        %new_global_sum = arith.addf %global_sum_rescaled, %tile_sum
           : vector<64xf32>
        scf.yield %new_global_max, %new_global_sum : vector<64xf32>, vector<64xf32>
      }
      // Loop 2: second pass -- normalize each 64x32 tile:
      //         output = exp(tile - global_max) / global_sum
      scf.for %k2 = %c0 to %c512 step %c32 {
        // Load 64x32 input tile
        %tile2 = xegpu.load_nd %in_desc[%row_base, %k2] <{layout = #lo_sg_8x1_data_8x32}> :
          !xegpu.tensor_desc<64x32xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
          -> vector<64x32xf32>
        // Broadcast global_max across columns
        %max_bc2 = vector.broadcast %result#0
           : vector<64xf32> to vector<32x64xf32>
        %max_bc2_t = vector.transpose %max_bc2, [1, 0]
          : vector<32x64xf32> to vector<64x32xf32>
        // exp(tile - global_max)
        %shifted2 = arith.subf %tile2, %max_bc2_t
          : vector<64x32xf32>
        %exp_tile2 = math.exp %shifted2
          : vector<64x32xf32>
        // Broadcast global_sum across columns
        %sum_bc2 = vector.broadcast %result#1
           : vector<64xf32> to vector<32x64xf32>
        %sum_bc2_t = vector.transpose %sum_bc2, [1, 0]
          : vector<32x64xf32> to vector<64x32xf32>
        // Normalize and store
        %out_tile = arith.divf %exp_tile2, %sum_bc2_t
          : vector<64x32xf32>
        xegpu.store_nd %out_tile, %out_desc[%row_base, %k2] <{layout = #lo_sg_8x1_data_8x32}>
          : vector<64x32xf32>, !xegpu.tensor_desc<64x32xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      }
      gpu.return
    }
  }
  func.func private @fillResource1DRandomF32(memref<*xf32>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
