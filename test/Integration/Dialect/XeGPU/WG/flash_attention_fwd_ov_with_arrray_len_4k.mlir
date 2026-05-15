// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup igc-cmd-options=-ze-opt-large-register-file" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --shared-libs=%irunner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s


#q = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64]>
#q_load = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [16, 32]>
#q_ord = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], order = [0, 1]>
#t_ord = #xegpu.layout<sg_layout = [1, 8], sg_data = [64, 16], order = [0, 1]>
#q_s1 = #xegpu.slice<#q_ord, dims = [1]>
#t_s0 = #xegpu.slice<#t_ord, dims = [0]>
#pf = #xegpu.layout<sg_layout = [2, 4], sg_data = [32, 16], inst_data = [32, 16]>
#kv_load = #xegpu.layout<sg_layout = [1, 1], sg_data = [64, 64], inst_data = [32, 32]>
#kv = #xegpu.layout<sg_layout = [1, 1], sg_data = [64, 64]>
#kv_ord = #xegpu.layout<sg_layout = [1, 1], sg_data = [64, 64], order = [0, 1]>

module @fragment_name attributes {gpu.container_module} {
  func.func @entry(%arg0: memref<16x4096x64xf16>, %arg1: memref<16x4096x64xf16>, %arg2: memref<16x4096x64xf16>, %arg3: memref<16x4096x64xf16>) attributes {gc.num_kernels = 1 : i32} {
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c16, %c32, %c1) threads in (%c128, %c1, %c1)  args(%arg0 : memref<16x4096x64xf16>, %arg1 : memref<16x4096x64xf16>, %arg2 : memref<16x4096x64xf16>, %arg3 : memref<16x4096x64xf16>)
    return
  }
  gpu.module @entry_kernel [#xevm.target<O = 3>, #xevm.target<flags = {"cmd-options" = ["-ze-opt-large-register-file"]}>] {
  gpu.func @entry_kernel(%Q: memref<16x4096x64xf16>, %K: memref<16x4096x64xf16>, %V: memref<16x4096x64xf16>, %Out: memref<16x4096x64xf16>) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c4096 = arith.constant 4096 : index
    %l_i_init = arith.constant dense<0.000000e+00> : vector<128xf32>
    // -inf constant for max reduction init
    %m_i_init = arith.constant dense<0xFF800000> : vector<128xf32>
    // constant scale propagated from OV (sm_scale folded with log2(e) for exp-via-exp2)
    %qk_scale = arith.constant dense<0.350097656> : vector<128x64xf32>
    %zero_acc = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %c0 = arith.constant 0 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y

    // Load Q tile
    %q_base_buffer, %q_offset, %q_sizes:3, %q_strides:3 = memref.extract_strided_metadata %Q : memref<16x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
    %c262144 = arith.constant 262144 : index
    %q_offset_bh = arith.muli %block_id_x, %c262144 overflow<nsw> : index
    %c8192 = arith.constant 8192 : index
    %q_offset_row = arith.muli %block_id_y, %c8192 overflow<nsw> : index
    %q_offset_base = arith.addi %q_offset_bh, %q_offset_row : index
    %q_intptr = memref.extract_aligned_pointer_as_index %q_base_buffer : memref<f16> -> index
    %q_offset_bytes = arith.muli %q_offset_base, %c2 : index
    %q_ptr_offset = arith.addi %q_intptr, %q_offset_bytes : index
    %q_ptr_i64 = arith.index_cast %q_ptr_offset : index to i64
    %q_tile = xegpu.create_nd_tdesc %q_ptr_i64, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #q>
    %q_load_tile = xegpu.create_nd_tdesc %q_ptr_i64, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #q_load>
    %q_value = xegpu.load_nd %q_load_tile[0, 0] <{layout = #q_load}> : !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #q_load> -> vector<128x64xf16>

    // Prefetch first K tile (iteration 0)
    %k_base_buffer, %k_offset_meta, %k_sizes:3, %k_strides:3 = memref.extract_strided_metadata %K : memref<16x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
    %k_intptr = memref.extract_aligned_pointer_as_index %k_base_buffer : memref<f16> -> index
    %c262144_kv = arith.constant 262144 : index
    %kv_offset_bh = arith.muli %block_id_x, %c262144_kv overflow<nsw> : index
    %kv_offset_bytes_bh = arith.muli %kv_offset_bh, %c2 : index
    %k_prefetch0_ptr = arith.addi %k_intptr, %kv_offset_bytes_bh : index
    %k_prefetch0_ptr_i64 = arith.index_cast %k_prefetch0_ptr : index to i64
    %k_prefetch0_tile = xegpu.create_nd_tdesc %k_prefetch0_ptr_i64, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #pf>
    xegpu.prefetch_nd %k_prefetch0_tile[0, 0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #pf}> : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #pf>

    // Prefetch first V tile (iteration 0)
    %v_base_buffer, %v_offset_meta, %v_sizes:3, %v_strides:3 = memref.extract_strided_metadata %V : memref<16x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
    %v_intptr = memref.extract_aligned_pointer_as_index %v_base_buffer : memref<f16> -> index
    %v_prefetch0_ptr = arith.addi %v_intptr, %kv_offset_bytes_bh : index
    %v_prefetch0_ptr_i64 = arith.index_cast %v_prefetch0_ptr : index to i64
    %v_prefetch0_tile = xegpu.create_nd_tdesc %v_prefetch0_ptr_i64, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #pf>
    xegpu.prefetch_nd %v_prefetch0_tile[0, 0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #pf}> : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #pf>

    // Main loop over K and V tiles
    %result:3 = scf.for %k_offset = %c0 to %c4096 step %c64 iter_args(%acc_in = %zero_acc, %m_i_in = %m_i_init, %l_i_in = %l_i_init) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
      // Prefetch next-iteration K tile
      %k_next_offset = arith.addi %k_offset, %c64 : index
      %c262144_next = arith.constant 262144 : index
      %kv_next_offset_bh = arith.muli %block_id_x, %c262144_next overflow<nsw> : index
      %c64_stride = arith.constant 64 : index
      %k_next_offset_row = arith.muli %k_next_offset, %c64_stride overflow<nsw> : index
      %k_next_offset_base = arith.addi %kv_next_offset_bh, %k_next_offset_row : index
      %k_next_offset_bytes = arith.muli %k_next_offset_base, %c2 : index
      %k_prefetch_ptr = arith.addi %k_intptr, %k_next_offset_bytes : index
      %k_prefetch_ptr_i64 = arith.index_cast %k_prefetch_ptr : index to i64
      %k_prefetch_tile = xegpu.create_nd_tdesc %k_prefetch_ptr_i64, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #pf>
      xegpu.prefetch_nd %k_prefetch_tile[0, 0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #pf}> : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #pf>

      // Prefetch next-iteration V tile
      %v_prefetch_ptr = arith.addi %v_intptr, %k_next_offset_bytes : index
      %v_prefetch_ptr_i64 = arith.index_cast %v_prefetch_ptr : index to i64
      %v_prefetch_tile = xegpu.create_nd_tdesc %v_prefetch_ptr_i64, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #pf>
      xegpu.prefetch_nd %v_prefetch_tile[0, 0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #pf}> : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #pf>

      // Load K tile (current iteration)
      %c262144_curr = arith.constant 262144 : index
      %kv_curr_offset_bh = arith.muli %block_id_x, %c262144_curr overflow<nsw> : index
      %c64_stride_curr = arith.constant 64 : index
      %k_curr_offset_row = arith.muli %k_offset, %c64_stride_curr overflow<nsw> : index
      %k_curr_offset_base = arith.addi %kv_curr_offset_bh, %k_curr_offset_row : index
      %k_curr_offset_bytes = arith.muli %k_curr_offset_base, %c2 : index
      %k_ptr_offset = arith.addi %k_intptr, %k_curr_offset_bytes : index
      %k_ptr_i64 = arith.index_cast %k_ptr_offset : index to i64
      %k_tile = xegpu.create_nd_tdesc %k_ptr_i64, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #kv_ord>
      %k_value = xegpu.load_nd %k_tile[0, 0] <{layout = #kv_ord}> : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #kv_ord> -> vector<64x64xf16>

      // Load V tile (current iteration)
      %v_ptr_offset = arith.addi %v_intptr, %k_curr_offset_bytes : index
      %v_ptr_i64 = arith.index_cast %v_ptr_offset : index to i64
      %v_tile_unused = xegpu.create_nd_tdesc %v_ptr_i64, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #kv>
      %v_load_tile = xegpu.create_nd_tdesc %v_ptr_i64, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #kv_load>
      %v_value = xegpu.load_nd %v_load_tile[0, 0] <{layout = #kv_load}> : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #kv_load> -> vector<64x64xf16>

      // Compute Q * K^T
      %k_value_t = vector.transpose %k_value, [1, 0] : vector<64x64xf16> to vector<64x64xf16>
      %qk_out = xegpu.dpas %q_value, %k_value_t, %zero_acc {layout_a = #q, layout_b = #kv, layout_cd = #q} : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
      %qk_scaled = arith.mulf %qk_out, %qk_scale : vector<128x64xf32>

      // Online softmax: compute row-wise max
      %qk_row_max = vector.multi_reduction <maximumf>, %qk_scaled, %m_i_init [1] : vector<128x64xf32> to vector<128xf32>
      %m_ij = arith.maximumf %m_i_in, %qk_row_max : vector<128xf32>

      // Center by m_ij and compute exp
      %m_ij_broadcasted_t = vector.broadcast %m_ij : vector<128xf32> to vector<64x128xf32>
      %m_ij_broadcasted = vector.transpose %m_ij_broadcasted_t, [1, 0] : vector<64x128xf32> to vector<128x64xf32>
      %qk_centered = arith.subf %qk_scaled, %m_ij_broadcasted : vector<128x64xf32>
      %p_out = math.exp %qk_centered fastmath<fast> : vector<128x64xf32>

      // Sum exp values
      %l_ij = vector.multi_reduction <add>, %p_out, %l_i_init [1] : vector<128x64xf32> to vector<128xf32>

      // Update running statistics
      %m_diff = arith.subf %m_i_in, %m_ij : vector<128xf32>
      %alpha = math.exp %m_diff fastmath<fast> : vector<128xf32>
      %l_i_scaled = arith.mulf %l_i_in, %alpha : vector<128xf32>
      %l_i_new = arith.addf %l_i_scaled, %l_ij : vector<128xf32>

      // Scale previous accumulator by alpha
      %alpha_broadcasted_t = vector.broadcast %alpha : vector<128xf32> to vector<64x128xf32>
      %alpha_broadcasted = vector.transpose %alpha_broadcasted_t, [1, 0] : vector<64x128xf32> to vector<128x64xf32>
      %acc_scaled = arith.mulf %acc_in, %alpha_broadcasted : vector<128x64xf32>

      // Convert P to f16 for DPAS
      %p_out_f16 = arith.truncf %p_out : vector<128x64xf32> to vector<128x64xf16>

      // Compute P * V and add to scaled accumulator
      %pv_out = xegpu.dpas %p_out_f16, %v_value, %acc_scaled {layout_a = #q, layout_b = #kv, layout_cd = #q} : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>

      scf.yield %pv_out, %m_ij, %l_i_new : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
    }

    // Normalize output by l_i
    %l_i_broadcasted_t = vector.broadcast %result#2 : vector<128xf32> to vector<64x128xf32>
    %l_i_broadcasted = vector.transpose %l_i_broadcasted_t, [1, 0] : vector<64x128xf32> to vector<128x64xf32>
    %out_normalized = arith.divf %result#0, %l_i_broadcasted : vector<128x64xf32>
    %out_f16 = arith.truncf %out_normalized : vector<128x64xf32> to vector<128x64xf16>

    // Store output
    %o_base_buffer, %o_offset_meta, %o_sizes:3, %o_strides:3 = memref.extract_strided_metadata %Out : memref<16x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
    %o_intptr = memref.extract_aligned_pointer_as_index %o_base_buffer : memref<f16> -> index
    %o_ptr_offset = arith.addi %o_intptr, %q_offset_bytes : index
    %o_ptr_i64 = arith.index_cast %o_ptr_offset : index to i64
    %o_tile = xegpu.create_nd_tdesc %o_ptr_i64, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #q>
    xegpu.store_nd %out_f16, %o_tile[0, 0] <{layout = #q}> : vector<128x64xf16>, !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #q>
    gpu.return
  }
}
  func.func @gpu_impl(%arg0: memref<16x4096x64xf16>, %arg1: memref<16x4096x64xf16>, %arg2: memref<16x4096x64xf16>, %arg3: memref<16x4096x64xf16>) -> memref<16x4096x64xf16> {
    // Allocate GPU buffers
    %memref = gpu.alloc  () : memref<16x4096x64xf16>
    %memref_0 = gpu.alloc  () : memref<16x4096x64xf16>
    %memref_1 = gpu.alloc  () : memref<16x4096x64xf16>
    %memref_2 = gpu.alloc  () : memref<16x4096x64xf16>

    // Copy from CPU to GPU
    gpu.memcpy  %memref, %arg0 : memref<16x4096x64xf16>, memref<16x4096x64xf16>
    gpu.memcpy  %memref_0, %arg1 : memref<16x4096x64xf16>, memref<16x4096x64xf16>
    gpu.memcpy  %memref_1, %arg2 : memref<16x4096x64xf16>, memref<16x4096x64xf16>

    // Call the entry function which launches the kernel
    call @entry(%memref, %memref_0, %memref_1, %memref_2) : (memref<16x4096x64xf16>, memref<16x4096x64xf16>, memref<16x4096x64xf16>, memref<16x4096x64xf16>) -> ()

    // Wait for GPU to finish
    gpu.wait

    // Copy output back to CPU
    gpu.memcpy  %arg3, %memref_2 : memref<16x4096x64xf16>, memref<16x4096x64xf16>

    // Cleanup GPU buffers
    gpu.dealloc  %memref : memref<16x4096x64xf16>
    gpu.dealloc  %memref_0 : memref<16x4096x64xf16>
    gpu.dealloc  %memref_1 : memref<16x4096x64xf16>
    gpu.dealloc  %memref_2 : memref<16x4096x64xf16>
    return %arg3 : memref<16x4096x64xf16>
  }
  func.func @cpu_impl(%arg0: memref<16x4096x64xf16>, %arg1: memref<16x4096x64xf16>, %arg2: memref<16x4096x64xf16>, %arg3: memref<16x4096x64xf16>, %arg4: f32) -> memref<16x4096x64xf16> {
    %cst = arith.constant 0xFF800000 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %cst_0 = arith.constant 0.000000e+00 : f32

    // Buffer for QK intermediate results
    %alloc = memref.alloc(%c4096, %c4096) : memref<?x?xf32>

    // Iterate over all batch*head combinations
    scf.for %arg5 = %c0 to %c16 step %c1 {
      // Reset buffer
      scf.for %arg6 = %c0 to %c4096 step %c1 {
        scf.for %arg7 = %c0 to %c4096 step %c1 {
          memref.store %cst_0, %alloc[%arg6, %arg7] : memref<?x?xf32>
        }
      }

      // Compute P = Q*K^T
      scf.for %arg6 = %c0 to %c4096 step %c1 {
        scf.for %arg7 = %c0 to %c4096 step %c1 {
          %0 = scf.for %arg8 = %c0 to %c64 step %c1 iter_args(%arg9 = %cst_0) -> (f32) {
            %2 = memref.load %arg0[%arg5, %arg6, %arg8] : memref<16x4096x64xf16>
            %3 = memref.load %arg1[%arg5, %arg7, %arg8] : memref<16x4096x64xf16>
            %4 = arith.extf %2 : f16 to f32
            %5 = arith.extf %3 : f16 to f32
            %6 = arith.mulf %4, %5 : f32
            %7 = arith.addf %arg9, %6 : f32
            scf.yield %7 : f32
          }
          // Scale by sm_scale
          %1 = arith.mulf %0, %arg4 : f32
          memref.store %1, %alloc[%arg6, %arg7] : memref<?x?xf32>
        }
      }

      // Compute softmax
      scf.for %arg6 = %c0 to %c4096 step %c1 {
        // Max reduce
        %0 = scf.for %arg7 = %c0 to %c4096 step %c1 iter_args(%arg8 = %cst) -> (f32) {
          %2 = memref.load %alloc[%arg6, %arg7] : memref<?x?xf32>
          %3 = arith.maximumf %arg8, %2 : f32
          scf.yield %3 : f32
        }
        // Center by max and exp
        scf.for %arg7 = %c0 to %c4096 step %c1 {
          %2 = memref.load %alloc[%arg6, %arg7] : memref<?x?xf32>
          %3 = arith.subf %2, %0 : f32
          %4 = math.exp%3 : f32
          memref.store %4, %alloc[%arg6, %arg7] : memref<?x?xf32>
        }
        // Take sum of row
        %1 = scf.for %arg7 = %c0 to %c4096 step %c1 iter_args(%arg8 = %cst_0) -> (f32) {
          %2 = memref.load %alloc[%arg6, %arg7] : memref<?x?xf32>
          %3 = arith.addf %arg8, %2 : f32
          scf.yield %3 : f32
        }
        // Divide by sum
        scf.for %arg7 = %c0 to %c4096 step %c1 {
          %2 = memref.load %alloc[%arg6, %arg7] : memref<?x?xf32>
          %3 = arith.divf %2, %1 : f32
          memref.store %3, %alloc[%arg6, %arg7] : memref<?x?xf32>
        }
      }

      // Compute P*V
      scf.for %arg6 = %c0 to %c4096 step %c1 {
        scf.for %arg7 = %c0 to %c64 step %c1 {
          %0 = scf.for %arg8 = %c0 to %c4096 step %c1 iter_args(%arg9 = %cst_0) -> (f32) {
            %2 = memref.load %alloc[%arg6, %arg8] : memref<?x?xf32>
            %3 = memref.load %arg2[%arg5, %arg8, %arg7] : memref<16x4096x64xf16>
            %4 = arith.extf %3 : f16 to f32
            %5 = arith.mulf %2, %4 : f32
            %6 = arith.addf %5, %arg9 : f32
            scf.yield %6 : f32
          }
          %1 = arith.truncf %0 : f32 to f16
          memref.store %1, %arg3[%arg5, %arg6, %arg7] : memref<16x4096x64xf16>
        }
      }
    }
    memref.dealloc %alloc : memref<?x?xf32>
    return %arg3 : memref<16x4096x64xf16>
  }
  func.func @init_3d_memref_to_const_f16(%arg0: memref<16x4096x64xf16>, %arg1: f16) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    scf.for %arg2 = %c0 to %c16 step %c1 {
      scf.for %arg3 = %c0 to %c4096 step %c1 {
        scf.for %arg4 = %c0 to %c64 step %c1 {
          memref.store %arg1, %arg0[%arg2, %arg3, %arg4] : memref<16x4096x64xf16>
        }
      }
    }
    return
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 0.350097656 : f32 // Match the hardcoded scale in the kernel

    // Random number generator parameters
    %cst_1 = arith.constant -1.000000e+00 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %false = arith.constant false

    // Allocate Q, K, V, O buffers
    %alloc = memref.alloc() : memref<16x4096x64xf16>
    %alloc_3 = memref.alloc() : memref<16x4096x64xf16>
    %alloc_4 = memref.alloc() : memref<16x4096x64xf16>
    %alloc_5 = memref.alloc() : memref<16x4096x64xf16>
    %alloc_6 = memref.alloc() : memref<16x4096x64xf16>
    %alloc_7 = memref.alloc() : memref<16x4096x64xf32>

    // Initialize Q, K, V with random numbers in range [-1, 1]
    %cast = memref.cast %alloc : memref<16x4096x64xf16> to memref<*xf16>
    %cast_8 = memref.cast %alloc_3 : memref<16x4096x64xf16> to memref<*xf16>
    %cast_9 = memref.cast %alloc_4 : memref<16x4096x64xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%cast, %cst_1, %cst_2, %false) : (memref<*xf16>, f32, f32, i1) -> ()
    call @fillResource1DRandomF16(%cast_8, %cst_1, %cst_2, %false) : (memref<*xf16>, f32, f32, i1) -> ()
    call @fillResource1DRandomF16(%cast_9, %cst_1, %cst_2, %false) : (memref<*xf16>, f32, f32, i1) -> ()

    // Initialize output buffers to 0.0
    call @init_3d_memref_to_const_f16(%alloc_5, %cst) : (memref<16x4096x64xf16>, f16) -> ()
    call @init_3d_memref_to_const_f16(%alloc_6, %cst) : (memref<16x4096x64xf16>, f16) -> ()

    // Run GPU version
    %0 = call @gpu_impl(%alloc, %alloc_3, %alloc_4, %alloc_5) : (memref<16x4096x64xf16>, memref<16x4096x64xf16>, memref<16x4096x64xf16>, memref<16x4096x64xf16>) -> memref<16x4096x64xf16>

    // Run CPU reference version
    %1 = call @cpu_impl(%alloc, %alloc_3, %alloc_4, %alloc_6, %cst_0) : (memref<16x4096x64xf16>, memref<16x4096x64xf16>, memref<16x4096x64xf16>, memref<16x4096x64xf16>, f32) -> memref<16x4096x64xf16>

    // Convert CPU output to f32 for comparison
    scf.for %arg0 = %c0 to %c16 step %c1 {
      scf.for %arg1 = %c0 to %c4096 step %c1 {
        scf.for %arg2 = %c0 to %c64 step %c1 {
          %2 = memref.load %alloc_6[%arg0, %arg1, %arg2] : memref<16x4096x64xf16>
          %3 = arith.extf %2 : f16 to f32
          memref.store %3, %alloc_7[%arg0, %arg1, %arg2] : memref<16x4096x64xf32>
        }
      }
    }

    // Compare results
    %cast_10 = memref.cast %0 : memref<16x4096x64xf16> to memref<*xf16>
    %cast_11 = memref.cast %alloc_7 : memref<16x4096x64xf32> to memref<*xf32>
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF16(%cast_10, %cast_11) : (memref<*xf16>, memref<*xf32>) -> ()

    // Cleanup
    memref.dealloc %alloc : memref<16x4096x64xf16>
    memref.dealloc %alloc_3 : memref<16x4096x64xf16>
    memref.dealloc %alloc_4 : memref<16x4096x64xf16>
    memref.dealloc %alloc_5 : memref<16x4096x64xf16>
    memref.dealloc %alloc_6 : memref<16x4096x64xf16>
    memref.dealloc %alloc_7 : memref<16x4096x64xf32>
    return
  }

  // Helper function declarations
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF16(memref<*xf16>, f32) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF32(memref<*xf32>, f32) attributes {llvm.emit_c_interface}
}
