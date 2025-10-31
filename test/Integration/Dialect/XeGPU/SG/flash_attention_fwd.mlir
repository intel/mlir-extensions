// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime,spirv-backend -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck
module @flash_attention attributes {gpu.container_module} {
  gpu.module @flash_attention_fwd {
    gpu.func @flash_attention_fwd(
      %Q : memref<?x?xf16>,
      %K : memref<?x?xf16>,
      %V : memref<?x?xf16>,
      %Out : memref<?x?xf16>,
      %sm_scale : f32,
      %stride_qz : index, %stride_qh : index, %stride_qm : index, %stride_qk : index,
      %stride_kz : index, %stride_kh : index, %stride_kn : index, %stride_kk : index,
      %stride_vz : index, %stride_vh : index, %stride_vk : index, %stride_vn : index,
      %stride_oz : index, %stride_oh : index, %stride_om : index, %stride_on : index,
      %Z : index, %H : index,
      %N_CTX : index,
      %BLOCK_M : index,
      %BLOCK_DMODEL : index,
      %BLOCK_N : index
      ) kernel attributes {intel_reqd_sub_group_size = 16 : i32} {

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c24 = arith.constant 24 : index
      %c32 = arith.constant 32 : index
      %c48 = arith.constant 48 : index
      %c64 = arith.constant 64 : index
      %start_m = gpu.block_id x
      %off_hz = gpu.block_id y
      %sg_id = gpu.subgroup_id : index

      // Memref sizes in x dim
      %size_x_t0 = arith.muli %Z, %H : index
      %size_x = arith.muli %size_x_t0, %N_CTX : index

      // Calculate the WG x offset of the q tile. This is equal to off_hz * N_CTX + start_m * BLOCK_M
      %wg_x_offset = arith.muli %off_hz, %N_CTX : index
      %offset_m = arith.muli %start_m, %BLOCK_M : index
      %wg_q_x_offset = arith.addi %wg_x_offset, %offset_m : index

      // For k and v offsets are off_zh * N_CTX because inside the K loop we will consume N_CTX length
      // this is equal to wg_x_offset

      // Compute the SG x offset for the q tile.
      %sg_x_slice_size = arith.divui %BLOCK_M, %c8 : index
      %sg_q_x_offset_t0 = arith.muli %sg_id, %sg_x_slice_size : index
      %sg_q_x_offset = arith.addi %wg_q_x_offset, %sg_q_x_offset_t0 : index

      // Init Q tile
      %q_tile  = xegpu.create_nd_tdesc %Q, shape: [%size_x, %BLOCK_DMODEL], strides: [%BLOCK_DMODEL, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<16x16xf16>

      // Init K tile. K is transposed, HW only supports 32-bit transpose. So K tile is created in f32 dtype.
      %k_ptr_index = memref.extract_aligned_pointer_as_index %K : memref<?x?xf16> -> index
      %k_ptr_i64 = arith.index_cast %k_ptr_index : index to i64
      %BLOCK_MODEL_DIV_2 = arith.divui %BLOCK_DMODEL, %c2 : index
      %k_tile_slice = xegpu.create_nd_tdesc %k_ptr_i64, shape: [%size_x, %BLOCK_MODEL_DIV_2], strides: [%BLOCK_MODEL_DIV_2, %c1] : i64 -> !xegpu.tensor_desc<16x8xf32>

      // Init V tile.
      %v_tile_slice = xegpu.create_nd_tdesc %V, shape: [%size_x, %BLOCK_DMODEL], strides: [%BLOCK_DMODEL, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<16x16xf16>

      // K prefetch.
      // Prefetch 16x32 tiles in 4x2 layout to cover 64x64
      // x offset for prefetch is same as for q tiles. This means that WGs assigned to same batch also collaborate on prefetching
      // the K, V tiles.
      // NOTE: We also tried WGs prefetching from the begining of the K, V tiles but that did not work well because multiple
      // WGs compete to prefetch the same data.
      %sg_layout_x = arith.divui %sg_id, %c2 : index
      %sg_layout_y = arith.remui %sg_id, %c2 : index

      %prefetch_offset_x_t0 = arith.muli %sg_layout_x, %c16 : index
      %prefetch_offset_x = arith.addi %wg_q_x_offset, %prefetch_offset_x_t0 : index
      %prefetch_offset_y = arith.muli %sg_layout_y, %c32 : index

      %k_prefetch_tile = xegpu.create_nd_tdesc %K , shape: [%size_x, %BLOCK_DMODEL], strides: [%BLOCK_DMODEL, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
      xegpu.prefetch_nd %k_prefetch_tile[%prefetch_offset_x, %prefetch_offset_y]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
      %prefetch_offset_x_plus_BLOCK_N = arith.addi %prefetch_offset_x, %BLOCK_N : index
      xegpu.prefetch_nd %k_prefetch_tile[%prefetch_offset_x_plus_BLOCK_N, %prefetch_offset_y]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
      %prefetch_offset_x_plus_2_BLOCK_N = arith.addi %prefetch_offset_x_plus_BLOCK_N, %BLOCK_N : index
      xegpu.prefetch_nd %k_prefetch_tile[%prefetch_offset_x_plus_2_BLOCK_N, %prefetch_offset_y]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>

      // V prefetch is similar to K
      %v_prefetch_tile = xegpu.create_nd_tdesc %V , shape: [%size_x, %BLOCK_DMODEL], strides: [%BLOCK_DMODEL, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
      xegpu.prefetch_nd %v_prefetch_tile[%prefetch_offset_x, %prefetch_offset_y]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
      xegpu.prefetch_nd %v_prefetch_tile[%prefetch_offset_x_plus_BLOCK_N, %prefetch_offset_y]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
      xegpu.prefetch_nd %v_prefetch_tile[%prefetch_offset_x_plus_2_BLOCK_N, %prefetch_offset_y]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
      %BLOCK_N_3_t = arith.addi %BLOCK_N, %BLOCK_N : index
      %BLOCK_N_3 = arith.addi %BLOCK_N_3_t, %BLOCK_N : index


      // Initialize m, l and acc
      %m_i_row_0_in = arith.constant dense<0xFF800000> : vector<8x1xf32> // -inf
      %m_i_row_1_in = arith.constant dense<0xFF800000> : vector<8x1xf32> // -inf
      %l_i_row_0_in = arith.constant dense<1.0> : vector<8x1xf32> // 1.0
      %l_i_row_1_in = arith.constant dense<1.0> : vector<8x1xf32> // 1.0
      %zero_dpas = arith.constant dense<0.0> : vector<8x16xf32>
      %minus_inf = arith.constant dense<0xFF800000> : vector<8x1xf32> // -inf
      %zero_8 = arith.constant dense<0.000000e+00> : vector<8xf32>
      %minus_inf_8 = arith.constant dense<0xFF800000> : vector<8xf32> // -inf

      // Softmax scaling
      // FIXME: value 0.5 is hard coded. need to take it from %sm_scale
      %qk_scale_8x1 = arith.constant dense<0.5> : vector<8x1xf32>
      %qk_scale_8x16 = arith.constant dense<0.5> : vector<8x16xf32>


      // Load 4 Q 16x16 tiles.
      %q_block_value_0 = xegpu.load_nd %q_tile[%sg_q_x_offset, %c0] {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
      %q_block_value_1 = xegpu.load_nd %q_tile[%sg_q_x_offset, %c16] {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
      %q_block_value_2 = xegpu.load_nd %q_tile[%sg_q_x_offset, %c32] {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
      %q_block_value_3 = xegpu.load_nd %q_tile[%sg_q_x_offset, %c48] {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>

      %q_block_value_0_0 = vector.extract_strided_slice %q_block_value_0 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
      %q_block_value_1_0 = vector.extract_strided_slice %q_block_value_0 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
      %q_block_value_0_1 = vector.extract_strided_slice %q_block_value_1 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
      %q_block_value_1_1 = vector.extract_strided_slice %q_block_value_1 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
      %q_block_value_0_2 = vector.extract_strided_slice %q_block_value_2 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
      %q_block_value_1_2 = vector.extract_strided_slice %q_block_value_2 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
      %q_block_value_0_3 = vector.extract_strided_slice %q_block_value_3 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
      %q_block_value_1_3 = vector.extract_strided_slice %q_block_value_3 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>

      // Inner loop. This loop iterate over K and V tiles and update the accumulator by computing softmax(q*k^T)*v
      %result:12 = scf.for %k = %c0 to %N_CTX step %BLOCK_N iter_args
        (
          %acc_in_0_0 = %zero_dpas,
          %acc_in_0_1 = %zero_dpas,
          %acc_in_0_2 = %zero_dpas,
          %acc_in_0_3 = %zero_dpas,
          %acc_in_1_0 = %zero_dpas,
          %acc_in_1_1 = %zero_dpas,
          %acc_in_1_2 = %zero_dpas,
          %acc_in_1_3 = %zero_dpas,

          %m_i_row_0 = %m_i_row_0_in,
          %m_i_row_1 = %m_i_row_1_in,
          %l_i_row_0 = %l_i_row_0_in,
          %l_i_row_1 = %l_i_row_1_in
          )
         -> (
          vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>,
          vector<8x1xf32>, vector<8x1xf32>, vector<8x1xf32>, vector<8x1xf32>
         ) {
          gpu.barrier

          // K prefetch
          %prefetch_offset_x_running_t = arith.addi %BLOCK_N_3, %k : index
          %prefetch_offset_x_running = arith.addi %wg_q_x_offset, %prefetch_offset_x_running_t : index
          xegpu.prefetch_nd %k_prefetch_tile[%prefetch_offset_x_running, %prefetch_offset_y] : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>

          // V prefetch
          xegpu.prefetch_nd %v_prefetch_tile[%prefetch_offset_x_running, %prefetch_offset_y] : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>

          // Load first 16x64xf16 (i.e. 16x32xf32) K slice.
          %wg_x_offset_running = arith.addi %wg_x_offset, %k : index
          %k_value_slice_0_0_t0 = xegpu.load_nd %k_tile_slice[%wg_x_offset_running, %c0]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32> -> vector<16x8xf32>
          %k_value_slice_0_1_t0 = xegpu.load_nd %k_tile_slice[%wg_x_offset_running, %c8]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32> -> vector<16x8xf32>
          %k_value_slice_0_2_t0 = xegpu.load_nd %k_tile_slice[%wg_x_offset_running, %c16]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32> -> vector<16x8xf32>
          %k_value_slice_0_3_t0 = xegpu.load_nd %k_tile_slice[%wg_x_offset_running, %c24]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32> -> vector<16x8xf32>

          %k_value_slice_0_0_t1 = vector.bitcast %k_value_slice_0_0_t0 : vector<16x8xf32> to vector<16x16xf16>
          %k_value_slice_0_1_t1 = vector.bitcast %k_value_slice_0_1_t0 : vector<16x8xf32> to vector<16x16xf16>
          %k_value_slice_0_2_t1 = vector.bitcast %k_value_slice_0_2_t0 : vector<16x8xf32> to vector<16x16xf16>
          %k_value_slice_0_3_t1 = vector.bitcast %k_value_slice_0_3_t0 : vector<16x8xf32> to vector<16x16xf16>

          %k_value_slice_0_0 = vector.transpose %k_value_slice_0_0_t1, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
          %k_value_slice_0_1 = vector.transpose %k_value_slice_0_1_t1, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
          %k_value_slice_0_2 = vector.transpose %k_value_slice_0_2_t1, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
          %k_value_slice_0_3 = vector.transpose %k_value_slice_0_3_t1, [1, 0] : vector<16x16xf16> to vector<16x16xf16>

          // Compute first 16x16 of Q * K^T using DPAS
          %qk_out_0_0_t0 = xegpu.dpas %q_block_value_0_0, %k_value_slice_0_0, %zero_dpas : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_0_t0 = xegpu.dpas %q_block_value_1_0, %k_value_slice_0_0, %zero_dpas : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_0_t1 = xegpu.dpas %q_block_value_0_1, %k_value_slice_0_1, %qk_out_0_0_t0 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_0_t1 = xegpu.dpas %q_block_value_1_1, %k_value_slice_0_1, %qk_out_1_0_t0 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_0_t2 = xegpu.dpas %q_block_value_0_2, %k_value_slice_0_2, %qk_out_0_0_t1 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_0_t2 = xegpu.dpas %q_block_value_1_2, %k_value_slice_0_2, %qk_out_1_0_t1 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_0 = xegpu.dpas %q_block_value_0_3, %k_value_slice_0_3, %qk_out_0_0_t2 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_0 = xegpu.dpas %q_block_value_1_3, %k_value_slice_0_3, %qk_out_1_0_t2 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>


          // Load second 16x64xf16 K slice
          %wg_x_offset_running_plus_16 = arith.addi %wg_x_offset_running, %c16 : index
          %k_value_slice_1_0_t0 = xegpu.load_nd %k_tile_slice[%wg_x_offset_running_plus_16, %c0]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32> -> vector<16x8xf32>
          %k_value_slice_1_1_t0 = xegpu.load_nd %k_tile_slice[%wg_x_offset_running_plus_16, %c8]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32> -> vector<16x8xf32>
          %k_value_slice_1_2_t0 = xegpu.load_nd %k_tile_slice[%wg_x_offset_running_plus_16, %c16]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32> -> vector<16x8xf32>
          %k_value_slice_1_3_t0 = xegpu.load_nd %k_tile_slice[%wg_x_offset_running_plus_16, %c24]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32> -> vector<16x8xf32>

          %k_value_slice_1_0_t1 = vector.bitcast %k_value_slice_1_0_t0 : vector<16x8xf32> to vector<16x16xf16>
          %k_value_slice_1_1_t1 = vector.bitcast %k_value_slice_1_1_t0 : vector<16x8xf32> to vector<16x16xf16>
          %k_value_slice_1_2_t1 = vector.bitcast %k_value_slice_1_2_t0 : vector<16x8xf32> to vector<16x16xf16>
          %k_value_slice_1_3_t1 = vector.bitcast %k_value_slice_1_3_t0 : vector<16x8xf32> to vector<16x16xf16>

          %k_value_slice_1_0 = vector.transpose %k_value_slice_1_0_t1, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
          %k_value_slice_1_1 = vector.transpose %k_value_slice_1_1_t1, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
          %k_value_slice_1_2 = vector.transpose %k_value_slice_1_2_t1, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
          %k_value_slice_1_3 = vector.transpose %k_value_slice_1_3_t1, [1, 0] : vector<16x16xf16> to vector<16x16xf16>

          // Compute second 16x16 of Q * K^T using DPAS
          %qk_out_0_1_t0 = xegpu.dpas %q_block_value_0_0, %k_value_slice_1_0, %zero_dpas     : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_1_t1 = xegpu.dpas %q_block_value_0_1, %k_value_slice_1_1, %qk_out_0_1_t0 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_1_t2 = xegpu.dpas %q_block_value_0_2, %k_value_slice_1_2, %qk_out_0_1_t1 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_1 = xegpu.dpas %q_block_value_0_3, %k_value_slice_1_3, %qk_out_0_1_t2    : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>

          %qk_out_1_1_t0 = xegpu.dpas %q_block_value_1_0, %k_value_slice_1_0, %zero_dpas      : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_1_t1 = xegpu.dpas %q_block_value_1_1, %k_value_slice_1_1, %qk_out_1_1_t0  : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_1_t2 = xegpu.dpas %q_block_value_1_2, %k_value_slice_1_2, %qk_out_1_1_t1  : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_1 = xegpu.dpas %q_block_value_1_3, %k_value_slice_1_3, %qk_out_1_1_t2     : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>

          // Load third  16x64xf16 K slice
          %wg_x_offset_running_plus_32 = arith.addi %wg_x_offset_running_plus_16, %c16 : index
          %k_value_slice_2_0_t0 = xegpu.load_nd %k_tile_slice[%wg_x_offset_running_plus_32, %c0]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32> -> vector<16x8xf32>
          %k_value_slice_2_1_t0 = xegpu.load_nd %k_tile_slice[%wg_x_offset_running_plus_32, %c8]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32> -> vector<16x8xf32>
          %k_value_slice_2_2_t0 = xegpu.load_nd %k_tile_slice[%wg_x_offset_running_plus_32, %c16]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32> -> vector<16x8xf32>
          %k_value_slice_2_3_t0 = xegpu.load_nd %k_tile_slice[%wg_x_offset_running_plus_32, %c24]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32> -> vector<16x8xf32>

          %k_value_slice_2_0_t1 = vector.bitcast %k_value_slice_2_0_t0 : vector<16x8xf32> to vector<16x16xf16>
          %k_value_slice_2_1_t1 = vector.bitcast %k_value_slice_2_1_t0 : vector<16x8xf32> to vector<16x16xf16>
          %k_value_slice_2_2_t1 = vector.bitcast %k_value_slice_2_2_t0 : vector<16x8xf32> to vector<16x16xf16>
          %k_value_slice_2_3_t1 = vector.bitcast %k_value_slice_2_3_t0 : vector<16x8xf32> to vector<16x16xf16>

          %k_value_slice_2_0 = vector.transpose %k_value_slice_2_0_t1, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
          %k_value_slice_2_1 = vector.transpose %k_value_slice_2_1_t1, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
          %k_value_slice_2_2 = vector.transpose %k_value_slice_2_2_t1, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
          %k_value_slice_2_3 = vector.transpose %k_value_slice_2_3_t1, [1, 0] : vector<16x16xf16> to vector<16x16xf16>

          // Compute third 16x16 of Q * K^T using DPAS
          %qk_out_0_2_t0 = xegpu.dpas %q_block_value_0_0, %k_value_slice_2_0, %zero_dpas : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_2_t1 = xegpu.dpas %q_block_value_0_1, %k_value_slice_2_1, %qk_out_0_2_t0 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_2_t2 = xegpu.dpas %q_block_value_0_2, %k_value_slice_2_2, %qk_out_0_2_t1 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_2 = xegpu.dpas %q_block_value_0_3, %k_value_slice_2_3, %qk_out_0_2_t2 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>

          %qk_out_1_2_t0 = xegpu.dpas %q_block_value_1_0, %k_value_slice_2_0, %zero_dpas : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_2_t1 = xegpu.dpas %q_block_value_1_1, %k_value_slice_2_1, %qk_out_1_2_t0 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_2_t2 = xegpu.dpas %q_block_value_1_2, %k_value_slice_2_2, %qk_out_1_2_t1 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_2 = xegpu.dpas %q_block_value_1_3, %k_value_slice_2_3, %qk_out_1_2_t2 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>

          // Load forth  16x64 K slice
          %wg_x_offset_running_plus_48 = arith.addi %wg_x_offset_running_plus_32, %c16 : index
          %k_value_slice_3_0_t0 = xegpu.load_nd %k_tile_slice[%wg_x_offset_running_plus_48, %c0]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32> -> vector<16x8xf32>
          %k_value_slice_3_1_t0 = xegpu.load_nd %k_tile_slice[%wg_x_offset_running_plus_48, %c8]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32> -> vector<16x8xf32>
          %k_value_slice_3_2_t0 = xegpu.load_nd %k_tile_slice[%wg_x_offset_running_plus_48, %c16]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32> -> vector<16x8xf32>
          %k_value_slice_3_3_t0 = xegpu.load_nd %k_tile_slice[%wg_x_offset_running_plus_48, %c24]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32> -> vector<16x8xf32>

          %k_value_slice_3_0_t1 = vector.bitcast %k_value_slice_3_0_t0 : vector<16x8xf32> to vector<16x16xf16>
          %k_value_slice_3_1_t1 = vector.bitcast %k_value_slice_3_1_t0 : vector<16x8xf32> to vector<16x16xf16>
          %k_value_slice_3_2_t1 = vector.bitcast %k_value_slice_3_2_t0 : vector<16x8xf32> to vector<16x16xf16>
          %k_value_slice_3_3_t1 = vector.bitcast %k_value_slice_3_3_t0 : vector<16x8xf32> to vector<16x16xf16>

          %k_value_slice_3_0 = vector.transpose %k_value_slice_3_0_t1, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
          %k_value_slice_3_1 = vector.transpose %k_value_slice_3_1_t1, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
          %k_value_slice_3_2 = vector.transpose %k_value_slice_3_2_t1, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
          %k_value_slice_3_3 = vector.transpose %k_value_slice_3_3_t1, [1, 0] : vector<16x16xf16> to vector<16x16xf16>

          // Compute forth 16x16 of Q * K^T using DPAS
          %qk_out_0_3_t0 = xegpu.dpas %q_block_value_0_0, %k_value_slice_3_0, %zero_dpas : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_3_t1 = xegpu.dpas %q_block_value_0_1, %k_value_slice_3_1, %qk_out_0_3_t0 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_3_t2 = xegpu.dpas %q_block_value_0_2, %k_value_slice_3_2, %qk_out_0_3_t1 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_3 = xegpu.dpas %q_block_value_0_3, %k_value_slice_3_3, %qk_out_0_3_t2  : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>

          %qk_out_1_3_t0 = xegpu.dpas %q_block_value_1_0, %k_value_slice_3_0, %zero_dpas : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_3_t1 = xegpu.dpas %q_block_value_1_1, %k_value_slice_3_1, %qk_out_1_3_t0 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_3_t2 = xegpu.dpas %q_block_value_1_2, %k_value_slice_3_2, %qk_out_1_3_t1 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_3 = xegpu.dpas %q_block_value_1_3, %k_value_slice_3_3, %qk_out_1_3_t2 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>

          // Process row 0 of QK_out
          // Do max reduction on qk_out row 0
          %qk_out_max_0_t0 = arith.maximumf %qk_out_0_0, %qk_out_0_1 fastmath<fast> : vector<8x16xf32>
          %qk_out_max_0_t1 = arith.maximumf %qk_out_0_2, %qk_out_0_3 fastmath<fast> : vector<8x16xf32>
          %qk_out_max_0_t2 = arith.maximumf %qk_out_max_0_t0, %qk_out_max_0_t1 fastmath<fast> : vector<8x16xf32>
          %qk_out_max_0_t3 = vector.multi_reduction <maximumf>, %qk_out_max_0_t2, %minus_inf_8 [1] : vector<8x16xf32> to vector<8xf32>
          %qk_out_max_0 = vector.shape_cast %qk_out_max_0_t3 : vector<8xf32> to vector<8x1xf32>

          // Scale
          %qk_out_max_0_scaled = arith.mulf %qk_out_max_0, %qk_scale_8x1 : vector<8x1xf32>
          // Find m_ij_row_0
          %m_ij_row_0 = arith.maximumf %qk_out_max_0_scaled, %m_i_row_0 fastmath<fast> : vector<8x1xf32>
          // Scale qk row 0 by qk_scale
          %qk_out_0_0_scaled = arith.mulf %qk_out_0_0, %qk_scale_8x16 : vector<8x16xf32>
          %qk_out_0_1_scaled = arith.mulf %qk_out_0_1, %qk_scale_8x16 : vector<8x16xf32>
          %qk_out_0_2_scaled = arith.mulf %qk_out_0_2, %qk_scale_8x16 : vector<8x16xf32>
          %qk_out_0_3_scaled = arith.mulf %qk_out_0_3, %qk_scale_8x16 : vector<8x16xf32>
          // Broadcast m_ij_row_0 to 8x16
          %m_ij_row_0_broadcasted = vector.broadcast %m_ij_row_0 : vector<8x1xf32> to vector<8x16xf32>
          // Center qk_out by m_ij_row_0
          %qk_out_0_0_centered = arith.subf %qk_out_0_0_scaled, %m_ij_row_0_broadcasted : vector<8x16xf32>
          %qk_out_0_1_centered = arith.subf %qk_out_0_1_scaled, %m_ij_row_0_broadcasted : vector<8x16xf32>
          %qk_out_0_2_centered = arith.subf %qk_out_0_2_scaled, %m_ij_row_0_broadcasted : vector<8x16xf32>
          %qk_out_0_3_centered = arith.subf %qk_out_0_3_scaled, %m_ij_row_0_broadcasted : vector<8x16xf32>
          // Take exp
          %qk_out_0_0_exp = math.exp %qk_out_0_0_centered fastmath<fast> : vector<8x16xf32>
          %qk_out_0_1_exp = math.exp %qk_out_0_1_centered fastmath<fast> : vector<8x16xf32>
          %qk_out_0_2_exp = math.exp %qk_out_0_2_centered fastmath<fast> : vector<8x16xf32>
          %qk_out_0_3_exp = math.exp %qk_out_0_3_centered fastmath<fast> : vector<8x16xf32>
          // Do a sum reduction on exp output
          %l_ij_row_0_t0 = arith.addf %qk_out_0_0_exp, %qk_out_0_1_exp : vector<8x16xf32>
          %l_ij_row_0_t1 = arith.addf %qk_out_0_2_exp, %qk_out_0_3_exp : vector<8x16xf32>
          %l_ij_row_0_t2 = arith.addf %l_ij_row_0_t0, %l_ij_row_0_t1 : vector<8x16xf32>

          %l_ij_row_0_t3 = vector.multi_reduction <add>, %l_ij_row_0_t2, %zero_8 [1] : vector<8x16xf32> to vector<8xf32>
          %l_ij_row_0 = vector.shape_cast %l_ij_row_0_t3 : vector<8xf32> to vector<8x1xf32>
          // Compute alpha
          %alpha_row_0_t1 = arith.subf %m_i_row_0, %m_ij_row_0 : vector<8x1xf32>
          %alpha_row_0 = math.exp %alpha_row_0_t1 fastmath<fast>  : vector<8x1xf32>
          // Update l_i
          %l_i_row_0_new_t1 = arith.mulf %l_i_row_0, %alpha_row_0 : vector<8x1xf32>
          %l_i_row_0_new = arith.addf %l_i_row_0_new_t1, %l_ij_row_0 : vector<8x1xf32>
          // Update acc
          %alpha_row_0_broadcasted = vector.broadcast %alpha_row_0 : vector<8x1xf32> to vector<8x16xf32>
          %acc_in_0_0_updated = arith.mulf %acc_in_0_0, %alpha_row_0_broadcasted : vector<8x16xf32>
          %acc_in_0_1_updated = arith.mulf %acc_in_0_1, %alpha_row_0_broadcasted : vector<8x16xf32>
          %acc_in_0_2_updated = arith.mulf %acc_in_0_2, %alpha_row_0_broadcasted : vector<8x16xf32>
          %acc_in_0_3_updated = arith.mulf %acc_in_0_3, %alpha_row_0_broadcasted : vector<8x16xf32>

          // Process row 1 of QK_out
          // Do max reduction on qk_out row 1
          %qk_out_max_1_t0 = arith.maximumf %qk_out_1_0, %qk_out_1_1 fastmath<fast> : vector<8x16xf32>
          %qk_out_max_1_t1 = arith.maximumf %qk_out_1_2, %qk_out_1_3 fastmath<fast> : vector<8x16xf32>
          %qk_out_max_1_t2 = arith.maximumf %qk_out_max_1_t0, %qk_out_max_1_t1 fastmath<fast> : vector<8x16xf32>
          %qk_out_max_1_t3 = vector.multi_reduction <maximumf>, %qk_out_max_1_t2, %minus_inf_8 [1] : vector<8x16xf32> to vector<8xf32>
          %qk_out_max_1 = vector.shape_cast %qk_out_max_1_t3 : vector<8xf32> to vector<8x1xf32>
          // Scale
          %qk_out_max_1_scaled = arith.mulf %qk_out_max_1, %qk_scale_8x1 : vector<8x1xf32>
          // Find m_ij_row_0
          %m_ij_row_1 = arith.maximumf %qk_out_max_1_scaled, %m_i_row_1 fastmath<fast> : vector<8x1xf32>
          // Scale qk row 0 by qk_scale
          %qk_out_1_0_scaled = arith.mulf %qk_out_1_0, %qk_scale_8x16 : vector<8x16xf32>
          %qk_out_1_1_scaled = arith.mulf %qk_out_1_1, %qk_scale_8x16 : vector<8x16xf32>
          %qk_out_1_2_scaled = arith.mulf %qk_out_1_2, %qk_scale_8x16 : vector<8x16xf32>
          %qk_out_1_3_scaled = arith.mulf %qk_out_1_3, %qk_scale_8x16 : vector<8x16xf32>
          // Broadcast m_ij_row_0 to 8x16
          %m_ij_row_1_broadcasted = vector.broadcast %m_ij_row_1 : vector<8x1xf32> to vector<8x16xf32>
          // Center qk_out by m_ij_row_0
          %qk_out_1_0_centered = arith.subf %qk_out_1_0_scaled, %m_ij_row_1_broadcasted : vector<8x16xf32>
          %qk_out_1_1_centered = arith.subf %qk_out_1_1_scaled, %m_ij_row_1_broadcasted : vector<8x16xf32>
          %qk_out_1_2_centered = arith.subf %qk_out_1_2_scaled, %m_ij_row_1_broadcasted : vector<8x16xf32>
          %qk_out_1_3_centered = arith.subf %qk_out_1_3_scaled, %m_ij_row_1_broadcasted : vector<8x16xf32>
          // Take exp
          %qk_out_1_0_exp = math.exp %qk_out_1_0_centered fastmath<fast> : vector<8x16xf32>
          %qk_out_1_1_exp = math.exp %qk_out_1_1_centered fastmath<fast> : vector<8x16xf32>
          %qk_out_1_2_exp = math.exp %qk_out_1_2_centered fastmath<fast> : vector<8x16xf32>
          %qk_out_1_3_exp = math.exp %qk_out_1_3_centered fastmath<fast> : vector<8x16xf32>
          // Do a sum reduction on exp output
          %l_ij_row_1_t0 = arith.addf %qk_out_1_0_exp, %qk_out_1_1_exp : vector<8x16xf32>
          %l_ij_row_1_t1 = arith.addf %qk_out_1_2_exp, %qk_out_1_3_exp : vector<8x16xf32>
          %l_ij_row_1_t2 = arith.addf %l_ij_row_1_t0, %l_ij_row_1_t1 : vector<8x16xf32>
          %l_ij_row_1_t3 = vector.multi_reduction <add>, %l_ij_row_1_t2, %zero_8 [1] : vector<8x16xf32> to vector<8xf32>
          %l_ij_row_1 = vector.shape_cast %l_ij_row_1_t3 : vector<8xf32> to vector<8x1xf32>
          // Compute alpha
          %alpha_row_1_t1 = arith.subf %m_i_row_1, %m_ij_row_1 : vector<8x1xf32>
          %alpha_row_1 = math.exp %alpha_row_1_t1 fastmath<fast> : vector<8x1xf32>
          // Update l_i
          %l_i_row_1_new_t1 = arith.mulf %l_i_row_1, %alpha_row_1 : vector<8x1xf32>
          %l_i_row_1_new = arith.addf %l_i_row_1_new_t1, %l_ij_row_1 : vector<8x1xf32>
          // Update acc
          %alpha_row_1_broadcasted = vector.broadcast %alpha_row_1 : vector<8x1xf32> to vector<8x16xf32>
          %acc_in_1_0_updated = arith.mulf %acc_in_1_0, %alpha_row_1_broadcasted : vector<8x16xf32>
          %acc_in_1_1_updated = arith.mulf %acc_in_1_1, %alpha_row_1_broadcasted : vector<8x16xf32>
          %acc_in_1_2_updated = arith.mulf %acc_in_1_2, %alpha_row_1_broadcasted : vector<8x16xf32>
          %acc_in_1_3_updated = arith.mulf %acc_in_1_3, %alpha_row_1_broadcasted : vector<8x16xf32>

          // Convert qk_out_tile to A format for DPAS for p * v computation
          %qk_out_0_0_f16 = arith.truncf %qk_out_0_0_exp : vector<8x16xf32> to vector<8x16xf16>
          %qk_out_0_1_f16 = arith.truncf %qk_out_0_1_exp : vector<8x16xf32> to vector<8x16xf16>
          %qk_out_0_2_f16 = arith.truncf %qk_out_0_2_exp : vector<8x16xf32> to vector<8x16xf16>
          %qk_out_0_3_f16 = arith.truncf %qk_out_0_3_exp : vector<8x16xf32> to vector<8x16xf16>
          %qk_out_1_0_f16 = arith.truncf %qk_out_1_0_exp : vector<8x16xf32> to vector<8x16xf16>
          %qk_out_1_1_f16 = arith.truncf %qk_out_1_1_exp : vector<8x16xf32> to vector<8x16xf16>
          %qk_out_1_2_f16 = arith.truncf %qk_out_1_2_exp : vector<8x16xf32> to vector<8x16xf16>
          %qk_out_1_3_f16 = arith.truncf %qk_out_1_3_exp : vector<8x16xf32> to vector<8x16xf16>


          // Load first 16x64 V slices
          %v_val_slice_0_0 = xegpu.load_nd %v_tile_slice[%wg_x_offset_running, %c0] { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
          %v_val_slice_0_1 = xegpu.load_nd %v_tile_slice[%wg_x_offset_running, %c16] { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
          %v_val_slice_0_2 = xegpu.load_nd %v_tile_slice[%wg_x_offset_running, %c32] { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
          %v_val_slice_0_3 = xegpu.load_nd %v_tile_slice[%wg_x_offset_running, %c48] { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>

          // Compute first iteration update of 16x64 of P * V
          %pv_out_0_0_iter0 = xegpu.dpas %qk_out_0_0_f16, %v_val_slice_0_0, %acc_in_0_0_updated : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_0_iter0 = xegpu.dpas %qk_out_1_0_f16, %v_val_slice_0_0, %acc_in_1_0_updated : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_1_iter0 = xegpu.dpas %qk_out_0_0_f16, %v_val_slice_0_1, %acc_in_0_1_updated : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_1_iter0 = xegpu.dpas %qk_out_1_0_f16, %v_val_slice_0_1, %acc_in_1_1_updated : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_2_iter0 = xegpu.dpas %qk_out_0_0_f16, %v_val_slice_0_2, %acc_in_0_2_updated : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_2_iter0 = xegpu.dpas %qk_out_1_0_f16, %v_val_slice_0_2, %acc_in_1_2_updated : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_3_iter0 = xegpu.dpas %qk_out_0_0_f16, %v_val_slice_0_3, %acc_in_0_3_updated : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_3_iter0 = xegpu.dpas %qk_out_1_0_f16, %v_val_slice_0_3, %acc_in_1_3_updated : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>

          // Load second 16x64 V slices
          %v_val_slice_1_0 = xegpu.load_nd %v_tile_slice[%wg_x_offset_running_plus_16, %c0] { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
          %v_val_slice_1_1 = xegpu.load_nd %v_tile_slice[%wg_x_offset_running_plus_16, %c16] { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
          %v_val_slice_1_2 = xegpu.load_nd %v_tile_slice[%wg_x_offset_running_plus_16, %c32] { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
          %v_val_slice_1_3 = xegpu.load_nd %v_tile_slice[%wg_x_offset_running_plus_16, %c48] { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>

          // Compute second iteration update of 16x64 of P * V
          %pv_out_0_0_iter1 = xegpu.dpas %qk_out_0_1_f16, %v_val_slice_1_0, %pv_out_0_0_iter0 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_0_iter1 = xegpu.dpas %qk_out_1_1_f16, %v_val_slice_1_0, %pv_out_1_0_iter0 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_1_iter1 = xegpu.dpas %qk_out_0_1_f16, %v_val_slice_1_1, %pv_out_0_1_iter0 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_1_iter1 = xegpu.dpas %qk_out_1_1_f16, %v_val_slice_1_1, %pv_out_1_1_iter0 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_2_iter1 = xegpu.dpas %qk_out_0_1_f16, %v_val_slice_1_2, %pv_out_0_2_iter0 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_2_iter1 = xegpu.dpas %qk_out_1_1_f16, %v_val_slice_1_2, %pv_out_1_2_iter0 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_3_iter1 = xegpu.dpas %qk_out_0_1_f16, %v_val_slice_1_3, %pv_out_0_3_iter0 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_3_iter1 = xegpu.dpas %qk_out_1_1_f16, %v_val_slice_1_3, %pv_out_1_3_iter0 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>

          // Load third 16x64 V slices
          %v_val_slice_2_0 = xegpu.load_nd %v_tile_slice[%wg_x_offset_running_plus_32, %c0]  { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
          %v_val_slice_2_1 = xegpu.load_nd %v_tile_slice[%wg_x_offset_running_plus_32, %c16]  { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
          %v_val_slice_2_2 = xegpu.load_nd %v_tile_slice[%wg_x_offset_running_plus_32, %c32]  { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
          %v_val_slice_2_3 = xegpu.load_nd %v_tile_slice[%wg_x_offset_running_plus_32, %c48]  { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>

          // Compute third iteration update of 16x64 of P * V
          %pv_out_0_0_iter2 = xegpu.dpas %qk_out_0_2_f16, %v_val_slice_2_0, %pv_out_0_0_iter1 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_0_iter2 = xegpu.dpas %qk_out_1_2_f16, %v_val_slice_2_0, %pv_out_1_0_iter1 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_1_iter2 = xegpu.dpas %qk_out_0_2_f16, %v_val_slice_2_1, %pv_out_0_1_iter1 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_1_iter2 = xegpu.dpas %qk_out_1_2_f16, %v_val_slice_2_1, %pv_out_1_1_iter1 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_2_iter2 = xegpu.dpas %qk_out_0_2_f16, %v_val_slice_2_2, %pv_out_0_2_iter1 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_2_iter2 = xegpu.dpas %qk_out_1_2_f16, %v_val_slice_2_2, %pv_out_1_2_iter1 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_3_iter2 = xegpu.dpas %qk_out_0_2_f16, %v_val_slice_2_3, %pv_out_0_3_iter1 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_3_iter2 = xegpu.dpas %qk_out_1_2_f16, %v_val_slice_2_3, %pv_out_1_3_iter1 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>

          // Load forth 16x64 V slices
          %v_val_slice_3_0 = xegpu.load_nd %v_tile_slice[%wg_x_offset_running_plus_48, %c0]  { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
          %v_val_slice_3_1 = xegpu.load_nd %v_tile_slice[%wg_x_offset_running_plus_48, %c16]  { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
          %v_val_slice_3_2 = xegpu.load_nd %v_tile_slice[%wg_x_offset_running_plus_48, %c32]  { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
          %v_val_slice_3_3 = xegpu.load_nd %v_tile_slice[%wg_x_offset_running_plus_48, %c48]  { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>

          // Compute third iteration update of 16x64 of P * V
          %pv_out_0_0_iter3 = xegpu.dpas %qk_out_0_3_f16, %v_val_slice_3_0, %pv_out_0_0_iter2 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_0_iter3 = xegpu.dpas %qk_out_1_3_f16, %v_val_slice_3_0, %pv_out_1_0_iter2 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_1_iter3 = xegpu.dpas %qk_out_0_3_f16, %v_val_slice_3_1, %pv_out_0_1_iter2 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_1_iter3 = xegpu.dpas %qk_out_1_3_f16, %v_val_slice_3_1, %pv_out_1_1_iter2 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_2_iter3 = xegpu.dpas %qk_out_0_3_f16, %v_val_slice_3_2, %pv_out_0_2_iter2 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_2_iter3 = xegpu.dpas %qk_out_1_3_f16, %v_val_slice_3_2, %pv_out_1_2_iter2 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_3_iter3 = xegpu.dpas %qk_out_0_3_f16, %v_val_slice_3_3, %pv_out_0_3_iter2 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_3_iter3 = xegpu.dpas %qk_out_1_3_f16, %v_val_slice_3_3, %pv_out_1_3_iter2 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>

          scf.yield
                  %pv_out_0_0_iter3, %pv_out_0_1_iter3,  %pv_out_0_2_iter3,  %pv_out_0_3_iter3,
                  %pv_out_1_0_iter3, %pv_out_1_1_iter3,  %pv_out_1_2_iter3,  %pv_out_1_3_iter3,
                  %m_ij_row_0, %m_ij_row_1, %l_i_row_0_new, %l_i_row_1_new :
                  vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>,
                  vector<8x1xf32>, vector<8x1xf32>, vector<8x1xf32>, vector<8x1xf32>

      }
      // Divide acc output by l_i
      %l_i_row_0_broadcast = vector.broadcast %result#10 : vector<8x1xf32> to vector<8x16xf32>
      %l_i_row_1_broadcast = vector.broadcast %result#11 : vector<8x1xf32> to vector<8x16xf32>
      %o_val_final_0_0_t = arith.divf %result#0, %l_i_row_0_broadcast : vector<8x16xf32>
      %o_val_final_0_1_t = arith.divf %result#1, %l_i_row_0_broadcast : vector<8x16xf32>
      %o_val_final_0_2_t = arith.divf %result#2, %l_i_row_0_broadcast : vector<8x16xf32>
      %o_val_final_0_3_t = arith.divf %result#3, %l_i_row_0_broadcast : vector<8x16xf32>
      %o_val_final_1_0_t = arith.divf %result#4, %l_i_row_1_broadcast : vector<8x16xf32>
      %o_val_final_1_1_t = arith.divf %result#5, %l_i_row_1_broadcast : vector<8x16xf32>
      %o_val_final_1_2_t = arith.divf %result#6, %l_i_row_1_broadcast : vector<8x16xf32>
      %o_val_final_1_3_t = arith.divf %result#7, %l_i_row_1_broadcast : vector<8x16xf32>

      %o_val_final_0_0 = arith.truncf %o_val_final_0_0_t : vector<8x16xf32> to vector<8x16xf16>
      %o_val_final_0_1 = arith.truncf %o_val_final_0_1_t : vector<8x16xf32> to vector<8x16xf16>
      %o_val_final_0_2 = arith.truncf %o_val_final_0_2_t : vector<8x16xf32> to vector<8x16xf16>
      %o_val_final_0_3 = arith.truncf %o_val_final_0_3_t : vector<8x16xf32> to vector<8x16xf16>
      %o_val_final_1_0 = arith.truncf %o_val_final_1_0_t : vector<8x16xf32> to vector<8x16xf16>
      %o_val_final_1_1 = arith.truncf %o_val_final_1_1_t : vector<8x16xf32> to vector<8x16xf16>
      %o_val_final_1_2 = arith.truncf %o_val_final_1_2_t : vector<8x16xf32> to vector<8x16xf16>
      %o_val_final_1_3 = arith.truncf %o_val_final_1_3_t : vector<8x16xf32> to vector<8x16xf16>

      // FIXME: Output is stored in 8x16 shape even though HW can do 8x32 stores. This is due to limitation in vector
      // distribution. Inserting 8x16 into 8x32 is not supported in subgroup distribution.
      %o_tile  = xegpu.create_nd_tdesc %Out, shape: [%size_x, %BLOCK_DMODEL], strides: [%BLOCK_DMODEL, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<8x16xf16>

      xegpu.store_nd %o_val_final_0_0, %o_tile[%sg_q_x_offset, %c0]  {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %o_val_final_0_1, %o_tile[%sg_q_x_offset, %c16]  {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %o_val_final_0_2, %o_tile[%sg_q_x_offset, %c32]  {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %o_val_final_0_3, %o_tile[%sg_q_x_offset, %c48]  {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>

      %sg_q_x_offset_plus_8 = arith.addi %sg_q_x_offset, %c8 : index
      xegpu.store_nd %o_val_final_1_0, %o_tile[%sg_q_x_offset_plus_8, %c0]  {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %o_val_final_1_1, %o_tile[%sg_q_x_offset_plus_8, %c16]  {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %o_val_final_1_2, %o_tile[%sg_q_x_offset_plus_8, %c32]  {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %o_val_final_1_3, %o_tile[%sg_q_x_offset_plus_8, %c48]  {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>

      gpu.return
    }
  }

  func.func @gpu_impl(%q : memref<?x?xf16>, %k : memref<?x?xf16>, %v : memref<?x?xf16>,
    %o : memref<?x?xf16>, %Z : index, %H : index, %N_CTX : index, %D_HEAD : index,
    %sm_scale : f32) -> memref<?x?xf16> {

    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c128 = arith.constant 128 : index
    %c1_i64 = arith.constant 1 : i64

    %Z_H_N_t0 = arith.muli %Z, %H : index
    %Z_H_N = arith.muli %Z_H_N_t0, %N_CTX : index

    // %Z_i64 = index.castu %Z : index to i64
    // %H_i64 = index.castu %H : index to i64
    // %N_CTX_i64 = index.castu %N_CTX : index to i64
    // %D_HEAD_i64 = index.castu %D_HEAD : index to i64

    //strides
    %stride_1 = arith.muli %N_CTX, %D_HEAD : index
    %stride_2 = arith.muli %stride_1, %H : index

    %q_gpu = gpu.alloc (%Z_H_N, %D_HEAD) : memref<?x?xf16>
    %k_gpu = gpu.alloc (%Z_H_N, %D_HEAD) : memref<?x?xf16>
    %v_gpu = gpu.alloc (%Z_H_N, %D_HEAD) : memref<?x?xf16>
    %o_gpu = gpu.alloc (%Z_H_N, %D_HEAD) : memref<?x?xf16>
    // %m_gpu = gpu.alloc host_shared (%Z, %H, %N_CTX) : memref<?x?x?xf32>

    // copy from CPU to
    gpu.memcpy %q_gpu, %q : memref<?x?xf16>, memref<?x?xf16>
    gpu.memcpy %k_gpu, %k : memref<?x?xf16>, memref<?x?xf16>
    gpu.memcpy %v_gpu, %v : memref<?x?xf16>, memref<?x?xf16>
    gpu.memcpy %o_gpu, %o : memref<?x?xf16>, memref<?x?xf16>
    // memref.copy %m, %m_gpu : memref<?x?x?xf32> to memref<?x?x?xf32>

    // GPU params
    %BLOCK_M = arith.constant 128 : index
    %BLOCK_N = arith.constant 64 : index
    %N_CTX_i64 = index.castu %N_CTX : index to i64
    %BLOCK_M_i64 = index.castu %BLOCK_M : index to i64
    // do a ceiling div to figure out blocks_x
    // blocks_x = (N_CTX + BLOCKS_M - 1) / BLOCKS_M
    %blocks_x_t1 = arith.subi %BLOCK_M_i64, %c1_i64 : i64
    %blocks_x_t2 = arith.addi %N_CTX_i64, %blocks_x_t1 : i64
    %blocks_x_i64 = arith.divui %blocks_x_t2, %BLOCK_M_i64 : i64
    %blocks_x = index.castu %blocks_x_i64 : i64 to index
    %blocks_y = arith.muli %Z, %H : index
    // %blocks_x = arith.constant 32 : index

    // %BLOCK_M_i64 = index.castu %BLOCK_M : index to i64
    // %BLOCK_N_i64 = index.castu %BLOCK_N : index to i64

    // launch GPU func
    // Z (batch size) * H (num heads) is mapped to blocks_y. Each WG is doing a BLOCK_M * D_HEAD output tile.
    // Therefore, blocks_x is mapped to number of WGs needed to cover N_CTX dimension.
    // There are 8 SGs (128 work items). Each SG is computing (BLOCK_M / 8) * D_HEAD output tile.
    // Eg. For D_HEAD=64, BLOCK_M=64 case each SG computes 8x64 output tile.
    gpu.launch_func @flash_attention_fwd::@flash_attention_fwd blocks in (%blocks_x, %blocks_y, %c1)
      threads in (%c128, %c1, %c1) args(
      %q_gpu : memref<?x?xf16>, %k_gpu : memref<?x?xf16>, %v_gpu : memref<?x?xf16>, %o_gpu : memref<?x?xf16>,
      %sm_scale : f32,
      %stride_2 : index, %stride_1 : index, %D_HEAD : index, %c1 : index,
      %stride_2 : index, %stride_1 : index, %D_HEAD : index, %c1 : index,
      %stride_2 : index, %stride_1 : index, %D_HEAD : index, %c1 : index,
      %stride_2 : index, %stride_1 : index, %D_HEAD : index, %c1 : index,
      %Z : index, %H : index, %N_CTX : index, %BLOCK_M : index, %D_HEAD : index, %BLOCK_N : index
      )
    // wait for GPU to finish
    gpu.wait
    // copy output to CPU
    gpu.memcpy %o, %o_gpu : memref<?x?xf16>, memref<?x?xf16>

    gpu.dealloc %q_gpu : memref<?x?xf16>
    gpu.dealloc %k_gpu : memref<?x?xf16>
    gpu.dealloc %v_gpu : memref<?x?xf16>
    gpu.dealloc %o_gpu : memref<?x?xf16>
    // gpu.dealloc %m_gpu : memref<?x?x?xf32>

    return %o : memref<?x?xf16>
  }

  func.func @cpu_impl(%Q : memref<?x?xf16>, %K : memref<?x?xf16>, %V : memref<?x?xf16>,
    %o : memref<?x?xf16>, %Z : index, %H : index, %N_CTX : index, %D_HEAD : index,
    %sm_scale : f32) -> memref<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_f32 = arith.constant 0.0 : f32
    %Z_H = arith.muli %Z, %H : index
    %BLOCK_N = arith.constant 64 : index
    %log2e = arith.constant 1.442695040888963 : f32

    // buffer
    %qk_buffer = memref.alloc(%N_CTX, %N_CTX) : memref<?x?xf32>
    scf.for %zh = %c0 to %Z_H step %c1 {
      // reset memref
      scf.for %i = %c0 to %N_CTX step %c1 {
        scf.for %j = %c0 to %N_CTX step %c1 {
          memref.store %c0_f32, %qk_buffer[%i, %j] : memref<?x?xf32>
        }
      }
      %x_offset = arith.muli %N_CTX, %zh : index
      // compute p = q*k^T
      scf.for %i = %c0 to %N_CTX step %c1 {
        scf.for %j = %c0 to %N_CTX step %c1 {
          %qk_init = arith.constant 0.0 : f32
          %result = scf.for %k = %c0 to %D_HEAD step %c1 iter_args(%qk = %qk_init) -> f32 {
            %zh_i = arith.addi %i, %x_offset : index
            %zh_j = arith.addi %j, %x_offset : index
            %q_val = memref.load %Q [%zh_i, %k] : memref<?x?xf16>
            %k_val = memref.load %K [%zh_j, %k] : memref<?x?xf16>
            %q_val_f32 = arith.extf %q_val : f16 to f32
            %k_val_f32 = arith.extf %k_val : f16 to f32
            %t = arith.mulf %q_val_f32, %k_val_f32 : f32
            %t1 = arith.addf %qk, %t  : f32
            scf.yield %t1 : f32
          }
          %scaled = arith.mulf %result, %sm_scale : f32
          memref.store %scaled, %qk_buffer [%i, %j] : memref<?x?xf32>
        }
      }
      // compute the softmax
      scf.for %i = %c0 to %N_CTX step %c1 {
        %qk_row_max_in = arith.constant 0xFF800000 : f32
        // max reduce
        %qk_row_max = scf.for %j = %c0 to %N_CTX step %c1 iter_args(%curr = %qk_row_max_in) -> f32 {
          %qk_val = memref.load %qk_buffer [%i, %j] : memref<?x?xf32>
          %new_max = arith.maximumf %curr, %qk_val : f32
          scf.yield %new_max : f32
        }
        // center by max and exp
        scf.for %j = %c0 to %N_CTX step %c1 {
          %qk_val = memref.load %qk_buffer [%i, %j] : memref<?x?xf32>
          %t = arith.subf %qk_val, %qk_row_max : f32
          // scale by log2e to emulate exp2
          %t1 = arith.mulf %t, %log2e : f32
          %t2 = math.exp2 %t1 : f32
          memref.store %t2, %qk_buffer [%i, %j] : memref<?x?xf32>
        }
        // take sum of row
        %qk_row_sum_in = arith.constant 0.0 : f32
        %qk_row_sum = scf.for %j = %c0 to %N_CTX step %c1 iter_args(%curr = %qk_row_sum_in) -> f32 {
          %qk_val = memref.load %qk_buffer [%i, %j] : memref<?x?xf32>
          %sum_new = arith.addf %curr, %qk_val : f32
          scf.yield %sum_new : f32
        }
        // div by sum
        scf.for %j = %c0 to %N_CTX step %c1 {
          %qk_val = memref.load %qk_buffer [%i, %j] : memref<?x?xf32>
          %t = arith.divf %qk_val, %qk_row_sum : f32
          memref.store %t, %qk_buffer [%i, %j] : memref<?x?xf32>
        }
      }
      // compute p*v
      scf.for %i = %c0 to %N_CTX step %c1 {
        scf.for %j = %c0 to %D_HEAD step %c1 {
          %pv_init = arith.constant 0.0 : f32
          %result = scf.for %k = %c0 to %N_CTX step %c1 iter_args (%pv = %pv_init) -> f32 {
            %qk_val = memref.load %qk_buffer [%i, %k] : memref<?x?xf32>
            %qk_val_f16 = arith.truncf %qk_val : f32 to f16
            %zh_k = arith.addi %k, %x_offset : index
            %v_val = memref.load %V [%zh_k, %j] : memref<?x?xf16>
            %qk_val_f32 = arith.extf %qk_val_f16 : f16 to f32
            %v_val_f32 = arith.extf %v_val : f16 to f32
            %t = arith.mulf %qk_val_f32, %v_val_f32 : f32
            %t1 = arith.addf %t, %pv : f32
            scf.yield %t1 : f32
          }
          %zh_i = arith.addi %i, %x_offset : index
          %pv_f16 = arith.truncf %result : f32 to f16
          memref.store %pv_f16, %o [%zh_i, %j] : memref<?x?xf16>
        }
      }

    }

    memref.dealloc %qk_buffer : memref<?x?xf32>

    return %o : memref<?x?xf16>
  }


  func.func @init_2d_dynamic_memref_to_const_f16(%m : memref<?x?xf16>,
    %d0 : index, %d1 : index, %value : f16) -> () {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg0 = %c0 to %d0 step %c1 {
      scf.for %arg1 = %c0 to %d1 step %c1 {
        memref.store %value, %m [%arg0, %arg1] : memref<?x?xf16>
      }
    }
    return
  }

  func.func @main() attributes {llvm.emit_c_interface}  {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %magic = arith.constant 0.625 : f32
    %c0_f16 = arith.constant 0.0 : f16
    %c1_f32 = arith.constant 0.5 : f32
    %Z = arith.constant 2 : index // number of batches
    %H = arith.constant 2 : index // number of heads
    %N_CTX = arith.constant 4096 : index // sequence len
    %D_HEAD = arith.constant 64 : index // head dim
    %sm_scale = arith.constant 0.5 : f32 // softmax scale

    // random number generator params
    %rand_low = arith.constant -1.0 : f32
    %rand_high = arith.constant 1.0 : f32
    %gen_int = arith.constant 0 : i1

    // collapse the first 3 dims of the inputs to make it 2D.
    // Z x H x N_CTX x D_HEAD -> (Z * H * N_CTX) x D_HEAD
    %Z_H_N_t0 = arith.muli %Z, %H : index
    %Z_H_N = arith.muli %Z_H_N_t0, %N_CTX : index

    // allocate q, k, v, o
    %q = memref.alloc(%Z_H_N, %D_HEAD) : memref<?x?xf16>
    %k = memref.alloc(%Z_H_N, %D_HEAD) : memref<?x?xf16>
    %v = memref.alloc(%Z_H_N, %D_HEAD) : memref<?x?xf16>
    %o = memref.alloc(%Z_H_N, %D_HEAD) : memref<?x?xf16>
    %o_cpu = memref.alloc(%Z_H_N, %D_HEAD) : memref<?x?xf16>
    %o_cpu_f32 = memref.alloc(%Z_H_N, %D_HEAD) : memref<?x?xf32>
    // FIXME : m is unused for now
    // %m = memref.alloc(%Z, %H, %N_CTX) : memref<?x?x?xf32>

    // initialize q, k, v
    %q_random = memref.cast %q : memref<?x?xf16> to memref<*xf16>
    %k_random = memref.cast %k : memref<?x?xf16> to memref<*xf16>
    %v_random = memref.cast %v : memref<?x?xf16> to memref<*xf16>
    // Option 1: fill with random numbers
    // call @fillResource1DRandomF16(%q_random, %rand_low, %rand_high, %gen_int) : (memref<*xf16>, f32, f32, i1) -> ()
    // call @fillResource1DRandomF16(%k_random, %rand_low, %rand_high, %gen_int) : (memref<*xf16>, f32, f32, i1) -> ()
    // call @fillResource1DRandomF16(%v_random, %rand_low, %rand_high, %gen_int) : (memref<*xf16>, f32, f32, i1) -> ()
    // Option 2: fill with some magic constant for validation
    call @fillResource1DF16(%q_random, %magic) : (memref<*xf16>, f32) -> ()
    call @fillResource1DF16(%k_random, %magic) : (memref<*xf16>, f32) -> ()
    call @fillResource1DF16(%v_random, %magic) : (memref<*xf16>, f32) -> ()

    // // initialize output to 0.0
    // %o_random = memref.collapse_shape %o [[0, 1, 2, 3]] : memref<?x?x?x?xf16> into memref<?xf16>
    call @init_2d_dynamic_memref_to_const_f16(%o, %Z_H_N, %D_HEAD, %c0_f16)
      : (memref<?x?xf16>, index, index, f16) -> ()
    call @init_2d_dynamic_memref_to_const_f16(%o_cpu, %Z_H_N, %D_HEAD, %c0_f16)
      : (memref<?x?xf16>, index, index, f16) -> ()

    // initialize m to 1.0 (FIXME : masking is not used)
    // %c1_f32 = arith.constant 1.0 : f32
    // %m_random = memref.collapse_shape %m [[0, 1, 2]] : memref<?x?x?xf32> into memref<?xf32>
    // call @fillResource1DF32(%m_random, %c1_f32) : (memref<?xf32>, f32) -> ()

    // run fused version
    %out = call @gpu_impl( %q, %k, %v, %o, %Z, %H, %N_CTX, %D_HEAD, %sm_scale) :
      (memref<?x?xf16>, memref<?x?xf16>, memref<?x?xf16>, memref<?x?xf16>,
       index, index, index, index, f32) -> memref<?x?xf16>

    // run cpu version
    %out_cpu = call @cpu_impl( %q, %k, %v, %o_cpu, %Z, %H, %N_CTX, %D_HEAD, %sm_scale) :
      (memref<?x?xf16>, memref<?x?xf16>, memref<?x?xf16>, memref<?x?xf16>,
       index, index, index, index, f32) -> memref<?x?xf16>

    %out_cast = memref.cast %out : memref<?x?xf16> to memref<*xf16>
    %q_cast = memref.cast %q : memref<?x?xf16> to memref<*xf16>
    %out_cpu_cast = memref.cast %out_cpu : memref<?x?xf16> to memref<*xf16>
    // call @printMemrefF16(%q_cast) : (memref<*xf16>) -> ()
    // call @printMemrefF16(%out_cast) : (memref<*xf16>) -> ()
    // call @printMemrefF16(%out_cpu_cast) : (memref<*xf16>) -> ()
    // call @printMaxErrorF16(%out_cast, %out_cpu_cast) : (memref<*xf16>, memref<*xf16>) -> ()
    // sign extend CPU output to f32
    scf.for %i = %c0 to %Z_H_N step %c1 {
      scf.for %j = %c0 to %D_HEAD step %c1 {
        %o_val = memref.load %o_cpu [%i, %j] : memref<?x?xf16>
        %o_val_f32 = arith.extf %o_val : f16 to f32
        memref.store %o_val_f32, %o_cpu_f32 [%i, %j] : memref<?x?xf32>
      }
    }
    %out_cpu_f32_cast = memref.cast %o_cpu_f32 : memref<?x?xf32> to memref<*xf32>
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF16(%out_cast, %out_cpu_f32_cast) : (memref<*xf16>, memref<*xf32>) -> ()


    memref.dealloc %q : memref<?x?xf16>
    memref.dealloc %k : memref<?x?xf16>
    memref.dealloc %v : memref<?x?xf16>
    memref.dealloc %o : memref<?x?xf16>
    memref.dealloc %o_cpu : memref<?x?xf16>
    memref.dealloc %o_cpu_f32 : memref<?x?xf32>
    // memref.dealloc %m : memref<?x?x?xf32>

    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF16(memref<*xf16>, f32) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF32(memref<*xf32>, f32) attributes {llvm.emit_c_interface}
  func.func private @printMaxErrorF16(memref<*xf16>, memref<*xf16>) attributes {llvm.emit_c_interface}

}
