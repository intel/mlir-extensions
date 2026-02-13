// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime,spirv-backend -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

#q = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [8, 16]>
#k = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [16, 16]>
#v = #k
#kt = #xegpu.layout<sg_layout = [1, 8], sg_data = [64, 16], inst_data = [16, 16]>
#k_prefetch = #xegpu.layout<sg_layout = [4, 2], sg_data = [16, 32], inst_data = [16, 16]>
#v_prefetch = #k_prefetch
#out = #q
#layout_128x1 = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 1], inst_data = [8, 1]>
#layout_128x16 = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 16], inst_data = [8, 16]>
#layout_128 = #xegpu.layout<sg_layout = [8], sg_data = [16], inst_data = [8]>
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
      %c16 = arith.constant 16 : index

      %start_m = gpu.block_id x
      %off_hz = gpu.block_id y

      // Memref sizes in x dim
      %size_x_t0 = arith.muli %Z, %H : index
      %size_x = arith.muli %size_x_t0, %N_CTX : index

      // Calculate the WG x offset of the q tile. This is equal to off_hz * N_CTX + start_m * BLOCK_M
      %wg_x_offset = arith.muli %off_hz, %N_CTX : index
      %offset_m = arith.muli %start_m, %BLOCK_M : index
      %wg_q_x_offset = arith.addi %wg_x_offset, %offset_m : index

      // Init Q tile. Each WG must load 128x64xf16 tile of Q
      %q_ptr = memref.extract_aligned_pointer_as_index %Q : memref<?x?xf16> -> index
      %q_ptr_i64 = arith.index_cast %q_ptr : index to i64
      %q_tile  = xegpu.create_nd_tdesc %q_ptr_i64, shape: [%size_x, %BLOCK_DMODEL], strides: [%BLOCK_DMODEL, %c1] : i64 -> !xegpu.tensor_desc<128x64xf16, #q>

      // Init K tile. Each WG must load 64x64xf16 tile of K per iteration of inner loop. However, we block this as 16x64xf16 slices to
      // better utilize registers.
      %k_ptr = memref.extract_aligned_pointer_as_index %K : memref<?x?xf16> -> index
      %k_ptr_i64 = arith.index_cast %k_ptr : index to i64
      %k_tile_slice = xegpu.create_nd_tdesc %k_ptr_i64, shape: [%size_x, %BLOCK_DMODEL], strides: [%BLOCK_DMODEL, %c1] : i64 -> !xegpu.tensor_desc<16x64xf16, #k>

      // Init V tile. Same as K tile.
      %v_ptr = memref.extract_aligned_pointer_as_index %V : memref<?x?xf16> -> index
      %v_ptr_i64 = arith.index_cast %v_ptr : index to i64
      %v_tile_slice = xegpu.create_nd_tdesc %v_ptr_i64, shape: [%size_x, %BLOCK_DMODEL], strides: [%BLOCK_DMODEL, %c1] : i64 -> !xegpu.tensor_desc<16x64xf16, #v>

      // K prefetch. Each WG must prefetch 64x64xf16 tile of K per iteration of inner loop.
      // For prefetch SG layout is 4x2. Each SG prefetch 16x32xf16 tile.
      // Note that prefetch x offset is same as Q x offset. This is because WGs in same batch colloborate on K and V prefetch.
      %k_prefetch_tile = xegpu.create_nd_tdesc %K , shape: [%size_x, %BLOCK_DMODEL], strides: [%BLOCK_DMODEL, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<64x64xf16, #k_prefetch>
      xegpu.prefetch_nd %k_prefetch_tile[%wg_q_x_offset, %c0]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #k_prefetch} : !xegpu.tensor_desc<64x64xf16, #k_prefetch>
      %wg_q_x_offset_plus_BLOCK_N = arith.addi %wg_q_x_offset, %BLOCK_N : index
      xegpu.prefetch_nd %k_prefetch_tile[%wg_q_x_offset_plus_BLOCK_N, %c0]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #k_prefetch} : !xegpu.tensor_desc<64x64xf16, #k_prefetch>
      %wg_q_x_offset_plus_2_BLOCK_N = arith.addi %wg_q_x_offset_plus_BLOCK_N, %BLOCK_N : index
      xegpu.prefetch_nd %k_prefetch_tile[%wg_q_x_offset_plus_2_BLOCK_N, %c0]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #k_prefetch} : !xegpu.tensor_desc<64x64xf16, #k_prefetch>

      // V prefetch is similar to K
      %v_prefetch_tile = xegpu.create_nd_tdesc %V , shape: [%size_x, %BLOCK_DMODEL], strides: [%BLOCK_DMODEL, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<64x64xf16, #v_prefetch>
      xegpu.prefetch_nd %v_prefetch_tile[%wg_q_x_offset, %c0]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #v_prefetch} : !xegpu.tensor_desc<64x64xf16, #v_prefetch>
      xegpu.prefetch_nd %v_prefetch_tile[%wg_q_x_offset_plus_BLOCK_N, %c0] {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #v_prefetch} : !xegpu.tensor_desc<64x64xf16, #v_prefetch>
      xegpu.prefetch_nd %v_prefetch_tile[%wg_q_x_offset_plus_2_BLOCK_N, %c0]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #v_prefetch} : !xegpu.tensor_desc<64x64xf16, #v_prefetch>
      %BLOCK_N_3_t = arith.addi %BLOCK_N, %BLOCK_N : index
      %BLOCK_N_3 = arith.addi %BLOCK_N_3_t, %BLOCK_N : index


      // Initialize m, l and acc
      %m_i_row_in = arith.constant {layout_result_0 = #layout_128} dense<0xFF800000> : vector<128xf32> // -inf
      %l_i_row_in = arith.constant {layout_result_0 = #layout_128} dense<1.0>  : vector<128xf32> // 1.0
      %zero_dpas_128x16 = arith.constant {layout_result_0 = #layout_128x16} dense<0.0>  : vector<128x16xf32>
      %zero_128x64 = arith.constant {layout_result_0 = #out} dense<0.0>  : vector<128x64xf32>
      %zero_128 = arith.constant {layout_result_0 = #layout_128} dense<0.000000e+00>  : vector<128xf32>
      %minus_inf_128 = arith.constant {layout_result_0 = #layout_128} dense<0xFF800000> : vector<128xf32> // -inf

      // Softmax scaling
      // FIXME: value 0.5 is hard coded. need to take it from %sm_scale
      %qk_scale_128 = arith.constant {layout_result_0 = #layout_128} dense<0.5> : vector<128xf32>
      %qk_scale_128x1 = arith.constant {layout_result_0 = #layout_128x1}  dense<0.5> : vector<128x1xf32>
      %qk_scale_128x16 = arith.constant {layout_result_0 = #layout_128x16} dense<0.5>  : vector<128x16xf32>


      // Load Q tile. Each WG loads 128x64xf16 tile of Q.
      %q_value = xegpu.load_nd %q_tile[%wg_q_x_offset, %c0] {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #q}  : !xegpu.tensor_desc<128x64xf16, #q> -> vector<128x64xf16>

      // Inner loop. This loop iterate over K and V tiles and update the accumulator by computing softmax(q*k^T)*v
      // K and V tiles are accessed in 64x64xf16 blocks (BLOCK_N=64). However, we load them in 16x64xf16 slices.
      %result:3 = scf.for %k = %c0 to %N_CTX step %BLOCK_N iter_args
        (
          %acc_in = %zero_128x64,
          %m_i_row = %m_i_row_in,
          %l_i_row = %l_i_row_in
          )
         -> (
          vector<128x64xf32>, vector<128xf32>, vector<128xf32>
         ) {
          gpu.barrier

          // K prefetch
          %prefetch_offset_x_running_t = arith.addi %BLOCK_N_3, %k : index
          %prefetch_offset_x_running = arith.addi %wg_q_x_offset, %prefetch_offset_x_running_t : index
          xegpu.prefetch_nd %k_prefetch_tile[%prefetch_offset_x_running, %c0]  {layout = #k_prefetch}: !xegpu.tensor_desc<64x64xf16, #k_prefetch>

          // V prefetch
          xegpu.prefetch_nd %v_prefetch_tile[%prefetch_offset_x_running, %c0]  {layout = #v_prefetch}: !xegpu.tensor_desc<64x64xf16, #v_prefetch>

          // Load first 16x64xf16 K slice. K is in column major layout, so we need to transpose after loading.
          %wg_x_offset_running = arith.addi %wg_x_offset, %k : index
          %k_value_slice_0_t0 = xegpu.load_nd %k_tile_slice[%wg_x_offset_running, %c0]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #k} : !xegpu.tensor_desc<16x64xf16, #k> -> vector<16x64xf16>
          %k_value_slice_0 = vector.transpose %k_value_slice_0_t0, [1, 0] {layout_result_0 = #kt} : vector<16x64xf16> to vector<64x16xf16>

          // Compute first 128x16 of Q * K^T using DPAS.
          %qk_out_0 = xegpu.dpas %q_value, %k_value_slice_0, %zero_dpas_128x16 {layout_a = #q, layout_b = #kt, layout_cd = #layout_128x16} : vector<128x64xf16>, vector<64x16xf16>, vector<128x16xf32> -> vector<128x16xf32>

          // Load second 16x64xf16 K slice.
          %wg_x_offset_running_plus_16 = arith.addi %wg_x_offset_running, %c16 : index
          %k_value_slice_1_t0 = xegpu.load_nd %k_tile_slice[%wg_x_offset_running_plus_16, %c0]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #k} : !xegpu.tensor_desc<16x64xf16, #k> -> vector<16x64xf16>
          %k_value_slice_1 = vector.transpose %k_value_slice_1_t0, [1, 0] {layout_result_0 = #kt} : vector<16x64xf16> to vector<64x16xf16>

          // Compute second 128x16 of Q * K^T using DPAS
          %qk_out_1 = xegpu.dpas %q_value, %k_value_slice_1, %zero_dpas_128x16 {layout_a = #q, layout_b = #kt, layout_cd = #layout_128x16} : vector<128x64xf16>, vector<64x16xf16>, vector<128x16xf32> -> vector<128x16xf32>

          // Load third  16x64xf16 K slice
          %wg_x_offset_running_plus_32 = arith.addi %wg_x_offset_running_plus_16, %c16 : index
          %k_value_slice_2_t0 = xegpu.load_nd %k_tile_slice[%wg_x_offset_running_plus_32, %c0]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #k} : !xegpu.tensor_desc<16x64xf16, #k> -> vector<16x64xf16>
          %k_value_slice_2 = vector.transpose %k_value_slice_2_t0, [1, 0] {layout_result_0 = #kt}  : vector<16x64xf16> to vector<64x16xf16>

          // Compute third 128x16 of Q * K^T using DPAS
          %qk_out_2 = xegpu.dpas %q_value, %k_value_slice_2, %zero_dpas_128x16 {layout_a = #q, layout_b = #kt, layout_cd = #layout_128x16}  : vector<128x64xf16>, vector<64x16xf16>, vector<128x16xf32> -> vector<128x16xf32>

          // Load forth  16x64 K slice
          %wg_x_offset_running_plus_48 = arith.addi %wg_x_offset_running_plus_32, %c16 : index
          %k_value_slice_3_t0 = xegpu.load_nd %k_tile_slice[%wg_x_offset_running_plus_48, %c0]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #k} : !xegpu.tensor_desc<16x64xf16, #k> -> vector<16x64xf16>
          %k_value_slice_3 = vector.transpose %k_value_slice_3_t0, [1, 0] {layout_result_0 = #kt}  : vector<16x64xf16> to vector<64x16xf16>

          // Compute forth 128x16 of Q * K^T using DPAS
          %qk_out_3 = xegpu.dpas %q_value, %k_value_slice_3, %zero_dpas_128x16 {layout_a = #q, layout_b = #kt, layout_cd = #layout_128x16} : vector<128x64xf16>, vector<64x16xf16>, vector<128x16xf32> -> vector<128x16xf32>

          // Softmax computation on QK_out tile
          // Do max reduction on qk_out
          %qk_out_max_t0 = arith.maximumf %qk_out_0, %qk_out_1 fastmath<fast> {layout_result_0 = #layout_128x16}  : vector<128x16xf32>
          %qk_out_max_t1 = arith.maximumf %qk_out_2, %qk_out_3 fastmath<fast> {layout_result_0 = #layout_128x16} : vector<128x16xf32>
          %qk_out_max_t2 = arith.maximumf %qk_out_max_t0, %qk_out_max_t1 fastmath<fast> {layout_result_0 = #layout_128x16} : vector<128x16xf32>
          %qk_out_max_t3 = vector.multi_reduction <maximumf>, %qk_out_max_t2, %minus_inf_128
            {layout_result_0 = #xegpu.slice<#layout_128x16, dims = [1]>}
            [1] : vector<128x16xf32> to vector<128xf32>
          // %qk_out_max = vector.shape_cast %qk_out_max_t3 {layout_result_0 = #layout_128x1} : vector<128xf32> to vector<128x1xf32>

          // Scale
          %qk_out_max_scaled = arith.mulf %qk_out_max_t3, %qk_scale_128 {layout_result_0 = #layout_128} : vector<128xf32>
          // Find m_ij_row
          %m_ij_row = arith.maximumf %qk_out_max_scaled, %m_i_row fastmath<fast> {layout_result_0 = #layout_128} : vector<128xf32>
          // Scale qk_out by qk_scale
          %qk_out_0_scaled = arith.mulf %qk_out_0, %qk_scale_128x16 {layout_result_0 = #layout_128x16} : vector<128x16xf32>
          %qk_out_1_scaled = arith.mulf %qk_out_1, %qk_scale_128x16 {layout_result_0 = #layout_128x16} : vector<128x16xf32>
          %qk_out_2_scaled = arith.mulf %qk_out_2, %qk_scale_128x16 {layout_result_0 = #layout_128x16} : vector<128x16xf32>
          %qk_out_3_scaled = arith.mulf %qk_out_3, %qk_scale_128x16 {layout_result_0 = #layout_128x16} : vector<128x16xf32>
          // Broadcast m_ij_row to 128x16
          %m_ij_row_broadcasted0 = vector.shape_cast %m_ij_row {layout_result_0 = #layout_128x1, layout_operand_0 = #xegpu.slice<#layout_128x1, dims=[1]>} : vector<128xf32> to vector<128x1xf32>
          %m_ij_row_broadcasted = vector.broadcast %m_ij_row_broadcasted0 {layout_result_0 = #layout_128x16} : vector<128x1xf32> to vector<128x16xf32>
          // Center qk_out by m_ij_row
          %qk_out_0_centered = arith.subf %qk_out_0_scaled, %m_ij_row_broadcasted {layout_result_0 = #layout_128x16} : vector<128x16xf32>
          %qk_out_1_centered = arith.subf %qk_out_1_scaled, %m_ij_row_broadcasted {layout_result_0 = #layout_128x16} : vector<128x16xf32>
          %qk_out_2_centered = arith.subf %qk_out_2_scaled, %m_ij_row_broadcasted {layout_result_0 = #layout_128x16} : vector<128x16xf32>
          %qk_out_3_centered = arith.subf %qk_out_3_scaled, %m_ij_row_broadcasted {layout_result_0 = #layout_128x16} : vector<128x16xf32>
          // Take exp
          %qk_out_0_exp = math.exp %qk_out_0_centered fastmath<fast> {layout_result_0 = #layout_128x16} : vector<128x16xf32>
          %qk_out_1_exp = math.exp %qk_out_1_centered fastmath<fast> {layout_result_0 = #layout_128x16} : vector<128x16xf32>
          %qk_out_2_exp = math.exp %qk_out_2_centered fastmath<fast> {layout_result_0 = #layout_128x16} : vector<128x16xf32>
          %qk_out_3_exp = math.exp %qk_out_3_centered fastmath<fast> {layout_result_0 = #layout_128x16} : vector<128x16xf32>
          // Do a sum reduction on exp output
          %l_ij_row_t0 = arith.addf %qk_out_0_exp, %qk_out_1_exp {layout_result_0 = #layout_128x16} : vector<128x16xf32>
          %l_ij_row_t1 = arith.addf %qk_out_2_exp, %qk_out_3_exp {layout_result_0 = #layout_128x16} : vector<128x16xf32>
          %l_ij_row_t2 = arith.addf %l_ij_row_t0, %l_ij_row_t1 {layout_result_0 = #layout_128x16} : vector<128x16xf32>
          %l_ij_row_t3 = vector.multi_reduction <add>, %l_ij_row_t2, %zero_128
            {layout_result_0 = #xegpu.slice<#layout_128x16, dims = [1]>}
            [1]  : vector<128x16xf32> to vector<128xf32>
          // %l_ij_row = vector.shape_cast %l_ij_row_t3 {layout_result_0 = #layout_128x1} : vector<128xf32> to vector<128x1xf32>
          // Compute alpha
          %alpha_row_t1 = arith.subf %m_i_row, %m_ij_row {layout_result_0 = #layout_128} : vector<128xf32>
          %alpha_row = math.exp %alpha_row_t1 fastmath<fast> {layout_result_0 = #layout_128} : vector<128xf32>
          // Update l_i
          %l_i_row_new_t1 = arith.mulf %l_i_row, %alpha_row {layout_result_0 = #layout_128} : vector<128xf32>
          %l_i_row_new = arith.addf %l_i_row_new_t1, %l_ij_row_t3 {layout_result_0 = #layout_128} : vector<128xf32>
          // Update acc
          %alpha_row_broadcasted0 = vector.shape_cast %alpha_row {layout_result_0 = #layout_128x1, layout_operand_0 = #xegpu.slice<#layout_128x1, dims=[1]>} : vector<128xf32> to vector<128x1xf32>
          %alpha_row_broadcasted = vector.broadcast %alpha_row_broadcasted0 {layout_result_0 = #out} : vector<128x1xf32> to vector<128x64xf32>
          %acc_in_updated = arith.mulf %acc_in, %alpha_row_broadcasted {layout_result_0 = #out} : vector<128x64xf32>

          // Convert qk_out_tile to DPAS-A precision for P*V computation.
          %qk_out_0_f16 = arith.truncf %qk_out_0_exp {layout_result_0 = #layout_128x16} : vector<128x16xf32> to vector<128x16xf16>
          %qk_out_1_f16 = arith.truncf %qk_out_1_exp {layout_result_0 = #layout_128x16} : vector<128x16xf32> to vector<128x16xf16>
          %qk_out_2_f16 = arith.truncf %qk_out_2_exp {layout_result_0 = #layout_128x16} : vector<128x16xf32> to vector<128x16xf16>
          %qk_out_3_f16 = arith.truncf %qk_out_3_exp {layout_result_0 = #layout_128x16} : vector<128x16xf32> to vector<128x16xf16>

          // Load first 16x64 V slice.
          %v_val_slice_0 = xegpu.load_nd %v_tile_slice[%wg_x_offset_running, %c0] {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #v} : !xegpu.tensor_desc<16x64xf16, #v> -> vector<16x64xf16>
          // Compute first iteration update of 128x64 of P * V
          %pv_out_iter0 = xegpu.dpas %qk_out_0_f16, %v_val_slice_0, %acc_in_updated {layout_a = #q, layout_b = #v, layout_cd = #out} : vector<128x16xf16>, vector<16x64xf16>, vector<128x64xf32> -> vector<128x64xf32>

          // Load second 16x64 V slice.
          %v_val_slice_1 = xegpu.load_nd %v_tile_slice[%wg_x_offset_running_plus_16, %c0] { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #v} : !xegpu.tensor_desc<16x64xf16, #v> -> vector<16x64xf16>
          // Compute second iteration update of 128x64 of P * V
          %pv_out_iter1 = xegpu.dpas %qk_out_1_f16, %v_val_slice_1, %pv_out_iter0 {layout_a = #q, layout_b = #v, layout_cd = #out} : vector<128x16xf16>, vector<16x64xf16>, vector<128x64xf32> -> vector<128x64xf32>

          // Load third 16x64 V slice.
          %v_val_slice_2 = xegpu.load_nd %v_tile_slice[%wg_x_offset_running_plus_32, %c0]  { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #v} : !xegpu.tensor_desc<16x64xf16, #v> -> vector<16x64xf16>
          // Compute third iteration update of 128x64 of P * V
          %pv_out_iter2 = xegpu.dpas %qk_out_2_f16, %v_val_slice_2, %pv_out_iter1 {layout_a = #q, layout_b = #v, layout_cd = #out} : vector<128x16xf16>, vector<16x64xf16>, vector<128x64xf32> -> vector<128x64xf32>

          // Load forth 16x64 V slice.
          %v_val_slice_3 = xegpu.load_nd %v_tile_slice[%wg_x_offset_running_plus_48, %c0]  { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #v} : !xegpu.tensor_desc<16x64xf16, #v> -> vector<16x64xf16>
          // Compute forth iteration update of 128x64 of P * V
          %pv_out_iter3 = xegpu.dpas %qk_out_3_f16, %v_val_slice_3, %pv_out_iter2 {layout_a = #q, layout_b = #v, layout_cd = #out} : vector<128x16xf16>, vector<16x64xf16>, vector<128x64xf32> -> vector<128x64xf32>

          scf.yield %pv_out_iter3, %m_ij_row, %l_i_row_new : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
        } {layout_result_0 = #out, layout_result_1 = #layout_128, layout_result_2 = #layout_128}// end of inner loop
      // Divide acc output by l_i
      %l_i_row_broadcast0 = vector.shape_cast %result#2 {layout_result_0 = #layout_128x1, layout_operand_0 = #xegpu.slice<#layout_128x1, dims=[0]>} : vector<128xf32> to vector<128x1xf32>
      %l_i_row_broadcast = vector.broadcast %l_i_row_broadcast0 {layout_result_0 = #out} : vector<128x1xf32> to vector<128x64xf32>
      %o_val_final_t = arith.divf %result#0, %l_i_row_broadcast {layout_result_0 = #out} : vector<128x64xf32>
      // Store output tile.
      %o_val_final = arith.truncf %o_val_final_t {layout_result_0 = #out} : vector<128x64xf32> to vector<128x64xf16>
      %Out_ptr = memref.extract_aligned_pointer_as_index %Out : memref<?x?xf16> -> index
      %Out_ptr_i64 = arith.index_cast %Out_ptr : index to i64
      %o_tile  = xegpu.create_nd_tdesc %Out_ptr_i64, shape: [%size_x, %BLOCK_DMODEL], strides: [%BLOCK_DMODEL, %c1] : i64 -> !xegpu.tensor_desc<128x64xf16, #out>
      xegpu.store_nd %o_val_final, %o_tile[%wg_q_x_offset, %c0]  {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<128x64xf16>, !xegpu.tensor_desc<128x64xf16, #out>

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
