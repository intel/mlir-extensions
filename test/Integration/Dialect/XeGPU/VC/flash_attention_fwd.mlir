// RUN: IMEX_ENABLE_LARGE_REG_FILE=1 %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: IMEX_ENABLE_LARGE_REG_FILE=1 %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @flash_attention attributes {gpu.container_module} {
  gpu.module @flash_attention_fwd attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
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
      ) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %start_m = gpu.block_id x
      %off_hz = gpu.block_id y
      %sg_id = gpu.subgroup_id : index

      // memref sizes in x dim
      %size_x_t0 = arith.muli %Z, %H : index
      %size_x = arith.muli %size_x_t0, %N_CTX : index

      // calculate the WG x offset of the q tile. This is equal to off_hz * N_CTX + start_m * BLOCK_M
      %wg_x_offset = arith.muli %off_hz, %N_CTX : index
      %offset_m = arith.muli %start_m, %BLOCK_M : index
      %wg_q_x_offset = arith.addi %wg_x_offset, %offset_m : index

      // for k and v offsets are off_zh * N_CTX because inside the K loop we will consume N_CTX length
      // this is eqaul to wg_x_offset

      // compute the SG x offset for the q tile.
      // wg_q_offset + sg_x_slice_size * sg_id
      %sg_x_slice_size = arith.divui %BLOCK_M, %c8 : index
      %sg_q_x_offset_t0 = arith.muli %sg_id, %sg_x_slice_size : index
      %sg_q_x_offset = arith.addi %wg_q_x_offset, %sg_q_x_offset_t0 : index

      // init tile for 16x64 Q tiles
      %q_tile_init_0  = xegpu.create_nd_tdesc %Q[%sg_q_x_offset, %c0], shape: [%size_x, %BLOCK_DMODEL], strides: [%BLOCK_DMODEL, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<16x16xf16>
      %q_tile_init_1 = xegpu.update_nd_offset %q_tile_init_0, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %q_tile_init_2 = xegpu.update_nd_offset %q_tile_init_1, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %q_tile_init_3 = xegpu.update_nd_offset %q_tile_init_2, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>

      // init tile for 64x64 K tiles. We do this in 4 stages of 16x64 tiles to reduce register pressure.
      // k is reused by all SGs
      %k_tile_slice_0_0_init = xegpu.create_nd_tdesc %K [%wg_x_offset, %c0], shape: [%size_x, %BLOCK_DMODEL], strides: [%BLOCK_DMODEL, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<16x16xf16>
      %k_tile_slice_0_1_init = xegpu.update_nd_offset %k_tile_slice_0_0_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %k_tile_slice_0_2_init = xegpu.update_nd_offset %k_tile_slice_0_1_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %k_tile_slice_0_3_init = xegpu.update_nd_offset %k_tile_slice_0_2_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>

      %k_tile_slice_1_0_init = xegpu.update_nd_offset %k_tile_slice_0_0_init, [%c16, %c0] : !xegpu.tensor_desc<16x16xf16>
      %k_tile_slice_1_1_init = xegpu.update_nd_offset %k_tile_slice_1_0_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %k_tile_slice_1_2_init = xegpu.update_nd_offset %k_tile_slice_1_1_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %k_tile_slice_1_3_init = xegpu.update_nd_offset %k_tile_slice_1_2_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>

      %k_tile_slice_2_0_init = xegpu.update_nd_offset %k_tile_slice_1_0_init, [%c16, %c0] : !xegpu.tensor_desc<16x16xf16>
      %k_tile_slice_2_1_init = xegpu.update_nd_offset %k_tile_slice_2_0_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %k_tile_slice_2_2_init = xegpu.update_nd_offset %k_tile_slice_2_1_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %k_tile_slice_2_3_init = xegpu.update_nd_offset %k_tile_slice_2_2_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>

      %k_tile_slice_3_0_init = xegpu.update_nd_offset %k_tile_slice_2_0_init, [%c16, %c0] : !xegpu.tensor_desc<16x16xf16>
      %k_tile_slice_3_1_init = xegpu.update_nd_offset %k_tile_slice_3_0_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %k_tile_slice_3_2_init = xegpu.update_nd_offset %k_tile_slice_3_1_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %k_tile_slice_3_3_init = xegpu.update_nd_offset %k_tile_slice_3_2_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>

      // same for V tiles
      %v_tile_slice_0_0_init = xegpu.create_nd_tdesc %V [%wg_x_offset, %c0], shape: [%size_x, %BLOCK_DMODEL], strides: [%BLOCK_DMODEL, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<16x16xf16>
      %v_tile_slice_0_1_init = xegpu.update_nd_offset %v_tile_slice_0_0_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %v_tile_slice_0_2_init = xegpu.update_nd_offset %v_tile_slice_0_1_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %v_tile_slice_0_3_init = xegpu.update_nd_offset %v_tile_slice_0_2_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>

      %v_tile_slice_1_0_init = xegpu.update_nd_offset %v_tile_slice_0_0_init, [%c16, %c0] : !xegpu.tensor_desc<16x16xf16>
      %v_tile_slice_1_1_init = xegpu.update_nd_offset %v_tile_slice_1_0_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %v_tile_slice_1_2_init = xegpu.update_nd_offset %v_tile_slice_1_1_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %v_tile_slice_1_3_init = xegpu.update_nd_offset %v_tile_slice_1_2_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>

      %v_tile_slice_2_0_init = xegpu.update_nd_offset %v_tile_slice_1_0_init, [%c16, %c0] : !xegpu.tensor_desc<16x16xf16>
      %v_tile_slice_2_1_init = xegpu.update_nd_offset %v_tile_slice_2_0_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %v_tile_slice_2_2_init = xegpu.update_nd_offset %v_tile_slice_2_1_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %v_tile_slice_2_3_init = xegpu.update_nd_offset %v_tile_slice_2_2_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>

      %v_tile_slice_3_0_init = xegpu.update_nd_offset %v_tile_slice_2_0_init, [%c16, %c0] : !xegpu.tensor_desc<16x16xf16>
      %v_tile_slice_3_1_init = xegpu.update_nd_offset %v_tile_slice_3_0_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %v_tile_slice_3_2_init = xegpu.update_nd_offset %v_tile_slice_3_1_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %v_tile_slice_3_3_init = xegpu.update_nd_offset %v_tile_slice_3_2_init, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>

      // k preftech
      // prefetch 16x32 tiles in 4x2 layout to cover 64x64
      // x offset for prefetch is same as for q tiles. This means that WGs assigned to same bacth also colloborate on prefetching
      // the K, V tiles.
      // We also tried WGs prefetching from the begining of the K, V tiles but that did not work well because multiple
      // WGs compete to prefetch the same data.
      %c2 = arith.constant 2 : index
      %sg_layout_x = arith.divui %sg_id, %c2 : index
      %sg_layout_y = arith.remui %sg_id, %c2 : index

      %prefetch_offset_x_t0 = arith.muli %sg_layout_x, %c16 : index
      %prefetch_offset_x = arith.addi %wg_q_x_offset, %prefetch_offset_x_t0 : index
      %prefetch_offset_y = arith.muli %sg_layout_y, %c32 : index

      %k_prefetch_iter0 = xegpu.create_nd_tdesc %K [%prefetch_offset_x, %prefetch_offset_y], shape: [%size_x, %BLOCK_DMODEL], strides: [%BLOCK_DMODEL, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<16x32xf16>
      xegpu.prefetch_nd %k_prefetch_iter0  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x32xf16>
      %k_prefetch_iter1 = xegpu.update_nd_offset %k_prefetch_iter0, [%BLOCK_N, %c0] : !xegpu.tensor_desc<16x32xf16>
      xegpu.prefetch_nd %k_prefetch_iter1  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x32xf16>
      %k_prefetch_iter2 = xegpu.update_nd_offset %k_prefetch_iter1, [%BLOCK_N, %c0] : !xegpu.tensor_desc<16x32xf16>
      xegpu.prefetch_nd %k_prefetch_iter2  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x32xf16>
      %k_prefetch_iter3 = xegpu.update_nd_offset %k_prefetch_iter2, [%BLOCK_N, %c0] : !xegpu.tensor_desc<16x32xf16>

      // V prefetch is similar to K
      %v_prefetch_iter0 = xegpu.create_nd_tdesc %V [%prefetch_offset_x, %prefetch_offset_y], shape: [%size_x, %BLOCK_DMODEL], strides: [%BLOCK_DMODEL, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<16x32xf16>
      xegpu.prefetch_nd %v_prefetch_iter0  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x32xf16>
      %v_prefetch_iter1 = xegpu.update_nd_offset %v_prefetch_iter0, [%BLOCK_N, %c0]  : !xegpu.tensor_desc<16x32xf16>
      xegpu.prefetch_nd %v_prefetch_iter1  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x32xf16>
      %v_prefetch_iter2 = xegpu.update_nd_offset %v_prefetch_iter1, [%BLOCK_N, %c0] : !xegpu.tensor_desc<16x32xf16>
      xegpu.prefetch_nd %v_prefetch_iter2  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x32xf16>
      %v_prefetch_iter3 = xegpu.update_nd_offset %v_prefetch_iter2, [%BLOCK_N, %c0] : !xegpu.tensor_desc<16x32xf16>


      // initialize m, l and acc
      %m_i_row_0_in_flat = arith.constant dense<0xFF800000> : vector<8xf32> // -inf
      %m_i_row_1_in_flat = arith.constant dense<0xFF800000> : vector<8xf32> // -inf
      %l_i_row_0_in_flat = arith.constant dense<1.0> : vector<8xf32> // 1.0
      %l_i_row_1_in_flat = arith.constant dense<1.0> : vector<8xf32> // 1.0
      %m_i_row_0_in = vector.shape_cast %m_i_row_0_in_flat : vector<8xf32> to vector<8x1xf32>
      %m_i_row_1_in = vector.shape_cast %m_i_row_1_in_flat : vector<8xf32> to vector<8x1xf32>
      %l_i_row_0_in = vector.shape_cast %l_i_row_0_in_flat : vector<8xf32> to vector<8x1xf32>
      %l_i_row_1_in = vector.shape_cast %l_i_row_1_in_flat : vector<8xf32> to vector<8x1xf32>
      %zero = arith.constant dense<0.0> : vector<128xf32>
      %zero_dpas = vector.shape_cast %zero : vector<128xf32> to vector<8x16xf32>

      // softmax scaling
      // %qk_scale_8 = spirv.CompositeConstruct %sm_scale, %sm_scale, %sm_scale, %sm_scale, %sm_scale, %sm_scale, %sm_scale, %sm_scale : (f32, f32, f32, f32, f32, f32, f32, f32) -> vector<8xf32>
      // %qk_scale_16 = spirv.CompositeConstruct %sm_scale, %sm_scale, %sm_scale, %sm_scale, %sm_scale, %sm_scale, %sm_scale, %sm_scale,%sm_scale, %sm_scale, %sm_scale, %sm_scale,%sm_scale, %sm_scale, %sm_scale, %sm_scale : (f32, f32, f32, f32,f32, f32, f32, f32,f32, f32, f32, f32,f32, f32, f32, f32 ) -> vector<16xf32>
      // FIXME: value 0.5 is hard coded. need to take it from %sm_scale
      %qk_scale_8 = arith.constant dense<0.5> : vector<8xf32>
      %qk_scale_16 = arith.constant dense<0.5> : vector<16xf32>
      %qk_scale_8x1 = vector.shape_cast %qk_scale_8 : vector<8xf32> to vector<8x1xf32>
      %qk_scale_1x16 = vector.shape_cast %qk_scale_16 : vector<16xf32> to vector<1x16xf32>
      %qk_scale_8x16 = vector.shuffle %qk_scale_1x16, %qk_scale_1x16 [0, 0, 0, 0, 0, 0, 0, 0] : vector<1x16xf32>, vector<1x16xf32>


      // load Q tiles
      %q_block_value_0 = xegpu.load_nd %q_tile_init_0 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
      %q_block_value_1 = xegpu.load_nd %q_tile_init_1 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
      %q_block_value_2 = xegpu.load_nd %q_tile_init_2 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
      %q_block_value_3 = xegpu.load_nd %q_tile_init_3 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>

      %q_block_value_0_flat = vector.shape_cast %q_block_value_0 : vector<16x16xf16> to vector<256xf16>
      %q_block_value_1_flat = vector.shape_cast %q_block_value_1 : vector<16x16xf16> to vector<256xf16>
      %q_block_value_2_flat = vector.shape_cast %q_block_value_2 : vector<16x16xf16> to vector<256xf16>
      %q_block_value_3_flat = vector.shape_cast %q_block_value_3 : vector<16x16xf16> to vector<256xf16>

      %q_block_value_0_0_t0 = vector.extract_strided_slice %q_block_value_0_flat { offsets = [0], sizes = [128], strides = [1]} : vector<256xf16> to vector<128xf16>
      %q_block_value_0_0 = vector.shape_cast %q_block_value_0_0_t0 : vector<128xf16> to vector<8x16xf16>

      %q_block_value_1_0_t0 = vector.extract_strided_slice %q_block_value_0_flat { offsets = [128], sizes = [128], strides = [1]} : vector<256xf16> to vector<128xf16>
      %q_block_value_1_0 = vector.shape_cast %q_block_value_1_0_t0 : vector<128xf16> to vector<8x16xf16>

      %q_block_value_0_1_t0 = vector.extract_strided_slice %q_block_value_1_flat { offsets = [0], sizes = [128], strides = [1]} : vector<256xf16> to vector<128xf16>
      %q_block_value_0_1 = vector.shape_cast %q_block_value_0_1_t0 : vector<128xf16> to vector<8x16xf16>

      %q_block_value_1_1_t0 = vector.extract_strided_slice %q_block_value_1_flat { offsets = [128], sizes = [128], strides = [1]} : vector<256xf16> to vector<128xf16>
      %q_block_value_1_1 = vector.shape_cast %q_block_value_1_1_t0 : vector<128xf16> to vector<8x16xf16>

      // ----
      %q_block_value_0_2_t0 = vector.extract_strided_slice %q_block_value_2_flat { offsets = [0], sizes = [128], strides = [1]} : vector<256xf16> to vector<128xf16>
      %q_block_value_0_2 = vector.shape_cast %q_block_value_0_2_t0 : vector<128xf16> to vector<8x16xf16>

      %q_block_value_1_2_t0 = vector.extract_strided_slice %q_block_value_2_flat { offsets = [128], sizes = [128], strides = [1]} : vector<256xf16> to vector<128xf16>
      %q_block_value_1_2 = vector.shape_cast %q_block_value_1_2_t0 : vector<128xf16> to vector<8x16xf16>

      %q_block_value_0_3_t0 = vector.extract_strided_slice %q_block_value_3_flat { offsets = [0], sizes = [128], strides = [1]} : vector<256xf16> to vector<128xf16>
      %q_block_value_0_3 = vector.shape_cast %q_block_value_0_3_t0 : vector<128xf16> to vector<8x16xf16>

      %q_block_value_1_3_t0 = vector.extract_strided_slice %q_block_value_3_flat { offsets = [128], sizes = [128], strides = [1]} : vector<256xf16> to vector<128xf16>
      %q_block_value_1_3 = vector.shape_cast %q_block_value_1_3_t0 : vector<128xf16> to vector<8x16xf16>

      xegpu.alloc_nbarrier 16
      %nbarrier_id = arith.constant 1 : i8
      %num_threads = arith.constant 8 : i8
      %nbarrier = xegpu.init_nbarrier %nbarrier_id, %num_threads : i8, i8 -> !xegpu.nbarrier


      // inner loop. This loop iterate over K and V tiles and update the accumulator by computing softmax(q*k^T)*v
      %result:46 = scf.for %k = %c0 to %N_CTX step %BLOCK_N iter_args
        (
          %acc_in_0_0 = %zero_dpas,
          %acc_in_0_1 = %zero_dpas,
          %acc_in_0_2 = %zero_dpas,
          %acc_in_0_3 = %zero_dpas,
          %acc_in_1_0 = %zero_dpas,
          %acc_in_1_1 = %zero_dpas,
          %acc_in_1_2 = %zero_dpas,
          %acc_in_1_3 = %zero_dpas,

          %k_tile_slice_0_0 = %k_tile_slice_0_0_init,
          %k_tile_slice_0_1 = %k_tile_slice_0_1_init,
          %k_tile_slice_0_2 = %k_tile_slice_0_2_init,
          %k_tile_slice_0_3 = %k_tile_slice_0_3_init,
          %k_tile_slice_1_0 = %k_tile_slice_1_0_init,
          %k_tile_slice_1_1 = %k_tile_slice_1_1_init,
          %k_tile_slice_1_2 = %k_tile_slice_1_2_init,
          %k_tile_slice_1_3 = %k_tile_slice_1_3_init,
          %k_tile_slice_2_0 = %k_tile_slice_2_0_init,
          %k_tile_slice_2_1 = %k_tile_slice_2_1_init,
          %k_tile_slice_2_2 = %k_tile_slice_2_2_init,
          %k_tile_slice_2_3 = %k_tile_slice_2_3_init,
          %k_tile_slice_3_0 = %k_tile_slice_3_0_init,
          %k_tile_slice_3_1 = %k_tile_slice_3_1_init,
          %k_tile_slice_3_2 = %k_tile_slice_3_2_init,
          %k_tile_slice_3_3 = %k_tile_slice_3_3_init,

          %v_tile_slice_0_0 = %v_tile_slice_0_0_init,
          %v_tile_slice_0_1 = %v_tile_slice_0_1_init,
          %v_tile_slice_0_2 = %v_tile_slice_0_2_init,
          %v_tile_slice_0_3 = %v_tile_slice_0_3_init,
          %v_tile_slice_1_0 = %v_tile_slice_1_0_init,
          %v_tile_slice_1_1 = %v_tile_slice_1_1_init,
          %v_tile_slice_1_2 = %v_tile_slice_1_2_init,
          %v_tile_slice_1_3 = %v_tile_slice_1_3_init,
          %v_tile_slice_2_0 = %v_tile_slice_2_0_init,
          %v_tile_slice_2_1 = %v_tile_slice_2_1_init,
          %v_tile_slice_2_2 = %v_tile_slice_2_2_init,
          %v_tile_slice_2_3 = %v_tile_slice_2_3_init,
          %v_tile_slice_3_0 = %v_tile_slice_3_0_init,
          %v_tile_slice_3_1 = %v_tile_slice_3_1_init,
          %v_tile_slice_3_2 = %v_tile_slice_3_2_init,
          %v_tile_slice_3_3 = %v_tile_slice_3_3_init,

          /// prefetch
          %k_prefetch_tile = %k_prefetch_iter3,
          %v_prefetch_tile = %v_prefetch_iter3,

          %m_i_row_0 = %m_i_row_0_in,
          %m_i_row_1 = %m_i_row_1_in,
          %l_i_row_0 = %l_i_row_0_in,
          %l_i_row_1 = %l_i_row_1_in
          )
         -> (
          vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>,
         !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>,

         !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>,

         !xegpu.tensor_desc<16x32xf16>, !xegpu.tensor_desc<16x32xf16>,
         vector<8x1xf32>, vector<8x1xf32>, vector<8x1xf32>, vector<8x1xf32>

         ) {
          xegpu.nbarrier_arrive %nbarrier : !xegpu.nbarrier

          // k prefetch
          xegpu.prefetch_nd %k_prefetch_tile : !xegpu.tensor_desc<16x32xf16>
          %k_prefetch_tile_new = xegpu.update_nd_offset %k_prefetch_tile, [%BLOCK_N, %c0] : !xegpu.tensor_desc<16x32xf16>
          xegpu.compile_hint
          // V prefetch
          xegpu.prefetch_nd %v_prefetch_tile : !xegpu.tensor_desc<16x32xf16>
          %v_prefetch_tile_new = xegpu.update_nd_offset %v_prefetch_tile, [%BLOCK_N, %c0] : !xegpu.tensor_desc<16x32xf16>

          xegpu.compile_hint

          // load first 16x64 K slice
          %k_value_slice_0_0 = xegpu.load_nd %k_tile_slice_0_0  {transpose_bit_width = 32 : i32, transpose = array<i64: 1, 0>, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %k_value_slice_0_1 = xegpu.load_nd %k_tile_slice_0_1  {transpose_bit_width = 32 : i32, transpose = array<i64: 1, 0>, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %k_value_slice_0_2 = xegpu.load_nd %k_tile_slice_0_2  {transpose_bit_width = 32 : i32, transpose = array<i64: 1, 0>, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %k_value_slice_0_3 = xegpu.load_nd %k_tile_slice_0_3  {transpose_bit_width = 32 : i32, transpose = array<i64: 1, 0>, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          xegpu.compile_hint
          // update offsets
          %k_tile_slice_0_0_new = xegpu.update_nd_offset %k_tile_slice_0_0, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %k_tile_slice_0_1_new = xegpu.update_nd_offset %k_tile_slice_0_1, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %k_tile_slice_0_2_new = xegpu.update_nd_offset %k_tile_slice_0_2, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %k_tile_slice_0_3_new = xegpu.update_nd_offset %k_tile_slice_0_3, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          xegpu.compile_hint



          // compute first 16x16 of Q * K^T using DPAS
          %qk_out_0_0_t0 = xegpu.dpas %q_block_value_0_0, %k_value_slice_0_0, %zero_dpas : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_0_t0 = xegpu.dpas %q_block_value_1_0, %k_value_slice_0_0, %zero_dpas : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_0_t1 = xegpu.dpas %q_block_value_0_1, %k_value_slice_0_1, %qk_out_0_0_t0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_0_t1 = xegpu.dpas %q_block_value_1_1, %k_value_slice_0_1, %qk_out_1_0_t0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_0_t2 = xegpu.dpas %q_block_value_0_2, %k_value_slice_0_2, %qk_out_0_0_t1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_0_t2 = xegpu.dpas %q_block_value_1_2, %k_value_slice_0_2, %qk_out_1_0_t1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_0 = xegpu.dpas %q_block_value_0_3, %k_value_slice_0_3, %qk_out_0_0_t2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_0 = xegpu.dpas %q_block_value_1_3, %k_value_slice_0_3, %qk_out_1_0_t2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          xegpu.compile_hint

          // load second 16x64 K slice
          %k_value_slice_1_0 = xegpu.load_nd %k_tile_slice_1_0  {transpose_bit_width = 32 : i32, transpose = array<i64: 1, 0>, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %k_value_slice_1_1 = xegpu.load_nd %k_tile_slice_1_1  {transpose_bit_width = 32 : i32, transpose = array<i64: 1, 0>, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %k_value_slice_1_2 = xegpu.load_nd %k_tile_slice_1_2  {transpose_bit_width = 32 : i32, transpose = array<i64: 1, 0>, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %k_value_slice_1_3 = xegpu.load_nd %k_tile_slice_1_3  {transpose_bit_width = 32 : i32, transpose = array<i64: 1, 0>, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          xegpu.compile_hint
          // update offsets
          %k_tile_slice_1_0_new = xegpu.update_nd_offset %k_tile_slice_1_0, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %k_tile_slice_1_1_new = xegpu.update_nd_offset %k_tile_slice_1_1, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %k_tile_slice_1_2_new = xegpu.update_nd_offset %k_tile_slice_1_2, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %k_tile_slice_1_3_new = xegpu.update_nd_offset %k_tile_slice_1_3, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          xegpu.compile_hint

          // compute second 16x16 of Q * K^T using DPAS
          %qk_out_0_1_t0 = xegpu.dpas %q_block_value_0_0, %k_value_slice_1_0, %zero_dpas     : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_1_t1 = xegpu.dpas %q_block_value_0_1, %k_value_slice_1_1, %qk_out_0_1_t0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_1_t2 = xegpu.dpas %q_block_value_0_2, %k_value_slice_1_2, %qk_out_0_1_t1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_1 = xegpu.dpas %q_block_value_0_3, %k_value_slice_1_3, %qk_out_0_1_t2    : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

          %qk_out_1_1_t0 = xegpu.dpas %q_block_value_1_0, %k_value_slice_1_0, %zero_dpas      : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_1_t1 = xegpu.dpas %q_block_value_1_1, %k_value_slice_1_1, %qk_out_1_1_t0  : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_1_t2 = xegpu.dpas %q_block_value_1_2, %k_value_slice_1_2, %qk_out_1_1_t1  : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_1 = xegpu.dpas %q_block_value_1_3, %k_value_slice_1_3, %qk_out_1_1_t2     : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

          xegpu.compile_hint

          // load third  16x64 K slice
          %k_value_slice_2_0 = xegpu.load_nd %k_tile_slice_2_0  {transpose_bit_width = 32 : i32, transpose = array<i64: 1, 0>, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %k_value_slice_2_1 = xegpu.load_nd %k_tile_slice_2_1  {transpose_bit_width = 32 : i32, transpose = array<i64: 1, 0>, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %k_value_slice_2_2 = xegpu.load_nd %k_tile_slice_2_2  {transpose_bit_width = 32 : i32, transpose = array<i64: 1, 0>, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %k_value_slice_2_3 = xegpu.load_nd %k_tile_slice_2_3  {transpose_bit_width = 32 : i32, transpose = array<i64: 1, 0>, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          xegpu.compile_hint
          // update offsets
          %k_tile_slice_2_0_new = xegpu.update_nd_offset %k_tile_slice_2_0, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %k_tile_slice_2_1_new = xegpu.update_nd_offset %k_tile_slice_2_1, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %k_tile_slice_2_2_new = xegpu.update_nd_offset %k_tile_slice_2_2, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %k_tile_slice_2_3_new = xegpu.update_nd_offset %k_tile_slice_2_3, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          xegpu.compile_hint

          // compute third 16x16 of Q * K^T using DPAS
          %qk_out_0_2_t0 = xegpu.dpas %q_block_value_0_0, %k_value_slice_2_0, %zero_dpas : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_2_t1 = xegpu.dpas %q_block_value_0_1, %k_value_slice_2_1, %qk_out_0_2_t0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_2_t2 = xegpu.dpas %q_block_value_0_2, %k_value_slice_2_2, %qk_out_0_2_t1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_2 = xegpu.dpas %q_block_value_0_3, %k_value_slice_2_3, %qk_out_0_2_t2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

          %qk_out_1_2_t0 = xegpu.dpas %q_block_value_1_0, %k_value_slice_2_0, %zero_dpas : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_2_t1 = xegpu.dpas %q_block_value_1_1, %k_value_slice_2_1, %qk_out_1_2_t0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_2_t2 = xegpu.dpas %q_block_value_1_2, %k_value_slice_2_2, %qk_out_1_2_t1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_2 = xegpu.dpas %q_block_value_1_3, %k_value_slice_2_3, %qk_out_1_2_t2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

          xegpu.compile_hint

          // load forth  16x64 K slice
          %k_value_slice_3_0 = xegpu.load_nd %k_tile_slice_3_0  {transpose_bit_width = 32 : i32, transpose = array<i64: 1, 0>, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %k_value_slice_3_1 = xegpu.load_nd %k_tile_slice_3_1  {transpose_bit_width = 32 : i32, transpose = array<i64: 1, 0>, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %k_value_slice_3_2 = xegpu.load_nd %k_tile_slice_3_2  {transpose_bit_width = 32 : i32, transpose = array<i64: 1, 0>, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %k_value_slice_3_3 = xegpu.load_nd %k_tile_slice_3_3  {transpose_bit_width = 32 : i32, transpose = array<i64: 1, 0>, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          xegpu.compile_hint
          // update offsets
          %k_tile_slice_3_0_new = xegpu.update_nd_offset %k_tile_slice_3_0, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %k_tile_slice_3_1_new = xegpu.update_nd_offset %k_tile_slice_3_1, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %k_tile_slice_3_2_new = xegpu.update_nd_offset %k_tile_slice_3_2, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %k_tile_slice_3_3_new = xegpu.update_nd_offset %k_tile_slice_3_3, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          xegpu.compile_hint

          // compute forth 16x16 of Q * K^T using DPAS
          %qk_out_0_3_t0 = xegpu.dpas %q_block_value_0_0, %k_value_slice_3_0, %zero_dpas : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_3_t1 = xegpu.dpas %q_block_value_0_1, %k_value_slice_3_1, %qk_out_0_3_t0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_3_t2 = xegpu.dpas %q_block_value_0_2, %k_value_slice_3_2, %qk_out_0_3_t1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_0_3 = xegpu.dpas %q_block_value_0_3, %k_value_slice_3_3, %qk_out_0_3_t2  : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

          %qk_out_1_3_t0 = xegpu.dpas %q_block_value_1_0, %k_value_slice_3_0, %zero_dpas : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_3_t1 = xegpu.dpas %q_block_value_1_1, %k_value_slice_3_1, %qk_out_1_3_t0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_3_t2 = xegpu.dpas %q_block_value_1_2, %k_value_slice_3_2, %qk_out_1_3_t1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %qk_out_1_3 = xegpu.dpas %q_block_value_1_3, %k_value_slice_3_3, %qk_out_1_3_t2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

          xegpu.compile_hint

          // process row 0 of QK_out
          // do max reduction on qk_out row 0
          %qk_out_max_0_t0 = arith.maximumf %qk_out_0_0, %qk_out_0_1 fastmath<nnan> : vector<8x16xf32>
          %qk_out_max_0_t1 = arith.maximumf %qk_out_0_2, %qk_out_0_3 fastmath<nnan> : vector<8x16xf32>
          %qk_out_max_0_t2 = arith.maximumf %qk_out_max_0_t0, %qk_out_max_0_t1 fastmath<nnan> : vector<8x16xf32>
          %qk_out_max_0_t4 = vector.extract_strided_slice %qk_out_max_0_t2 {sizes = [8, 8], strides = [1, 1], offsets = [0, 0]} : vector<8x16xf32> to vector<8x8xf32>
          %qk_out_max_0_t5 = vector.extract_strided_slice %qk_out_max_0_t2 {sizes = [8, 8], strides = [1, 1], offsets = [0, 8]} : vector<8x16xf32> to vector<8x8xf32>
          %qk_out_max_0_t6 = arith.maximumf %qk_out_max_0_t4, %qk_out_max_0_t5 fastmath<nnan> : vector<8x8xf32>
          %qk_out_max_0_t7 = vector.extract_strided_slice %qk_out_max_0_t6 {sizes = [8, 4], strides = [1, 1], offsets = [0, 0]} : vector<8x8xf32> to vector<8x4xf32>
          %qk_out_max_0_t8 = vector.extract_strided_slice %qk_out_max_0_t6 {sizes = [8, 4], strides = [1, 1], offsets = [0, 4]} : vector<8x8xf32> to vector<8x4xf32>
          %qk_out_max_0_t9 = arith.maximumf %qk_out_max_0_t7, %qk_out_max_0_t8 fastmath<nnan>  : vector<8x4xf32>
          %qk_out_max_0_t10 = vector.extract_strided_slice %qk_out_max_0_t9 {sizes = [8, 2], strides = [1, 1], offsets = [0, 0]} : vector<8x4xf32> to vector<8x2xf32>
          %qk_out_max_0_t11 = vector.extract_strided_slice %qk_out_max_0_t9 {sizes = [8, 2], strides = [1, 1], offsets = [0, 2]} : vector<8x4xf32> to vector<8x2xf32>
          %qk_out_max_0_t12 = arith.maximumf %qk_out_max_0_t10, %qk_out_max_0_t11 fastmath<nnan> : vector<8x2xf32>
          %qk_out_max_0_t13 = vector.extract_strided_slice %qk_out_max_0_t12 {sizes = [8, 1], strides = [1, 1], offsets = [0, 0]} : vector<8x2xf32> to vector<8x1xf32>
          %qk_out_max_0_t14 = vector.extract_strided_slice %qk_out_max_0_t12 {sizes = [8, 1], strides = [1, 1], offsets = [0, 1]} : vector<8x2xf32> to vector<8x1xf32>
          %qk_out_max_0 = arith.maximumf %qk_out_max_0_t13, %qk_out_max_0_t14 fastmath<nnan> : vector<8x1xf32>
          // scale
          %qk_out_max_0_scaled = arith.mulf %qk_out_max_0, %qk_scale_8x1 : vector<8x1xf32>
          // find m_ij_row_0
          %m_ij_row_0 = arith.maximumf %qk_out_max_0_scaled, %m_i_row_0 fastmath<nnan> : vector<8x1xf32>
          // scale qk row 0 by qk_scale
          %qk_out_0_0_scaled = arith.mulf %qk_out_0_0, %qk_scale_8x16 : vector<8x16xf32>
          %qk_out_0_1_scaled = arith.mulf %qk_out_0_1, %qk_scale_8x16 : vector<8x16xf32>
          %qk_out_0_2_scaled = arith.mulf %qk_out_0_2, %qk_scale_8x16 : vector<8x16xf32>
          %qk_out_0_3_scaled = arith.mulf %qk_out_0_3, %qk_scale_8x16 : vector<8x16xf32>
          // broadcast m_ij_row_0 to 8x16
          %m_ij_row_0_broadcasted_t1 = vector.shape_cast %m_ij_row_0 : vector<8x1xf32> to vector<8xf32>
          %m_ij_row_0_broadcasted_t2 = vector.shuffle %m_ij_row_0_broadcasted_t1, %m_ij_row_0_broadcasted_t1
            [ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
              2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
              3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
              4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
              5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
              6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
              7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7] : vector<8xf32>, vector<8xf32>
          %m_ij_row_0_broadcasted = vector.shape_cast %m_ij_row_0_broadcasted_t2 : vector<128xf32> to vector<8x16xf32>
          // center qk_out by m_ij_row_0
          %qk_out_0_0_centered = arith.subf %qk_out_0_0_scaled, %m_ij_row_0_broadcasted : vector<8x16xf32>
          %qk_out_0_1_centered = arith.subf %qk_out_0_1_scaled, %m_ij_row_0_broadcasted : vector<8x16xf32>
          %qk_out_0_2_centered = arith.subf %qk_out_0_2_scaled, %m_ij_row_0_broadcasted : vector<8x16xf32>
          %qk_out_0_3_centered = arith.subf %qk_out_0_3_scaled, %m_ij_row_0_broadcasted : vector<8x16xf32>
          // take exp
          %qk_out_0_0_exp = math.exp %qk_out_0_0_centered : vector<8x16xf32>
          %qk_out_0_1_exp = math.exp %qk_out_0_1_centered : vector<8x16xf32>
          %qk_out_0_2_exp = math.exp %qk_out_0_2_centered : vector<8x16xf32>
          %qk_out_0_3_exp = math.exp %qk_out_0_3_centered : vector<8x16xf32>
          // do a sum reduction on exp output
          %l_ij_row_0_t0 = arith.addf %qk_out_0_0_exp, %qk_out_0_1_exp : vector<8x16xf32>
          %l_ij_row_0_t1 = arith.addf %qk_out_0_2_exp, %qk_out_0_3_exp : vector<8x16xf32>
          %l_ij_row_0_t2 = arith.addf %l_ij_row_0_t0, %l_ij_row_0_t1 : vector<8x16xf32>
          %l_ij_row_0_t3 = vector.extract_strided_slice %l_ij_row_0_t2 {sizes = [8, 8], strides = [1, 1], offsets = [0, 0]} : vector<8x16xf32> to vector<8x8xf32>
          %l_ij_row_0_t4 = vector.extract_strided_slice %l_ij_row_0_t2 {sizes = [8, 8], strides = [1, 1], offsets = [0, 8]} : vector<8x16xf32> to vector<8x8xf32>
          %l_ij_row_0_t5 = arith.addf %l_ij_row_0_t3, %l_ij_row_0_t4 : vector<8x8xf32>
          %l_ij_row_0_t6 = vector.extract_strided_slice %l_ij_row_0_t5 {sizes = [8, 4], strides = [1, 1], offsets = [0, 0]} : vector<8x8xf32> to vector<8x4xf32>
          %l_ij_row_0_t7 = vector.extract_strided_slice %l_ij_row_0_t5 {sizes = [8, 4], strides = [1, 1], offsets = [0, 4]} : vector<8x8xf32> to vector<8x4xf32>
          %l_ij_row_0_t8 = arith.addf %l_ij_row_0_t6, %l_ij_row_0_t7 : vector<8x4xf32>
          %l_ij_row_0_t9 = vector.extract_strided_slice %l_ij_row_0_t8 {sizes = [8, 2], strides = [1, 1], offsets = [0, 0]} : vector<8x4xf32> to vector<8x2xf32>
          %l_ij_row_0_t10 = vector.extract_strided_slice %l_ij_row_0_t8 {sizes = [8, 2], strides = [1, 1], offsets = [0, 2]} : vector<8x4xf32> to vector<8x2xf32>
          %l_ij_row_0_t11 = arith.addf %l_ij_row_0_t9, %l_ij_row_0_t10 : vector<8x2xf32>
          %l_ij_row_0_t12 = vector.extract_strided_slice %l_ij_row_0_t11 {sizes = [8, 1], strides = [1, 1], offsets = [0, 0]} : vector<8x2xf32> to vector<8x1xf32>
          %l_ij_row_0_t13 = vector.extract_strided_slice %l_ij_row_0_t11 {sizes = [8, 1], strides = [1, 1], offsets = [0, 1]} : vector<8x2xf32> to vector<8x1xf32>
          %l_ij_row_0 = arith.addf %l_ij_row_0_t12, %l_ij_row_0_t13 : vector<8x1xf32>
          // compute alpha
          %alpha_row_0_t1 = arith.subf %m_i_row_0, %m_ij_row_0 : vector<8x1xf32>
          %alpha_row_0 = math.exp %alpha_row_0_t1 : vector<8x1xf32>
          // update l_i
          %l_i_row_0_new_t1 = arith.mulf %l_i_row_0, %alpha_row_0 : vector<8x1xf32>
          %l_i_row_0_new = arith.addf %l_i_row_0_new_t1, %l_ij_row_0 : vector<8x1xf32>
          // update acc
          %alpha_row_0_broadcasted_t1 = vector.shape_cast %alpha_row_0 : vector<8x1xf32> to vector<8xf32>
          %alpha_row_0_broadcasted_t2 = vector.shuffle %alpha_row_0_broadcasted_t1, %alpha_row_0_broadcasted_t1
            [ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
              2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
              3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
              4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
              5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
              6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
              7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7] : vector<8xf32>, vector<8xf32>
          %alpha_row_0_broadcasted = vector.shape_cast %alpha_row_0_broadcasted_t2 : vector<128xf32> to vector<8x16xf32>
          %acc_in_0_0_updated = arith.mulf %acc_in_0_0, %alpha_row_0_broadcasted : vector<8x16xf32>
          %acc_in_0_1_updated = arith.mulf %acc_in_0_1, %alpha_row_0_broadcasted : vector<8x16xf32>
          %acc_in_0_2_updated = arith.mulf %acc_in_0_2, %alpha_row_0_broadcasted : vector<8x16xf32>
          %acc_in_0_3_updated = arith.mulf %acc_in_0_3, %alpha_row_0_broadcasted : vector<8x16xf32>

          xegpu.compile_hint

          // process row 1 of QK_out
          // do max reduction on qk_out row 1
          %qk_out_max_1_t0 = arith.maximumf %qk_out_1_0, %qk_out_1_1 fastmath<nnan> : vector<8x16xf32>
          %qk_out_max_1_t1 = arith.maximumf %qk_out_1_2, %qk_out_1_3 fastmath<nnan> : vector<8x16xf32>
          %qk_out_max_1_t2 = arith.maximumf %qk_out_max_1_t0, %qk_out_max_1_t1 fastmath<nnan> : vector<8x16xf32>
          %qk_out_max_1_t4 = vector.extract_strided_slice %qk_out_max_1_t2 {sizes = [8, 8], strides = [1, 1], offsets = [0, 0]} : vector<8x16xf32> to vector<8x8xf32>
          %qk_out_max_1_t5 = vector.extract_strided_slice %qk_out_max_1_t2 {sizes = [8, 8], strides = [1, 1], offsets = [0, 8]} : vector<8x16xf32> to vector<8x8xf32>
          %qk_out_max_1_t6 = arith.maximumf %qk_out_max_1_t4, %qk_out_max_1_t5 fastmath<nnan> : vector<8x8xf32>
          %qk_out_max_1_t7 = vector.extract_strided_slice %qk_out_max_1_t6 {sizes = [8, 4], strides = [1, 1], offsets = [0, 0]} : vector<8x8xf32> to vector<8x4xf32>
          %qk_out_max_1_t8 = vector.extract_strided_slice %qk_out_max_1_t6 {sizes = [8, 4], strides = [1, 1], offsets = [0, 4]} : vector<8x8xf32> to vector<8x4xf32>
          %qk_out_max_1_t9 = arith.maximumf %qk_out_max_1_t7, %qk_out_max_1_t8 fastmath<nnan> : vector<8x4xf32>
          %qk_out_max_1_t10 = vector.extract_strided_slice %qk_out_max_1_t9 {sizes = [8, 2], strides = [1, 1], offsets = [0, 0]} : vector<8x4xf32> to vector<8x2xf32>
          %qk_out_max_1_t11 = vector.extract_strided_slice %qk_out_max_1_t9 {sizes = [8, 2], strides = [1, 1], offsets = [0, 2]} : vector<8x4xf32> to vector<8x2xf32>
          %qk_out_max_1_t12 = arith.maximumf %qk_out_max_1_t10, %qk_out_max_1_t11 fastmath<nnan> : vector<8x2xf32>
          %qk_out_max_1_t13 = vector.extract_strided_slice %qk_out_max_1_t12 {sizes = [8, 1], strides = [1, 1], offsets = [0, 0]} : vector<8x2xf32> to vector<8x1xf32>
          %qk_out_max_1_t14 = vector.extract_strided_slice %qk_out_max_1_t12 {sizes = [8, 1], strides = [1, 1], offsets = [0, 1]} : vector<8x2xf32> to vector<8x1xf32>
          %qk_out_max_1 = arith.maximumf %qk_out_max_1_t13, %qk_out_max_1_t14 fastmath<nnan> : vector<8x1xf32>
          // scale
          %qk_out_max_1_scaled = arith.mulf %qk_out_max_1, %qk_scale_8x1 : vector<8x1xf32>
          // find m_ij_row_0
          %m_ij_row_1 = arith.maximumf %qk_out_max_1_scaled, %m_i_row_1 fastmath<nnan> : vector<8x1xf32>
          // scale qk row 0 by qk_scale
          %qk_out_1_0_scaled = arith.mulf %qk_out_1_0, %qk_scale_8x16 : vector<8x16xf32>
          %qk_out_1_1_scaled = arith.mulf %qk_out_1_1, %qk_scale_8x16 : vector<8x16xf32>
          %qk_out_1_2_scaled = arith.mulf %qk_out_1_2, %qk_scale_8x16 : vector<8x16xf32>
          %qk_out_1_3_scaled = arith.mulf %qk_out_1_3, %qk_scale_8x16 : vector<8x16xf32>
          // broadcast m_ij_row_0 to 8x16
          %m_ij_row_1_broadcasted_t1 = vector.shape_cast %m_ij_row_1 : vector<8x1xf32> to vector<8xf32>
          %m_ij_row_1_broadcasted_t2 = vector.shuffle %m_ij_row_1_broadcasted_t1, %m_ij_row_1_broadcasted_t1
            [ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
              2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
              3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
              4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
              5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
              6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
              7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7] : vector<8xf32>, vector<8xf32>
          %m_ij_row_1_broadcasted = vector.shape_cast %m_ij_row_1_broadcasted_t2 : vector<128xf32> to vector<8x16xf32>
          // center qk_out by m_ij_row_0
          %qk_out_1_0_centered = arith.subf %qk_out_1_0_scaled, %m_ij_row_1_broadcasted : vector<8x16xf32>
          %qk_out_1_1_centered = arith.subf %qk_out_1_1_scaled, %m_ij_row_1_broadcasted : vector<8x16xf32>
          %qk_out_1_2_centered = arith.subf %qk_out_1_2_scaled, %m_ij_row_1_broadcasted : vector<8x16xf32>
          %qk_out_1_3_centered = arith.subf %qk_out_1_3_scaled, %m_ij_row_1_broadcasted : vector<8x16xf32>
          // take exp
          %qk_out_1_0_exp = math.exp %qk_out_1_0_centered : vector<8x16xf32>
          %qk_out_1_1_exp = math.exp %qk_out_1_1_centered : vector<8x16xf32>
          %qk_out_1_2_exp = math.exp %qk_out_1_2_centered : vector<8x16xf32>
          %qk_out_1_3_exp = math.exp %qk_out_1_3_centered : vector<8x16xf32>
          // do a sum reduction on exp output
          %l_ij_row_1_t0 = arith.addf %qk_out_1_0_exp, %qk_out_1_1_exp : vector<8x16xf32>
          %l_ij_row_1_t1 = arith.addf %qk_out_1_2_exp, %qk_out_1_3_exp : vector<8x16xf32>
          %l_ij_row_1_t2 = arith.addf %l_ij_row_1_t0, %l_ij_row_1_t1 : vector<8x16xf32>
          %l_ij_row_1_t3 = vector.extract_strided_slice %l_ij_row_1_t2 {sizes = [8, 8], strides = [1, 1], offsets = [0, 0]} : vector<8x16xf32> to vector<8x8xf32>
          %l_ij_row_1_t4 = vector.extract_strided_slice %l_ij_row_1_t2 {sizes = [8, 8], strides = [1, 1], offsets = [0, 8]} : vector<8x16xf32> to vector<8x8xf32>
          %l_ij_row_1_t5 = arith.addf %l_ij_row_1_t3, %l_ij_row_1_t4 : vector<8x8xf32>
          %l_ij_row_1_t6 = vector.extract_strided_slice %l_ij_row_1_t5 {sizes = [8, 4], strides = [1, 1], offsets = [0, 0]} : vector<8x8xf32> to vector<8x4xf32>
          %l_ij_row_1_t7 = vector.extract_strided_slice %l_ij_row_1_t5 {sizes = [8, 4], strides = [1, 1], offsets = [0, 4]} : vector<8x8xf32> to vector<8x4xf32>
          %l_ij_row_1_t8 = arith.addf %l_ij_row_1_t6, %l_ij_row_1_t7 : vector<8x4xf32>
          %l_ij_row_1_t9 = vector.extract_strided_slice %l_ij_row_1_t8 {sizes = [8, 2], strides = [1, 1], offsets = [0, 0]} : vector<8x4xf32> to vector<8x2xf32>
          %l_ij_row_1_t10 = vector.extract_strided_slice %l_ij_row_1_t8 {sizes = [8, 2], strides = [1, 1], offsets = [0, 2]} : vector<8x4xf32> to vector<8x2xf32>
          %l_ij_row_1_t11 = arith.addf %l_ij_row_1_t9, %l_ij_row_1_t10 : vector<8x2xf32>
          %l_ij_row_1_t12 = vector.extract_strided_slice %l_ij_row_1_t11 {sizes = [8, 1], strides = [1, 1], offsets = [0, 0]} : vector<8x2xf32> to vector<8x1xf32>
          %l_ij_row_1_t13 = vector.extract_strided_slice %l_ij_row_1_t11 {sizes = [8, 1], strides = [1, 1], offsets = [0, 1]} : vector<8x2xf32> to vector<8x1xf32>
          %l_ij_row_1 = arith.addf %l_ij_row_1_t12, %l_ij_row_1_t13 : vector<8x1xf32>
          // compute alpha
          %alpha_row_1_t1 = arith.subf %m_i_row_1, %m_ij_row_1 : vector<8x1xf32>
          %alpha_row_1 = math.exp %alpha_row_1_t1 : vector<8x1xf32>
          // update l_i
          %l_i_row_1_new_t1 = arith.mulf %l_i_row_1, %alpha_row_1 : vector<8x1xf32>
          %l_i_row_1_new = arith.addf %l_i_row_1_new_t1, %l_ij_row_1 : vector<8x1xf32>
          // update acc
          %alpha_row_1_broadcasted_t1 = vector.shape_cast %alpha_row_1 : vector<8x1xf32> to vector<8xf32>
          %alpha_row_1_broadcasted_t2 = vector.shuffle %alpha_row_1_broadcasted_t1, %alpha_row_1_broadcasted_t1
            [ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
              2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
              3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
              4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
              5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
              6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
              7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7] : vector<8xf32>, vector<8xf32>
          %alpha_row_1_broadcasted = vector.shape_cast %alpha_row_1_broadcasted_t2 : vector<128xf32> to vector<8x16xf32>
          %acc_in_1_0_updated = arith.mulf %acc_in_1_0, %alpha_row_1_broadcasted : vector<8x16xf32>
          %acc_in_1_1_updated = arith.mulf %acc_in_1_1, %alpha_row_1_broadcasted : vector<8x16xf32>
          %acc_in_1_2_updated = arith.mulf %acc_in_1_2, %alpha_row_1_broadcasted : vector<8x16xf32>
          %acc_in_1_3_updated = arith.mulf %acc_in_1_3, %alpha_row_1_broadcasted : vector<8x16xf32>

          // convert qk_out_tile to A format for DPAS for p * v computation
          %qk_out_0_0_f16 = arith.truncf %qk_out_0_0_exp : vector<8x16xf32> to vector<8x16xf16>
          %qk_out_0_1_f16 = arith.truncf %qk_out_0_1_exp : vector<8x16xf32> to vector<8x16xf16>
          %qk_out_0_2_f16 = arith.truncf %qk_out_0_2_exp : vector<8x16xf32> to vector<8x16xf16>
          %qk_out_0_3_f16 = arith.truncf %qk_out_0_3_exp : vector<8x16xf32> to vector<8x16xf16>
          %qk_out_1_0_f16 = arith.truncf %qk_out_1_0_exp : vector<8x16xf32> to vector<8x16xf16>
          %qk_out_1_1_f16 = arith.truncf %qk_out_1_1_exp : vector<8x16xf32> to vector<8x16xf16>
          %qk_out_1_2_f16 = arith.truncf %qk_out_1_2_exp : vector<8x16xf32> to vector<8x16xf16>
          %qk_out_1_3_f16 = arith.truncf %qk_out_1_3_exp : vector<8x16xf32> to vector<8x16xf16>

          xegpu.compile_hint
          // load first 16x64 V slices
          %v_val_slice_0_0 = xegpu.load_nd %v_tile_slice_0_0 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %v_val_slice_0_1 = xegpu.load_nd %v_tile_slice_0_1 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %v_val_slice_0_2 = xegpu.load_nd %v_tile_slice_0_2 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %v_val_slice_0_3 = xegpu.load_nd %v_tile_slice_0_3 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          xegpu.compile_hint
          // update offsets
          %v_tile_slice_0_0_new = xegpu.update_nd_offset %v_tile_slice_0_0, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %v_tile_slice_0_1_new = xegpu.update_nd_offset %v_tile_slice_0_1, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %v_tile_slice_0_2_new = xegpu.update_nd_offset %v_tile_slice_0_2, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %v_tile_slice_0_3_new = xegpu.update_nd_offset %v_tile_slice_0_3, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          xegpu.compile_hint


          xegpu.compile_hint
          // compute first iteration update of 16x64 of P * V
          %pv_out_0_0_iter0 = xegpu.dpas %qk_out_0_0_f16, %v_val_slice_0_0, %acc_in_0_0_updated : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_0_iter0 = xegpu.dpas %qk_out_1_0_f16, %v_val_slice_0_0, %acc_in_1_0_updated : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_1_iter0 = xegpu.dpas %qk_out_0_0_f16, %v_val_slice_0_1, %acc_in_0_1_updated : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_1_iter0 = xegpu.dpas %qk_out_1_0_f16, %v_val_slice_0_1, %acc_in_1_1_updated : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_2_iter0 = xegpu.dpas %qk_out_0_0_f16, %v_val_slice_0_2, %acc_in_0_2_updated : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_2_iter0 = xegpu.dpas %qk_out_1_0_f16, %v_val_slice_0_2, %acc_in_1_2_updated : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_3_iter0 = xegpu.dpas %qk_out_0_0_f16, %v_val_slice_0_3, %acc_in_0_3_updated : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_3_iter0 = xegpu.dpas %qk_out_1_0_f16, %v_val_slice_0_3, %acc_in_1_3_updated : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          xegpu.compile_hint

          // load second 16x64 V slices
          %v_val_slice_1_0 = xegpu.load_nd %v_tile_slice_1_0 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %v_val_slice_1_1 = xegpu.load_nd %v_tile_slice_1_1 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %v_val_slice_1_2 = xegpu.load_nd %v_tile_slice_1_2 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %v_val_slice_1_3 = xegpu.load_nd %v_tile_slice_1_3 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          xegpu.compile_hint
          // update offsets
          %v_tile_slice_1_0_new = xegpu.update_nd_offset %v_tile_slice_1_0, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %v_tile_slice_1_1_new = xegpu.update_nd_offset %v_tile_slice_1_1, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %v_tile_slice_1_2_new = xegpu.update_nd_offset %v_tile_slice_1_2, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %v_tile_slice_1_3_new = xegpu.update_nd_offset %v_tile_slice_1_3, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          xegpu.compile_hint
          // compute second iteration update of 16x64 of P * V
          %pv_out_0_0_iter1 = xegpu.dpas %qk_out_0_1_f16, %v_val_slice_1_0, %pv_out_0_0_iter0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_0_iter1 = xegpu.dpas %qk_out_1_1_f16, %v_val_slice_1_0, %pv_out_1_0_iter0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_1_iter1 = xegpu.dpas %qk_out_0_1_f16, %v_val_slice_1_1, %pv_out_0_1_iter0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_1_iter1 = xegpu.dpas %qk_out_1_1_f16, %v_val_slice_1_1, %pv_out_1_1_iter0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_2_iter1 = xegpu.dpas %qk_out_0_1_f16, %v_val_slice_1_2, %pv_out_0_2_iter0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_2_iter1 = xegpu.dpas %qk_out_1_1_f16, %v_val_slice_1_2, %pv_out_1_2_iter0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_3_iter1 = xegpu.dpas %qk_out_0_1_f16, %v_val_slice_1_3, %pv_out_0_3_iter0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_3_iter1 = xegpu.dpas %qk_out_1_1_f16, %v_val_slice_1_3, %pv_out_1_3_iter0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

          xegpu.compile_hint

          // load third 16x64 V slices
          %v_val_slice_2_0 = xegpu.load_nd %v_tile_slice_2_0  {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %v_val_slice_2_1 = xegpu.load_nd %v_tile_slice_2_1  {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %v_val_slice_2_2 = xegpu.load_nd %v_tile_slice_2_2  {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %v_val_slice_2_3 = xegpu.load_nd %v_tile_slice_2_3  {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          xegpu.compile_hint
          // update offsets
          %v_tile_slice_2_0_new = xegpu.update_nd_offset %v_tile_slice_2_0, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %v_tile_slice_2_1_new = xegpu.update_nd_offset %v_tile_slice_2_1, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %v_tile_slice_2_2_new = xegpu.update_nd_offset %v_tile_slice_2_2, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %v_tile_slice_2_3_new = xegpu.update_nd_offset %v_tile_slice_2_3, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          xegpu.compile_hint
          // compute third iteration update of 16x64 of P * V
          %pv_out_0_0_iter2 = xegpu.dpas %qk_out_0_2_f16, %v_val_slice_2_0, %pv_out_0_0_iter1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_0_iter2 = xegpu.dpas %qk_out_1_2_f16, %v_val_slice_2_0, %pv_out_1_0_iter1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_1_iter2 = xegpu.dpas %qk_out_0_2_f16, %v_val_slice_2_1, %pv_out_0_1_iter1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_1_iter2 = xegpu.dpas %qk_out_1_2_f16, %v_val_slice_2_1, %pv_out_1_1_iter1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_2_iter2 = xegpu.dpas %qk_out_0_2_f16, %v_val_slice_2_2, %pv_out_0_2_iter1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_2_iter2 = xegpu.dpas %qk_out_1_2_f16, %v_val_slice_2_2, %pv_out_1_2_iter1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_3_iter2 = xegpu.dpas %qk_out_0_2_f16, %v_val_slice_2_3, %pv_out_0_3_iter1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_3_iter2 = xegpu.dpas %qk_out_1_2_f16, %v_val_slice_2_3, %pv_out_1_3_iter1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

          xegpu.compile_hint

          // load forth 16x64 V slices
          %v_val_slice_3_0 = xegpu.load_nd %v_tile_slice_3_0  {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %v_val_slice_3_1 = xegpu.load_nd %v_tile_slice_3_1  {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %v_val_slice_3_2 = xegpu.load_nd %v_tile_slice_3_2  {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %v_val_slice_3_3 = xegpu.load_nd %v_tile_slice_3_3  {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          xegpu.compile_hint
          // update offsets
          %v_tile_slice_3_0_new = xegpu.update_nd_offset %v_tile_slice_3_0, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %v_tile_slice_3_1_new = xegpu.update_nd_offset %v_tile_slice_3_1, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %v_tile_slice_3_2_new = xegpu.update_nd_offset %v_tile_slice_3_2, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          %v_tile_slice_3_3_new = xegpu.update_nd_offset %v_tile_slice_3_3, [%BLOCK_N , %c0] : !xegpu.tensor_desc<16x16xf16>
          xegpu.compile_hint
          // compute third iteration update of 16x64 of P * V
          %pv_out_0_0_iter3 = xegpu.dpas %qk_out_0_3_f16, %v_val_slice_3_0, %pv_out_0_0_iter2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_0_iter3 = xegpu.dpas %qk_out_1_3_f16, %v_val_slice_3_0, %pv_out_1_0_iter2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_1_iter3 = xegpu.dpas %qk_out_0_3_f16, %v_val_slice_3_1, %pv_out_0_1_iter2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_1_iter3 = xegpu.dpas %qk_out_1_3_f16, %v_val_slice_3_1, %pv_out_1_1_iter2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_2_iter3 = xegpu.dpas %qk_out_0_3_f16, %v_val_slice_3_2, %pv_out_0_2_iter2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_2_iter3 = xegpu.dpas %qk_out_1_3_f16, %v_val_slice_3_2, %pv_out_1_2_iter2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_0_3_iter3 = xegpu.dpas %qk_out_0_3_f16, %v_val_slice_3_3, %pv_out_0_3_iter2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %pv_out_1_3_iter3 = xegpu.dpas %qk_out_1_3_f16, %v_val_slice_3_3, %pv_out_1_3_iter2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          xegpu.compile_hint

          xegpu.nbarrier_wait %nbarrier : !xegpu.nbarrier

          scf.yield
                  %pv_out_0_0_iter3, %pv_out_0_1_iter3,  %pv_out_0_2_iter3,  %pv_out_0_3_iter3,
                  %pv_out_1_0_iter3, %pv_out_1_1_iter3,  %pv_out_1_2_iter3,  %pv_out_1_3_iter3,
                  %k_tile_slice_0_0_new, %k_tile_slice_0_1_new, %k_tile_slice_0_2_new, %k_tile_slice_0_3_new,
                  %k_tile_slice_1_0_new, %k_tile_slice_1_1_new, %k_tile_slice_1_2_new, %k_tile_slice_1_3_new,
                  %k_tile_slice_2_0_new, %k_tile_slice_2_1_new, %k_tile_slice_2_2_new, %k_tile_slice_2_3_new,
                  %k_tile_slice_3_0_new, %k_tile_slice_3_1_new, %k_tile_slice_3_2_new, %k_tile_slice_3_3_new,

                  %v_tile_slice_0_0_new, %v_tile_slice_0_1_new, %v_tile_slice_0_2_new, %v_tile_slice_0_3_new,
                  %v_tile_slice_1_0_new, %v_tile_slice_1_1_new, %v_tile_slice_1_2_new, %v_tile_slice_1_3_new,
                  %v_tile_slice_2_0_new, %v_tile_slice_2_1_new, %v_tile_slice_2_2_new, %v_tile_slice_2_3_new,
                  %v_tile_slice_3_0_new, %v_tile_slice_3_1_new, %v_tile_slice_3_2_new, %v_tile_slice_3_3_new,

                  %k_prefetch_tile_new, %v_prefetch_tile_new,
                  %m_ij_row_0, %m_ij_row_1, %l_i_row_0_new, %l_i_row_1_new :
                  vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>,

                  !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>,!xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>,
                  !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>,!xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>,
                  !xegpu.tensor_desc<16x32xf16>, !xegpu.tensor_desc<16x32xf16>,
                  vector<8x1xf32>, vector<8x1xf32>, vector<8x1xf32>, vector<8x1xf32>

      }
      // divide acc output by l_i
      %l_i_row_0_broadcast_t1 = vector.shape_cast %result#44 : vector<8x1xf32> to vector<8xf32>
      %l_i_row_0_broadcast_t2 = vector.shuffle %l_i_row_0_broadcast_t1, %l_i_row_0_broadcast_t1
            [ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
              2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
              3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
              4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
              5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
              6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
              7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7] : vector<8xf32>, vector<8xf32>
      %l_i_row_0_broadcast = vector.shape_cast %l_i_row_0_broadcast_t2 : vector<128xf32> to vector<8x16xf32>

      %l_i_row_1_broadcast_t1 = vector.shape_cast %result#45 : vector<8x1xf32> to vector<8xf32>
      %l_i_row_1_broadcast_t2 = vector.shuffle %l_i_row_1_broadcast_t1, %l_i_row_1_broadcast_t1
            [ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
              2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
              3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
              4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
              5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
              6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
              7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7] : vector<8xf32>, vector<8xf32>
      %l_i_row_1_broadcast = vector.shape_cast %l_i_row_1_broadcast_t2 : vector<128xf32> to vector<8x16xf32>
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

      // O tile, max size is 8x32
      %o_tile_init_0_0  = xegpu.create_nd_tdesc %Out [%sg_q_x_offset, %c0], shape: [%size_x, %BLOCK_DMODEL], strides: [%BLOCK_DMODEL, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<8x32xf16>
      %o_tile_init_0_1 = xegpu.update_nd_offset %o_tile_init_0_0, [%c0, %c32] : !xegpu.tensor_desc<8x32xf16>
      %o_tile_init_1_0 = xegpu.update_nd_offset %o_tile_init_0_0, [%c8, %c0] : !xegpu.tensor_desc<8x32xf16>
      %o_tile_init_1_1 = xegpu.update_nd_offset %o_tile_init_1_0, [%c0, %c32] : !xegpu.tensor_desc<8x32xf16>

      %o_val_8x32_0_0_t1 = vector.shuffle %o_val_final_0_0, %o_val_final_0_1 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %o_val_8x32_0_0_t2 = vector.shape_cast %o_val_8x32_0_0_t1 : vector<16x16xf16> to vector<256xf16>
      %o_val_8x32_0_0_t3 = vector.shape_cast %o_val_8x32_0_0_t2 : vector<256xf16> to vector<8x32xf16>
      xegpu.store_nd %o_val_8x32_0_0_t3, %o_tile_init_0_0 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %o_val_8x32_0_1_t1 = vector.shuffle %o_val_final_0_2, %o_val_final_0_3 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %o_val_8x32_0_1_t2 = vector.shape_cast %o_val_8x32_0_1_t1 : vector<16x16xf16> to vector<256xf16>
      %o_val_8x32_0_1_t3 = vector.shape_cast %o_val_8x32_0_1_t2 : vector<256xf16> to vector<8x32xf16>
      xegpu.store_nd %o_val_8x32_0_1_t3, %o_tile_init_0_1 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %o_val_8x32_1_0_t1 = vector.shuffle %o_val_final_1_0, %o_val_final_1_1 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %o_val_8x32_1_0_t2 = vector.shape_cast %o_val_8x32_1_0_t1 : vector<16x16xf16> to vector<256xf16>
      %o_val_8x32_1_0_t3 = vector.shape_cast %o_val_8x32_1_0_t2 : vector<256xf16> to vector<8x32xf16>
      xegpu.store_nd %o_val_8x32_1_0_t3, %o_tile_init_1_0 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %o_val_8x32_1_1_t1 = vector.shuffle %o_val_final_1_2, %o_val_final_1_3 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %o_val_8x32_1_1_t2 = vector.shape_cast %o_val_8x32_1_1_t1 : vector<16x16xf16> to vector<256xf16>
      %o_val_8x32_1_1_t3 = vector.shape_cast %o_val_8x32_1_1_t2 : vector<256xf16> to vector<8x32xf16>
      xegpu.store_nd %o_val_8x32_1_1_t3, %o_tile_init_1_1 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint


      gpu.return
    }
  }

  func.func @gpu_impl(%q : memref<?x?xf16>, %k : memref<?x?xf16>, %v : memref<?x?xf16>,
    %o : memref<?x?xf16>, %Z : index, %H : index, %N_CTX : index, %D_HEAD : index,
    %sm_scale : f32) -> memref<?x?xf16> {

    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
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

    %q_gpu = gpu.alloc host_shared (%Z_H_N, %D_HEAD) : memref<?x?xf16>
    %k_gpu = gpu.alloc host_shared (%Z_H_N, %D_HEAD) : memref<?x?xf16>
    %v_gpu = gpu.alloc host_shared (%Z_H_N, %D_HEAD) : memref<?x?xf16>
    %o_gpu = gpu.alloc host_shared (%Z_H_N, %D_HEAD) : memref<?x?xf16>
    // %m_gpu = gpu.alloc host_shared (%Z, %H, %N_CTX) : memref<?x?x?xf32>

    // copy from CPU to
    memref.copy %q, %q_gpu : memref<?x?xf16> to memref<?x?xf16>
    memref.copy %k, %k_gpu : memref<?x?xf16> to memref<?x?xf16>
    memref.copy %v, %v_gpu : memref<?x?xf16> to memref<?x?xf16>
    memref.copy %o, %o_gpu : memref<?x?xf16> to memref<?x?xf16>
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
    gpu.launch_func @flash_attention_fwd::@flash_attention_fwd blocks in (%blocks_x, %blocks_y, %c1)
      threads in (%c8, %c1, %c1) args(
      %q_gpu : memref<?x?xf16>, %k_gpu : memref<?x?xf16>, %v_gpu : memref<?x?xf16>, %o_gpu : memref<?x?xf16>,
      %sm_scale : f32,
      %stride_2 : index, %stride_1 : index, %D_HEAD : index, %c1 : index,
      %stride_2 : index, %stride_1 : index, %D_HEAD : index, %c1 : index,
      %stride_2 : index, %stride_1 : index, %D_HEAD : index, %c1 : index,
      %stride_2 : index, %stride_1 : index, %D_HEAD : index, %c1 : index,
      %Z : index, %H : index, %N_CTX : index, %BLOCK_M : index, %D_HEAD : index, %BLOCK_N : index
      )

    // copy output to CPU
    memref.copy %o_gpu, %o : memref<?x?xf16> to memref<?x?xf16>

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

    // xegpu only supports 2d memrefs. So we collapse the first 3 dims of the inputs
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
