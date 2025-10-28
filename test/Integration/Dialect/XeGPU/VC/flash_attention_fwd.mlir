// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @flash_attention attributes {gpu.container_module} {
  gpu.module @flash_attention_fwd  {
    gpu.func @flash_attention_fwd(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?x?xf16>, %arg3: memref<?x?xf16>, %arg4: f32, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index, %arg12: index, %arg13: index, %arg14: index, %arg15: index, %arg16: index, %arg17: index, %arg18: index, %arg19: index, %arg20: index, %arg21: index, %arg22: index, %arg23: index, %arg24: index, %arg25: index, %arg26: index) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      // memref sizes in x dim
      // calculate the WG x offset of the q tile. This is equal to off_hz * N_CTX + start_m * BLOCK_M
      // for k and v offsets are off_zh * N_CTX because inside the K loop we will consume N_CTX length
      // this is eqaul to wg_x_offset
      // compute the SG x offset for the q tile.
      // wg_q_offset + sg_x_slice_size * sg_id
      // init tile for 16x64 Q tiles
      // init tile for 64x64 K tiles. We do this in 4 stages of 16x64 tiles to reduce register pressure.
      // k is reused by all SGs
      // same for V tiles
      // k preftech
      // prefetch 16x32 tiles in 4x2 layout to cover 64x64
      // x offset for prefetch is same as for q tiles. This means that WGs assigned to same bacth also colloborate on prefetching
      // the K, V tiles.
      // We also tried WGs prefetching from the begining of the K, V tiles but that did not work well because multiple
      // WGs compete to prefetch the same data.
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = gpu.subgroup_id : index
      %1 = arith.muli %arg21, %arg22 : index
      %2 = arith.muli %1, %arg23 : index
      %3 = arith.muli %block_id_y, %arg23 : index
      %4 = arith.muli %block_id_x, %arg24 : index
      %5 = arith.addi %3, %4 : index
      %6 = arith.divui %arg24, %c8 : index
      %7 = arith.muli %0, %6 : index
      %8 = arith.addi %5, %7 : index
      %9 = xegpu.create_nd_tdesc %arg0[%8, %c0], shape : [%2, %arg25], strides : [%arg25, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<16x16xf16>
      %10 = xegpu.update_nd_offset %9, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %11 = xegpu.update_nd_offset %10, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %12 = xegpu.update_nd_offset %11, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %13 = xegpu.create_nd_tdesc %arg1[%3, %c0], shape : [%2, %arg25], strides : [%arg25, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<16x16xf16>
      %14 = xegpu.update_nd_offset %13, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %15 = xegpu.update_nd_offset %14, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %16 = xegpu.update_nd_offset %15, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %17 = xegpu.update_nd_offset %13, [%c16, %c0] : !xegpu.tensor_desc<16x16xf16>
      %18 = xegpu.update_nd_offset %17, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %19 = xegpu.update_nd_offset %18, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %20 = xegpu.update_nd_offset %19, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %21 = xegpu.update_nd_offset %17, [%c16, %c0] : !xegpu.tensor_desc<16x16xf16>
      %22 = xegpu.update_nd_offset %21, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %23 = xegpu.update_nd_offset %22, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %24 = xegpu.update_nd_offset %23, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %25 = xegpu.update_nd_offset %21, [%c16, %c0] : !xegpu.tensor_desc<16x16xf16>
      %26 = xegpu.update_nd_offset %25, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %27 = xegpu.update_nd_offset %26, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %28 = xegpu.update_nd_offset %27, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %29 = xegpu.create_nd_tdesc %arg2[%3, %c0], shape : [%2, %arg25], strides : [%arg25, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<16x16xf16>
      %30 = xegpu.update_nd_offset %29, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %31 = xegpu.update_nd_offset %30, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %32 = xegpu.update_nd_offset %31, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %33 = xegpu.update_nd_offset %29, [%c16, %c0] : !xegpu.tensor_desc<16x16xf16>
      %34 = xegpu.update_nd_offset %33, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %35 = xegpu.update_nd_offset %34, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %36 = xegpu.update_nd_offset %35, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %37 = xegpu.update_nd_offset %33, [%c16, %c0] : !xegpu.tensor_desc<16x16xf16>
      %38 = xegpu.update_nd_offset %37, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %39 = xegpu.update_nd_offset %38, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %40 = xegpu.update_nd_offset %39, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %41 = xegpu.update_nd_offset %37, [%c16, %c0] : !xegpu.tensor_desc<16x16xf16>
      %42 = xegpu.update_nd_offset %41, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %43 = xegpu.update_nd_offset %42, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %44 = xegpu.update_nd_offset %43, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %c2 = arith.constant 2 : index
      // V prefetch is similar to K
      // initialize m, l and acc
      // softmax scaling
      // %qk_scale_8 = spirv.CompositeConstruct %sm_scale, %sm_scale, %sm_scale, %sm_scale, %sm_scale, %sm_scale, %sm_scale, %sm_scale : (f32, f32, f32, f32, f32, f32, f32, f32) -> vector<8xf32>
      // %qk_scale_16 = spirv.CompositeConstruct %sm_scale, %sm_scale, %sm_scale, %sm_scale, %sm_scale, %sm_scale, %sm_scale, %sm_scale,%sm_scale, %sm_scale, %sm_scale, %sm_scale,%sm_scale, %sm_scale, %sm_scale, %sm_scale : (f32, f32, f32, f32,f32, f32, f32, f32,f32, f32, f32, f32,f32, f32, f32, f32 ) -> vector<16xf32>
      // FIXME: value 0.5 is hard coded. need to take it from %sm_scale
      // load Q tiles
      // ----
      %45 = arith.divui %0, %c2 : index
      %46 = arith.remui %0, %c2 : index
      %47 = arith.muli %45, %c16 : index
      %48 = arith.addi %5, %47 : index
      %49 = arith.muli %46, %c32 : index
      %50 = xegpu.create_nd_tdesc %arg1[%48, %49], shape : [%2, %arg25], strides : [%arg25, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<16x32xf16>
      xegpu.prefetch_nd %50 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x32xf16>
      %51 = xegpu.update_nd_offset %50, [%arg26, %c0] : !xegpu.tensor_desc<16x32xf16>
      xegpu.prefetch_nd %51 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x32xf16>
      %52 = xegpu.update_nd_offset %51, [%arg26, %c0] : !xegpu.tensor_desc<16x32xf16>
      xegpu.prefetch_nd %52 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x32xf16>
      %53 = xegpu.update_nd_offset %52, [%arg26, %c0] : !xegpu.tensor_desc<16x32xf16>
      %54 = xegpu.create_nd_tdesc %arg2[%48, %49], shape : [%2, %arg25], strides : [%arg25, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<16x32xf16>
      xegpu.prefetch_nd %54 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x32xf16>
      %55 = xegpu.update_nd_offset %54, [%arg26, %c0] : !xegpu.tensor_desc<16x32xf16>
      xegpu.prefetch_nd %55 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x32xf16>
      %56 = xegpu.update_nd_offset %55, [%arg26, %c0] : !xegpu.tensor_desc<16x32xf16>
      xegpu.prefetch_nd %56 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x32xf16>
      %57 = xegpu.update_nd_offset %56, [%arg26, %c0] : !xegpu.tensor_desc<16x32xf16>
      %cst = arith.constant dense<0xFF800000> : vector<8xf32>
      %cst_0 = arith.constant dense<0xFF800000> : vector<8xf32>
      %cst_1 = arith.constant dense<1.000000e+00> : vector<8xf32>
      %cst_2 = arith.constant dense<1.000000e+00> : vector<8xf32>
      %58 = vector.shape_cast %cst : vector<8xf32> to vector<8x1xf32>
      %59 = vector.shape_cast %cst_0 : vector<8xf32> to vector<8x1xf32>
      %60 = vector.shape_cast %cst_1 : vector<8xf32> to vector<8x1xf32>
      %61 = vector.shape_cast %cst_2 : vector<8xf32> to vector<8x1xf32>
      %cst_3 = arith.constant dense<0.000000e+00> : vector<128xf32>
      %62 = vector.shape_cast %cst_3 : vector<128xf32> to vector<8x16xf32>
      %cst_4 = arith.constant dense<5.000000e-01> : vector<8xf32>
      %cst_5 = arith.constant dense<5.000000e-01> : vector<16xf32>
      %63 = vector.shape_cast %cst_4 : vector<8xf32> to vector<8x1xf32>
      %64 = vector.shape_cast %cst_5 : vector<16xf32> to vector<1x16xf32>
      %65 = vector.shuffle %64, %64 [0, 0, 0, 0, 0, 0, 0, 0] : vector<1x16xf32>, vector<1x16xf32>
      %66 = xegpu.load_nd %9 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
      %67 = xegpu.load_nd %10 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
      %68 = xegpu.load_nd %11 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
      %69 = xegpu.load_nd %12 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
      %70 = vector.shape_cast %66 : vector<16x16xf16> to vector<256xf16>
      %71 = vector.shape_cast %67 : vector<16x16xf16> to vector<256xf16>
      %72 = vector.shape_cast %68 : vector<16x16xf16> to vector<256xf16>
      %73 = vector.shape_cast %69 : vector<16x16xf16> to vector<256xf16>
      %74 = vector.extract_strided_slice %70 {offsets = [0], sizes = [128], strides = [1]} : vector<256xf16> to vector<128xf16>
      %75 = vector.shape_cast %74 : vector<128xf16> to vector<8x16xf16>
      %76 = vector.extract_strided_slice %70 {offsets = [128], sizes = [128], strides = [1]} : vector<256xf16> to vector<128xf16>
      %77 = vector.shape_cast %76 : vector<128xf16> to vector<8x16xf16>
      %78 = vector.extract_strided_slice %71 {offsets = [0], sizes = [128], strides = [1]} : vector<256xf16> to vector<128xf16>
      %79 = vector.shape_cast %78 : vector<128xf16> to vector<8x16xf16>
      %80 = vector.extract_strided_slice %71 {offsets = [128], sizes = [128], strides = [1]} : vector<256xf16> to vector<128xf16>
      %81 = vector.shape_cast %80 : vector<128xf16> to vector<8x16xf16>
      %82 = vector.extract_strided_slice %72 {offsets = [0], sizes = [128], strides = [1]} : vector<256xf16> to vector<128xf16>
      %83 = vector.shape_cast %82 : vector<128xf16> to vector<8x16xf16>
      %84 = vector.extract_strided_slice %72 {offsets = [128], sizes = [128], strides = [1]} : vector<256xf16> to vector<128xf16>
      %85 = vector.shape_cast %84 : vector<128xf16> to vector<8x16xf16>
      %86 = vector.extract_strided_slice %73 {offsets = [0], sizes = [128], strides = [1]} : vector<256xf16> to vector<128xf16>
      %87 = vector.shape_cast %86 : vector<128xf16> to vector<8x16xf16>
      %88 = vector.extract_strided_slice %73 {offsets = [128], sizes = [128], strides = [1]} : vector<256xf16> to vector<128xf16>
      %89 = vector.shape_cast %88 : vector<128xf16> to vector<8x16xf16>
      xegpu.alloc_nbarrier 16
      // inner loop. This loop iterate over K and V tiles and update the accumulator by computing softmax(q*k^T)*v
          /// prefetch
          // k prefetch
          // V prefetch
          // load first 16x64 K slice
          // update offsets
          // compute first 16x16 of Q * K^T using DPAS
          // load second 16x64 K slice
          // update offsets
          // compute second 16x16 of Q * K^T using DPAS
          // load third  16x64 K slice
          // update offsets
          // compute third 16x16 of Q * K^T using DPAS
          // load forth  16x64 K slice
          // update offsets
          // compute forth 16x16 of Q * K^T using DPAS
          // process row 0 of QK_out
          // do max reduction on qk_out row 0
          // scale
          // find m_ij_row_0
          // scale qk row 0 by qk_scale
          // broadcast m_ij_row_0 to 8x16
          // center qk_out by m_ij_row_0
          // take exp
          // do a sum reduction on exp output
          // compute alpha
          // update l_i
          // update acc
          // process row 1 of QK_out
          // do max reduction on qk_out row 1
          // scale
          // find m_ij_row_0
          // scale qk row 0 by qk_scale
          // broadcast m_ij_row_0 to 8x16
          // center qk_out by m_ij_row_0
          // take exp
          // do a sum reduction on exp output
          // compute alpha
          // update l_i
          // update acc
          // convert qk_out_tile to A format for DPAS for p * v computation
          // load first 16x64 V slices
          // update offsets
          // compute first iteration update of 16x64 of P * V
          // load second 16x64 V slices
          // update offsets
          // compute second iteration update of 16x64 of P * V
          // load third 16x64 V slices
          // update offsets
          // compute third iteration update of 16x64 of P * V
          // load forth 16x64 V slices
          // update offsets
          // compute third iteration update of 16x64 of P * V
      %c1_i8 = arith.constant 1 : i8
      %c8_i8 = arith.constant 8 : i8
      %90 = xegpu.init_nbarrier %c1_i8, %c8_i8 : i8, i8 -> !xegpu.nbarrier
      %91:46 = scf.for %arg27 = %c0 to %arg23 step %arg26 iter_args(%arg28 = %62, %arg29 = %62, %arg30 = %62, %arg31 = %62, %arg32 = %62, %arg33 = %62, %arg34 = %62, %arg35 = %62, %arg36 = %13, %arg37 = %14, %arg38 = %15, %arg39 = %16, %arg40 = %17, %arg41 = %18, %arg42 = %19, %arg43 = %20, %arg44 = %21, %arg45 = %22, %arg46 = %23, %arg47 = %24, %arg48 = %25, %arg49 = %26, %arg50 = %27, %arg51 = %28, %arg52 = %29, %arg53 = %30, %arg54 = %31, %arg55 = %32, %arg56 = %33, %arg57 = %34, %arg58 = %35, %arg59 = %36, %arg60 = %37, %arg61 = %38, %arg62 = %39, %arg63 = %40, %arg64 = %41, %arg65 = %42, %arg66 = %43, %arg67 = %44, %arg68 = %53, %arg69 = %57, %arg70 = %58, %arg71 = %59, %arg72 = %60, %arg73 = %61) -> (vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x32xf16>, !xegpu.tensor_desc<16x32xf16>, vector<8x1xf32>, vector<8x1xf32>, vector<8x1xf32>, vector<8x1xf32>) {
        xegpu.nbarrier_arrive %90 : !xegpu.nbarrier
        xegpu.prefetch_nd %arg68  : !xegpu.tensor_desc<16x32xf16>
        %130 = xegpu.update_nd_offset %arg68, [%arg26, %c0] : !xegpu.tensor_desc<16x32xf16>
        xegpu.compile_hint
        xegpu.prefetch_nd %arg69  : !xegpu.tensor_desc<16x32xf16>
        %131 = xegpu.update_nd_offset %arg69, [%arg26, %c0] : !xegpu.tensor_desc<16x32xf16>
        xegpu.compile_hint
        %132 = xegpu.load_nd %arg36 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %133 = xegpu.load_nd %arg37 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %134 = xegpu.load_nd %arg38 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %135 = xegpu.load_nd %arg39 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        xegpu.compile_hint
        %136 = xegpu.update_nd_offset %arg36, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %137 = xegpu.update_nd_offset %arg37, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %138 = xegpu.update_nd_offset %arg38, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %139 = xegpu.update_nd_offset %arg39, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        xegpu.compile_hint
        %140 = xegpu.dpas %75, %132, %62 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %141 = xegpu.dpas %77, %132, %62 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %142 = xegpu.dpas %79, %133, %140 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %143 = xegpu.dpas %81, %133, %141 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %144 = xegpu.dpas %83, %134, %142 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %145 = xegpu.dpas %85, %134, %143 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %146 = xegpu.dpas %87, %135, %144 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %147 = xegpu.dpas %89, %135, %145 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        xegpu.compile_hint
        %148 = xegpu.load_nd %arg40 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %149 = xegpu.load_nd %arg41 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %150 = xegpu.load_nd %arg42 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %151 = xegpu.load_nd %arg43 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        xegpu.compile_hint
        %152 = xegpu.update_nd_offset %arg40, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %153 = xegpu.update_nd_offset %arg41, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %154 = xegpu.update_nd_offset %arg42, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %155 = xegpu.update_nd_offset %arg43, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        xegpu.compile_hint
        %156 = xegpu.dpas %75, %148, %62 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %157 = xegpu.dpas %79, %149, %156 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %158 = xegpu.dpas %83, %150, %157 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %159 = xegpu.dpas %87, %151, %158 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %160 = xegpu.dpas %77, %148, %62 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %161 = xegpu.dpas %81, %149, %160 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %162 = xegpu.dpas %85, %150, %161 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %163 = xegpu.dpas %89, %151, %162 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        xegpu.compile_hint
        %164 = xegpu.load_nd %arg44 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %165 = xegpu.load_nd %arg45 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %166 = xegpu.load_nd %arg46 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %167 = xegpu.load_nd %arg47 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        xegpu.compile_hint
        %168 = xegpu.update_nd_offset %arg44, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %169 = xegpu.update_nd_offset %arg45, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %170 = xegpu.update_nd_offset %arg46, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %171 = xegpu.update_nd_offset %arg47, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        xegpu.compile_hint
        %172 = xegpu.dpas %75, %164, %62 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %173 = xegpu.dpas %79, %165, %172 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %174 = xegpu.dpas %83, %166, %173 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %175 = xegpu.dpas %87, %167, %174 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %176 = xegpu.dpas %77, %164, %62 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %177 = xegpu.dpas %81, %165, %176 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %178 = xegpu.dpas %85, %166, %177 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %179 = xegpu.dpas %89, %167, %178 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        xegpu.compile_hint
        %180 = xegpu.load_nd %arg48 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %181 = xegpu.load_nd %arg49 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %182 = xegpu.load_nd %arg50 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %183 = xegpu.load_nd %arg51 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        xegpu.compile_hint
        %184 = xegpu.update_nd_offset %arg48, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %185 = xegpu.update_nd_offset %arg49, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %186 = xegpu.update_nd_offset %arg50, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %187 = xegpu.update_nd_offset %arg51, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        xegpu.compile_hint
        %188 = xegpu.dpas %75, %180, %62 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %189 = xegpu.dpas %79, %181, %188 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %190 = xegpu.dpas %83, %182, %189 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %191 = xegpu.dpas %87, %183, %190 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %192 = xegpu.dpas %77, %180, %62 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %193 = xegpu.dpas %81, %181, %192 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %194 = xegpu.dpas %85, %182, %193 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %195 = xegpu.dpas %89, %183, %194 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        xegpu.compile_hint
        %196 = arith.maximumf %146, %159 fastmath<nnan> : vector<8x16xf32>
        %197 = arith.maximumf %175, %191 fastmath<nnan> : vector<8x16xf32>
        %198 = arith.maximumf %196, %197 fastmath<nnan> : vector<8x16xf32>
        %199 = vector.extract_strided_slice %198 {offsets = [0, 0], sizes = [8, 8], strides = [1, 1]} : vector<8x16xf32> to vector<8x8xf32>
        %200 = vector.extract_strided_slice %198 {offsets = [0, 8], sizes = [8, 8], strides = [1, 1]} : vector<8x16xf32> to vector<8x8xf32>
        %201 = arith.maximumf %199, %200 fastmath<nnan> : vector<8x8xf32>
        %202 = vector.extract_strided_slice %201 {offsets = [0, 0], sizes = [8, 4], strides = [1, 1]} : vector<8x8xf32> to vector<8x4xf32>
        %203 = vector.extract_strided_slice %201 {offsets = [0, 4], sizes = [8, 4], strides = [1, 1]} : vector<8x8xf32> to vector<8x4xf32>
        %204 = arith.maximumf %202, %203 fastmath<nnan> : vector<8x4xf32>
        %205 = vector.extract_strided_slice %204 {offsets = [0, 0], sizes = [8, 2], strides = [1, 1]} : vector<8x4xf32> to vector<8x2xf32>
        %206 = vector.extract_strided_slice %204 {offsets = [0, 2], sizes = [8, 2], strides = [1, 1]} : vector<8x4xf32> to vector<8x2xf32>
        %207 = arith.maximumf %205, %206 fastmath<nnan> : vector<8x2xf32>
        %208 = vector.extract_strided_slice %207 {offsets = [0, 0], sizes = [8, 1], strides = [1, 1]} : vector<8x2xf32> to vector<8x1xf32>
        %209 = vector.extract_strided_slice %207 {offsets = [0, 1], sizes = [8, 1], strides = [1, 1]} : vector<8x2xf32> to vector<8x1xf32>
        %210 = arith.maximumf %208, %209 fastmath<nnan> : vector<8x1xf32>
        %211 = arith.mulf %210, %63 : vector<8x1xf32>
        %212 = arith.maximumf %211, %arg70 fastmath<nnan> : vector<8x1xf32>
        %213 = arith.mulf %146, %65 : vector<8x16xf32>
        %214 = arith.mulf %159, %65 : vector<8x16xf32>
        %215 = arith.mulf %175, %65 : vector<8x16xf32>
        %216 = arith.mulf %191, %65 : vector<8x16xf32>
        %217 = vector.shape_cast %212 : vector<8x1xf32> to vector<8xf32>
        %218 = vector.shuffle %217, %217 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7] : vector<8xf32>, vector<8xf32>
        %219 = vector.shape_cast %218 : vector<128xf32> to vector<8x16xf32>
        %220 = arith.subf %213, %219 : vector<8x16xf32>
        %221 = arith.subf %214, %219 : vector<8x16xf32>
        %222 = arith.subf %215, %219 : vector<8x16xf32>
        %223 = arith.subf %216, %219 : vector<8x16xf32>
        %224 = math.exp %220 : vector<8x16xf32>
        %225 = math.exp %221 : vector<8x16xf32>
        %226 = math.exp %222 : vector<8x16xf32>
        %227 = math.exp %223 : vector<8x16xf32>
        %228 = arith.addf %224, %225 : vector<8x16xf32>
        %229 = arith.addf %226, %227 : vector<8x16xf32>
        %230 = arith.addf %228, %229 : vector<8x16xf32>
        %231 = vector.extract_strided_slice %230 {offsets = [0, 0], sizes = [8, 8], strides = [1, 1]} : vector<8x16xf32> to vector<8x8xf32>
        %232 = vector.extract_strided_slice %230 {offsets = [0, 8], sizes = [8, 8], strides = [1, 1]} : vector<8x16xf32> to vector<8x8xf32>
        %233 = arith.addf %231, %232 : vector<8x8xf32>
        %234 = vector.extract_strided_slice %233 {offsets = [0, 0], sizes = [8, 4], strides = [1, 1]} : vector<8x8xf32> to vector<8x4xf32>
        %235 = vector.extract_strided_slice %233 {offsets = [0, 4], sizes = [8, 4], strides = [1, 1]} : vector<8x8xf32> to vector<8x4xf32>
        %236 = arith.addf %234, %235 : vector<8x4xf32>
        %237 = vector.extract_strided_slice %236 {offsets = [0, 0], sizes = [8, 2], strides = [1, 1]} : vector<8x4xf32> to vector<8x2xf32>
        %238 = vector.extract_strided_slice %236 {offsets = [0, 2], sizes = [8, 2], strides = [1, 1]} : vector<8x4xf32> to vector<8x2xf32>
        %239 = arith.addf %237, %238 : vector<8x2xf32>
        %240 = vector.extract_strided_slice %239 {offsets = [0, 0], sizes = [8, 1], strides = [1, 1]} : vector<8x2xf32> to vector<8x1xf32>
        %241 = vector.extract_strided_slice %239 {offsets = [0, 1], sizes = [8, 1], strides = [1, 1]} : vector<8x2xf32> to vector<8x1xf32>
        %242 = arith.addf %240, %241 : vector<8x1xf32>
        %243 = arith.subf %arg70, %212 : vector<8x1xf32>
        %244 = math.exp %243 : vector<8x1xf32>
        %245 = arith.mulf %arg72, %244 : vector<8x1xf32>
        %246 = arith.addf %245, %242 : vector<8x1xf32>
        %247 = vector.shape_cast %244 : vector<8x1xf32> to vector<8xf32>
        %248 = vector.shuffle %247, %247 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7] : vector<8xf32>, vector<8xf32>
        %249 = vector.shape_cast %248 : vector<128xf32> to vector<8x16xf32>
        %250 = arith.mulf %arg28, %249 : vector<8x16xf32>
        %251 = arith.mulf %arg29, %249 : vector<8x16xf32>
        %252 = arith.mulf %arg30, %249 : vector<8x16xf32>
        %253 = arith.mulf %arg31, %249 : vector<8x16xf32>
        xegpu.compile_hint
        %254 = arith.maximumf %147, %163 fastmath<nnan> : vector<8x16xf32>
        %255 = arith.maximumf %179, %195 fastmath<nnan> : vector<8x16xf32>
        %256 = arith.maximumf %254, %255 fastmath<nnan> : vector<8x16xf32>
        %257 = vector.extract_strided_slice %256 {offsets = [0, 0], sizes = [8, 8], strides = [1, 1]} : vector<8x16xf32> to vector<8x8xf32>
        %258 = vector.extract_strided_slice %256 {offsets = [0, 8], sizes = [8, 8], strides = [1, 1]} : vector<8x16xf32> to vector<8x8xf32>
        %259 = arith.maximumf %257, %258 fastmath<nnan> : vector<8x8xf32>
        %260 = vector.extract_strided_slice %259 {offsets = [0, 0], sizes = [8, 4], strides = [1, 1]} : vector<8x8xf32> to vector<8x4xf32>
        %261 = vector.extract_strided_slice %259 {offsets = [0, 4], sizes = [8, 4], strides = [1, 1]} : vector<8x8xf32> to vector<8x4xf32>
        %262 = arith.maximumf %260, %261 fastmath<nnan> : vector<8x4xf32>
        %263 = vector.extract_strided_slice %262 {offsets = [0, 0], sizes = [8, 2], strides = [1, 1]} : vector<8x4xf32> to vector<8x2xf32>
        %264 = vector.extract_strided_slice %262 {offsets = [0, 2], sizes = [8, 2], strides = [1, 1]} : vector<8x4xf32> to vector<8x2xf32>
        %265 = arith.maximumf %263, %264 fastmath<nnan> : vector<8x2xf32>
        %266 = vector.extract_strided_slice %265 {offsets = [0, 0], sizes = [8, 1], strides = [1, 1]} : vector<8x2xf32> to vector<8x1xf32>
        %267 = vector.extract_strided_slice %265 {offsets = [0, 1], sizes = [8, 1], strides = [1, 1]} : vector<8x2xf32> to vector<8x1xf32>
        %268 = arith.maximumf %266, %267 fastmath<nnan> : vector<8x1xf32>
        %269 = arith.mulf %268, %63 : vector<8x1xf32>
        %270 = arith.maximumf %269, %arg71 fastmath<nnan> : vector<8x1xf32>
        %271 = arith.mulf %147, %65 : vector<8x16xf32>
        %272 = arith.mulf %163, %65 : vector<8x16xf32>
        %273 = arith.mulf %179, %65 : vector<8x16xf32>
        %274 = arith.mulf %195, %65 : vector<8x16xf32>
        %275 = vector.shape_cast %270 : vector<8x1xf32> to vector<8xf32>
        %276 = vector.shuffle %275, %275 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7] : vector<8xf32>, vector<8xf32>
        %277 = vector.shape_cast %276 : vector<128xf32> to vector<8x16xf32>
        %278 = arith.subf %271, %277 : vector<8x16xf32>
        %279 = arith.subf %272, %277 : vector<8x16xf32>
        %280 = arith.subf %273, %277 : vector<8x16xf32>
        %281 = arith.subf %274, %277 : vector<8x16xf32>
        %282 = math.exp %278 : vector<8x16xf32>
        %283 = math.exp %279 : vector<8x16xf32>
        %284 = math.exp %280 : vector<8x16xf32>
        %285 = math.exp %281 : vector<8x16xf32>
        %286 = arith.addf %282, %283 : vector<8x16xf32>
        %287 = arith.addf %284, %285 : vector<8x16xf32>
        %288 = arith.addf %286, %287 : vector<8x16xf32>
        %289 = vector.extract_strided_slice %288 {offsets = [0, 0], sizes = [8, 8], strides = [1, 1]} : vector<8x16xf32> to vector<8x8xf32>
        %290 = vector.extract_strided_slice %288 {offsets = [0, 8], sizes = [8, 8], strides = [1, 1]} : vector<8x16xf32> to vector<8x8xf32>
        %291 = arith.addf %289, %290 : vector<8x8xf32>
        %292 = vector.extract_strided_slice %291 {offsets = [0, 0], sizes = [8, 4], strides = [1, 1]} : vector<8x8xf32> to vector<8x4xf32>
        %293 = vector.extract_strided_slice %291 {offsets = [0, 4], sizes = [8, 4], strides = [1, 1]} : vector<8x8xf32> to vector<8x4xf32>
        %294 = arith.addf %292, %293 : vector<8x4xf32>
        %295 = vector.extract_strided_slice %294 {offsets = [0, 0], sizes = [8, 2], strides = [1, 1]} : vector<8x4xf32> to vector<8x2xf32>
        %296 = vector.extract_strided_slice %294 {offsets = [0, 2], sizes = [8, 2], strides = [1, 1]} : vector<8x4xf32> to vector<8x2xf32>
        %297 = arith.addf %295, %296 : vector<8x2xf32>
        %298 = vector.extract_strided_slice %297 {offsets = [0, 0], sizes = [8, 1], strides = [1, 1]} : vector<8x2xf32> to vector<8x1xf32>
        %299 = vector.extract_strided_slice %297 {offsets = [0, 1], sizes = [8, 1], strides = [1, 1]} : vector<8x2xf32> to vector<8x1xf32>
        %300 = arith.addf %298, %299 : vector<8x1xf32>
        %301 = arith.subf %arg71, %270 : vector<8x1xf32>
        %302 = math.exp %301 : vector<8x1xf32>
        %303 = arith.mulf %arg73, %302 : vector<8x1xf32>
        %304 = arith.addf %303, %300 : vector<8x1xf32>
        %305 = vector.shape_cast %302 : vector<8x1xf32> to vector<8xf32>
        %306 = vector.shuffle %305, %305 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7] : vector<8xf32>, vector<8xf32>
        %307 = vector.shape_cast %306 : vector<128xf32> to vector<8x16xf32>
        %308 = arith.mulf %arg32, %307 : vector<8x16xf32>
        %309 = arith.mulf %arg33, %307 : vector<8x16xf32>
        %310 = arith.mulf %arg34, %307 : vector<8x16xf32>
        %311 = arith.mulf %arg35, %307 : vector<8x16xf32>
        %312 = arith.truncf %224 : vector<8x16xf32> to vector<8x16xf16>
        %313 = arith.truncf %225 : vector<8x16xf32> to vector<8x16xf16>
        %314 = arith.truncf %226 : vector<8x16xf32> to vector<8x16xf16>
        %315 = arith.truncf %227 : vector<8x16xf32> to vector<8x16xf16>
        %316 = arith.truncf %282 : vector<8x16xf32> to vector<8x16xf16>
        %317 = arith.truncf %283 : vector<8x16xf32> to vector<8x16xf16>
        %318 = arith.truncf %284 : vector<8x16xf32> to vector<8x16xf16>
        %319 = arith.truncf %285 : vector<8x16xf32> to vector<8x16xf16>
        xegpu.compile_hint
        %320 = xegpu.load_nd %arg52 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %321 = xegpu.load_nd %arg53 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %322 = xegpu.load_nd %arg54 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %323 = xegpu.load_nd %arg55 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        xegpu.compile_hint
        %324 = xegpu.update_nd_offset %arg52, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %325 = xegpu.update_nd_offset %arg53, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %326 = xegpu.update_nd_offset %arg54, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %327 = xegpu.update_nd_offset %arg55, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        xegpu.compile_hint
        xegpu.compile_hint
        %328 = xegpu.dpas %312, %320, %250 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %329 = xegpu.dpas %316, %320, %308 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %330 = xegpu.dpas %312, %321, %251 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %331 = xegpu.dpas %316, %321, %309 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %332 = xegpu.dpas %312, %322, %252 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %333 = xegpu.dpas %316, %322, %310 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %334 = xegpu.dpas %312, %323, %253 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %335 = xegpu.dpas %316, %323, %311 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        xegpu.compile_hint
        %336 = xegpu.load_nd %arg56 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %337 = xegpu.load_nd %arg57 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %338 = xegpu.load_nd %arg58 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %339 = xegpu.load_nd %arg59 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        xegpu.compile_hint
        %340 = xegpu.update_nd_offset %arg56, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %341 = xegpu.update_nd_offset %arg57, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %342 = xegpu.update_nd_offset %arg58, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %343 = xegpu.update_nd_offset %arg59, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        xegpu.compile_hint
        %344 = xegpu.dpas %313, %336, %328 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %345 = xegpu.dpas %317, %336, %329 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %346 = xegpu.dpas %313, %337, %330 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %347 = xegpu.dpas %317, %337, %331 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %348 = xegpu.dpas %313, %338, %332 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %349 = xegpu.dpas %317, %338, %333 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %350 = xegpu.dpas %313, %339, %334 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %351 = xegpu.dpas %317, %339, %335 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        xegpu.compile_hint
        %352 = xegpu.load_nd %arg60 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %353 = xegpu.load_nd %arg61 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %354 = xegpu.load_nd %arg62 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %355 = xegpu.load_nd %arg63 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        xegpu.compile_hint
        %356 = xegpu.update_nd_offset %arg60, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %357 = xegpu.update_nd_offset %arg61, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %358 = xegpu.update_nd_offset %arg62, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %359 = xegpu.update_nd_offset %arg63, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        xegpu.compile_hint
        %360 = xegpu.dpas %314, %352, %344 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %361 = xegpu.dpas %318, %352, %345 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %362 = xegpu.dpas %314, %353, %346 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %363 = xegpu.dpas %318, %353, %347 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %364 = xegpu.dpas %314, %354, %348 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %365 = xegpu.dpas %318, %354, %349 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %366 = xegpu.dpas %314, %355, %350 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %367 = xegpu.dpas %318, %355, %351 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        xegpu.compile_hint
        %368 = xegpu.load_nd %arg64 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %369 = xegpu.load_nd %arg65 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %370 = xegpu.load_nd %arg66 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %371 = xegpu.load_nd %arg67 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        xegpu.compile_hint
        %372 = xegpu.update_nd_offset %arg64, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %373 = xegpu.update_nd_offset %arg65, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %374 = xegpu.update_nd_offset %arg66, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        %375 = xegpu.update_nd_offset %arg67, [%arg26, %c0] : !xegpu.tensor_desc<16x16xf16>
        xegpu.compile_hint
        %376 = xegpu.dpas %315, %368, %360 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %377 = xegpu.dpas %319, %368, %361 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %378 = xegpu.dpas %315, %369, %362 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %379 = xegpu.dpas %319, %369, %363 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %380 = xegpu.dpas %315, %370, %364 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %381 = xegpu.dpas %319, %370, %365 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %382 = xegpu.dpas %315, %371, %366 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %383 = xegpu.dpas %319, %371, %367 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        xegpu.compile_hint
        xegpu.nbarrier_wait %90 : !xegpu.nbarrier
        scf.yield %376, %378, %380, %382, %377, %379, %381, %383, %136, %137, %138, %139, %152, %153, %154, %155, %168, %169, %170, %171, %184, %185, %186, %187, %324, %325, %326, %327, %340, %341, %342, %343, %356, %357, %358, %359, %372, %373, %374, %375, %130, %131, %212, %270, %246, %304 : vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x32xf16>, !xegpu.tensor_desc<16x32xf16>, vector<8x1xf32>, vector<8x1xf32>, vector<8x1xf32>, vector<8x1xf32>
      }
      // divide acc output by l_i
      // O tile, max size is 8x32
      %92 = vector.shape_cast %91#44 : vector<8x1xf32> to vector<8xf32>
      %93 = vector.shuffle %92, %92 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7] : vector<8xf32>, vector<8xf32>
      %94 = vector.shape_cast %93 : vector<128xf32> to vector<8x16xf32>
      %95 = vector.shape_cast %91#45 : vector<8x1xf32> to vector<8xf32>
      %96 = vector.shuffle %95, %95 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7] : vector<8xf32>, vector<8xf32>
      %97 = vector.shape_cast %96 : vector<128xf32> to vector<8x16xf32>
      %98 = arith.divf %91#0, %94 : vector<8x16xf32>
      %99 = arith.divf %91#1, %94 : vector<8x16xf32>
      %100 = arith.divf %91#2, %94 : vector<8x16xf32>
      %101 = arith.divf %91#3, %94 : vector<8x16xf32>
      %102 = arith.divf %91#4, %97 : vector<8x16xf32>
      %103 = arith.divf %91#5, %97 : vector<8x16xf32>
      %104 = arith.divf %91#6, %97 : vector<8x16xf32>
      %105 = arith.divf %91#7, %97 : vector<8x16xf32>
      %106 = arith.truncf %98 : vector<8x16xf32> to vector<8x16xf16>
      %107 = arith.truncf %99 : vector<8x16xf32> to vector<8x16xf16>
      %108 = arith.truncf %100 : vector<8x16xf32> to vector<8x16xf16>
      %109 = arith.truncf %101 : vector<8x16xf32> to vector<8x16xf16>
      %110 = arith.truncf %102 : vector<8x16xf32> to vector<8x16xf16>
      %111 = arith.truncf %103 : vector<8x16xf32> to vector<8x16xf16>
      %112 = arith.truncf %104 : vector<8x16xf32> to vector<8x16xf16>
      %113 = arith.truncf %105 : vector<8x16xf32> to vector<8x16xf16>
      %114 = xegpu.create_nd_tdesc %arg3[%8, %c0], shape : [%2, %arg25], strides : [%arg25, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<8x32xf16>
      %115 = xegpu.update_nd_offset %114, [%c0, %c32] : !xegpu.tensor_desc<8x32xf16>
      %116 = xegpu.update_nd_offset %114, [%c8, %c0] : !xegpu.tensor_desc<8x32xf16>
      %117 = xegpu.update_nd_offset %116, [%c0, %c32] : !xegpu.tensor_desc<8x32xf16>
      %118 = vector.shuffle %106, %107 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %119 = vector.shape_cast %118 : vector<16x16xf16> to vector<256xf16>
      %120 = vector.shape_cast %119 : vector<256xf16> to vector<8x32xf16>
      xegpu.store_nd %120, %114 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint
      %121 = vector.shuffle %108, %109 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %122 = vector.shape_cast %121 : vector<16x16xf16> to vector<256xf16>
      %123 = vector.shape_cast %122 : vector<256xf16> to vector<8x32xf16>
      xegpu.store_nd %123, %115 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint
      %124 = vector.shuffle %110, %111 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %125 = vector.shape_cast %124 : vector<16x16xf16> to vector<256xf16>
      %126 = vector.shape_cast %125 : vector<256xf16> to vector<8x32xf16>
      xegpu.store_nd %126, %116 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint
      %127 = vector.shuffle %112, %113 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %128 = vector.shape_cast %127 : vector<16x16xf16> to vector<256xf16>
      %129 = vector.shape_cast %128 : vector<256xf16> to vector<8x32xf16>
      xegpu.store_nd %129, %117 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint
      gpu.return
    }
  }
  func.func @gpu_impl(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?x?xf16>, %arg3: memref<?x?xf16>, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: f32) -> memref<?x?xf16> {
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c1_i64 = arith.constant 1 : i64
    // %Z_i64 = index.castu %Z : index to i64
    // %H_i64 = index.castu %H : index to i64
    // %N_CTX_i64 = index.castu %N_CTX : index to i64
    // %D_HEAD_i64 = index.castu %D_HEAD : index to i64
    //strides
    // %m_gpu = gpu.alloc host_shared (%Z, %H, %N_CTX) : memref<?x?x?xf32>
    // copy from CPU to
    // memref.copy %m, %m_gpu : memref<?x?x?xf32> to memref<?x?x?xf32>
    // GPU params
    // do a ceiling div to figure out blocks_x
    // blocks_x = (N_CTX + BLOCKS_M - 1) / BLOCKS_M
    // %blocks_x = arith.constant 32 : index
    // %BLOCK_M_i64 = index.castu %BLOCK_M : index to i64
    // %BLOCK_N_i64 = index.castu %BLOCK_N : index to i64
    // launch GPU func
    // copy output to CPU
    // gpu.dealloc %m_gpu : memref<?x?x?xf32>
    %0 = arith.muli %arg4, %arg5 : index
    %1 = arith.muli %0, %arg6 : index
    %2 = arith.muli %arg6, %arg7 : index
    %3 = arith.muli %2, %arg5 : index
    %memref = gpu.alloc  (%1, %arg7) : memref<?x?xf16>
    %memref_0 = gpu.alloc  (%1, %arg7) : memref<?x?xf16>
    %memref_1 = gpu.alloc  (%1, %arg7) : memref<?x?xf16>
    %memref_2 = gpu.alloc  (%1, %arg7) : memref<?x?xf16>
    gpu.memcpy  %memref, %arg0 : memref<?x?xf16>, memref<?x?xf16>
    gpu.memcpy  %memref_0, %arg1 : memref<?x?xf16>, memref<?x?xf16>
    gpu.memcpy  %memref_1, %arg2 : memref<?x?xf16>, memref<?x?xf16>
    gpu.memcpy  %memref_2, %arg3 : memref<?x?xf16>, memref<?x?xf16>
    %4 = index.castu %arg6 : index to i64
    %5 = index.castu %c128 : index to i64
    %6 = arith.subi %5, %c1_i64 : i64
    %7 = arith.addi %4, %6 : i64
    %8 = arith.divui %7, %5 : i64
    %9 = index.castu %8 : i64 to index
    %10 = arith.muli %arg4, %arg5 : index
    gpu.launch_func  @flash_attention_fwd::@flash_attention_fwd blocks in (%9, %10, %c1) threads in (%c8, %c1, %c1)  args(%memref : memref<?x?xf16>, %memref_0 : memref<?x?xf16>, %memref_1 : memref<?x?xf16>, %memref_2 : memref<?x?xf16>, %arg8 : f32, %3 : index, %2 : index, %arg7 : index, %c1 : index, %3 : index, %2 : index, %arg7 : index, %c1 : index, %3 : index, %2 : index, %arg7 : index, %c1 : index, %3 : index, %2 : index, %arg7 : index, %c1 : index, %arg4 : index, %arg5 : index, %arg6 : index, %c128 : index, %arg7 : index, %c64 : index)
    gpu.memcpy  %arg3, %memref_2 : memref<?x?xf16>, memref<?x?xf16>
    gpu.dealloc  %memref : memref<?x?xf16>
    gpu.dealloc  %memref_0 : memref<?x?xf16>
    gpu.dealloc  %memref_1 : memref<?x?xf16>
    gpu.dealloc  %memref_2 : memref<?x?xf16>
    return %arg3 : memref<?x?xf16>
  }
  func.func @cpu_impl(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?x?xf16>, %arg3: memref<?x?xf16>, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: f32) -> memref<?x?xf16> {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 1.44269502 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // buffer
      // reset memref
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = arith.muli %arg4, %arg5 : index
    %alloc = memref.alloc(%arg6, %arg6) : memref<?x?xf32>
    scf.for %arg9 = %c0 to %0 step %c1 {
      scf.for %arg10 = %c0 to %arg6 step %c1 {
        scf.for %arg11 = %c0 to %arg6 step %c1 {
          memref.store %cst_1, %alloc[%arg10, %arg11] : memref<?x?xf32>
        }
      }
      // compute p = q*k^T
      %1 = arith.muli %arg6, %arg9 : index
      scf.for %arg10 = %c0 to %arg6 step %c1 {
        scf.for %arg11 = %c0 to %arg6 step %c1 {
          %2 = scf.for %arg12 = %c0 to %arg7 step %c1 iter_args(%arg13 = %cst_1) -> (f32) {
            %4 = arith.addi %arg10, %1 : index
            %5 = arith.addi %arg11, %1 : index
            %6 = memref.load %arg0[%4, %arg12] : memref<?x?xf16>
            %7 = memref.load %arg1[%5, %arg12] : memref<?x?xf16>
            %8 = arith.extf %6 : f16 to f32
            %9 = arith.extf %7 : f16 to f32
            %10 = arith.mulf %8, %9 : f32
            %11 = arith.addf %arg13, %10 : f32
            scf.yield %11 : f32
          }
          %3 = arith.mulf %2, %arg8 : f32
          memref.store %3, %alloc[%arg10, %arg11] : memref<?x?xf32>
        }
      }
      // compute the softmax
        // max reduce
      scf.for %arg10 = %c0 to %arg6 step %c1 {
        %2 = scf.for %arg11 = %c0 to %arg6 step %c1 iter_args(%arg12 = %cst) -> (f32) {
          %4 = memref.load %alloc[%arg10, %arg11] : memref<?x?xf32>
          %5 = arith.maximumf %arg12, %4 : f32
          scf.yield %5 : f32
        }
        // center by max and exp
          // scale by log2e to emulate exp2
        scf.for %arg11 = %c0 to %arg6 step %c1 {
          %4 = memref.load %alloc[%arg10, %arg11] : memref<?x?xf32>
          %5 = arith.subf %4, %2 : f32
          %6 = arith.mulf %5, %cst_0 : f32
          %7 = math.exp2 %6 : f32
          memref.store %7, %alloc[%arg10, %arg11] : memref<?x?xf32>
        }
        // take sum of row
        %3 = scf.for %arg11 = %c0 to %arg6 step %c1 iter_args(%arg12 = %cst_1) -> (f32) {
          %4 = memref.load %alloc[%arg10, %arg11] : memref<?x?xf32>
          %5 = arith.addf %arg12, %4 : f32
          scf.yield %5 : f32
        }
        // div by sum
        scf.for %arg11 = %c0 to %arg6 step %c1 {
          %4 = memref.load %alloc[%arg10, %arg11] : memref<?x?xf32>
          %5 = arith.divf %4, %3 : f32
          memref.store %5, %alloc[%arg10, %arg11] : memref<?x?xf32>
        }
      }
      // compute p*v
      scf.for %arg10 = %c0 to %arg6 step %c1 {
        scf.for %arg11 = %c0 to %arg7 step %c1 {
          %2 = scf.for %arg12 = %c0 to %arg6 step %c1 iter_args(%arg13 = %cst_1) -> (f32) {
            %5 = memref.load %alloc[%arg10, %arg12] : memref<?x?xf32>
            %6 = arith.truncf %5 : f32 to f16
            %7 = arith.addi %arg12, %1 : index
            %8 = memref.load %arg2[%7, %arg11] : memref<?x?xf16>
            %9 = arith.extf %6 : f16 to f32
            %10 = arith.extf %8 : f16 to f32
            %11 = arith.mulf %9, %10 : f32
            %12 = arith.addf %11, %arg13 : f32
            scf.yield %12 : f32
          }
          %3 = arith.addi %arg10, %1 : index
          %4 = arith.truncf %2 : f32 to f16
          memref.store %4, %arg3[%3, %arg11] : memref<?x?xf16>
        }
      }
    }
    memref.dealloc %alloc : memref<?x?xf32>
    return %arg3 : memref<?x?xf16>
  }
  func.func @init_2d_dynamic_memref_to_const_f16(%arg0: memref<?x?xf16>, %arg1: index, %arg2: index, %arg3: f16) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg4 = %c0 to %arg1 step %c1 {
      scf.for %arg5 = %c0 to %arg2 step %c1 {
        memref.store %arg3, %arg0[%arg4, %arg5] : memref<?x?xf16>
      }
    }
    return
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // random number generator params
    // xegpu only supports 2d memrefs. So we collapse the first 3 dims of the inputs
    // Z x H x N_CTX x D_HEAD -> (Z * H * N_CTX) x D_HEAD
    // allocate q, k, v, o
    // FIXME : m is unused for now
    // %m = memref.alloc(%Z, %H, %N_CTX) : memref<?x?x?xf32>
    // initialize q, k, v
    // Option 1: fill with random numbers
    // call @fillResource1DRandomF16(%q_random, %rand_low, %rand_high, %gen_int) : (memref<*xf16>, f32, f32, i1) -> ()
    // call @fillResource1DRandomF16(%k_random, %rand_low, %rand_high, %gen_int) : (memref<*xf16>, f32, f32, i1) -> ()
    // call @fillResource1DRandomF16(%v_random, %rand_low, %rand_high, %gen_int) : (memref<*xf16>, f32, f32, i1) -> ()
    // Option 2: fill with some magic constant for validation
    // // initialize output to 0.0
    // %o_random = memref.collapse_shape %o [[0, 1, 2, 3]] : memref<?x?x?x?xf16> into memref<?xf16>
    // initialize m to 1.0 (FIXME : masking is not used)
    // %c1_f32 = arith.constant 1.0 : f32
    // %m_random = memref.collapse_shape %m [[0, 1, 2]] : memref<?x?x?xf32> into memref<?xf32>
    // call @fillResource1DF32(%m_random, %c1_f32) : (memref<?xf32>, f32) -> ()
    // run fused version
    // run cpu version
    // call @printMemrefF16(%q_cast) : (memref<*xf16>) -> ()
    // call @printMemrefF16(%out_cast) : (memref<*xf16>) -> ()
    // call @printMemrefF16(%out_cpu_cast) : (memref<*xf16>) -> ()
    // call @printMaxErrorF16(%out_cast, %out_cpu_cast) : (memref<*xf16>, memref<*xf16>) -> ()
    // sign extend CPU output to f32
    %cst = arith.constant 6.250000e-01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f16
    %cst_1 = arith.constant 5.000000e-01 : f32
    %c2 = arith.constant 2 : index
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c16384 = arith.constant 16384 : index
    %alloc = memref.alloc(%c16384, %c64) : memref<?x?xf16>
    %alloc_2 = memref.alloc(%c16384, %c64) : memref<?x?xf16>
    %alloc_3 = memref.alloc(%c16384, %c64) : memref<?x?xf16>
    %alloc_4 = memref.alloc(%c16384, %c64) : memref<?x?xf16>
    %alloc_5 = memref.alloc(%c16384, %c64) : memref<?x?xf16>
    %alloc_6 = memref.alloc(%c16384, %c64) : memref<?x?xf32>
    %cast = memref.cast %alloc : memref<?x?xf16> to memref<*xf16>
    %cast_7 = memref.cast %alloc_2 : memref<?x?xf16> to memref<*xf16>
    %cast_8 = memref.cast %alloc_3 : memref<?x?xf16> to memref<*xf16>
    call @fillResource1DF16(%cast, %cst) : (memref<*xf16>, f32) -> ()
    call @fillResource1DF16(%cast_7, %cst) : (memref<*xf16>, f32) -> ()
    call @fillResource1DF16(%cast_8, %cst) : (memref<*xf16>, f32) -> ()
    call @init_2d_dynamic_memref_to_const_f16(%alloc_4, %c16384, %c64, %cst_0) : (memref<?x?xf16>, index, index, f16) -> ()
    call @init_2d_dynamic_memref_to_const_f16(%alloc_5, %c16384, %c64, %cst_0) : (memref<?x?xf16>, index, index, f16) -> ()
    %0 = call @gpu_impl(%alloc, %alloc_2, %alloc_3, %alloc_4, %c2, %c2, %c4096, %c64, %cst_1) : (memref<?x?xf16>, memref<?x?xf16>, memref<?x?xf16>, memref<?x?xf16>, index, index, index, index, f32) -> memref<?x?xf16>
    %1 = call @cpu_impl(%alloc, %alloc_2, %alloc_3, %alloc_5, %c2, %c2, %c4096, %c64, %cst_1) : (memref<?x?xf16>, memref<?x?xf16>, memref<?x?xf16>, memref<?x?xf16>, index, index, index, index, f32) -> memref<?x?xf16>
    %cast_9 = memref.cast %0 : memref<?x?xf16> to memref<*xf16>
    scf.for %arg0 = %c0 to %c16384 step %c1 {
      scf.for %arg1 = %c0 to %c64 step %c1 {
        %2 = memref.load %alloc_5[%arg0, %arg1] : memref<?x?xf16>
        %3 = arith.extf %2 : f16 to f32
        memref.store %3, %alloc_6[%arg0, %arg1] : memref<?x?xf32>
      }
    }
    // CHECK: [ALLCLOSE: TRUE]
    // memref.dealloc %m : memref<?x?x?xf32>
    %cast_10 = memref.cast %alloc_6 : memref<?x?xf32> to memref<*xf32>
    call @printAllcloseF16(%cast_9, %cast_10) : (memref<*xf16>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<?x?xf16>
    memref.dealloc %alloc_2 : memref<?x?xf16>
    memref.dealloc %alloc_3 : memref<?x?xf16>
    memref.dealloc %alloc_4 : memref<?x?xf16>
    memref.dealloc %alloc_5 : memref<?x?xf16>
    memref.dealloc %alloc_6 : memref<?x?xf32>
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
