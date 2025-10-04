// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime,spirv-backend -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<4096x4096xf16>, %B: memref<4096x4096xf16>, %C: memref<4096x4096xf16>) -> memref<4096x4096xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %A_gpu= gpu.alloc () : memref<4096x4096xf16>
    gpu.memcpy %A_gpu, %A : memref<4096x4096xf16>, memref<4096x4096xf16>
    %B_gpu= gpu.alloc () : memref<4096x4096xf16>
    gpu.memcpy %B_gpu, %B : memref<4096x4096xf16>, memref<4096x4096xf16>
    %C_gpu= gpu.alloc () : memref<4096x4096xf16>
    gpu.memcpy %C_gpu, %C : memref<4096x4096xf16>, memref<4096x4096xf16>
    // NOTE: Here we can't use [8, 64] wi threads following the SG thread layout of [8, 4]. Because runtime will linearize the x dimension first (we need y dimension to be linearized first).
    // So just use linearized thread layout of [512, 1] wi threads.
    gpu.launch_func @test_kernel::@test_kernel blocks in (%c16, %c16, %c1) threads in (%c512, %c1, %c1) args(%A_gpu : memref<4096x4096xf16>, %B_gpu : memref<4096x4096xf16>, %C_gpu : memref<4096x4096xf16>)
    gpu.wait // Wait for the kernel to finish.
    gpu.memcpy %C, %C_gpu : memref<4096x4096xf16>, memref<4096x4096xf16>
    gpu.dealloc %A_gpu : memref<4096x4096xf16>
    gpu.dealloc %B_gpu : memref<4096x4096xf16>
    gpu.dealloc %C_gpu : memref<4096x4096xf16>
    return %C : memref<4096x4096xf16>
  }

  gpu.module @test_kernel   {
    gpu.func @test_kernel(%A: memref<4096x4096xf16>, %B: memref<4096x4096xf16>, %C: memref<4096x4096xf16>) kernel  {
      // constants
      %c256 = arith.constant 256 : index
      %c512 = arith.constant 512 : index
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %c2048 = arith.constant 2048 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c48 = arith.constant 48 : index
      %c16 = arith.constant 16 : index
      %c24 = arith.constant 24 : index
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      // get IDs
      %wg_id_x = gpu.block_id x
      %wg_id_y = gpu.block_id y

      // each C wg tile is 256x256 and 32 SGs update it in 8x4 layout
      // C sg tile size is 32x64
      // SG layout for one C tile update
      // |0|1|2|3|
      // |4|5|6|7|
      // .........
      // |28|29|30|31|
      // --> y means cols
      // |
      // V x means rows

      // get unique sg ID in global context
      %sg_id = gpu.subgroup_id : index
      %thread_id = gpu.thread_id x
      // Compute the x and y thread ID assuming a [8, 64] wi layout.
      %thread_id_x = arith.divui %thread_id, %c64 : index
      %thread_id_y = arith.remui %thread_id, %c64 : index

      %local_sg_id_x = arith.divui %thread_id, %c64 : index
      %local_sg_id_y = arith.divui %thread_id_y, %c16 : index

      // compute SG C tile offsets in x and y dims
      %C_sg_tile_offset_x_t0 = arith.muli %wg_id_x, %c256 : index
      %C_sg_tile_offset_x_t1 = arith.muli %local_sg_id_x, %c32 : index
      %C_sg_tile_offset_x = arith.addi %C_sg_tile_offset_x_t0, %C_sg_tile_offset_x_t1 : index
      %C_sg_tile_offset_y_t0 = arith.muli %wg_id_y, %c256 : index
      %C_sg_tile_offset_y_t1 = arith.muli %local_sg_id_y, %c64 : index
      %C_sg_tile_offset_y = arith.addi %C_sg_tile_offset_y_t0, %C_sg_tile_offset_y_t1 : index


      // each SG needs to do the follwoing compute to update its 32x64 sub tile
      // (32x4096)x(4096x64)=(32x64)
      // DPAS size is (8x16)x(16x16)=(8x16)
      // K loop adavances in steps of 32, so inside the compute is (32x32)x(32x64) = (32x64)
      // So we need to (4x2) A tiles of size (8x16) and (2x4) B tiles of size (16x16)
      // tiled compute for a SG is (4x2x8x16)x(2x4x16x16)=(4x4x8x16)
      // this will require 32 DPAS ops (4x2x2) inside the K loop

      // WG tiles are 256x256 so there offsets are same for A, B and C
      %wg_tile_offset_x = arith.muli %wg_id_x, %c256 : index
      %wg_tile_offset_y = arith.muli %wg_id_y, %c256 : index

      %local_sg_id_temp = arith.muli %local_sg_id_x, %c4 : index
      %local_sg_id = arith.addi %local_sg_id_temp, %local_sg_id_y : index

      // TODO: Add prefetch support. B prefetch requires sightly different prefetch tile calculation.

      // two 32x16 A tiles from 256x32 WG slice
      %A_tile_0 = xegpu.create_nd_tdesc %A : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
      // create B tiles considering the transposed B matrix.
      // 1. First view the B matrix as 4096x2048xf32 because we need to transpose it in 32 bits.
      // 2. Then create 16x8 B tiles from the 4096x2048 view. Note that we can not use array length > 1 and max size supported for 32 bitwidth is 16x8.
      %B_ptr_index = memref.extract_aligned_pointer_as_index %B : memref<4096x4096xf16> -> index
      %b_ptr_i64 = arith.index_cast %B_ptr_index : index to i64

      %B_tile = xegpu.create_nd_tdesc %b_ptr_i64, shape: [%c4096, %c2048], strides: [%c2048, %c1] : i64 -> !xegpu.tensor_desc<16x8xf32, #xegpu.block_tdesc_attr<array_length = 1>>
      %C_sg_tile_offset_y_plus_16_t0 = arith.addi %C_sg_tile_offset_y, %c16 : index
      %C_sg_tile_offset_y_plus_32_t0 = arith.addi %C_sg_tile_offset_y_plus_16_t0, %c16 : index
      %C_sg_tile_offset_y_plus_48_t0 = arith.addi %C_sg_tile_offset_y_plus_32_t0, %c16 : index

      // init 16 C tiles of size 8x16 each is initialized to 0.0 assuming a zero C matrix
      %c_init_val_0_0 = arith.constant dense<0.0> : vector<8x16xf32>
      %c_init_val_0_1 = arith.constant dense<0.0> : vector<8x16xf32>
      %c_init_val_0_2 = arith.constant dense<0.0> : vector<8x16xf32>
      %c_init_val_0_3 = arith.constant dense<0.0> : vector<8x16xf32>
      %c_init_val_1_0 = arith.constant dense<0.0> : vector<8x16xf32>
      %c_init_val_1_1 = arith.constant dense<0.0> : vector<8x16xf32>
      %c_init_val_1_2 = arith.constant dense<0.0> : vector<8x16xf32>
      %c_init_val_1_3 = arith.constant dense<0.0> : vector<8x16xf32>
      %c_init_val_2_0 = arith.constant dense<0.0> : vector<8x16xf32>
      %c_init_val_2_1 = arith.constant dense<0.0> : vector<8x16xf32>
      %c_init_val_2_2 = arith.constant dense<0.0> : vector<8x16xf32>
      %c_init_val_2_3 = arith.constant dense<0.0> : vector<8x16xf32>
      %c_init_val_3_0 = arith.constant dense<0.0> : vector<8x16xf32>
      %c_init_val_3_1 = arith.constant dense<0.0> : vector<8x16xf32>
      %c_init_val_3_2 = arith.constant dense<0.0> : vector<8x16xf32>
      %c_init_val_3_3 = arith.constant dense<0.0> : vector<8x16xf32>

      %k_loop_result:16 = scf.for %k = %c0 to %c4096 step %c32 iter_args (
          %c_val_0_0 = %c_init_val_0_0,
          %c_val_0_1 = %c_init_val_0_1,
          %c_val_0_2 = %c_init_val_0_2,
          %c_val_0_3 = %c_init_val_0_3,
          %c_val_1_0 = %c_init_val_1_0,
          %c_val_1_1 = %c_init_val_1_1,
          %c_val_1_2 = %c_init_val_1_2,
          %c_val_1_3 = %c_init_val_1_3,
          %c_val_2_0 = %c_init_val_2_0,
          %c_val_2_1 = %c_init_val_2_1,
          %c_val_2_2 = %c_init_val_2_2,
          %c_val_2_3 = %c_init_val_2_3,
          %c_val_3_0 = %c_init_val_3_0,
          %c_val_3_1 = %c_init_val_3_1,
          %c_val_3_2 = %c_init_val_3_2,
          %c_val_3_3 = %c_init_val_3_3
          ) ->
          (vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>)
          {
        // sync all threads
        gpu.barrier
        // load A tiles
        %a_val = xegpu.load_nd %A_tile_0[%C_sg_tile_offset_x, %k]  {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>> -> vector<2x32x16xf16>
        // %a_val = vector.shape_cast %a_val_t0 : vector<64xf16> to vector<2x32xf16>
        %a_val_0 = vector.extract %a_val [0] : vector<32x16xf16> from vector<2x32x16xf16>
        %a_val_1 = vector.extract %a_val [1] : vector<32x16xf16> from vector<2x32x16xf16>

        // load B tiles (transposed view)
        %k_div_2 = index.shru %k, %c1 // B tile moves by 16 because of f32 cast.
        %k_div_2_plus_8 = arith.addi %k_div_2, %c8 : index
        %b_val_0_0_t0 = xegpu.load_nd %B_tile[%C_sg_tile_offset_y, %k_div_2] { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32, #xegpu.block_tdesc_attr<array_length = 1>> -> vector<16x8xf32>
        %b_val_1_0_t0 = xegpu.load_nd %B_tile[%C_sg_tile_offset_y, %k_div_2_plus_8] { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32, #xegpu.block_tdesc_attr<array_length = 1>> -> vector<16x8xf32>
        %b_val_0_1_t0 = xegpu.load_nd %B_tile[%C_sg_tile_offset_y_plus_16_t0, %k_div_2] { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32, #xegpu.block_tdesc_attr<array_length = 1>> -> vector<16x8xf32>
        %b_val_1_1_t0 = xegpu.load_nd %B_tile[%C_sg_tile_offset_y_plus_16_t0, %k_div_2_plus_8] { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32, #xegpu.block_tdesc_attr<array_length = 1>> -> vector<16x8xf32>
        %b_val_0_2_t0 = xegpu.load_nd %B_tile[%C_sg_tile_offset_y_plus_32_t0, %k_div_2] { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32, #xegpu.block_tdesc_attr<array_length = 1>> -> vector<16x8xf32>
        %b_val_1_2_t0 = xegpu.load_nd %B_tile[%C_sg_tile_offset_y_plus_32_t0, %k_div_2_plus_8] { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32, #xegpu.block_tdesc_attr<array_length = 1>> -> vector<16x8xf32>
        %b_val_0_3_t0 = xegpu.load_nd %B_tile[%C_sg_tile_offset_y_plus_48_t0, %k_div_2] { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32, #xegpu.block_tdesc_attr<array_length = 1>> -> vector<16x8xf32>
        %b_val_1_3_t0 = xegpu.load_nd %B_tile[%C_sg_tile_offset_y_plus_48_t0, %k_div_2_plus_8] { l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x8xf32, #xegpu.block_tdesc_attr<array_length = 1>> -> vector<16x8xf32>

        // Bitcast B data slices to f16.
        %b_val_0_0_bitcast = vector.bitcast %b_val_0_0_t0 : vector<16x8xf32> to vector<16x16xf16>
        %b_val_1_0_bitcast = vector.bitcast %b_val_1_0_t0 : vector<16x8xf32> to vector<16x16xf16>
        %b_val_0_1_bitcast = vector.bitcast %b_val_0_1_t0 : vector<16x8xf32> to vector<16x16xf16>
        %b_val_1_1_bitcast = vector.bitcast %b_val_1_1_t0 : vector<16x8xf32> to vector<16x16xf16>
        %b_val_0_2_bitcast = vector.bitcast %b_val_0_2_t0 : vector<16x8xf32> to vector<16x16xf16>
        %b_val_1_2_bitcast = vector.bitcast %b_val_1_2_t0 : vector<16x8xf32> to vector<16x16xf16>
        %b_val_0_3_bitcast = vector.bitcast %b_val_0_3_t0 : vector<16x8xf32> to vector<16x16xf16>
        %b_val_1_3_bitcast = vector.bitcast %b_val_1_3_t0 : vector<16x8xf32> to vector<16x16xf16>

        // Transpose B slices.
        %b_val_0_0 = vector.transpose %b_val_0_0_bitcast, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
        %b_val_1_0 = vector.transpose %b_val_1_0_bitcast, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
        %b_val_0_1 = vector.transpose %b_val_0_1_bitcast, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
        %b_val_1_1 = vector.transpose %b_val_1_1_bitcast, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
        %b_val_0_2 = vector.transpose %b_val_0_2_bitcast, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
        %b_val_1_2 = vector.transpose %b_val_1_2_bitcast, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
        %b_val_0_3 = vector.transpose %b_val_0_3_bitcast, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
        %b_val_1_3 = vector.transpose %b_val_1_3_bitcast, [1, 0] : vector<16x16xf16> to vector<16x16xf16>

        %a_val_0_0 = vector.extract_strided_slice %a_val_0 { offsets = [0], sizes = [8], strides = [1]} :
          vector<32x16xf16> to vector<8x16xf16>
        %a_val_1_0 = vector.extract_strided_slice %a_val_0 { offsets = [8], sizes = [8], strides = [1]} :
          vector<32x16xf16> to vector<8x16xf16>
        %a_val_2_0 = vector.extract_strided_slice  %a_val_0 { offsets = [16], sizes = [8], strides = [1]} :
          vector<32x16xf16> to vector<8x16xf16>
        %a_val_3_0 = vector.extract_strided_slice %a_val_0 { offsets = [24], sizes = [8], strides = [1]} :
          vector<32x16xf16> to vector<8x16xf16>
        %a_val_0_1 = vector.extract_strided_slice %a_val_1 { offsets = [0], sizes = [8], strides = [1]} :
          vector<32x16xf16> to vector<8x16xf16>
        %a_val_1_1 = vector.extract_strided_slice %a_val_1 {offsets = [8], sizes = [8], strides = [1]} :
          vector<32x16xf16> to vector<8x16xf16>
        %a_val_2_1 = vector.extract_strided_slice %a_val_1 { offsets = [16], sizes = [8], strides = [1]} :
          vector<32x16xf16> to vector<8x16xf16>
        %a_val_3_1 = vector.extract_strided_slice %a_val_1 { offsets = [24], sizes = [8], strides = [1]} :
          vector<32x16xf16> to vector<8x16xf16>

        // do DPAS
        %new_c_val_0_0_temp = xegpu.dpas %a_val_0_0, %b_val_0_0, %c_val_0_0 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_0 = xegpu.dpas %a_val_0_1, %b_val_1_0, %new_c_val_0_0_temp : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_0_temp = xegpu.dpas %a_val_1_0, %b_val_0_0, %c_val_1_0 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_0 = xegpu.dpas %a_val_1_1, %b_val_1_0, %new_c_val_1_0_temp : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_0_temp = xegpu.dpas %a_val_2_0, %b_val_0_0, %c_val_2_0 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_0 = xegpu.dpas %a_val_2_1, %b_val_1_0, %new_c_val_2_0_temp : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_0_temp = xegpu.dpas %a_val_3_0, %b_val_0_0, %c_val_3_0 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_0 = xegpu.dpas %a_val_3_1, %b_val_1_0, %new_c_val_3_0_temp : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>

        %new_c_val_0_1_temp = xegpu.dpas %a_val_0_0, %b_val_0_1, %c_val_0_1 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_1 = xegpu.dpas %a_val_0_1, %b_val_1_1, %new_c_val_0_1_temp : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_1_temp = xegpu.dpas %a_val_1_0, %b_val_0_1, %c_val_1_1 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_1 = xegpu.dpas %a_val_1_1, %b_val_1_1, %new_c_val_1_1_temp : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_1_temp = xegpu.dpas %a_val_2_0, %b_val_0_1, %c_val_2_1 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_1 = xegpu.dpas %a_val_2_1, %b_val_1_1, %new_c_val_2_1_temp : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_1_temp = xegpu.dpas %a_val_3_0, %b_val_0_1, %c_val_3_1 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_1 = xegpu.dpas %a_val_3_1, %b_val_1_1, %new_c_val_3_1_temp : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>

        %new_c_val_0_2_temp = xegpu.dpas %a_val_0_0, %b_val_0_2, %c_val_0_2 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_2 = xegpu.dpas %a_val_0_1, %b_val_1_2, %new_c_val_0_2_temp : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_2_temp = xegpu.dpas %a_val_1_0, %b_val_0_2, %c_val_1_2 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_2 = xegpu.dpas %a_val_1_1, %b_val_1_2, %new_c_val_1_2_temp : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_2_temp = xegpu.dpas %a_val_2_0, %b_val_0_2, %c_val_2_2 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_2 = xegpu.dpas %a_val_2_1, %b_val_1_2, %new_c_val_2_2_temp : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_2_temp = xegpu.dpas %a_val_3_0, %b_val_0_2, %c_val_3_2 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_2 = xegpu.dpas %a_val_3_1, %b_val_1_2, %new_c_val_3_2_temp : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>

        %new_c_val_0_3_temp = xegpu.dpas %a_val_0_0, %b_val_0_3, %c_val_0_3 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_3 = xegpu.dpas %a_val_0_1, %b_val_1_3, %new_c_val_0_3_temp : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_3_temp = xegpu.dpas %a_val_1_0, %b_val_0_3, %c_val_1_3 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_3 = xegpu.dpas %a_val_1_1, %b_val_1_3, %new_c_val_1_3_temp : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_3_temp = xegpu.dpas %a_val_2_0, %b_val_0_3, %c_val_2_3 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_3 = xegpu.dpas %a_val_2_1, %b_val_1_3, %new_c_val_2_3_temp : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_3_temp = xegpu.dpas %a_val_3_0, %b_val_0_3, %c_val_3_3 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_3 = xegpu.dpas %a_val_3_1, %b_val_1_3, %new_c_val_3_3_temp : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        scf.yield %new_c_val_0_0, %new_c_val_0_1, %new_c_val_0_2, %new_c_val_0_3, %new_c_val_1_0, %new_c_val_1_1, %new_c_val_1_2, %new_c_val_1_3, %new_c_val_2_0, %new_c_val_2_1, %new_c_val_2_2, %new_c_val_2_3, %new_c_val_3_0, %new_c_val_3_1, %new_c_val_3_2, %new_c_val_3_3
                  :
                  vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>
      }

      // trunc to f16
      %c_result_0_0_f16 = arith.truncf %k_loop_result#0 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_0_1_f16 = arith.truncf %k_loop_result#1 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_0_2_f16 = arith.truncf %k_loop_result#2 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_0_3_f16 = arith.truncf %k_loop_result#3 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_0_f16 = arith.truncf %k_loop_result#4 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_1_f16 = arith.truncf %k_loop_result#5 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_2_f16 = arith.truncf %k_loop_result#6 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_3_f16 = arith.truncf %k_loop_result#7 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_0_f16 = arith.truncf %k_loop_result#8 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_1_f16 = arith.truncf %k_loop_result#9 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_2_f16 = arith.truncf %k_loop_result#10 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_3_f16 = arith.truncf %k_loop_result#11 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_0_f16 = arith.truncf %k_loop_result#12 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_1_f16 = arith.truncf %k_loop_result#13 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_2_f16 = arith.truncf %k_loop_result#14 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_3_f16 = arith.truncf %k_loop_result#15 : vector<8x16xf32> to vector<8x16xf16>

      // each SG needs to write to 32x64 C tile.
      // DPAS output size is 8x16. So each SG needs to write 16 (4x4) DPAS outputs.
      // create 16 address descriptions to cover 8x16 tiles in 4x4 layout within the 32x64 SG C tile.
      // advance 8 in x dim and, advance 16 in y dim
      // row 1
      %C_tile = xegpu.create_nd_tdesc %C: memref<4096x4096xf16> -> !xegpu.tensor_desc<8x16xf16>
      %C_sg_tile_offset_y_plus_16 = arith.addi %C_sg_tile_offset_y, %c16 : index
      %C_sg_tile_offset_y_plus_32 = arith.addi %C_sg_tile_offset_y_plus_16, %c16 : index
      %C_sg_tile_offset_y_plus_48 = arith.addi %C_sg_tile_offset_y_plus_32, %c16 : index
      %C_sg_tile_offset_x_plus_8 = arith.addi %C_sg_tile_offset_x, %c8 : index
      %C_sg_tile_offset_x_plus_16 = arith.addi %C_sg_tile_offset_x_plus_8, %c8 : index
      %C_sg_tile_offset_x_plus_24 = arith.addi %C_sg_tile_offset_x_plus_16, %c8 : index

      // do store_nd
      xegpu.store_nd %c_result_0_0_f16, %C_tile[%C_sg_tile_offset_x, %C_sg_tile_offset_y] {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_0_1_f16, %C_tile[%C_sg_tile_offset_x, %C_sg_tile_offset_y_plus_16] {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_0_2_f16, %C_tile[%C_sg_tile_offset_x, %C_sg_tile_offset_y_plus_32] {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_0_3_f16, %C_tile[%C_sg_tile_offset_x, %C_sg_tile_offset_y_plus_48] {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_1_0_f16, %C_tile[%C_sg_tile_offset_x_plus_8, %C_sg_tile_offset_y] {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_1_1_f16, %C_tile[%C_sg_tile_offset_x_plus_8, %C_sg_tile_offset_y_plus_16] {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_1_2_f16, %C_tile[%C_sg_tile_offset_x_plus_8, %C_sg_tile_offset_y_plus_32] {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_1_3_f16, %C_tile[%C_sg_tile_offset_x_plus_8, %C_sg_tile_offset_y_plus_48] {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_2_0_f16, %C_tile[%C_sg_tile_offset_x_plus_16, %C_sg_tile_offset_y] {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_2_1_f16, %C_tile[%C_sg_tile_offset_x_plus_16, %C_sg_tile_offset_y_plus_16] {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_2_2_f16, %C_tile[%C_sg_tile_offset_x_plus_16, %C_sg_tile_offset_y_plus_32] {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_2_3_f16, %C_tile[%C_sg_tile_offset_x_plus_16, %C_sg_tile_offset_y_plus_48] {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_3_0_f16, %C_tile[%C_sg_tile_offset_x_plus_24, %C_sg_tile_offset_y] {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_3_1_f16, %C_tile[%C_sg_tile_offset_x_plus_24, %C_sg_tile_offset_y_plus_16] {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_3_2_f16, %C_tile[%C_sg_tile_offset_x_plus_24, %C_sg_tile_offset_y_plus_32] {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_3_3_f16, %C_tile[%C_sg_tile_offset_x_plus_24, %C_sg_tile_offset_y_plus_48] {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
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
    %C = memref.alloc() : memref<4096x4096xf16>
    %C_ref = memref.alloc() : memref<4096x4096xf32>
    %c_gen_int = arith.constant 0 : i1
    %cf_lower = arith.constant -0.5 : f32
    %cf_upper = arith.constant 0.5 : f32
    // Use one of the two options to initialize the A matrix
    // Option 1: intialize matrix A ; A[i, j] = j
    // scf.for %i = %c0 to %c4096 step %c1 {
    //   scf.for %j = %c0 to %c4096 step %c1 {
    //     %t = index.castu %j : index to i16
    //     %val = arith.uitofp %t : i16 to f16
    //     memref.store %val, %A[%i, %j] : memref<4096x4096xf16>
    //     // memref.store %c1_f16, %A[%i, %j] : memref<4096x4096xf16>
    //     // memref.store %c2_f16, %B[%i, %j] : memref<4096x4096xf16>
    //   }
    // }
    // Option 2:  convert the memref to 1D and fill with random values in (-0.5, 0.5)
    %A_random = memref.cast %A : memref<4096x4096xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%A_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf16>, f32, f32, i1) -> ()


    // Use one of the two options below to initialize the B matrix
    // Option 1: make matrix B an identity matrix
    // scf.for %i = %c0 to %c4096 step %c1 {
    //   scf.for %j = %c0 to %c4096 step %c1 {
    //     %i_i32 = index.castu %i : index to i32
    //     %j_i32 = index.castu %j : index to i32
    //     %i_j_same = arith.cmpi eq, %i_i32, %j_i32 : i32

    //     scf.if %i_j_same {
    //       memref.store %cf_1, %B[%i, %j] : memref<4096x4096xf16>
    //     } else {
    //       memref.store %cf_0, %B[%i, %j] : memref<4096x4096xf16>
    //     }
    //   }
    // }
    // Option 2:  convert the memref to 1D and fill with random values in (-0.5, 0.5)
    %B_random = memref.cast %B : memref<4096x4096xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%B_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf16>, f32, f32, i1) -> ()


    // Initialize matrix C and C_ref ; C[i, j] = 0
    %c0_f16 = arith.constant 0.0 : f16
    %c0_f32 = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c4096 step %c1 {
      scf.for %j = %c0 to %c4096 step %c1 {
        memref.store %c0_f16, %C[%i, %j] : memref<4096x4096xf16>
        memref.store %c0_f32, %C_ref[%i, %j] : memref<4096x4096xf32>
      }
    }

    // Run GPU version.
    %2 = call @test(%A, %B, %C) : (memref<4096x4096xf16>, memref<4096x4096xf16>, memref<4096x4096xf16>) -> memref<4096x4096xf16>
    %gpu_result_cast = memref.cast %2 : memref<4096x4096xf16> to memref<*xf16>

    // Run CPU version.
    // Construct a non transposed version of B for validating the results using imex runtime calls.
    %B_non_tranposed = memref.alloc() : memref<4096x4096xf16>
    scf.for %i = %c0 to %c4096 step %c1 {
      scf.for %j = %c0 to %c4096 step %c1 {
        %b_val = memref.load %B[%j, %i] : memref<4096x4096xf16>
        memref.store %b_val, %B_non_tranposed[%i, %j] : memref<4096x4096xf16>
      }
    }
    %A_cast = memref.cast %A : memref<4096x4096xf16> to memref<*xf16>
    %B_cast = memref.cast %B_non_tranposed : memref<4096x4096xf16> to memref<*xf16>
    %C_cast = memref.cast %C_ref : memref<4096x4096xf32> to memref<*xf32>
    call @gemmF16F16F16(%A_cast, %B_cast, %C_cast) : (memref<*xf16>, memref<*xf16>, memref<*xf32>) -> ()

    %C_row_0 = memref.subview %C_ref[0, 0][1, 4096][1, 1] : memref<4096x4096xf32> to memref<1x4096xf32, strided<[4096, 1], offset:0>>
    %C_row_0_cast = memref.cast %C_row_0 : memref<1x4096xf32, strided<[4096, 1], offset: 0>> to memref<*xf32>
    // call @printMemrefF32(%C_row_0_cast) : (memref<*xf32>) -> ()

    %C_row_0_gpu  = memref.subview %2[0, 0][1, 4096][1, 1] : memref<4096x4096xf16> to memref<1x4096xf16, strided<[4096, 1], offset:0>>
    %C_row_0_cast_gpu = memref.cast %C_row_0_gpu : memref<1x4096xf16, strided<[4096, 1], offset: 0>> to memref<*xf16>
    // call @printMemrefF16(%C_row_0_cast_gpu) : (memref<*xf16>) -> ()

    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF16(%gpu_result_cast, %C_cast) : (memref<*xf16>, memref<*xf32>) -> ()
    memref.dealloc %A : memref<4096x4096xf16>
    memref.dealloc %B : memref<4096x4096xf16>
    memref.dealloc %C : memref<4096x4096xf16>
    memref.dealloc %C_ref : memref<4096x4096xf32>
    memref.dealloc %B_non_tranposed : memref<4096x4096xf16>
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMaxErrorF16(memref<*xf16>, memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @gemmF16F16F16(memref<*xf16>, memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}

}
