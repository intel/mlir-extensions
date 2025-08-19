// RUN: IMEX_ENABLE_LARGE_REG_FILE=1 %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: IMEX_ENABLE_LARGE_REG_FILE=1 %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
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
    %A_gpu = gpu.alloc  host_shared () : memref<4096x4096xf16>
    memref.copy %A, %A_gpu : memref<4096x4096xf16> to memref<4096x4096xf16>
    %B_gpu = gpu.alloc  host_shared () : memref<4096x4096xf16>
    memref.copy %B, %B_gpu : memref<4096x4096xf16> to memref<4096x4096xf16>
    %C_gpu = gpu.alloc  host_shared () : memref<4096x4096xf16>
    memref.copy %C, %C_gpu : memref<4096x4096xf16> to memref<4096x4096xf16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c16, %c16, %c1) threads in (%c8, %c4, %c1) args(%A_gpu : memref<4096x4096xf16>, %B_gpu : memref<4096x4096xf16>, %C_gpu : memref<4096x4096xf16>)
    gpu.dealloc  %A_gpu : memref<4096x4096xf16>
    gpu.dealloc  %B_gpu : memref<4096x4096xf16>
    return %C_gpu : memref<4096x4096xf16>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%A: memref<4096x4096xf16>, %B: memref<4096x4096xf16>, %C: memref<4096x4096xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // constants
      %c256 = arith.constant 256 : index
      %c512 = arith.constant 512 : index
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
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
      // %sg_id = gpu.subgroup_id : index

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
      %global_sg_id_x = gpu.global_id x
      %global_sg_id_y = gpu.global_id y
      %local_sg_id_x = arith.remui %global_sg_id_x, %c8 : index
      %local_sg_id_y = arith.remui %global_sg_id_y, %c4 : index

      // compute SG C tile offsets in x and y dims
      %C_sg_tile_offset_x = arith.muli %global_sg_id_x, %c32 : index
      %C_sg_tile_offset_y = arith.muli %global_sg_id_y, %c64 : index

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

      // prefetching A and B slice within the 256x256 WG tile
      //
      // prefetch the entire 256x32 slice of A WG tile, this means each subgroups needs to prefetch 8x32 slice
      // each 1x4 row of SGs do a colloborative prefetch of 8x32 slice of the 32x32 tile
      // SG 0 -> slice 0 |
      // SG 1 -> slice 1 |
      // SG 2 -> slice 2  > SG 0,1,2,3 share data prefetch from the top 32x32 tile.
      // SG 3 -> slice 3 |
      // SG 4 -> slice 4
      // ....
      // SG 31 -> slice 31
      %A_sg_prefetch_offset_x_temp = arith.muli %local_sg_id, %c8 : index
      %A_sg_prefetch_offset_x = arith.addi %A_sg_prefetch_offset_x_temp, %wg_tile_offset_x : index
      // create A preftech tiles and prefetch
      // stage 1
      %A_sg_prefetch_tile_iter0 = xegpu.create_nd_tdesc %A[%A_sg_prefetch_offset_x, %c0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<8x32xf16>
      xegpu.prefetch_nd %A_sg_prefetch_tile_iter0 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
      // stage 2 (move 32 elements in the y direction and prefetch next 8x32 tile)
      %A_sg_prefetch_tile_iter1 = xegpu.update_nd_offset %A_sg_prefetch_tile_iter0, [%c0, %c32] : !xegpu.tensor_desc<8x32xf16>
      xegpu.prefetch_nd %A_sg_prefetch_tile_iter1 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
      // stage 3
      %A_sg_prefetch_tile_iter2 = xegpu.update_nd_offset %A_sg_prefetch_tile_iter1, [%c0, %c32] : !xegpu.tensor_desc<8x32xf16>
      xegpu.prefetch_nd %A_sg_prefetch_tile_iter2 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
      // compute the next tile to prefetch within K loop
      %A_sg_prefetch_tile_iter3 = xegpu.update_nd_offset %A_sg_prefetch_tile_iter2, [%c0, %c32] : !xegpu.tensor_desc<8x32xf16>

      // ---- Simpler prefetch scheme for B prefetch ----
      // Original SG layout is 8x4. And we need to prefetch 32x256 slice of B. Best prefetch size for the data type is
      // is 8x32. This makes the prefetch layout 4x8. To avoid complex prefetching address calculation, we convert the
      // SG layout to 4x8 from original 8x4 and assign the SGs to prefetch slices in a round robin fashion.
      // This approach results in the following SG to prefetch slice mapping:
      // | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
      // | 8 | 9 | 10| 11| 12| 13| 14| 15|
      // | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
      // | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 |

      // calculate the linear index of the SG
      %linear_sg_id_t0 = arith.muli %local_sg_id_x, %c4 : index
      %linear_sg_id = arith.muli %linear_sg_id_t0, %local_sg_id_y : index
      // convert layout to 4x8 from 8x4
      %sg_id_4x8_x = arith.divui %linear_sg_id, %c8 : index
      %sg_id_4x8_y = arith.remui %linear_sg_id, %c8 : index
      // compute address for 8x32 slice
      %B_sg_prefetch_offset_x = arith.muli %sg_id_4x8_x, %c8 : index
      %B_sg_prefetch_offset_y_t0 = arith.muli %sg_id_4x8_y, %c32 : index
      %B_sg_prefetch_offset_y = arith.addi %wg_tile_offset_y, %B_sg_prefetch_offset_y_t0 : index

      // create B prefetch tiles and prefetch
      %B_sg_prefetch_tile_iter0 = xegpu.create_nd_tdesc %B[%B_sg_prefetch_offset_x, %B_sg_prefetch_offset_y] : memref<4096x4096xf16> -> !xegpu.tensor_desc<8x32xf16>
      xegpu.prefetch_nd %B_sg_prefetch_tile_iter0 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
      // stage 2 (move 32 elements in the x direction and prefetch next 8x32 tile)
      %B_sg_prefetch_tile_iter1 = xegpu.update_nd_offset %B_sg_prefetch_tile_iter0, [%c32, %c0] : !xegpu.tensor_desc<8x32xf16>
      xegpu.prefetch_nd %B_sg_prefetch_tile_iter1 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
      // stage 3
      %B_sg_prefetch_tile_iter2 = xegpu.update_nd_offset %B_sg_prefetch_tile_iter1, [%c32, %c0] : !xegpu.tensor_desc<8x32xf16>
      xegpu.prefetch_nd %B_sg_prefetch_tile_iter2 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
      // compute the next tile to prefetch inside K loop
      %B_sg_prefetch_tile_iter3 = xegpu.update_nd_offset %B_sg_prefetch_tile_iter2, [%c32, %c0] : !xegpu.tensor_desc<8x32xf16>


      // two 32x16 A tiles from 256x32 WG slice
      %A_sg_init_tile_0 = xegpu.create_nd_tdesc %A[%C_sg_tile_offset_x, %c0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length =2>>
      // %A_sg_init_tile_1 = xegpu.create_nd_tdesc %A[%C_sg_tile_offset_x, %c16] : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x16xf16>

      //create B tiles
      %B_sg_init_tile_0 = xegpu.create_nd_tdesc %B[%c0, %C_sg_tile_offset_y] : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
      %B_sg_init_tile_1 = xegpu.update_nd_offset %B_sg_init_tile_0, [%c0, %c32] :  !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr< array_length = 2>>
      // %B_sg_init_tile_2 = xegpu.update_nd_offset %B_sg_init_tile_1, [%c0, %c16] :  !xegpu.tensor_desc<32x16xf16>
      // %B_sg_init_tile_3 = xegpu.update_nd_offset %B_sg_init_tile_2, [%c0, %c16] :  !xegpu.tensor_desc<32x16xf16>

      // init 16 C tiles of size 8x16 each is initialized to 0.0 assuming a zero C matrix
      %zero_vec = arith.constant dense<0.0> : vector<128xf32>
      %c_init_val_0_0 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_0_1 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_0_2 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_0_3 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_1_0 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_1_1 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_1_2 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_1_3 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_2_0 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_2_1 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_2_2 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_2_3 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_3_0 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_3_1 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_3_2 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_3_3 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>


      xegpu.alloc_nbarrier 16
      %nbarrier_id = arith.constant 1 : i8
      %num_threads = arith.constant 32 : i8
      %nbarrier = xegpu.init_nbarrier %nbarrier_id, %num_threads : i8, i8 -> !xegpu.nbarrier
      // K loop advances in 32 steps
      %k_loop_result:21 = scf.for %k = %c0 to %c4096 step %c32 iter_args (
          %A_tile_0 = %A_sg_init_tile_0,
          // %A_tile_1 = %A_sg_init_tile_1,

          %B_tile_0 = %B_sg_init_tile_0,
          %B_tile_1 = %B_sg_init_tile_1,
          // %B_tile_2 = %B_sg_init_tile_2,
          // %B_tile_3 = %B_sg_init_tile_3,

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
          %c_val_3_3 = %c_init_val_3_3,

          %A_prefetch_tile = %A_sg_prefetch_tile_iter3,
          %B_prefetch_tile = %B_sg_prefetch_tile_iter3
          ) ->
          (!xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
          !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
          !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
          vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,
          !xegpu.tensor_desc<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
          )
          {
        // all SGs must arrive here first
        %every_8th_iter = arith.remui %k, %c256 : index
        %every_8th_iter_i32 = arith.index_cast %every_8th_iter : index to i32
        %every_8th_iter_cond = arith.cmpi eq, %every_8th_iter_i32, %c0_i32 : i32
        scf.if %every_8th_iter_cond  {
          xegpu.nbarrier_arrive %nbarrier : !xegpu.nbarrier
        }
        // load A tiles
        %a_val = xegpu.load_nd %A_tile_0 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>> -> vector<2x32x16xf16>
        %a_val_0 = vector.extract %a_val [0] : vector<32x16xf16> from vector<2x32x16xf16>
        %a_val_1 = vector.extract %a_val [1] : vector<32x16xf16> from vector<2x32x16xf16>

        // load B tiles
        %b_val_arr_0 = xegpu.load_nd %B_tile_0 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>> -> vector<2x16x16x2xf16>
        %b_val_arr_1 = xegpu.load_nd %B_tile_1 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>> -> vector<2x16x16x2xf16>

        %b_val_0 = vector.extract %b_val_arr_0 [0] : vector<16x16x2xf16> from vector<2x16x16x2xf16>
        %b_val_1 = vector.extract %b_val_arr_0 [1] : vector<16x16x2xf16> from vector<2x16x16x2xf16>
        %b_val_2 = vector.extract %b_val_arr_1 [0] : vector<16x16x2xf16> from vector<2x16x16x2xf16>
        %b_val_3 = vector.extract %b_val_arr_1 [1] : vector<16x16x2xf16> from vector<2x16x16x2xf16>

        xegpu.compile_hint

        // prefetch A and B tiles
        // xegpu.prefetch_nd %A_prefetch_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
        // xegpu.prefetch_nd %B_prefetch_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>

        //
        xegpu.compile_hint

        // advance A and B prefetch tiles
        %next_A_prefetch_tile = xegpu.update_nd_offset %A_prefetch_tile, [%c0, %c32] : !xegpu.tensor_desc<8x32xf16>
        %next_B_prefetch_tile = xegpu.update_nd_offset %B_prefetch_tile, [%c32, %c0] : !xegpu.tensor_desc<8x32xf16>
        // advance A and B tiles
        %next_A_tile_0 = xegpu.update_nd_offset %A_tile_0, [%c0, %c32] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
        // %next_A_tile_1 = xegpu.update_nd_offset %A_tile_1, [%c0, %c32] : !xegpu.tensor_desc<32x16xf16>

        %next_B_tile_0 = xegpu.update_nd_offset %B_tile_0, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
        %next_B_tile_1 = xegpu.update_nd_offset %B_tile_1, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
        // %next_B_tile_2 = xegpu.update_nd_offset %B_tile_2, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16>
        // %next_B_tile_3 = xegpu.update_nd_offset %B_tile_3, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16>

        xegpu.compile_hint
        %a_val_0_flat = vector.shape_cast %a_val_0 : vector<32x16xf16> to vector<512xf16>
        %a_val_1_flat = vector.shape_cast %a_val_1 : vector<32x16xf16> to vector<512xf16>
        %a_val_0_0_flat = vector.extract_strided_slice %a_val_0_flat { offsets = [0], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_0_0 = vector.shape_cast %a_val_0_0_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_1_0_flat = vector.extract_strided_slice %a_val_0_flat { offsets = [128], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_1_0 = vector.shape_cast %a_val_1_0_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_2_0_flat = vector.extract_strided_slice  %a_val_0_flat { offsets = [256], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_2_0 = vector.shape_cast %a_val_2_0_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_3_0_flat = vector.extract_strided_slice %a_val_0_flat { offsets = [384], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_3_0 = vector.shape_cast %a_val_3_0_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_0_1_flat = vector.extract_strided_slice %a_val_1_flat { offsets = [0], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_0_1 = vector.shape_cast %a_val_0_1_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_1_1_flat = vector.extract_strided_slice %a_val_1_flat {offsets = [128], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_1_1 = vector.shape_cast %a_val_1_1_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_2_1_flat = vector.extract_strided_slice %a_val_1_flat { offsets = [256], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_2_1 = vector.shape_cast %a_val_2_1_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_3_1_flat = vector.extract_strided_slice %a_val_1_flat { offsets = [384], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_3_1 = vector.shape_cast %a_val_3_1_flat : vector<128xf16> to vector<8x16xf16>


        %b_val_0_flat = vector.shape_cast %b_val_0 : vector<16x16x2xf16> to vector<512xf16>
        %b_val_1_flat = vector.shape_cast %b_val_1 : vector<16x16x2xf16> to vector<512xf16>
        %b_val_2_flat = vector.shape_cast %b_val_2 : vector<16x16x2xf16> to vector<512xf16>
        %b_val_3_flat = vector.shape_cast %b_val_3 : vector<16x16x2xf16> to vector<512xf16>
        %b_val_0_0_flat = vector.extract_strided_slice %b_val_0_flat { offsets = [0], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_0_0 = vector.shape_cast %b_val_0_0_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_1_0_flat = vector.extract_strided_slice %b_val_0_flat { offsets = [256], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_1_0 = vector.shape_cast %b_val_1_0_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_0_1_flat = vector.extract_strided_slice %b_val_1_flat { offsets = [0], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_0_1 = vector.shape_cast %b_val_0_1_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_1_1_flat = vector.extract_strided_slice %b_val_1_flat { offsets = [256], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_1_1 = vector.shape_cast %b_val_1_1_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_0_2_flat = vector.extract_strided_slice %b_val_2_flat { offsets = [0], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_0_2 = vector.shape_cast %b_val_0_2_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_1_2_flat = vector.extract_strided_slice %b_val_2_flat { offsets = [256], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_1_2 = vector.shape_cast %b_val_1_2_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_0_3_flat = vector.extract_strided_slice  %b_val_3_flat { offsets = [0], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_0_3 = vector.shape_cast %b_val_0_3_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_1_3_flat = vector.extract_strided_slice %b_val_3_flat {offsets = [256], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_1_3 = vector.shape_cast %b_val_1_3_flat : vector<256xf16> to vector<8x16x2xf16>

        // do DPAS
        xegpu.compile_hint
        %new_c_val_0_0_temp = xegpu.dpas %a_val_0_0, %b_val_0_0, %c_val_0_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_0 = xegpu.dpas %a_val_0_1, %b_val_1_0, %new_c_val_0_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_0_temp = xegpu.dpas %a_val_1_0, %b_val_0_0, %c_val_1_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_0 = xegpu.dpas %a_val_1_1, %b_val_1_0, %new_c_val_1_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_0_temp = xegpu.dpas %a_val_2_0, %b_val_0_0, %c_val_2_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_0 = xegpu.dpas %a_val_2_1, %b_val_1_0, %new_c_val_2_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_0_temp = xegpu.dpas %a_val_3_0, %b_val_0_0, %c_val_3_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_0 = xegpu.dpas %a_val_3_1, %b_val_1_0, %new_c_val_3_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        %new_c_val_0_1_temp = xegpu.dpas %a_val_0_0, %b_val_0_1, %c_val_0_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_1 = xegpu.dpas %a_val_0_1, %b_val_1_1, %new_c_val_0_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_1_temp = xegpu.dpas %a_val_1_0, %b_val_0_1, %c_val_1_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_1 = xegpu.dpas %a_val_1_1, %b_val_1_1, %new_c_val_1_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_1_temp = xegpu.dpas %a_val_2_0, %b_val_0_1, %c_val_2_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_1 = xegpu.dpas %a_val_2_1, %b_val_1_1, %new_c_val_2_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_1_temp = xegpu.dpas %a_val_3_0, %b_val_0_1, %c_val_3_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_1 = xegpu.dpas %a_val_3_1, %b_val_1_1, %new_c_val_3_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        %new_c_val_0_2_temp = xegpu.dpas %a_val_0_0, %b_val_0_2, %c_val_0_2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_2 = xegpu.dpas %a_val_0_1, %b_val_1_2, %new_c_val_0_2_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_2_temp = xegpu.dpas %a_val_1_0, %b_val_0_2, %c_val_1_2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_2 = xegpu.dpas %a_val_1_1, %b_val_1_2, %new_c_val_1_2_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_2_temp = xegpu.dpas %a_val_2_0, %b_val_0_2, %c_val_2_2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_2 = xegpu.dpas %a_val_2_1, %b_val_1_2, %new_c_val_2_2_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_2_temp = xegpu.dpas %a_val_3_0, %b_val_0_2, %c_val_3_2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_2 = xegpu.dpas %a_val_3_1, %b_val_1_2, %new_c_val_3_2_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        %new_c_val_0_3_temp = xegpu.dpas %a_val_0_0, %b_val_0_3, %c_val_0_3 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_3 = xegpu.dpas %a_val_0_1, %b_val_1_3, %new_c_val_0_3_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_3_temp = xegpu.dpas %a_val_1_0, %b_val_0_3, %c_val_1_3 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_3 = xegpu.dpas %a_val_1_1, %b_val_1_3, %new_c_val_1_3_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_3_temp = xegpu.dpas %a_val_2_0, %b_val_0_3, %c_val_2_3 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_3 = xegpu.dpas %a_val_2_1, %b_val_1_3, %new_c_val_2_3_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_3_temp = xegpu.dpas %a_val_3_0, %b_val_0_3, %c_val_3_3 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        xegpu.compile_hint
        %new_c_val_3_3 = xegpu.dpas %a_val_3_1, %b_val_1_3, %new_c_val_3_3_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        xegpu.compile_hint
        //  barrier wait
        scf.if %every_8th_iter_cond {
          xegpu.nbarrier_wait %nbarrier : !xegpu.nbarrier
        }

        scf.yield %next_A_tile_0, %next_B_tile_0, %next_B_tile_1,
                  %new_c_val_0_0, %new_c_val_0_1, %new_c_val_0_2, %new_c_val_0_3, %new_c_val_1_0, %new_c_val_1_1, %new_c_val_1_2, %new_c_val_1_3, %new_c_val_2_0, %new_c_val_2_1, %new_c_val_2_2, %new_c_val_2_3, %new_c_val_3_0, %new_c_val_3_1, %new_c_val_3_2, %new_c_val_3_3,
                  %next_A_prefetch_tile, %next_B_prefetch_tile
                  : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
                  !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
                  !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
                  vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,
                  !xegpu.tensor_desc<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      }

      // trunc all DPAS output tiles to f16
      %c_result_0_0_f16 = arith.truncf %k_loop_result#3 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_0_1_f16 = arith.truncf %k_loop_result#4 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_0_2_f16 = arith.truncf %k_loop_result#5 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_0_3_f16 = arith.truncf %k_loop_result#6 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_0_f16 = arith.truncf %k_loop_result#7 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_1_f16 = arith.truncf %k_loop_result#8 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_2_f16 = arith.truncf %k_loop_result#9 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_3_f16 = arith.truncf %k_loop_result#10 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_0_f16 = arith.truncf %k_loop_result#11 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_1_f16 = arith.truncf %k_loop_result#12 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_2_f16 = arith.truncf %k_loop_result#13 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_3_f16 = arith.truncf %k_loop_result#14 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_0_f16 = arith.truncf %k_loop_result#15 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_1_f16 = arith.truncf %k_loop_result#16 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_2_f16 = arith.truncf %k_loop_result#17 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_3_f16 = arith.truncf %k_loop_result#18 : vector<8x16xf32> to vector<8x16xf16>

      // each SG needs to store the result of K loop into a 32x64 tile in C matrix. This is organized in 8x16 DPAS tiles
      // in the layout of 4x4x8x16. The max store size HW supoprt in f16 is 8x32. So we combine two 8x16 DPAS tiles
      // horizontally using vector.shuffle to get the required store size. The store layout then will 4x2x8x32 i.e.
      // we have 8 stores of size 8x32 in the layout 4x2.

      %c_result_8x32_0_0_t1 = vector.shuffle %c_result_0_0_f16, %c_result_0_1_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_0_0_t2 = vector.shape_cast %c_result_8x32_0_0_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_0_0 = vector.shape_cast %c_result_8x32_0_0_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_00 = xegpu.create_nd_tdesc %C[%C_sg_tile_offset_x, %C_sg_tile_offset_y] : memref<4096x4096xf16> -> !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_0_0, %c_sg_tile_00 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %c_result_8x32_0_1_t1 = vector.shuffle %c_result_0_2_f16, %c_result_0_3_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_0_1_t2 = vector.shape_cast %c_result_8x32_0_1_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_0_1 = vector.shape_cast %c_result_8x32_0_1_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_01 = xegpu.update_nd_offset %c_sg_tile_00, [%c0, %c32]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_0_1, %c_sg_tile_01 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %c_result_8x32_1_0_t1 = vector.shuffle %c_result_1_0_f16, %c_result_1_1_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_1_0_t2 = vector.shape_cast %c_result_8x32_1_0_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_1_0 = vector.shape_cast %c_result_8x32_1_0_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_10 = xegpu.update_nd_offset %c_sg_tile_00, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_1_0, %c_sg_tile_10 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint


      %c_result_8x32_1_1_t1 = vector.shuffle %c_result_1_2_f16, %c_result_1_3_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_1_1_t2 = vector.shape_cast %c_result_8x32_1_1_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_1_1 = vector.shape_cast %c_result_8x32_1_1_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_11 = xegpu.update_nd_offset %c_sg_tile_01, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_1_1, %c_sg_tile_11 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %c_result_8x32_2_0_t1 = vector.shuffle %c_result_2_0_f16, %c_result_2_1_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_2_0_t2 = vector.shape_cast %c_result_8x32_2_0_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_2_0 = vector.shape_cast %c_result_8x32_2_0_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_20 = xegpu.update_nd_offset %c_sg_tile_10, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_2_0, %c_sg_tile_20 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %c_result_8x32_2_1_t1 = vector.shuffle %c_result_2_2_f16, %c_result_2_3_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_2_1_t2 = vector.shape_cast %c_result_8x32_2_1_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_2_1 = vector.shape_cast %c_result_8x32_2_1_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_21 = xegpu.update_nd_offset %c_sg_tile_11, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_2_1, %c_sg_tile_21 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %c_result_8x32_3_0_t1 = vector.shuffle %c_result_3_0_f16, %c_result_3_1_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_3_0_t2 = vector.shape_cast %c_result_8x32_3_0_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_3_0 = vector.shape_cast %c_result_8x32_3_0_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_30 = xegpu.update_nd_offset %c_sg_tile_20, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_3_0, %c_sg_tile_30 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %c_result_8x32_3_1_t1 = vector.shuffle %c_result_3_2_f16, %c_result_3_3_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_3_1_t2 = vector.shape_cast %c_result_8x32_3_1_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_3_1 = vector.shape_cast %c_result_8x32_3_1_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_31 = xegpu.update_nd_offset %c_sg_tile_21, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_3_1, %c_sg_tile_31 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>

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
    %c_gen_int = arith.constant 1 : i1
    %cf_lower = arith.constant -0.5 : f32
    %cf_upper = arith.constant 0.5 : f32

    %A = memref.alloc() : memref<4096x4096xf16>
    %B = memref.alloc() : memref<4096x4096xf16>
    %C = memref.alloc() : memref<4096x4096xf16>
    %C_ref = memref.alloc() : memref<4096x4096xf32>

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
    %B_random = memref.cast %B : memref<4096x4096xf16>  to memref<*xf16>
    call @fillResource1DRandomF16(%B_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf16>, f32, f32, i1) -> ()

    // intialize matrix C and C_ref ; C[i, j] = 0
    %c0_f16 = arith.constant 0.0 : f16
    %c0_f32 = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c4096 step %c1 {
      scf.for %j = %c0 to %c4096 step %c1 {
        memref.store %c0_f16, %C[%i, %j] : memref<4096x4096xf16>
        memref.store %c0_f32, %C_ref[%i, %j] : memref<4096x4096xf32>
      }
    }

    // Run GPU.
    %2 = call @test(%A, %B, %C) : (memref<4096x4096xf16>, memref<4096x4096xf16>, memref<4096x4096xf16>) -> memref<4096x4096xf16>
    %cast_C = memref.cast %2 : memref<4096x4096xf16> to memref<*xf16>

    // Run CPU.
    %A_cast = memref.cast %A : memref<4096x4096xf16> to memref<*xf16>
    %B_cast = memref.cast %B : memref<4096x4096xf16> to memref<*xf16>
    %C_cast = memref.cast %C_ref : memref<4096x4096xf32> to memref<*xf32>
    call @gemmF16F16F16(%A_cast, %B_cast, %C_cast) : (memref<*xf16>, memref<*xf16>, memref<*xf32>) -> ()


    %C_row_0 = memref.subview %C_ref[0, 0][1, 4096][1, 1] : memref<4096x4096xf32> to memref<1x4096xf32, strided<[4096, 1], offset:0>>
    %C_row_0_cast = memref.cast %C_row_0 : memref<1x4096xf32, strided<[4096, 1], offset: 0>> to memref<*xf32>
    // call @printMemrefF32(%C_row_0_cast) : (memref<*xf32>) -> ()

    %C_row_0_gpu  = memref.subview %2[0, 0][1, 4096][1, 1] : memref<4096x4096xf16> to memref<1x4096xf16, strided<[4096, 1], offset:0>>
    %C_row_0_cast_gpu = memref.cast %C_row_0_gpu : memref<1x4096xf16, strided<[4096, 1], offset: 0>> to memref<*xf16>
    // call @printMemrefF16(%C_row_0_cast_gpu) : (memref<*xf16>) -> ()

    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF16(%cast_C, %C_cast) : (memref<*xf16>, memref<*xf32>) -> ()
    memref.dealloc %A : memref<4096x4096xf16>
    memref.dealloc %B : memref<4096x4096xf16>
    memref.dealloc %C : memref<4096x4096xf16>
    memref.dealloc %C_ref : memref<4096x4096xf32>
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @gemmF16F16F16(memref<*xf16>, memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}

}
