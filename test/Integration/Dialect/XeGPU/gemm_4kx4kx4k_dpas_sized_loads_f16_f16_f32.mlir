// RUN: IMEX_ENABLE_LARGE_REG_FILE=1 %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: IMEX_ENABLE_LARGE_REG_FILE=1 %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
// RUN: IMEX_ENABLE_LARGE_REG_FILE=1 %python_executable %imex_runner --requires=opencl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%opencl_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<4096x4096xf16>, %B: memref<4096x4096xf16>, %C: memref<4096x4096xf32>) -> memref<4096x4096xf32> attributes {llvm.emit_c_interface} {
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
    %C_gpu = gpu.alloc  host_shared () : memref<4096x4096xf32>
    memref.copy %C, %C_gpu : memref<4096x4096xf32> to memref<4096x4096xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c16, %c16, %c1) threads in (%c8, %c4, %c1) args(%A_gpu : memref<4096x4096xf16>, %B_gpu : memref<4096x4096xf16>, %C_gpu : memref<4096x4096xf32>)
    gpu.dealloc  %A_gpu : memref<4096x4096xf16>
    gpu.dealloc  %B_gpu : memref<4096x4096xf16>
    return %C_gpu : memref<4096x4096xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%A: memref<4096x4096xf16>, %B: memref<4096x4096xf16>, %C: memref<4096x4096xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // constants
      %c256 = arith.constant 256 : index
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
      %local_sg_id_x = arith.remsi %global_sg_id_x, %c8 : index
      %local_sg_id_y = arith.remsi %global_sg_id_y, %c4 : index

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

      // prefetch the entire 32x256 slice of B WG tile, we still use the prefetch size 8x32.
      // SGs have 8x4 layout. In this case 8 subgroups must do a colloborative  prefetch of 32x64 tile.
      // this because the B tile arrangement within the 32x256 slice is as follows
      // 32x64 | 32x64 | 32x64 | 32x64
      // in terms of 8x32 slices the arrangement is,
      // 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32
      // 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32
      // 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32
      // 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32
      // So SGs 0,1,2,3,....31 prefetch in following fashion
      // | 0  | 16||  1 | 17 || 2  | 18 || 3 | 19 |
      // | 4  | 20||  5 | 21 || 6  | 22 || 7 | 23 |
      // | 8  | 24||  9 | 25 || 10 | 26 || 11| 27 |
      // | 12 | 28|| 13 | 29 || 14 | 30 || 15| 31 |
      // For example, SGs 0,4,8,12,16,20,24,28 share the data in left 32x64 tile of B slice.

      // calculate the x offsets and y offsets within the 32x256 slice
      %B_sg_prefetch_offset_x_temp0 = arith.remsi %local_sg_id_x, %c4 : index
      %B_sg_prefetch_offset_x = arith.muli %B_sg_prefetch_offset_x_temp0, %c8 : index
      %B_sg_prefetch_offset_y_temp0 = arith.muli %local_sg_id_y, %c64 : index
      %B_sg_prefetch_offset_y_temp1 = arith.divsi %local_sg_id_x, %c4 : index
      %B_sg_prefetch_offset_y_temp2 = arith.muli %B_sg_prefetch_offset_y_temp1, %c32 : index
      %B_sg_prefetch_offset_y_temp3 = arith.addi %B_sg_prefetch_offset_y_temp0, %B_sg_prefetch_offset_y_temp2 : index
      %B_sg_prefetch_offset_y = arith.addi %wg_tile_offset_y, %B_sg_prefetch_offset_y_temp3 : index

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


      // create A tiles
      %A_sg_init_tile_0_0 = xegpu.create_nd_tdesc %A[%C_sg_tile_offset_x, %c0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<8x16xf16>
      %A_sg_init_tile_1_0 =  xegpu.update_nd_offset %A_sg_init_tile_0_0, [%c8, %c0]  : !xegpu.tensor_desc<8x16xf16>
      %A_sg_init_tile_2_0 =  xegpu.update_nd_offset %A_sg_init_tile_1_0, [%c8, %c0]  : !xegpu.tensor_desc<8x16xf16>
      %A_sg_init_tile_3_0 =  xegpu.update_nd_offset %A_sg_init_tile_2_0, [%c8, %c0]  : !xegpu.tensor_desc<8x16xf16>
      %A_sg_init_tile_0_1 =  xegpu.update_nd_offset %A_sg_init_tile_0_0, [%c0, %c16]  : !xegpu.tensor_desc<8x16xf16>
      %A_sg_init_tile_1_1 =  xegpu.update_nd_offset %A_sg_init_tile_0_1, [%c8, %c0]  : !xegpu.tensor_desc<8x16xf16>
      %A_sg_init_tile_2_1 =  xegpu.update_nd_offset %A_sg_init_tile_1_1, [%c8, %c0]  : !xegpu.tensor_desc<8x16xf16>
      %A_sg_init_tile_3_1 =  xegpu.update_nd_offset %A_sg_init_tile_2_1, [%c8, %c0]  : !xegpu.tensor_desc<8x16xf16>

      //create B tiles
      %B_sg_init_tile_0_0 = xegpu.create_nd_tdesc %B[%c0, %C_sg_tile_offset_y] : memref<4096x4096xf16> -> !xegpu.tensor_desc<16x16xf16>
      %B_sg_init_tile_0_1 =  xegpu.update_nd_offset %B_sg_init_tile_0_0, [%c0, %c16]  : !xegpu.tensor_desc<16x16xf16>
      %B_sg_init_tile_0_2 =  xegpu.update_nd_offset %B_sg_init_tile_0_1, [%c0, %c16]  : !xegpu.tensor_desc<16x16xf16>
      %B_sg_init_tile_0_3 =  xegpu.update_nd_offset %B_sg_init_tile_0_2, [%c0, %c16]  : !xegpu.tensor_desc<16x16xf16>
      %B_sg_init_tile_1_0 =  xegpu.update_nd_offset %B_sg_init_tile_0_0, [%c16, %c0]  : !xegpu.tensor_desc<16x16xf16>
      %B_sg_init_tile_1_1 =  xegpu.update_nd_offset %B_sg_init_tile_1_0, [%c0, %c16]  : !xegpu.tensor_desc<16x16xf16>
      %B_sg_init_tile_1_2 =  xegpu.update_nd_offset %B_sg_init_tile_1_1, [%c0, %c16]  : !xegpu.tensor_desc<16x16xf16>
      %B_sg_init_tile_1_3 =  xegpu.update_nd_offset %B_sg_init_tile_1_2, [%c0, %c16]  : !xegpu.tensor_desc<16x16xf16>

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
      %num_threads = arith.constant 8 : i8
      %nbarrier = xegpu.init_nbarrier %nbarrier_id, %num_threads : i8, i8 -> !xegpu.nbarrier
      // K loop advances in 32 steps
      %k_loop_result:34 = scf.for %k = %c0 to %c4096 step %c32 iter_args (
          %A_tile_0_0 = %A_sg_init_tile_0_0,
          %A_tile_1_0 = %A_sg_init_tile_1_0,
          %A_tile_2_0 = %A_sg_init_tile_2_0,
          %A_tile_3_0 = %A_sg_init_tile_3_0,
          %A_tile_0_1 = %A_sg_init_tile_0_1,
          %A_tile_1_1 = %A_sg_init_tile_1_1,
          %A_tile_2_1 = %A_sg_init_tile_2_1,
          %A_tile_3_1 = %A_sg_init_tile_3_1,

          %B_tile_0_0 = %B_sg_init_tile_0_0,
          %B_tile_0_1 = %B_sg_init_tile_0_1,
          %B_tile_0_2 = %B_sg_init_tile_0_2,
          %B_tile_0_3 = %B_sg_init_tile_0_3,
          %B_tile_1_0 = %B_sg_init_tile_1_0,
          %B_tile_1_1 = %B_sg_init_tile_1_1,
          %B_tile_1_2 = %B_sg_init_tile_1_2,
          %B_tile_1_3 = %B_sg_init_tile_1_3,

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
          (!xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>,  !xegpu.tensor_desc<8x16xf16>,  !xegpu.tensor_desc<8x16xf16>,  !xegpu.tensor_desc<8x16xf16>,  !xegpu.tensor_desc<8x16xf16>,
          !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>,!xegpu.tensor_desc<16x16xf16>,!xegpu.tensor_desc<16x16xf16>,!xegpu.tensor_desc<16x16xf16>,!xegpu.tensor_desc<16x16xf16>,!xegpu.tensor_desc<16x16xf16>,
          vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,
          !xegpu.tensor_desc<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
          )
          {
        // all SGs must arrive here first
        xegpu.nbarrier_arrive %nbarrier : !xegpu.nbarrier
        // load A tiles
        %a_val_0_0 = xegpu.load_nd %A_tile_0_0 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
        %a_val_1_0 = xegpu.load_nd %A_tile_1_0 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
        %a_val_2_0 = xegpu.load_nd %A_tile_2_0 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
        %a_val_3_0 = xegpu.load_nd %A_tile_3_0 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
        %a_val_0_1 = xegpu.load_nd %A_tile_0_1 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
        %a_val_1_1 = xegpu.load_nd %A_tile_1_1 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
        %a_val_2_1 = xegpu.load_nd %A_tile_2_1 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
        %a_val_3_1 = xegpu.load_nd %A_tile_3_1 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>

        // load B tiles
        %b_val_0_0 = xegpu.load_nd %B_tile_0_0 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %b_val_0_1 = xegpu.load_nd %B_tile_0_1 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %b_val_0_2 = xegpu.load_nd %B_tile_0_2 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %b_val_0_3 = xegpu.load_nd %B_tile_0_3 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %b_val_1_0 = xegpu.load_nd %B_tile_1_0 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %b_val_1_1 = xegpu.load_nd %B_tile_1_1 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %b_val_1_2 = xegpu.load_nd %B_tile_1_2 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %b_val_1_3 = xegpu.load_nd %B_tile_1_3 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>

        // prefetch A and B tiles
        xegpu.prefetch_nd %A_prefetch_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
        xegpu.prefetch_nd %B_prefetch_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>

        //
        xegpu.compile_hint

        // advance A and B prefetch tiles
        %next_A_prefetch_tile = xegpu.update_nd_offset %A_prefetch_tile, [%c0, %c32] : !xegpu.tensor_desc<8x32xf16>
        %next_B_prefetch_tile = xegpu.update_nd_offset %B_prefetch_tile, [%c32, %c0] : !xegpu.tensor_desc<8x32xf16>
        // advance A and B tiles
        %next_A_tile_0_0 = xegpu.update_nd_offset %A_tile_0_0, [%c0, %c32] : !xegpu.tensor_desc<8x16xf16>
        %next_A_tile_1_0 = xegpu.update_nd_offset %A_tile_1_0, [%c0, %c32] : !xegpu.tensor_desc<8x16xf16>
        %next_A_tile_2_0 = xegpu.update_nd_offset %A_tile_2_0, [%c0, %c32] : !xegpu.tensor_desc<8x16xf16>
        %next_A_tile_3_0 = xegpu.update_nd_offset %A_tile_3_0, [%c0, %c32] : !xegpu.tensor_desc<8x16xf16>
        %next_A_tile_0_1 = xegpu.update_nd_offset %A_tile_0_1, [%c0, %c32] : !xegpu.tensor_desc<8x16xf16>
        %next_A_tile_1_1 = xegpu.update_nd_offset %A_tile_1_1, [%c0, %c32] : !xegpu.tensor_desc<8x16xf16>
        %next_A_tile_2_1 = xegpu.update_nd_offset %A_tile_2_1, [%c0, %c32] : !xegpu.tensor_desc<8x16xf16>
        %next_A_tile_3_1 = xegpu.update_nd_offset %A_tile_3_1, [%c0, %c32] : !xegpu.tensor_desc<8x16xf16>

        %next_B_tile_0_0 = xegpu.update_nd_offset %B_tile_0_0, [%c32, %c0] : !xegpu.tensor_desc<16x16xf16>
        %next_B_tile_0_1 = xegpu.update_nd_offset %B_tile_0_1, [%c32, %c0] : !xegpu.tensor_desc<16x16xf16>
        %next_B_tile_0_2 = xegpu.update_nd_offset %B_tile_0_2, [%c32, %c0] : !xegpu.tensor_desc<16x16xf16>
        %next_B_tile_0_3 = xegpu.update_nd_offset %B_tile_0_3, [%c32, %c0] : !xegpu.tensor_desc<16x16xf16>
        %next_B_tile_1_0 = xegpu.update_nd_offset %B_tile_1_0, [%c32, %c0] : !xegpu.tensor_desc<16x16xf16>
        %next_B_tile_1_1 = xegpu.update_nd_offset %B_tile_1_1, [%c32, %c0] : !xegpu.tensor_desc<16x16xf16>
        %next_B_tile_1_2 = xegpu.update_nd_offset %B_tile_1_2, [%c32, %c0] : !xegpu.tensor_desc<16x16xf16>
        %next_B_tile_1_3 = xegpu.update_nd_offset %B_tile_1_3, [%c32, %c0] : !xegpu.tensor_desc<16x16xf16>

        xegpu.compile_hint

        // do DPAS
        %new_c_val_0_0_temp = xegpu.dpas %a_val_0_0, %b_val_0_0, %c_val_0_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_1_temp = xegpu.dpas %a_val_0_0, %b_val_0_1, %c_val_0_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_2_temp = xegpu.dpas %a_val_0_0, %b_val_0_2, %c_val_0_2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_3_temp = xegpu.dpas %a_val_0_0, %b_val_0_3, %c_val_0_3 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_0_temp = xegpu.dpas %a_val_1_0, %b_val_0_0, %c_val_1_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_1_temp = xegpu.dpas %a_val_1_0, %b_val_0_1, %c_val_1_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_2_temp = xegpu.dpas %a_val_1_0, %b_val_0_2, %c_val_1_2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_3_temp = xegpu.dpas %a_val_1_0, %b_val_0_3, %c_val_1_3 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_0_temp = xegpu.dpas %a_val_2_0, %b_val_0_0, %c_val_2_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_1_temp = xegpu.dpas %a_val_2_0, %b_val_0_1, %c_val_2_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_2_temp = xegpu.dpas %a_val_2_0, %b_val_0_2, %c_val_2_2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_3_temp = xegpu.dpas %a_val_2_0, %b_val_0_3, %c_val_2_3 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_0_temp = xegpu.dpas %a_val_3_0, %b_val_0_0, %c_val_3_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_1_temp = xegpu.dpas %a_val_3_0, %b_val_0_1, %c_val_3_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_2_temp = xegpu.dpas %a_val_3_0, %b_val_0_2, %c_val_3_2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_3_temp = xegpu.dpas %a_val_3_0, %b_val_0_3, %c_val_3_3 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        %new_c_val_0_0 = xegpu.dpas %a_val_0_1, %b_val_1_0, %new_c_val_0_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_1 = xegpu.dpas %a_val_0_1, %b_val_1_1, %new_c_val_0_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_2 = xegpu.dpas %a_val_0_1, %b_val_1_2, %new_c_val_0_2_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_3 = xegpu.dpas %a_val_0_1, %b_val_1_3, %new_c_val_0_3_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_0 = xegpu.dpas %a_val_1_1, %b_val_1_0, %new_c_val_1_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_1 = xegpu.dpas %a_val_1_1, %b_val_1_1, %new_c_val_1_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_2 = xegpu.dpas %a_val_1_1, %b_val_1_2, %new_c_val_1_2_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_3 = xegpu.dpas %a_val_1_1, %b_val_1_3, %new_c_val_1_3_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_0 = xegpu.dpas %a_val_2_1, %b_val_1_0, %new_c_val_2_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_1 = xegpu.dpas %a_val_2_1, %b_val_1_1, %new_c_val_2_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_2 = xegpu.dpas %a_val_2_1, %b_val_1_2, %new_c_val_2_2_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_3 = xegpu.dpas %a_val_2_1, %b_val_1_3, %new_c_val_2_3_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_0 = xegpu.dpas %a_val_3_1, %b_val_1_0, %new_c_val_3_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_1 = xegpu.dpas %a_val_3_1, %b_val_1_1, %new_c_val_3_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_2 = xegpu.dpas %a_val_3_1, %b_val_1_2, %new_c_val_3_2_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_3 = xegpu.dpas %a_val_3_1, %b_val_1_3, %new_c_val_3_3_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        xegpu.compile_hint
        //  barrier wait
        xegpu.nbarrier_wait %nbarrier : !xegpu.nbarrier

        scf.yield %next_A_tile_0_0, %next_A_tile_1_0, %next_A_tile_2_0, %next_A_tile_3_0, %next_A_tile_0_1, %next_A_tile_1_1, %next_A_tile_2_1, %next_A_tile_3_1,
                  %next_B_tile_0_0, %next_B_tile_0_1, %next_B_tile_0_2, %next_B_tile_0_3, %next_B_tile_1_0, %next_B_tile_1_1, %next_B_tile_1_2, %next_B_tile_1_3,
                  %new_c_val_0_0, %new_c_val_0_1, %new_c_val_0_2, %new_c_val_0_3, %new_c_val_1_0, %new_c_val_1_1, %new_c_val_1_2, %new_c_val_1_3, %new_c_val_2_0, %new_c_val_2_1, %new_c_val_2_2, %new_c_val_2_3, %new_c_val_3_0, %new_c_val_3_1, %new_c_val_3_2, %new_c_val_3_3,
                  %next_A_prefetch_tile, %next_B_prefetch_tile
                  : !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>,  !xegpu.tensor_desc<8x16xf16>,  !xegpu.tensor_desc<8x16xf16>,  !xegpu.tensor_desc<8x16xf16>,  !xegpu.tensor_desc<8x16xf16>,
                  !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>,!xegpu.tensor_desc<16x16xf16>,!xegpu.tensor_desc<16x16xf16>,!xegpu.tensor_desc<16x16xf16>,!xegpu.tensor_desc<16x16xf16>,!xegpu.tensor_desc<16x16xf16>,
                  vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,
                  !xegpu.tensor_desc<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      }

      // each SG needs to write to 32x64 C tile.
      // DPAS output size is 8x16. So each SG needs to write 16 (4x4) DPAS outputs.
      // create 16 address descriptions to cover 8x16 tiles in 4x4 layout within the 32x64 SG C tile.
      // advance 8 in x dim and, advance 16 in y dim
      // row 1
      %c_sg_tile_0_0 = xegpu.create_nd_tdesc %C[%C_sg_tile_offset_x, %C_sg_tile_offset_y] : memref<4096x4096xf32> -> !xegpu.tensor_desc<8x16xf32>
      %c_sg_tile_0_1 = xegpu.update_nd_offset %c_sg_tile_0_0, [%c0, %c16]  : !xegpu.tensor_desc<8x16xf32>
      %c_sg_tile_0_2 =  xegpu.update_nd_offset %c_sg_tile_0_1, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      %c_sg_tile_0_3 =  xegpu.update_nd_offset %c_sg_tile_0_2, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      // row 2
      %c_sg_tile_1_0 = xegpu.update_nd_offset %c_sg_tile_0_0, [%c8, %c0]  : !xegpu.tensor_desc<8x16xf32>
      %c_sg_tile_1_1 = xegpu.update_nd_offset %c_sg_tile_1_0, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      %c_sg_tile_1_2 = xegpu.update_nd_offset %c_sg_tile_1_1, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      %c_sg_tile_1_3 = xegpu.update_nd_offset %c_sg_tile_1_2, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      // row 3
      %c_sg_tile_2_0 = xegpu.update_nd_offset %c_sg_tile_0_0, [%c16, %c0] : !xegpu.tensor_desc<8x16xf32>
      %c_sg_tile_2_1 = xegpu.update_nd_offset %c_sg_tile_2_0, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      %c_sg_tile_2_2 = xegpu.update_nd_offset %c_sg_tile_2_1, [%c0, %c16]  : !xegpu.tensor_desc<8x16xf32>
      %c_sg_tile_2_3 = xegpu.update_nd_offset %c_sg_tile_2_2, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      // row 4
      %c_sg_tile_3_0 = xegpu.update_nd_offset %c_sg_tile_0_0, [%c24, %c0] : !xegpu.tensor_desc<8x16xf32>
      %c_sg_tile_3_1 = xegpu.update_nd_offset %c_sg_tile_3_0, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      %c_sg_tile_3_2 = xegpu.update_nd_offset %c_sg_tile_3_1, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      %c_sg_tile_3_3 = xegpu.update_nd_offset %c_sg_tile_3_2, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      // do store_nd
      xegpu.store_nd %k_loop_result#16, %c_sg_tile_0_0 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %k_loop_result#17, %c_sg_tile_0_1 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %k_loop_result#18, %c_sg_tile_0_2 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %k_loop_result#19, %c_sg_tile_0_3 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %k_loop_result#20, %c_sg_tile_1_0 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %k_loop_result#21, %c_sg_tile_1_1 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %k_loop_result#22, %c_sg_tile_1_2 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %k_loop_result#23, %c_sg_tile_1_3 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %k_loop_result#24, %c_sg_tile_2_0 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %k_loop_result#25, %c_sg_tile_2_1 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %k_loop_result#26, %c_sg_tile_2_2 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %k_loop_result#27, %c_sg_tile_2_3  {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %k_loop_result#28, %c_sg_tile_3_0 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %k_loop_result#29, %c_sg_tile_3_1 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %k_loop_result#30, %c_sg_tile_3_2 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %k_loop_result#31, %c_sg_tile_3_3 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }

  // compute CPU reference (takes minutes)
  func.func @cpu_reference(%A : memref<4096x4096xf16>, %B : memref<4096x4096xf16>, %C : memref<4096x4096xf32>) {
    %c4096 = arith.constant 4096 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    scf.for %i = %c0 to %c4096 step %c1 {
      scf.for %j = %c0 to %c4096 step %c1 {
        %c_curr = memref.load %C[%i, %j] : memref<4096x4096xf32>
        %c_val = scf.for %k_tile = %c0 to %c4096 step %c16 iter_args(%c_partial = %c_curr) -> f32 {
          %c_val_dpas = scf.for %k = %c0 to %c16 step %c1 iter_args(%c_dpas_partial = %c_partial) -> f32 {
            %k_dpas = arith.addi %k_tile, %k : index
            %a_val = memref.load %A[%i, %k_dpas] : memref<4096x4096xf16>
            %b_val = memref.load %B[%k_dpas, %j] : memref<4096x4096xf16>
            %a_cast = arith.extf %a_val : f16 to f32
            %b_cast = arith.extf %b_val : f16 to f32
            %t = arith.mulf %a_cast, %b_cast : f32
            // %t_cast = arith.extf %t : f16 to f16
            %c_sum = arith.addf %t, %c_dpas_partial : f32
            scf.yield %c_sum : f32
          }
          scf.yield %c_val_dpas : f32
        }
        // %c_val_f16 = arith.truncf %c_val : f32 to f16
        // %c_val_ = arith.extf %c_val_f16 : f16 to f32
        memref.store %c_val , %C[%i, %j] : memref<4096x4096xf32>
      }
    }
    return
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
    %C = memref.alloc() : memref<4096x4096xf32>
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


    // intialize matrix C and C_ref ; C[i, j] = 0
    %c0_f32 = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c4096 step %c1 {
      scf.for %j = %c0 to %c4096 step %c1 {
        memref.store %c0_f32, %C[%i, %j] : memref<4096x4096xf32>
        memref.store %c0_f32, %C_ref[%i, %j] : memref<4096x4096xf32>
      }
    }
    // print input fror debug
    // %A_row_0 = memref.subview %A[1, 0][1, 4096][1, 1] : memref<4096x4096xf16> to memref<1x4096xf16, strided<[4096, 1], offset: 4096>>
    // %A_row_0_cast = memref.cast %A_row_0 : memref<1x4096xf16, strided<[4096, 1], offset: 4096>> to memref<*xf16>
    // call @printMemrefF16(%A_row_0_cast) : (memref<*xf16>) -> ()

    // run GPU
    %2 = call @test(%A, %B, %C) : (memref<4096x4096xf16>, memref<4096x4096xf16>, memref<4096x4096xf32>) -> memref<4096x4096xf32>

    // run CPU
    call @cpu_reference(%A, %B, %C_ref) : (memref<4096x4096xf16>, memref<4096x4096xf16>, memref<4096x4096xf32>) -> ()

    // %cast = memref.cast %A : memref<4096x4096xf16> to memref<*xf16>
    // call @printMemrefF16(%cast) : (memref<*xf16>) -> ()
    %cast_C = memref.cast %2 : memref<4096x4096xf32> to memref<*xf32>
    %cast_C_ref = memref.cast %C_ref : memref<4096x4096xf32> to memref<*xf32>
    // call @printMemrefF32(%cast_C) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%cast_C_ref) : (memref<*xf32>) -> ()
    // %C_row_0 = memref.subview %2[1, 0][1, 4096][1, 1] : memref<4096x4096xf32> to memref<1x4096xf32, strided<[4096, 1], offset: 4096>>
    // %C_row_0_cast = memref.cast %C_row_0 : memref<1x4096xf32, strided<[4096, 1], offset: 4096>> to memref<*xf32>
    // call @printMemrefF32(%C_row_0_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast_C, %cast_C_ref) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %A : memref<4096x4096xf16>
    memref.dealloc %B : memref<4096x4096xf16>
    memref.dealloc %C : memref<4096x4096xf32>
    memref.dealloc %C_ref : memref<4096x4096xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}

}
