// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
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
      %A_sg_prefetch_tile_iter0 = xegpu.create_nd_tdesc %A[%A_sg_prefetch_offset_x, %c0] {mode = vc} : memref<4096x4096xf16> -> !xegpu.tensor_desc<8x32xf16>
      xegpu.prefetch_nd %A_sg_prefetch_tile_iter0 {l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<8x32xf16>
      // stage 2 (move 32 elements in the y direction and prefetch next 8x32 tile)
      %A_sg_prefetch_tile_iter1 = xegpu.update_nd_offset %A_sg_prefetch_tile_iter0, [%c0, %c32] {mode = vc}: !xegpu.tensor_desc<8x32xf16> -> !xegpu.tensor_desc<8x32xf16>
      xegpu.prefetch_nd %A_sg_prefetch_tile_iter1 {l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<8x32xf16>
      // stage 3
      %A_sg_prefetch_tile_iter2 = xegpu.update_nd_offset %A_sg_prefetch_tile_iter1, [%c0, %c32] {mode = vc}: !xegpu.tensor_desc<8x32xf16> -> !xegpu.tensor_desc<8x32xf16>
      xegpu.prefetch_nd %A_sg_prefetch_tile_iter2 {l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<8x32xf16>
      // compute the next tile to prefetch within K loop
      %A_sg_prefetch_tile_iter3 = xegpu.update_nd_offset %A_sg_prefetch_tile_iter2, [%c0, %c32] {mode = vc}: !xegpu.tensor_desc<8x32xf16> -> !xegpu.tensor_desc<8x32xf16>

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
      %B_sg_prefetch_offset_x_temp0 = arith.remui %local_sg_id_x, %c4 : index
      %B_sg_prefetch_offset_x = arith.muli %B_sg_prefetch_offset_x_temp0, %c8 : index
      %B_sg_prefetch_offset_y_temp0 = arith.muli %local_sg_id_y, %c64 : index
      %B_sg_prefetch_offset_y_temp1 = arith.divui %local_sg_id_x, %c4 : index
      %B_sg_prefetch_offset_y_temp2 = arith.muli %B_sg_prefetch_offset_y_temp1, %c32 : index
      %B_sg_prefetch_offset_y_temp3 = arith.addi %B_sg_prefetch_offset_y_temp0, %B_sg_prefetch_offset_y_temp2 : index
      %B_sg_prefetch_offset_y = arith.addi %wg_tile_offset_y, %B_sg_prefetch_offset_y_temp3 : index

      // create B prefetch tiles and prefetch
      %B_sg_prefetch_tile_iter0 = xegpu.create_nd_tdesc %B[%B_sg_prefetch_offset_x, %B_sg_prefetch_offset_y] {mode = vc} : memref<4096x4096xf16> -> !xegpu.tensor_desc<8x32xf16>
      xegpu.prefetch_nd %B_sg_prefetch_tile_iter0 {l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<8x32xf16>
      // stage 2 (move 32 elements in the x direction and prefetch next 8x32 tile)
      %B_sg_prefetch_tile_iter1 = xegpu.update_nd_offset %B_sg_prefetch_tile_iter0, [%c32, %c0] {mode = vc}: !xegpu.tensor_desc<8x32xf16> -> !xegpu.tensor_desc<8x32xf16>
      xegpu.prefetch_nd %B_sg_prefetch_tile_iter1 {l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<8x32xf16>
      // stage 3
      %B_sg_prefetch_tile_iter2 = xegpu.update_nd_offset %B_sg_prefetch_tile_iter1, [%c32, %c0] {mode = vc}: !xegpu.tensor_desc<8x32xf16> -> !xegpu.tensor_desc<8x32xf16>
      xegpu.prefetch_nd %B_sg_prefetch_tile_iter2 {l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<8x32xf16>
      // compute the next tile to prefetch inside K loop
      %B_sg_prefetch_tile_iter3 = xegpu.update_nd_offset %B_sg_prefetch_tile_iter2, [%c32, %c0] {mode = vc}: !xegpu.tensor_desc<8x32xf16> -> !xegpu.tensor_desc<8x32xf16>


      // two 32x16 A tiles from 256x32 WG slice
      %A_sg_init_tile_0 = xegpu.create_nd_tdesc %A[%C_sg_tile_offset_x, %c0] {mode = vc}: memref<4096x4096xf16> -> !xegpu.tensor_desc<32x16xf16>
      %A_sg_init_tile_1 = xegpu.create_nd_tdesc %A[%C_sg_tile_offset_x, %c16] {mode = vc} : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x16xf16>
      // %A_sg_init_tile_1_0 =  xegpu.update_nd_offset %A_sg_init_tile_0_0, [%c8, %c0]  {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      // %A_sg_init_tile_2_0 =  xegpu.update_nd_offset %A_sg_init_tile_1_0, [%c8, %c0]  {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      // %A_sg_init_tile_3_0 =  xegpu.update_nd_offset %A_sg_init_tile_2_0, [%c8, %c0]  {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      // %A_sg_init_tile_0_1 =  xegpu.update_nd_offset %A_sg_init_tile_0_0, [%c0, %c16]  {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      // %A_sg_init_tile_1_1 =  xegpu.update_nd_offset %A_sg_init_tile_0_1, [%c8, %c0]  {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      // %A_sg_init_tile_2_1 =  xegpu.update_nd_offset %A_sg_init_tile_1_1, [%c8, %c0]  {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      // %A_sg_init_tile_3_1 =  xegpu.update_nd_offset %A_sg_init_tile_2_1, [%c8, %c0]  {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>

      //create B tiles
      %B_sg_init_tile_0 = xegpu.create_nd_tdesc %B[%c0, %C_sg_tile_offset_y] {mode = vc}: memref<4096x4096xf16> -> !xegpu.tensor_desc<32x16xf16>
      %B_sg_init_tile_1 = xegpu.update_nd_offset %B_sg_init_tile_0, [%c0, %c16] {mode = vc}:  !xegpu.tensor_desc<32x16xf16> -> !xegpu.tensor_desc<32x16xf16>
      %B_sg_init_tile_2 = xegpu.update_nd_offset %B_sg_init_tile_1, [%c0, %c16] {mode = vc}:  !xegpu.tensor_desc<32x16xf16> -> !xegpu.tensor_desc<32x16xf16>
      %B_sg_init_tile_3 = xegpu.update_nd_offset %B_sg_init_tile_2, [%c0, %c16] {mode = vc}:  !xegpu.tensor_desc<32x16xf16> -> !xegpu.tensor_desc<32x16xf16>

      // %B_sg_init_tile_0_1 =  xegpu.update_nd_offset %B_sg_init_tile_0_0, [%c0, %c16]  {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
      // %B_sg_init_tile_0_2 =  xegpu.update_nd_offset %B_sg_init_tile_0_1, [%c0, %c16]  {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
      // %B_sg_init_tile_0_3 =  xegpu.update_nd_offset %B_sg_init_tile_0_2, [%c0, %c16]  {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
      // %B_sg_init_tile_1_0 =  xegpu.update_nd_offset %B_sg_init_tile_0_0, [%c16, %c0]  {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
      // %B_sg_init_tile_1_1 =  xegpu.update_nd_offset %B_sg_init_tile_1_0, [%c0, %c16]  {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
      // %B_sg_init_tile_1_2 =  xegpu.update_nd_offset %B_sg_init_tile_1_1, [%c0, %c16]  {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
      // %B_sg_init_tile_1_3 =  xegpu.update_nd_offset %B_sg_init_tile_1_2, [%c0, %c16]  {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>

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
      %nbarrier_role = arith.constant 0 : i8
      %nbarrier = xegpu.create_nbarrier %nbarrier_id, %nbarrier_role {num_producers = 32 : i8, num_consumers = 32 : i8} : (i8, i8) -> !xegpu.nbarrier
      // K loop advances in 32 steps
      %k_loop_result:24 = scf.for %k = %c0 to %c4096 step %c32 iter_args (
          %A_tile_0 = %A_sg_init_tile_0,
          %A_tile_1 = %A_sg_init_tile_1,
          // %A_tile_2_0 = %A_sg_init_tile_2_0,
          // %A_tile_3_0 = %A_sg_init_tile_3_0,
          // %A_tile_0_1 = %A_sg_init_tile_0_1,
          // %A_tile_1_1 = %A_sg_init_tile_1_1,
          // %A_tile_2_1 = %A_sg_init_tile_2_1,
          // %A_tile_3_1 = %A_sg_init_tile_3_1,

          %B_tile_0 = %B_sg_init_tile_0,
          %B_tile_1 = %B_sg_init_tile_1,
          %B_tile_2 = %B_sg_init_tile_2,
          %B_tile_3 = %B_sg_init_tile_3,
          // %B_tile_1_0 = %B_sg_init_tile_1_0,
          // %B_tile_1_1 = %B_sg_init_tile_1_1,
          // %B_tile_1_2 = %B_sg_init_tile_1_2,
          // %B_tile_1_3 = %B_sg_init_tile_1_3,

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
          (!xegpu.tensor_desc<32x16xf16>, !xegpu.tensor_desc<32x16xf16>, !xegpu.tensor_desc<32x16xf16>, !xegpu.tensor_desc<32x16xf16>,  !xegpu.tensor_desc<32x16xf16>,  !xegpu.tensor_desc<32x16xf16>,
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
        %a_val_0 = xegpu.load_nd %A_tile_0 {vnni_axis = 1, l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<32x16xf16> -> vector<32x8x2xf16>
        %a_val_1 = xegpu.load_nd %A_tile_1 {vnni_axis = 1, l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<32x16xf16> -> vector<32x8x2xf16>

        // %a_val_2_0 = xegpu.load_nd %A_tile_2_0 {vnni_axis = 1, l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        // %a_val_3_0 = xegpu.load_nd %A_tile_3_0 {vnni_axis = 1, l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        // %a_val_0_1 = xegpu.load_nd %A_tile_0_1 {vnni_axis = 1, l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        // %a_val_1_1 = xegpu.load_nd %A_tile_1_1 {vnni_axis = 1, l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        // %a_val_2_1 = xegpu.load_nd %A_tile_2_1 {vnni_axis = 1, l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        // %a_val_3_1 = xegpu.load_nd %A_tile_3_1 {vnni_axis = 1, l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>

        // load B tiles
        %b_val_0 = xegpu.load_nd %B_tile_0 {vnni_axis = 0, l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<32x16xf16> -> vector<16x16x2xf16>
        %b_val_1 = xegpu.load_nd %B_tile_1 {vnni_axis = 0, l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<32x16xf16> -> vector<16x16x2xf16>
        %b_val_2 = xegpu.load_nd %B_tile_2 {vnni_axis = 0, l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<32x16xf16> -> vector<16x16x2xf16>
        %b_val_3 = xegpu.load_nd %B_tile_3 {vnni_axis = 0, l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<32x16xf16> -> vector<16x16x2xf16>
        // %b_val_1_0 = xegpu.load_nd %B_tile_1_0 {vnni_axis = 0, l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        // %b_val_1_1 = xegpu.load_nd %B_tile_1_1 {vnni_axis = 0, l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        // %b_val_1_2 = xegpu.load_nd %B_tile_1_2 {vnni_axis = 0, l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        // %b_val_1_3 = xegpu.load_nd %B_tile_1_3 {vnni_axis = 0, l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>

        xegpu.compile_hint

        // prefetch A and B tiles
        xegpu.prefetch_nd %A_prefetch_tile {l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<8x32xf16>
        xegpu.prefetch_nd %B_prefetch_tile {l1_hint = cached, l2_hint = cached, l3_hint = cached, mode = vc} : !xegpu.tensor_desc<8x32xf16>

        //
        xegpu.compile_hint

        // advance A and B prefetch tiles
        %next_A_prefetch_tile = xegpu.update_nd_offset %A_prefetch_tile, [%c0, %c32] {mode = vc}: !xegpu.tensor_desc<8x32xf16> -> !xegpu.tensor_desc<8x32xf16>
        %next_B_prefetch_tile = xegpu.update_nd_offset %B_prefetch_tile, [%c32, %c0] {mode = vc}: !xegpu.tensor_desc<8x32xf16> -> !xegpu.tensor_desc<8x32xf16>
        // advance A and B tiles
        %next_A_tile_0 = xegpu.update_nd_offset %A_tile_0, [%c0, %c32] {mode = vc} : !xegpu.tensor_desc<32x16xf16> -> !xegpu.tensor_desc<32x16xf16>
        %next_A_tile_1 = xegpu.update_nd_offset %A_tile_1, [%c0, %c32] {mode = vc} : !xegpu.tensor_desc<32x16xf16> -> !xegpu.tensor_desc<32x16xf16>
        // %next_A_tile_2_0 = xegpu.update_nd_offset %A_tile_2_0, [%c0, %c32] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        // %next_A_tile_3_0 = xegpu.update_nd_offset %A_tile_3_0, [%c0, %c32] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        // %next_A_tile_0_1 = xegpu.update_nd_offset %A_tile_0_1, [%c0, %c32] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        // %next_A_tile_1_1 = xegpu.update_nd_offset %A_tile_1_1, [%c0, %c32] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        // %next_A_tile_2_1 = xegpu.update_nd_offset %A_tile_2_1, [%c0, %c32] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        // %next_A_tile_3_1 = xegpu.update_nd_offset %A_tile_3_1, [%c0, %c32] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>

        %next_B_tile_0 = xegpu.update_nd_offset %B_tile_0, [%c32, %c0] {mode = vc} : !xegpu.tensor_desc<32x16xf16> -> !xegpu.tensor_desc<32x16xf16>
        %next_B_tile_1 = xegpu.update_nd_offset %B_tile_1, [%c32, %c0] {mode = vc} : !xegpu.tensor_desc<32x16xf16> -> !xegpu.tensor_desc<32x16xf16>
        %next_B_tile_2 = xegpu.update_nd_offset %B_tile_2, [%c32, %c0] {mode = vc} : !xegpu.tensor_desc<32x16xf16> -> !xegpu.tensor_desc<32x16xf16>
        %next_B_tile_3 = xegpu.update_nd_offset %B_tile_3, [%c32, %c0] {mode = vc} : !xegpu.tensor_desc<32x16xf16> -> !xegpu.tensor_desc<32x16xf16>
        // %next_B_tile_0_2 = xegpu.update_nd_offset %B_tile_0_2, [%c32, %c0] {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        // %next_B_tile_0_3 = xegpu.update_nd_offset %B_tile_0_3, [%c32, %c0] {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        // %next_B_tile_1_0 = xegpu.update_nd_offset %B_tile_1_0, [%c32, %c0] {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        // %next_B_tile_1_1 = xegpu.update_nd_offset %B_tile_1_1, [%c32, %c0] {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        // %next_B_tile_1_2 = xegpu.update_nd_offset %B_tile_1_2, [%c32, %c0] {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        // %next_B_tile_1_3 = xegpu.update_nd_offset %B_tile_1_3, [%c32, %c0] {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>

        xegpu.compile_hint
        %a_val_0_flat = vector.shape_cast %a_val_0 : vector<32x8x2xf16> to vector<512xf16>
        %a_val_1_flat = vector.shape_cast %a_val_1 : vector<32x8x2xf16> to vector<512xf16>
        %a_val_0_0_flat = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32] %a_val_0_flat, %a_val_0_flat : vector<512xf16>, vector<512xf16> -> vector<128xf16>
        %a_val_0_0 = vector.shape_cast %a_val_0_0_flat : vector<128xf16> to vector<8x8x2xf16>
        %a_val_1_0_flat = spirv.VectorShuffle [128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %a_val_0_flat , %a_val_0_flat : vector<512xf16>, vector<512xf16> -> vector<128xf16>
        %a_val_1_0 = vector.shape_cast %a_val_1_0_flat : vector<128xf16> to vector<8x8x2xf16>
        %a_val_2_0_flat = spirv.VectorShuffle [256 : i32, 257 : i32, 258 : i32, 259 : i32, 260 : i32, 261 : i32, 262 : i32, 263 : i32, 264 : i32, 265 : i32, 266 : i32, 267 : i32, 268 : i32, 269 : i32, 270 : i32, 271 : i32, 272 : i32, 273 : i32, 274 : i32, 275 : i32, 276 : i32, 277 : i32, 278 : i32, 279 : i32, 280 : i32, 281 : i32, 282 : i32, 283 : i32, 284 : i32, 285 : i32, 286 : i32, 287 : i32, 288 : i32, 289 : i32, 290 : i32, 291 : i32, 292 : i32, 293 : i32, 294 : i32, 295 : i32, 296 : i32, 297 : i32, 298 : i32, 299 : i32, 300 : i32, 301 : i32, 302 : i32, 303 : i32, 304 : i32, 305 : i32, 306 : i32, 307 : i32, 308 : i32, 309 : i32, 310 : i32, 311 : i32, 312 : i32, 313 : i32, 314 : i32, 315 : i32, 316 : i32, 317 : i32, 318 : i32, 319 : i32, 320 : i32, 321 : i32, 322 : i32, 323 : i32, 324 : i32, 325 : i32, 326 : i32, 327 : i32, 328 : i32, 329 : i32, 330 : i32, 331 : i32, 332 : i32, 333 : i32, 334 : i32, 335 : i32, 336 : i32, 337 : i32, 338 : i32, 339 : i32, 340 : i32, 341 : i32, 342 : i32, 343 : i32, 344 : i32, 345 : i32, 346 : i32, 347 : i32, 348 : i32, 349 : i32, 350 : i32, 351 : i32, 352 : i32, 353 : i32, 354 : i32, 355 : i32, 356 : i32, 357 : i32, 358 : i32, 359 : i32, 360 : i32, 361 : i32, 362 : i32, 363 : i32, 364 : i32, 365 : i32, 366 : i32, 367 : i32, 368 : i32, 369 : i32, 370 : i32, 371 : i32, 372 : i32, 373 : i32, 374 : i32, 375 : i32, 376 : i32, 377 : i32, 378 : i32, 379 : i32, 380 : i32, 381 : i32, 382 : i32, 383 : i32] %a_val_0_flat , %a_val_0_flat : vector<512xf16>, vector<512xf16> -> vector<128xf16>
        %a_val_2_0 = vector.shape_cast %a_val_2_0_flat : vector<128xf16> to vector<8x8x2xf16>
        %a_val_3_0_flat = spirv.VectorShuffle [384 : i32, 385 : i32, 386 : i32, 387 : i32, 388 : i32, 389 : i32, 390 : i32, 391 : i32, 392 : i32, 393 : i32, 394 : i32, 395 : i32, 396 : i32, 397 : i32, 398 : i32, 399 : i32, 400 : i32, 401 : i32, 402 : i32, 403 : i32, 404 : i32, 405 : i32, 406 : i32, 407 : i32, 408 : i32, 409 : i32, 410 : i32, 411 : i32, 412 : i32, 413 : i32, 414 : i32, 415 : i32, 416 : i32, 417 : i32, 418 : i32, 419 : i32, 420 : i32, 421 : i32, 422 : i32, 423 : i32, 424 : i32, 425 : i32, 426 : i32, 427 : i32, 428 : i32, 429 : i32, 430 : i32, 431 : i32, 432 : i32, 433 : i32, 434 : i32, 435 : i32, 436 : i32, 437 : i32, 438 : i32, 439 : i32, 440 : i32, 441 : i32, 442 : i32, 443 : i32, 444 : i32, 445 : i32, 446 : i32, 447 : i32, 448 : i32, 449 : i32, 450 : i32, 451 : i32, 452 : i32, 453 : i32, 454 : i32, 455 : i32, 456 : i32, 457 : i32, 458 : i32, 459 : i32, 460 : i32, 461 : i32, 462 : i32, 463 : i32, 464 : i32, 465 : i32, 466 : i32, 467 : i32, 468 : i32, 469 : i32, 470 : i32, 471 : i32, 472 : i32, 473 : i32, 474 : i32, 475 : i32, 476 : i32, 477 : i32, 478 : i32, 479 : i32, 480 : i32, 481 : i32, 482 : i32, 483 : i32, 484 : i32, 485 : i32, 486 : i32, 487 : i32, 488 : i32, 489 : i32, 490 : i32, 491 : i32, 492 : i32, 493 : i32, 494 : i32, 495 : i32, 496 : i32, 497 : i32, 498 : i32, 499 : i32, 500 : i32, 501 : i32, 502 : i32, 503 : i32, 504 : i32, 505 : i32, 506 : i32, 507 : i32, 508 : i32, 509 : i32, 510 : i32, 511 : i32] %a_val_0_flat , %a_val_0_flat : vector<512xf16>, vector<512xf16> -> vector<128xf16>
        %a_val_3_0 = vector.shape_cast %a_val_3_0_flat : vector<128xf16> to vector<8x8x2xf16>
        %a_val_0_1_flat = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32] %a_val_1_flat , %a_val_1_flat : vector<512xf16>, vector<512xf16> -> vector<128xf16>
        %a_val_0_1 = vector.shape_cast %a_val_0_1_flat : vector<128xf16> to vector<8x8x2xf16>
        %a_val_1_1_flat = spirv.VectorShuffle [128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %a_val_1_flat, %a_val_1_flat : vector<512xf16>, vector<512xf16> -> vector<128xf16>
        %a_val_1_1 = vector.shape_cast %a_val_1_1_flat : vector<128xf16> to vector<8x8x2xf16>
        %a_val_2_1_flat = spirv.VectorShuffle [256 : i32, 257 : i32, 258 : i32, 259 : i32, 260 : i32, 261 : i32, 262 : i32, 263 : i32, 264 : i32, 265 : i32, 266 : i32, 267 : i32, 268 : i32, 269 : i32, 270 : i32, 271 : i32, 272 : i32, 273 : i32, 274 : i32, 275 : i32, 276 : i32, 277 : i32, 278 : i32, 279 : i32, 280 : i32, 281 : i32, 282 : i32, 283 : i32, 284 : i32, 285 : i32, 286 : i32, 287 : i32, 288 : i32, 289 : i32, 290 : i32, 291 : i32, 292 : i32, 293 : i32, 294 : i32, 295 : i32, 296 : i32, 297 : i32, 298 : i32, 299 : i32, 300 : i32, 301 : i32, 302 : i32, 303 : i32, 304 : i32, 305 : i32, 306 : i32, 307 : i32, 308 : i32, 309 : i32, 310 : i32, 311 : i32, 312 : i32, 313 : i32, 314 : i32, 315 : i32, 316 : i32, 317 : i32, 318 : i32, 319 : i32, 320 : i32, 321 : i32, 322 : i32, 323 : i32, 324 : i32, 325 : i32, 326 : i32, 327 : i32, 328 : i32, 329 : i32, 330 : i32, 331 : i32, 332 : i32, 333 : i32, 334 : i32, 335 : i32, 336 : i32, 337 : i32, 338 : i32, 339 : i32, 340 : i32, 341 : i32, 342 : i32, 343 : i32, 344 : i32, 345 : i32, 346 : i32, 347 : i32, 348 : i32, 349 : i32, 350 : i32, 351 : i32, 352 : i32, 353 : i32, 354 : i32, 355 : i32, 356 : i32, 357 : i32, 358 : i32, 359 : i32, 360 : i32, 361 : i32, 362 : i32, 363 : i32, 364 : i32, 365 : i32, 366 : i32, 367 : i32, 368 : i32, 369 : i32, 370 : i32, 371 : i32, 372 : i32, 373 : i32, 374 : i32, 375 : i32, 376 : i32, 377 : i32, 378 : i32, 379 : i32, 380 : i32, 381 : i32, 382 : i32, 383 : i32] %a_val_1_flat, %a_val_1_flat : vector<512xf16>, vector<512xf16> -> vector<128xf16>
        %a_val_2_1 = vector.shape_cast %a_val_2_1_flat : vector<128xf16> to vector<8x8x2xf16>
        %a_val_3_1_flat = spirv.VectorShuffle [384 : i32, 385 : i32, 386 : i32, 387 : i32, 388 : i32, 389 : i32, 390 : i32, 391 : i32, 392 : i32, 393 : i32, 394 : i32, 395 : i32, 396 : i32, 397 : i32, 398 : i32, 399 : i32, 400 : i32, 401 : i32, 402 : i32, 403 : i32, 404 : i32, 405 : i32, 406 : i32, 407 : i32, 408 : i32, 409 : i32, 410 : i32, 411 : i32, 412 : i32, 413 : i32, 414 : i32, 415 : i32, 416 : i32, 417 : i32, 418 : i32, 419 : i32, 420 : i32, 421 : i32, 422 : i32, 423 : i32, 424 : i32, 425 : i32, 426 : i32, 427 : i32, 428 : i32, 429 : i32, 430 : i32, 431 : i32, 432 : i32, 433 : i32, 434 : i32, 435 : i32, 436 : i32, 437 : i32, 438 : i32, 439 : i32, 440 : i32, 441 : i32, 442 : i32, 443 : i32, 444 : i32, 445 : i32, 446 : i32, 447 : i32, 448 : i32, 449 : i32, 450 : i32, 451 : i32, 452 : i32, 453 : i32, 454 : i32, 455 : i32, 456 : i32, 457 : i32, 458 : i32, 459 : i32, 460 : i32, 461 : i32, 462 : i32, 463 : i32, 464 : i32, 465 : i32, 466 : i32, 467 : i32, 468 : i32, 469 : i32, 470 : i32, 471 : i32, 472 : i32, 473 : i32, 474 : i32, 475 : i32, 476 : i32, 477 : i32, 478 : i32, 479 : i32, 480 : i32, 481 : i32, 482 : i32, 483 : i32, 484 : i32, 485 : i32, 486 : i32, 487 : i32, 488 : i32, 489 : i32, 490 : i32, 491 : i32, 492 : i32, 493 : i32, 494 : i32, 495 : i32, 496 : i32, 497 : i32, 498 : i32, 499 : i32, 500 : i32, 501 : i32, 502 : i32, 503 : i32, 504 : i32, 505 : i32, 506 : i32, 507 : i32, 508 : i32, 509 : i32, 510 : i32, 511 : i32] %a_val_1_flat, %a_val_1_flat : vector<512xf16>, vector<512xf16> -> vector<128xf16>
        %a_val_3_1 = vector.shape_cast %a_val_3_1_flat : vector<128xf16> to vector<8x8x2xf16>


        %b_val_0_flat = vector.shape_cast %b_val_0 : vector<16x16x2xf16> to vector<512xf16>
        %b_val_1_flat = vector.shape_cast %b_val_1 : vector<16x16x2xf16> to vector<512xf16>
        %b_val_2_flat = vector.shape_cast %b_val_2 : vector<16x16x2xf16> to vector<512xf16>
        %b_val_3_flat = vector.shape_cast %b_val_3 : vector<16x16x2xf16> to vector<512xf16>
        %b_val_0_0_flat = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32, 128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %b_val_0_flat, %b_val_0_flat : vector<512xf16>, vector<512xf16> -> vector<256xf16>
        %b_val_0_0 = vector.shape_cast %b_val_0_0_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_1_0_flat = spirv.VectorShuffle [256 : i32, 257 : i32, 258 : i32, 259 : i32, 260 : i32, 261 : i32, 262 : i32, 263 : i32, 264 : i32, 265 : i32, 266 : i32, 267 : i32, 268 : i32, 269 : i32, 270 : i32, 271 : i32, 272 : i32, 273 : i32, 274 : i32, 275 : i32, 276 : i32, 277 : i32, 278 : i32, 279 : i32, 280 : i32, 281 : i32, 282 : i32, 283 : i32, 284 : i32, 285 : i32, 286 : i32, 287 : i32, 288 : i32, 289 : i32, 290 : i32, 291 : i32, 292 : i32, 293 : i32, 294 : i32, 295 : i32, 296 : i32, 297 : i32, 298 : i32, 299 : i32, 300 : i32, 301 : i32, 302 : i32, 303 : i32, 304 : i32, 305 : i32, 306 : i32, 307 : i32, 308 : i32, 309 : i32, 310 : i32, 311 : i32, 312 : i32, 313 : i32, 314 : i32, 315 : i32, 316 : i32, 317 : i32, 318 : i32, 319 : i32, 320 : i32, 321 : i32, 322 : i32, 323 : i32, 324 : i32, 325 : i32, 326 : i32, 327 : i32, 328 : i32, 329 : i32, 330 : i32, 331 : i32, 332 : i32, 333 : i32, 334 : i32, 335 : i32, 336 : i32, 337 : i32, 338 : i32, 339 : i32, 340 : i32, 341 : i32, 342 : i32, 343 : i32, 344 : i32, 345 : i32, 346 : i32, 347 : i32, 348 : i32, 349 : i32, 350 : i32, 351 : i32, 352 : i32, 353 : i32, 354 : i32, 355 : i32, 356 : i32, 357 : i32, 358 : i32, 359 : i32, 360 : i32, 361 : i32, 362 : i32, 363 : i32, 364 : i32, 365 : i32, 366 : i32, 367 : i32, 368 : i32, 369 : i32, 370 : i32, 371 : i32, 372 : i32, 373 : i32, 374 : i32, 375 : i32, 376 : i32, 377 : i32, 378 : i32, 379 : i32, 380 : i32, 381 : i32, 382 : i32, 383 : i32, 384 : i32, 385 : i32, 386 : i32, 387 : i32, 388 : i32, 389 : i32, 390 : i32, 391 : i32, 392 : i32, 393 : i32, 394 : i32, 395 : i32, 396 : i32, 397 : i32, 398 : i32, 399 : i32, 400 : i32, 401 : i32, 402 : i32, 403 : i32, 404 : i32, 405 : i32, 406 : i32, 407 : i32, 408 : i32, 409 : i32, 410 : i32, 411 : i32, 412 : i32, 413 : i32, 414 : i32, 415 : i32, 416 : i32, 417 : i32, 418 : i32, 419 : i32, 420 : i32, 421 : i32, 422 : i32, 423 : i32, 424 : i32, 425 : i32, 426 : i32, 427 : i32, 428 : i32, 429 : i32, 430 : i32, 431 : i32, 432 : i32, 433 : i32, 434 : i32, 435 : i32, 436 : i32, 437 : i32, 438 : i32, 439 : i32, 440 : i32, 441 : i32, 442 : i32, 443 : i32, 444 : i32, 445 : i32, 446 : i32, 447 : i32, 448 : i32, 449 : i32, 450 : i32, 451 : i32, 452 : i32, 453 : i32, 454 : i32, 455 : i32, 456 : i32, 457 : i32, 458 : i32, 459 : i32, 460 : i32, 461 : i32, 462 : i32, 463 : i32, 464 : i32, 465 : i32, 466 : i32, 467 : i32, 468 : i32, 469 : i32, 470 : i32, 471 : i32, 472 : i32, 473 : i32, 474 : i32, 475 : i32, 476 : i32, 477 : i32, 478 : i32, 479 : i32, 480 : i32, 481 : i32, 482 : i32, 483 : i32, 484 : i32, 485 : i32, 486 : i32, 487 : i32, 488 : i32, 489 : i32, 490 : i32, 491 : i32, 492 : i32, 493 : i32, 494 : i32, 495 : i32, 496 : i32, 497 : i32, 498 : i32, 499 : i32, 500 : i32, 501 : i32, 502 : i32, 503 : i32, 504 : i32, 505 : i32, 506 : i32, 507 : i32, 508 : i32, 509 : i32, 510 : i32, 511 : i32] %b_val_0_flat, %b_val_0_flat : vector<512xf16>, vector<512xf16> -> vector<256xf16>
        %b_val_1_0 = vector.shape_cast %b_val_1_0_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_0_1_flat = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32, 128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %b_val_1_flat, %b_val_1_flat : vector<512xf16>, vector<512xf16> -> vector<256xf16>
        %b_val_0_1 = vector.shape_cast %b_val_0_1_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_1_1_flat = spirv.VectorShuffle [256 : i32, 257 : i32, 258 : i32, 259 : i32, 260 : i32, 261 : i32, 262 : i32, 263 : i32, 264 : i32, 265 : i32, 266 : i32, 267 : i32, 268 : i32, 269 : i32, 270 : i32, 271 : i32, 272 : i32, 273 : i32, 274 : i32, 275 : i32, 276 : i32, 277 : i32, 278 : i32, 279 : i32, 280 : i32, 281 : i32, 282 : i32, 283 : i32, 284 : i32, 285 : i32, 286 : i32, 287 : i32, 288 : i32, 289 : i32, 290 : i32, 291 : i32, 292 : i32, 293 : i32, 294 : i32, 295 : i32, 296 : i32, 297 : i32, 298 : i32, 299 : i32, 300 : i32, 301 : i32, 302 : i32, 303 : i32, 304 : i32, 305 : i32, 306 : i32, 307 : i32, 308 : i32, 309 : i32, 310 : i32, 311 : i32, 312 : i32, 313 : i32, 314 : i32, 315 : i32, 316 : i32, 317 : i32, 318 : i32, 319 : i32, 320 : i32, 321 : i32, 322 : i32, 323 : i32, 324 : i32, 325 : i32, 326 : i32, 327 : i32, 328 : i32, 329 : i32, 330 : i32, 331 : i32, 332 : i32, 333 : i32, 334 : i32, 335 : i32, 336 : i32, 337 : i32, 338 : i32, 339 : i32, 340 : i32, 341 : i32, 342 : i32, 343 : i32, 344 : i32, 345 : i32, 346 : i32, 347 : i32, 348 : i32, 349 : i32, 350 : i32, 351 : i32, 352 : i32, 353 : i32, 354 : i32, 355 : i32, 356 : i32, 357 : i32, 358 : i32, 359 : i32, 360 : i32, 361 : i32, 362 : i32, 363 : i32, 364 : i32, 365 : i32, 366 : i32, 367 : i32, 368 : i32, 369 : i32, 370 : i32, 371 : i32, 372 : i32, 373 : i32, 374 : i32, 375 : i32, 376 : i32, 377 : i32, 378 : i32, 379 : i32, 380 : i32, 381 : i32, 382 : i32, 383 : i32, 384 : i32, 385 : i32, 386 : i32, 387 : i32, 388 : i32, 389 : i32, 390 : i32, 391 : i32, 392 : i32, 393 : i32, 394 : i32, 395 : i32, 396 : i32, 397 : i32, 398 : i32, 399 : i32, 400 : i32, 401 : i32, 402 : i32, 403 : i32, 404 : i32, 405 : i32, 406 : i32, 407 : i32, 408 : i32, 409 : i32, 410 : i32, 411 : i32, 412 : i32, 413 : i32, 414 : i32, 415 : i32, 416 : i32, 417 : i32, 418 : i32, 419 : i32, 420 : i32, 421 : i32, 422 : i32, 423 : i32, 424 : i32, 425 : i32, 426 : i32, 427 : i32, 428 : i32, 429 : i32, 430 : i32, 431 : i32, 432 : i32, 433 : i32, 434 : i32, 435 : i32, 436 : i32, 437 : i32, 438 : i32, 439 : i32, 440 : i32, 441 : i32, 442 : i32, 443 : i32, 444 : i32, 445 : i32, 446 : i32, 447 : i32, 448 : i32, 449 : i32, 450 : i32, 451 : i32, 452 : i32, 453 : i32, 454 : i32, 455 : i32, 456 : i32, 457 : i32, 458 : i32, 459 : i32, 460 : i32, 461 : i32, 462 : i32, 463 : i32, 464 : i32, 465 : i32, 466 : i32, 467 : i32, 468 : i32, 469 : i32, 470 : i32, 471 : i32, 472 : i32, 473 : i32, 474 : i32, 475 : i32, 476 : i32, 477 : i32, 478 : i32, 479 : i32, 480 : i32, 481 : i32, 482 : i32, 483 : i32, 484 : i32, 485 : i32, 486 : i32, 487 : i32, 488 : i32, 489 : i32, 490 : i32, 491 : i32, 492 : i32, 493 : i32, 494 : i32, 495 : i32, 496 : i32, 497 : i32, 498 : i32, 499 : i32, 500 : i32, 501 : i32, 502 : i32, 503 : i32, 504 : i32, 505 : i32, 506 : i32, 507 : i32, 508 : i32, 509 : i32, 510 : i32, 511 : i32] %b_val_1_flat, %b_val_1_flat : vector<512xf16>, vector<512xf16> -> vector<256xf16>
        %b_val_1_1 = vector.shape_cast %b_val_1_1_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_0_2_flat = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32, 128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %b_val_2_flat, %b_val_2_flat : vector<512xf16>, vector<512xf16> -> vector<256xf16>
        %b_val_0_2 = vector.shape_cast %b_val_0_2_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_1_2_flat = spirv.VectorShuffle [256 : i32, 257 : i32, 258 : i32, 259 : i32, 260 : i32, 261 : i32, 262 : i32, 263 : i32, 264 : i32, 265 : i32, 266 : i32, 267 : i32, 268 : i32, 269 : i32, 270 : i32, 271 : i32, 272 : i32, 273 : i32, 274 : i32, 275 : i32, 276 : i32, 277 : i32, 278 : i32, 279 : i32, 280 : i32, 281 : i32, 282 : i32, 283 : i32, 284 : i32, 285 : i32, 286 : i32, 287 : i32, 288 : i32, 289 : i32, 290 : i32, 291 : i32, 292 : i32, 293 : i32, 294 : i32, 295 : i32, 296 : i32, 297 : i32, 298 : i32, 299 : i32, 300 : i32, 301 : i32, 302 : i32, 303 : i32, 304 : i32, 305 : i32, 306 : i32, 307 : i32, 308 : i32, 309 : i32, 310 : i32, 311 : i32, 312 : i32, 313 : i32, 314 : i32, 315 : i32, 316 : i32, 317 : i32, 318 : i32, 319 : i32, 320 : i32, 321 : i32, 322 : i32, 323 : i32, 324 : i32, 325 : i32, 326 : i32, 327 : i32, 328 : i32, 329 : i32, 330 : i32, 331 : i32, 332 : i32, 333 : i32, 334 : i32, 335 : i32, 336 : i32, 337 : i32, 338 : i32, 339 : i32, 340 : i32, 341 : i32, 342 : i32, 343 : i32, 344 : i32, 345 : i32, 346 : i32, 347 : i32, 348 : i32, 349 : i32, 350 : i32, 351 : i32, 352 : i32, 353 : i32, 354 : i32, 355 : i32, 356 : i32, 357 : i32, 358 : i32, 359 : i32, 360 : i32, 361 : i32, 362 : i32, 363 : i32, 364 : i32, 365 : i32, 366 : i32, 367 : i32, 368 : i32, 369 : i32, 370 : i32, 371 : i32, 372 : i32, 373 : i32, 374 : i32, 375 : i32, 376 : i32, 377 : i32, 378 : i32, 379 : i32, 380 : i32, 381 : i32, 382 : i32, 383 : i32, 384 : i32, 385 : i32, 386 : i32, 387 : i32, 388 : i32, 389 : i32, 390 : i32, 391 : i32, 392 : i32, 393 : i32, 394 : i32, 395 : i32, 396 : i32, 397 : i32, 398 : i32, 399 : i32, 400 : i32, 401 : i32, 402 : i32, 403 : i32, 404 : i32, 405 : i32, 406 : i32, 407 : i32, 408 : i32, 409 : i32, 410 : i32, 411 : i32, 412 : i32, 413 : i32, 414 : i32, 415 : i32, 416 : i32, 417 : i32, 418 : i32, 419 : i32, 420 : i32, 421 : i32, 422 : i32, 423 : i32, 424 : i32, 425 : i32, 426 : i32, 427 : i32, 428 : i32, 429 : i32, 430 : i32, 431 : i32, 432 : i32, 433 : i32, 434 : i32, 435 : i32, 436 : i32, 437 : i32, 438 : i32, 439 : i32, 440 : i32, 441 : i32, 442 : i32, 443 : i32, 444 : i32, 445 : i32, 446 : i32, 447 : i32, 448 : i32, 449 : i32, 450 : i32, 451 : i32, 452 : i32, 453 : i32, 454 : i32, 455 : i32, 456 : i32, 457 : i32, 458 : i32, 459 : i32, 460 : i32, 461 : i32, 462 : i32, 463 : i32, 464 : i32, 465 : i32, 466 : i32, 467 : i32, 468 : i32, 469 : i32, 470 : i32, 471 : i32, 472 : i32, 473 : i32, 474 : i32, 475 : i32, 476 : i32, 477 : i32, 478 : i32, 479 : i32, 480 : i32, 481 : i32, 482 : i32, 483 : i32, 484 : i32, 485 : i32, 486 : i32, 487 : i32, 488 : i32, 489 : i32, 490 : i32, 491 : i32, 492 : i32, 493 : i32, 494 : i32, 495 : i32, 496 : i32, 497 : i32, 498 : i32, 499 : i32, 500 : i32, 501 : i32, 502 : i32, 503 : i32, 504 : i32, 505 : i32, 506 : i32, 507 : i32, 508 : i32, 509 : i32, 510 : i32, 511 : i32] %b_val_2_flat, %b_val_2_flat : vector<512xf16>, vector<512xf16> -> vector<256xf16>
        %b_val_1_2 = vector.shape_cast %b_val_1_2_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_0_3_flat = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32, 128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %b_val_3_flat, %b_val_3_flat : vector<512xf16>, vector<512xf16> -> vector<256xf16>
        %b_val_0_3 = vector.shape_cast %b_val_0_3_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_1_3_flat = spirv.VectorShuffle [256 : i32, 257 : i32, 258 : i32, 259 : i32, 260 : i32, 261 : i32, 262 : i32, 263 : i32, 264 : i32, 265 : i32, 266 : i32, 267 : i32, 268 : i32, 269 : i32, 270 : i32, 271 : i32, 272 : i32, 273 : i32, 274 : i32, 275 : i32, 276 : i32, 277 : i32, 278 : i32, 279 : i32, 280 : i32, 281 : i32, 282 : i32, 283 : i32, 284 : i32, 285 : i32, 286 : i32, 287 : i32, 288 : i32, 289 : i32, 290 : i32, 291 : i32, 292 : i32, 293 : i32, 294 : i32, 295 : i32, 296 : i32, 297 : i32, 298 : i32, 299 : i32, 300 : i32, 301 : i32, 302 : i32, 303 : i32, 304 : i32, 305 : i32, 306 : i32, 307 : i32, 308 : i32, 309 : i32, 310 : i32, 311 : i32, 312 : i32, 313 : i32, 314 : i32, 315 : i32, 316 : i32, 317 : i32, 318 : i32, 319 : i32, 320 : i32, 321 : i32, 322 : i32, 323 : i32, 324 : i32, 325 : i32, 326 : i32, 327 : i32, 328 : i32, 329 : i32, 330 : i32, 331 : i32, 332 : i32, 333 : i32, 334 : i32, 335 : i32, 336 : i32, 337 : i32, 338 : i32, 339 : i32, 340 : i32, 341 : i32, 342 : i32, 343 : i32, 344 : i32, 345 : i32, 346 : i32, 347 : i32, 348 : i32, 349 : i32, 350 : i32, 351 : i32, 352 : i32, 353 : i32, 354 : i32, 355 : i32, 356 : i32, 357 : i32, 358 : i32, 359 : i32, 360 : i32, 361 : i32, 362 : i32, 363 : i32, 364 : i32, 365 : i32, 366 : i32, 367 : i32, 368 : i32, 369 : i32, 370 : i32, 371 : i32, 372 : i32, 373 : i32, 374 : i32, 375 : i32, 376 : i32, 377 : i32, 378 : i32, 379 : i32, 380 : i32, 381 : i32, 382 : i32, 383 : i32, 384 : i32, 385 : i32, 386 : i32, 387 : i32, 388 : i32, 389 : i32, 390 : i32, 391 : i32, 392 : i32, 393 : i32, 394 : i32, 395 : i32, 396 : i32, 397 : i32, 398 : i32, 399 : i32, 400 : i32, 401 : i32, 402 : i32, 403 : i32, 404 : i32, 405 : i32, 406 : i32, 407 : i32, 408 : i32, 409 : i32, 410 : i32, 411 : i32, 412 : i32, 413 : i32, 414 : i32, 415 : i32, 416 : i32, 417 : i32, 418 : i32, 419 : i32, 420 : i32, 421 : i32, 422 : i32, 423 : i32, 424 : i32, 425 : i32, 426 : i32, 427 : i32, 428 : i32, 429 : i32, 430 : i32, 431 : i32, 432 : i32, 433 : i32, 434 : i32, 435 : i32, 436 : i32, 437 : i32, 438 : i32, 439 : i32, 440 : i32, 441 : i32, 442 : i32, 443 : i32, 444 : i32, 445 : i32, 446 : i32, 447 : i32, 448 : i32, 449 : i32, 450 : i32, 451 : i32, 452 : i32, 453 : i32, 454 : i32, 455 : i32, 456 : i32, 457 : i32, 458 : i32, 459 : i32, 460 : i32, 461 : i32, 462 : i32, 463 : i32, 464 : i32, 465 : i32, 466 : i32, 467 : i32, 468 : i32, 469 : i32, 470 : i32, 471 : i32, 472 : i32, 473 : i32, 474 : i32, 475 : i32, 476 : i32, 477 : i32, 478 : i32, 479 : i32, 480 : i32, 481 : i32, 482 : i32, 483 : i32, 484 : i32, 485 : i32, 486 : i32, 487 : i32, 488 : i32, 489 : i32, 490 : i32, 491 : i32, 492 : i32, 493 : i32, 494 : i32, 495 : i32, 496 : i32, 497 : i32, 498 : i32, 499 : i32, 500 : i32, 501 : i32, 502 : i32, 503 : i32, 504 : i32, 505 : i32, 506 : i32, 507 : i32, 508 : i32, 509 : i32, 510 : i32, 511 : i32] %b_val_3_flat, %b_val_3_flat : vector<512xf16>, vector<512xf16> -> vector<256xf16>
        %b_val_1_3 = vector.shape_cast %b_val_1_3_flat : vector<256xf16> to vector<8x16x2xf16>

        // do DPAS
        xegpu.compile_hint
        %new_c_val_0_0_temp = xegpu.dpas %a_val_0_0, %b_val_0_0, %c_val_0_0 : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_0 = xegpu.dpas %a_val_0_1, %b_val_1_0, %new_c_val_0_0_temp : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_0_temp = xegpu.dpas %a_val_1_0, %b_val_0_0, %c_val_1_0 : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_0 = xegpu.dpas %a_val_1_1, %b_val_1_0, %new_c_val_1_0_temp : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_0_temp = xegpu.dpas %a_val_2_0, %b_val_0_0, %c_val_2_0 : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_0 = xegpu.dpas %a_val_2_1, %b_val_1_0, %new_c_val_2_0_temp : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_0_temp = xegpu.dpas %a_val_3_0, %b_val_0_0, %c_val_3_0 : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_0 = xegpu.dpas %a_val_3_1, %b_val_1_0, %new_c_val_3_0_temp : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        %new_c_val_0_1_temp = xegpu.dpas %a_val_0_0, %b_val_0_1, %c_val_0_1 : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_1 = xegpu.dpas %a_val_0_1, %b_val_1_1, %new_c_val_0_1_temp : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_1_temp = xegpu.dpas %a_val_1_0, %b_val_0_1, %c_val_1_1 : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_1 = xegpu.dpas %a_val_1_1, %b_val_1_1, %new_c_val_1_1_temp : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_1_temp = xegpu.dpas %a_val_2_0, %b_val_0_1, %c_val_2_1 : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_1 = xegpu.dpas %a_val_2_1, %b_val_1_1, %new_c_val_2_1_temp : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_1_temp = xegpu.dpas %a_val_3_0, %b_val_0_1, %c_val_3_1 : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_1 = xegpu.dpas %a_val_3_1, %b_val_1_1, %new_c_val_3_1_temp : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        %new_c_val_0_2_temp = xegpu.dpas %a_val_0_0, %b_val_0_2, %c_val_0_2 : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_2 = xegpu.dpas %a_val_0_1, %b_val_1_2, %new_c_val_0_2_temp : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_2_temp = xegpu.dpas %a_val_1_0, %b_val_0_2, %c_val_1_2 : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_2 = xegpu.dpas %a_val_1_1, %b_val_1_2, %new_c_val_1_2_temp : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_2_temp = xegpu.dpas %a_val_2_0, %b_val_0_2, %c_val_2_2 : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_2 = xegpu.dpas %a_val_2_1, %b_val_1_2, %new_c_val_2_2_temp : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_2_temp = xegpu.dpas %a_val_3_0, %b_val_0_2, %c_val_3_2 : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_2 = xegpu.dpas %a_val_3_1, %b_val_1_2, %new_c_val_3_2_temp : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        %new_c_val_0_3_temp = xegpu.dpas %a_val_0_0, %b_val_0_3, %c_val_0_3 : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_3 = xegpu.dpas %a_val_0_1, %b_val_1_3, %new_c_val_0_3_temp : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_3_temp = xegpu.dpas %a_val_1_0, %b_val_0_3, %c_val_1_3 : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_3 = xegpu.dpas %a_val_1_1, %b_val_1_3, %new_c_val_1_3_temp : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_3_temp = xegpu.dpas %a_val_2_0, %b_val_0_3, %c_val_2_3 : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_3 = xegpu.dpas %a_val_2_1, %b_val_1_3, %new_c_val_2_3_temp : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_3_temp = xegpu.dpas %a_val_3_0, %b_val_0_3, %c_val_3_3 : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        xegpu.compile_hint
        %new_c_val_3_3 = xegpu.dpas %a_val_3_1, %b_val_1_3, %new_c_val_3_3_temp : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        xegpu.compile_hint
        //  barrier wait
        scf.if %every_8th_iter_cond {
          xegpu.nbarrier_wait %nbarrier : !xegpu.nbarrier
        }

        scf.yield %next_A_tile_0, %next_A_tile_1, %next_B_tile_0, %next_B_tile_1, %next_B_tile_2, %next_B_tile_3,
                  %new_c_val_0_0, %new_c_val_0_1, %new_c_val_0_2, %new_c_val_0_3, %new_c_val_1_0, %new_c_val_1_1, %new_c_val_1_2, %new_c_val_1_3, %new_c_val_2_0, %new_c_val_2_1, %new_c_val_2_2, %new_c_val_2_3, %new_c_val_3_0, %new_c_val_3_1, %new_c_val_3_2, %new_c_val_3_3,
                  %next_A_prefetch_tile, %next_B_prefetch_tile
                  : !xegpu.tensor_desc<32x16xf16>, !xegpu.tensor_desc<32x16xf16>, !xegpu.tensor_desc<32x16xf16>, !xegpu.tensor_desc<32x16xf16>,  !xegpu.tensor_desc<32x16xf16>,  !xegpu.tensor_desc<32x16xf16>,
                  vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,
                  !xegpu.tensor_desc<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      }

      // trunc to f16
      %c_result_0_0_f16 = arith.truncf %k_loop_result#6 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_0_1_f16 = arith.truncf %k_loop_result#7 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_0_2_f16 = arith.truncf %k_loop_result#8 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_0_3_f16 = arith.truncf %k_loop_result#9 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_0_f16 = arith.truncf %k_loop_result#10 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_1_f16 = arith.truncf %k_loop_result#11 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_2_f16 = arith.truncf %k_loop_result#12 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_3_f16 = arith.truncf %k_loop_result#13 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_0_f16 = arith.truncf %k_loop_result#14 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_1_f16 = arith.truncf %k_loop_result#15 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_2_f16 = arith.truncf %k_loop_result#16 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_3_f16 = arith.truncf %k_loop_result#17 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_0_f16 = arith.truncf %k_loop_result#18 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_1_f16 = arith.truncf %k_loop_result#19 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_2_f16 = arith.truncf %k_loop_result#20 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_3_f16 = arith.truncf %k_loop_result#21 : vector<8x16xf32> to vector<8x16xf16>

      // each SG needs to write to 32x64 C tile.
      // DPAS output size is 8x16. So each SG needs to write 16 (4x4) DPAS outputs.
      // create 16 address descriptions to cover 8x16 tiles in 4x4 layout within the 32x64 SG C tile.
      // advance 8 in x dim and, advance 16 in y dim
      // row 1
      %c_sg_tile_0_0 = xegpu.create_nd_tdesc %C[%C_sg_tile_offset_x, %C_sg_tile_offset_y] {mode = vc}: memref<4096x4096xf16> -> !xegpu.tensor_desc<8x16xf16>
      %c_sg_tile_0_1 = xegpu.update_nd_offset %c_sg_tile_0_0, [%c0, %c16]  {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      %c_sg_tile_0_2 =  xegpu.update_nd_offset %c_sg_tile_0_1, [%c0, %c16] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      %c_sg_tile_0_3 =  xegpu.update_nd_offset %c_sg_tile_0_2, [%c0, %c16] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      // row 2
      %c_sg_tile_1_0 = xegpu.update_nd_offset %c_sg_tile_0_0, [%c8, %c0]  {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      %c_sg_tile_1_1 = xegpu.update_nd_offset %c_sg_tile_1_0, [%c0, %c16] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      %c_sg_tile_1_2 = xegpu.update_nd_offset %c_sg_tile_1_1, [%c0, %c16] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      %c_sg_tile_1_3 = xegpu.update_nd_offset %c_sg_tile_1_2, [%c0, %c16] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      // row 3
      %c_sg_tile_2_0 = xegpu.update_nd_offset %c_sg_tile_0_0, [%c16, %c0] {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      %c_sg_tile_2_1 = xegpu.update_nd_offset %c_sg_tile_2_0, [%c0, %c16] {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      %c_sg_tile_2_2 = xegpu.update_nd_offset %c_sg_tile_2_1, [%c0, %c16]  {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      %c_sg_tile_2_3 = xegpu.update_nd_offset %c_sg_tile_2_2, [%c0, %c16] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      // row 4
      %c_sg_tile_3_0 = xegpu.update_nd_offset %c_sg_tile_0_0, [%c24, %c0] {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      %c_sg_tile_3_1 = xegpu.update_nd_offset %c_sg_tile_3_0, [%c0, %c16] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      %c_sg_tile_3_2 = xegpu.update_nd_offset %c_sg_tile_3_1, [%c0, %c16] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      %c_sg_tile_3_3 = xegpu.update_nd_offset %c_sg_tile_3_2, [%c0, %c16] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>


      // do store_nd
      xegpu.store_nd %c_result_0_0_f16, %c_sg_tile_0_0 {l1_hint = write_back, l2_hint = write_back, l3_hint = write_back, mode=vc} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_0_1_f16, %c_sg_tile_0_1 {l1_hint = write_back, l2_hint = write_back, l3_hint = write_back, mode=vc} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_0_2_f16, %c_sg_tile_0_2 {l1_hint = write_back, l2_hint = write_back, l3_hint = write_back, mode=vc} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_0_3_f16, %c_sg_tile_0_3 {l1_hint = write_back, l2_hint = write_back, l3_hint = write_back, mode=vc} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_1_0_f16, %c_sg_tile_1_0 {l1_hint = write_back, l2_hint = write_back, l3_hint = write_back, mode=vc} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_1_1_f16, %c_sg_tile_1_1 {l1_hint = write_back, l2_hint = write_back, l3_hint = write_back, mode=vc} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_1_2_f16, %c_sg_tile_1_2 {l1_hint = write_back, l2_hint = write_back, l3_hint = write_back, mode=vc} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_1_3_f16, %c_sg_tile_1_3 {l1_hint = write_back, l2_hint = write_back, l3_hint = write_back, mode=vc} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_2_0_f16, %c_sg_tile_2_0 {l1_hint = write_back, l2_hint = write_back, l3_hint = write_back, mode=vc} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_2_1_f16, %c_sg_tile_2_1 {l1_hint = write_back, l2_hint = write_back, l3_hint = write_back, mode=vc} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_2_2_f16, %c_sg_tile_2_2 {l1_hint = write_back, l2_hint = write_back, l3_hint = write_back, mode=vc} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_2_3_f16, %c_sg_tile_2_3  {l1_hint = write_back, l2_hint = write_back, l3_hint = write_back, mode=vc} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_3_0_f16, %c_sg_tile_3_0 {l1_hint = write_back, l2_hint = write_back, l3_hint = write_back, mode=vc} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_3_1_f16, %c_sg_tile_3_1 {l1_hint = write_back, l2_hint = write_back, l3_hint = write_back, mode=vc} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_3_2_f16, %c_sg_tile_3_2 {l1_hint = write_back, l2_hint = write_back, l3_hint = write_back, mode=vc} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c_result_3_3_f16, %c_sg_tile_3_3 {l1_hint = write_back, l2_hint = write_back, l3_hint = write_back, mode=vc} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
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
        %c_val_f16 = arith.truncf %c_val : f32 to f16
        %c_val_ = arith.extf %c_val_f16 : f16 to f32
        memref.store %c_val_ , %C[%i, %j] : memref<4096x4096xf32>
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
    %A_random_ = memref.collapse_shape %A [[0, 1]] :memref<4096x4096xf16> into memref<16777216xf16>
    %A_random = memref.cast %A_random_ : memref<16777216xf16> to memref<?xf16>
    call @fillMatrixRandomF16(%A_random) : (memref<?xf16>) -> ()

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
    %B_random_ = memref.collapse_shape %B [[0, 1]] :memref<4096x4096xf16> into memref<16777216xf16>
    %B_random = memref.cast %B_random_ : memref<16777216xf16> to memref<?xf16>
    call @fillMatrixRandomF16(%B_random) : (memref<?xf16>) -> ()

    // intialize matrix C and C_ref ; C[i, j] = 0
    %c0_f16 = arith.constant 0.0 : f16
    %c0_f32 = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c4096 step %c1 {
      scf.for %j = %c0 to %c4096 step %c1 {
        memref.store %c0_f16, %C[%i, %j] : memref<4096x4096xf16>
        memref.store %c0_f32, %C_ref[%i, %j] : memref<4096x4096xf32>
      }
    }
    // print input fror debug
    // %A_row_0 = memref.subview %A[1, 0][1, 4096][1, 1] : memref<4096x4096xf16> to memref<1x4096xf16, strided<[4096, 1], offset: 4096>>
    // %A_row_0_cast = memref.cast %A_row_0 : memref<1x4096xf16, strided<[4096, 1], offset: 4096>> to memref<*xf16>
    // call @printMemrefF16(%A_row_0_cast) : (memref<*xf16>) -> ()

    // run GPU
    %2 = call @test(%A, %B, %C) : (memref<4096x4096xf16>, memref<4096x4096xf16>, memref<4096x4096xf16>) -> memref<4096x4096xf16>

    call @cpu_reference(%A, %B, %C_ref) : (memref<4096x4096xf16>, memref<4096x4096xf16>, memref<4096x4096xf32>) -> ()

    // %cast = memref.cast %A : memref<4096x4096xf16> to memref<*xf16>
    // call @printMemrefF16(%cast) : (memref<*xf16>) -> ()
    %cast_C = memref.cast %2 : memref<4096x4096xf16> to memref<*xf16>
    %cast_C_ref = memref.cast %C_ref : memref<4096x4096xf32> to memref<*xf32>
    // call @printMemrefF16(%cast_C) : (memref<*xf16>) -> ()
    // call @printMemrefF32(%cast_C_ref) : (memref<*xf32>) -> ()

    %C_row_0 = memref.subview %C_ref[0, 0][1, 4096][1, 1] : memref<4096x4096xf32> to memref<1x4096xf32, strided<[4096, 1], offset:0>>
    %C_row_0_cast = memref.cast %C_row_0 : memref<1x4096xf32, strided<[4096, 1], offset: 0>> to memref<*xf32>
    // call @printMemrefF32(%C_row_0_cast) : (memref<*xf32>) -> ()

    %C_row_0_gpu  = memref.subview %2[0, 0][1, 4096][1, 1] : memref<4096x4096xf16> to memref<1x4096xf16, strided<[4096, 1], offset:0>>
    %C_row_0_cast_gpu = memref.cast %C_row_0_gpu : memref<1x4096xf16, strided<[4096, 1], offset: 0>> to memref<*xf16>
    // call @printMemrefF16(%C_row_0_cast_gpu) : (memref<*xf16>) -> ()

    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF16(%cast_C, %cast_C_ref) : (memref<*xf16>, memref<*xf32>) -> ()
    memref.dealloc %A : memref<4096x4096xf16>
    memref.dealloc %B : memref<4096x4096xf16>
    memref.dealloc %C : memref<4096x4096xf16>
    memref.dealloc %C_ref : memref<4096x4096xf32>
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillMatrixRandomF16(memref<?xf16>) attributes {llvm.emit_c_interface}

}
