// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<256x256xbf16>, %arg1: memref<256x256xbf16>, %arg2: memref<256x256xf32>) -> memref<256x256xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %memref = gpu.alloc  () : memref<256x256xbf16>
    gpu.memcpy  %memref, %arg0 : memref<256x256xbf16>, memref<256x256xbf16>
    %memref_0 = gpu.alloc  () : memref<256x256xbf16>
    gpu.memcpy  %memref_0, %arg1 : memref<256x256xbf16>, memref<256x256xbf16>
    %memref_1 = gpu.alloc  () : memref<256x256xf32>
    gpu.memcpy  %memref_1, %arg2 : memref<256x256xf32>, memref<256x256xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c8, %c4, %c1)  args(%memref : memref<256x256xbf16>, %memref_0 : memref<256x256xbf16>, %memref_1 : memref<256x256xf32>)
    gpu.dealloc  %memref : memref<256x256xbf16>
    gpu.dealloc  %memref_0 : memref<256x256xbf16>
    %alloc = memref.alloc() : memref<256x256xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<256x256xf32>, memref<256x256xf32>
    gpu.dealloc  %memref_1 : memref<256x256xf32>
    return %alloc : memref<256x256xf32>
  }
  gpu.module @test_kernel  {
      // constants
    gpu.func @test_kernel(%arg0: memref<256x256xbf16>, %arg1: memref<256x256xbf16>, %arg2: memref<256x256xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c256 = arith.constant 256 : index
      %c512 = arith.constant 512 : index
      %c1024 = arith.constant 1024 : index
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
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
      // compute SG C tile offsets in x and y dims
      // each SG needs to do the follwoing compute to update its 32x64 sub tile
      // (32x256)x(256x64)=(32x64)
      // DPAS size is (8x16)x(16x16)=(8x16)
      // K loop adavances in steps of 32, so inside the compute is (32x32)x(32x64) = (32x64)
      // So we need to (4x2) A tiles of size (8x16) and (2x4) B tiles of size (16x16)
      // tiled compute for a SG is (4x2x8x16)x(2x4x16x16)=(4x4x8x16)
      // this will require 32 DPAS ops (4x2x2) inside the K loop
      // WG tiles are 256x256 so there offsets are same for A, B and C
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
      // create A preftech tiles and prefetch
      // stage 1
      // stage 2 (move 32 elements in the y direction and prefetch next 8x32 tile)
      // stage 3
      // stage 4
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
      // XeTLA like co-operative prefetch for B
      // create B prefetch tiles and prefetch
      // stage 2 (move 32 elements in the x direction and prefetch next 8x32 tile)
      // stage 3
      // stage 4
      //create B tiles
      // ************************* //
      // init 16 C tiles of size 8x16 each is initialized to 0.0 assuming a zero C matrix
      // Multi nbarrier implementation,
      // one set nbarrier is used to sync subgroups with same sg_id_x (local_sg_id_x)
      // second set nbarrier us used to sync subgroups with same sg_id_y (local_sg_id_y)
      // In this case wg_size = 8,4 (wg_size_x = 8; wg_size_y = 4)
      // So in Y-direction we need 4 nbarrier (to sync subgroups with same sg_id_y)
      // In X-direction we need 8 nbarrier (to sync subgroups with same sg_id_x)
      // First set of nbarriers work across coloumns, we have 4 coloums of subgroups,
      // Hnece 4 nbrrier
      // Each nbarrier has 8 producers and consumers
      // nbarrier type is Producer_Consumer (https://gfxspecs.intel.com/Predator/Home/Index/57499)
      // %nbarrier_role = arith.constant 0 : i8
      // Second set of barriers work on across rows of subgroups,
      // we have 8 rows of subgroups. Hnece, 8 nbarrier
      // Each nbarrier has 4 producers and consumers
      // nbarrier type is Producer_Consumer (https://gfxspecs.intel.com/Predator/Home/Index/57499)
      // We already have 4 (=%c_wg_size_y) nbarriers with id 0-3,
      // Now the next set of barrier id would start from 4, hence,
      // K loop advances in 32 steps
        // all SGs must arrive here first
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %global_id_x = gpu.global_id  x
      %global_id_y = gpu.global_id  y
      %0 = arith.remui %global_id_x, %c8 : index
      %1 = arith.remui %global_id_y, %c4 : index
      %2 = arith.muli %global_id_x, %c32 : index
      %3 = arith.muli %global_id_y, %c64 : index
      %4 = arith.muli %block_id_x, %c256 : index
      %5 = arith.muli %block_id_y, %c256 : index
      %6 = arith.muli %0, %c4 : index
      %7 = arith.addi %6, %1 : index
      %8 = arith.muli %7, %c8 : index
      %9 = arith.addi %8, %4 : index
      %10 = xegpu.create_nd_tdesc %arg0[%9, %c0] : memref<256x256xbf16> -> !xegpu.tensor_desc<8x32xbf16>
      xegpu.prefetch_nd %10 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xbf16>
      %11 = xegpu.update_nd_offset %10, [%c0, %c32] : !xegpu.tensor_desc<8x32xbf16>
      xegpu.prefetch_nd %11 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xbf16>
      %12 = xegpu.update_nd_offset %11, [%c0, %c32] : !xegpu.tensor_desc<8x32xbf16>
      xegpu.prefetch_nd %12 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xbf16>
      %13 = xegpu.update_nd_offset %12, [%c0, %c32] : !xegpu.tensor_desc<8x32xbf16>
      %14 = arith.divui %0, %c2 : index
      %15 = arith.muli %14, %c8 : index
      %16 = arith.muli %1, %c64 : index
      %17 = arith.remui %0, %c2 : index
      %18 = arith.muli %17, %c32 : index
      %19 = arith.addi %16, %18 : index
      %20 = arith.addi %5, %19 : index
      %21 = xegpu.create_nd_tdesc %arg1[%15, %20] : memref<256x256xbf16> -> !xegpu.tensor_desc<8x32xbf16>
      xegpu.prefetch_nd %21 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xbf16>
      %22 = xegpu.update_nd_offset %21, [%c32, %c0] : !xegpu.tensor_desc<8x32xbf16>
      xegpu.prefetch_nd %22 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xbf16>
      %23 = xegpu.update_nd_offset %22, [%c32, %c0] : !xegpu.tensor_desc<8x32xbf16>
      xegpu.prefetch_nd %23 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xbf16>
      %24 = xegpu.update_nd_offset %23, [%c32, %c0] : !xegpu.tensor_desc<8x32xbf16>
      %25 = xegpu.create_nd_tdesc %arg0[%2, %c0] : memref<256x256xbf16> -> !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
      %26 = xegpu.update_nd_offset %25, [%c16, %c0] : !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
      %27 = xegpu.create_nd_tdesc %arg1[%c0, %3] : memref<256x256xbf16> -> !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
      %28 = xegpu.update_nd_offset %27, [%c0, %c32] : !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
      %29 = xegpu.update_nd_offset %27, [%c16, %c0] : !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
      %30 = xegpu.update_nd_offset %29, [%c0, %c32] : !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
      %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
      %31 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %32 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %33 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %34 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %35 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %36 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %37 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %38 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %39 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %40 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %41 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %42 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %43 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %44 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %45 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %46 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %c8_0 = arith.constant 8 : index
      %c4_1 = arith.constant 4 : index
      %47 = arith.addi %c4_1, %c8_0 : index
      xegpu.alloc_nbarrier 12
      %c8_i8 = arith.constant 8 : i8
      %48 = arith.index_cast %1 : index to i8
      %49 = xegpu.init_nbarrier %48, %c8_i8 : i8, i8 -> !xegpu.nbarrier
      %c4_i8 = arith.constant 4 : i8
      %50 = arith.addi %c4_1, %0 : index
      %51 = arith.index_cast %50 : index to i8
      %52 = xegpu.init_nbarrier %51, %c4_i8 : i8, i8 -> !xegpu.nbarrier
      %53:24 = scf.for %arg3 = %c0 to %c256 step %c32 iter_args(%arg4 = %25, %arg5 = %26, %arg6 = %27, %arg7 = %28, %arg8 = %29, %arg9 = %30, %arg10 = %31, %arg11 = %32, %arg12 = %33, %arg13 = %34, %arg14 = %35, %arg15 = %36, %arg16 = %37, %arg17 = %38, %arg18 = %39, %arg19 = %40, %arg20 = %41, %arg21 = %42, %arg22 = %43, %arg23 = %44, %arg24 = %45, %arg25 = %46, %arg26 = %12, %arg27 = %23) -> (!xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, !xegpu.tensor_desc<8x32xbf16>, !xegpu.tensor_desc<8x32xbf16>) {
        %70 = arith.remui %arg3, %c32 : index
        %71 = arith.index_cast %70 : index to i32
        %72 = arith.cmpi eq, %71, %c0_i32 : i32
        scf.if %72 {
          xegpu.nbarrier_arrive %49 : !xegpu.nbarrier
          xegpu.nbarrier_arrive %52 : !xegpu.nbarrier
        }
        // Load smaller load (16 registers) with cache line size width : 64 bytes, 32 elements
        // Although maximum load size supported is 2KB or 32 registers, we use smaller loads, for 2 main reasons:
        // ** 1. Hide load latency: we do smaller load means for B, we do 4 loads, we set up the loads and dpas orderring in
        //       such a way that, the first set of DPAS works on data loaded by first 2 load operations, as a result the
        //       second set of loads' latency can be hidden by the first set of DPAS operations.
        //
        // ** 2. Reduce the impact of L3 miss: Larger load means more cache lines to be loaded, more chance of potential L3 miss
        //       which could increase the load time
        // load B tiles
        // load A tiles
        // prefetch A and B tiles
        // advance A and B prefetch tiles
        // advance A and B tiles
        // b[0,0], b[0,1]
        // b[0,2], b[0,3]
        // b[1,0], b[1,1]
        // b[1,2], b[1,3]
        // xegpu.compile_hint
        // do DPAS
        //  barrier wait
        %73 = xegpu.load_nd %arg6 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x8x16x2xbf16>
        %74 = xegpu.load_nd %arg7 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x8x16x2xbf16>
        %75 = xegpu.load_nd %arg8 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x8x16x2xbf16>
        %76 = xegpu.load_nd %arg9 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x8x16x2xbf16>
        %77 = xegpu.load_nd %arg4 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x16x16xbf16>
        %78 = xegpu.load_nd %arg5 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x16x16xbf16>
        xegpu.compile_hint
        xegpu.prefetch_nd %arg26 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xbf16>
        xegpu.prefetch_nd %arg27 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xbf16>
        xegpu.compile_hint
        %79 = xegpu.update_nd_offset %arg26, [%c0, %c32] : !xegpu.tensor_desc<8x32xbf16>
        %80 = xegpu.update_nd_offset %arg27, [%c32, %c0] : !xegpu.tensor_desc<8x32xbf16>
        %81 = xegpu.update_nd_offset %arg4, [%c0, %c32] : !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
        %82 = xegpu.update_nd_offset %arg5, [%c0, %c32] : !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
        %83 = xegpu.update_nd_offset %arg6, [%c32, %c0] : !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
        %84 = xegpu.update_nd_offset %arg7, [%c32, %c0] : !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
        %85 = xegpu.update_nd_offset %arg8, [%c32, %c0] : !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
        %86 = xegpu.update_nd_offset %arg9, [%c32, %c0] : !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
        %87 = vector.shape_cast %73 : vector<2x8x16x2xbf16> to vector<512xbf16>
        %88 = vector.shape_cast %74 : vector<2x8x16x2xbf16> to vector<512xbf16>
        %89 = vector.shape_cast %75 : vector<2x8x16x2xbf16> to vector<512xbf16>
        %90 = vector.shape_cast %76 : vector<2x8x16x2xbf16> to vector<512xbf16>
        %91 = vector.extract_strided_slice %87 {offsets = [0], sizes = [256], strides = [1]} : vector<512xbf16> to vector<256xbf16>
        %92 = vector.shape_cast %91 : vector<256xbf16> to vector<8x16x2xbf16>
        %93 = vector.extract_strided_slice %87 {offsets = [256], sizes = [256], strides = [1]} : vector<512xbf16> to vector<256xbf16>
        %94 = vector.shape_cast %93 : vector<256xbf16> to vector<8x16x2xbf16>
        %95 = vector.extract_strided_slice %88 {offsets = [0], sizes = [256], strides = [1]} : vector<512xbf16> to vector<256xbf16>
        %96 = vector.shape_cast %95 : vector<256xbf16> to vector<8x16x2xbf16>
        %97 = vector.extract_strided_slice %88 {offsets = [256], sizes = [256], strides = [1]} : vector<512xbf16> to vector<256xbf16>
        %98 = vector.shape_cast %97 : vector<256xbf16> to vector<8x16x2xbf16>
        %99 = vector.extract_strided_slice %89 {offsets = [0], sizes = [256], strides = [1]} : vector<512xbf16> to vector<256xbf16>
        %100 = vector.shape_cast %99 : vector<256xbf16> to vector<8x16x2xbf16>
        %101 = vector.extract_strided_slice %89 {offsets = [256], sizes = [256], strides = [1]} : vector<512xbf16> to vector<256xbf16>
        %102 = vector.shape_cast %101 : vector<256xbf16> to vector<8x16x2xbf16>
        %103 = vector.extract_strided_slice %90 {offsets = [0], sizes = [256], strides = [1]} : vector<512xbf16> to vector<256xbf16>
        %104 = vector.shape_cast %103 : vector<256xbf16> to vector<8x16x2xbf16>
        %105 = vector.extract_strided_slice %90 {offsets = [256], sizes = [256], strides = [1]} : vector<512xbf16> to vector<256xbf16>
        %106 = vector.shape_cast %105 : vector<256xbf16> to vector<8x16x2xbf16>
        %107 = vector.shape_cast %77 : vector<2x16x16xbf16> to vector<512xbf16>
        %108 = vector.shape_cast %78 : vector<2x16x16xbf16> to vector<512xbf16>
        %109 = vector.extract_strided_slice %107 {offsets = [0], sizes = [128], strides = [1]} : vector<512xbf16> to vector<128xbf16>
        %110 = vector.shape_cast %109 : vector<128xbf16> to vector<8x16xbf16>
        %111 = vector.extract_strided_slice %107 {offsets = [128], sizes = [128], strides = [1]} : vector<512xbf16> to vector<128xbf16>
        %112 = vector.shape_cast %111 : vector<128xbf16> to vector<8x16xbf16>
        %113 = vector.extract_strided_slice %107 {offsets = [256], sizes = [128], strides = [1]} : vector<512xbf16> to vector<128xbf16>
        %114 = vector.shape_cast %113 : vector<128xbf16> to vector<8x16xbf16>
        %115 = vector.extract_strided_slice %107 {offsets = [384], sizes = [128], strides = [1]} : vector<512xbf16> to vector<128xbf16>
        %116 = vector.shape_cast %115 : vector<128xbf16> to vector<8x16xbf16>
        %117 = vector.extract_strided_slice %108 {offsets = [0], sizes = [128], strides = [1]} : vector<512xbf16> to vector<128xbf16>
        %118 = vector.shape_cast %117 : vector<128xbf16> to vector<8x16xbf16>
        %119 = vector.extract_strided_slice %108 {offsets = [128], sizes = [128], strides = [1]} : vector<512xbf16> to vector<128xbf16>
        %120 = vector.shape_cast %119 : vector<128xbf16> to vector<8x16xbf16>
        %121 = vector.extract_strided_slice %108 {offsets = [256], sizes = [128], strides = [1]} : vector<512xbf16> to vector<128xbf16>
        %122 = vector.shape_cast %121 : vector<128xbf16> to vector<8x16xbf16>
        %123 = vector.extract_strided_slice %108 {offsets = [384], sizes = [128], strides = [1]} : vector<512xbf16> to vector<128xbf16>
        %124 = vector.shape_cast %123 : vector<128xbf16> to vector<8x16xbf16>
        xegpu.compile_hint
        %125 = xegpu.dpas %110, %92, %arg10 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %126 = xegpu.dpas %112, %92, %arg14 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %127 = xegpu.dpas %118, %92, %arg18 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %128 = xegpu.dpas %120, %92, %arg22 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %129 = xegpu.dpas %110, %94, %arg11 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %130 = xegpu.dpas %112, %94, %arg15 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %131 = xegpu.dpas %118, %94, %arg19 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %132 = xegpu.dpas %120, %94, %arg23 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %133 = xegpu.dpas %110, %96, %arg12 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %134 = xegpu.dpas %112, %96, %arg16 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %135 = xegpu.dpas %118, %96, %arg20 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %136 = xegpu.dpas %120, %96, %arg24 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %137 = xegpu.dpas %110, %98, %arg13 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %138 = xegpu.dpas %112, %98, %arg17 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %139 = xegpu.dpas %118, %98, %arg21 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %140 = xegpu.dpas %120, %98, %arg25 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        xegpu.compile_hint
        %141 = xegpu.dpas %114, %100, %125 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %142 = xegpu.dpas %116, %100, %126 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %143 = xegpu.dpas %122, %100, %127 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %144 = xegpu.dpas %124, %100, %128 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %145 = xegpu.dpas %114, %102, %129 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %146 = xegpu.dpas %116, %102, %130 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %147 = xegpu.dpas %122, %102, %131 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %148 = xegpu.dpas %124, %102, %132 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %149 = xegpu.dpas %114, %104, %133 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %150 = xegpu.dpas %116, %104, %134 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %151 = xegpu.dpas %122, %104, %135 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %152 = xegpu.dpas %124, %104, %136 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %153 = xegpu.dpas %114, %106, %137 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %154 = xegpu.dpas %116, %106, %138 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %155 = xegpu.dpas %122, %106, %139 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %156 = xegpu.dpas %124, %106, %140 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        scf.if %72 {
          xegpu.nbarrier_wait %49 : !xegpu.nbarrier
          xegpu.nbarrier_wait %52 : !xegpu.nbarrier
        }
        scf.yield %81, %82, %83, %84, %85, %86, %141, %145, %149, %153, %142, %146, %150, %154, %143, %147, %151, %155, %144, %148, %152, %156, %79, %80 : !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, !xegpu.tensor_desc<16x16xbf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, !xegpu.tensor_desc<8x32xbf16>, !xegpu.tensor_desc<8x32xbf16>
      }
      // each SG needs to store the result of K loop into a 32x64 tile in C matrix. This is organized in 8x16 DPAS tiles
      // in the layout of 4x4x8x16. The max store size HW supoprt in f32 is 8x16.
      %54 = xegpu.create_nd_tdesc %arg2[%2, %3] : memref<256x256xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %53#6, %54 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.compile_hint
      %55 = xegpu.update_nd_offset %54, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %53#7, %55 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      %56 = xegpu.update_nd_offset %55, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %53#8, %56 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      %57 = xegpu.update_nd_offset %56, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %53#9, %57 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      %58 = xegpu.update_nd_offset %54, [%c8, %c0] : !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %53#10, %58 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.compile_hint
      %59 = xegpu.update_nd_offset %55, [%c8, %c0] : !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %53#11, %59 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      %60 = xegpu.update_nd_offset %56, [%c8, %c0] : !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %53#12, %60 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      %61 = xegpu.update_nd_offset %57, [%c8, %c0] : !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %53#13, %61 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      %62 = xegpu.update_nd_offset %58, [%c8, %c0] : !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %53#14, %62 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.compile_hint
      %63 = xegpu.update_nd_offset %59, [%c8, %c0] : !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %53#15, %63 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      %64 = xegpu.update_nd_offset %60, [%c8, %c0] : !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %53#16, %64 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      %65 = xegpu.update_nd_offset %61, [%c8, %c0] : !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %53#17, %65 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      %66 = xegpu.update_nd_offset %62, [%c8, %c0] : !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %53#18, %66 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.compile_hint
      %67 = xegpu.update_nd_offset %63, [%c8, %c0] : !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %53#19, %67 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      %68 = xegpu.update_nd_offset %64, [%c8, %c0] : !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %53#20, %68 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      %69 = xegpu.update_nd_offset %65, [%c8, %c0] : !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %53#21, %69 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    // Use one of the two options to initialize the A matrix
    // Option 1: intialize matrix A ; A[i, j] = j
    // scf.for %i = %c0 to %c256 step %c1 {
    //   scf.for %j = %c0 to %c256 step %c1 {
    //     %t = index.castu %j : index to i16
    //     %val = arith.uitofp %t : i16 to bf16
    //     memref.store %val, %A[%i, %j] : memref<256x256xbf16>
    //     // memref.store %c1_f16, %A[%i, %j] : memref<256x256xbf16>
    //     // memref.store %c2_f16, %B[%i, %j] : memref<256x256xbf16>
    //   }
    // }
    // Option 2:  convert the memref to 1D and fill with random values in (0.0, 1.0)
    // Use one of the two options below to initialize the B matrix
    // Option 1: make matrix B an identity matrix
    // scf.for %i = %c0 to %c256 step %c1 {
    //   scf.for %j = %c0 to %c256 step %c1 {
    //     %i_i32 = index.castu %i : index to i32
    //     %j_i32 = index.castu %j : index to i32
    //     %i_j_same = arith.cmpi eq, %i_i32, %j_i32 : i32
    //     scf.if %i_j_same {
    //       memref.store %cf_1, %B[%i, %j] : memref<256x256xbf16>
    //     } else {
    //       memref.store %cf_0, %B[%i, %j] : memref<256x256xbf16>
    //     }
    //   }
    // }
    // Option 2:  convert the memref to 1D and fill with random values in (0.0, 1.0)
    // intialize matrix C and C_ref ; C[i, j] = 0
    %false = arith.constant false
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %alloc = memref.alloc() : memref<256x256xbf16>
    %alloc_1 = memref.alloc() : memref<256x256xbf16>
    %alloc_2 = memref.alloc() : memref<256x256xf32>
    %alloc_3 = memref.alloc() : memref<256x256xf32>
    %cast = memref.cast %alloc : memref<256x256xbf16> to memref<*xbf16>
    call @fillResource1DRandomBF16(%cast, %cst, %cst_0, %false) : (memref<*xbf16>, f32, f32, i1) -> ()
    %cast_4 = memref.cast %alloc_1 : memref<256x256xbf16> to memref<*xbf16>
    call @fillResource1DRandomBF16(%cast_4, %cst, %cst_0, %false) : (memref<*xbf16>, f32, f32, i1) -> ()
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c256 step %c1 {
        memref.store %cst, %alloc_2[%arg0, %arg1] : memref<256x256xf32>
        memref.store %cst, %alloc_3[%arg0, %arg1] : memref<256x256xf32>
      }
    }
    // Run GPU
    // Run CPU
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @test(%alloc, %alloc_1, %alloc_2) : (memref<256x256xbf16>, memref<256x256xbf16>, memref<256x256xf32>) -> memref<256x256xf32>
    %cast_5 = memref.cast %0 : memref<256x256xf32> to memref<*xf32>
    %cast_6 = memref.cast %alloc : memref<256x256xbf16> to memref<*xbf16>
    %cast_7 = memref.cast %alloc_1 : memref<256x256xbf16> to memref<*xbf16>
    %cast_8 = memref.cast %alloc_3 : memref<256x256xf32> to memref<*xf32>
    call @gemmBF16BF16F32(%cast_6, %cast_7, %cast_8) : (memref<*xbf16>, memref<*xbf16>, memref<*xf32>) -> ()
    call @printAllcloseF32(%cast_5, %cast_8) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<256x256xbf16>
    memref.dealloc %alloc_1 : memref<256x256xbf16>
    memref.dealloc %alloc_2 : memref<256x256xf32>
    memref.dealloc %alloc_3 : memref<256x256xf32>
    memref.dealloc  %0 : memref<256x256xf32>
    return
  }
  func.func private @printMemrefBF16(memref<*xbf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseBF16(memref<*xbf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomBF16(memref<*xbf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @gemmBF16BF16F32(memref<*xbf16>, memref<*xbf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

