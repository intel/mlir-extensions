// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) -> memref<4096x4096xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %memref = gpu.alloc  () : memref<4096x4096xf16>
    gpu.memcpy  %memref, %arg0 : memref<4096x4096xf16>, memref<4096x4096xf16>
    %memref_0 = gpu.alloc  () : memref<4096x4096xf16>
    gpu.memcpy  %memref_0, %arg1 : memref<4096x4096xf16>, memref<4096x4096xf16>
    %memref_1 = gpu.alloc  () : memref<4096x4096xf16>
    gpu.memcpy  %memref_1, %arg2 : memref<4096x4096xf16>, memref<4096x4096xf16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c16, %c16, %c1) threads in (%c8, %c4, %c1)  args(%memref : memref<4096x4096xf16>, %memref_0 : memref<4096x4096xf16>, %memref_1 : memref<4096x4096xf16>)
    gpu.dealloc  %memref : memref<4096x4096xf16>
    gpu.dealloc  %memref_0 : memref<4096x4096xf16>
    %alloc = memref.alloc() : memref<4096x4096xf16>
    gpu.memcpy  %alloc, %memref_1 : memref<4096x4096xf16>, memref<4096x4096xf16>
    gpu.dealloc  %memref_1 : memref<4096x4096xf16>
    return %alloc : memref<4096x4096xf16>
  }
  gpu.module @test_kernel  {
      // constants
    gpu.func @test_kernel(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
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
      // (32x4096)x(4096x64)=(32x64)
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
      // compute the next tile to prefetch within K loop
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
      // create B prefetch tiles and prefetch
      // stage 2 (move 32 elements in the x direction and prefetch next 8x32 tile)
      // stage 3
      // compute the next tile to prefetch inside K loop
      // two 32x16 A tiles from 256x32 WG slice
      // %A_sg_init_tile_1 = xegpu.create_nd_tdesc %A[%C_sg_tile_offset_x, %c16] : memref<4096x4096xf16>
      //create B tiles
      // %B_sg_init_tile_2 = xegpu.update_nd_offset %B_sg_init_tile_1, [%c0, %c16] : !xegpu.tensor_desc<32x16xf16>
      // %B_sg_init_tile_3 = xegpu.update_nd_offset %B_sg_init_tile_2, [%c0, %c16] : !xegpu.tensor_desc<32x16xf16>
      // init 16 C tiles of size 8x16 each is initialized to 0.0 assuming a zero C matrix
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
      %10 = xegpu.create_nd_tdesc %arg0[%9, %c0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<8x32xf16>
      xegpu.prefetch_nd %10 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xf16>
      %11 = xegpu.update_nd_offset %10, [%c0, %c32] : !xegpu.tensor_desc<8x32xf16>
      xegpu.prefetch_nd %11 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xf16>
      %12 = xegpu.update_nd_offset %11, [%c0, %c32] : !xegpu.tensor_desc<8x32xf16>
      xegpu.prefetch_nd %12 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xf16>
      %13 = xegpu.update_nd_offset %12, [%c0, %c32] : !xegpu.tensor_desc<8x32xf16>
      %14 = arith.remui %0, %c4 : index
      %15 = arith.muli %14, %c8 : index
      %16 = arith.muli %1, %c64 : index
      %17 = arith.divui %0, %c4 : index
      %18 = arith.muli %17, %c32 : index
      %19 = arith.addi %16, %18 : index
      %20 = arith.addi %5, %19 : index
      %21 = xegpu.create_nd_tdesc %arg1[%15, %20] : memref<4096x4096xf16> -> !xegpu.tensor_desc<8x32xf16>
      xegpu.prefetch_nd %21 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xf16>
      %22 = xegpu.update_nd_offset %21, [%c32, %c0] : !xegpu.tensor_desc<8x32xf16>
      xegpu.prefetch_nd %22 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xf16>
      %23 = xegpu.update_nd_offset %22, [%c32, %c0] : !xegpu.tensor_desc<8x32xf16>
      xegpu.prefetch_nd %23 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xf16>
      %24 = xegpu.update_nd_offset %23, [%c32, %c0] : !xegpu.tensor_desc<8x32xf16>
      %25 = xegpu.create_nd_tdesc %arg0[%2, %c0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
      %26 = xegpu.create_nd_tdesc %arg1[%c0, %3] : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
      %27 = xegpu.update_nd_offset %26, [%c0, %c32] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
      %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
      %28 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %29 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %30 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
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
      xegpu.alloc_nbarrier 16
      // K loop advances in 32 steps
          // %A_tile_1 = %A_sg_init_tile_1,
          // %B_tile_2 = %B_sg_init_tile_2,
          // %B_tile_3 = %B_sg_init_tile_3,
        // all SGs must arrive here first
      %c1_i8 = arith.constant 1 : i8
      %c32_i8 = arith.constant 32 : i8
      %44 = xegpu.init_nbarrier %c1_i8, %c32_i8 : i8, i8 -> !xegpu.nbarrier
      %45:21 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %25, %arg5 = %26, %arg6 = %27, %arg7 = %28, %arg8 = %29, %arg9 = %30, %arg10 = %31, %arg11 = %32, %arg12 = %33, %arg13 = %34, %arg14 = %35, %arg15 = %36, %arg16 = %37, %arg17 = %38, %arg18 = %39, %arg19 = %40, %arg20 = %41, %arg21 = %42, %arg22 = %43, %arg23 = %13, %arg24 = %24) -> (!xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, !xegpu.tensor_desc<8x32xf16>, !xegpu.tensor_desc<8x32xf16>) {
        %78 = arith.remui %arg3, %c256 : index
        %79 = arith.index_cast %78 : index to i32
        %80 = arith.cmpi eq, %79, %c0_i32 : i32
        scf.if %80 {
          xegpu.nbarrier_arrive %44 : !xegpu.nbarrier
        }
        // load A tiles
        // load B tiles
        // prefetch A and B tiles
        //
        // advance A and B prefetch tiles
        // advance A and B tiles
        // %next_A_tile_1 = xegpu.update_nd_offset %A_tile_1, [%c0, %c32] : !xegpu.tensor_desc<32x16xf16>
        // %next_B_tile_2 = xegpu.update_nd_offset %B_tile_2, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16>
        // %next_B_tile_3 = xegpu.update_nd_offset %B_tile_3, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16>
        // do DPAS
        //  barrier wait
        %81 = xegpu.load_nd %arg4 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x32x16xf16>
        %82 = vector.extract %81[0] : vector<32x16xf16> from vector<2x32x16xf16>
        %83 = vector.extract %81[1] : vector<32x16xf16> from vector<2x32x16xf16>
        %84 = xegpu.load_nd %arg5 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x16x16x2xf16>
        %85 = xegpu.load_nd %arg6 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x16x16x2xf16>
        %86 = vector.extract %84[0] : vector<16x16x2xf16> from vector<2x16x16x2xf16>
        %87 = vector.extract %84[1] : vector<16x16x2xf16> from vector<2x16x16x2xf16>
        %88 = vector.extract %85[0] : vector<16x16x2xf16> from vector<2x16x16x2xf16>
        %89 = vector.extract %85[1] : vector<16x16x2xf16> from vector<2x16x16x2xf16>
        xegpu.compile_hint
        xegpu.prefetch_nd %arg23 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xf16>
        xegpu.prefetch_nd %arg24 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xf16>
        xegpu.compile_hint
        %90 = xegpu.update_nd_offset %arg23, [%c0, %c32] : !xegpu.tensor_desc<8x32xf16>
        %91 = xegpu.update_nd_offset %arg24, [%c32, %c0] : !xegpu.tensor_desc<8x32xf16>
        %92 = xegpu.update_nd_offset %arg4, [%c0, %c32] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
        %93 = xegpu.update_nd_offset %arg5, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
        %94 = xegpu.update_nd_offset %arg6, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
        xegpu.compile_hint
        %95 = vector.shape_cast %82 : vector<32x16xf16> to vector<512xf16>
        %96 = vector.shape_cast %83 : vector<32x16xf16> to vector<512xf16>
        %97 = vector.extract_strided_slice %95 {offsets = [0], sizes = [128], strides = [1]} : vector<512xf16> to vector<128xf16>
        %98 = vector.shape_cast %97 : vector<128xf16> to vector<8x16xf16>
        %99 = vector.extract_strided_slice %95 {offsets = [128], sizes = [128], strides = [1]} : vector<512xf16> to vector<128xf16>
        %100 = vector.shape_cast %99 : vector<128xf16> to vector<8x16xf16>
        %101 = vector.extract_strided_slice %95 {offsets = [256], sizes = [128], strides = [1]} : vector<512xf16> to vector<128xf16>
        %102 = vector.shape_cast %101 : vector<128xf16> to vector<8x16xf16>
        %103 = vector.extract_strided_slice %95 {offsets = [384], sizes = [128], strides = [1]} : vector<512xf16> to vector<128xf16>
        %104 = vector.shape_cast %103 : vector<128xf16> to vector<8x16xf16>
        %105 = vector.extract_strided_slice %96 {offsets = [0], sizes = [128], strides = [1]} : vector<512xf16> to vector<128xf16>
        %106 = vector.shape_cast %105 : vector<128xf16> to vector<8x16xf16>
        %107 = vector.extract_strided_slice %96 {offsets = [128], sizes = [128], strides = [1]} : vector<512xf16> to vector<128xf16>
        %108 = vector.shape_cast %107 : vector<128xf16> to vector<8x16xf16>
        %109 = vector.extract_strided_slice %96 {offsets = [256], sizes = [128], strides = [1]} : vector<512xf16> to vector<128xf16>
        %110 = vector.shape_cast %109 : vector<128xf16> to vector<8x16xf16>
        %111 = vector.extract_strided_slice %96 {offsets = [384], sizes = [128], strides = [1]} : vector<512xf16> to vector<128xf16>
        %112 = vector.shape_cast %111 : vector<128xf16> to vector<8x16xf16>
        %113 = vector.shape_cast %86 : vector<16x16x2xf16> to vector<512xf16>
        %114 = vector.shape_cast %87 : vector<16x16x2xf16> to vector<512xf16>
        %115 = vector.shape_cast %88 : vector<16x16x2xf16> to vector<512xf16>
        %116 = vector.shape_cast %89 : vector<16x16x2xf16> to vector<512xf16>
        %117 = vector.extract_strided_slice %113 {offsets = [0], sizes = [256], strides = [1]} : vector<512xf16> to vector<256xf16>
        %118 = vector.shape_cast %117 : vector<256xf16> to vector<8x16x2xf16>
        %119 = vector.extract_strided_slice %113 {offsets = [256], sizes = [256], strides = [1]} : vector<512xf16> to vector<256xf16>
        %120 = vector.shape_cast %119 : vector<256xf16> to vector<8x16x2xf16>
        %121 = vector.extract_strided_slice %114 {offsets = [0], sizes = [256], strides = [1]} : vector<512xf16> to vector<256xf16>
        %122 = vector.shape_cast %121 : vector<256xf16> to vector<8x16x2xf16>
        %123 = vector.extract_strided_slice %114 {offsets = [256], sizes = [256], strides = [1]} : vector<512xf16> to vector<256xf16>
        %124 = vector.shape_cast %123 : vector<256xf16> to vector<8x16x2xf16>
        %125 = vector.extract_strided_slice %115 {offsets = [0], sizes = [256], strides = [1]} : vector<512xf16> to vector<256xf16>
        %126 = vector.shape_cast %125 : vector<256xf16> to vector<8x16x2xf16>
        %127 = vector.extract_strided_slice %115 {offsets = [256], sizes = [256], strides = [1]} : vector<512xf16> to vector<256xf16>
        %128 = vector.shape_cast %127 : vector<256xf16> to vector<8x16x2xf16>
        %129 = vector.extract_strided_slice %116 {offsets = [0], sizes = [256], strides = [1]} : vector<512xf16> to vector<256xf16>
        %130 = vector.shape_cast %129 : vector<256xf16> to vector<8x16x2xf16>
        %131 = vector.extract_strided_slice %116 {offsets = [256], sizes = [256], strides = [1]} : vector<512xf16> to vector<256xf16>
        %132 = vector.shape_cast %131 : vector<256xf16> to vector<8x16x2xf16>
        xegpu.compile_hint
        %133 = xegpu.dpas %98, %118, %arg7 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %134 = xegpu.dpas %106, %120, %133 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %135 = xegpu.dpas %100, %118, %arg11 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %136 = xegpu.dpas %108, %120, %135 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %137 = xegpu.dpas %102, %118, %arg15 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %138 = xegpu.dpas %110, %120, %137 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %139 = xegpu.dpas %104, %118, %arg19 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %140 = xegpu.dpas %112, %120, %139 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %141 = xegpu.dpas %98, %122, %arg8 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %142 = xegpu.dpas %106, %124, %141 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %143 = xegpu.dpas %100, %122, %arg12 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %144 = xegpu.dpas %108, %124, %143 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %145 = xegpu.dpas %102, %122, %arg16 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %146 = xegpu.dpas %110, %124, %145 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %147 = xegpu.dpas %104, %122, %arg20 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %148 = xegpu.dpas %112, %124, %147 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %149 = xegpu.dpas %98, %126, %arg9 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %150 = xegpu.dpas %106, %128, %149 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %151 = xegpu.dpas %100, %126, %arg13 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %152 = xegpu.dpas %108, %128, %151 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %153 = xegpu.dpas %102, %126, %arg17 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %154 = xegpu.dpas %110, %128, %153 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %155 = xegpu.dpas %104, %126, %arg21 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %156 = xegpu.dpas %112, %128, %155 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %157 = xegpu.dpas %98, %130, %arg10 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %158 = xegpu.dpas %106, %132, %157 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %159 = xegpu.dpas %100, %130, %arg14 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %160 = xegpu.dpas %108, %132, %159 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %161 = xegpu.dpas %102, %130, %arg18 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %162 = xegpu.dpas %110, %132, %161 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %163 = xegpu.dpas %104, %130, %arg22 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        xegpu.compile_hint
        %164 = xegpu.dpas %112, %132, %163 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        xegpu.compile_hint
        scf.if %80 {
          xegpu.nbarrier_wait %44 : !xegpu.nbarrier
        }
        scf.yield %92, %93, %94, %134, %142, %150, %158, %136, %144, %152, %160, %138, %146, %154, %162, %140, %148, %156, %164, %90, %91 : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, !xegpu.tensor_desc<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      }
      // trunc to f16
      // each SG needs to write to 32x64 C tile.
      // DPAS output size is 8x16. So each SG needs to write 16 (4x4) DPAS outputs.
      // create 16 address descriptions to cover 8x16 tiles in 4x4 layout within the 32x64 SG C tile.
      // advance 8 in x dim and, advance 16 in y dim
      // row 1
      // row 2
      // row 3
      // row 4
      // do store_nd
      %46 = arith.truncf %45#3 : vector<8x16xf32> to vector<8x16xf16>
      %47 = arith.truncf %45#4 : vector<8x16xf32> to vector<8x16xf16>
      %48 = arith.truncf %45#5 : vector<8x16xf32> to vector<8x16xf16>
      %49 = arith.truncf %45#6 : vector<8x16xf32> to vector<8x16xf16>
      %50 = arith.truncf %45#7 : vector<8x16xf32> to vector<8x16xf16>
      %51 = arith.truncf %45#8 : vector<8x16xf32> to vector<8x16xf16>
      %52 = arith.truncf %45#9 : vector<8x16xf32> to vector<8x16xf16>
      %53 = arith.truncf %45#10 : vector<8x16xf32> to vector<8x16xf16>
      %54 = arith.truncf %45#11 : vector<8x16xf32> to vector<8x16xf16>
      %55 = arith.truncf %45#12 : vector<8x16xf32> to vector<8x16xf16>
      %56 = arith.truncf %45#13 : vector<8x16xf32> to vector<8x16xf16>
      %57 = arith.truncf %45#14 : vector<8x16xf32> to vector<8x16xf16>
      %58 = arith.truncf %45#15 : vector<8x16xf32> to vector<8x16xf16>
      %59 = arith.truncf %45#16 : vector<8x16xf32> to vector<8x16xf16>
      %60 = arith.truncf %45#17 : vector<8x16xf32> to vector<8x16xf16>
      %61 = arith.truncf %45#18 : vector<8x16xf32> to vector<8x16xf16>
      %62 = xegpu.create_nd_tdesc %arg2[%2, %3] : memref<4096x4096xf16> -> !xegpu.tensor_desc<8x16xf16>
      %63 = xegpu.update_nd_offset %62, [%c0, %c16] : !xegpu.tensor_desc<8x16xf16>
      %64 = xegpu.update_nd_offset %63, [%c0, %c16] : !xegpu.tensor_desc<8x16xf16>
      %65 = xegpu.update_nd_offset %64, [%c0, %c16] : !xegpu.tensor_desc<8x16xf16>
      %66 = xegpu.update_nd_offset %62, [%c8, %c0] : !xegpu.tensor_desc<8x16xf16>
      %67 = xegpu.update_nd_offset %66, [%c0, %c16] : !xegpu.tensor_desc<8x16xf16>
      %68 = xegpu.update_nd_offset %67, [%c0, %c16] : !xegpu.tensor_desc<8x16xf16>
      %69 = xegpu.update_nd_offset %68, [%c0, %c16] : !xegpu.tensor_desc<8x16xf16>
      %70 = xegpu.update_nd_offset %62, [%c16, %c0] : !xegpu.tensor_desc<8x16xf16>
      %71 = xegpu.update_nd_offset %70, [%c0, %c16] : !xegpu.tensor_desc<8x16xf16>
      %72 = xegpu.update_nd_offset %71, [%c0, %c16] : !xegpu.tensor_desc<8x16xf16>
      %73 = xegpu.update_nd_offset %72, [%c0, %c16] : !xegpu.tensor_desc<8x16xf16>
      %74 = xegpu.update_nd_offset %62, [%c24, %c0] : !xegpu.tensor_desc<8x16xf16>
      %75 = xegpu.update_nd_offset %74, [%c0, %c16] : !xegpu.tensor_desc<8x16xf16>
      %76 = xegpu.update_nd_offset %75, [%c0, %c16] : !xegpu.tensor_desc<8x16xf16>
      %77 = xegpu.update_nd_offset %76, [%c0, %c16] : !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %46, %62 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %47, %63 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %48, %64 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %49, %65 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %50, %66 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %51, %67 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %52, %68 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %53, %69 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %54, %70 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %55, %71 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %56, %72 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %57, %73 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %58, %74 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %59, %75 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %60, %76 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %61, %77 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant -5.000000e-01 : f32
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4096 = arith.constant 4096 : index
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
    // intialize matrix C and C_ref ; C[i, j] = 0
    %cst_2 = arith.constant 0.000000e+00 : f16
    %alloc = memref.alloc() : memref<4096x4096xf16>
    %alloc_3 = memref.alloc() : memref<4096x4096xf16>
    %alloc_4 = memref.alloc() : memref<4096x4096xf16>
    %alloc_5 = memref.alloc() : memref<4096x4096xf32>
    %cast = memref.cast %alloc : memref<4096x4096xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%cast, %cst_1, %cst_0, %false) : (memref<*xf16>, f32, f32, i1) -> ()
    %cast_6 = memref.cast %alloc_3 : memref<4096x4096xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%cast_6, %cst_1, %cst_0, %false) : (memref<*xf16>, f32, f32, i1) -> ()
    scf.for %arg0 = %c0 to %c4096 step %c1 {
      scf.for %arg1 = %c0 to %c4096 step %c1 {
        memref.store %cst_2, %alloc_4[%arg0, %arg1] : memref<4096x4096xf16>
        memref.store %cst, %alloc_5[%arg0, %arg1] : memref<4096x4096xf32>
      }
    }
    // Run GPU.
    // Run CPU.
    // call @printMemrefF32(%C_row_0_cast) : (memref<*xf32>) -> ()
    // call @printMemrefF16(%C_row_0_cast_gpu) : (memref<*xf16>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @test(%alloc, %alloc_3, %alloc_4) : (memref<4096x4096xf16>, memref<4096x4096xf16>, memref<4096x4096xf16>) -> memref<4096x4096xf16>
    %cast_7 = memref.cast %0 : memref<4096x4096xf16> to memref<*xf16>
    %cast_8 = memref.cast %alloc : memref<4096x4096xf16> to memref<*xf16>
    %cast_9 = memref.cast %alloc_3 : memref<4096x4096xf16> to memref<*xf16>
    %cast_10 = memref.cast %alloc_5 : memref<4096x4096xf32> to memref<*xf32>
    call @gemmF16F16F16(%cast_8, %cast_9, %cast_10) : (memref<*xf16>, memref<*xf16>, memref<*xf32>) -> ()
    call @printAllcloseF16(%cast_7, %cast_10) : (memref<*xf16>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<4096x4096xf16>
    memref.dealloc %alloc_3 : memref<4096x4096xf16>
    memref.dealloc %alloc_4 : memref<4096x4096xf16>
    memref.dealloc %alloc_5 : memref<4096x4096xf32>
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @gemmF16F16F16(memref<*xf16>, memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

