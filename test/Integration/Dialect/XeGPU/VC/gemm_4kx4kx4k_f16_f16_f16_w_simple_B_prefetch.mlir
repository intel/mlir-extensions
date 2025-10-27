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
      // convert layout to 4x8 from 8x4
      // compute address for 8x32 slice
      // create B prefetch tiles and prefetch
      // stage 2 (move 32 elements in the x direction and prefetch next 8x32 tile)
      // stage 3
      // compute the next tile to prefetch inside K loop
      // two 32x16 A tiles from 256x32 WG slice
      // %A_sg_init_tile_1 = xegpu.create_nd_tdesc %A[%C_sg_tile_offset_x, %c16] : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x16xf16>
      //create B tiles
      // %B_sg_init_tile_2 = xegpu.update_nd_offset %B_sg_init_tile_1, [%c0, %c16] :  !xegpu.tensor_desc<32x16xf16>
      // %B_sg_init_tile_3 = xegpu.update_nd_offset %B_sg_init_tile_2, [%c0, %c16] :  !xegpu.tensor_desc<32x16xf16>
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
      %14 = arith.muli %0, %c4 : index
      %15 = arith.muli %14, %1 : index
      %16 = arith.divui %15, %c8 : index
      %17 = arith.remui %15, %c8 : index
      %18 = arith.muli %16, %c8 : index
      %19 = arith.muli %17, %c32 : index
      %20 = arith.addi %5, %19 : index
      %21 = xegpu.create_nd_tdesc %arg1[%18, %20] : memref<4096x4096xf16> -> !xegpu.tensor_desc<8x32xf16>
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
        %94 = arith.remui %arg3, %c256 : index
        %95 = arith.index_cast %94 : index to i32
        %96 = arith.cmpi eq, %95, %c0_i32 : i32
        scf.if %96 {
          xegpu.nbarrier_arrive %44 : !xegpu.nbarrier
        }
        // load A tiles
        // load B tiles
        // prefetch A and B tiles
        // xegpu.prefetch_nd %A_prefetch_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
        // xegpu.prefetch_nd %B_prefetch_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
        //
        // advance A and B prefetch tiles
        // advance A and B tiles
        // %next_A_tile_1 = xegpu.update_nd_offset %A_tile_1, [%c0, %c32] : !xegpu.tensor_desc<32x16xf16>
        // %next_B_tile_2 = xegpu.update_nd_offset %B_tile_2, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16>
        // %next_B_tile_3 = xegpu.update_nd_offset %B_tile_3, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16>
        // do DPAS
        //  barrier wait
        %97 = xegpu.load_nd %arg4 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x32x16xf16>
        %98 = vector.extract %97[0] : vector<32x16xf16> from vector<2x32x16xf16>
        %99 = vector.extract %97[1] : vector<32x16xf16> from vector<2x32x16xf16>
        %100 = xegpu.load_nd %arg5 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x16x16x2xf16>
        %101 = xegpu.load_nd %arg6 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x16x16x2xf16>
        %102 = vector.extract %100[0] : vector<16x16x2xf16> from vector<2x16x16x2xf16>
        %103 = vector.extract %100[1] : vector<16x16x2xf16> from vector<2x16x16x2xf16>
        %104 = vector.extract %101[0] : vector<16x16x2xf16> from vector<2x16x16x2xf16>
        %105 = vector.extract %101[1] : vector<16x16x2xf16> from vector<2x16x16x2xf16>
        xegpu.compile_hint
        xegpu.compile_hint
        %106 = xegpu.update_nd_offset %arg23, [%c0, %c32] : !xegpu.tensor_desc<8x32xf16>
        %107 = xegpu.update_nd_offset %arg24, [%c32, %c0] : !xegpu.tensor_desc<8x32xf16>
        %108 = xegpu.update_nd_offset %arg4, [%c0, %c32] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
        %109 = xegpu.update_nd_offset %arg5, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
        %110 = xegpu.update_nd_offset %arg6, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
        xegpu.compile_hint
        %111 = vector.shape_cast %98 : vector<32x16xf16> to vector<512xf16>
        %112 = vector.shape_cast %99 : vector<32x16xf16> to vector<512xf16>
        %113 = vector.extract_strided_slice %111 {offsets = [0], sizes = [128], strides = [1]} : vector<512xf16> to vector<128xf16>
        %114 = vector.shape_cast %113 : vector<128xf16> to vector<8x16xf16>
        %115 = vector.extract_strided_slice %111 {offsets = [128], sizes = [128], strides = [1]} : vector<512xf16> to vector<128xf16>
        %116 = vector.shape_cast %115 : vector<128xf16> to vector<8x16xf16>
        %117 = vector.extract_strided_slice %111 {offsets = [256], sizes = [128], strides = [1]} : vector<512xf16> to vector<128xf16>
        %118 = vector.shape_cast %117 : vector<128xf16> to vector<8x16xf16>
        %119 = vector.extract_strided_slice %111 {offsets = [384], sizes = [128], strides = [1]} : vector<512xf16> to vector<128xf16>
        %120 = vector.shape_cast %119 : vector<128xf16> to vector<8x16xf16>
        %121 = vector.extract_strided_slice %112 {offsets = [0], sizes = [128], strides = [1]} : vector<512xf16> to vector<128xf16>
        %122 = vector.shape_cast %121 : vector<128xf16> to vector<8x16xf16>
        %123 = vector.extract_strided_slice %112 {offsets = [128], sizes = [128], strides = [1]} : vector<512xf16> to vector<128xf16>
        %124 = vector.shape_cast %123 : vector<128xf16> to vector<8x16xf16>
        %125 = vector.extract_strided_slice %112 {offsets = [256], sizes = [128], strides = [1]} : vector<512xf16> to vector<128xf16>
        %126 = vector.shape_cast %125 : vector<128xf16> to vector<8x16xf16>
        %127 = vector.extract_strided_slice %112 {offsets = [384], sizes = [128], strides = [1]} : vector<512xf16> to vector<128xf16>
        %128 = vector.shape_cast %127 : vector<128xf16> to vector<8x16xf16>
        %129 = vector.shape_cast %102 : vector<16x16x2xf16> to vector<512xf16>
        %130 = vector.shape_cast %103 : vector<16x16x2xf16> to vector<512xf16>
        %131 = vector.shape_cast %104 : vector<16x16x2xf16> to vector<512xf16>
        %132 = vector.shape_cast %105 : vector<16x16x2xf16> to vector<512xf16>
        %133 = vector.extract_strided_slice %129 {offsets = [0], sizes = [256], strides = [1]} : vector<512xf16> to vector<256xf16>
        %134 = vector.shape_cast %133 : vector<256xf16> to vector<8x16x2xf16>
        %135 = vector.extract_strided_slice %129 {offsets = [256], sizes = [256], strides = [1]} : vector<512xf16> to vector<256xf16>
        %136 = vector.shape_cast %135 : vector<256xf16> to vector<8x16x2xf16>
        %137 = vector.extract_strided_slice %130 {offsets = [0], sizes = [256], strides = [1]} : vector<512xf16> to vector<256xf16>
        %138 = vector.shape_cast %137 : vector<256xf16> to vector<8x16x2xf16>
        %139 = vector.extract_strided_slice %130 {offsets = [256], sizes = [256], strides = [1]} : vector<512xf16> to vector<256xf16>
        %140 = vector.shape_cast %139 : vector<256xf16> to vector<8x16x2xf16>
        %141 = vector.extract_strided_slice %131 {offsets = [0], sizes = [256], strides = [1]} : vector<512xf16> to vector<256xf16>
        %142 = vector.shape_cast %141 : vector<256xf16> to vector<8x16x2xf16>
        %143 = vector.extract_strided_slice %131 {offsets = [256], sizes = [256], strides = [1]} : vector<512xf16> to vector<256xf16>
        %144 = vector.shape_cast %143 : vector<256xf16> to vector<8x16x2xf16>
        %145 = vector.extract_strided_slice %132 {offsets = [0], sizes = [256], strides = [1]} : vector<512xf16> to vector<256xf16>
        %146 = vector.shape_cast %145 : vector<256xf16> to vector<8x16x2xf16>
        %147 = vector.extract_strided_slice %132 {offsets = [256], sizes = [256], strides = [1]} : vector<512xf16> to vector<256xf16>
        %148 = vector.shape_cast %147 : vector<256xf16> to vector<8x16x2xf16>
        xegpu.compile_hint
        %149 = xegpu.dpas %114, %134, %arg7 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %150 = xegpu.dpas %122, %136, %149 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %151 = xegpu.dpas %116, %134, %arg11 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %152 = xegpu.dpas %124, %136, %151 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %153 = xegpu.dpas %118, %134, %arg15 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %154 = xegpu.dpas %126, %136, %153 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %155 = xegpu.dpas %120, %134, %arg19 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %156 = xegpu.dpas %128, %136, %155 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %157 = xegpu.dpas %114, %138, %arg8 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %158 = xegpu.dpas %122, %140, %157 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %159 = xegpu.dpas %116, %138, %arg12 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %160 = xegpu.dpas %124, %140, %159 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %161 = xegpu.dpas %118, %138, %arg16 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %162 = xegpu.dpas %126, %140, %161 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %163 = xegpu.dpas %120, %138, %arg20 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %164 = xegpu.dpas %128, %140, %163 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %165 = xegpu.dpas %114, %142, %arg9 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %166 = xegpu.dpas %122, %144, %165 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %167 = xegpu.dpas %116, %142, %arg13 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %168 = xegpu.dpas %124, %144, %167 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %169 = xegpu.dpas %118, %142, %arg17 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %170 = xegpu.dpas %126, %144, %169 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %171 = xegpu.dpas %120, %142, %arg21 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %172 = xegpu.dpas %128, %144, %171 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %173 = xegpu.dpas %114, %146, %arg10 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %174 = xegpu.dpas %122, %148, %173 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %175 = xegpu.dpas %116, %146, %arg14 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %176 = xegpu.dpas %124, %148, %175 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %177 = xegpu.dpas %118, %146, %arg18 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %178 = xegpu.dpas %126, %148, %177 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %179 = xegpu.dpas %120, %146, %arg22 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        xegpu.compile_hint
        %180 = xegpu.dpas %128, %148, %179 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        xegpu.compile_hint
        scf.if %96 {
          xegpu.nbarrier_wait %44 : !xegpu.nbarrier
        }
        scf.yield %108, %109, %110, %150, %158, %166, %174, %152, %160, %168, %176, %154, %162, %170, %178, %156, %164, %172, %180, %106, %107 : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, !xegpu.tensor_desc<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      }
      // trunc all DPAS output tiles to f16
      // each SG needs to store the result of K loop into a 32x64 tile in C matrix. This is organized in 8x16 DPAS tiles
      // in the layout of 4x4x8x16. The max store size HW supoprt in f16 is 8x32. So we combine two 8x16 DPAS tiles
      // horizontally using vector.shuffle to get the required store size. The store layout then will 4x2x8x32 i.e.
      // we have 8 stores of size 8x32 in the layout 4x2.
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
      %62 = vector.shuffle %46, %47 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %63 = vector.shape_cast %62 : vector<16x16xf16> to vector<256xf16>
      %64 = vector.shape_cast %63 : vector<256xf16> to vector<8x32xf16>
      %65 = xegpu.create_nd_tdesc %arg2[%2, %3] : memref<4096x4096xf16> -> !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %64, %65 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint
      %66 = vector.shuffle %48, %49 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %67 = vector.shape_cast %66 : vector<16x16xf16> to vector<256xf16>
      %68 = vector.shape_cast %67 : vector<256xf16> to vector<8x32xf16>
      %69 = xegpu.update_nd_offset %65, [%c0, %c32] : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %68, %69 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint
      %70 = vector.shuffle %50, %51 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %71 = vector.shape_cast %70 : vector<16x16xf16> to vector<256xf16>
      %72 = vector.shape_cast %71 : vector<256xf16> to vector<8x32xf16>
      %73 = xegpu.update_nd_offset %65, [%c8, %c0] : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %72, %73 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint
      %74 = vector.shuffle %52, %53 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %75 = vector.shape_cast %74 : vector<16x16xf16> to vector<256xf16>
      %76 = vector.shape_cast %75 : vector<256xf16> to vector<8x32xf16>
      %77 = xegpu.update_nd_offset %69, [%c8, %c0] : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %76, %77 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint
      %78 = vector.shuffle %54, %55 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %79 = vector.shape_cast %78 : vector<16x16xf16> to vector<256xf16>
      %80 = vector.shape_cast %79 : vector<256xf16> to vector<8x32xf16>
      %81 = xegpu.update_nd_offset %73, [%c8, %c0] : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %80, %81 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint
      %82 = vector.shuffle %56, %57 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %83 = vector.shape_cast %82 : vector<16x16xf16> to vector<256xf16>
      %84 = vector.shape_cast %83 : vector<256xf16> to vector<8x32xf16>
      %85 = xegpu.update_nd_offset %77, [%c8, %c0] : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %84, %85 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint
      %86 = vector.shuffle %58, %59 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %87 = vector.shape_cast %86 : vector<16x16xf16> to vector<256xf16>
      %88 = vector.shape_cast %87 : vector<256xf16> to vector<8x32xf16>
      %89 = xegpu.update_nd_offset %81, [%c8, %c0] : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %88, %89 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint
      %90 = vector.shuffle %60, %61 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %91 = vector.shape_cast %90 : vector<16x16xf16> to vector<256xf16>
      %92 = vector.shape_cast %91 : vector<256xf16> to vector<8x32xf16>
      %93 = xegpu.update_nd_offset %85, [%c8, %c0] : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %92, %93 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
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
    %cst_0 = arith.constant 0.000000e+00 : f16
    %true = arith.constant true
    %cst_1 = arith.constant -5.000000e-01 : f32
    %cst_2 = arith.constant 5.000000e-01 : f32
    %alloc = memref.alloc() : memref<4096x4096xf16>
    %alloc_3 = memref.alloc() : memref<4096x4096xf16>
    %alloc_4 = memref.alloc() : memref<4096x4096xf16>
    %alloc_5 = memref.alloc() : memref<4096x4096xf32>
    %cast = memref.cast %alloc : memref<4096x4096xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%cast, %cst_1, %cst_2, %true) : (memref<*xf16>, f32, f32, i1) -> ()
    %cast_6 = memref.cast %alloc_3 : memref<4096x4096xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%cast_6, %cst_1, %cst_2, %true) : (memref<*xf16>, f32, f32, i1) -> ()
    scf.for %arg0 = %c0 to %c4096 step %c1 {
      scf.for %arg1 = %c0 to %c4096 step %c1 {
        memref.store %cst_0, %alloc_4[%arg0, %arg1] : memref<4096x4096xf16>
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

