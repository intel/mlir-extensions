// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf32>) -> memref<4096x4096xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %memref = gpu.alloc  () : memref<4096x4096xf16>
    gpu.memcpy  %memref, %arg0 : memref<4096x4096xf16>, memref<4096x4096xf16>
    %memref_0 = gpu.alloc  () : memref<4096x4096xf16>
    gpu.memcpy  %memref_0, %arg1 : memref<4096x4096xf16>, memref<4096x4096xf16>
    %memref_1 = gpu.alloc  () : memref<4096x4096xf32>
    gpu.memcpy  %memref_1, %arg2 : memref<4096x4096xf32>, memref<4096x4096xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c16, %c16, %c1) threads in (%c8, %c4, %c1)  args(%memref : memref<4096x4096xf16>, %memref_0 : memref<4096x4096xf16>, %memref_1 : memref<4096x4096xf32>)
    gpu.dealloc  %memref : memref<4096x4096xf16>
    gpu.dealloc  %memref_0 : memref<4096x4096xf16>
    %alloc = memref.alloc() : memref<4096x4096xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<4096x4096xf32>, memref<4096x4096xf32>
    gpu.dealloc  %memref_1 : memref<4096x4096xf32>
    return %alloc : memref<4096x4096xf32>
  }
  gpu.module @test_kernel  {
      // constants
    gpu.func @test_kernel(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
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
      // create A tiles
      //create B tiles
      // init 16 C tiles of size 8x16 each is initialized to 0.0 assuming a zero C matrix
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %global_id_x = gpu.global_id  x
      %global_id_y = gpu.global_id  y
      %0 = arith.remsi %global_id_x, %c8 : index
      %1 = arith.remsi %global_id_y, %c4 : index
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
      %14 = arith.remsi %0, %c4 : index
      %15 = arith.muli %14, %c8 : index
      %16 = arith.muli %1, %c64 : index
      %17 = arith.divsi %0, %c4 : index
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
      %25 = xegpu.create_nd_tdesc %arg0[%2, %c0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<8x16xf16>
      %26 = xegpu.update_nd_offset %25, [%c8, %c0] : !xegpu.tensor_desc<8x16xf16>
      %27 = xegpu.update_nd_offset %26, [%c8, %c0] : !xegpu.tensor_desc<8x16xf16>
      %28 = xegpu.update_nd_offset %27, [%c8, %c0] : !xegpu.tensor_desc<8x16xf16>
      %29 = xegpu.update_nd_offset %25, [%c0, %c16] : !xegpu.tensor_desc<8x16xf16>
      %30 = xegpu.update_nd_offset %29, [%c8, %c0] : !xegpu.tensor_desc<8x16xf16>
      %31 = xegpu.update_nd_offset %30, [%c8, %c0] : !xegpu.tensor_desc<8x16xf16>
      %32 = xegpu.update_nd_offset %31, [%c8, %c0] : !xegpu.tensor_desc<8x16xf16>
      %33 = xegpu.create_nd_tdesc %arg1[%c0, %3] : memref<4096x4096xf16> -> !xegpu.tensor_desc<16x16xf16>
      %34 = xegpu.update_nd_offset %33, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %35 = xegpu.update_nd_offset %34, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %36 = xegpu.update_nd_offset %35, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %37 = xegpu.update_nd_offset %33, [%c16, %c0] : !xegpu.tensor_desc<16x16xf16>
      %38 = xegpu.update_nd_offset %37, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %39 = xegpu.update_nd_offset %38, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %40 = xegpu.update_nd_offset %39, [%c0, %c16] : !xegpu.tensor_desc<16x16xf16>
      %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
      %41 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %42 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %43 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %44 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %45 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %46 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %47 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %48 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %49 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %50 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %51 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %52 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %53 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %54 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %55 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      %56 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      xegpu.alloc_nbarrier 16
      // K loop advances in 32 steps
        // all SGs must arrive here first
        // load A tiles
        // load B tiles
        // prefetch A and B tiles
        //
      %c1_i8 = arith.constant 1 : i8
      %c8_i8 = arith.constant 8 : i8
      %57 = xegpu.init_nbarrier %c1_i8, %c8_i8 : i8, i8 -> !xegpu.nbarrier
      %58:34 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %25, %arg5 = %26, %arg6 = %27, %arg7 = %28, %arg8 = %29, %arg9 = %30, %arg10 = %31, %arg11 = %32, %arg12 = %33, %arg13 = %34, %arg14 = %35, %arg15 = %36, %arg16 = %37, %arg17 = %38, %arg18 = %39, %arg19 = %40, %arg20 = %41, %arg21 = %42, %arg22 = %43, %arg23 = %44, %arg24 = %45, %arg25 = %46, %arg26 = %47, %arg27 = %48, %arg28 = %49, %arg29 = %50, %arg30 = %51, %arg31 = %52, %arg32 = %53, %arg33 = %54, %arg34 = %55, %arg35 = %56, %arg36 = %13, %arg37 = %24) -> (!xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, !xegpu.tensor_desc<8x32xf16>, !xegpu.tensor_desc<8x32xf16>) {
        xegpu.nbarrier_arrive %57 : !xegpu.nbarrier
        %75 = xegpu.load_nd %arg4 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
        %76 = xegpu.load_nd %arg5 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
        %77 = xegpu.load_nd %arg6 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
        %78 = xegpu.load_nd %arg7 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
        %79 = xegpu.load_nd %arg8 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
        %80 = xegpu.load_nd %arg9 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
        %81 = xegpu.load_nd %arg10 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
        %82 = xegpu.load_nd %arg11 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
        %83 = xegpu.load_nd %arg12 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %84 = xegpu.load_nd %arg13 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %85 = xegpu.load_nd %arg14 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %86 = xegpu.load_nd %arg15 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %87 = xegpu.load_nd %arg16 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %88 = xegpu.load_nd %arg17 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %89 = xegpu.load_nd %arg18 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %90 = xegpu.load_nd %arg19 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        xegpu.prefetch_nd %arg36 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xf16>
        xegpu.prefetch_nd %arg37 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xf16>
        xegpu.compile_hint
        // advance A and B prefetch tiles
        // advance A and B tiles
        %91 = xegpu.update_nd_offset %arg36, [%c0, %c32] : !xegpu.tensor_desc<8x32xf16>
        %92 = xegpu.update_nd_offset %arg37, [%c32, %c0] : !xegpu.tensor_desc<8x32xf16>
        %93 = xegpu.update_nd_offset %arg4, [%c0, %c32] : !xegpu.tensor_desc<8x16xf16>
        %94 = xegpu.update_nd_offset %arg5, [%c0, %c32] : !xegpu.tensor_desc<8x16xf16>
        %95 = xegpu.update_nd_offset %arg6, [%c0, %c32] : !xegpu.tensor_desc<8x16xf16>
        %96 = xegpu.update_nd_offset %arg7, [%c0, %c32] : !xegpu.tensor_desc<8x16xf16>
        %97 = xegpu.update_nd_offset %arg8, [%c0, %c32] : !xegpu.tensor_desc<8x16xf16>
        %98 = xegpu.update_nd_offset %arg9, [%c0, %c32] : !xegpu.tensor_desc<8x16xf16>
        %99 = xegpu.update_nd_offset %arg10, [%c0, %c32] : !xegpu.tensor_desc<8x16xf16>
        %100 = xegpu.update_nd_offset %arg11, [%c0, %c32] : !xegpu.tensor_desc<8x16xf16>
        %101 = xegpu.update_nd_offset %arg12, [%c32, %c0] : !xegpu.tensor_desc<16x16xf16>
        %102 = xegpu.update_nd_offset %arg13, [%c32, %c0] : !xegpu.tensor_desc<16x16xf16>
        %103 = xegpu.update_nd_offset %arg14, [%c32, %c0] : !xegpu.tensor_desc<16x16xf16>
        %104 = xegpu.update_nd_offset %arg15, [%c32, %c0] : !xegpu.tensor_desc<16x16xf16>
        %105 = xegpu.update_nd_offset %arg16, [%c32, %c0] : !xegpu.tensor_desc<16x16xf16>
        %106 = xegpu.update_nd_offset %arg17, [%c32, %c0] : !xegpu.tensor_desc<16x16xf16>
        %107 = xegpu.update_nd_offset %arg18, [%c32, %c0] : !xegpu.tensor_desc<16x16xf16>
        %108 = xegpu.update_nd_offset %arg19, [%c32, %c0] : !xegpu.tensor_desc<16x16xf16>
        xegpu.compile_hint
        // do DPAS
        %109 = xegpu.dpas %75, %83, %arg20 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %110 = xegpu.dpas %75, %84, %arg21 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %111 = xegpu.dpas %75, %85, %arg22 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %112 = xegpu.dpas %75, %86, %arg23 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %113 = xegpu.dpas %76, %83, %arg24 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %114 = xegpu.dpas %76, %84, %arg25 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %115 = xegpu.dpas %76, %85, %arg26 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %116 = xegpu.dpas %76, %86, %arg27 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %117 = xegpu.dpas %77, %83, %arg28 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %118 = xegpu.dpas %77, %84, %arg29 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %119 = xegpu.dpas %77, %85, %arg30 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %120 = xegpu.dpas %77, %86, %arg31 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %121 = xegpu.dpas %78, %83, %arg32 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %122 = xegpu.dpas %78, %84, %arg33 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %123 = xegpu.dpas %78, %85, %arg34 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %124 = xegpu.dpas %78, %86, %arg35 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %125 = xegpu.dpas %79, %87, %109 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %126 = xegpu.dpas %79, %88, %110 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %127 = xegpu.dpas %79, %89, %111 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %128 = xegpu.dpas %79, %90, %112 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %129 = xegpu.dpas %80, %87, %113 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %130 = xegpu.dpas %80, %88, %114 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %131 = xegpu.dpas %80, %89, %115 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %132 = xegpu.dpas %80, %90, %116 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %133 = xegpu.dpas %81, %87, %117 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %134 = xegpu.dpas %81, %88, %118 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %135 = xegpu.dpas %81, %89, %119 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %136 = xegpu.dpas %81, %90, %120 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %137 = xegpu.dpas %82, %87, %121 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %138 = xegpu.dpas %82, %88, %122 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %139 = xegpu.dpas %82, %89, %123 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %140 = xegpu.dpas %82, %90, %124 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        xegpu.compile_hint
        //  barrier wait
        xegpu.nbarrier_wait %57 : !xegpu.nbarrier
        scf.yield %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %125, %126, %127, %128, %129, %130, %131, %132, %133, %134, %135, %136, %137, %138, %139, %140, %91, %92 : !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, !xegpu.tensor_desc<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      }
      // each SG needs to write to 32x64 C tile.
      // DPAS output size is 8x16. So each SG needs to write 16 (4x4) DPAS outputs.
      // create 16 address descriptions to cover 8x16 tiles in 4x4 layout within the 32x64 SG C tile.
      // advance 8 in x dim and, advance 16 in y dim
      // row 1
      // row 2
      // row 3
      // row 4
      // do store_nd
      %59 = xegpu.create_nd_tdesc %arg2[%2, %3] : memref<4096x4096xf32> -> !xegpu.tensor_desc<8x16xf32>
      %60 = xegpu.update_nd_offset %59, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      %61 = xegpu.update_nd_offset %60, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      %62 = xegpu.update_nd_offset %61, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      %63 = xegpu.update_nd_offset %59, [%c8, %c0] : !xegpu.tensor_desc<8x16xf32>
      %64 = xegpu.update_nd_offset %63, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      %65 = xegpu.update_nd_offset %64, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      %66 = xegpu.update_nd_offset %65, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      %67 = xegpu.update_nd_offset %59, [%c16, %c0] : !xegpu.tensor_desc<8x16xf32>
      %68 = xegpu.update_nd_offset %67, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      %69 = xegpu.update_nd_offset %68, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      %70 = xegpu.update_nd_offset %69, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      %71 = xegpu.update_nd_offset %59, [%c24, %c0] : !xegpu.tensor_desc<8x16xf32>
      %72 = xegpu.update_nd_offset %71, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      %73 = xegpu.update_nd_offset %72, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      %74 = xegpu.update_nd_offset %73, [%c0, %c16] : !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %58#16, %59 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %58#17, %60 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %58#18, %61 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %58#19, %62 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %58#20, %63 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %58#21, %64 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %58#22, %65 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %58#23, %66 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %58#24, %67 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %58#25, %68 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %58#26, %69 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %58#27, %70 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %58#28, %71 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %58#29, %72 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %58#30, %73 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %58#31, %74 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
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
    %alloc = memref.alloc() : memref<4096x4096xf16>
    %alloc_2 = memref.alloc() : memref<4096x4096xf16>
    %alloc_3 = memref.alloc() : memref<4096x4096xf32>
    %alloc_4 = memref.alloc() : memref<4096x4096xf32>
    %cast = memref.cast %alloc : memref<4096x4096xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%cast, %cst_1, %cst_0, %false) : (memref<*xf16>, f32, f32, i1) -> ()
    %cast_5 = memref.cast %alloc_2 : memref<4096x4096xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%cast_5, %cst_1, %cst_0, %false) : (memref<*xf16>, f32, f32, i1) -> ()
    scf.for %arg0 = %c0 to %c4096 step %c1 {
      scf.for %arg1 = %c0 to %c4096 step %c1 {
        memref.store %cst, %alloc_3[%arg0, %arg1] : memref<4096x4096xf32>
        memref.store %cst, %alloc_4[%arg0, %arg1] : memref<4096x4096xf32>
      }
    }
    // Run GPU.
    // Run CPU.
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @test(%alloc, %alloc_2, %alloc_3) : (memref<4096x4096xf16>, memref<4096x4096xf16>, memref<4096x4096xf32>) -> memref<4096x4096xf32>
    %cast_6 = memref.cast %0 : memref<4096x4096xf32> to memref<*xf32>
    %cast_7 = memref.cast %alloc : memref<4096x4096xf16> to memref<*xf16>
    %cast_8 = memref.cast %alloc_2 : memref<4096x4096xf16> to memref<*xf16>
    %cast_9 = memref.cast %alloc_4 : memref<4096x4096xf32> to memref<*xf32>
    call @gemmF16F16F32(%cast_7, %cast_8, %cast_9) : (memref<*xf16>, memref<*xf16>, memref<*xf32>) -> ()
    call @printAllcloseF32(%cast_6, %cast_9) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<4096x4096xf16>
    memref.dealloc %alloc_2 : memref<4096x4096xf16>
    memref.dealloc %alloc_3 : memref<4096x4096xf32>
    memref.dealloc %alloc_4 : memref<4096x4096xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @gemmF16F16F32(memref<*xf16>, memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

