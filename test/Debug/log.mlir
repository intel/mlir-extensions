// -----// IR Dump After CSE (cse) //----- //
module {
  gpu.module @test_kernels {
    gpu.func @bcast_add_exp(%arg0: vector<1x16xf16>, %arg1: vector<16x1xf16>, %arg2: memref<8x16xf32>) kernel {
      %0 = xetile.broadcast %arg0 [0] : vector<1x16xf16> -> vector<8x16xf16>
      %1 = math.exp %arg1 : vector<16x1xf16>
      %2 = xetile.broadcast %1 [1] : vector<16x1xf16> -> vector<16x16xf16>
      %3 = xetile.tile_mma %0, %2 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
      %4 = xetile.init_tile %arg2[0, 0] : memref<8x16xf32> -> !xetile.tile<8x16xf32>
      xetile.store_tile %3,  %4 : vector<8x16xf32>, !xetile.tile<8x16xf32>
      gpu.return
    }
  }
}


// -----// IR Dump After XeTileInitDuplicate (xetile-init-duplicate) //----- //
gpu.module @test_kernels {
  gpu.func @bcast_add_exp(%arg0: vector<1x16xf16>, %arg1: vector<16x1xf16>, %arg2: memref<8x16xf32>) kernel {
    %0 = xetile.broadcast %arg0 [0] : vector<1x16xf16> -> vector<8x16xf16>
    %1 = math.exp %arg1 : vector<16x1xf16>
    %2 = xetile.broadcast %1 [1] : vector<16x1xf16> -> vector<16x16xf16>
    %3 = xetile.tile_mma %0, %2 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    %4 = xetile.init_tile %arg2[0, 0] : memref<8x16xf32> -> !xetile.tile<8x16xf32>
    xetile.store_tile %3,  %4 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    gpu.return
  }
}

// -----// IR Dump After XeTileCanonicalization (xetile-canonicalization) //----- //
gpu.module @test_kernels {
  gpu.func @bcast_add_exp(%arg0: vector<1x16xf16>, %arg1: vector<16x1xf16>, %arg2: memref<8x16xf32>) kernel {
    %0 = xetile.broadcast %arg0 [0] : vector<1x16xf16> -> vector<8x16xf16>
    %1 = math.exp %arg1 : vector<16x1xf16>
    %2 = xetile.broadcast %1 [1] : vector<16x1xf16> -> vector<16x16xf16>
    %3 = xetile.tile_mma %0, %2 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    %4 = xetile.init_tile %arg2[0, 0] : memref<8x16xf32> -> !xetile.tile<8x16xf32>
    xetile.store_tile %3,  %4 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    gpu.return
  }
}

// -----// IR Dump After XeTileBlocking (xetile-blocking) //----- //
gpu.module @test_kernels {
  gpu.func @bcast_add_exp(%arg0: vector<1x16xf16>, %arg1: vector<16x1xf16>, %arg2: memref<8x16xf32>) kernel {
    %0 = xetile.broadcast %arg0 [0] : vector<1x16xf16> -> vector<8x16xf16>
    %1 = vector.extract_strided_slice %arg1 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %2 = vector.extract_strided_slice %arg1 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %3 = vector.extract_strided_slice %arg1 {offsets = [2, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %4 = vector.extract_strided_slice %arg1 {offsets = [3, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %5 = vector.extract_strided_slice %arg1 {offsets = [4, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %6 = vector.extract_strided_slice %arg1 {offsets = [5, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %7 = vector.extract_strided_slice %arg1 {offsets = [6, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %8 = vector.extract_strided_slice %arg1 {offsets = [7, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %9 = vector.extract_strided_slice %arg1 {offsets = [8, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %10 = vector.extract_strided_slice %arg1 {offsets = [9, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %11 = vector.extract_strided_slice %arg1 {offsets = [10, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %12 = vector.extract_strided_slice %arg1 {offsets = [11, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %13 = vector.extract_strided_slice %arg1 {offsets = [12, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %14 = vector.extract_strided_slice %arg1 {offsets = [13, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %15 = vector.extract_strided_slice %arg1 {offsets = [14, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %16 = vector.extract_strided_slice %arg1 {offsets = [15, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %17 = math.exp %1 : vector<1x1xf16>
    %18 = math.exp %2 : vector<1x1xf16>
    %19 = math.exp %3 : vector<1x1xf16>
    %20 = math.exp %4 : vector<1x1xf16>
    %21 = math.exp %5 : vector<1x1xf16>
    %22 = math.exp %6 : vector<1x1xf16>
    %23 = math.exp %7 : vector<1x1xf16>
    %24 = math.exp %8 : vector<1x1xf16>
    %25 = math.exp %9 : vector<1x1xf16>
    %26 = math.exp %10 : vector<1x1xf16>
    %27 = math.exp %11 : vector<1x1xf16>
    %28 = math.exp %12 : vector<1x1xf16>
    %29 = math.exp %13 : vector<1x1xf16>
    %30 = math.exp %14 : vector<1x1xf16>
    %31 = math.exp %15 : vector<1x1xf16>
    %32 = math.exp %16 : vector<1x1xf16>
    %33 = vector.extract %17[0, 0] : f16 from vector<1x1xf16>
    %34 = vector.splat %33 : vector<1x16xf16>
    %35 = vector.extract %18[0, 0] : f16 from vector<1x1xf16>
    %36 = vector.splat %35 : vector<1x16xf16>
    %37 = vector.extract %19[0, 0] : f16 from vector<1x1xf16>
    %38 = vector.splat %37 : vector<1x16xf16>
    %39 = vector.extract %20[0, 0] : f16 from vector<1x1xf16>
    %40 = vector.splat %39 : vector<1x16xf16>
    %41 = vector.extract %21[0, 0] : f16 from vector<1x1xf16>
    %42 = vector.splat %41 : vector<1x16xf16>
    %43 = vector.extract %22[0, 0] : f16 from vector<1x1xf16>
    %44 = vector.splat %43 : vector<1x16xf16>
    %45 = vector.extract %23[0, 0] : f16 from vector<1x1xf16>
    %46 = vector.splat %45 : vector<1x16xf16>
    %47 = vector.extract %24[0, 0] : f16 from vector<1x1xf16>
    %48 = vector.splat %47 : vector<1x16xf16>
    %49 = vector.extract %25[0, 0] : f16 from vector<1x1xf16>
    %50 = vector.splat %49 : vector<1x16xf16>
    %51 = vector.extract %26[0, 0] : f16 from vector<1x1xf16>
    %52 = vector.splat %51 : vector<1x16xf16>
    %53 = vector.extract %27[0, 0] : f16 from vector<1x1xf16>
    %54 = vector.splat %53 : vector<1x16xf16>
    %55 = vector.extract %28[0, 0] : f16 from vector<1x1xf16>
    %56 = vector.splat %55 : vector<1x16xf16>
    %57 = vector.extract %29[0, 0] : f16 from vector<1x1xf16>
    %58 = vector.splat %57 : vector<1x16xf16>
    %59 = vector.extract %30[0, 0] : f16 from vector<1x1xf16>
    %60 = vector.splat %59 : vector<1x16xf16>
    %61 = vector.extract %31[0, 0] : f16 from vector<1x1xf16>
    %62 = vector.splat %61 : vector<1x16xf16>
    %63 = vector.extract %32[0, 0] : f16 from vector<1x1xf16>
    %64 = vector.splat %63 : vector<1x16xf16>
    %65 = vector.shuffle %34, %36 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %66 = vector.shuffle %38, %40 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %67 = vector.shuffle %42, %44 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %68 = vector.shuffle %46, %48 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %69 = vector.shuffle %50, %52 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %70 = vector.shuffle %54, %56 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %71 = vector.shuffle %58, %60 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %72 = vector.shuffle %62, %64 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %73 = vector.shuffle %65, %66 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %74 = vector.shuffle %67, %68 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %75 = vector.shuffle %69, %70 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %76 = vector.shuffle %71, %72 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %77 = vector.shuffle %73, %74 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x16xf16>, vector<4x16xf16>
    %78 = vector.shuffle %75, %76 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x16xf16>, vector<4x16xf16>
    %79 = vector.shuffle %77, %78 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf16>, vector<8x16xf16>
    %80 = xetile.tile_mma %0, %79 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    %81 = xetile.init_tile %arg2[0, 0] : memref<8x16xf32> -> !xetile.tile<8x16xf32>
    xetile.store_tile %80,  %81 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    gpu.return
  }
}

// -----// IR Dump After CSE (cse) //----- //
gpu.module @test_kernels {
  gpu.func @bcast_add_exp(%arg0: vector<1x16xf16>, %arg1: vector<16x1xf16>, %arg2: memref<8x16xf32>) kernel {
    %0 = xetile.broadcast %arg0 [0] : vector<1x16xf16> -> vector<8x16xf16>
    %1 = vector.extract_strided_slice %arg1 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %2 = vector.extract_strided_slice %arg1 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %3 = vector.extract_strided_slice %arg1 {offsets = [2, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %4 = vector.extract_strided_slice %arg1 {offsets = [3, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %5 = vector.extract_strided_slice %arg1 {offsets = [4, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %6 = vector.extract_strided_slice %arg1 {offsets = [5, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %7 = vector.extract_strided_slice %arg1 {offsets = [6, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %8 = vector.extract_strided_slice %arg1 {offsets = [7, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %9 = vector.extract_strided_slice %arg1 {offsets = [8, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %10 = vector.extract_strided_slice %arg1 {offsets = [9, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %11 = vector.extract_strided_slice %arg1 {offsets = [10, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %12 = vector.extract_strided_slice %arg1 {offsets = [11, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %13 = vector.extract_strided_slice %arg1 {offsets = [12, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %14 = vector.extract_strided_slice %arg1 {offsets = [13, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %15 = vector.extract_strided_slice %arg1 {offsets = [14, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %16 = vector.extract_strided_slice %arg1 {offsets = [15, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %17 = math.exp %1 : vector<1x1xf16>
    %18 = math.exp %2 : vector<1x1xf16>
    %19 = math.exp %3 : vector<1x1xf16>
    %20 = math.exp %4 : vector<1x1xf16>
    %21 = math.exp %5 : vector<1x1xf16>
    %22 = math.exp %6 : vector<1x1xf16>
    %23 = math.exp %7 : vector<1x1xf16>
    %24 = math.exp %8 : vector<1x1xf16>
    %25 = math.exp %9 : vector<1x1xf16>
    %26 = math.exp %10 : vector<1x1xf16>
    %27 = math.exp %11 : vector<1x1xf16>
    %28 = math.exp %12 : vector<1x1xf16>
    %29 = math.exp %13 : vector<1x1xf16>
    %30 = math.exp %14 : vector<1x1xf16>
    %31 = math.exp %15 : vector<1x1xf16>
    %32 = math.exp %16 : vector<1x1xf16>
    %33 = vector.extract %17[0, 0] : f16 from vector<1x1xf16>
    %34 = vector.splat %33 : vector<1x16xf16>
    %35 = vector.extract %18[0, 0] : f16 from vector<1x1xf16>
    %36 = vector.splat %35 : vector<1x16xf16>
    %37 = vector.extract %19[0, 0] : f16 from vector<1x1xf16>
    %38 = vector.splat %37 : vector<1x16xf16>
    %39 = vector.extract %20[0, 0] : f16 from vector<1x1xf16>
    %40 = vector.splat %39 : vector<1x16xf16>
    %41 = vector.extract %21[0, 0] : f16 from vector<1x1xf16>
    %42 = vector.splat %41 : vector<1x16xf16>
    %43 = vector.extract %22[0, 0] : f16 from vector<1x1xf16>
    %44 = vector.splat %43 : vector<1x16xf16>
    %45 = vector.extract %23[0, 0] : f16 from vector<1x1xf16>
    %46 = vector.splat %45 : vector<1x16xf16>
    %47 = vector.extract %24[0, 0] : f16 from vector<1x1xf16>
    %48 = vector.splat %47 : vector<1x16xf16>
    %49 = vector.extract %25[0, 0] : f16 from vector<1x1xf16>
    %50 = vector.splat %49 : vector<1x16xf16>
    %51 = vector.extract %26[0, 0] : f16 from vector<1x1xf16>
    %52 = vector.splat %51 : vector<1x16xf16>
    %53 = vector.extract %27[0, 0] : f16 from vector<1x1xf16>
    %54 = vector.splat %53 : vector<1x16xf16>
    %55 = vector.extract %28[0, 0] : f16 from vector<1x1xf16>
    %56 = vector.splat %55 : vector<1x16xf16>
    %57 = vector.extract %29[0, 0] : f16 from vector<1x1xf16>
    %58 = vector.splat %57 : vector<1x16xf16>
    %59 = vector.extract %30[0, 0] : f16 from vector<1x1xf16>
    %60 = vector.splat %59 : vector<1x16xf16>
    %61 = vector.extract %31[0, 0] : f16 from vector<1x1xf16>
    %62 = vector.splat %61 : vector<1x16xf16>
    %63 = vector.extract %32[0, 0] : f16 from vector<1x1xf16>
    %64 = vector.splat %63 : vector<1x16xf16>
    %65 = vector.shuffle %34, %36 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %66 = vector.shuffle %38, %40 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %67 = vector.shuffle %42, %44 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %68 = vector.shuffle %46, %48 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %69 = vector.shuffle %50, %52 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %70 = vector.shuffle %54, %56 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %71 = vector.shuffle %58, %60 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %72 = vector.shuffle %62, %64 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %73 = vector.shuffle %65, %66 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %74 = vector.shuffle %67, %68 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %75 = vector.shuffle %69, %70 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %76 = vector.shuffle %71, %72 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %77 = vector.shuffle %73, %74 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x16xf16>, vector<4x16xf16>
    %78 = vector.shuffle %75, %76 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x16xf16>, vector<4x16xf16>
    %79 = vector.shuffle %77, %78 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf16>, vector<8x16xf16>
    %80 = xetile.tile_mma %0, %79 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    %81 = xetile.init_tile %arg2[0, 0] : memref<8x16xf32> -> !xetile.tile<8x16xf32>
    xetile.store_tile %80,  %81 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    gpu.return
  }
}

// -----// IR Dump After ConvertXeTileToXeGPU (convert-xetile-to-xegpu) //----- //
gpu.module @test_kernels {
  gpu.func @bcast_add_exp(%arg0: vector<1x16xf16>, %arg1: vector<16x1xf16>, %arg2: memref<8x16xf32>) kernel {
    %0 = vector.shape_cast %arg0 : vector<1x16xf16> to vector<16xf16>
    %1 = vector.broadcast %0 : vector<16xf16> to vector<8x16xf16>
    %2 = vector.extract_strided_slice %arg1 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %3 = vector.extract_strided_slice %arg1 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %4 = vector.extract_strided_slice %arg1 {offsets = [2, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %5 = vector.extract_strided_slice %arg1 {offsets = [3, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %6 = vector.extract_strided_slice %arg1 {offsets = [4, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %7 = vector.extract_strided_slice %arg1 {offsets = [5, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %8 = vector.extract_strided_slice %arg1 {offsets = [6, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %9 = vector.extract_strided_slice %arg1 {offsets = [7, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %10 = vector.extract_strided_slice %arg1 {offsets = [8, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %11 = vector.extract_strided_slice %arg1 {offsets = [9, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %12 = vector.extract_strided_slice %arg1 {offsets = [10, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %13 = vector.extract_strided_slice %arg1 {offsets = [11, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %14 = vector.extract_strided_slice %arg1 {offsets = [12, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %15 = vector.extract_strided_slice %arg1 {offsets = [13, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %16 = vector.extract_strided_slice %arg1 {offsets = [14, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %17 = vector.extract_strided_slice %arg1 {offsets = [15, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %18 = math.exp %2 : vector<1x1xf16>
    %19 = math.exp %3 : vector<1x1xf16>
    %20 = math.exp %4 : vector<1x1xf16>
    %21 = math.exp %5 : vector<1x1xf16>
    %22 = math.exp %6 : vector<1x1xf16>
    %23 = math.exp %7 : vector<1x1xf16>
    %24 = math.exp %8 : vector<1x1xf16>
    %25 = math.exp %9 : vector<1x1xf16>
    %26 = math.exp %10 : vector<1x1xf16>
    %27 = math.exp %11 : vector<1x1xf16>
    %28 = math.exp %12 : vector<1x1xf16>
    %29 = math.exp %13 : vector<1x1xf16>
    %30 = math.exp %14 : vector<1x1xf16>
    %31 = math.exp %15 : vector<1x1xf16>
    %32 = math.exp %16 : vector<1x1xf16>
    %33 = math.exp %17 : vector<1x1xf16>
    %34 = vector.extract %18[0, 0] : f16 from vector<1x1xf16>
    %35 = vector.splat %34 : vector<1x16xf16>
    %36 = vector.extract %19[0, 0] : f16 from vector<1x1xf16>
    %37 = vector.splat %36 : vector<1x16xf16>
    %38 = vector.extract %20[0, 0] : f16 from vector<1x1xf16>
    %39 = vector.splat %38 : vector<1x16xf16>
    %40 = vector.extract %21[0, 0] : f16 from vector<1x1xf16>
    %41 = vector.splat %40 : vector<1x16xf16>
    %42 = vector.extract %22[0, 0] : f16 from vector<1x1xf16>
    %43 = vector.splat %42 : vector<1x16xf16>
    %44 = vector.extract %23[0, 0] : f16 from vector<1x1xf16>
    %45 = vector.splat %44 : vector<1x16xf16>
    %46 = vector.extract %24[0, 0] : f16 from vector<1x1xf16>
    %47 = vector.splat %46 : vector<1x16xf16>
    %48 = vector.extract %25[0, 0] : f16 from vector<1x1xf16>
    %49 = vector.splat %48 : vector<1x16xf16>
    %50 = vector.extract %26[0, 0] : f16 from vector<1x1xf16>
    %51 = vector.splat %50 : vector<1x16xf16>
    %52 = vector.extract %27[0, 0] : f16 from vector<1x1xf16>
    %53 = vector.splat %52 : vector<1x16xf16>
    %54 = vector.extract %28[0, 0] : f16 from vector<1x1xf16>
    %55 = vector.splat %54 : vector<1x16xf16>
    %56 = vector.extract %29[0, 0] : f16 from vector<1x1xf16>
    %57 = vector.splat %56 : vector<1x16xf16>
    %58 = vector.extract %30[0, 0] : f16 from vector<1x1xf16>
    %59 = vector.splat %58 : vector<1x16xf16>
    %60 = vector.extract %31[0, 0] : f16 from vector<1x1xf16>
    %61 = vector.splat %60 : vector<1x16xf16>
    %62 = vector.extract %32[0, 0] : f16 from vector<1x1xf16>
    %63 = vector.splat %62 : vector<1x16xf16>
    %64 = vector.extract %33[0, 0] : f16 from vector<1x1xf16>
    %65 = vector.splat %64 : vector<1x16xf16>
    %66 = vector.shuffle %35, %37 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %67 = vector.shuffle %39, %41 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %68 = vector.shuffle %43, %45 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %69 = vector.shuffle %47, %49 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %70 = vector.shuffle %51, %53 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %71 = vector.shuffle %55, %57 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %72 = vector.shuffle %59, %61 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %73 = vector.shuffle %63, %65 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %74 = vector.shuffle %66, %67 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %75 = vector.shuffle %68, %69 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %76 = vector.shuffle %70, %71 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %77 = vector.shuffle %72, %73 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %78 = vector.shuffle %74, %75 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x16xf16>, vector<4x16xf16>
    %79 = vector.shuffle %76, %77 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x16xf16>, vector<4x16xf16>
    %80 = vector.shuffle %78, %79 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf16>, vector<8x16xf16>
    %81 = xegpu.dpas %1, %80 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    %82 = xegpu.create_nd_tdesc %arg2[0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %81, %82 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    gpu.return
  }
}

// -----// IR Dump After CSE (cse) //----- //
gpu.module @test_kernels {
  gpu.func @bcast_add_exp(%arg0: vector<1x16xf16>, %arg1: vector<16x1xf16>, %arg2: memref<8x16xf32>) kernel {
    %0 = vector.shape_cast %arg0 : vector<1x16xf16> to vector<16xf16>
    %1 = vector.broadcast %0 : vector<16xf16> to vector<8x16xf16>
    %2 = vector.extract_strided_slice %arg1 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %3 = vector.extract_strided_slice %arg1 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %4 = vector.extract_strided_slice %arg1 {offsets = [2, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %5 = vector.extract_strided_slice %arg1 {offsets = [3, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %6 = vector.extract_strided_slice %arg1 {offsets = [4, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %7 = vector.extract_strided_slice %arg1 {offsets = [5, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %8 = vector.extract_strided_slice %arg1 {offsets = [6, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %9 = vector.extract_strided_slice %arg1 {offsets = [7, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %10 = vector.extract_strided_slice %arg1 {offsets = [8, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %11 = vector.extract_strided_slice %arg1 {offsets = [9, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %12 = vector.extract_strided_slice %arg1 {offsets = [10, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %13 = vector.extract_strided_slice %arg1 {offsets = [11, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %14 = vector.extract_strided_slice %arg1 {offsets = [12, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %15 = vector.extract_strided_slice %arg1 {offsets = [13, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %16 = vector.extract_strided_slice %arg1 {offsets = [14, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %17 = vector.extract_strided_slice %arg1 {offsets = [15, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %18 = math.exp %2 : vector<1x1xf16>
    %19 = math.exp %3 : vector<1x1xf16>
    %20 = math.exp %4 : vector<1x1xf16>
    %21 = math.exp %5 : vector<1x1xf16>
    %22 = math.exp %6 : vector<1x1xf16>
    %23 = math.exp %7 : vector<1x1xf16>
    %24 = math.exp %8 : vector<1x1xf16>
    %25 = math.exp %9 : vector<1x1xf16>
    %26 = math.exp %10 : vector<1x1xf16>
    %27 = math.exp %11 : vector<1x1xf16>
    %28 = math.exp %12 : vector<1x1xf16>
    %29 = math.exp %13 : vector<1x1xf16>
    %30 = math.exp %14 : vector<1x1xf16>
    %31 = math.exp %15 : vector<1x1xf16>
    %32 = math.exp %16 : vector<1x1xf16>
    %33 = math.exp %17 : vector<1x1xf16>
    %34 = vector.extract %18[0, 0] : f16 from vector<1x1xf16>
    %35 = vector.splat %34 : vector<1x16xf16>
    %36 = vector.extract %19[0, 0] : f16 from vector<1x1xf16>
    %37 = vector.splat %36 : vector<1x16xf16>
    %38 = vector.extract %20[0, 0] : f16 from vector<1x1xf16>
    %39 = vector.splat %38 : vector<1x16xf16>
    %40 = vector.extract %21[0, 0] : f16 from vector<1x1xf16>
    %41 = vector.splat %40 : vector<1x16xf16>
    %42 = vector.extract %22[0, 0] : f16 from vector<1x1xf16>
    %43 = vector.splat %42 : vector<1x16xf16>
    %44 = vector.extract %23[0, 0] : f16 from vector<1x1xf16>
    %45 = vector.splat %44 : vector<1x16xf16>
    %46 = vector.extract %24[0, 0] : f16 from vector<1x1xf16>
    %47 = vector.splat %46 : vector<1x16xf16>
    %48 = vector.extract %25[0, 0] : f16 from vector<1x1xf16>
    %49 = vector.splat %48 : vector<1x16xf16>
    %50 = vector.extract %26[0, 0] : f16 from vector<1x1xf16>
    %51 = vector.splat %50 : vector<1x16xf16>
    %52 = vector.extract %27[0, 0] : f16 from vector<1x1xf16>
    %53 = vector.splat %52 : vector<1x16xf16>
    %54 = vector.extract %28[0, 0] : f16 from vector<1x1xf16>
    %55 = vector.splat %54 : vector<1x16xf16>
    %56 = vector.extract %29[0, 0] : f16 from vector<1x1xf16>
    %57 = vector.splat %56 : vector<1x16xf16>
    %58 = vector.extract %30[0, 0] : f16 from vector<1x1xf16>
    %59 = vector.splat %58 : vector<1x16xf16>
    %60 = vector.extract %31[0, 0] : f16 from vector<1x1xf16>
    %61 = vector.splat %60 : vector<1x16xf16>
    %62 = vector.extract %32[0, 0] : f16 from vector<1x1xf16>
    %63 = vector.splat %62 : vector<1x16xf16>
    %64 = vector.extract %33[0, 0] : f16 from vector<1x1xf16>
    %65 = vector.splat %64 : vector<1x16xf16>
    %66 = vector.shuffle %35, %37 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %67 = vector.shuffle %39, %41 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %68 = vector.shuffle %43, %45 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %69 = vector.shuffle %47, %49 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %70 = vector.shuffle %51, %53 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %71 = vector.shuffle %55, %57 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %72 = vector.shuffle %59, %61 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %73 = vector.shuffle %63, %65 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %74 = vector.shuffle %66, %67 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %75 = vector.shuffle %68, %69 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %76 = vector.shuffle %70, %71 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %77 = vector.shuffle %72, %73 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %78 = vector.shuffle %74, %75 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x16xf16>, vector<4x16xf16>
    %79 = vector.shuffle %76, %77 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x16xf16>, vector<4x16xf16>
    %80 = vector.shuffle %78, %79 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf16>, vector<8x16xf16>
    %81 = xegpu.dpas %1, %80 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    %82 = xegpu.create_nd_tdesc %arg2[0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %81, %82 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    gpu.return
  }
}

// -----// IR Dump After HoistTranspose (imex-xegpu-hoist-transpose) //----- //
gpu.module @test_kernels {
  gpu.func @bcast_add_exp(%arg0: vector<1x16xf16>, %arg1: vector<16x1xf16>, %arg2: memref<8x16xf32>) kernel {
    %0 = vector.shape_cast %arg0 : vector<1x16xf16> to vector<16xf16>
    %1 = vector.broadcast %0 : vector<16xf16> to vector<8x16xf16>
    %2 = vector.extract_strided_slice %arg1 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %3 = vector.extract_strided_slice %arg1 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %4 = vector.extract_strided_slice %arg1 {offsets = [2, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %5 = vector.extract_strided_slice %arg1 {offsets = [3, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %6 = vector.extract_strided_slice %arg1 {offsets = [4, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %7 = vector.extract_strided_slice %arg1 {offsets = [5, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %8 = vector.extract_strided_slice %arg1 {offsets = [6, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %9 = vector.extract_strided_slice %arg1 {offsets = [7, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %10 = vector.extract_strided_slice %arg1 {offsets = [8, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %11 = vector.extract_strided_slice %arg1 {offsets = [9, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %12 = vector.extract_strided_slice %arg1 {offsets = [10, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %13 = vector.extract_strided_slice %arg1 {offsets = [11, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %14 = vector.extract_strided_slice %arg1 {offsets = [12, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %15 = vector.extract_strided_slice %arg1 {offsets = [13, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %16 = vector.extract_strided_slice %arg1 {offsets = [14, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %17 = vector.extract_strided_slice %arg1 {offsets = [15, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %18 = math.exp %2 : vector<1x1xf16>
    %19 = math.exp %3 : vector<1x1xf16>
    %20 = math.exp %4 : vector<1x1xf16>
    %21 = math.exp %5 : vector<1x1xf16>
    %22 = math.exp %6 : vector<1x1xf16>
    %23 = math.exp %7 : vector<1x1xf16>
    %24 = math.exp %8 : vector<1x1xf16>
    %25 = math.exp %9 : vector<1x1xf16>
    %26 = math.exp %10 : vector<1x1xf16>
    %27 = math.exp %11 : vector<1x1xf16>
    %28 = math.exp %12 : vector<1x1xf16>
    %29 = math.exp %13 : vector<1x1xf16>
    %30 = math.exp %14 : vector<1x1xf16>
    %31 = math.exp %15 : vector<1x1xf16>
    %32 = math.exp %16 : vector<1x1xf16>
    %33 = math.exp %17 : vector<1x1xf16>
    %34 = vector.extract %18[0, 0] : f16 from vector<1x1xf16>
    %35 = vector.splat %34 : vector<1x16xf16>
    %36 = vector.extract %19[0, 0] : f16 from vector<1x1xf16>
    %37 = vector.splat %36 : vector<1x16xf16>
    %38 = vector.extract %20[0, 0] : f16 from vector<1x1xf16>
    %39 = vector.splat %38 : vector<1x16xf16>
    %40 = vector.extract %21[0, 0] : f16 from vector<1x1xf16>
    %41 = vector.splat %40 : vector<1x16xf16>
    %42 = vector.extract %22[0, 0] : f16 from vector<1x1xf16>
    %43 = vector.splat %42 : vector<1x16xf16>
    %44 = vector.extract %23[0, 0] : f16 from vector<1x1xf16>
    %45 = vector.splat %44 : vector<1x16xf16>
    %46 = vector.extract %24[0, 0] : f16 from vector<1x1xf16>
    %47 = vector.splat %46 : vector<1x16xf16>
    %48 = vector.extract %25[0, 0] : f16 from vector<1x1xf16>
    %49 = vector.splat %48 : vector<1x16xf16>
    %50 = vector.extract %26[0, 0] : f16 from vector<1x1xf16>
    %51 = vector.splat %50 : vector<1x16xf16>
    %52 = vector.extract %27[0, 0] : f16 from vector<1x1xf16>
    %53 = vector.splat %52 : vector<1x16xf16>
    %54 = vector.extract %28[0, 0] : f16 from vector<1x1xf16>
    %55 = vector.splat %54 : vector<1x16xf16>
    %56 = vector.extract %29[0, 0] : f16 from vector<1x1xf16>
    %57 = vector.splat %56 : vector<1x16xf16>
    %58 = vector.extract %30[0, 0] : f16 from vector<1x1xf16>
    %59 = vector.splat %58 : vector<1x16xf16>
    %60 = vector.extract %31[0, 0] : f16 from vector<1x1xf16>
    %61 = vector.splat %60 : vector<1x16xf16>
    %62 = vector.extract %32[0, 0] : f16 from vector<1x1xf16>
    %63 = vector.splat %62 : vector<1x16xf16>
    %64 = vector.extract %33[0, 0] : f16 from vector<1x1xf16>
    %65 = vector.splat %64 : vector<1x16xf16>
    %66 = vector.shuffle %35, %37 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %67 = vector.shuffle %39, %41 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %68 = vector.shuffle %43, %45 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %69 = vector.shuffle %47, %49 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %70 = vector.shuffle %51, %53 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %71 = vector.shuffle %55, %57 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %72 = vector.shuffle %59, %61 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %73 = vector.shuffle %63, %65 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %74 = vector.shuffle %66, %67 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %75 = vector.shuffle %68, %69 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %76 = vector.shuffle %70, %71 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %77 = vector.shuffle %72, %73 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %78 = vector.shuffle %74, %75 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x16xf16>, vector<4x16xf16>
    %79 = vector.shuffle %76, %77 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x16xf16>, vector<4x16xf16>
    %80 = vector.shuffle %78, %79 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf16>, vector<8x16xf16>
    %81 = xegpu.dpas %1, %80 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    %82 = xegpu.create_nd_tdesc %arg2[0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %81, %82 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    gpu.return
  }
}

// -----// IR Dump After VnniTransformation (imex-xegpu-apply-vnni-transformation) //----- //
gpu.module @test_kernels {
  gpu.func @bcast_add_exp(%arg0: vector<1x16xf16>, %arg1: vector<16x1xf16>, %arg2: memref<8x16xf32>) kernel {
    %0 = vector.shape_cast %arg0 : vector<1x16xf16> to vector<16xf16>
    %1 = vector.broadcast %0 : vector<16xf16> to vector<8x16xf16>
    %2 = vector.extract_strided_slice %arg1 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %3 = vector.extract_strided_slice %arg1 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %4 = vector.extract_strided_slice %arg1 {offsets = [2, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %5 = vector.extract_strided_slice %arg1 {offsets = [3, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %6 = vector.extract_strided_slice %arg1 {offsets = [4, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %7 = vector.extract_strided_slice %arg1 {offsets = [5, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %8 = vector.extract_strided_slice %arg1 {offsets = [6, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %9 = vector.extract_strided_slice %arg1 {offsets = [7, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %10 = vector.extract_strided_slice %arg1 {offsets = [8, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %11 = vector.extract_strided_slice %arg1 {offsets = [9, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %12 = vector.extract_strided_slice %arg1 {offsets = [10, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %13 = vector.extract_strided_slice %arg1 {offsets = [11, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %14 = vector.extract_strided_slice %arg1 {offsets = [12, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %15 = vector.extract_strided_slice %arg1 {offsets = [13, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %16 = vector.extract_strided_slice %arg1 {offsets = [14, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %17 = vector.extract_strided_slice %arg1 {offsets = [15, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %18 = math.exp %2 : vector<1x1xf16>
    %19 = math.exp %3 : vector<1x1xf16>
    %20 = math.exp %4 : vector<1x1xf16>
    %21 = math.exp %5 : vector<1x1xf16>
    %22 = math.exp %6 : vector<1x1xf16>
    %23 = math.exp %7 : vector<1x1xf16>
    %24 = math.exp %8 : vector<1x1xf16>
    %25 = math.exp %9 : vector<1x1xf16>
    %26 = math.exp %10 : vector<1x1xf16>
    %27 = math.exp %11 : vector<1x1xf16>
    %28 = math.exp %12 : vector<1x1xf16>
    %29 = math.exp %13 : vector<1x1xf16>
    %30 = math.exp %14 : vector<1x1xf16>
    %31 = math.exp %15 : vector<1x1xf16>
    %32 = math.exp %16 : vector<1x1xf16>
    %33 = math.exp %17 : vector<1x1xf16>
    %34 = vector.extract %18[0, 0] : f16 from vector<1x1xf16>
    %35 = vector.splat %34 : vector<1x16xf16>
    %36 = vector.extract %19[0, 0] : f16 from vector<1x1xf16>
    %37 = vector.splat %36 : vector<1x16xf16>
    %38 = vector.extract %20[0, 0] : f16 from vector<1x1xf16>
    %39 = vector.splat %38 : vector<1x16xf16>
    %40 = vector.extract %21[0, 0] : f16 from vector<1x1xf16>
    %41 = vector.splat %40 : vector<1x16xf16>
    %42 = vector.extract %22[0, 0] : f16 from vector<1x1xf16>
    %43 = vector.splat %42 : vector<1x16xf16>
    %44 = vector.extract %23[0, 0] : f16 from vector<1x1xf16>
    %45 = vector.splat %44 : vector<1x16xf16>
    %46 = vector.extract %24[0, 0] : f16 from vector<1x1xf16>
    %47 = vector.splat %46 : vector<1x16xf16>
    %48 = vector.extract %25[0, 0] : f16 from vector<1x1xf16>
    %49 = vector.splat %48 : vector<1x16xf16>
    %50 = vector.extract %26[0, 0] : f16 from vector<1x1xf16>
    %51 = vector.splat %50 : vector<1x16xf16>
    %52 = vector.extract %27[0, 0] : f16 from vector<1x1xf16>
    %53 = vector.splat %52 : vector<1x16xf16>
    %54 = vector.extract %28[0, 0] : f16 from vector<1x1xf16>
    %55 = vector.splat %54 : vector<1x16xf16>
    %56 = vector.extract %29[0, 0] : f16 from vector<1x1xf16>
    %57 = vector.splat %56 : vector<1x16xf16>
    %58 = vector.extract %30[0, 0] : f16 from vector<1x1xf16>
    %59 = vector.splat %58 : vector<1x16xf16>
    %60 = vector.extract %31[0, 0] : f16 from vector<1x1xf16>
    %61 = vector.splat %60 : vector<1x16xf16>
    %62 = vector.extract %32[0, 0] : f16 from vector<1x1xf16>
    %63 = vector.splat %62 : vector<1x16xf16>
    %64 = vector.extract %33[0, 0] : f16 from vector<1x1xf16>
    %65 = vector.splat %64 : vector<1x16xf16>
    %66 = vector.shuffle %35, %37 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %67 = vector.shuffle %39, %41 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %68 = vector.shuffle %43, %45 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %69 = vector.shuffle %47, %49 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %70 = vector.shuffle %51, %53 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %71 = vector.shuffle %55, %57 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %72 = vector.shuffle %59, %61 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %73 = vector.shuffle %63, %65 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %74 = vector.shuffle %66, %67 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %75 = vector.shuffle %68, %69 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %76 = vector.shuffle %70, %71 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %77 = vector.shuffle %72, %73 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %78 = vector.shuffle %74, %75 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x16xf16>, vector<4x16xf16>
    %79 = vector.shuffle %76, %77 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x16xf16>, vector<4x16xf16>
    %80 = vector.shuffle %78, %79 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf16>, vector<8x16xf16>
    %81 = vector.shape_cast %80 {packed} : vector<16x16xf16> to vector<256xf16>
    %82 = vector.shuffle %81, %81 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
    %83 = vector.shape_cast %82 {packed} : vector<256xf16> to vector<8x16x2xf16>
    %84 = xegpu.dpas %1, %83 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    %85 = xegpu.create_nd_tdesc %arg2[0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %84, %85 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    gpu.return
  }
}

// -----// IR Dump After OptimizeTranspose (imex-xegpu-optimize-transpose) //----- //
gpu.module @test_kernels {
  gpu.func @bcast_add_exp(%arg0: vector<1x16xf16>, %arg1: vector<16x1xf16>, %arg2: memref<8x16xf32>) kernel {
    %0 = vector.shape_cast %arg0 : vector<1x16xf16> to vector<16xf16>
    %1 = vector.broadcast %0 : vector<16xf16> to vector<8x16xf16>
    %2 = vector.extract_strided_slice %arg1 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %3 = vector.extract_strided_slice %arg1 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %4 = vector.extract_strided_slice %arg1 {offsets = [2, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %5 = vector.extract_strided_slice %arg1 {offsets = [3, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %6 = vector.extract_strided_slice %arg1 {offsets = [4, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %7 = vector.extract_strided_slice %arg1 {offsets = [5, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %8 = vector.extract_strided_slice %arg1 {offsets = [6, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %9 = vector.extract_strided_slice %arg1 {offsets = [7, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %10 = vector.extract_strided_slice %arg1 {offsets = [8, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %11 = vector.extract_strided_slice %arg1 {offsets = [9, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %12 = vector.extract_strided_slice %arg1 {offsets = [10, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %13 = vector.extract_strided_slice %arg1 {offsets = [11, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %14 = vector.extract_strided_slice %arg1 {offsets = [12, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %15 = vector.extract_strided_slice %arg1 {offsets = [13, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %16 = vector.extract_strided_slice %arg1 {offsets = [14, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %17 = vector.extract_strided_slice %arg1 {offsets = [15, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
    %18 = math.exp %2 : vector<1x1xf16>
    %19 = math.exp %3 : vector<1x1xf16>
    %20 = math.exp %4 : vector<1x1xf16>
    %21 = math.exp %5 : vector<1x1xf16>
    %22 = math.exp %6 : vector<1x1xf16>
    %23 = math.exp %7 : vector<1x1xf16>
    %24 = math.exp %8 : vector<1x1xf16>
    %25 = math.exp %9 : vector<1x1xf16>
    %26 = math.exp %10 : vector<1x1xf16>
    %27 = math.exp %11 : vector<1x1xf16>
    %28 = math.exp %12 : vector<1x1xf16>
    %29 = math.exp %13 : vector<1x1xf16>
    %30 = math.exp %14 : vector<1x1xf16>
    %31 = math.exp %15 : vector<1x1xf16>
    %32 = math.exp %16 : vector<1x1xf16>
    %33 = math.exp %17 : vector<1x1xf16>
    %34 = vector.extract %18[0, 0] : f16 from vector<1x1xf16>
    %35 = vector.splat %34 : vector<1x16xf16>
    %36 = vector.extract %19[0, 0] : f16 from vector<1x1xf16>
    %37 = vector.splat %36 : vector<1x16xf16>
    %38 = vector.extract %20[0, 0] : f16 from vector<1x1xf16>
    %39 = vector.splat %38 : vector<1x16xf16>
    %40 = vector.extract %21[0, 0] : f16 from vector<1x1xf16>
    %41 = vector.splat %40 : vector<1x16xf16>
    %42 = vector.extract %22[0, 0] : f16 from vector<1x1xf16>
    %43 = vector.splat %42 : vector<1x16xf16>
    %44 = vector.extract %23[0, 0] : f16 from vector<1x1xf16>
    %45 = vector.splat %44 : vector<1x16xf16>
    %46 = vector.extract %24[0, 0] : f16 from vector<1x1xf16>
    %47 = vector.splat %46 : vector<1x16xf16>
    %48 = vector.extract %25[0, 0] : f16 from vector<1x1xf16>
    %49 = vector.splat %48 : vector<1x16xf16>
    %50 = vector.extract %26[0, 0] : f16 from vector<1x1xf16>
    %51 = vector.splat %50 : vector<1x16xf16>
    %52 = vector.extract %27[0, 0] : f16 from vector<1x1xf16>
    %53 = vector.splat %52 : vector<1x16xf16>
    %54 = vector.extract %28[0, 0] : f16 from vector<1x1xf16>
    %55 = vector.splat %54 : vector<1x16xf16>
    %56 = vector.extract %29[0, 0] : f16 from vector<1x1xf16>
    %57 = vector.splat %56 : vector<1x16xf16>
    %58 = vector.extract %30[0, 0] : f16 from vector<1x1xf16>
    %59 = vector.splat %58 : vector<1x16xf16>
    %60 = vector.extract %31[0, 0] : f16 from vector<1x1xf16>
    %61 = vector.splat %60 : vector<1x16xf16>
    %62 = vector.extract %32[0, 0] : f16 from vector<1x1xf16>
    %63 = vector.splat %62 : vector<1x16xf16>
    %64 = vector.extract %33[0, 0] : f16 from vector<1x1xf16>
    %65 = vector.splat %64 : vector<1x16xf16>
    %66 = vector.shuffle %35, %37 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %67 = vector.shuffle %39, %41 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %68 = vector.shuffle %43, %45 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %69 = vector.shuffle %47, %49 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %70 = vector.shuffle %51, %53 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %71 = vector.shuffle %55, %57 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %72 = vector.shuffle %59, %61 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %73 = vector.shuffle %63, %65 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
    %74 = vector.shuffle %66, %67 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %75 = vector.shuffle %68, %69 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %76 = vector.shuffle %70, %71 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %77 = vector.shuffle %72, %73 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
    %78 = vector.shuffle %74, %75 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x16xf16>, vector<4x16xf16>
    %79 = vector.shuffle %76, %77 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x16xf16>, vector<4x16xf16>
    %80 = vector.shuffle %78, %79 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf16>, vector<8x16xf16>
    %81 = vector.shape_cast %80 {packed} : vector<16x16xf16> to vector<256xf16>
    %82 = vector.shuffle %81, %81 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
    %83 = vector.shape_cast %82 {packed} : vector<256xf16> to vector<8x16x2xf16>
    %84 = xegpu.dpas %1, %83 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    %85 = xegpu.create_nd_tdesc %arg2[0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %84, %85 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    gpu.return
  }
}

// -----// IR Dump After CSE (cse) //----- //
module {
  gpu.module @test_kernels {
    gpu.func @bcast_add_exp(%arg0: vector<1x16xf16>, %arg1: vector<16x1xf16>, %arg2: memref<8x16xf32>) kernel {
      %0 = vector.shape_cast %arg0 : vector<1x16xf16> to vector<16xf16>
      %1 = vector.broadcast %0 : vector<16xf16> to vector<8x16xf16>
      %2 = vector.extract_strided_slice %arg1 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
      %3 = vector.extract_strided_slice %arg1 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
      %4 = vector.extract_strided_slice %arg1 {offsets = [2, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
      %5 = vector.extract_strided_slice %arg1 {offsets = [3, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
      %6 = vector.extract_strided_slice %arg1 {offsets = [4, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
      %7 = vector.extract_strided_slice %arg1 {offsets = [5, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
      %8 = vector.extract_strided_slice %arg1 {offsets = [6, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
      %9 = vector.extract_strided_slice %arg1 {offsets = [7, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
      %10 = vector.extract_strided_slice %arg1 {offsets = [8, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
      %11 = vector.extract_strided_slice %arg1 {offsets = [9, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
      %12 = vector.extract_strided_slice %arg1 {offsets = [10, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
      %13 = vector.extract_strided_slice %arg1 {offsets = [11, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
      %14 = vector.extract_strided_slice %arg1 {offsets = [12, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
      %15 = vector.extract_strided_slice %arg1 {offsets = [13, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
      %16 = vector.extract_strided_slice %arg1 {offsets = [14, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
      %17 = vector.extract_strided_slice %arg1 {offsets = [15, 0], sizes = [1, 1], strides = [1, 1]} : vector<16x1xf16> to vector<1x1xf16>
      %18 = math.exp %2 : vector<1x1xf16>
      %19 = math.exp %3 : vector<1x1xf16>
      %20 = math.exp %4 : vector<1x1xf16>
      %21 = math.exp %5 : vector<1x1xf16>
      %22 = math.exp %6 : vector<1x1xf16>
      %23 = math.exp %7 : vector<1x1xf16>
      %24 = math.exp %8 : vector<1x1xf16>
      %25 = math.exp %9 : vector<1x1xf16>
      %26 = math.exp %10 : vector<1x1xf16>
      %27 = math.exp %11 : vector<1x1xf16>
      %28 = math.exp %12 : vector<1x1xf16>
      %29 = math.exp %13 : vector<1x1xf16>
      %30 = math.exp %14 : vector<1x1xf16>
      %31 = math.exp %15 : vector<1x1xf16>
      %32 = math.exp %16 : vector<1x1xf16>
      %33 = math.exp %17 : vector<1x1xf16>
      %34 = vector.extract %18[0, 0] : f16 from vector<1x1xf16>
      %35 = vector.splat %34 : vector<1x16xf16>
      %36 = vector.extract %19[0, 0] : f16 from vector<1x1xf16>
      %37 = vector.splat %36 : vector<1x16xf16>
      %38 = vector.extract %20[0, 0] : f16 from vector<1x1xf16>
      %39 = vector.splat %38 : vector<1x16xf16>
      %40 = vector.extract %21[0, 0] : f16 from vector<1x1xf16>
      %41 = vector.splat %40 : vector<1x16xf16>
      %42 = vector.extract %22[0, 0] : f16 from vector<1x1xf16>
      %43 = vector.splat %42 : vector<1x16xf16>
      %44 = vector.extract %23[0, 0] : f16 from vector<1x1xf16>
      %45 = vector.splat %44 : vector<1x16xf16>
      %46 = vector.extract %24[0, 0] : f16 from vector<1x1xf16>
      %47 = vector.splat %46 : vector<1x16xf16>
      %48 = vector.extract %25[0, 0] : f16 from vector<1x1xf16>
      %49 = vector.splat %48 : vector<1x16xf16>
      %50 = vector.extract %26[0, 0] : f16 from vector<1x1xf16>
      %51 = vector.splat %50 : vector<1x16xf16>
      %52 = vector.extract %27[0, 0] : f16 from vector<1x1xf16>
      %53 = vector.splat %52 : vector<1x16xf16>
      %54 = vector.extract %28[0, 0] : f16 from vector<1x1xf16>
      %55 = vector.splat %54 : vector<1x16xf16>
      %56 = vector.extract %29[0, 0] : f16 from vector<1x1xf16>
      %57 = vector.splat %56 : vector<1x16xf16>
      %58 = vector.extract %30[0, 0] : f16 from vector<1x1xf16>
      %59 = vector.splat %58 : vector<1x16xf16>
      %60 = vector.extract %31[0, 0] : f16 from vector<1x1xf16>
      %61 = vector.splat %60 : vector<1x16xf16>
      %62 = vector.extract %32[0, 0] : f16 from vector<1x1xf16>
      %63 = vector.splat %62 : vector<1x16xf16>
      %64 = vector.extract %33[0, 0] : f16 from vector<1x1xf16>
      %65 = vector.splat %64 : vector<1x16xf16>
      %66 = vector.shuffle %35, %37 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
      %67 = vector.shuffle %39, %41 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
      %68 = vector.shuffle %43, %45 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
      %69 = vector.shuffle %47, %49 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
      %70 = vector.shuffle %51, %53 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
      %71 = vector.shuffle %55, %57 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
      %72 = vector.shuffle %59, %61 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
      %73 = vector.shuffle %63, %65 [0, 1] : vector<1x16xf16>, vector<1x16xf16>
      %74 = vector.shuffle %66, %67 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
      %75 = vector.shuffle %68, %69 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
      %76 = vector.shuffle %70, %71 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
      %77 = vector.shuffle %72, %73 [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
      %78 = vector.shuffle %74, %75 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x16xf16>, vector<4x16xf16>
      %79 = vector.shuffle %76, %77 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x16xf16>, vector<4x16xf16>
      %80 = vector.shuffle %78, %79 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf16>, vector<8x16xf16>
      %81 = vector.shape_cast %80 {packed} : vector<16x16xf16> to vector<256xf16>
      %82 = vector.shuffle %81, %81 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
      %83 = vector.shape_cast %82 {packed} : vector<256xf16> to vector<8x16x2xf16>
      %84 = xegpu.dpas %1, %83 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
      %85 = xegpu.create_nd_tdesc %arg2[0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      xegpu.store_nd %84, %85 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      gpu.return
    }
  }
}


// -----// IR Dump After VectorLinearize (imex-vector-linearize) //----- //
module {
  gpu.module @test_kernels {
    gpu.func @bcast_add_exp(%arg0: vector<1x16xf16>, %arg1: vector<16x1xf16>, %arg2: memref<8x16xf32>) kernel {
      %0 = vector.shape_cast %arg1 : vector<16x1xf16> to vector<16xf16>
      %cst = arith.constant dense<0.000000e+00> : vector<128xf16>
      %1 = vector.shape_cast %arg0 : vector<1x16xf16> to vector<16xf16>
      %2 = vector.shuffle %1, %1 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf16>, vector<16xf16>
      %3 = vector.shuffle %cst, %2 [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %4 = vector.shuffle %1, %1 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf16>, vector<16xf16>
      %5 = vector.shuffle %3, %4 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %6 = vector.shuffle %1, %1 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf16>, vector<16xf16>
      %7 = vector.shuffle %5, %6 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %8 = vector.shuffle %1, %1 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf16>, vector<16xf16>
      %9 = vector.shuffle %7, %8 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %10 = vector.shuffle %1, %1 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf16>, vector<16xf16>
      %11 = vector.shuffle %9, %10 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %12 = vector.shuffle %1, %1 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf16>, vector<16xf16>
      %13 = vector.shuffle %11, %12 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %14 = vector.shuffle %1, %1 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf16>, vector<16xf16>
      %15 = vector.shuffle %13, %14 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %16 = vector.shuffle %1, %1 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf16>, vector<16xf16>
      %17 = vector.shuffle %15, %16 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143] : vector<128xf16>, vector<128xf16>
      %18 = vector.shape_cast %17 : vector<128xf16> to vector<8x16xf16>
      %19 = vector.extract_strided_slice %0 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %20 = vector.extract_strided_slice %0 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %21 = vector.extract_strided_slice %0 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %22 = vector.extract_strided_slice %0 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %23 = vector.extract_strided_slice %0 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %24 = vector.extract_strided_slice %0 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %25 = vector.extract_strided_slice %0 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %26 = vector.extract_strided_slice %0 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %27 = vector.extract_strided_slice %0 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %28 = vector.extract_strided_slice %0 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %29 = vector.extract_strided_slice %0 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %30 = vector.extract_strided_slice %0 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %31 = vector.extract_strided_slice %0 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %32 = vector.extract_strided_slice %0 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %33 = vector.extract_strided_slice %0 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %34 = vector.extract_strided_slice %0 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %35 = math.exp %19 : vector<1xf16>
      %36 = math.exp %20 : vector<1xf16>
      %37 = math.exp %21 : vector<1xf16>
      %38 = math.exp %22 : vector<1xf16>
      %39 = math.exp %23 : vector<1xf16>
      %40 = math.exp %24 : vector<1xf16>
      %41 = math.exp %25 : vector<1xf16>
      %42 = math.exp %26 : vector<1xf16>
      %43 = math.exp %27 : vector<1xf16>
      %44 = math.exp %28 : vector<1xf16>
      %45 = math.exp %29 : vector<1xf16>
      %46 = math.exp %30 : vector<1xf16>
      %47 = math.exp %31 : vector<1xf16>
      %48 = math.exp %32 : vector<1xf16>
      %49 = math.exp %33 : vector<1xf16>
      %50 = math.exp %34 : vector<1xf16>
      %c0_i32 = arith.constant 0 : i32
      %51 = vector.extractelement %35[%c0_i32 : i32] : vector<1xf16>
      %52 = vector.splat %51 : vector<16xf16>
      %c0_i32_0 = arith.constant 0 : i32
      %53 = vector.extractelement %36[%c0_i32_0 : i32] : vector<1xf16>
      %54 = vector.splat %53 : vector<16xf16>
      %c0_i32_1 = arith.constant 0 : i32
      %55 = vector.extractelement %37[%c0_i32_1 : i32] : vector<1xf16>
      %56 = vector.splat %55 : vector<16xf16>
      %c0_i32_2 = arith.constant 0 : i32
      %57 = vector.extractelement %38[%c0_i32_2 : i32] : vector<1xf16>
      %58 = vector.splat %57 : vector<16xf16>
      %c0_i32_3 = arith.constant 0 : i32
      %59 = vector.extractelement %39[%c0_i32_3 : i32] : vector<1xf16>
      %60 = vector.splat %59 : vector<16xf16>
      %c0_i32_4 = arith.constant 0 : i32
      %61 = vector.extractelement %40[%c0_i32_4 : i32] : vector<1xf16>
      %62 = vector.splat %61 : vector<16xf16>
      %c0_i32_5 = arith.constant 0 : i32
      %63 = vector.extractelement %41[%c0_i32_5 : i32] : vector<1xf16>
      %64 = vector.splat %63 : vector<16xf16>
      %c0_i32_6 = arith.constant 0 : i32
      %65 = vector.extractelement %42[%c0_i32_6 : i32] : vector<1xf16>
      %66 = vector.splat %65 : vector<16xf16>
      %c0_i32_7 = arith.constant 0 : i32
      %67 = vector.extractelement %43[%c0_i32_7 : i32] : vector<1xf16>
      %68 = vector.splat %67 : vector<16xf16>
      %c0_i32_8 = arith.constant 0 : i32
      %69 = vector.extractelement %44[%c0_i32_8 : i32] : vector<1xf16>
      %70 = vector.splat %69 : vector<16xf16>
      %c0_i32_9 = arith.constant 0 : i32
      %71 = vector.extractelement %45[%c0_i32_9 : i32] : vector<1xf16>
      %72 = vector.splat %71 : vector<16xf16>
      %c0_i32_10 = arith.constant 0 : i32
      %73 = vector.extractelement %46[%c0_i32_10 : i32] : vector<1xf16>
      %74 = vector.splat %73 : vector<16xf16>
      %c0_i32_11 = arith.constant 0 : i32
      %75 = vector.extractelement %47[%c0_i32_11 : i32] : vector<1xf16>
      %76 = vector.splat %75 : vector<16xf16>
      %c0_i32_12 = arith.constant 0 : i32
      %77 = vector.extractelement %48[%c0_i32_12 : i32] : vector<1xf16>
      %78 = vector.splat %77 : vector<16xf16>
      %c0_i32_13 = arith.constant 0 : i32
      %79 = vector.extractelement %49[%c0_i32_13 : i32] : vector<1xf16>
      %80 = vector.splat %79 : vector<16xf16>
      %c0_i32_14 = arith.constant 0 : i32
      %81 = vector.extractelement %50[%c0_i32_14 : i32] : vector<1xf16>
      %82 = vector.splat %81 : vector<16xf16>
      %83 = vector.shuffle %52, %54 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %84 = vector.shuffle %56, %58 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %85 = vector.shuffle %60, %62 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %86 = vector.shuffle %64, %66 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %87 = vector.shuffle %68, %70 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %88 = vector.shuffle %72, %74 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %89 = vector.shuffle %76, %78 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %90 = vector.shuffle %80, %82 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %91 = vector.shuffle %83, %84 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %92 = vector.shuffle %85, %86 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %93 = vector.shuffle %87, %88 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %94 = vector.shuffle %89, %90 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %95 = vector.shuffle %91, %92 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<64xf16>, vector<64xf16>
      %96 = vector.shuffle %93, %94 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<64xf16>, vector<64xf16>
      %97 = vector.shuffle %95, %96 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<128xf16>, vector<128xf16>
      %98 = vector.shape_cast %97 : vector<256xf16> to vector<16x16xf16>
      %99 = vector.shape_cast %98 {packed} : vector<16x16xf16> to vector<256xf16>
      %100 = vector.shuffle %99, %99 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
      %101 = vector.shape_cast %100 {packed} : vector<256xf16> to vector<8x16x2xf16>
      %102 = xegpu.dpas %18, %101 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
      %103 = xegpu.create_nd_tdesc %arg2[0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      xegpu.store_nd %102, %103 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      gpu.return
    }
  }
}


// -----// IR Dump After CSE (cse) //----- //
module {
  gpu.module @test_kernels {
    gpu.func @bcast_add_exp(%arg0: vector<1x16xf16>, %arg1: vector<16x1xf16>, %arg2: memref<8x16xf32>) kernel {
      %0 = vector.shape_cast %arg1 : vector<16x1xf16> to vector<16xf16>
      %cst = arith.constant dense<0.000000e+00> : vector<128xf16>
      %1 = vector.shape_cast %arg0 : vector<1x16xf16> to vector<16xf16>
      %2 = vector.shuffle %1, %1 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf16>, vector<16xf16>
      %3 = vector.shuffle %cst, %2 [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %4 = vector.shuffle %3, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %5 = vector.shuffle %4, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %6 = vector.shuffle %5, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %7 = vector.shuffle %6, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %8 = vector.shuffle %7, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %9 = vector.shuffle %8, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %10 = vector.shuffle %9, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143] : vector<128xf16>, vector<128xf16>
      %11 = vector.shape_cast %10 : vector<128xf16> to vector<8x16xf16>
      %12 = vector.extract_strided_slice %0 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %13 = vector.extract_strided_slice %0 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %14 = vector.extract_strided_slice %0 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %15 = vector.extract_strided_slice %0 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %16 = vector.extract_strided_slice %0 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %17 = vector.extract_strided_slice %0 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %18 = vector.extract_strided_slice %0 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %19 = vector.extract_strided_slice %0 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %20 = vector.extract_strided_slice %0 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %21 = vector.extract_strided_slice %0 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %22 = vector.extract_strided_slice %0 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %23 = vector.extract_strided_slice %0 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %24 = vector.extract_strided_slice %0 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %25 = vector.extract_strided_slice %0 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %26 = vector.extract_strided_slice %0 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %27 = vector.extract_strided_slice %0 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %28 = math.exp %12 : vector<1xf16>
      %29 = math.exp %13 : vector<1xf16>
      %30 = math.exp %14 : vector<1xf16>
      %31 = math.exp %15 : vector<1xf16>
      %32 = math.exp %16 : vector<1xf16>
      %33 = math.exp %17 : vector<1xf16>
      %34 = math.exp %18 : vector<1xf16>
      %35 = math.exp %19 : vector<1xf16>
      %36 = math.exp %20 : vector<1xf16>
      %37 = math.exp %21 : vector<1xf16>
      %38 = math.exp %22 : vector<1xf16>
      %39 = math.exp %23 : vector<1xf16>
      %40 = math.exp %24 : vector<1xf16>
      %41 = math.exp %25 : vector<1xf16>
      %42 = math.exp %26 : vector<1xf16>
      %43 = math.exp %27 : vector<1xf16>
      %c0_i32 = arith.constant 0 : i32
      %44 = vector.extractelement %28[%c0_i32 : i32] : vector<1xf16>
      %45 = vector.splat %44 : vector<16xf16>
      %46 = vector.extractelement %29[%c0_i32 : i32] : vector<1xf16>
      %47 = vector.splat %46 : vector<16xf16>
      %48 = vector.extractelement %30[%c0_i32 : i32] : vector<1xf16>
      %49 = vector.splat %48 : vector<16xf16>
      %50 = vector.extractelement %31[%c0_i32 : i32] : vector<1xf16>
      %51 = vector.splat %50 : vector<16xf16>
      %52 = vector.extractelement %32[%c0_i32 : i32] : vector<1xf16>
      %53 = vector.splat %52 : vector<16xf16>
      %54 = vector.extractelement %33[%c0_i32 : i32] : vector<1xf16>
      %55 = vector.splat %54 : vector<16xf16>
      %56 = vector.extractelement %34[%c0_i32 : i32] : vector<1xf16>
      %57 = vector.splat %56 : vector<16xf16>
      %58 = vector.extractelement %35[%c0_i32 : i32] : vector<1xf16>
      %59 = vector.splat %58 : vector<16xf16>
      %60 = vector.extractelement %36[%c0_i32 : i32] : vector<1xf16>
      %61 = vector.splat %60 : vector<16xf16>
      %62 = vector.extractelement %37[%c0_i32 : i32] : vector<1xf16>
      %63 = vector.splat %62 : vector<16xf16>
      %64 = vector.extractelement %38[%c0_i32 : i32] : vector<1xf16>
      %65 = vector.splat %64 : vector<16xf16>
      %66 = vector.extractelement %39[%c0_i32 : i32] : vector<1xf16>
      %67 = vector.splat %66 : vector<16xf16>
      %68 = vector.extractelement %40[%c0_i32 : i32] : vector<1xf16>
      %69 = vector.splat %68 : vector<16xf16>
      %70 = vector.extractelement %41[%c0_i32 : i32] : vector<1xf16>
      %71 = vector.splat %70 : vector<16xf16>
      %72 = vector.extractelement %42[%c0_i32 : i32] : vector<1xf16>
      %73 = vector.splat %72 : vector<16xf16>
      %74 = vector.extractelement %43[%c0_i32 : i32] : vector<1xf16>
      %75 = vector.splat %74 : vector<16xf16>
      %76 = vector.shuffle %45, %47 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %77 = vector.shuffle %49, %51 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %78 = vector.shuffle %53, %55 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %79 = vector.shuffle %57, %59 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %80 = vector.shuffle %61, %63 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %81 = vector.shuffle %65, %67 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %82 = vector.shuffle %69, %71 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %83 = vector.shuffle %73, %75 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %84 = vector.shuffle %76, %77 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %85 = vector.shuffle %78, %79 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %86 = vector.shuffle %80, %81 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %87 = vector.shuffle %82, %83 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %88 = vector.shuffle %84, %85 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<64xf16>, vector<64xf16>
      %89 = vector.shuffle %86, %87 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<64xf16>, vector<64xf16>
      %90 = vector.shuffle %88, %89 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<128xf16>, vector<128xf16>
      %91 = vector.shape_cast %90 : vector<256xf16> to vector<16x16xf16>
      %92 = vector.shape_cast %91 {packed} : vector<16x16xf16> to vector<256xf16>
      %93 = vector.shuffle %92, %92 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
      %94 = vector.shape_cast %93 {packed} : vector<256xf16> to vector<8x16x2xf16>
      %95 = xegpu.dpas %11, %94 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
      %96 = xegpu.create_nd_tdesc %arg2[0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      xegpu.store_nd %95, %96 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      gpu.return
    }
  }
}


// -----// IR Dump After RemoveSingleElemVector (imex-remove-single-elem-vector) //----- //
module {
  gpu.module @test_kernels {
    gpu.func @bcast_add_exp(%arg0: vector<1x16xf16>, %arg1: vector<16x1xf16>, %arg2: memref<8x16xf32>) kernel {
      %0 = vector.shape_cast %arg1 : vector<16x1xf16> to vector<16xf16>
      %cst = arith.constant dense<0.000000e+00> : vector<128xf16>
      %1 = vector.shape_cast %arg0 : vector<1x16xf16> to vector<16xf16>
      %2 = vector.shuffle %1, %1 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf16>, vector<16xf16>
      %3 = vector.shuffle %cst, %2 [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %4 = vector.shuffle %3, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %5 = vector.shuffle %4, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %6 = vector.shuffle %5, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %7 = vector.shuffle %6, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %8 = vector.shuffle %7, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %9 = vector.shuffle %8, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %10 = vector.shuffle %9, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143] : vector<128xf16>, vector<128xf16>
      %11 = vector.shape_cast %10 : vector<128xf16> to vector<8x16xf16>
      %12 = vector.extract_strided_slice %0 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %c0 = arith.constant 0 : index
      %13 = vector.extractelement %12[%c0 : index] : vector<1xf16>
      %14 = vector.extract_strided_slice %0 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %c0_0 = arith.constant 0 : index
      %15 = vector.extractelement %14[%c0_0 : index] : vector<1xf16>
      %16 = vector.extract_strided_slice %0 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %c0_1 = arith.constant 0 : index
      %17 = vector.extractelement %16[%c0_1 : index] : vector<1xf16>
      %18 = vector.extract_strided_slice %0 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %c0_2 = arith.constant 0 : index
      %19 = vector.extractelement %18[%c0_2 : index] : vector<1xf16>
      %20 = vector.extract_strided_slice %0 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %c0_3 = arith.constant 0 : index
      %21 = vector.extractelement %20[%c0_3 : index] : vector<1xf16>
      %22 = vector.extract_strided_slice %0 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %c0_4 = arith.constant 0 : index
      %23 = vector.extractelement %22[%c0_4 : index] : vector<1xf16>
      %24 = vector.extract_strided_slice %0 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %c0_5 = arith.constant 0 : index
      %25 = vector.extractelement %24[%c0_5 : index] : vector<1xf16>
      %26 = vector.extract_strided_slice %0 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %c0_6 = arith.constant 0 : index
      %27 = vector.extractelement %26[%c0_6 : index] : vector<1xf16>
      %28 = vector.extract_strided_slice %0 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %c0_7 = arith.constant 0 : index
      %29 = vector.extractelement %28[%c0_7 : index] : vector<1xf16>
      %30 = vector.extract_strided_slice %0 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %c0_8 = arith.constant 0 : index
      %31 = vector.extractelement %30[%c0_8 : index] : vector<1xf16>
      %32 = vector.extract_strided_slice %0 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %c0_9 = arith.constant 0 : index
      %33 = vector.extractelement %32[%c0_9 : index] : vector<1xf16>
      %34 = vector.extract_strided_slice %0 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %c0_10 = arith.constant 0 : index
      %35 = vector.extractelement %34[%c0_10 : index] : vector<1xf16>
      %36 = vector.extract_strided_slice %0 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %c0_11 = arith.constant 0 : index
      %37 = vector.extractelement %36[%c0_11 : index] : vector<1xf16>
      %38 = vector.extract_strided_slice %0 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %c0_12 = arith.constant 0 : index
      %39 = vector.extractelement %38[%c0_12 : index] : vector<1xf16>
      %40 = vector.extract_strided_slice %0 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %c0_13 = arith.constant 0 : index
      %41 = vector.extractelement %40[%c0_13 : index] : vector<1xf16>
      %42 = vector.extract_strided_slice %0 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %c0_14 = arith.constant 0 : index
      %43 = vector.extractelement %42[%c0_14 : index] : vector<1xf16>
      %44 = math.exp %13 : f16
      %45 = math.exp %15 : f16
      %46 = math.exp %17 : f16
      %47 = math.exp %19 : f16
      %48 = math.exp %21 : f16
      %49 = math.exp %23 : f16
      %50 = math.exp %25 : f16
      %51 = math.exp %27 : f16
      %52 = math.exp %29 : f16
      %53 = math.exp %31 : f16
      %54 = math.exp %33 : f16
      %55 = math.exp %35 : f16
      %56 = math.exp %37 : f16
      %57 = math.exp %39 : f16
      %58 = math.exp %41 : f16
      %59 = math.exp %43 : f16
      %c0_i32 = arith.constant 0 : i32
      %60 = vector.splat %44 : vector<16xf16>
      %61 = vector.splat %45 : vector<16xf16>
      %62 = vector.splat %46 : vector<16xf16>
      %63 = vector.splat %47 : vector<16xf16>
      %64 = vector.splat %48 : vector<16xf16>
      %65 = vector.splat %49 : vector<16xf16>
      %66 = vector.splat %50 : vector<16xf16>
      %67 = vector.splat %51 : vector<16xf16>
      %68 = vector.splat %52 : vector<16xf16>
      %69 = vector.splat %53 : vector<16xf16>
      %70 = vector.splat %54 : vector<16xf16>
      %71 = vector.splat %55 : vector<16xf16>
      %72 = vector.splat %56 : vector<16xf16>
      %73 = vector.splat %57 : vector<16xf16>
      %74 = vector.splat %58 : vector<16xf16>
      %75 = vector.splat %59 : vector<16xf16>
      %76 = vector.shuffle %60, %61 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %77 = vector.shuffle %62, %63 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %78 = vector.shuffle %64, %65 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %79 = vector.shuffle %66, %67 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %80 = vector.shuffle %68, %69 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %81 = vector.shuffle %70, %71 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %82 = vector.shuffle %72, %73 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %83 = vector.shuffle %74, %75 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %84 = vector.shuffle %76, %77 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %85 = vector.shuffle %78, %79 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %86 = vector.shuffle %80, %81 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %87 = vector.shuffle %82, %83 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %88 = vector.shuffle %84, %85 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<64xf16>, vector<64xf16>
      %89 = vector.shuffle %86, %87 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<64xf16>, vector<64xf16>
      %90 = vector.shuffle %88, %89 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<128xf16>, vector<128xf16>
      %91 = vector.shape_cast %90 : vector<256xf16> to vector<16x16xf16>
      %92 = vector.shuffle %90, %90 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
      %93 = vector.shape_cast %92 {packed} : vector<256xf16> to vector<8x16x2xf16>
      %94 = xegpu.dpas %11, %93 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
      %95 = xegpu.create_nd_tdesc %arg2[0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      xegpu.store_nd %94, %95 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      gpu.return
    }
  }
}


// -----// IR Dump After CSE (cse) //----- //
module {
  gpu.module @test_kernels {
    gpu.func @bcast_add_exp(%arg0: vector<1x16xf16>, %arg1: vector<16x1xf16>, %arg2: memref<8x16xf32>) kernel {
      %0 = vector.shape_cast %arg1 : vector<16x1xf16> to vector<16xf16>
      %cst = arith.constant dense<0.000000e+00> : vector<128xf16>
      %1 = vector.shape_cast %arg0 : vector<1x16xf16> to vector<16xf16>
      %2 = vector.shuffle %1, %1 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf16>, vector<16xf16>
      %3 = vector.shuffle %cst, %2 [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %4 = vector.shuffle %3, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %5 = vector.shuffle %4, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %6 = vector.shuffle %5, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %7 = vector.shuffle %6, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %8 = vector.shuffle %7, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %9 = vector.shuffle %8, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %10 = vector.shuffle %9, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143] : vector<128xf16>, vector<128xf16>
      %11 = vector.shape_cast %10 : vector<128xf16> to vector<8x16xf16>
      %12 = vector.extract_strided_slice %0 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %c0 = arith.constant 0 : index
      %13 = vector.extractelement %12[%c0 : index] : vector<1xf16>
      %14 = vector.extract_strided_slice %0 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %15 = vector.extractelement %14[%c0 : index] : vector<1xf16>
      %16 = vector.extract_strided_slice %0 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %17 = vector.extractelement %16[%c0 : index] : vector<1xf16>
      %18 = vector.extract_strided_slice %0 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %19 = vector.extractelement %18[%c0 : index] : vector<1xf16>
      %20 = vector.extract_strided_slice %0 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %21 = vector.extractelement %20[%c0 : index] : vector<1xf16>
      %22 = vector.extract_strided_slice %0 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %23 = vector.extractelement %22[%c0 : index] : vector<1xf16>
      %24 = vector.extract_strided_slice %0 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %25 = vector.extractelement %24[%c0 : index] : vector<1xf16>
      %26 = vector.extract_strided_slice %0 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %27 = vector.extractelement %26[%c0 : index] : vector<1xf16>
      %28 = vector.extract_strided_slice %0 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %29 = vector.extractelement %28[%c0 : index] : vector<1xf16>
      %30 = vector.extract_strided_slice %0 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %31 = vector.extractelement %30[%c0 : index] : vector<1xf16>
      %32 = vector.extract_strided_slice %0 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %33 = vector.extractelement %32[%c0 : index] : vector<1xf16>
      %34 = vector.extract_strided_slice %0 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %35 = vector.extractelement %34[%c0 : index] : vector<1xf16>
      %36 = vector.extract_strided_slice %0 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %37 = vector.extractelement %36[%c0 : index] : vector<1xf16>
      %38 = vector.extract_strided_slice %0 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %39 = vector.extractelement %38[%c0 : index] : vector<1xf16>
      %40 = vector.extract_strided_slice %0 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %41 = vector.extractelement %40[%c0 : index] : vector<1xf16>
      %42 = vector.extract_strided_slice %0 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %43 = vector.extractelement %42[%c0 : index] : vector<1xf16>
      %44 = math.exp %13 : f16
      %45 = math.exp %15 : f16
      %46 = math.exp %17 : f16
      %47 = math.exp %19 : f16
      %48 = math.exp %21 : f16
      %49 = math.exp %23 : f16
      %50 = math.exp %25 : f16
      %51 = math.exp %27 : f16
      %52 = math.exp %29 : f16
      %53 = math.exp %31 : f16
      %54 = math.exp %33 : f16
      %55 = math.exp %35 : f16
      %56 = math.exp %37 : f16
      %57 = math.exp %39 : f16
      %58 = math.exp %41 : f16
      %59 = math.exp %43 : f16
      %60 = vector.splat %44 : vector<16xf16>
      %61 = vector.splat %45 : vector<16xf16>
      %62 = vector.splat %46 : vector<16xf16>
      %63 = vector.splat %47 : vector<16xf16>
      %64 = vector.splat %48 : vector<16xf16>
      %65 = vector.splat %49 : vector<16xf16>
      %66 = vector.splat %50 : vector<16xf16>
      %67 = vector.splat %51 : vector<16xf16>
      %68 = vector.splat %52 : vector<16xf16>
      %69 = vector.splat %53 : vector<16xf16>
      %70 = vector.splat %54 : vector<16xf16>
      %71 = vector.splat %55 : vector<16xf16>
      %72 = vector.splat %56 : vector<16xf16>
      %73 = vector.splat %57 : vector<16xf16>
      %74 = vector.splat %58 : vector<16xf16>
      %75 = vector.splat %59 : vector<16xf16>
      %76 = vector.shuffle %60, %61 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %77 = vector.shuffle %62, %63 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %78 = vector.shuffle %64, %65 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %79 = vector.shuffle %66, %67 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %80 = vector.shuffle %68, %69 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %81 = vector.shuffle %70, %71 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %82 = vector.shuffle %72, %73 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %83 = vector.shuffle %74, %75 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %84 = vector.shuffle %76, %77 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %85 = vector.shuffle %78, %79 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %86 = vector.shuffle %80, %81 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %87 = vector.shuffle %82, %83 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %88 = vector.shuffle %84, %85 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<64xf16>, vector<64xf16>
      %89 = vector.shuffle %86, %87 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<64xf16>, vector<64xf16>
      %90 = vector.shuffle %88, %89 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<128xf16>, vector<128xf16>
      %91 = vector.shuffle %90, %90 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
      %92 = vector.shape_cast %91 {packed} : vector<256xf16> to vector<8x16x2xf16>
      %93 = xegpu.dpas %11, %92 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
      %94 = xegpu.create_nd_tdesc %arg2[0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      xegpu.store_nd %93, %94 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      gpu.return
    }
  }
}


// -----// IR Dump After ConvertXeGPUToVC (convert-xegpu-to-vc) //----- //
gpu.module @test_kernels {
  func.func private @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32", linkage_type = <Import>>}
  func.func private @llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32(vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32", linkage_type = <Import>>}
  gpu.func @bcast_add_exp(%arg0: vector<1x16xf16>, %arg1: vector<16x1xf16>, %arg2: memref<8x16xf32>) kernel {
    %0 = builtin.unrealized_conversion_cast %arg0 : vector<1x16xf16> to vector<16xf16>
    %1 = builtin.unrealized_conversion_cast %arg1 : vector<16x1xf16> to vector<16xf16>
    %cst = arith.constant dense<0.000000e+00> : vector<128xf16>
    %2 = vector.shuffle %0, %0 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf16>, vector<16xf16>
    %3 = vector.shuffle %cst, %2 [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
    %4 = vector.shuffle %3, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
    %5 = vector.shuffle %4, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
    %6 = vector.shuffle %5, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
    %7 = vector.shuffle %6, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
    %8 = vector.shuffle %7, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
    %9 = vector.shuffle %8, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
    %10 = vector.shuffle %9, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143] : vector<128xf16>, vector<128xf16>
    %11 = vector.extract_strided_slice %1 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
    %c0 = arith.constant 0 : index
    %12 = vector.extractelement %11[%c0 : index] : vector<1xf16>
    %13 = vector.extract_strided_slice %1 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
    %14 = vector.extractelement %13[%c0 : index] : vector<1xf16>
    %15 = vector.extract_strided_slice %1 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
    %16 = vector.extractelement %15[%c0 : index] : vector<1xf16>
    %17 = vector.extract_strided_slice %1 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
    %18 = vector.extractelement %17[%c0 : index] : vector<1xf16>
    %19 = vector.extract_strided_slice %1 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
    %20 = vector.extractelement %19[%c0 : index] : vector<1xf16>
    %21 = vector.extract_strided_slice %1 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
    %22 = vector.extractelement %21[%c0 : index] : vector<1xf16>
    %23 = vector.extract_strided_slice %1 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
    %24 = vector.extractelement %23[%c0 : index] : vector<1xf16>
    %25 = vector.extract_strided_slice %1 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
    %26 = vector.extractelement %25[%c0 : index] : vector<1xf16>
    %27 = vector.extract_strided_slice %1 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
    %28 = vector.extractelement %27[%c0 : index] : vector<1xf16>
    %29 = vector.extract_strided_slice %1 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
    %30 = vector.extractelement %29[%c0 : index] : vector<1xf16>
    %31 = vector.extract_strided_slice %1 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
    %32 = vector.extractelement %31[%c0 : index] : vector<1xf16>
    %33 = vector.extract_strided_slice %1 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
    %34 = vector.extractelement %33[%c0 : index] : vector<1xf16>
    %35 = vector.extract_strided_slice %1 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
    %36 = vector.extractelement %35[%c0 : index] : vector<1xf16>
    %37 = vector.extract_strided_slice %1 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
    %38 = vector.extractelement %37[%c0 : index] : vector<1xf16>
    %39 = vector.extract_strided_slice %1 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
    %40 = vector.extractelement %39[%c0 : index] : vector<1xf16>
    %41 = vector.extract_strided_slice %1 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
    %42 = vector.extractelement %41[%c0 : index] : vector<1xf16>
    %43 = math.exp %12 : f16
    %44 = math.exp %14 : f16
    %45 = math.exp %16 : f16
    %46 = math.exp %18 : f16
    %47 = math.exp %20 : f16
    %48 = math.exp %22 : f16
    %49 = math.exp %24 : f16
    %50 = math.exp %26 : f16
    %51 = math.exp %28 : f16
    %52 = math.exp %30 : f16
    %53 = math.exp %32 : f16
    %54 = math.exp %34 : f16
    %55 = math.exp %36 : f16
    %56 = math.exp %38 : f16
    %57 = math.exp %40 : f16
    %58 = math.exp %42 : f16
    %59 = vector.splat %43 : vector<16xf16>
    %60 = vector.splat %44 : vector<16xf16>
    %61 = vector.splat %45 : vector<16xf16>
    %62 = vector.splat %46 : vector<16xf16>
    %63 = vector.splat %47 : vector<16xf16>
    %64 = vector.splat %48 : vector<16xf16>
    %65 = vector.splat %49 : vector<16xf16>
    %66 = vector.splat %50 : vector<16xf16>
    %67 = vector.splat %51 : vector<16xf16>
    %68 = vector.splat %52 : vector<16xf16>
    %69 = vector.splat %53 : vector<16xf16>
    %70 = vector.splat %54 : vector<16xf16>
    %71 = vector.splat %55 : vector<16xf16>
    %72 = vector.splat %56 : vector<16xf16>
    %73 = vector.splat %57 : vector<16xf16>
    %74 = vector.splat %58 : vector<16xf16>
    %75 = vector.shuffle %59, %60 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
    %76 = vector.shuffle %61, %62 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
    %77 = vector.shuffle %63, %64 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
    %78 = vector.shuffle %65, %66 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
    %79 = vector.shuffle %67, %68 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
    %80 = vector.shuffle %69, %70 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
    %81 = vector.shuffle %71, %72 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
    %82 = vector.shuffle %73, %74 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
    %83 = vector.shuffle %75, %76 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    %84 = vector.shuffle %77, %78 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    %85 = vector.shuffle %79, %80 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    %86 = vector.shuffle %81, %82 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    %87 = vector.shuffle %83, %84 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<64xf16>, vector<64xf16>
    %88 = vector.shuffle %85, %86 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<64xf16>, vector<64xf16>
    %89 = vector.shuffle %87, %88 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<128xf16>, vector<128xf16>
    %90 = vector.shuffle %89, %89 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
    %c134744586_i32 = arith.constant 134744586 : i32
    %91 = vector.bitcast %10 : vector<128xf16> to vector<64xi32>
    %92 = vector.bitcast %90 : vector<256xf16> to vector<128xi32>
    %93 = func.call @llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32(%92, %91, %c134744586_i32) : (vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32>
    %intptr = memref.extract_aligned_pointer_as_index %arg2 : memref<8x16xf32> -> index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %94 = arith.index_castui %intptr : index to i64
    %cst_0 = arith.constant dense<0> : vector<8xi64>
    %95 = vector.insert %94, %cst_0 [0] : i64 into vector<8xi64>
    %96 = vector.bitcast %95 : vector<8xi64> to vector<16xi32>
    %c63_i32 = arith.constant 63 : i32
    %c7_i32 = arith.constant 7 : i32
    %c63_i32_1 = arith.constant 63 : i32
    %97 = vector.insert %c63_i32, %96 [2] : i32 into vector<16xi32>
    %98 = vector.insert %c7_i32, %97 [3] : i32 into vector<16xi32>
    %99 = vector.insert %c63_i32_1, %98 [4] : i32 into vector<16xi32>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_2 = arith.constant 0 : i32
    %100 = vector.insert %c0_i32, %99 [5] : i32 into vector<16xi32>
    %101 = vector.insert %c0_i32_2, %100 [6] : i32 into vector<16xi32>
    %c1807_i32 = arith.constant 1807 : i32
    %102 = vector.insert %c1807_i32, %101 [7] : i32 into vector<16xi32>
    %true = arith.constant true
    %c3_i8 = arith.constant 3 : i8
    %c3_i8_3 = arith.constant 3 : i8
    %103 = vector.from_elements %c3_i8, %c3_i8_3 : vector<2xi8>
    %c1_i8 = arith.constant 1 : i8
    %c16_i16 = arith.constant 16 : i16
    %c8_i16 = arith.constant 8 : i16
    %c0_i32_4 = arith.constant 0 : i32
    %c0_i32_5 = arith.constant 0 : i32
    func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %103, %c1_i8, %c16_i16, %c8_i16, %102, %c0_i32_4, %c0_i32_5, %93) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
    gpu.return
  }
}

// -----// IR Dump After ReconcileUnrealizedCasts (reconcile-unrealized-casts) //----- //
module {
  gpu.module @test_kernels {
    func.func private @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32", linkage_type = <Import>>}
    func.func private @llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32(vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32", linkage_type = <Import>>}
    gpu.func @bcast_add_exp(%arg0: vector<1x16xf16>, %arg1: vector<16x1xf16>, %arg2: memref<8x16xf32>) kernel {
      %0 = builtin.unrealized_conversion_cast %arg0 : vector<1x16xf16> to vector<16xf16>
      %1 = builtin.unrealized_conversion_cast %arg1 : vector<16x1xf16> to vector<16xf16>
      %cst = arith.constant dense<0.000000e+00> : vector<128xf16>
      %2 = vector.shuffle %0, %0 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf16>, vector<16xf16>
      %3 = vector.shuffle %cst, %2 [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %4 = vector.shuffle %3, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %5 = vector.shuffle %4, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %6 = vector.shuffle %5, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %7 = vector.shuffle %6, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %8 = vector.shuffle %7, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %9 = vector.shuffle %8, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %10 = vector.shuffle %9, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143] : vector<128xf16>, vector<128xf16>
      %11 = vector.extract_strided_slice %1 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %c0 = arith.constant 0 : index
      %12 = vector.extractelement %11[%c0 : index] : vector<1xf16>
      %13 = vector.extract_strided_slice %1 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %14 = vector.extractelement %13[%c0 : index] : vector<1xf16>
      %15 = vector.extract_strided_slice %1 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %16 = vector.extractelement %15[%c0 : index] : vector<1xf16>
      %17 = vector.extract_strided_slice %1 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %18 = vector.extractelement %17[%c0 : index] : vector<1xf16>
      %19 = vector.extract_strided_slice %1 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %20 = vector.extractelement %19[%c0 : index] : vector<1xf16>
      %21 = vector.extract_strided_slice %1 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %22 = vector.extractelement %21[%c0 : index] : vector<1xf16>
      %23 = vector.extract_strided_slice %1 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %24 = vector.extractelement %23[%c0 : index] : vector<1xf16>
      %25 = vector.extract_strided_slice %1 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %26 = vector.extractelement %25[%c0 : index] : vector<1xf16>
      %27 = vector.extract_strided_slice %1 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %28 = vector.extractelement %27[%c0 : index] : vector<1xf16>
      %29 = vector.extract_strided_slice %1 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %30 = vector.extractelement %29[%c0 : index] : vector<1xf16>
      %31 = vector.extract_strided_slice %1 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %32 = vector.extractelement %31[%c0 : index] : vector<1xf16>
      %33 = vector.extract_strided_slice %1 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %34 = vector.extractelement %33[%c0 : index] : vector<1xf16>
      %35 = vector.extract_strided_slice %1 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %36 = vector.extractelement %35[%c0 : index] : vector<1xf16>
      %37 = vector.extract_strided_slice %1 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %38 = vector.extractelement %37[%c0 : index] : vector<1xf16>
      %39 = vector.extract_strided_slice %1 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %40 = vector.extractelement %39[%c0 : index] : vector<1xf16>
      %41 = vector.extract_strided_slice %1 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %42 = vector.extractelement %41[%c0 : index] : vector<1xf16>
      %43 = math.exp %12 : f16
      %44 = math.exp %14 : f16
      %45 = math.exp %16 : f16
      %46 = math.exp %18 : f16
      %47 = math.exp %20 : f16
      %48 = math.exp %22 : f16
      %49 = math.exp %24 : f16
      %50 = math.exp %26 : f16
      %51 = math.exp %28 : f16
      %52 = math.exp %30 : f16
      %53 = math.exp %32 : f16
      %54 = math.exp %34 : f16
      %55 = math.exp %36 : f16
      %56 = math.exp %38 : f16
      %57 = math.exp %40 : f16
      %58 = math.exp %42 : f16
      %59 = vector.splat %43 : vector<16xf16>
      %60 = vector.splat %44 : vector<16xf16>
      %61 = vector.splat %45 : vector<16xf16>
      %62 = vector.splat %46 : vector<16xf16>
      %63 = vector.splat %47 : vector<16xf16>
      %64 = vector.splat %48 : vector<16xf16>
      %65 = vector.splat %49 : vector<16xf16>
      %66 = vector.splat %50 : vector<16xf16>
      %67 = vector.splat %51 : vector<16xf16>
      %68 = vector.splat %52 : vector<16xf16>
      %69 = vector.splat %53 : vector<16xf16>
      %70 = vector.splat %54 : vector<16xf16>
      %71 = vector.splat %55 : vector<16xf16>
      %72 = vector.splat %56 : vector<16xf16>
      %73 = vector.splat %57 : vector<16xf16>
      %74 = vector.splat %58 : vector<16xf16>
      %75 = vector.shuffle %59, %60 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %76 = vector.shuffle %61, %62 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %77 = vector.shuffle %63, %64 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %78 = vector.shuffle %65, %66 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %79 = vector.shuffle %67, %68 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %80 = vector.shuffle %69, %70 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %81 = vector.shuffle %71, %72 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %82 = vector.shuffle %73, %74 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %83 = vector.shuffle %75, %76 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %84 = vector.shuffle %77, %78 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %85 = vector.shuffle %79, %80 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %86 = vector.shuffle %81, %82 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %87 = vector.shuffle %83, %84 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<64xf16>, vector<64xf16>
      %88 = vector.shuffle %85, %86 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<64xf16>, vector<64xf16>
      %89 = vector.shuffle %87, %88 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<128xf16>, vector<128xf16>
      %90 = vector.shuffle %89, %89 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
      %c134744586_i32 = arith.constant 134744586 : i32
      %91 = vector.bitcast %10 : vector<128xf16> to vector<64xi32>
      %92 = vector.bitcast %90 : vector<256xf16> to vector<128xi32>
      %93 = func.call @llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32(%92, %91, %c134744586_i32) : (vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32>
      %intptr = memref.extract_aligned_pointer_as_index %arg2 : memref<8x16xf32> -> index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %c4 = arith.constant 4 : index
      %94 = arith.index_castui %intptr : index to i64
      %cst_0 = arith.constant dense<0> : vector<8xi64>
      %95 = vector.insert %94, %cst_0 [0] : i64 into vector<8xi64>
      %96 = vector.bitcast %95 : vector<8xi64> to vector<16xi32>
      %c63_i32 = arith.constant 63 : i32
      %c7_i32 = arith.constant 7 : i32
      %c63_i32_1 = arith.constant 63 : i32
      %97 = vector.insert %c63_i32, %96 [2] : i32 into vector<16xi32>
      %98 = vector.insert %c7_i32, %97 [3] : i32 into vector<16xi32>
      %99 = vector.insert %c63_i32_1, %98 [4] : i32 into vector<16xi32>
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_2 = arith.constant 0 : i32
      %100 = vector.insert %c0_i32, %99 [5] : i32 into vector<16xi32>
      %101 = vector.insert %c0_i32_2, %100 [6] : i32 into vector<16xi32>
      %c1807_i32 = arith.constant 1807 : i32
      %102 = vector.insert %c1807_i32, %101 [7] : i32 into vector<16xi32>
      %true = arith.constant true
      %c3_i8 = arith.constant 3 : i8
      %c3_i8_3 = arith.constant 3 : i8
      %103 = vector.from_elements %c3_i8, %c3_i8_3 : vector<2xi8>
      %c1_i8 = arith.constant 1 : i8
      %c16_i16 = arith.constant 16 : i16
      %c8_i16 = arith.constant 8 : i16
      %c0_i32_4 = arith.constant 0 : i32
      %c0_i32_5 = arith.constant 0 : i32
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %103, %c1_i8, %c16_i16, %c8_i16, %102, %c0_i32_4, %c0_i32_5, %93) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      gpu.return
    }
  }
}


// -----// IR Dump After BF16ToGPU (bf16-to-gpu) //----- //
module {
  gpu.module @test_kernels {
    func.func private @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32", linkage_type = <Import>>}
    func.func private @llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32(vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32", linkage_type = <Import>>}
    gpu.func @bcast_add_exp(%arg0: vector<1x16xf16>, %arg1: vector<16x1xf16>, %arg2: memref<8x16xf32>) kernel {
      %0 = builtin.unrealized_conversion_cast %arg0 : vector<1x16xf16> to vector<16xf16>
      %1 = builtin.unrealized_conversion_cast %arg1 : vector<16x1xf16> to vector<16xf16>
      %cst = arith.constant dense<0.000000e+00> : vector<128xf16>
      %2 = vector.shuffle %0, %0 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf16>, vector<16xf16>
      %3 = vector.shuffle %cst, %2 [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %4 = vector.shuffle %3, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %5 = vector.shuffle %4, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %6 = vector.shuffle %5, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %7 = vector.shuffle %6, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %8 = vector.shuffle %7, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %9 = vector.shuffle %8, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf16>, vector<128xf16>
      %10 = vector.shuffle %9, %2 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143] : vector<128xf16>, vector<128xf16>
      %11 = vector.extract_strided_slice %1 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %c0 = arith.constant 0 : index
      %12 = vector.extractelement %11[%c0 : index] : vector<1xf16>
      %13 = vector.extract_strided_slice %1 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %14 = vector.extractelement %13[%c0 : index] : vector<1xf16>
      %15 = vector.extract_strided_slice %1 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %16 = vector.extractelement %15[%c0 : index] : vector<1xf16>
      %17 = vector.extract_strided_slice %1 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %18 = vector.extractelement %17[%c0 : index] : vector<1xf16>
      %19 = vector.extract_strided_slice %1 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %20 = vector.extractelement %19[%c0 : index] : vector<1xf16>
      %21 = vector.extract_strided_slice %1 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %22 = vector.extractelement %21[%c0 : index] : vector<1xf16>
      %23 = vector.extract_strided_slice %1 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %24 = vector.extractelement %23[%c0 : index] : vector<1xf16>
      %25 = vector.extract_strided_slice %1 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %26 = vector.extractelement %25[%c0 : index] : vector<1xf16>
      %27 = vector.extract_strided_slice %1 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %28 = vector.extractelement %27[%c0 : index] : vector<1xf16>
      %29 = vector.extract_strided_slice %1 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %30 = vector.extractelement %29[%c0 : index] : vector<1xf16>
      %31 = vector.extract_strided_slice %1 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %32 = vector.extractelement %31[%c0 : index] : vector<1xf16>
      %33 = vector.extract_strided_slice %1 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %34 = vector.extractelement %33[%c0 : index] : vector<1xf16>
      %35 = vector.extract_strided_slice %1 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %36 = vector.extractelement %35[%c0 : index] : vector<1xf16>
      %37 = vector.extract_strided_slice %1 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %38 = vector.extractelement %37[%c0 : index] : vector<1xf16>
      %39 = vector.extract_strided_slice %1 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %40 = vector.extractelement %39[%c0 : index] : vector<1xf16>
      %41 = vector.extract_strided_slice %1 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      %42 = vector.extractelement %41[%c0 : index] : vector<1xf16>
      %43 = math.exp %12 : f16
      %44 = math.exp %14 : f16
      %45 = math.exp %16 : f16
      %46 = math.exp %18 : f16
      %47 = math.exp %20 : f16
      %48 = math.exp %22 : f16
      %49 = math.exp %24 : f16
      %50 = math.exp %26 : f16
      %51 = math.exp %28 : f16
      %52 = math.exp %30 : f16
      %53 = math.exp %32 : f16
      %54 = math.exp %34 : f16
      %55 = math.exp %36 : f16
      %56 = math.exp %38 : f16
      %57 = math.exp %40 : f16
      %58 = math.exp %42 : f16
      %59 = vector.splat %43 : vector<16xf16>
      %60 = vector.splat %44 : vector<16xf16>
      %61 = vector.splat %45 : vector<16xf16>
      %62 = vector.splat %46 : vector<16xf16>
      %63 = vector.splat %47 : vector<16xf16>
      %64 = vector.splat %48 : vector<16xf16>
      %65 = vector.splat %49 : vector<16xf16>
      %66 = vector.splat %50 : vector<16xf16>
      %67 = vector.splat %51 : vector<16xf16>
      %68 = vector.splat %52 : vector<16xf16>
      %69 = vector.splat %53 : vector<16xf16>
      %70 = vector.splat %54 : vector<16xf16>
      %71 = vector.splat %55 : vector<16xf16>
      %72 = vector.splat %56 : vector<16xf16>
      %73 = vector.splat %57 : vector<16xf16>
      %74 = vector.splat %58 : vector<16xf16>
      %75 = vector.shuffle %59, %60 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %76 = vector.shuffle %61, %62 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %77 = vector.shuffle %63, %64 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %78 = vector.shuffle %65, %66 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %79 = vector.shuffle %67, %68 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %80 = vector.shuffle %69, %70 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %81 = vector.shuffle %71, %72 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %82 = vector.shuffle %73, %74 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      %83 = vector.shuffle %75, %76 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %84 = vector.shuffle %77, %78 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %85 = vector.shuffle %79, %80 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %86 = vector.shuffle %81, %82 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      %87 = vector.shuffle %83, %84 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<64xf16>, vector<64xf16>
      %88 = vector.shuffle %85, %86 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<64xf16>, vector<64xf16>
      %89 = vector.shuffle %87, %88 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<128xf16>, vector<128xf16>
      %90 = vector.shuffle %89, %89 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
      %c134744586_i32 = arith.constant 134744586 : i32
      %91 = vector.bitcast %10 : vector<128xf16> to vector<64xi32>
      %92 = vector.bitcast %90 : vector<256xf16> to vector<128xi32>
      %93 = func.call @llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32(%92, %91, %c134744586_i32) : (vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32>
      %intptr = memref.extract_aligned_pointer_as_index %arg2 : memref<8x16xf32> -> index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %c4 = arith.constant 4 : index
      %94 = arith.index_castui %intptr : index to i64
      %cst_0 = arith.constant dense<0> : vector<8xi64>
      %95 = vector.insert %94, %cst_0 [0] : i64 into vector<8xi64>
      %96 = vector.bitcast %95 : vector<8xi64> to vector<16xi32>
      %c63_i32 = arith.constant 63 : i32
      %c7_i32 = arith.constant 7 : i32
      %c63_i32_1 = arith.constant 63 : i32
      %97 = vector.insert %c63_i32, %96 [2] : i32 into vector<16xi32>
      %98 = vector.insert %c7_i32, %97 [3] : i32 into vector<16xi32>
      %99 = vector.insert %c63_i32_1, %98 [4] : i32 into vector<16xi32>
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_2 = arith.constant 0 : i32
      %100 = vector.insert %c0_i32, %99 [5] : i32 into vector<16xi32>
      %101 = vector.insert %c0_i32_2, %100 [6] : i32 into vector<16xi32>
      %c1807_i32 = arith.constant 1807 : i32
      %102 = vector.insert %c1807_i32, %101 [7] : i32 into vector<16xi32>
      %true = arith.constant true
      %c3_i8 = arith.constant 3 : i8
      %c3_i8_3 = arith.constant 3 : i8
      %103 = vector.from_elements %c3_i8, %c3_i8_3 : vector<2xi8>
      %c1_i8 = arith.constant 1 : i8
      %c16_i16 = arith.constant 16 : i16
      %c8_i16 = arith.constant 8 : i16
      %c0_i32_4 = arith.constant 0 : i32
      %c0_i32_5 = arith.constant 0 : i32
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %103, %c1_i8, %c16_i16, %c8_i16, %102, %c0_i32_4, %c0_i32_5, %93) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      gpu.return
    }
  }
}


../Transforms/RemoveSingleElemVector/unit_tests.mlir:4:3: error: failed to legalize operation 'func.func'
  gpu.module @test_kernels {
  ^
../Transforms/RemoveSingleElemVector/unit_tests.mlir:4:3: note: see current operation:
"func.func"() <{function_type = (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> (), sym_name = "llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32", sym_visibility = "private"}> ({
}) {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32", linkage_type = <Import>>} : () -> ()
// -----// IR Dump After ConvertGPUXToSPIRV Failed (imex-convert-gpu-to-spirv) //----- //
"builtin.module"() ({
  "gpu.module"() <{sym_name = "test_kernels"}> ({
    "func.func"() <{function_type = (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> (), sym_name = "llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32", sym_visibility = "private"}> ({
    }) {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32", linkage_type = <Import>>} : () -> ()
    "func.func"() <{function_type = (vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32>, sym_name = "llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32", sym_visibility = "private"}> ({
    }) {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32", linkage_type = <Import>>} : () -> ()
    "gpu.func"() <{function_type = (vector<1x16xf16>, vector<16x1xf16>, memref<8x16xf32, #spirv.storage_class<CrossWorkgroup>>) -> ()}> ({
    ^bb0(%arg3: vector<1x16xf16>, %arg4: vector<16x1xf16>, %arg5: memref<8x16xf32, #spirv.storage_class<CrossWorkgroup>>):
      %127 = "builtin.unrealized_conversion_cast"(%arg3) : (vector<1x16xf16>) -> vector<16xf16>
      %128 = "builtin.unrealized_conversion_cast"(%arg4) : (vector<16x1xf16>) -> vector<16xf16>
      %129 = "arith.constant"() <{value = dense<0.000000e+00> : vector<128xf16>}> : () -> vector<128xf16>
      %130 = "vector.shuffle"(%127, %127) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> : (vector<16xf16>, vector<16xf16>) -> vector<128xf16>
      %131 = "vector.shuffle"(%129, %130) <{mask = array<i64: 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>}> : (vector<128xf16>, vector<128xf16>) -> vector<128xf16>
      %132 = "vector.shuffle"(%131, %130) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>}> : (vector<128xf16>, vector<128xf16>) -> vector<128xf16>
      %133 = "vector.shuffle"(%132, %130) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>}> : (vector<128xf16>, vector<128xf16>) -> vector<128xf16>
      %134 = "vector.shuffle"(%133, %130) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>}> : (vector<128xf16>, vector<128xf16>) -> vector<128xf16>
      %135 = "vector.shuffle"(%134, %130) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>}> : (vector<128xf16>, vector<128xf16>) -> vector<128xf16>
      %136 = "vector.shuffle"(%135, %130) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>}> : (vector<128xf16>, vector<128xf16>) -> vector<128xf16>
      %137 = "vector.shuffle"(%136, %130) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>}> : (vector<128xf16>, vector<128xf16>) -> vector<128xf16>
      %138 = "vector.shuffle"(%137, %130) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143>}> : (vector<128xf16>, vector<128xf16>) -> vector<128xf16>
      %139 = "vector.extract_strided_slice"(%128) <{offsets = [0], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %140 = "arith.constant"() <{value = 0 : index}> : () -> index
      %141 = "vector.extractelement"(%139, %140) : (vector<1xf16>, index) -> f16
      %142 = "vector.extract_strided_slice"(%128) <{offsets = [1], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %143 = "vector.extractelement"(%142, %140) : (vector<1xf16>, index) -> f16
      %144 = "vector.extract_strided_slice"(%128) <{offsets = [2], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %145 = "vector.extractelement"(%144, %140) : (vector<1xf16>, index) -> f16
      %146 = "vector.extract_strided_slice"(%128) <{offsets = [3], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %147 = "vector.extractelement"(%146, %140) : (vector<1xf16>, index) -> f16
      %148 = "vector.extract_strided_slice"(%128) <{offsets = [4], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %149 = "vector.extractelement"(%148, %140) : (vector<1xf16>, index) -> f16
      %150 = "vector.extract_strided_slice"(%128) <{offsets = [5], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %151 = "vector.extractelement"(%150, %140) : (vector<1xf16>, index) -> f16
      %152 = "vector.extract_strided_slice"(%128) <{offsets = [6], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %153 = "vector.extractelement"(%152, %140) : (vector<1xf16>, index) -> f16
      %154 = "vector.extract_strided_slice"(%128) <{offsets = [7], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %155 = "vector.extractelement"(%154, %140) : (vector<1xf16>, index) -> f16
      %156 = "vector.extract_strided_slice"(%128) <{offsets = [8], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %157 = "vector.extractelement"(%156, %140) : (vector<1xf16>, index) -> f16
      %158 = "vector.extract_strided_slice"(%128) <{offsets = [9], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %159 = "vector.extractelement"(%158, %140) : (vector<1xf16>, index) -> f16
      %160 = "vector.extract_strided_slice"(%128) <{offsets = [10], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %161 = "vector.extractelement"(%160, %140) : (vector<1xf16>, index) -> f16
      %162 = "vector.extract_strided_slice"(%128) <{offsets = [11], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %163 = "vector.extractelement"(%162, %140) : (vector<1xf16>, index) -> f16
      %164 = "vector.extract_strided_slice"(%128) <{offsets = [12], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %165 = "vector.extractelement"(%164, %140) : (vector<1xf16>, index) -> f16
      %166 = "vector.extract_strided_slice"(%128) <{offsets = [13], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %167 = "vector.extractelement"(%166, %140) : (vector<1xf16>, index) -> f16
      %168 = "vector.extract_strided_slice"(%128) <{offsets = [14], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %169 = "vector.extractelement"(%168, %140) : (vector<1xf16>, index) -> f16
      %170 = "vector.extract_strided_slice"(%128) <{offsets = [15], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %171 = "vector.extractelement"(%170, %140) : (vector<1xf16>, index) -> f16
      %172 = "math.exp"(%141) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %173 = "math.exp"(%143) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %174 = "math.exp"(%145) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %175 = "math.exp"(%147) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %176 = "math.exp"(%149) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %177 = "math.exp"(%151) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %178 = "math.exp"(%153) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %179 = "math.exp"(%155) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %180 = "math.exp"(%157) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %181 = "math.exp"(%159) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %182 = "math.exp"(%161) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %183 = "math.exp"(%163) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %184 = "math.exp"(%165) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %185 = "math.exp"(%167) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %186 = "math.exp"(%169) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %187 = "math.exp"(%171) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %188 = "vector.splat"(%172) : (f16) -> vector<16xf16>
      %189 = "vector.splat"(%173) : (f16) -> vector<16xf16>
      %190 = "vector.splat"(%174) : (f16) -> vector<16xf16>
      %191 = "vector.splat"(%175) : (f16) -> vector<16xf16>
      %192 = "vector.splat"(%176) : (f16) -> vector<16xf16>
      %193 = "vector.splat"(%177) : (f16) -> vector<16xf16>
      %194 = "vector.splat"(%178) : (f16) -> vector<16xf16>
      %195 = "vector.splat"(%179) : (f16) -> vector<16xf16>
      %196 = "vector.splat"(%180) : (f16) -> vector<16xf16>
      %197 = "vector.splat"(%181) : (f16) -> vector<16xf16>
      %198 = "vector.splat"(%182) : (f16) -> vector<16xf16>
      %199 = "vector.splat"(%183) : (f16) -> vector<16xf16>
      %200 = "vector.splat"(%184) : (f16) -> vector<16xf16>
      %201 = "vector.splat"(%185) : (f16) -> vector<16xf16>
      %202 = "vector.splat"(%186) : (f16) -> vector<16xf16>
      %203 = "vector.splat"(%187) : (f16) -> vector<16xf16>
      %204 = "vector.shuffle"(%188, %189) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31>}> : (vector<16xf16>, vector<16xf16>) -> vector<32xf16>
      %205 = "vector.shuffle"(%190, %191) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31>}> : (vector<16xf16>, vector<16xf16>) -> vector<32xf16>
      %206 = "vector.shuffle"(%192, %193) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31>}> : (vector<16xf16>, vector<16xf16>) -> vector<32xf16>
      %207 = "vector.shuffle"(%194, %195) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31>}> : (vector<16xf16>, vector<16xf16>) -> vector<32xf16>
      %208 = "vector.shuffle"(%196, %197) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31>}> : (vector<16xf16>, vector<16xf16>) -> vector<32xf16>
      %209 = "vector.shuffle"(%198, %199) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31>}> : (vector<16xf16>, vector<16xf16>) -> vector<32xf16>
      %210 = "vector.shuffle"(%200, %201) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31>}> : (vector<16xf16>, vector<16xf16>) -> vector<32xf16>
      %211 = "vector.shuffle"(%202, %203) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31>}> : (vector<16xf16>, vector<16xf16>) -> vector<32xf16>
      %212 = "vector.shuffle"(%204, %205) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63>}> : (vector<32xf16>, vector<32xf16>) -> vector<64xf16>
      %213 = "vector.shuffle"(%206, %207) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63>}> : (vector<32xf16>, vector<32xf16>) -> vector<64xf16>
      %214 = "vector.shuffle"(%208, %209) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63>}> : (vector<32xf16>, vector<32xf16>) -> vector<64xf16>
      %215 = "vector.shuffle"(%210, %211) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63>}> : (vector<32xf16>, vector<32xf16>) -> vector<64xf16>
      %216 = "vector.shuffle"(%212, %213) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>}> : (vector<64xf16>, vector<64xf16>) -> vector<128xf16>
      %217 = "vector.shuffle"(%214, %215) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>}> : (vector<64xf16>, vector<64xf16>) -> vector<128xf16>
      %218 = "vector.shuffle"(%216, %217) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255>}> : (vector<128xf16>, vector<128xf16>) -> vector<256xf16>
      %219 = "vector.shuffle"(%218, %218) <{mask = array<i64: 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255>}> {packed} : (vector<256xf16>, vector<256xf16>) -> vector<256xf16>
      %220 = "arith.constant"() <{value = 134744586 : i32}> : () -> i32
      %221 = "vector.bitcast"(%138) : (vector<128xf16>) -> vector<64xi32>
      %222 = "vector.bitcast"(%219) : (vector<256xf16>) -> vector<128xi32>
      %223 = "func.call"(%222, %221, %220) <{callee = @llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32}> : (vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32>
      %224 = "memref.extract_aligned_pointer_as_index"(%arg5) : (memref<8x16xf32, #spirv.storage_class<CrossWorkgroup>>) -> index
      %225 = "arith.constant"() <{value = 16 : index}> : () -> index
      %226 = "arith.constant"() <{value = 1 : index}> : () -> index
      %227 = "arith.constant"() <{value = 32 : index}> : () -> index
      %228 = "arith.constant"() <{value = 4 : index}> : () -> index
      %229 = "arith.index_castui"(%224) : (index) -> i64
      %230 = "arith.constant"() <{value = dense<0> : vector<8xi64>}> : () -> vector<8xi64>
      %231 = "vector.insert"(%229, %230) <{static_position = array<i64: 0>}> : (i64, vector<8xi64>) -> vector<8xi64>
      %232 = "vector.bitcast"(%231) : (vector<8xi64>) -> vector<16xi32>
      %233 = "arith.constant"() <{value = 63 : i32}> : () -> i32
      %234 = "arith.constant"() <{value = 7 : i32}> : () -> i32
      %235 = "arith.constant"() <{value = 63 : i32}> : () -> i32
      %236 = "vector.insert"(%233, %232) <{static_position = array<i64: 2>}> : (i32, vector<16xi32>) -> vector<16xi32>
      %237 = "vector.insert"(%234, %236) <{static_position = array<i64: 3>}> : (i32, vector<16xi32>) -> vector<16xi32>
      %238 = "vector.insert"(%235, %237) <{static_position = array<i64: 4>}> : (i32, vector<16xi32>) -> vector<16xi32>
      %239 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %240 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %241 = "vector.insert"(%239, %238) <{static_position = array<i64: 5>}> : (i32, vector<16xi32>) -> vector<16xi32>
      %242 = "vector.insert"(%240, %241) <{static_position = array<i64: 6>}> : (i32, vector<16xi32>) -> vector<16xi32>
      %243 = "arith.constant"() <{value = 1807 : i32}> : () -> i32
      %244 = "vector.insert"(%243, %242) <{static_position = array<i64: 7>}> : (i32, vector<16xi32>) -> vector<16xi32>
      %245 = "arith.constant"() <{value = true}> : () -> i1
      %246 = "arith.constant"() <{value = 3 : i8}> : () -> i8
      %247 = "arith.constant"() <{value = 3 : i8}> : () -> i8
      %248 = "vector.from_elements"(%246, %247) : (i8, i8) -> vector<2xi8>
      %249 = "arith.constant"() <{value = 1 : i8}> : () -> i8
      %250 = "arith.constant"() <{value = 16 : i16}> : () -> i16
      %251 = "arith.constant"() <{value = 8 : i16}> : () -> i16
      %252 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %253 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      "func.call"(%245, %248, %249, %250, %251, %244, %252, %253, %223) <{callee = @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32}> : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      "gpu.return"() : () -> ()
    }) {gpu.kernel, sym_name = "bcast_add_exp", workgroup_attributions = 0 : i64} : () -> ()
  }) : () -> ()
  "gpu.module"() <{sym_name = "test_kernels"}> ({
    "func.func"() <{function_type = (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> (), sym_name = "llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32", sym_visibility = "private"}> ({
    }) {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32", linkage_type = <Import>>} : () -> ()
    "func.func"() <{function_type = (vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32>, sym_name = "llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32", sym_visibility = "private"}> ({
    }) {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32", linkage_type = <Import>>} : () -> ()
    "gpu.func"() <{function_type = (vector<1x16xf16>, vector<16x1xf16>, memref<8x16xf32>) -> ()}> ({
    ^bb0(%arg0: vector<1x16xf16>, %arg1: vector<16x1xf16>, %arg2: memref<8x16xf32>):
      %0 = "builtin.unrealized_conversion_cast"(%arg0) : (vector<1x16xf16>) -> vector<16xf16>
      %1 = "builtin.unrealized_conversion_cast"(%arg1) : (vector<16x1xf16>) -> vector<16xf16>
      %2 = "arith.constant"() <{value = dense<0.000000e+00> : vector<128xf16>}> : () -> vector<128xf16>
      %3 = "vector.shuffle"(%0, %0) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> : (vector<16xf16>, vector<16xf16>) -> vector<128xf16>
      %4 = "vector.shuffle"(%2, %3) <{mask = array<i64: 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>}> : (vector<128xf16>, vector<128xf16>) -> vector<128xf16>
      %5 = "vector.shuffle"(%4, %3) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>}> : (vector<128xf16>, vector<128xf16>) -> vector<128xf16>
      %6 = "vector.shuffle"(%5, %3) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>}> : (vector<128xf16>, vector<128xf16>) -> vector<128xf16>
      %7 = "vector.shuffle"(%6, %3) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>}> : (vector<128xf16>, vector<128xf16>) -> vector<128xf16>
      %8 = "vector.shuffle"(%7, %3) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>}> : (vector<128xf16>, vector<128xf16>) -> vector<128xf16>
      %9 = "vector.shuffle"(%8, %3) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>}> : (vector<128xf16>, vector<128xf16>) -> vector<128xf16>
      %10 = "vector.shuffle"(%9, %3) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>}> : (vector<128xf16>, vector<128xf16>) -> vector<128xf16>
      %11 = "vector.shuffle"(%10, %3) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143>}> : (vector<128xf16>, vector<128xf16>) -> vector<128xf16>
      %12 = "vector.extract_strided_slice"(%1) <{offsets = [0], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %13 = "arith.constant"() <{value = 0 : index}> : () -> index
      %14 = "vector.extractelement"(%12, %13) : (vector<1xf16>, index) -> f16
      %15 = "vector.extract_strided_slice"(%1) <{offsets = [1], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %16 = "vector.extractelement"(%15, %13) : (vector<1xf16>, index) -> f16
      %17 = "vector.extract_strided_slice"(%1) <{offsets = [2], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %18 = "vector.extractelement"(%17, %13) : (vector<1xf16>, index) -> f16
      %19 = "vector.extract_strided_slice"(%1) <{offsets = [3], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %20 = "vector.extractelement"(%19, %13) : (vector<1xf16>, index) -> f16
      %21 = "vector.extract_strided_slice"(%1) <{offsets = [4], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %22 = "vector.extractelement"(%21, %13) : (vector<1xf16>, index) -> f16
      %23 = "vector.extract_strided_slice"(%1) <{offsets = [5], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %24 = "vector.extractelement"(%23, %13) : (vector<1xf16>, index) -> f16
      %25 = "vector.extract_strided_slice"(%1) <{offsets = [6], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %26 = "vector.extractelement"(%25, %13) : (vector<1xf16>, index) -> f16
      %27 = "vector.extract_strided_slice"(%1) <{offsets = [7], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %28 = "vector.extractelement"(%27, %13) : (vector<1xf16>, index) -> f16
      %29 = "vector.extract_strided_slice"(%1) <{offsets = [8], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %30 = "vector.extractelement"(%29, %13) : (vector<1xf16>, index) -> f16
      %31 = "vector.extract_strided_slice"(%1) <{offsets = [9], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %32 = "vector.extractelement"(%31, %13) : (vector<1xf16>, index) -> f16
      %33 = "vector.extract_strided_slice"(%1) <{offsets = [10], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %34 = "vector.extractelement"(%33, %13) : (vector<1xf16>, index) -> f16
      %35 = "vector.extract_strided_slice"(%1) <{offsets = [11], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %36 = "vector.extractelement"(%35, %13) : (vector<1xf16>, index) -> f16
      %37 = "vector.extract_strided_slice"(%1) <{offsets = [12], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %38 = "vector.extractelement"(%37, %13) : (vector<1xf16>, index) -> f16
      %39 = "vector.extract_strided_slice"(%1) <{offsets = [13], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %40 = "vector.extractelement"(%39, %13) : (vector<1xf16>, index) -> f16
      %41 = "vector.extract_strided_slice"(%1) <{offsets = [14], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %42 = "vector.extractelement"(%41, %13) : (vector<1xf16>, index) -> f16
      %43 = "vector.extract_strided_slice"(%1) <{offsets = [15], sizes = [1], strides = [1]}> : (vector<16xf16>) -> vector<1xf16>
      %44 = "vector.extractelement"(%43, %13) : (vector<1xf16>, index) -> f16
      %45 = "math.exp"(%14) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %46 = "math.exp"(%16) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %47 = "math.exp"(%18) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %48 = "math.exp"(%20) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %49 = "math.exp"(%22) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %50 = "math.exp"(%24) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %51 = "math.exp"(%26) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %52 = "math.exp"(%28) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %53 = "math.exp"(%30) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %54 = "math.exp"(%32) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %55 = "math.exp"(%34) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %56 = "math.exp"(%36) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %57 = "math.exp"(%38) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %58 = "math.exp"(%40) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %59 = "math.exp"(%42) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %60 = "math.exp"(%44) <{fastmath = #arith.fastmath<none>}> : (f16) -> f16
      %61 = "vector.splat"(%45) : (f16) -> vector<16xf16>
      %62 = "vector.splat"(%46) : (f16) -> vector<16xf16>
      %63 = "vector.splat"(%47) : (f16) -> vector<16xf16>
      %64 = "vector.splat"(%48) : (f16) -> vector<16xf16>
      %65 = "vector.splat"(%49) : (f16) -> vector<16xf16>
      %66 = "vector.splat"(%50) : (f16) -> vector<16xf16>
      %67 = "vector.splat"(%51) : (f16) -> vector<16xf16>
      %68 = "vector.splat"(%52) : (f16) -> vector<16xf16>
      %69 = "vector.splat"(%53) : (f16) -> vector<16xf16>
      %70 = "vector.splat"(%54) : (f16) -> vector<16xf16>
      %71 = "vector.splat"(%55) : (f16) -> vector<16xf16>
      %72 = "vector.splat"(%56) : (f16) -> vector<16xf16>
      %73 = "vector.splat"(%57) : (f16) -> vector<16xf16>
      %74 = "vector.splat"(%58) : (f16) -> vector<16xf16>
      %75 = "vector.splat"(%59) : (f16) -> vector<16xf16>
      %76 = "vector.splat"(%60) : (f16) -> vector<16xf16>
      %77 = "vector.shuffle"(%61, %62) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31>}> : (vector<16xf16>, vector<16xf16>) -> vector<32xf16>
      %78 = "vector.shuffle"(%63, %64) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31>}> : (vector<16xf16>, vector<16xf16>) -> vector<32xf16>
      %79 = "vector.shuffle"(%65, %66) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31>}> : (vector<16xf16>, vector<16xf16>) -> vector<32xf16>
      %80 = "vector.shuffle"(%67, %68) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31>}> : (vector<16xf16>, vector<16xf16>) -> vector<32xf16>
      %81 = "vector.shuffle"(%69, %70) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31>}> : (vector<16xf16>, vector<16xf16>) -> vector<32xf16>
      %82 = "vector.shuffle"(%71, %72) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31>}> : (vector<16xf16>, vector<16xf16>) -> vector<32xf16>
      %83 = "vector.shuffle"(%73, %74) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31>}> : (vector<16xf16>, vector<16xf16>) -> vector<32xf16>
      %84 = "vector.shuffle"(%75, %76) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31>}> : (vector<16xf16>, vector<16xf16>) -> vector<32xf16>
      %85 = "vector.shuffle"(%77, %78) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63>}> : (vector<32xf16>, vector<32xf16>) -> vector<64xf16>
      %86 = "vector.shuffle"(%79, %80) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63>}> : (vector<32xf16>, vector<32xf16>) -> vector<64xf16>
      %87 = "vector.shuffle"(%81, %82) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63>}> : (vector<32xf16>, vector<32xf16>) -> vector<64xf16>
      %88 = "vector.shuffle"(%83, %84) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63>}> : (vector<32xf16>, vector<32xf16>) -> vector<64xf16>
      %89 = "vector.shuffle"(%85, %86) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>}> : (vector<64xf16>, vector<64xf16>) -> vector<128xf16>
      %90 = "vector.shuffle"(%87, %88) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>}> : (vector<64xf16>, vector<64xf16>) -> vector<128xf16>
      %91 = "vector.shuffle"(%89, %90) <{mask = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255>}> : (vector<128xf16>, vector<128xf16>) -> vector<256xf16>
      %92 = "vector.shuffle"(%91, %91) <{mask = array<i64: 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255>}> {packed} : (vector<256xf16>, vector<256xf16>) -> vector<256xf16>
      %93 = "arith.constant"() <{value = 134744586 : i32}> : () -> i32
      %94 = "vector.bitcast"(%11) : (vector<128xf16>) -> vector<64xi32>
      %95 = "vector.bitcast"(%92) : (vector<256xf16>) -> vector<128xi32>
      %96 = "func.call"(%95, %94, %93) <{callee = @llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32}> : (vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32>
      %97 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<8x16xf32>) -> index
      %98 = "arith.constant"() <{value = 16 : index}> : () -> index
      %99 = "arith.constant"() <{value = 1 : index}> : () -> index
      %100 = "arith.constant"() <{value = 32 : index}> : () -> index
      %101 = "arith.constant"() <{value = 4 : index}> : () -> index
      %102 = "arith.index_castui"(%97) : (index) -> i64
      %103 = "arith.constant"() <{value = dense<0> : vector<8xi64>}> : () -> vector<8xi64>
      %104 = "vector.insert"(%102, %103) <{static_position = array<i64: 0>}> : (i64, vector<8xi64>) -> vector<8xi64>
      %105 = "vector.bitcast"(%104) : (vector<8xi64>) -> vector<16xi32>
      %106 = "arith.constant"() <{value = 63 : i32}> : () -> i32
      %107 = "arith.constant"() <{value = 7 : i32}> : () -> i32
      %108 = "arith.constant"() <{value = 63 : i32}> : () -> i32
      %109 = "vector.insert"(%106, %105) <{static_position = array<i64: 2>}> : (i32, vector<16xi32>) -> vector<16xi32>
      %110 = "vector.insert"(%107, %109) <{static_position = array<i64: 3>}> : (i32, vector<16xi32>) -> vector<16xi32>
      %111 = "vector.insert"(%108, %110) <{static_position = array<i64: 4>}> : (i32, vector<16xi32>) -> vector<16xi32>
      %112 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %113 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %114 = "vector.insert"(%112, %111) <{static_position = array<i64: 5>}> : (i32, vector<16xi32>) -> vector<16xi32>
      %115 = "vector.insert"(%113, %114) <{static_position = array<i64: 6>}> : (i32, vector<16xi32>) -> vector<16xi32>
      %116 = "arith.constant"() <{value = 1807 : i32}> : () -> i32
      %117 = "vector.insert"(%116, %115) <{static_position = array<i64: 7>}> : (i32, vector<16xi32>) -> vector<16xi32>
      %118 = "arith.constant"() <{value = true}> : () -> i1
      %119 = "arith.constant"() <{value = 3 : i8}> : () -> i8
      %120 = "arith.constant"() <{value = 3 : i8}> : () -> i8
      %121 = "vector.from_elements"(%119, %120) : (i8, i8) -> vector<2xi8>
      %122 = "arith.constant"() <{value = 1 : i8}> : () -> i8
      %123 = "arith.constant"() <{value = 16 : i16}> : () -> i16
      %124 = "arith.constant"() <{value = 8 : i16}> : () -> i16
      %125 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %126 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      "func.call"(%118, %121, %122, %123, %124, %117, %125, %126, %96) <{callee = @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32}> : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      "gpu.return"() : () -> ()
    }) {gpu.kernel, sym_name = "bcast_add_exp", workgroup_attributions = 0 : i64} : () -> ()
  }) : () -> ()
}) : () -> ()


Error: entry point not found
