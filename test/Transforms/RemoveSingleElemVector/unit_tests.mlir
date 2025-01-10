// RUN: imex-opt %s -split-input-file -imex-remove-single-elem-vector -canonicalize | FileCheck %s

module {
  gpu.module @test_kernels {
    gpu.func @bcast_add_exp(%arg0: vector<1x16xf16>, %arg1: vector<16x1xf16>, %arg2: memref<8x16xf32>) kernel {
      // Corresponding xetile code:
      // %0 = xetile.broadcast %a [0] : vector<1x16xf16> -> vector<8x16xf16>
      // %1 = math.exp %b: vector<16x1xf16>
      // %2 = xetile.broadcast %1 [1] : vector<16x1xf16> -> vector<16x16xf16>
      // %3 = xetile.tile_mma %0, %2: vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
      // %4 = xetile.init_tile %c[0, 0] : memref<8x16xf32> -> !xetile.tile<8x16xf32>
      // xetile.store_tile %3, %4: vector<8x16xf32>, !xetile.tile<8x16xf32>

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
      //CHECK: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [0], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extractelement %{{.*}}[%{{.*}} : index] : vector<1xf16>
      %12 = vector.extract_strided_slice %0 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [1], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extractelement %{{.*}}[%{{.*}} : index] : vector<1xf16>
      %13 = vector.extract_strided_slice %0 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [2], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extractelement %{{.*}}[%{{.*}} : index] : vector<1xf16>
      %14 = vector.extract_strided_slice %0 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [3], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extractelement %{{.*}}[%{{.*}} : index] : vector<1xf16>
      %15 = vector.extract_strided_slice %0 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [4], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extractelement %{{.*}}[%{{.*}} : index] : vector<1xf16>
      %16 = vector.extract_strided_slice %0 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [5], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extractelement %{{.*}}[%{{.*}} : index] : vector<1xf16>
      %17 = vector.extract_strided_slice %0 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [6], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extractelement %{{.*}}[%{{.*}} : index] : vector<1xf16>
      %18 = vector.extract_strided_slice %0 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [7], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extractelement %{{.*}}[%{{.*}} : index] : vector<1xf16>
      %19 = vector.extract_strided_slice %0 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [8], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extractelement %{{.*}}[%{{.*}} : index] : vector<1xf16>
      %20 = vector.extract_strided_slice %0 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [9], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extractelement %{{.*}}[%{{.*}} : index] : vector<1xf16>
      %21 = vector.extract_strided_slice %0 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [10], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extractelement %{{.*}}[%{{.*}} : index] : vector<1xf16>
      %22 = vector.extract_strided_slice %0 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [11], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extractelement %{{.*}}[%{{.*}} : index] : vector<1xf16>
      %23 = vector.extract_strided_slice %0 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [12], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extractelement %{{.*}}[%{{.*}} : index] : vector<1xf16>
      %24 = vector.extract_strided_slice %0 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [13], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extractelement %{{.*}}[%{{.*}} : index] : vector<1xf16>
      %25 = vector.extract_strided_slice %0 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [14], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extractelement %{{.*}}[%{{.*}} : index] : vector<1xf16>
      %26 = vector.extract_strided_slice %0 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [15], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK: %{{.*}} = vector.extractelement %{{.*}}[%{{.*}} : index] : vector<1xf16>
      %27 = vector.extract_strided_slice %0 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf16> to vector<1xf16>
      //CHECK-COUNT-16: {{.*}} = math.exp %{{.*}} : f16
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
      // CHECK-COUNT-16: %{{.*}} = vector.splat %{{.*}} : vector<16xf16>
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
