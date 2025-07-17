// RUN: imex-opt %s -split-input-file -imex-xegpu-apply-vnni-transformation | FileCheck %s

gpu.module @test_kernel {
  gpu.func @test_gemm(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %0 = arith.muli %block_id_x, %c32 : index
    %1 = arith.muli %block_id_y, %c32 : index
    %2 = arith.addi %0, %c0 : index
    %3 = arith.addi %1, %c0 : index
    %4 = xegpu.create_nd_tdesc %arg2[%2, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %c16 = arith.constant 16 : index
    %5 = arith.addi %1, %c16 : index
    %6 = xegpu.create_nd_tdesc %arg2[%2, %5] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %c8 = arith.constant 8 : index
    %7 = arith.addi %0, %c8 : index
    %8 = xegpu.create_nd_tdesc %arg2[%7, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %9 = xegpu.create_nd_tdesc %arg2[%7, %5] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %10 = arith.addi %0, %c16 : index
    %11 = xegpu.create_nd_tdesc %arg2[%10, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %12 = xegpu.create_nd_tdesc %arg2[%10, %5] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %c24 = arith.constant 24 : index
    %13 = arith.addi %0, %c24 : index
    %14 = xegpu.create_nd_tdesc %arg2[%13, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %15 = xegpu.create_nd_tdesc %arg2[%13, %5] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %16 = xegpu.create_nd_tdesc %arg2[%2, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %17 = xegpu.create_nd_tdesc %arg2[%2, %5] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %18 = xegpu.load_nd %16 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    %19 = xegpu.load_nd %17 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    %20 = xegpu.create_nd_tdesc %arg0[%2, %c0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>
    %21 = xegpu.create_nd_tdesc %arg1[%c0, %3] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>
    %22:4 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %20, %arg5 = %21, %arg6 = %18, %arg7 = %19) -> (!xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>, vector<32x16xf32>, vector<32x16xf32>) {
      %31 = vector.extract_strided_slice %arg6 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %32 = vector.extract_strided_slice %arg6 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %33 = vector.extract_strided_slice %arg6 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %34 = vector.extract_strided_slice %arg6 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %35 = vector.extract_strided_slice %arg7 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %36 = vector.extract_strided_slice %arg7 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %37 = vector.extract_strided_slice %arg7 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %38 = vector.extract_strided_slice %arg7 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %39 = xegpu.load_nd %arg4 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
      %40 = vector.extract %39[0] : vector<32x16xf16> from vector<2x32x16xf16>
      %41 = vector.extract %39[1] : vector<32x16xf16> from vector<2x32x16xf16>
      %42 = vector.extract_strided_slice %40 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      %43 = vector.extract_strided_slice %40 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      %44 = vector.extract_strided_slice %40 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      %45 = vector.extract_strided_slice %40 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      %46 = vector.extract_strided_slice %41 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      %47 = vector.extract_strided_slice %41 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      %48 = vector.extract_strided_slice %41 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      %49 = vector.extract_strided_slice %41 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>

      //CHECK: %[[R50:.*]] = xegpu.load_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x16x16x2xf16>
      %50 = xegpu.load_nd %arg5 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>

      //CHECK: %[[R51:.*]] = vector.extract %[[R50]][0] : vector<16x16x2xf16> from vector<2x16x16x2xf16>
      //CHECK: %[[R52:.*]] = vector.extract %[[R50]][1] : vector<16x16x2xf16> from vector<2x16x16x2xf16>
      %51 = vector.extract %50[0] : vector<32x16xf16> from vector<2x32x16xf16>
      %52 = vector.extract %50[1] : vector<32x16xf16> from vector<2x32x16xf16>

      //CHECK: %[[R53:.*]] = vector.extract_strided_slice %[[R51]] {offsets = [0, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
      //CHECK: %[[R54:.*]] = vector.extract_strided_slice %[[R51]] {offsets = [8, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
      //CHECK: %[[R55:.*]] = vector.extract_strided_slice %[[R52]] {offsets = [0, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
      //CHECK: %[[R56:.*]] = vector.extract_strided_slice %[[R52]] {offsets = [8, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
      %53 = vector.extract_strided_slice %51 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %54 = vector.extract_strided_slice %51 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %55 = vector.extract_strided_slice %52 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %56 = vector.extract_strided_slice %52 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>

      //CHECK: %[[R57:.*]] = math.exp %[[R53]] : vector<8x16x2xf16>
      //CHECK: %[[R58:.*]] = math.exp %[[R55]] : vector<8x16x2xf16>
      //CHECK: %[[R59:.*]] = math.exp %[[R54]] : vector<8x16x2xf16>
      //CHECK: %[[R60:.*]] = math.exp %[[R56]] : vector<8x16x2xf16>
      %57 = math.exp %53 : vector<16x16xf16>
      %58 = math.exp %55 : vector<16x16xf16>
      %59 = math.exp %54 : vector<16x16xf16>
      %60 = math.exp %56 : vector<16x16xf16>

      //CHECK: %[[R61:.*]] = xegpu.dpas {{.*}}, %[[R57]], {{.*}} : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[R62:.*]] = xegpu.dpas {{.*}}, %[[R59]], {{.*}} : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[R63:.*]] = xegpu.dpas {{.*}}, %[[R58]], {{.*}} : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[R64:.*]] = xegpu.dpas {{.*}}, %[[R60]], {{.*}} : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[R65:.*]] = xegpu.dpas {{.*}}, %[[R57]], {{.*}} : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[R66:.*]] = xegpu.dpas {{.*}}, %[[R59]], {{.*}} : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[R67:.*]] = xegpu.dpas {{.*}}, %[[R58]], {{.*}} : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[R68:.*]] = xegpu.dpas {{.*}}, %[[R60]], {{.*}} : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[R69:.*]] = xegpu.dpas {{.*}}, %[[R57]], {{.*}} : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[R70:.*]] = xegpu.dpas {{.*}}, %[[R59]], {{.*}} : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[R71:.*]] = xegpu.dpas {{.*}}, %[[R58]], {{.*}} : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[R72:.*]] = xegpu.dpas {{.*}}, %[[R60]], {{.*}} : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[R73:.*]] = xegpu.dpas {{.*}}, %[[R57]], {{.*}} : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[R74:.*]] = xegpu.dpas {{.*}}, %[[R59]], {{.*}} : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[R75:.*]] = xegpu.dpas {{.*}}, %[[R58]], {{.*}} : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[R76:.*]] = xegpu.dpas {{.*}}, %[[R60]], {{.*}} : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %61 = xegpu.dpas %42, %57, %31 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %62 = xegpu.dpas %46, %59, %61 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %63 = xegpu.dpas %42, %58, %35 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %64 = xegpu.dpas %46, %60, %63 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %65 = xegpu.dpas %43, %57, %32 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %66 = xegpu.dpas %47, %59, %65 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %67 = xegpu.dpas %43, %58, %36 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %68 = xegpu.dpas %47, %60, %67 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %69 = xegpu.dpas %44, %57, %33 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %70 = xegpu.dpas %48, %59, %69 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %71 = xegpu.dpas %44, %58, %37 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %72 = xegpu.dpas %48, %60, %71 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %73 = xegpu.dpas %45, %57, %34 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %74 = xegpu.dpas %49, %59, %73 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %75 = xegpu.dpas %45, %58, %38 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %76 = xegpu.dpas %49, %60, %75 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %77 = vector.shuffle %62, %66 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      %78 = vector.shuffle %70, %74 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      %79 = vector.shuffle %77, %78 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16x16xf32>, vector<16x16xf32>
      %80 = vector.shuffle %64, %68 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      %81 = vector.shuffle %72, %76 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      %82 = vector.shuffle %80, %81 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16x16xf32>, vector<16x16xf32>
      %83 = xegpu.update_nd_offset %arg4, [%c0, %c32] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>
      %84 = xegpu.update_nd_offset %arg5, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>
      scf.yield %83, %84, %79, %82 : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>, vector<32x16xf32>, vector<32x16xf32>
    }
    %23 = vector.extract_strided_slice %22#2 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %24 = vector.extract_strided_slice %22#2 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %25 = vector.extract_strided_slice %22#2 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %26 = vector.extract_strided_slice %22#2 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %27 = vector.extract_strided_slice %22#3 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %28 = vector.extract_strided_slice %22#3 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %29 = vector.extract_strided_slice %22#3 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %30 = vector.extract_strided_slice %22#3 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    xegpu.store_nd %23, %4 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %27, %6 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %24, %8 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %28, %9 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %25, %11 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %29, %12 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %26, %14 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %30, %15 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    gpu.return
  }
}
