// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<256x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<256x256xf32>) -> memref<256x256xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %memref = gpu.alloc  host_shared () : memref<256x256xf16>
    memref.copy %arg0, %memref : memref<256x256xf16> to memref<256x256xf16>
    %memref_0 = gpu.alloc  host_shared () : memref<256x256xf16>
    memref.copy %arg1, %memref_0 : memref<256x256xf16> to memref<256x256xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<256x256xf32>
    memref.copy %arg2, %memref_1 : memref<256x256xf32> to memref<256x256xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c16, %c8, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<256x256xf16>, %memref_0 : memref<256x256xf16>, %memref_1 : memref<256x256xf32>)
    gpu.dealloc  %memref : memref<256x256xf16>
    gpu.dealloc  %memref_0 : memref<256x256xf16>
    return %memref_1 : memref<256x256xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<256x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<256x256xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c16 : index
      %1 = arith.muli %block_id_y, %c32 : index
      %2 = arith.addi %0, %c0 : index
      %3 = arith.addi %1, %c0 : index
      %4 = xegpu.create_nd_tdesc %arg2[%2, %3] : memref<256x256xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      %5 = arith.addi %1, %c16 : index
      %6 = xegpu.create_nd_tdesc %arg2[%2, %5] : memref<256x256xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      %c8 = arith.constant 8 : index
      %7 = arith.addi %0, %c8 : index
      %8 = xegpu.create_nd_tdesc %arg2[%7, %3] : memref<256x256xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      %9 = xegpu.create_nd_tdesc %arg2[%7, %5] : memref<256x256xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      %10 = xegpu.create_nd_tdesc %arg2[%2, %3] : memref<256x256xf32> -> !xegpu.tensor_desc<16x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      %11 = xegpu.create_nd_tdesc %arg2[%2, %5] : memref<256x256xf32> -> !xegpu.tensor_desc<16x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      %12 = xegpu.load_nd %10 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<16x16xf32>
      %13 = xegpu.load_nd %11 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<16x16xf32>
      %14 = xegpu.create_nd_tdesc %arg0[%2, %c0] : memref<256x256xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>
      %15 = xegpu.create_nd_tdesc %arg1[%3, %c0] : memref<256x256xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      %16 = xegpu.create_nd_tdesc %arg1[%3, %c16] : memref<256x256xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      %17:5 = scf.for %arg3 = %c0 to %c256 step %c32 iter_args(%arg4 = %14, %arg5 = %15, %arg6 = %16, %arg7 = %12, %arg8 = %13) -> (!xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>, vector<16x16xf32>, vector<16x16xf32>) {
        %22 = vector.extract_strided_slice %arg7 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf32> to vector<8x16xf32>
        %23 = vector.extract_strided_slice %arg7 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf32> to vector<8x16xf32>
        %24 = vector.extract_strided_slice %arg8 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf32> to vector<8x16xf32>
        %25 = vector.extract_strided_slice %arg8 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf32> to vector<8x16xf32>
        %26 = xegpu.load_nd %arg4 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x16x16xf16>
        %27 = vector.extract %26[0] : vector<16x16xf16> from vector<2x16x16xf16>
        %28 = vector.extract %26[1] : vector<16x16xf16> from vector<2x16x16xf16>
        %29 = vector.extract_strided_slice %27 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %30 = vector.extract_strided_slice %27 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %31 = vector.extract_strided_slice %28 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %32 = vector.extract_strided_slice %28 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %33 = xegpu.load_nd %arg5 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf16>
        %34 = xegpu.load_nd %arg6 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf16>
        %35 = vector.transpose %33, [1, 0] : vector<32x16xf16> to vector<16x32xf16>
        %36 = vector.shape_cast %35 {packed} : vector<16x32xf16> to vector<512xf16>
        %37 = vector.shuffle %36, %36 [0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47, 16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63, 64, 96, 65, 97, 66, 98, 67, 99, 68, 100, 69, 101, 70, 102, 71, 103, 72, 104, 73, 105, 74, 106, 75, 107, 76, 108, 77, 109, 78, 110, 79, 111, 80, 112, 81, 113, 82, 114, 83, 115, 84, 116, 85, 117, 86, 118, 87, 119, 88, 120, 89, 121, 90, 122, 91, 123, 92, 124, 93, 125, 94, 126, 95, 127, 128, 160, 129, 161, 130, 162, 131, 163, 132, 164, 133, 165, 134, 166, 135, 167, 136, 168, 137, 169, 138, 170, 139, 171, 140, 172, 141, 173, 142, 174, 143, 175, 144, 176, 145, 177, 146, 178, 147, 179, 148, 180, 149, 181, 150, 182, 151, 183, 152, 184, 153, 185, 154, 186, 155, 187, 156, 188, 157, 189, 158, 190, 159, 191, 192, 224, 193, 225, 194, 226, 195, 227, 196, 228, 197, 229, 198, 230, 199, 231, 200, 232, 201, 233, 202, 234, 203, 235, 204, 236, 205, 237, 206, 238, 207, 239, 208, 240, 209, 241, 210, 242, 211, 243, 212, 244, 213, 245, 214, 246, 215, 247, 216, 248, 217, 249, 218, 250, 219, 251, 220, 252, 221, 253, 222, 254, 223, 255, 256, 288, 257, 289, 258, 290, 259, 291, 260, 292, 261, 293, 262, 294, 263, 295, 264, 296, 265, 297, 266, 298, 267, 299, 268, 300, 269, 301, 270, 302, 271, 303, 272, 304, 273, 305, 274, 306, 275, 307, 276, 308, 277, 309, 278, 310, 279, 311, 280, 312, 281, 313, 282, 314, 283, 315, 284, 316, 285, 317, 286, 318, 287, 319, 320, 352, 321, 353, 322, 354, 323, 355, 324, 356, 325, 357, 326, 358, 327, 359, 328, 360, 329, 361, 330, 362, 331, 363, 332, 364, 333, 365, 334, 366, 335, 367, 336, 368, 337, 369, 338, 370, 339, 371, 340, 372, 341, 373, 342, 374, 343, 375, 344, 376, 345, 377, 346, 378, 347, 379, 348, 380, 349, 381, 350, 382, 351, 383, 384, 416, 385, 417, 386, 418, 387, 419, 388, 420, 389, 421, 390, 422, 391, 423, 392, 424, 393, 425, 394, 426, 395, 427, 396, 428, 397, 429, 398, 430, 399, 431, 400, 432, 401, 433, 402, 434, 403, 435, 404, 436, 405, 437, 406, 438, 407, 439, 408, 440, 409, 441, 410, 442, 411, 443, 412, 444, 413, 445, 414, 446, 415, 447, 448, 480, 449, 481, 450, 482, 451, 483, 452, 484, 453, 485, 454, 486, 455, 487, 456, 488, 457, 489, 458, 490, 459, 491, 460, 492, 461, 493, 462, 494, 463, 495, 464, 496, 465, 497, 466, 498, 467, 499, 468, 500, 469, 501, 470, 502, 471, 503, 472, 504, 473, 505, 474, 506, 475, 507, 476, 508, 477, 509, 478, 510, 479, 511] {packed} : vector<512xf16>, vector<512xf16>
        %38 = vector.shape_cast %37 {packed} : vector<512xf16> to vector<8x32x2xf16>
        %39 = vector.transpose %34, [1, 0] : vector<32x16xf16> to vector<16x32xf16>
        %40 = vector.shape_cast %39 {packed} : vector<16x32xf16> to vector<512xf16>
        %41 = vector.shuffle %40, %40 [0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47, 16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63, 64, 96, 65, 97, 66, 98, 67, 99, 68, 100, 69, 101, 70, 102, 71, 103, 72, 104, 73, 105, 74, 106, 75, 107, 76, 108, 77, 109, 78, 110, 79, 111, 80, 112, 81, 113, 82, 114, 83, 115, 84, 116, 85, 117, 86, 118, 87, 119, 88, 120, 89, 121, 90, 122, 91, 123, 92, 124, 93, 125, 94, 126, 95, 127, 128, 160, 129, 161, 130, 162, 131, 163, 132, 164, 133, 165, 134, 166, 135, 167, 136, 168, 137, 169, 138, 170, 139, 171, 140, 172, 141, 173, 142, 174, 143, 175, 144, 176, 145, 177, 146, 178, 147, 179, 148, 180, 149, 181, 150, 182, 151, 183, 152, 184, 153, 185, 154, 186, 155, 187, 156, 188, 157, 189, 158, 190, 159, 191, 192, 224, 193, 225, 194, 226, 195, 227, 196, 228, 197, 229, 198, 230, 199, 231, 200, 232, 201, 233, 202, 234, 203, 235, 204, 236, 205, 237, 206, 238, 207, 239, 208, 240, 209, 241, 210, 242, 211, 243, 212, 244, 213, 245, 214, 246, 215, 247, 216, 248, 217, 249, 218, 250, 219, 251, 220, 252, 221, 253, 222, 254, 223, 255, 256, 288, 257, 289, 258, 290, 259, 291, 260, 292, 261, 293, 262, 294, 263, 295, 264, 296, 265, 297, 266, 298, 267, 299, 268, 300, 269, 301, 270, 302, 271, 303, 272, 304, 273, 305, 274, 306, 275, 307, 276, 308, 277, 309, 278, 310, 279, 311, 280, 312, 281, 313, 282, 314, 283, 315, 284, 316, 285, 317, 286, 318, 287, 319, 320, 352, 321, 353, 322, 354, 323, 355, 324, 356, 325, 357, 326, 358, 327, 359, 328, 360, 329, 361, 330, 362, 331, 363, 332, 364, 333, 365, 334, 366, 335, 367, 336, 368, 337, 369, 338, 370, 339, 371, 340, 372, 341, 373, 342, 374, 343, 375, 344, 376, 345, 377, 346, 378, 347, 379, 348, 380, 349, 381, 350, 382, 351, 383, 384, 416, 385, 417, 386, 418, 387, 419, 388, 420, 389, 421, 390, 422, 391, 423, 392, 424, 393, 425, 394, 426, 395, 427, 396, 428, 397, 429, 398, 430, 399, 431, 400, 432, 401, 433, 402, 434, 403, 435, 404, 436, 405, 437, 406, 438, 407, 439, 408, 440, 409, 441, 410, 442, 411, 443, 412, 444, 413, 445, 414, 446, 415, 447, 448, 480, 449, 481, 450, 482, 451, 483, 452, 484, 453, 485, 454, 486, 455, 487, 456, 488, 457, 489, 458, 490, 459, 491, 460, 492, 461, 493, 462, 494, 463, 495, 464, 496, 465, 497, 466, 498, 467, 499, 468, 500, 469, 501, 470, 502, 471, 503, 472, 504, 473, 505, 474, 506, 475, 507, 476, 508, 477, 509, 478, 510, 479, 511] {packed} : vector<512xf16>, vector<512xf16>
        %42 = vector.shape_cast %41 {packed} : vector<512xf16> to vector<8x32x2xf16>
        %43 = vector.extract_strided_slice %38 {offsets = [0, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
        %44 = vector.extract_strided_slice %38 {offsets = [0, 16, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
        %45 = vector.extract_strided_slice %42 {offsets = [0, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
        %46 = vector.extract_strided_slice %42 {offsets = [0, 16, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
        %47 = xegpu.dpas %29, %43, %22 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %48 = xegpu.dpas %31, %45, %47 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %49 = xegpu.dpas %29, %44, %24 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %50 = xegpu.dpas %31, %46, %49 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %51 = xegpu.dpas %30, %43, %23 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %52 = xegpu.dpas %32, %45, %51 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %53 = xegpu.dpas %30, %44, %25 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %54 = xegpu.dpas %32, %46, %53 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %55 = vector.shuffle %48, %52 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
        %56 = vector.shuffle %50, %54 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
        %57 = xegpu.update_nd_offset %arg4, [%c0, %c32] : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>
        %58 = xegpu.update_nd_offset %arg5, [%c0, %c32] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
        %59 = xegpu.update_nd_offset %arg6, [%c0, %c32] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
        scf.yield %57, %58, %59, %55, %56 : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>, vector<16x16xf32>, vector<16x16xf32>
      }
      %18 = vector.extract_strided_slice %17#3 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf32> to vector<8x16xf32>
      %19 = vector.extract_strided_slice %17#3 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf32> to vector<8x16xf32>
      %20 = vector.extract_strided_slice %17#4 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf32> to vector<8x16xf32>
      %21 = vector.extract_strided_slice %17#4 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf32> to vector<8x16xf32>
      xegpu.store_nd %18, %4 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      xegpu.store_nd %20, %6 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      xegpu.store_nd %19, %8 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      xegpu.store_nd %21, %9 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 1.000000e+00 : f16
    %alloc = memref.alloc() : memref<256x256xf16>
    %alloc_1 = memref.alloc() : memref<256x256xf16>
    %alloc_2 = memref.alloc() : memref<256x256xf32>
    %alloc_3 = memref.alloc() : memref<256x256xf32>
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c256 step %c1 {
        %1 = index.castu %arg1 : index to i16
        %2 = arith.uitofp %1 : i16 to f16
        memref.store %2, %alloc_1[%arg0, %arg1] : memref<256x256xf16>
      }
    }
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c256 step %c1 {
        %1 = index.castu %arg0 : index to i32
        %2 = index.castu %arg1 : index to i32
        %3 = arith.cmpi eq, %1, %2 : i32
        scf.if %3 {
          memref.store %cst_0, %alloc[%arg0, %arg1] : memref<256x256xf16>
        } else {
          memref.store %cst, %alloc[%arg0, %arg1] : memref<256x256xf16>
        }
      }
    }
    %cst_4 = arith.constant 0.000000e+00 : f32
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c256 step %c1 {
        memref.store %cst_4, %alloc_2[%arg0, %arg1] : memref<256x256xf32>
        memref.store %cst_4, %alloc_3[%arg0, %arg1] : memref<256x256xf32>
      }
    }
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c256 step %c1 {
        %1 = memref.load %alloc_3[%arg0, %arg1] : memref<256x256xf32>
        %2 = scf.for %arg2 = %c0 to %c256 step %c1 iter_args(%arg3 = %1) -> (f32) {
          %3 = memref.load %alloc[%arg0, %arg2] : memref<256x256xf16>
          %4 = memref.load %alloc_1[%arg1, %arg2] : memref<256x256xf16>
          %5 = arith.mulf %3, %4 : f16
          %6 = arith.extf %5 : f16 to f32
          %7 = arith.addf %6, %arg3 : f32
          scf.yield %7 : f32
        }
        memref.store %2, %alloc_3[%arg0, %arg1] : memref<256x256xf32>
      }
    }
    %0 = call @test(%alloc, %alloc_1, %alloc_2) : (memref<256x256xf16>, memref<256x256xf16>, memref<256x256xf32>) -> memref<256x256xf32>
    %cast = memref.cast %0 : memref<256x256xf32> to memref<*xf32>
    %cast_5 = memref.cast %alloc_3 : memref<256x256xf32> to memref<*xf32>
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast, %cast_5) : (memref<*xf32>, memref<*xf32>) -> ()
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<256x256xf16>
    memref.dealloc %alloc_1 : memref<256x256xf16>
    memref.dealloc %alloc_2 : memref<256x256xf32>
    memref.dealloc %alloc_3 : memref<256x256xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
