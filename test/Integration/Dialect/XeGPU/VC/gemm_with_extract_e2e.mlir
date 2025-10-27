// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  memref.global "private" @__constant_1024x1024xf16 : memref<1024x1024xf16> = dense<0.000000e+00>
  memref.global "private" @__constant_1024x1024xf16_ : memref<1024x1024xf16> = dense<0.000000e+00>
  memref.global "private" @__constant_1024x1024xf32 : memref<1024x1024xf32> = dense<0.000000e+00>
  func.func @test(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>) -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %memref = gpu.alloc  () : memref<1024x1024xf16>
    gpu.memcpy  %memref, %arg0 : memref<1024x1024xf16>, memref<1024x1024xf16>
    %memref_0 = gpu.alloc  () : memref<1024x1024xf16>
    gpu.memcpy  %memref_0, %arg1 : memref<1024x1024xf16>, memref<1024x1024xf16>
    %memref_1 = gpu.alloc  () : memref<1024x1024xf32>
    gpu.launch_func  @test_kernel::@test_gemm blocks in (%c32, %c32, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<1024x1024xf16>, %memref_0 : memref<1024x1024xf16>, %memref_1 : memref<1024x1024xf32>)
    gpu.dealloc  %memref : memref<1024x1024xf16>
    gpu.dealloc  %memref_0 : memref<1024x1024xf16>
    %alloc = memref.alloc() : memref<1024x1024xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<1024x1024xf32>, memref<1024x1024xf32>
    gpu.dealloc  %memref_1 : memref<1024x1024xf32>
    return %alloc : memref<1024x1024xf32>
  }
  gpu.module @test_kernel  {
    gpu.func @test_gemm(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 128, 64, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c32 : index
      %1 = arith.muli %block_id_y, %c32 : index
      %2 = arith.addi %0, %c0 : index
      %3 = arith.addi %1, %c0 : index
      %4 = xegpu.create_nd_tdesc %arg2[%2, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
      %c16 = arith.constant 16 : index
      %5 = arith.addi %1, %c16 : index
      %6 = xegpu.create_nd_tdesc %arg2[%2, %5] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
      %c8 = arith.constant 8 : index
      %7 = arith.addi %0, %c8 : index
      %8 = xegpu.create_nd_tdesc %arg2[%7, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
      %9 = xegpu.create_nd_tdesc %arg2[%7, %5] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
      %10 = arith.addi %0, %c16 : index
      %11 = xegpu.create_nd_tdesc %arg2[%10, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
      %12 = xegpu.create_nd_tdesc %arg2[%10, %5] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
      %c24 = arith.constant 24 : index
      %13 = arith.addi %0, %c24 : index
      %14 = xegpu.create_nd_tdesc %arg2[%13, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
      %15 = xegpu.create_nd_tdesc %arg2[%13, %5] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
      %16 = xegpu.create_nd_tdesc %arg2[%2, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32>
      %17 = xegpu.create_nd_tdesc %arg2[%2, %5] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32>
      %18 = xegpu.load_nd %16 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32> -> vector<32x16xf32>
      %19 = xegpu.load_nd %17 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32> -> vector<32x16xf32>
      %20 = xegpu.create_nd_tdesc %arg0[%2, %c0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
      %21 = xegpu.create_nd_tdesc %arg1[%c0, %3] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
      %22:4 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %20, %arg5 = %21, %arg6 = %18, %arg7 = %19) -> (!xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, vector<32x16xf32>, vector<32x16xf32>) {
        %31 = vector.extract_strided_slice %arg6 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
        %32 = vector.extract_strided_slice %arg6 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
        %33 = vector.extract_strided_slice %arg6 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
        %34 = vector.extract_strided_slice %arg6 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
        %35 = vector.extract_strided_slice %arg7 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
        %36 = vector.extract_strided_slice %arg7 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
        %37 = vector.extract_strided_slice %arg7 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
        %38 = vector.extract_strided_slice %arg7 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
        %39 = xegpu.load_nd %arg4 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x32x16xf16>
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
        %50 = xegpu.load_nd %arg5 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x32x16xf16>
        %51 = vector.extract %50[0] : vector<32x16xf16> from vector<2x32x16xf16>
        %52 = vector.extract %50[1] : vector<32x16xf16> from vector<2x32x16xf16>
        %53 = vector.extract_strided_slice %51 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
        %54 = vector.extract_strided_slice %51 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
        %55 = vector.extract_strided_slice %52 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
        %56 = vector.extract_strided_slice %52 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
        %57 = xegpu.dpas %42, %53, %31 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %58 = xegpu.dpas %46, %54, %57 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %59 = xegpu.dpas %42, %55, %35 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %60 = xegpu.dpas %46, %56, %59 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %61 = xegpu.dpas %43, %53, %32 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %62 = xegpu.dpas %47, %54, %61 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %63 = xegpu.dpas %43, %55, %36 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %64 = xegpu.dpas %47, %56, %63 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %65 = xegpu.dpas %44, %53, %33 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %66 = xegpu.dpas %48, %54, %65 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %67 = xegpu.dpas %44, %55, %37 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %68 = xegpu.dpas %48, %56, %67 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %69 = xegpu.dpas %45, %53, %34 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %70 = xegpu.dpas %49, %54, %69 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %71 = xegpu.dpas %45, %55, %38 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %72 = xegpu.dpas %49, %56, %71 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %73 = vector.shuffle %58, %62 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
        %74 = vector.shuffle %66, %70 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
        %75 = vector.shuffle %73, %74 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16x16xf32>, vector<16x16xf32>
        %76 = vector.shuffle %60, %64 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
        %77 = vector.shuffle %68, %72 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
        %78 = vector.shuffle %76, %77 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16x16xf32>, vector<16x16xf32>
        %79 = xegpu.update_nd_offset %arg4, [%c0, %c32] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
        %80 = xegpu.update_nd_offset %arg5, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
        scf.yield %79, %80, %75, %78 : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, vector<32x16xf32>, vector<32x16xf32>
      }
      %23 = vector.extract_strided_slice %22#2 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %24 = vector.extract_strided_slice %22#2 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %25 = vector.extract_strided_slice %22#2 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %26 = vector.extract_strided_slice %22#2 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %27 = vector.extract_strided_slice %22#3 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %28 = vector.extract_strided_slice %22#3 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %29 = vector.extract_strided_slice %22#3 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %30 = vector.extract_strided_slice %22#3 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      xegpu.store_nd %23, %4 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %27, %6 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %24, %8 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %28, %9 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %25, %11 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %29, %12 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %26, %14 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %30, %15 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 1.000000e+00 : f16
    %cst_0 = arith.constant 1.000000e+02 : f16
    %c128_i16 = arith.constant 128 : i16
    %c1024 = arith.constant 1024 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_1024x1024xf16 : memref<1024x1024xf16>
    %1 = memref.get_global @__constant_1024x1024xf16_ : memref<1024x1024xf16>
    // fill the top-left block 128x128
    // A matrix: row-major, start from 0.0, increase 0.01 per element
    // B matrix: A matrix + 1.0
    %2 = memref.get_global @__constant_1024x1024xf32 : memref<1024x1024xf32>
    scf.for %arg0 = %c0 to %c128 step %c1 {
      scf.for %arg1 = %c0 to %c128 step %c1 {
        %4 = arith.index_cast %arg0 : index to i16
        %5 = arith.index_cast %arg1 : index to i16
        %6 = arith.muli %4, %c128_i16 : i16
        %7 = arith.addi %5, %6 : i16
        %8 = arith.uitofp %7 : i16 to f16
        %9 = arith.divf %8, %cst_0 : f16
        %10 = arith.addf %9, %cst : f16
        memref.store %9, %0[%arg0, %arg1] : memref<1024x1024xf16>
        memref.store %10, %1[%arg0, %arg1] : memref<1024x1024xf16>
      }
    }
    // caculate the result C matrix
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %4 = memref.load %2[%arg0, %arg1] : memref<1024x1024xf32>
        %5 = scf.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %4) -> (f32) {
          %6 = memref.load %0[%arg0, %arg2] : memref<1024x1024xf16>
          %7 = memref.load %1[%arg2, %arg1] : memref<1024x1024xf16>
          %8 = arith.mulf %6, %7 : f16
          %9 = arith.extf %8 : f16 to f32
          %10 = arith.addf %9, %arg3 : f32
          scf.yield %10 : f32
        }
        memref.store %5, %2[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    //call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    //call @printMemrefF32(%cast_ref) : (memref<*xf32>) -> ()
    // CHECK:   [ALLCLOSE: TRUE]
    %3 = call @test(%0, %1) : (memref<1024x1024xf16>, memref<1024x1024xf16>) -> memref<1024x1024xf32>
    %cast = memref.cast %3 : memref<1024x1024xf32> to memref<*xf32>
    %cast_1 = memref.cast %2 : memref<1024x1024xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_1) : (memref<*xf32>, memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

