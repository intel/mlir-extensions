// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<256x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<256x256xf32>) -> memref<256x256xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %memref = gpu.alloc  () : memref<256x256xf16>
    gpu.memcpy  %memref, %arg0 : memref<256x256xf16>, memref<256x256xf16>
    %memref_0 = gpu.alloc  () : memref<256x256xf16>
    gpu.memcpy  %memref_0, %arg1 : memref<256x256xf16>, memref<256x256xf16>
    %memref_1 = gpu.alloc  () : memref<256x256xf32>
    gpu.memcpy  %memref_1, %arg2 : memref<256x256xf32>, memref<256x256xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c64, %c32, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<256x256xf16>, %memref_0 : memref<256x256xf16>, %memref_1 : memref<256x256xf32>)
    gpu.dealloc  %memref : memref<256x256xf16>
    gpu.dealloc  %memref_0 : memref<256x256xf16>
    %alloc = memref.alloc() : memref<256x256xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<256x256xf32>, memref<256x256xf32>
    gpu.dealloc  %memref_1 : memref<256x256xf32>
    return %alloc : memref<256x256xf32>
  }
  gpu.module @test_kernel  {
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
      %4 = xegpu.create_nd_tdesc %arg2[%2, %3] : memref<256x256xf32> -> !xegpu.tensor_desc<8x16xf32>
      %5 = arith.addi %1, %c16 : index
      %6 = xegpu.create_nd_tdesc %arg2[%2, %5] : memref<256x256xf32> -> !xegpu.tensor_desc<8x16xf32>
      %c8 = arith.constant 8 : index
      %7 = arith.addi %0, %c8 : index
      %8 = xegpu.create_nd_tdesc %arg2[%7, %3] : memref<256x256xf32> -> !xegpu.tensor_desc<8x16xf32>
      %9 = xegpu.create_nd_tdesc %arg2[%7, %5] : memref<256x256xf32> -> !xegpu.tensor_desc<8x16xf32>
      %10 = xegpu.create_nd_tdesc %arg2[%2, %3] : memref<256x256xf32> -> !xegpu.tensor_desc<16x16xf32>
      %11 = xegpu.create_nd_tdesc %arg2[%2, %5] : memref<256x256xf32> -> !xegpu.tensor_desc<16x16xf32>
      %12 = xegpu.load_nd %10 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x16xf32> -> vector<16x16xf32>
      %13 = xegpu.load_nd %11 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x16xf32> -> vector<16x16xf32>
      %14 = vector.extract_strided_slice %12 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf32> to vector<8x16xf32>
      %15 = vector.extract_strided_slice %12 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf32> to vector<8x16xf32>
      %16 = vector.extract_strided_slice %13 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf32> to vector<8x16xf32>
      %17 = vector.extract_strided_slice %13 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf32> to vector<8x16xf32>
      %18 = xegpu.create_nd_tdesc %arg0[%2, %c0] : memref<256x256xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
      %19 = xegpu.create_nd_tdesc %arg1[%3, %c0] : memref<256x256xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
      %20:6 = scf.for %arg3 = %c0 to %c256 step %c32 iter_args(%arg4 = %18, %arg5 = %19, %arg6 = %14, %arg7 = %16, %arg8 = %15, %arg9 = %17) -> (!xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>) {
        %21 = xegpu.load_nd %arg4 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x16x16xf16>
        %22 = vector.extract %21[0] : vector<16x16xf16> from vector<2x16x16xf16>
        %23 = vector.extract %21[1] : vector<16x16xf16> from vector<2x16x16xf16>
        %24 = vector.extract_strided_slice %22 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %25 = vector.extract_strided_slice %22 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %26 = vector.extract_strided_slice %23 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %27 = vector.extract_strided_slice %23 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
        %28 = xegpu.load_nd %arg5 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x32x16xf16>
        %29 = vector.extract %28[0] : vector<32x16xf16> from vector<2x32x16xf16>
        %30 = vector.extract %28[1] : vector<32x16xf16> from vector<2x32x16xf16>
        %31 = vector.extract_strided_slice %29 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
        %32 = vector.extract_strided_slice %29 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
        %33 = vector.extract_strided_slice %30 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
        %34 = vector.extract_strided_slice %30 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
        %35 = vector.transpose %31, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
        %36 = vector.transpose %32, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
        %37 = vector.transpose %33, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
        %38 = vector.transpose %34, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
        %39 = xegpu.dpas %24, %35, %arg6 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %40 = xegpu.dpas %26, %37, %39 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %41 = xegpu.dpas %24, %36, %arg7 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %42 = xegpu.dpas %26, %38, %41 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %43 = xegpu.dpas %25, %35, %arg8 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %44 = xegpu.dpas %27, %37, %43 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %45 = xegpu.dpas %25, %36, %arg9 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %46 = xegpu.dpas %27, %38, %45 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %47 = xegpu.update_nd_offset %arg4, [%c0, %c32] : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
        %48 = xegpu.update_nd_offset %arg5, [%c0, %c32] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
        scf.yield %47, %48, %40, %42, %44, %46 : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>
      }
      xegpu.store_nd %20#2, %4 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %20#3, %6 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %20#4, %8 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %20#5, %9 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %cst_0 = arith.constant 0.000000e+00 : f16
    %cst_1 = arith.constant 1.000000e+00 : f16
    %alloc = memref.alloc() : memref<256x256xf16>
    %alloc_2 = memref.alloc() : memref<256x256xf16>
    %alloc_3 = memref.alloc() : memref<256x256xf32>
    %alloc_4 = memref.alloc() : memref<256x256xf32>
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c256 step %c1 {
        %1 = index.castu %arg1 : index to i16
        %2 = arith.uitofp %1 : i16 to f16
        memref.store %2, %alloc_2[%arg0, %arg1] : memref<256x256xf16>
      }
    }
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c256 step %c1 {
        %1 = index.castu %arg0 : index to i32
        %2 = index.castu %arg1 : index to i32
        %3 = arith.cmpi eq, %1, %2 : i32
        scf.if %3 {
          memref.store %cst_1, %alloc[%arg0, %arg1] : memref<256x256xf16>
        } else {
          memref.store %cst_0, %alloc[%arg0, %arg1] : memref<256x256xf16>
        }
      }
    }
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c256 step %c1 {
        memref.store %cst, %alloc_3[%arg0, %arg1] : memref<256x256xf32>
        memref.store %cst, %alloc_4[%arg0, %arg1] : memref<256x256xf32>
      }
    }
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c256 step %c1 {
        %1 = memref.load %alloc_4[%arg0, %arg1] : memref<256x256xf32>
        %2 = scf.for %arg2 = %c0 to %c256 step %c1 iter_args(%arg3 = %1) -> (f32) {
          %3 = memref.load %alloc[%arg0, %arg2] : memref<256x256xf16>
          %4 = memref.load %alloc_2[%arg1, %arg2] : memref<256x256xf16>
          %5 = arith.mulf %3, %4 : f16
          %6 = arith.extf %5 : f16 to f32
          %7 = arith.addf %6, %arg3 : f32
          scf.yield %7 : f32
        }
        memref.store %2, %alloc_4[%arg0, %arg1] : memref<256x256xf32>
      }
    }
    %0 = call @test(%alloc, %alloc_2, %alloc_3) : (memref<256x256xf16>, memref<256x256xf16>, memref<256x256xf32>) -> memref<256x256xf32>
    %cast = memref.cast %0 : memref<256x256xf32> to memref<*xf32>
    // CHECK: [ALLCLOSE: TRUE]
    %cast_5 = memref.cast %alloc_4 : memref<256x256xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_5) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<256x256xf16>
    memref.dealloc %alloc_2 : memref<256x256xf16>
    memref.dealloc %alloc_3 : memref<256x256xf32>
    memref.dealloc %alloc_4 : memref<256x256xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
