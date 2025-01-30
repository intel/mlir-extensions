// -----// IR Dump After CSE (cse) //----- //
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %memref = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %arg0, %memref : memref<1024x1024xf16> to memref<1024x1024xf16>
    %memref_0 = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %arg1, %memref_0 : memref<1024x1024xf16> to memref<1024x1024xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<1024x1024xf32>
    memref.copy %arg2, %memref_1 : memref<1024x1024xf32> to memref<1024x1024xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c32, %c32, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<1024x1024xf16>, %memref_0 : memref<1024x1024xf16>, %memref_1 : memref<1024x1024xf32>)
    gpu.dealloc  %memref : memref<1024x1024xf16>
    gpu.dealloc  %memref_0 : memref<1024x1024xf16>
    return %memref_1 : memref<1024x1024xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c32 : index
      %1 = arith.muli %block_id_y, %c32 : index
      %2 = xetile.init_tile %arg2[%0, %1] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
      %3 = xetile.load_tile %2 : !xetile.tile<32x32xf32> -> vector<32x32xf32>
      %4 = xetile.init_tile %arg0[%c0, %0] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
      %5 = xetile.init_tile %arg1[%c0, %1] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
      %6:3 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %4, %arg5 = %5, %arg6 = %3) -> (!xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>) {
        %7 = xetile.load_tile %arg4 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
        %8 = xetile.load_tile %arg5 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
        %9 = vector.transpose %7, [1, 0] : vector<32x32xf16> to vector<32x32xf16>
        %10 = xetile.tile_mma %9, %8, %arg6 : vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
        %11 = xetile.update_tile_offset %arg4, [%c32, %c0] : !xetile.tile<32x32xf16>
        %12 = xetile.update_tile_offset %arg5, [%c32, %c0] : !xetile.tile<32x32xf16>
        scf.yield %11, %12, %10 : !xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>
      }
      xetile.store_tile %6#2,  %2 : vector<32x32xf32>, !xetile.tile<32x32xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 1.000000e+00 : f16
    %alloc = memref.alloc() : memref<1024x1024xf16>
    %alloc_1 = memref.alloc() : memref<1024x1024xf16>
    %alloc_2 = memref.alloc() : memref<1024x1024xf32>
    %alloc_3 = memref.alloc() : memref<1024x1024xf32>
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = index.castu %arg1 : index to i16
        %2 = arith.uitofp %1 : i16 to f16
        memref.store %2, %alloc[%arg0, %arg1] : memref<1024x1024xf16>
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = index.castu %arg0 : index to i32
        %2 = index.castu %arg1 : index to i32
        %3 = arith.cmpi eq, %1, %2 : i32
        scf.if %3 {
          memref.store %cst_0, %alloc_1[%arg0, %arg1] : memref<1024x1024xf16>
        } else {
          memref.store %cst, %alloc_1[%arg0, %arg1] : memref<1024x1024xf16>
        }
      }
    }
    %cst_4 = arith.constant 0.000000e+00 : f32
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        memref.store %cst_4, %alloc_2[%arg0, %arg1] : memref<1024x1024xf32>
        memref.store %cst_4, %alloc_3[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = memref.load %alloc_3[%arg0, %arg1] : memref<1024x1024xf32>
        %2 = scf.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %1) -> (f32) {
          %3 = memref.load %alloc[%arg2, %arg0] : memref<1024x1024xf16>
          %4 = memref.load %alloc_1[%arg2, %arg1] : memref<1024x1024xf16>
          %5 = arith.mulf %3, %4 : f16
          %6 = arith.extf %5 : f16 to f32
          %7 = arith.addf %6, %arg3 : f32
          scf.yield %7 : f32
        }
        memref.store %2, %alloc_3[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    %0 = call @test(%alloc, %alloc_1, %alloc_2) : (memref<1024x1024xf16>, memref<1024x1024xf16>, memref<1024x1024xf32>) -> memref<1024x1024xf32>
    %cast = memref.cast %0 : memref<1024x1024xf32> to memref<*xf32>
    %cast_5 = memref.cast %alloc_3 : memref<1024x1024xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_5) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<1024x1024xf16>
    memref.dealloc %alloc_1 : memref<1024x1024xf16>
    memref.dealloc %alloc_2 : memref<1024x1024xf32>
    memref.dealloc %alloc_3 : memref<1024x1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}


// -----// IR Dump After XeTileInitDuplicate (xetile-init-duplicate) //----- //
gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %0 = arith.muli %block_id_x, %c32 : index
    %1 = arith.muli %block_id_y, %c32 : index
    %2 = xetile.init_tile %arg2[%0, %1] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
    %3 = xetile.init_tile %arg2[%0, %1] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
    %4 = xetile.load_tile %3 : !xetile.tile<32x32xf32> -> vector<32x32xf32>
    %5 = xetile.init_tile %arg0[%c0, %0] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    %6 = xetile.init_tile %arg1[%c0, %1] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    %7:3 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %5, %arg5 = %6, %arg6 = %4) -> (!xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>) {
      %8 = xetile.load_tile %arg4 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
      %9 = xetile.load_tile %arg5 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
      %10 = vector.transpose %8, [1, 0] : vector<32x32xf16> to vector<32x32xf16>
      %11 = xetile.tile_mma %10, %9, %arg6 : vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
      %12 = xetile.update_tile_offset %arg4, [%c32, %c0] : !xetile.tile<32x32xf16>
      %13 = xetile.update_tile_offset %arg5, [%c32, %c0] : !xetile.tile<32x32xf16>
      scf.yield %12, %13, %11 : !xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>
    }
    xetile.store_tile %7#2,  %2 : vector<32x32xf32>, !xetile.tile<32x32xf32>
    gpu.return
  }
}

// -----// IR Dump After XeTileCanonicalization (xetile-canonicalization) //----- //
gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %0 = arith.muli %block_id_x, %c32 : index
    %1 = arith.muli %block_id_y, %c32 : index
    %2 = xetile.init_tile %arg2[%0, %1] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
    %3 = xetile.init_tile %arg2[%0, %1] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
    %4 = xetile.load_tile %3 : !xetile.tile<32x32xf32> -> vector<32x32xf32>
    %5 = xetile.init_tile %arg0[%c0, %0] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    %6 = xetile.init_tile %arg1[%c0, %1] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    %7:3 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %5, %arg5 = %6, %arg6 = %4) -> (!xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>) {
      %8 = xetile.load_tile %arg4 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
      %9 = xetile.load_tile %arg5 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
      %10 = xetile.transpose %8, [1, 0] : vector<32x32xf16> -> vector<32x32xf16>
      %11 = xetile.tile_mma %10, %9, %arg6 : vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
      %12 = xetile.update_tile_offset %arg4, [%c32, %c0] : !xetile.tile<32x32xf16>
      %13 = xetile.update_tile_offset %arg5, [%c32, %c0] : !xetile.tile<32x32xf16>
      scf.yield %12, %13, %11 : !xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>
    }
    xetile.store_tile %7#2,  %2 : vector<32x32xf32>, !xetile.tile<32x32xf32>
    gpu.return
  }
}

// -----// IR Dump After XeTileBlocking (xetile-blocking) //----- //
gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %c24 = arith.constant 24 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %0 = arith.muli %block_id_x, %c32 : index
    %1 = arith.muli %block_id_y, %c32 : index
    %2 = xetile.init_tile %arg2[%0, %1] : memref<1024x1024xf32> -> !xetile.tile<8x16xf32>
    %3 = arith.addi %1, %c16 : index
    %4 = xetile.init_tile %arg2[%0, %3] : memref<1024x1024xf32> -> !xetile.tile<8x16xf32>
    %5 = arith.addi %0, %c8 : index
    %6 = xetile.init_tile %arg2[%5, %1] : memref<1024x1024xf32> -> !xetile.tile<8x16xf32>
    %7 = xetile.init_tile %arg2[%5, %3] : memref<1024x1024xf32> -> !xetile.tile<8x16xf32>
    %8 = arith.addi %0, %c16 : index
    %9 = xetile.init_tile %arg2[%8, %1] : memref<1024x1024xf32> -> !xetile.tile<8x16xf32>
    %10 = xetile.init_tile %arg2[%8, %3] : memref<1024x1024xf32> -> !xetile.tile<8x16xf32>
    %11 = arith.addi %0, %c24 : index
    %12 = xetile.init_tile %arg2[%11, %1] : memref<1024x1024xf32> -> !xetile.tile<8x16xf32>
    %13 = xetile.init_tile %arg2[%11, %3] : memref<1024x1024xf32> -> !xetile.tile<8x16xf32>
    %14 = xetile.init_tile %arg2[%0, %1] : memref<1024x1024xf32> -> !xetile.tile<32x16xf32>
    %15 = xetile.init_tile %arg2[%0, %3] : memref<1024x1024xf32> -> !xetile.tile<32x16xf32>
    %16 = xetile.load_tile %14 : !xetile.tile<32x16xf32> -> vector<32x16xf32>
    %17 = xetile.load_tile %15 : !xetile.tile<32x16xf32> -> vector<32x16xf32>
    %18 = vector.extract_strided_slice %16 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %19 = vector.extract_strided_slice %16 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %20 = vector.extract_strided_slice %16 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %21 = vector.extract_strided_slice %16 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %22 = vector.extract_strided_slice %17 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %23 = vector.extract_strided_slice %17 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %24 = vector.extract_strided_slice %17 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %25 = vector.extract_strided_slice %17 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %26 = xetile.init_tile %arg0[%c0, %0] : memref<1024x1024xf16> -> !xetile.tile<16x8xf16>
    %27 = xetile.init_tile %arg0[%c0, %5] : memref<1024x1024xf16> -> !xetile.tile<16x8xf16>
    %28 = xetile.init_tile %arg0[%c0, %8] : memref<1024x1024xf16> -> !xetile.tile<16x8xf16>
    %29 = xetile.init_tile %arg0[%c0, %11] : memref<1024x1024xf16> -> !xetile.tile<16x8xf16>
    %30 = xetile.init_tile %arg0[%c16, %0] : memref<1024x1024xf16> -> !xetile.tile<16x8xf16>
    %31 = xetile.init_tile %arg0[%c16, %5] : memref<1024x1024xf16> -> !xetile.tile<16x8xf16>
    %32 = xetile.init_tile %arg0[%c16, %8] : memref<1024x1024xf16> -> !xetile.tile<16x8xf16>
    %33 = xetile.init_tile %arg0[%c16, %11] : memref<1024x1024xf16> -> !xetile.tile<16x8xf16>
    %34 = xetile.init_tile %arg1[%c0, %1] : memref<1024x1024xf16> -> !xetile.tile<32x16xf16>
    %35 = xetile.init_tile %arg1[%c0, %3] : memref<1024x1024xf16> -> !xetile.tile<32x16xf16>
    %36:18 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %26, %arg5 = %27, %arg6 = %28, %arg7 = %29, %arg8 = %30, %arg9 = %31, %arg10 = %32, %arg11 = %33, %arg12 = %34, %arg13 = %35, %arg14 = %18, %arg15 = %22, %arg16 = %19, %arg17 = %23, %arg18 = %20, %arg19 = %24, %arg20 = %21, %arg21 = %25) -> (!xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<32x16xf16>, !xetile.tile<32x16xf16>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>) {
      %37 = xetile.load_tile %arg4 : !xetile.tile<16x8xf16> -> vector<16x8xf16>
      %38 = xetile.load_tile %arg5 : !xetile.tile<16x8xf16> -> vector<16x8xf16>
      %39 = xetile.load_tile %arg6 : !xetile.tile<16x8xf16> -> vector<16x8xf16>
      %40 = xetile.load_tile %arg7 : !xetile.tile<16x8xf16> -> vector<16x8xf16>
      %41 = xetile.load_tile %arg8 : !xetile.tile<16x8xf16> -> vector<16x8xf16>
      %42 = xetile.load_tile %arg9 : !xetile.tile<16x8xf16> -> vector<16x8xf16>
      %43 = xetile.load_tile %arg10 : !xetile.tile<16x8xf16> -> vector<16x8xf16>
      %44 = xetile.load_tile %arg11 : !xetile.tile<16x8xf16> -> vector<16x8xf16>
      %45 = xetile.load_tile %arg12 : !xetile.tile<32x16xf16> -> vector<32x16xf16>
      %46 = xetile.load_tile %arg13 : !xetile.tile<32x16xf16> -> vector<32x16xf16>
      %47 = vector.extract_strided_slice %45 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %48 = vector.extract_strided_slice %45 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %49 = vector.extract_strided_slice %46 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %50 = vector.extract_strided_slice %46 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %51 = xetile.transpose %37, [1, 0] : vector<16x8xf16> -> vector<8x16xf16>
      %52 = xetile.transpose %41, [1, 0] : vector<16x8xf16> -> vector<8x16xf16>
      %53 = xetile.transpose %38, [1, 0] : vector<16x8xf16> -> vector<8x16xf16>
      %54 = xetile.transpose %42, [1, 0] : vector<16x8xf16> -> vector<8x16xf16>
      %55 = xetile.transpose %39, [1, 0] : vector<16x8xf16> -> vector<8x16xf16>
      %56 = xetile.transpose %43, [1, 0] : vector<16x8xf16> -> vector<8x16xf16>
      %57 = xetile.transpose %40, [1, 0] : vector<16x8xf16> -> vector<8x16xf16>
      %58 = xetile.transpose %44, [1, 0] : vector<16x8xf16> -> vector<8x16xf16>
      %59 = xetile.tile_mma %51, %47, %arg14 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %60 = xetile.tile_mma %52, %48, %59 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %61 = xetile.tile_mma %51, %49, %arg15 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %62 = xetile.tile_mma %52, %50, %61 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %63 = xetile.tile_mma %53, %47, %arg16 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %64 = xetile.tile_mma %54, %48, %63 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %65 = xetile.tile_mma %53, %49, %arg17 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %66 = xetile.tile_mma %54, %50, %65 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %67 = xetile.tile_mma %55, %47, %arg18 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %68 = xetile.tile_mma %56, %48, %67 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %69 = xetile.tile_mma %55, %49, %arg19 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %70 = xetile.tile_mma %56, %50, %69 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %71 = xetile.tile_mma %57, %47, %arg20 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %72 = xetile.tile_mma %58, %48, %71 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %73 = xetile.tile_mma %57, %49, %arg21 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %74 = xetile.tile_mma %58, %50, %73 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %75 = xetile.update_tile_offset %arg4, [%c32, %c0] : !xetile.tile<16x8xf16>
      %76 = xetile.update_tile_offset %arg5, [%c32, %c0] : !xetile.tile<16x8xf16>
      %77 = xetile.update_tile_offset %arg6, [%c32, %c0] : !xetile.tile<16x8xf16>
      %78 = xetile.update_tile_offset %arg7, [%c32, %c0] : !xetile.tile<16x8xf16>
      %79 = xetile.update_tile_offset %arg8, [%c32, %c0] : !xetile.tile<16x8xf16>
      %80 = xetile.update_tile_offset %arg9, [%c32, %c0] : !xetile.tile<16x8xf16>
      %81 = xetile.update_tile_offset %arg10, [%c32, %c0] : !xetile.tile<16x8xf16>
      %82 = xetile.update_tile_offset %arg11, [%c32, %c0] : !xetile.tile<16x8xf16>
      %83 = xetile.update_tile_offset %arg12, [%c32, %c0] : !xetile.tile<32x16xf16>
      %84 = xetile.update_tile_offset %arg13, [%c32, %c0] : !xetile.tile<32x16xf16>
      scf.yield %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %60, %62, %64, %66, %68, %70, %72, %74 : !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<32x16xf16>, !xetile.tile<32x16xf16>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>
    }
    xetile.store_tile %36#10,  %2 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    xetile.store_tile %36#11,  %4 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    xetile.store_tile %36#12,  %6 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    xetile.store_tile %36#13,  %7 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    xetile.store_tile %36#14,  %9 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    xetile.store_tile %36#15,  %10 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    xetile.store_tile %36#16,  %12 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    xetile.store_tile %36#17,  %13 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    gpu.return
  }
}

// -----// IR Dump After CSE (cse) //----- //
gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %c24 = arith.constant 24 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %0 = arith.muli %block_id_x, %c32 : index
    %1 = arith.muli %block_id_y, %c32 : index
    %2 = xetile.init_tile %arg2[%0, %1] : memref<1024x1024xf32> -> !xetile.tile<8x16xf32>
    %3 = arith.addi %1, %c16 : index
    %4 = xetile.init_tile %arg2[%0, %3] : memref<1024x1024xf32> -> !xetile.tile<8x16xf32>
    %5 = arith.addi %0, %c8 : index
    %6 = xetile.init_tile %arg2[%5, %1] : memref<1024x1024xf32> -> !xetile.tile<8x16xf32>
    %7 = xetile.init_tile %arg2[%5, %3] : memref<1024x1024xf32> -> !xetile.tile<8x16xf32>
    %8 = arith.addi %0, %c16 : index
    %9 = xetile.init_tile %arg2[%8, %1] : memref<1024x1024xf32> -> !xetile.tile<8x16xf32>
    %10 = xetile.init_tile %arg2[%8, %3] : memref<1024x1024xf32> -> !xetile.tile<8x16xf32>
    %11 = arith.addi %0, %c24 : index
    %12 = xetile.init_tile %arg2[%11, %1] : memref<1024x1024xf32> -> !xetile.tile<8x16xf32>
    %13 = xetile.init_tile %arg2[%11, %3] : memref<1024x1024xf32> -> !xetile.tile<8x16xf32>
    %14 = xetile.init_tile %arg2[%0, %1] : memref<1024x1024xf32> -> !xetile.tile<32x16xf32>
    %15 = xetile.init_tile %arg2[%0, %3] : memref<1024x1024xf32> -> !xetile.tile<32x16xf32>
    %16 = xetile.load_tile %14 : !xetile.tile<32x16xf32> -> vector<32x16xf32>
    %17 = xetile.load_tile %15 : !xetile.tile<32x16xf32> -> vector<32x16xf32>
    %18 = vector.extract_strided_slice %16 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %19 = vector.extract_strided_slice %16 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %20 = vector.extract_strided_slice %16 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %21 = vector.extract_strided_slice %16 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %22 = vector.extract_strided_slice %17 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %23 = vector.extract_strided_slice %17 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %24 = vector.extract_strided_slice %17 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %25 = vector.extract_strided_slice %17 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %26 = xetile.init_tile %arg0[%c0, %0] : memref<1024x1024xf16> -> !xetile.tile<16x8xf16>
    %27 = xetile.init_tile %arg0[%c0, %5] : memref<1024x1024xf16> -> !xetile.tile<16x8xf16>
    %28 = xetile.init_tile %arg0[%c0, %8] : memref<1024x1024xf16> -> !xetile.tile<16x8xf16>
    %29 = xetile.init_tile %arg0[%c0, %11] : memref<1024x1024xf16> -> !xetile.tile<16x8xf16>
    %30 = xetile.init_tile %arg0[%c16, %0] : memref<1024x1024xf16> -> !xetile.tile<16x8xf16>
    %31 = xetile.init_tile %arg0[%c16, %5] : memref<1024x1024xf16> -> !xetile.tile<16x8xf16>
    %32 = xetile.init_tile %arg0[%c16, %8] : memref<1024x1024xf16> -> !xetile.tile<16x8xf16>
    %33 = xetile.init_tile %arg0[%c16, %11] : memref<1024x1024xf16> -> !xetile.tile<16x8xf16>
    %34 = xetile.init_tile %arg1[%c0, %1] : memref<1024x1024xf16> -> !xetile.tile<32x16xf16>
    %35 = xetile.init_tile %arg1[%c0, %3] : memref<1024x1024xf16> -> !xetile.tile<32x16xf16>
    %36:18 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %26, %arg5 = %27, %arg6 = %28, %arg7 = %29, %arg8 = %30, %arg9 = %31, %arg10 = %32, %arg11 = %33, %arg12 = %34, %arg13 = %35, %arg14 = %18, %arg15 = %22, %arg16 = %19, %arg17 = %23, %arg18 = %20, %arg19 = %24, %arg20 = %21, %arg21 = %25) -> (!xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<32x16xf16>, !xetile.tile<32x16xf16>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>) {
      %37 = xetile.load_tile %arg4 : !xetile.tile<16x8xf16> -> vector<16x8xf16>
      %38 = xetile.load_tile %arg5 : !xetile.tile<16x8xf16> -> vector<16x8xf16>
      %39 = xetile.load_tile %arg6 : !xetile.tile<16x8xf16> -> vector<16x8xf16>
      %40 = xetile.load_tile %arg7 : !xetile.tile<16x8xf16> -> vector<16x8xf16>
      %41 = xetile.load_tile %arg8 : !xetile.tile<16x8xf16> -> vector<16x8xf16>
      %42 = xetile.load_tile %arg9 : !xetile.tile<16x8xf16> -> vector<16x8xf16>
      %43 = xetile.load_tile %arg10 : !xetile.tile<16x8xf16> -> vector<16x8xf16>
      %44 = xetile.load_tile %arg11 : !xetile.tile<16x8xf16> -> vector<16x8xf16>
      %45 = xetile.load_tile %arg12 : !xetile.tile<32x16xf16> -> vector<32x16xf16>
      %46 = xetile.load_tile %arg13 : !xetile.tile<32x16xf16> -> vector<32x16xf16>
      %47 = vector.extract_strided_slice %45 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %48 = vector.extract_strided_slice %45 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %49 = vector.extract_strided_slice %46 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %50 = vector.extract_strided_slice %46 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %51 = xetile.transpose %37, [1, 0] : vector<16x8xf16> -> vector<8x16xf16>
      %52 = xetile.transpose %41, [1, 0] : vector<16x8xf16> -> vector<8x16xf16>
      %53 = xetile.transpose %38, [1, 0] : vector<16x8xf16> -> vector<8x16xf16>
      %54 = xetile.transpose %42, [1, 0] : vector<16x8xf16> -> vector<8x16xf16>
      %55 = xetile.transpose %39, [1, 0] : vector<16x8xf16> -> vector<8x16xf16>
      %56 = xetile.transpose %43, [1, 0] : vector<16x8xf16> -> vector<8x16xf16>
      %57 = xetile.transpose %40, [1, 0] : vector<16x8xf16> -> vector<8x16xf16>
      %58 = xetile.transpose %44, [1, 0] : vector<16x8xf16> -> vector<8x16xf16>
      %59 = xetile.tile_mma %51, %47, %arg14 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %60 = xetile.tile_mma %52, %48, %59 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %61 = xetile.tile_mma %51, %49, %arg15 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %62 = xetile.tile_mma %52, %50, %61 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %63 = xetile.tile_mma %53, %47, %arg16 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %64 = xetile.tile_mma %54, %48, %63 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %65 = xetile.tile_mma %53, %49, %arg17 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %66 = xetile.tile_mma %54, %50, %65 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %67 = xetile.tile_mma %55, %47, %arg18 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %68 = xetile.tile_mma %56, %48, %67 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %69 = xetile.tile_mma %55, %49, %arg19 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %70 = xetile.tile_mma %56, %50, %69 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %71 = xetile.tile_mma %57, %47, %arg20 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %72 = xetile.tile_mma %58, %48, %71 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %73 = xetile.tile_mma %57, %49, %arg21 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %74 = xetile.tile_mma %58, %50, %73 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %75 = xetile.update_tile_offset %arg4, [%c32, %c0] : !xetile.tile<16x8xf16>
      %76 = xetile.update_tile_offset %arg5, [%c32, %c0] : !xetile.tile<16x8xf16>
      %77 = xetile.update_tile_offset %arg6, [%c32, %c0] : !xetile.tile<16x8xf16>
      %78 = xetile.update_tile_offset %arg7, [%c32, %c0] : !xetile.tile<16x8xf16>
      %79 = xetile.update_tile_offset %arg8, [%c32, %c0] : !xetile.tile<16x8xf16>
      %80 = xetile.update_tile_offset %arg9, [%c32, %c0] : !xetile.tile<16x8xf16>
      %81 = xetile.update_tile_offset %arg10, [%c32, %c0] : !xetile.tile<16x8xf16>
      %82 = xetile.update_tile_offset %arg11, [%c32, %c0] : !xetile.tile<16x8xf16>
      %83 = xetile.update_tile_offset %arg12, [%c32, %c0] : !xetile.tile<32x16xf16>
      %84 = xetile.update_tile_offset %arg13, [%c32, %c0] : !xetile.tile<32x16xf16>
      scf.yield %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %60, %62, %64, %66, %68, %70, %72, %74 : !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<16x8xf16>, !xetile.tile<32x16xf16>, !xetile.tile<32x16xf16>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>
    }
    xetile.store_tile %36#10,  %2 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    xetile.store_tile %36#11,  %4 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    xetile.store_tile %36#12,  %6 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    xetile.store_tile %36#13,  %7 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    xetile.store_tile %36#14,  %9 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    xetile.store_tile %36#15,  %10 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    xetile.store_tile %36#16,  %12 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    xetile.store_tile %36#17,  %13 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    gpu.return
  }
}

// -----// IR Dump After ConvertXeTileToXeGPU (convert-xetile-to-xegpu) //----- //
gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %c24 = arith.constant 24 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %0 = arith.muli %block_id_x, %c32 : index
    %1 = arith.muli %block_id_y, %c32 : index
    %2 = xegpu.create_nd_tdesc %arg2[%0, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %3 = arith.addi %1, %c16 : index
    %4 = xegpu.create_nd_tdesc %arg2[%0, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %5 = arith.addi %0, %c8 : index
    %6 = xegpu.create_nd_tdesc %arg2[%5, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %7 = xegpu.create_nd_tdesc %arg2[%5, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %8 = arith.addi %0, %c16 : index
    %9 = xegpu.create_nd_tdesc %arg2[%8, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %10 = xegpu.create_nd_tdesc %arg2[%8, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %11 = arith.addi %0, %c24 : index
    %12 = xegpu.create_nd_tdesc %arg2[%11, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %13 = xegpu.create_nd_tdesc %arg2[%11, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %14 = xegpu.create_nd_tdesc %arg2[%0, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %15 = xegpu.create_nd_tdesc %arg2[%0, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %16 = xegpu.load_nd %14 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    %17 = xegpu.load_nd %15 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    %18 = vector.extract_strided_slice %16 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %19 = vector.extract_strided_slice %16 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %20 = vector.extract_strided_slice %16 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %21 = vector.extract_strided_slice %16 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %22 = vector.extract_strided_slice %17 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %23 = vector.extract_strided_slice %17 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %24 = vector.extract_strided_slice %17 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %25 = vector.extract_strided_slice %17 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %26 = xegpu.create_nd_tdesc %arg0[%c0, %0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %27 = xegpu.create_nd_tdesc %arg0[%c0, %5] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %28 = xegpu.create_nd_tdesc %arg0[%c0, %8] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %29 = xegpu.create_nd_tdesc %arg0[%c0, %11] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %30 = xegpu.create_nd_tdesc %arg0[%c16, %0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %31 = xegpu.create_nd_tdesc %arg0[%c16, %5] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %32 = xegpu.create_nd_tdesc %arg0[%c16, %8] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %33 = xegpu.create_nd_tdesc %arg0[%c16, %11] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %34 = xegpu.create_nd_tdesc %arg1[%c0, %1] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %35 = xegpu.create_nd_tdesc %arg1[%c0, %3] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %36:18 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %26, %arg5 = %27, %arg6 = %28, %arg7 = %29, %arg8 = %30, %arg9 = %31, %arg10 = %32, %arg11 = %33, %arg12 = %34, %arg13 = %35, %arg14 = %18, %arg15 = %22, %arg16 = %19, %arg17 = %23, %arg18 = %20, %arg19 = %24, %arg20 = %21, %arg21 = %25) -> (!xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>) {
      %37 = xegpu.load_nd %arg4 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %38 = xegpu.load_nd %arg5 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %39 = xegpu.load_nd %arg6 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %40 = xegpu.load_nd %arg7 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %41 = xegpu.load_nd %arg8 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %42 = xegpu.load_nd %arg9 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %43 = xegpu.load_nd %arg10 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %44 = xegpu.load_nd %arg11 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %45 = xegpu.load_nd %arg12 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf16>
      %46 = xegpu.load_nd %arg13 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf16>
      %47 = vector.extract_strided_slice %45 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %48 = vector.extract_strided_slice %45 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %49 = vector.extract_strided_slice %46 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %50 = vector.extract_strided_slice %46 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %51 = vector.transpose %37, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %52 = vector.transpose %41, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %53 = vector.transpose %38, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %54 = vector.transpose %42, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %55 = vector.transpose %39, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %56 = vector.transpose %43, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %57 = vector.transpose %40, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %58 = vector.transpose %44, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %59 = xegpu.dpas %51, %47, %arg14 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %60 = xegpu.dpas %52, %48, %59 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %61 = xegpu.dpas %51, %49, %arg15 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %62 = xegpu.dpas %52, %50, %61 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %63 = xegpu.dpas %53, %47, %arg16 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %64 = xegpu.dpas %54, %48, %63 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %65 = xegpu.dpas %53, %49, %arg17 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %66 = xegpu.dpas %54, %50, %65 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %67 = xegpu.dpas %55, %47, %arg18 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %68 = xegpu.dpas %56, %48, %67 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %69 = xegpu.dpas %55, %49, %arg19 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %70 = xegpu.dpas %56, %50, %69 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %71 = xegpu.dpas %57, %47, %arg20 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %72 = xegpu.dpas %58, %48, %71 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %73 = xegpu.dpas %57, %49, %arg21 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %74 = xegpu.dpas %58, %50, %73 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %75 = xegpu.update_nd_offset %arg4, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %76 = xegpu.update_nd_offset %arg5, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %77 = xegpu.update_nd_offset %arg6, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %78 = xegpu.update_nd_offset %arg7, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %79 = xegpu.update_nd_offset %arg8, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %80 = xegpu.update_nd_offset %arg9, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %81 = xegpu.update_nd_offset %arg10, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %82 = xegpu.update_nd_offset %arg11, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %83 = xegpu.update_nd_offset %arg12, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %84 = xegpu.update_nd_offset %arg13, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      scf.yield %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %60, %62, %64, %66, %68, %70, %72, %74 : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>
    }
    xegpu.store_nd %36#10, %2 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#11, %4 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#12, %6 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#13, %7 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#14, %9 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#15, %10 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#16, %12 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#17, %13 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    gpu.return
  }
}

// -----// IR Dump After CSE (cse) //----- //
gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %c24 = arith.constant 24 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %0 = arith.muli %block_id_x, %c32 : index
    %1 = arith.muli %block_id_y, %c32 : index
    %2 = xegpu.create_nd_tdesc %arg2[%0, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %3 = arith.addi %1, %c16 : index
    %4 = xegpu.create_nd_tdesc %arg2[%0, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %5 = arith.addi %0, %c8 : index
    %6 = xegpu.create_nd_tdesc %arg2[%5, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %7 = xegpu.create_nd_tdesc %arg2[%5, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %8 = arith.addi %0, %c16 : index
    %9 = xegpu.create_nd_tdesc %arg2[%8, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %10 = xegpu.create_nd_tdesc %arg2[%8, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %11 = arith.addi %0, %c24 : index
    %12 = xegpu.create_nd_tdesc %arg2[%11, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %13 = xegpu.create_nd_tdesc %arg2[%11, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %14 = xegpu.create_nd_tdesc %arg2[%0, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %15 = xegpu.create_nd_tdesc %arg2[%0, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %16 = xegpu.load_nd %14 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    %17 = xegpu.load_nd %15 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    %18 = vector.extract_strided_slice %16 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %19 = vector.extract_strided_slice %16 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %20 = vector.extract_strided_slice %16 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %21 = vector.extract_strided_slice %16 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %22 = vector.extract_strided_slice %17 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %23 = vector.extract_strided_slice %17 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %24 = vector.extract_strided_slice %17 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %25 = vector.extract_strided_slice %17 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %26 = xegpu.create_nd_tdesc %arg0[%c0, %0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %27 = xegpu.create_nd_tdesc %arg0[%c0, %5] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %28 = xegpu.create_nd_tdesc %arg0[%c0, %8] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %29 = xegpu.create_nd_tdesc %arg0[%c0, %11] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %30 = xegpu.create_nd_tdesc %arg0[%c16, %0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %31 = xegpu.create_nd_tdesc %arg0[%c16, %5] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %32 = xegpu.create_nd_tdesc %arg0[%c16, %8] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %33 = xegpu.create_nd_tdesc %arg0[%c16, %11] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %34 = xegpu.create_nd_tdesc %arg1[%c0, %1] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %35 = xegpu.create_nd_tdesc %arg1[%c0, %3] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %36:18 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %26, %arg5 = %27, %arg6 = %28, %arg7 = %29, %arg8 = %30, %arg9 = %31, %arg10 = %32, %arg11 = %33, %arg12 = %34, %arg13 = %35, %arg14 = %18, %arg15 = %22, %arg16 = %19, %arg17 = %23, %arg18 = %20, %arg19 = %24, %arg20 = %21, %arg21 = %25) -> (!xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>) {
      %37 = xegpu.load_nd %arg4 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %38 = xegpu.load_nd %arg5 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %39 = xegpu.load_nd %arg6 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %40 = xegpu.load_nd %arg7 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %41 = xegpu.load_nd %arg8 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %42 = xegpu.load_nd %arg9 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %43 = xegpu.load_nd %arg10 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %44 = xegpu.load_nd %arg11 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %45 = xegpu.load_nd %arg12 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf16>
      %46 = xegpu.load_nd %arg13 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf16>
      %47 = vector.extract_strided_slice %45 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %48 = vector.extract_strided_slice %45 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %49 = vector.extract_strided_slice %46 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %50 = vector.extract_strided_slice %46 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %51 = vector.transpose %37, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %52 = vector.transpose %41, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %53 = vector.transpose %38, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %54 = vector.transpose %42, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %55 = vector.transpose %39, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %56 = vector.transpose %43, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %57 = vector.transpose %40, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %58 = vector.transpose %44, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %59 = xegpu.dpas %51, %47, %arg14 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %60 = xegpu.dpas %52, %48, %59 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %61 = xegpu.dpas %51, %49, %arg15 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %62 = xegpu.dpas %52, %50, %61 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %63 = xegpu.dpas %53, %47, %arg16 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %64 = xegpu.dpas %54, %48, %63 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %65 = xegpu.dpas %53, %49, %arg17 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %66 = xegpu.dpas %54, %50, %65 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %67 = xegpu.dpas %55, %47, %arg18 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %68 = xegpu.dpas %56, %48, %67 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %69 = xegpu.dpas %55, %49, %arg19 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %70 = xegpu.dpas %56, %50, %69 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %71 = xegpu.dpas %57, %47, %arg20 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %72 = xegpu.dpas %58, %48, %71 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %73 = xegpu.dpas %57, %49, %arg21 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %74 = xegpu.dpas %58, %50, %73 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %75 = xegpu.update_nd_offset %arg4, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %76 = xegpu.update_nd_offset %arg5, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %77 = xegpu.update_nd_offset %arg6, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %78 = xegpu.update_nd_offset %arg7, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %79 = xegpu.update_nd_offset %arg8, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %80 = xegpu.update_nd_offset %arg9, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %81 = xegpu.update_nd_offset %arg10, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %82 = xegpu.update_nd_offset %arg11, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %83 = xegpu.update_nd_offset %arg12, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %84 = xegpu.update_nd_offset %arg13, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      scf.yield %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %60, %62, %64, %66, %68, %70, %72, %74 : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>
    }
    xegpu.store_nd %36#10, %2 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#11, %4 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#12, %6 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#13, %7 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#14, %9 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#15, %10 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#16, %12 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#17, %13 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    gpu.return
  }
}

// -----// IR Dump After HoistTranspose (imex-xegpu-hoist-transpose) //----- //
gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %c24 = arith.constant 24 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %0 = arith.muli %block_id_x, %c32 : index
    %1 = arith.muli %block_id_y, %c32 : index
    %2 = xegpu.create_nd_tdesc %arg2[%0, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %3 = arith.addi %1, %c16 : index
    %4 = xegpu.create_nd_tdesc %arg2[%0, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %5 = arith.addi %0, %c8 : index
    %6 = xegpu.create_nd_tdesc %arg2[%5, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %7 = xegpu.create_nd_tdesc %arg2[%5, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %8 = arith.addi %0, %c16 : index
    %9 = xegpu.create_nd_tdesc %arg2[%8, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %10 = xegpu.create_nd_tdesc %arg2[%8, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %11 = arith.addi %0, %c24 : index
    %12 = xegpu.create_nd_tdesc %arg2[%11, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %13 = xegpu.create_nd_tdesc %arg2[%11, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %14 = xegpu.create_nd_tdesc %arg2[%0, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %15 = xegpu.create_nd_tdesc %arg2[%0, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %16 = xegpu.load_nd %14 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    %17 = xegpu.load_nd %15 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    %18 = vector.extract_strided_slice %16 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %19 = vector.extract_strided_slice %16 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %20 = vector.extract_strided_slice %16 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %21 = vector.extract_strided_slice %16 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %22 = vector.extract_strided_slice %17 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %23 = vector.extract_strided_slice %17 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %24 = vector.extract_strided_slice %17 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %25 = vector.extract_strided_slice %17 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %26 = xegpu.create_nd_tdesc %arg0[%c0, %0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %27 = xegpu.create_nd_tdesc %arg0[%c0, %5] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %28 = xegpu.create_nd_tdesc %arg0[%c0, %8] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %29 = xegpu.create_nd_tdesc %arg0[%c0, %11] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %30 = xegpu.create_nd_tdesc %arg0[%c16, %0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %31 = xegpu.create_nd_tdesc %arg0[%c16, %5] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %32 = xegpu.create_nd_tdesc %arg0[%c16, %8] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %33 = xegpu.create_nd_tdesc %arg0[%c16, %11] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %34 = xegpu.create_nd_tdesc %arg1[%c0, %1] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %35 = xegpu.create_nd_tdesc %arg1[%c0, %3] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %36:18 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %26, %arg5 = %27, %arg6 = %28, %arg7 = %29, %arg8 = %30, %arg9 = %31, %arg10 = %32, %arg11 = %33, %arg12 = %34, %arg13 = %35, %arg14 = %18, %arg15 = %22, %arg16 = %19, %arg17 = %23, %arg18 = %20, %arg19 = %24, %arg20 = %21, %arg21 = %25) -> (!xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>) {
      %37 = xegpu.load_nd %arg4 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %38 = xegpu.load_nd %arg5 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %39 = xegpu.load_nd %arg6 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %40 = xegpu.load_nd %arg7 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %41 = xegpu.load_nd %arg8 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %42 = xegpu.load_nd %arg9 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %43 = xegpu.load_nd %arg10 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %44 = xegpu.load_nd %arg11 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %45 = xegpu.load_nd %arg12 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf16>
      %46 = xegpu.load_nd %arg13 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf16>
      %47 = vector.extract_strided_slice %45 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %48 = vector.extract_strided_slice %45 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %49 = vector.extract_strided_slice %46 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %50 = vector.extract_strided_slice %46 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %51 = vector.transpose %37, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %52 = vector.transpose %41, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %53 = vector.transpose %38, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %54 = vector.transpose %42, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %55 = vector.transpose %39, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %56 = vector.transpose %43, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %57 = vector.transpose %40, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %58 = vector.transpose %44, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %59 = xegpu.dpas %51, %47, %arg14 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %60 = xegpu.dpas %52, %48, %59 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %61 = xegpu.dpas %51, %49, %arg15 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %62 = xegpu.dpas %52, %50, %61 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %63 = xegpu.dpas %53, %47, %arg16 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %64 = xegpu.dpas %54, %48, %63 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %65 = xegpu.dpas %53, %49, %arg17 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %66 = xegpu.dpas %54, %50, %65 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %67 = xegpu.dpas %55, %47, %arg18 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %68 = xegpu.dpas %56, %48, %67 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %69 = xegpu.dpas %55, %49, %arg19 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %70 = xegpu.dpas %56, %50, %69 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %71 = xegpu.dpas %57, %47, %arg20 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %72 = xegpu.dpas %58, %48, %71 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %73 = xegpu.dpas %57, %49, %arg21 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %74 = xegpu.dpas %58, %50, %73 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %75 = xegpu.update_nd_offset %arg4, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %76 = xegpu.update_nd_offset %arg5, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %77 = xegpu.update_nd_offset %arg6, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %78 = xegpu.update_nd_offset %arg7, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %79 = xegpu.update_nd_offset %arg8, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %80 = xegpu.update_nd_offset %arg9, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %81 = xegpu.update_nd_offset %arg10, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %82 = xegpu.update_nd_offset %arg11, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %83 = xegpu.update_nd_offset %arg12, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %84 = xegpu.update_nd_offset %arg13, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      scf.yield %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %60, %62, %64, %66, %68, %70, %72, %74 : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>
    }
    xegpu.store_nd %36#10, %2 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#11, %4 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#12, %6 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#13, %7 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#14, %9 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#15, %10 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#16, %12 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#17, %13 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    gpu.return
  }
}

// -----// IR Dump After VnniTransformation (imex-xegpu-apply-vnni-transformation) //----- //
gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %c24 = arith.constant 24 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %0 = arith.muli %block_id_x, %c32 : index
    %1 = arith.muli %block_id_y, %c32 : index
    %2 = xegpu.create_nd_tdesc %arg2[%0, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %3 = arith.addi %1, %c16 : index
    %4 = xegpu.create_nd_tdesc %arg2[%0, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %5 = arith.addi %0, %c8 : index
    %6 = xegpu.create_nd_tdesc %arg2[%5, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %7 = xegpu.create_nd_tdesc %arg2[%5, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %8 = arith.addi %0, %c16 : index
    %9 = xegpu.create_nd_tdesc %arg2[%8, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %10 = xegpu.create_nd_tdesc %arg2[%8, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %11 = arith.addi %0, %c24 : index
    %12 = xegpu.create_nd_tdesc %arg2[%11, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %13 = xegpu.create_nd_tdesc %arg2[%11, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %14 = xegpu.create_nd_tdesc %arg2[%0, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %15 = xegpu.create_nd_tdesc %arg2[%0, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %16 = xegpu.load_nd %14 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    %17 = xegpu.load_nd %15 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    %18 = vector.extract_strided_slice %16 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %19 = vector.extract_strided_slice %16 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %20 = vector.extract_strided_slice %16 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %21 = vector.extract_strided_slice %16 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %22 = vector.extract_strided_slice %17 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %23 = vector.extract_strided_slice %17 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %24 = vector.extract_strided_slice %17 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %25 = vector.extract_strided_slice %17 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %26 = xegpu.create_nd_tdesc %arg0[%c0, %0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %27 = xegpu.create_nd_tdesc %arg0[%c0, %5] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %28 = xegpu.create_nd_tdesc %arg0[%c0, %8] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %29 = xegpu.create_nd_tdesc %arg0[%c0, %11] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %30 = xegpu.create_nd_tdesc %arg0[%c16, %0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %31 = xegpu.create_nd_tdesc %arg0[%c16, %5] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %32 = xegpu.create_nd_tdesc %arg0[%c16, %8] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %33 = xegpu.create_nd_tdesc %arg0[%c16, %11] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %34 = xegpu.create_nd_tdesc %arg1[%c0, %1] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %35 = xegpu.create_nd_tdesc %arg1[%c0, %3] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %36:18 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %26, %arg5 = %27, %arg6 = %28, %arg7 = %29, %arg8 = %30, %arg9 = %31, %arg10 = %32, %arg11 = %33, %arg12 = %34, %arg13 = %35, %arg14 = %18, %arg15 = %22, %arg16 = %19, %arg17 = %23, %arg18 = %20, %arg19 = %24, %arg20 = %21, %arg21 = %25) -> (!xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>) {
      %37 = xegpu.load_nd %arg4 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %38 = xegpu.load_nd %arg5 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %39 = xegpu.load_nd %arg6 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %40 = xegpu.load_nd %arg7 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %41 = xegpu.load_nd %arg8 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %42 = xegpu.load_nd %arg9 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %43 = xegpu.load_nd %arg10 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %44 = xegpu.load_nd %arg11 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x8xf16>
      %45 = xegpu.load_nd %arg12 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x16x2xf16>
      %46 = xegpu.load_nd %arg13 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x16x2xf16>
      %47 = vector.extract_strided_slice %45 {offsets = [0, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
      %48 = vector.extract_strided_slice %45 {offsets = [8, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
      %49 = vector.extract_strided_slice %46 {offsets = [0, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
      %50 = vector.extract_strided_slice %46 {offsets = [8, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
      %51 = vector.transpose %37, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %52 = vector.transpose %41, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %53 = vector.transpose %38, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %54 = vector.transpose %42, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %55 = vector.transpose %39, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %56 = vector.transpose %43, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %57 = vector.transpose %40, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %58 = vector.transpose %44, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %59 = xegpu.dpas %51, %47, %arg14 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %60 = xegpu.dpas %52, %48, %59 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %61 = xegpu.dpas %51, %49, %arg15 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %62 = xegpu.dpas %52, %50, %61 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %63 = xegpu.dpas %53, %47, %arg16 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %64 = xegpu.dpas %54, %48, %63 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %65 = xegpu.dpas %53, %49, %arg17 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %66 = xegpu.dpas %54, %50, %65 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %67 = xegpu.dpas %55, %47, %arg18 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %68 = xegpu.dpas %56, %48, %67 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %69 = xegpu.dpas %55, %49, %arg19 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %70 = xegpu.dpas %56, %50, %69 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %71 = xegpu.dpas %57, %47, %arg20 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %72 = xegpu.dpas %58, %48, %71 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %73 = xegpu.dpas %57, %49, %arg21 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %74 = xegpu.dpas %58, %50, %73 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %75 = xegpu.update_nd_offset %arg4, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %76 = xegpu.update_nd_offset %arg5, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %77 = xegpu.update_nd_offset %arg6, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %78 = xegpu.update_nd_offset %arg7, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %79 = xegpu.update_nd_offset %arg8, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %80 = xegpu.update_nd_offset %arg9, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %81 = xegpu.update_nd_offset %arg10, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %82 = xegpu.update_nd_offset %arg11, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %83 = xegpu.update_nd_offset %arg12, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %84 = xegpu.update_nd_offset %arg13, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      scf.yield %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %60, %62, %64, %66, %68, %70, %72, %74 : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>
    }
    xegpu.store_nd %36#10, %2 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#11, %4 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#12, %6 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#13, %7 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#14, %9 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#15, %10 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#16, %12 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#17, %13 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    gpu.return
  }
}

// -----// IR Dump After OptimizeTranspose (imex-xegpu-optimize-transpose) //----- //
gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %cst = arith.constant dense<true> : vector<8xi1>
    %cst_0 = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56]> : vector<8xindex>
    %c64 = arith.constant 64 : index
    %c24 = arith.constant 24 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %0 = arith.muli %block_id_x, %c32 : index
    %1 = arith.muli %block_id_y, %c32 : index
    %2 = xegpu.create_nd_tdesc %arg2[%0, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %3 = arith.addi %1, %c16 : index
    %4 = xegpu.create_nd_tdesc %arg2[%0, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %5 = arith.addi %0, %c8 : index
    %6 = xegpu.create_nd_tdesc %arg2[%5, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %7 = xegpu.create_nd_tdesc %arg2[%5, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %8 = arith.addi %0, %c16 : index
    %9 = xegpu.create_nd_tdesc %arg2[%8, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %10 = xegpu.create_nd_tdesc %arg2[%8, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %11 = arith.addi %0, %c24 : index
    %12 = xegpu.create_nd_tdesc %arg2[%11, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %13 = xegpu.create_nd_tdesc %arg2[%11, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %14 = xegpu.create_nd_tdesc %arg2[%0, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %15 = xegpu.create_nd_tdesc %arg2[%0, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %16 = xegpu.load_nd %14 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    %17 = xegpu.load_nd %15 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    %18 = vector.extract_strided_slice %16 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %19 = vector.extract_strided_slice %16 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %20 = vector.extract_strided_slice %16 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %21 = vector.extract_strided_slice %16 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %22 = vector.extract_strided_slice %17 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %23 = vector.extract_strided_slice %17 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %24 = vector.extract_strided_slice %17 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %25 = vector.extract_strided_slice %17 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %26 = xegpu.create_nd_tdesc %arg0[%c0, %0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %27 = xegpu.create_nd_tdesc %arg0[%c0, %5] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %28 = xegpu.create_nd_tdesc %arg0[%c0, %8] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %29 = xegpu.create_nd_tdesc %arg0[%c0, %11] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %30 = xegpu.create_nd_tdesc %arg0[%c16, %0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %31 = xegpu.create_nd_tdesc %arg0[%c16, %5] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %32 = xegpu.create_nd_tdesc %arg0[%c16, %8] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %33 = xegpu.create_nd_tdesc %arg0[%c16, %11] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %34 = xegpu.create_nd_tdesc %arg1[%c0, %1] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %35 = xegpu.create_nd_tdesc %arg1[%c0, %3] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %36:18 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %26, %arg5 = %27, %arg6 = %28, %arg7 = %29, %arg8 = %30, %arg9 = %31, %arg10 = %32, %arg11 = %33, %arg12 = %34, %arg13 = %35, %arg14 = %18, %arg15 = %22, %arg16 = %19, %arg17 = %23, %arg18 = %20, %arg19 = %24, %arg20 = %21, %arg21 = %25) -> (!xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>) {
      %37 = xegpu.load_nd %arg4 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<8x8x2xf16>
      %38 = xegpu.load_nd %arg5 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<8x8x2xf16>
      %39 = xegpu.load_nd %arg6 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<8x8x2xf16>
      %40 = xegpu.load_nd %arg7 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<8x8x2xf16>
      %41 = xegpu.load_nd %arg8 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<8x8x2xf16>
      %42 = xegpu.load_nd %arg9 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<8x8x2xf16>
      %43 = xegpu.load_nd %arg10 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<8x8x2xf16>
      %44 = xegpu.load_nd %arg11 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<8x8x2xf16>
      %45 = xegpu.load_nd %arg12 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x16x2xf16>
      %46 = xegpu.load_nd %arg13 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x16x2xf16>
      %47 = vector.extract_strided_slice %45 {offsets = [0, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
      %48 = vector.extract_strided_slice %45 {offsets = [8, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
      %49 = vector.extract_strided_slice %46 {offsets = [0, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
      %50 = vector.extract_strided_slice %46 {offsets = [8, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
      %51 = vector.shape_cast %37 : vector<8x8x2xf16> to vector<128xf16>
      %52 = vector.bitcast %51 : vector<128xf16> to vector<64xf32>
      %53 = vector.shape_cast %52 : vector<64xf32> to vector<8x8xf32>
      %alloc = memref.alloc() : memref<4096xf32, 3>
      %54 = gpu.subgroup_id : index
      %55 = arith.muli %54, %c64 : index
      %56 = vector.broadcast %55 : index to vector<8xindex>
      %57 = arith.addi %56, %cst_0 : vector<8xindex>
      %58 = xegpu.create_tdesc %alloc, %57 : memref<4096xf32, 3>, vector<8xindex> -> !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>
      xegpu.store %53, %58, %cst <{transpose}> : vector<8x8xf32>, !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>, vector<8xi1>
      %59 = xegpu.create_nd_tdesc %alloc[%55] : memref<4096xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
      %60 = xegpu.load_nd %59  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
      %61 = vector.bitcast %60 : vector<64xf32> to vector<128xf16>
      %62 = vector.shape_cast %61 : vector<128xf16> to vector<8x16xf16>
      %63 = vector.shape_cast %41 : vector<8x8x2xf16> to vector<128xf16>
      %64 = vector.bitcast %63 : vector<128xf16> to vector<64xf32>
      %65 = vector.shape_cast %64 : vector<64xf32> to vector<8x8xf32>
      %alloc_1 = memref.alloc() : memref<4096xf32, 3>
      %66 = gpu.subgroup_id : index
      %67 = arith.muli %66, %c64 : index
      %68 = vector.broadcast %67 : index to vector<8xindex>
      %69 = arith.addi %68, %cst_0 : vector<8xindex>
      %70 = xegpu.create_tdesc %alloc_1, %69 : memref<4096xf32, 3>, vector<8xindex> -> !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>
      xegpu.store %65, %70, %cst <{transpose}> : vector<8x8xf32>, !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>, vector<8xi1>
      %71 = xegpu.create_nd_tdesc %alloc_1[%67] : memref<4096xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
      %72 = xegpu.load_nd %71  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
      %73 = vector.bitcast %72 : vector<64xf32> to vector<128xf16>
      %74 = vector.shape_cast %73 : vector<128xf16> to vector<8x16xf16>
      %75 = vector.shape_cast %38 : vector<8x8x2xf16> to vector<128xf16>
      %76 = vector.bitcast %75 : vector<128xf16> to vector<64xf32>
      %77 = vector.shape_cast %76 : vector<64xf32> to vector<8x8xf32>
      %alloc_2 = memref.alloc() : memref<4096xf32, 3>
      %78 = gpu.subgroup_id : index
      %79 = arith.muli %78, %c64 : index
      %80 = vector.broadcast %79 : index to vector<8xindex>
      %81 = arith.addi %80, %cst_0 : vector<8xindex>
      %82 = xegpu.create_tdesc %alloc_2, %81 : memref<4096xf32, 3>, vector<8xindex> -> !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>
      xegpu.store %77, %82, %cst <{transpose}> : vector<8x8xf32>, !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>, vector<8xi1>
      %83 = xegpu.create_nd_tdesc %alloc_2[%79] : memref<4096xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
      %84 = xegpu.load_nd %83  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
      %85 = vector.bitcast %84 : vector<64xf32> to vector<128xf16>
      %86 = vector.shape_cast %85 : vector<128xf16> to vector<8x16xf16>
      %87 = vector.shape_cast %42 : vector<8x8x2xf16> to vector<128xf16>
      %88 = vector.bitcast %87 : vector<128xf16> to vector<64xf32>
      %89 = vector.shape_cast %88 : vector<64xf32> to vector<8x8xf32>
      %alloc_3 = memref.alloc() : memref<4096xf32, 3>
      %90 = gpu.subgroup_id : index
      %91 = arith.muli %90, %c64 : index
      %92 = vector.broadcast %91 : index to vector<8xindex>
      %93 = arith.addi %92, %cst_0 : vector<8xindex>
      %94 = xegpu.create_tdesc %alloc_3, %93 : memref<4096xf32, 3>, vector<8xindex> -> !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>
      xegpu.store %89, %94, %cst <{transpose}> : vector<8x8xf32>, !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>, vector<8xi1>
      %95 = xegpu.create_nd_tdesc %alloc_3[%91] : memref<4096xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
      %96 = xegpu.load_nd %95  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
      %97 = vector.bitcast %96 : vector<64xf32> to vector<128xf16>
      %98 = vector.shape_cast %97 : vector<128xf16> to vector<8x16xf16>
      %99 = vector.shape_cast %39 : vector<8x8x2xf16> to vector<128xf16>
      %100 = vector.bitcast %99 : vector<128xf16> to vector<64xf32>
      %101 = vector.shape_cast %100 : vector<64xf32> to vector<8x8xf32>
      %alloc_4 = memref.alloc() : memref<4096xf32, 3>
      %102 = gpu.subgroup_id : index
      %103 = arith.muli %102, %c64 : index
      %104 = vector.broadcast %103 : index to vector<8xindex>
      %105 = arith.addi %104, %cst_0 : vector<8xindex>
      %106 = xegpu.create_tdesc %alloc_4, %105 : memref<4096xf32, 3>, vector<8xindex> -> !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>
      xegpu.store %101, %106, %cst <{transpose}> : vector<8x8xf32>, !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>, vector<8xi1>
      %107 = xegpu.create_nd_tdesc %alloc_4[%103] : memref<4096xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
      %108 = xegpu.load_nd %107  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
      %109 = vector.bitcast %108 : vector<64xf32> to vector<128xf16>
      %110 = vector.shape_cast %109 : vector<128xf16> to vector<8x16xf16>
      %111 = vector.shape_cast %43 : vector<8x8x2xf16> to vector<128xf16>
      %112 = vector.bitcast %111 : vector<128xf16> to vector<64xf32>
      %113 = vector.shape_cast %112 : vector<64xf32> to vector<8x8xf32>
      %alloc_5 = memref.alloc() : memref<4096xf32, 3>
      %114 = gpu.subgroup_id : index
      %115 = arith.muli %114, %c64 : index
      %116 = vector.broadcast %115 : index to vector<8xindex>
      %117 = arith.addi %116, %cst_0 : vector<8xindex>
      %118 = xegpu.create_tdesc %alloc_5, %117 : memref<4096xf32, 3>, vector<8xindex> -> !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>
      xegpu.store %113, %118, %cst <{transpose}> : vector<8x8xf32>, !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>, vector<8xi1>
      %119 = xegpu.create_nd_tdesc %alloc_5[%115] : memref<4096xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
      %120 = xegpu.load_nd %119  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
      %121 = vector.bitcast %120 : vector<64xf32> to vector<128xf16>
      %122 = vector.shape_cast %121 : vector<128xf16> to vector<8x16xf16>
      %123 = vector.shape_cast %40 : vector<8x8x2xf16> to vector<128xf16>
      %124 = vector.bitcast %123 : vector<128xf16> to vector<64xf32>
      %125 = vector.shape_cast %124 : vector<64xf32> to vector<8x8xf32>
      %alloc_6 = memref.alloc() : memref<4096xf32, 3>
      %126 = gpu.subgroup_id : index
      %127 = arith.muli %126, %c64 : index
      %128 = vector.broadcast %127 : index to vector<8xindex>
      %129 = arith.addi %128, %cst_0 : vector<8xindex>
      %130 = xegpu.create_tdesc %alloc_6, %129 : memref<4096xf32, 3>, vector<8xindex> -> !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>
      xegpu.store %125, %130, %cst <{transpose}> : vector<8x8xf32>, !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>, vector<8xi1>
      %131 = xegpu.create_nd_tdesc %alloc_6[%127] : memref<4096xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
      %132 = xegpu.load_nd %131  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
      %133 = vector.bitcast %132 : vector<64xf32> to vector<128xf16>
      %134 = vector.shape_cast %133 : vector<128xf16> to vector<8x16xf16>
      %135 = vector.shape_cast %44 : vector<8x8x2xf16> to vector<128xf16>
      %136 = vector.bitcast %135 : vector<128xf16> to vector<64xf32>
      %137 = vector.shape_cast %136 : vector<64xf32> to vector<8x8xf32>
      %alloc_7 = memref.alloc() : memref<4096xf32, 3>
      %138 = gpu.subgroup_id : index
      %139 = arith.muli %138, %c64 : index
      %140 = vector.broadcast %139 : index to vector<8xindex>
      %141 = arith.addi %140, %cst_0 : vector<8xindex>
      %142 = xegpu.create_tdesc %alloc_7, %141 : memref<4096xf32, 3>, vector<8xindex> -> !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>
      xegpu.store %137, %142, %cst <{transpose}> : vector<8x8xf32>, !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>, vector<8xi1>
      %143 = xegpu.create_nd_tdesc %alloc_7[%139] : memref<4096xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
      %144 = xegpu.load_nd %143  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
      %145 = vector.bitcast %144 : vector<64xf32> to vector<128xf16>
      %146 = vector.shape_cast %145 : vector<128xf16> to vector<8x16xf16>
      %147 = xegpu.dpas %62, %47, %arg14 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %148 = xegpu.dpas %74, %48, %147 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %149 = xegpu.dpas %62, %49, %arg15 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %150 = xegpu.dpas %74, %50, %149 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %151 = xegpu.dpas %86, %47, %arg16 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %152 = xegpu.dpas %98, %48, %151 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %153 = xegpu.dpas %86, %49, %arg17 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %154 = xegpu.dpas %98, %50, %153 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %155 = xegpu.dpas %110, %47, %arg18 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %156 = xegpu.dpas %122, %48, %155 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %157 = xegpu.dpas %110, %49, %arg19 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %158 = xegpu.dpas %122, %50, %157 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %159 = xegpu.dpas %134, %47, %arg20 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %160 = xegpu.dpas %146, %48, %159 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %161 = xegpu.dpas %134, %49, %arg21 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %162 = xegpu.dpas %146, %50, %161 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %163 = xegpu.update_nd_offset %arg4, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %164 = xegpu.update_nd_offset %arg5, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %165 = xegpu.update_nd_offset %arg6, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %166 = xegpu.update_nd_offset %arg7, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %167 = xegpu.update_nd_offset %arg8, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %168 = xegpu.update_nd_offset %arg9, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %169 = xegpu.update_nd_offset %arg10, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %170 = xegpu.update_nd_offset %arg11, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %171 = xegpu.update_nd_offset %arg12, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %172 = xegpu.update_nd_offset %arg13, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      scf.yield %163, %164, %165, %166, %167, %168, %169, %170, %171, %172, %148, %150, %152, %154, %156, %158, %160, %162 : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>
    }
    xegpu.store_nd %36#10, %2 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#11, %4 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#12, %6 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#13, %7 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#14, %9 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#15, %10 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#16, %12 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#17, %13 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    gpu.return
  }
}

// -----// IR Dump After CSE (cse) //----- //
gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %cst = arith.constant dense<true> : vector<8xi1>
    %cst_0 = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56]> : vector<8xindex>
    %c64 = arith.constant 64 : index
    %c24 = arith.constant 24 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %0 = arith.muli %block_id_x, %c32 : index
    %1 = arith.muli %block_id_y, %c32 : index
    %2 = xegpu.create_nd_tdesc %arg2[%0, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %3 = arith.addi %1, %c16 : index
    %4 = xegpu.create_nd_tdesc %arg2[%0, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %5 = arith.addi %0, %c8 : index
    %6 = xegpu.create_nd_tdesc %arg2[%5, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %7 = xegpu.create_nd_tdesc %arg2[%5, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %8 = arith.addi %0, %c16 : index
    %9 = xegpu.create_nd_tdesc %arg2[%8, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %10 = xegpu.create_nd_tdesc %arg2[%8, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %11 = arith.addi %0, %c24 : index
    %12 = xegpu.create_nd_tdesc %arg2[%11, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %13 = xegpu.create_nd_tdesc %arg2[%11, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %14 = xegpu.create_nd_tdesc %arg2[%0, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %15 = xegpu.create_nd_tdesc %arg2[%0, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %16 = xegpu.load_nd %14 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    %17 = xegpu.load_nd %15 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    %18 = vector.extract_strided_slice %16 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %19 = vector.extract_strided_slice %16 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %20 = vector.extract_strided_slice %16 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %21 = vector.extract_strided_slice %16 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %22 = vector.extract_strided_slice %17 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %23 = vector.extract_strided_slice %17 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %24 = vector.extract_strided_slice %17 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %25 = vector.extract_strided_slice %17 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %26 = xegpu.create_nd_tdesc %arg0[%c0, %0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %27 = xegpu.create_nd_tdesc %arg0[%c0, %5] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %28 = xegpu.create_nd_tdesc %arg0[%c0, %8] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %29 = xegpu.create_nd_tdesc %arg0[%c0, %11] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %30 = xegpu.create_nd_tdesc %arg0[%c16, %0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %31 = xegpu.create_nd_tdesc %arg0[%c16, %5] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %32 = xegpu.create_nd_tdesc %arg0[%c16, %8] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %33 = xegpu.create_nd_tdesc %arg0[%c16, %11] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %34 = xegpu.create_nd_tdesc %arg1[%c0, %1] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %35 = xegpu.create_nd_tdesc %arg1[%c0, %3] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %36:18 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %26, %arg5 = %27, %arg6 = %28, %arg7 = %29, %arg8 = %30, %arg9 = %31, %arg10 = %32, %arg11 = %33, %arg12 = %34, %arg13 = %35, %arg14 = %18, %arg15 = %22, %arg16 = %19, %arg17 = %23, %arg18 = %20, %arg19 = %24, %arg20 = %21, %arg21 = %25) -> (!xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>) {
      %37 = xegpu.load_nd %arg4 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<8x8x2xf16>
      %38 = xegpu.load_nd %arg5 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<8x8x2xf16>
      %39 = xegpu.load_nd %arg6 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<8x8x2xf16>
      %40 = xegpu.load_nd %arg7 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<8x8x2xf16>
      %41 = xegpu.load_nd %arg8 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<8x8x2xf16>
      %42 = xegpu.load_nd %arg9 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<8x8x2xf16>
      %43 = xegpu.load_nd %arg10 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<8x8x2xf16>
      %44 = xegpu.load_nd %arg11 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<8x8x2xf16>
      %45 = xegpu.load_nd %arg12 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x16x2xf16>
      %46 = xegpu.load_nd %arg13 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x16x2xf16>
      %47 = vector.extract_strided_slice %45 {offsets = [0, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
      %48 = vector.extract_strided_slice %45 {offsets = [8, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
      %49 = vector.extract_strided_slice %46 {offsets = [0, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
      %50 = vector.extract_strided_slice %46 {offsets = [8, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
      %51 = vector.shape_cast %37 : vector<8x8x2xf16> to vector<128xf16>
      %52 = vector.bitcast %51 : vector<128xf16> to vector<64xf32>
      %53 = vector.shape_cast %52 : vector<64xf32> to vector<8x8xf32>
      %alloc = memref.alloc() : memref<4096xf32, 3>
      %54 = gpu.subgroup_id : index
      %55 = arith.muli %54, %c64 : index
      %56 = vector.broadcast %55 : index to vector<8xindex>
      %57 = arith.addi %56, %cst_0 : vector<8xindex>
      %58 = xegpu.create_tdesc %alloc, %57 : memref<4096xf32, 3>, vector<8xindex> -> !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>
      xegpu.store %53, %58, %cst <{transpose}> : vector<8x8xf32>, !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>, vector<8xi1>
      %59 = xegpu.create_nd_tdesc %alloc[%55] : memref<4096xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
      %60 = xegpu.load_nd %59  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
      %61 = vector.bitcast %60 : vector<64xf32> to vector<128xf16>
      %62 = vector.shape_cast %61 : vector<128xf16> to vector<8x16xf16>
      %63 = vector.shape_cast %41 : vector<8x8x2xf16> to vector<128xf16>
      %64 = vector.bitcast %63 : vector<128xf16> to vector<64xf32>
      %65 = vector.shape_cast %64 : vector<64xf32> to vector<8x8xf32>
      %alloc_1 = memref.alloc() : memref<4096xf32, 3>
      %66 = xegpu.create_tdesc %alloc_1, %57 : memref<4096xf32, 3>, vector<8xindex> -> !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>
      xegpu.store %65, %66, %cst <{transpose}> : vector<8x8xf32>, !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>, vector<8xi1>
      %67 = xegpu.create_nd_tdesc %alloc_1[%55] : memref<4096xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
      %68 = xegpu.load_nd %67  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
      %69 = vector.bitcast %68 : vector<64xf32> to vector<128xf16>
      %70 = vector.shape_cast %69 : vector<128xf16> to vector<8x16xf16>
      %71 = vector.shape_cast %38 : vector<8x8x2xf16> to vector<128xf16>
      %72 = vector.bitcast %71 : vector<128xf16> to vector<64xf32>
      %73 = vector.shape_cast %72 : vector<64xf32> to vector<8x8xf32>
      %alloc_2 = memref.alloc() : memref<4096xf32, 3>
      %74 = xegpu.create_tdesc %alloc_2, %57 : memref<4096xf32, 3>, vector<8xindex> -> !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>
      xegpu.store %73, %74, %cst <{transpose}> : vector<8x8xf32>, !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>, vector<8xi1>
      %75 = xegpu.create_nd_tdesc %alloc_2[%55] : memref<4096xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
      %76 = xegpu.load_nd %75  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
      %77 = vector.bitcast %76 : vector<64xf32> to vector<128xf16>
      %78 = vector.shape_cast %77 : vector<128xf16> to vector<8x16xf16>
      %79 = vector.shape_cast %42 : vector<8x8x2xf16> to vector<128xf16>
      %80 = vector.bitcast %79 : vector<128xf16> to vector<64xf32>
      %81 = vector.shape_cast %80 : vector<64xf32> to vector<8x8xf32>
      %alloc_3 = memref.alloc() : memref<4096xf32, 3>
      %82 = xegpu.create_tdesc %alloc_3, %57 : memref<4096xf32, 3>, vector<8xindex> -> !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>
      xegpu.store %81, %82, %cst <{transpose}> : vector<8x8xf32>, !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>, vector<8xi1>
      %83 = xegpu.create_nd_tdesc %alloc_3[%55] : memref<4096xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
      %84 = xegpu.load_nd %83  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
      %85 = vector.bitcast %84 : vector<64xf32> to vector<128xf16>
      %86 = vector.shape_cast %85 : vector<128xf16> to vector<8x16xf16>
      %87 = vector.shape_cast %39 : vector<8x8x2xf16> to vector<128xf16>
      %88 = vector.bitcast %87 : vector<128xf16> to vector<64xf32>
      %89 = vector.shape_cast %88 : vector<64xf32> to vector<8x8xf32>
      %alloc_4 = memref.alloc() : memref<4096xf32, 3>
      %90 = xegpu.create_tdesc %alloc_4, %57 : memref<4096xf32, 3>, vector<8xindex> -> !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>
      xegpu.store %89, %90, %cst <{transpose}> : vector<8x8xf32>, !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>, vector<8xi1>
      %91 = xegpu.create_nd_tdesc %alloc_4[%55] : memref<4096xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
      %92 = xegpu.load_nd %91  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
      %93 = vector.bitcast %92 : vector<64xf32> to vector<128xf16>
      %94 = vector.shape_cast %93 : vector<128xf16> to vector<8x16xf16>
      %95 = vector.shape_cast %43 : vector<8x8x2xf16> to vector<128xf16>
      %96 = vector.bitcast %95 : vector<128xf16> to vector<64xf32>
      %97 = vector.shape_cast %96 : vector<64xf32> to vector<8x8xf32>
      %alloc_5 = memref.alloc() : memref<4096xf32, 3>
      %98 = xegpu.create_tdesc %alloc_5, %57 : memref<4096xf32, 3>, vector<8xindex> -> !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>
      xegpu.store %97, %98, %cst <{transpose}> : vector<8x8xf32>, !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>, vector<8xi1>
      %99 = xegpu.create_nd_tdesc %alloc_5[%55] : memref<4096xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
      %100 = xegpu.load_nd %99  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
      %101 = vector.bitcast %100 : vector<64xf32> to vector<128xf16>
      %102 = vector.shape_cast %101 : vector<128xf16> to vector<8x16xf16>
      %103 = vector.shape_cast %40 : vector<8x8x2xf16> to vector<128xf16>
      %104 = vector.bitcast %103 : vector<128xf16> to vector<64xf32>
      %105 = vector.shape_cast %104 : vector<64xf32> to vector<8x8xf32>
      %alloc_6 = memref.alloc() : memref<4096xf32, 3>
      %106 = xegpu.create_tdesc %alloc_6, %57 : memref<4096xf32, 3>, vector<8xindex> -> !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>
      xegpu.store %105, %106, %cst <{transpose}> : vector<8x8xf32>, !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>, vector<8xi1>
      %107 = xegpu.create_nd_tdesc %alloc_6[%55] : memref<4096xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
      %108 = xegpu.load_nd %107  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
      %109 = vector.bitcast %108 : vector<64xf32> to vector<128xf16>
      %110 = vector.shape_cast %109 : vector<128xf16> to vector<8x16xf16>
      %111 = vector.shape_cast %44 : vector<8x8x2xf16> to vector<128xf16>
      %112 = vector.bitcast %111 : vector<128xf16> to vector<64xf32>
      %113 = vector.shape_cast %112 : vector<64xf32> to vector<8x8xf32>
      %alloc_7 = memref.alloc() : memref<4096xf32, 3>
      %114 = xegpu.create_tdesc %alloc_7, %57 : memref<4096xf32, 3>, vector<8xindex> -> !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>
      xegpu.store %113, %114, %cst <{transpose}> : vector<8x8xf32>, !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>, vector<8xi1>
      %115 = xegpu.create_nd_tdesc %alloc_7[%55] : memref<4096xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
      %116 = xegpu.load_nd %115  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
      %117 = vector.bitcast %116 : vector<64xf32> to vector<128xf16>
      %118 = vector.shape_cast %117 : vector<128xf16> to vector<8x16xf16>
      %119 = xegpu.dpas %62, %47, %arg14 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %120 = xegpu.dpas %70, %48, %119 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %121 = xegpu.dpas %62, %49, %arg15 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %122 = xegpu.dpas %70, %50, %121 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %123 = xegpu.dpas %78, %47, %arg16 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %124 = xegpu.dpas %86, %48, %123 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %125 = xegpu.dpas %78, %49, %arg17 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %126 = xegpu.dpas %86, %50, %125 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %127 = xegpu.dpas %94, %47, %arg18 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %128 = xegpu.dpas %102, %48, %127 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %129 = xegpu.dpas %94, %49, %arg19 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %130 = xegpu.dpas %102, %50, %129 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %131 = xegpu.dpas %110, %47, %arg20 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %132 = xegpu.dpas %118, %48, %131 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %133 = xegpu.dpas %110, %49, %arg21 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %134 = xegpu.dpas %118, %50, %133 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %135 = xegpu.update_nd_offset %arg4, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %136 = xegpu.update_nd_offset %arg5, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %137 = xegpu.update_nd_offset %arg6, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %138 = xegpu.update_nd_offset %arg7, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %139 = xegpu.update_nd_offset %arg8, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %140 = xegpu.update_nd_offset %arg9, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %141 = xegpu.update_nd_offset %arg10, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %142 = xegpu.update_nd_offset %arg11, [%c32, %c0] : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %143 = xegpu.update_nd_offset %arg12, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      %144 = xegpu.update_nd_offset %arg13, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
      scf.yield %135, %136, %137, %138, %139, %140, %141, %142, %143, %144, %120, %122, %124, %126, %128, %130, %132, %134 : !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x8xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>
    }
    xegpu.store_nd %36#10, %2 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#11, %4 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#12, %6 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#13, %7 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#14, %9 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#15, %10 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#16, %12 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %36#17, %13 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    gpu.return
  }
}



DpasOp: %1061 = "xegpu.dpas"(%716, %659, <<UNKNOWN SSA VALUE>>) : (vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32>) -> vector<8x16xf32>
   lhs: %742 = "builtin.unrealized_conversion_cast"(%741) : (vector<8x16xf16>) -> vector<128xf16>
   rhs: %685 = "builtin.unrealized_conversion_cast"(%684) : (vector<8x16x2xf16>) -> vector<256xf16>


DpasOp: %1073 = "xegpu.dpas"(%767, %661, %1072) : (vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32>) -> vector<8x16xf32>
   lhs: %793 = "builtin.unrealized_conversion_cast"(%792) : (vector<8x16xf16>) -> vector<128xf16>
   rhs: %687 = "builtin.unrealized_conversion_cast"(%686) : (vector<8x16x2xf16>) -> vector<256xf16>


DpasOp: %1084 = "xegpu.dpas"(%718, %663, <<UNKNOWN SSA VALUE>>) : (vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32>) -> vector<8x16xf32>
   lhs: %744 = "builtin.unrealized_conversion_cast"(%743) : (vector<8x16xf16>) -> vector<128xf16>
   rhs: %689 = "builtin.unrealized_conversion_cast"(%688) : (vector<8x16x2xf16>) -> vector<256xf16>


DpasOp: %1095 = "xegpu.dpas"(%769, %665, %1094) : (vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32>) -> vector<8x16xf32>
   lhs: %795 = "builtin.unrealized_conversion_cast"(%794) : (vector<8x16xf16>) -> vector<128xf16>
   rhs: %691 = "builtin.unrealized_conversion_cast"(%690) : (vector<8x16x2xf16>) -> vector<256xf16>


DpasOp: %1106 = "xegpu.dpas"(%819, %659, <<UNKNOWN SSA VALUE>>) : (vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32>) -> vector<8x16xf32>
   lhs: %845 = "builtin.unrealized_conversion_cast"(%844) : (vector<8x16xf16>) -> vector<128xf16>
   rhs: %685 = "builtin.unrealized_conversion_cast"(%684) : (vector<8x16x2xf16>) -> vector<256xf16>


DpasOp: %1117 = "xegpu.dpas"(%869, %661, %1116) : (vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32>) -> vector<8x16xf32>
   lhs: %895 = "builtin.unrealized_conversion_cast"(%894) : (vector<8x16xf16>) -> vector<128xf16>
   rhs: %687 = "builtin.unrealized_conversion_cast"(%686) : (vector<8x16x2xf16>) -> vector<256xf16>


DpasOp: %1127 = "xegpu.dpas"(%819, %663, <<UNKNOWN SSA VALUE>>) : (vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32>) -> vector<8x16xf32>
   lhs: %845 = "builtin.unrealized_conversion_cast"(%844) : (vector<8x16xf16>) -> vector<128xf16>
   rhs: %689 = "builtin.unrealized_conversion_cast"(%688) : (vector<8x16x2xf16>) -> vector<256xf16>


DpasOp: %1137 = "xegpu.dpas"(%869, %665, %1136) : (vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32>) -> vector<8x16xf32>
   lhs: %895 = "builtin.unrealized_conversion_cast"(%894) : (vector<8x16xf16>) -> vector<128xf16>
   rhs: %691 = "builtin.unrealized_conversion_cast"(%690) : (vector<8x16x2xf16>) -> vector<256xf16>


DpasOp: %1148 = "xegpu.dpas"(%919, %659, <<UNKNOWN SSA VALUE>>) : (vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32>) -> vector<8x16xf32>
   lhs: %945 = "builtin.unrealized_conversion_cast"(%944) : (vector<8x16xf16>) -> vector<128xf16>
   rhs: %685 = "builtin.unrealized_conversion_cast"(%684) : (vector<8x16x2xf16>) -> vector<256xf16>


DpasOp: %1159 = "xegpu.dpas"(%969, %661, %1158) : (vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32>) -> vector<8x16xf32>
   lhs: %995 = "builtin.unrealized_conversion_cast"(%994) : (vector<8x16xf16>) -> vector<128xf16>
   rhs: %687 = "builtin.unrealized_conversion_cast"(%686) : (vector<8x16x2xf16>) -> vector<256xf16>


DpasOp: %1169 = "xegpu.dpas"(%919, %663, <<UNKNOWN SSA VALUE>>) : (vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32>) -> vector<8x16xf32>
   lhs: %945 = "builtin.unrealized_conversion_cast"(%944) : (vector<8x16xf16>) -> vector<128xf16>
   rhs: %689 = "builtin.unrealized_conversion_cast"(%688) : (vector<8x16x2xf16>) -> vector<256xf16>


DpasOp: %1179 = "xegpu.dpas"(%969, %665, %1178) : (vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32>) -> vector<8x16xf32>
   lhs: %995 = "builtin.unrealized_conversion_cast"(%994) : (vector<8x16xf16>) -> vector<128xf16>
   rhs: %691 = "builtin.unrealized_conversion_cast"(%690) : (vector<8x16x2xf16>) -> vector<256xf16>


DpasOp: %1190 = "xegpu.dpas"(%1019, %659, <<UNKNOWN SSA VALUE>>) : (vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32>) -> vector<8x16xf32>
   lhs: %1045 = "builtin.unrealized_conversion_cast"(%1044) : (vector<8x16xf16>) -> vector<128xf16>
   rhs: %685 = "builtin.unrealized_conversion_cast"(%684) : (vector<8x16x2xf16>) -> vector<256xf16>


DpasOp: %1201 = "xegpu.dpas"(%1069, %661, %1200) : (vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32>) -> vector<8x16xf32>
   lhs: %1095 = "builtin.unrealized_conversion_cast"(%1094) : (vector<8x16xf16>) -> vector<128xf16>
   rhs: %687 = "builtin.unrealized_conversion_cast"(%686) : (vector<8x16x2xf16>) -> vector<256xf16>


DpasOp: %1211 = "xegpu.dpas"(%1019, %663, <<UNKNOWN SSA VALUE>>) : (vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32>) -> vector<8x16xf32>
   lhs: %1045 = "builtin.unrealized_conversion_cast"(%1044) : (vector<8x16xf16>) -> vector<128xf16>
   rhs: %689 = "builtin.unrealized_conversion_cast"(%688) : (vector<8x16x2xf16>) -> vector<256xf16>


DpasOp: %1221 = "xegpu.dpas"(%1069, %665, %1220) : (vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32>) -> vector<8x16xf32>
   lhs: %1095 = "builtin.unrealized_conversion_cast"(%1094) : (vector<8x16xf16>) -> vector<128xf16>
   rhs: %691 = "builtin.unrealized_conversion_cast"(%690) : (vector<8x16x2xf16>) -> vector<256xf16>
// -----// IR Dump After ConvertXeGPUToVC (convert-xegpu-to-vc) //----- //
gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  func.func private @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32", linkage_type = <Import>>}
  func.func private @llvm.genx.dpas2.v128f32.v128i32.v64i32(vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas2.v128f32.v128i32.v64i32", linkage_type = <Import>>}
  func.func private @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32", linkage_type = <Import>>}
  func.func private @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32", linkage_type = <Import>>}
  func.func private @llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf16>) -> vector<512xf16> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8", linkage_type = <Import>>}
  func.func private @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8", linkage_type = <Import>>}
  func.func private @llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf32>) -> vector<512xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8", linkage_type = <Import>>}
  gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %cst = arith.constant dense<true> : vector<8xi1>
    %cst_0 = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56]> : vector<8xindex>
    %c64 = arith.constant 64 : index
    %c24 = arith.constant 24 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %0 = arith.muli %block_id_x, %c32 : index
    %1 = arith.muli %block_id_y, %c32 : index
    %intptr = memref.extract_aligned_pointer_as_index %arg2 : memref<1024x1024xf32> -> index
    %c1024_1 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c32_2 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %2 = arith.index_castui %intptr : index to i64
    %cst_3 = arith.constant dense<0> : vector<8xi64>
    %3 = vector.insert %2, %cst_3 [0] : i64 into vector<8xi64>
    %4 = vector.bitcast %3 : vector<8xi64> to vector<16xi32>
    %c4095_i32 = arith.constant 4095 : i32
    %c1023_i32 = arith.constant 1023 : i32
    %c4095_i32_4 = arith.constant 4095 : i32
    %5 = vector.insert %c4095_i32, %4 [2] : i32 into vector<16xi32>
    %6 = vector.insert %c1023_i32, %5 [3] : i32 into vector<16xi32>
    %7 = vector.insert %c4095_i32_4, %6 [4] : i32 into vector<16xi32>
    %8 = arith.index_castui %1 : index to i32
    %c1_i32 = arith.constant 1 : i32
    %9 = arith.index_castui %0 : index to i32
    %c1_i32_5 = arith.constant 1 : i32
    %10 = vector.insert %8, %7 [5] : i32 into vector<16xi32>
    %11 = vector.insert %9, %10 [6] : i32 into vector<16xi32>
    %c1807_i32 = arith.constant 1807 : i32
    %12 = vector.insert %c1807_i32, %11 [7] : i32 into vector<16xi32>
    %13 = arith.addi %1, %c16 : index
    %intptr_6 = memref.extract_aligned_pointer_as_index %arg2 : memref<1024x1024xf32> -> index
    %c1024_7 = arith.constant 1024 : index
    %c1_8 = arith.constant 1 : index
    %c32_9 = arith.constant 32 : index
    %c4_10 = arith.constant 4 : index
    %14 = arith.index_castui %intptr_6 : index to i64
    %cst_11 = arith.constant dense<0> : vector<8xi64>
    %15 = vector.insert %14, %cst_11 [0] : i64 into vector<8xi64>
    %16 = vector.bitcast %15 : vector<8xi64> to vector<16xi32>
    %c4095_i32_12 = arith.constant 4095 : i32
    %c1023_i32_13 = arith.constant 1023 : i32
    %c4095_i32_14 = arith.constant 4095 : i32
    %17 = vector.insert %c4095_i32_12, %16 [2] : i32 into vector<16xi32>
    %18 = vector.insert %c1023_i32_13, %17 [3] : i32 into vector<16xi32>
    %19 = vector.insert %c4095_i32_14, %18 [4] : i32 into vector<16xi32>
    %20 = arith.index_castui %13 : index to i32
    %c1_i32_15 = arith.constant 1 : i32
    %21 = arith.index_castui %0 : index to i32
    %c1_i32_16 = arith.constant 1 : i32
    %22 = vector.insert %20, %19 [5] : i32 into vector<16xi32>
    %23 = vector.insert %21, %22 [6] : i32 into vector<16xi32>
    %c1807_i32_17 = arith.constant 1807 : i32
    %24 = vector.insert %c1807_i32_17, %23 [7] : i32 into vector<16xi32>
    %25 = arith.addi %0, %c8 : index
    %intptr_18 = memref.extract_aligned_pointer_as_index %arg2 : memref<1024x1024xf32> -> index
    %c1024_19 = arith.constant 1024 : index
    %c1_20 = arith.constant 1 : index
    %c32_21 = arith.constant 32 : index
    %c4_22 = arith.constant 4 : index
    %26 = arith.index_castui %intptr_18 : index to i64
    %cst_23 = arith.constant dense<0> : vector<8xi64>
    %27 = vector.insert %26, %cst_23 [0] : i64 into vector<8xi64>
    %28 = vector.bitcast %27 : vector<8xi64> to vector<16xi32>
    %c4095_i32_24 = arith.constant 4095 : i32
    %c1023_i32_25 = arith.constant 1023 : i32
    %c4095_i32_26 = arith.constant 4095 : i32
    %29 = vector.insert %c4095_i32_24, %28 [2] : i32 into vector<16xi32>
    %30 = vector.insert %c1023_i32_25, %29 [3] : i32 into vector<16xi32>
    %31 = vector.insert %c4095_i32_26, %30 [4] : i32 into vector<16xi32>
    %32 = arith.index_castui %1 : index to i32
    %c1_i32_27 = arith.constant 1 : i32
    %33 = arith.index_castui %25 : index to i32
    %c1_i32_28 = arith.constant 1 : i32
    %34 = vector.insert %32, %31 [5] : i32 into vector<16xi32>
    %35 = vector.insert %33, %34 [6] : i32 into vector<16xi32>
    %c1807_i32_29 = arith.constant 1807 : i32
    %36 = vector.insert %c1807_i32_29, %35 [7] : i32 into vector<16xi32>
    %intptr_30 = memref.extract_aligned_pointer_as_index %arg2 : memref<1024x1024xf32> -> index
    %c1024_31 = arith.constant 1024 : index
    %c1_32 = arith.constant 1 : index
    %c32_33 = arith.constant 32 : index
    %c4_34 = arith.constant 4 : index
    %37 = arith.index_castui %intptr_30 : index to i64
    %cst_35 = arith.constant dense<0> : vector<8xi64>
    %38 = vector.insert %37, %cst_35 [0] : i64 into vector<8xi64>
    %39 = vector.bitcast %38 : vector<8xi64> to vector<16xi32>
    %c4095_i32_36 = arith.constant 4095 : i32
    %c1023_i32_37 = arith.constant 1023 : i32
    %c4095_i32_38 = arith.constant 4095 : i32
    %40 = vector.insert %c4095_i32_36, %39 [2] : i32 into vector<16xi32>
    %41 = vector.insert %c1023_i32_37, %40 [3] : i32 into vector<16xi32>
    %42 = vector.insert %c4095_i32_38, %41 [4] : i32 into vector<16xi32>
    %43 = arith.index_castui %13 : index to i32
    %c1_i32_39 = arith.constant 1 : i32
    %44 = arith.index_castui %25 : index to i32
    %c1_i32_40 = arith.constant 1 : i32
    %45 = vector.insert %43, %42 [5] : i32 into vector<16xi32>
    %46 = vector.insert %44, %45 [6] : i32 into vector<16xi32>
    %c1807_i32_41 = arith.constant 1807 : i32
    %47 = vector.insert %c1807_i32_41, %46 [7] : i32 into vector<16xi32>
    %48 = arith.addi %0, %c16 : index
    %intptr_42 = memref.extract_aligned_pointer_as_index %arg2 : memref<1024x1024xf32> -> index
    %c1024_43 = arith.constant 1024 : index
    %c1_44 = arith.constant 1 : index
    %c32_45 = arith.constant 32 : index
    %c4_46 = arith.constant 4 : index
    %49 = arith.index_castui %intptr_42 : index to i64
    %cst_47 = arith.constant dense<0> : vector<8xi64>
    %50 = vector.insert %49, %cst_47 [0] : i64 into vector<8xi64>
    %51 = vector.bitcast %50 : vector<8xi64> to vector<16xi32>
    %c4095_i32_48 = arith.constant 4095 : i32
    %c1023_i32_49 = arith.constant 1023 : i32
    %c4095_i32_50 = arith.constant 4095 : i32
    %52 = vector.insert %c4095_i32_48, %51 [2] : i32 into vector<16xi32>
    %53 = vector.insert %c1023_i32_49, %52 [3] : i32 into vector<16xi32>
    %54 = vector.insert %c4095_i32_50, %53 [4] : i32 into vector<16xi32>
    %55 = arith.index_castui %1 : index to i32
    %c1_i32_51 = arith.constant 1 : i32
    %56 = arith.index_castui %48 : index to i32
    %c1_i32_52 = arith.constant 1 : i32
    %57 = vector.insert %55, %54 [5] : i32 into vector<16xi32>
    %58 = vector.insert %56, %57 [6] : i32 into vector<16xi32>
    %c1807_i32_53 = arith.constant 1807 : i32
    %59 = vector.insert %c1807_i32_53, %58 [7] : i32 into vector<16xi32>
    %intptr_54 = memref.extract_aligned_pointer_as_index %arg2 : memref<1024x1024xf32> -> index
    %c1024_55 = arith.constant 1024 : index
    %c1_56 = arith.constant 1 : index
    %c32_57 = arith.constant 32 : index
    %c4_58 = arith.constant 4 : index
    %60 = arith.index_castui %intptr_54 : index to i64
    %cst_59 = arith.constant dense<0> : vector<8xi64>
    %61 = vector.insert %60, %cst_59 [0] : i64 into vector<8xi64>
    %62 = vector.bitcast %61 : vector<8xi64> to vector<16xi32>
    %c4095_i32_60 = arith.constant 4095 : i32
    %c1023_i32_61 = arith.constant 1023 : i32
    %c4095_i32_62 = arith.constant 4095 : i32
    %63 = vector.insert %c4095_i32_60, %62 [2] : i32 into vector<16xi32>
    %64 = vector.insert %c1023_i32_61, %63 [3] : i32 into vector<16xi32>
    %65 = vector.insert %c4095_i32_62, %64 [4] : i32 into vector<16xi32>
    %66 = arith.index_castui %13 : index to i32
    %c1_i32_63 = arith.constant 1 : i32
    %67 = arith.index_castui %48 : index to i32
    %c1_i32_64 = arith.constant 1 : i32
    %68 = vector.insert %66, %65 [5] : i32 into vector<16xi32>
    %69 = vector.insert %67, %68 [6] : i32 into vector<16xi32>
    %c1807_i32_65 = arith.constant 1807 : i32
    %70 = vector.insert %c1807_i32_65, %69 [7] : i32 into vector<16xi32>
    %71 = arith.addi %0, %c24 : index
    %intptr_66 = memref.extract_aligned_pointer_as_index %arg2 : memref<1024x1024xf32> -> index
    %c1024_67 = arith.constant 1024 : index
    %c1_68 = arith.constant 1 : index
    %c32_69 = arith.constant 32 : index
    %c4_70 = arith.constant 4 : index
    %72 = arith.index_castui %intptr_66 : index to i64
    %cst_71 = arith.constant dense<0> : vector<8xi64>
    %73 = vector.insert %72, %cst_71 [0] : i64 into vector<8xi64>
    %74 = vector.bitcast %73 : vector<8xi64> to vector<16xi32>
    %c4095_i32_72 = arith.constant 4095 : i32
    %c1023_i32_73 = arith.constant 1023 : i32
    %c4095_i32_74 = arith.constant 4095 : i32
    %75 = vector.insert %c4095_i32_72, %74 [2] : i32 into vector<16xi32>
    %76 = vector.insert %c1023_i32_73, %75 [3] : i32 into vector<16xi32>
    %77 = vector.insert %c4095_i32_74, %76 [4] : i32 into vector<16xi32>
    %78 = arith.index_castui %1 : index to i32
    %c1_i32_75 = arith.constant 1 : i32
    %79 = arith.index_castui %71 : index to i32
    %c1_i32_76 = arith.constant 1 : i32
    %80 = vector.insert %78, %77 [5] : i32 into vector<16xi32>
    %81 = vector.insert %79, %80 [6] : i32 into vector<16xi32>
    %c1807_i32_77 = arith.constant 1807 : i32
    %82 = vector.insert %c1807_i32_77, %81 [7] : i32 into vector<16xi32>
    %intptr_78 = memref.extract_aligned_pointer_as_index %arg2 : memref<1024x1024xf32> -> index
    %c1024_79 = arith.constant 1024 : index
    %c1_80 = arith.constant 1 : index
    %c32_81 = arith.constant 32 : index
    %c4_82 = arith.constant 4 : index
    %83 = arith.index_castui %intptr_78 : index to i64
    %cst_83 = arith.constant dense<0> : vector<8xi64>
    %84 = vector.insert %83, %cst_83 [0] : i64 into vector<8xi64>
    %85 = vector.bitcast %84 : vector<8xi64> to vector<16xi32>
    %c4095_i32_84 = arith.constant 4095 : i32
    %c1023_i32_85 = arith.constant 1023 : i32
    %c4095_i32_86 = arith.constant 4095 : i32
    %86 = vector.insert %c4095_i32_84, %85 [2] : i32 into vector<16xi32>
    %87 = vector.insert %c1023_i32_85, %86 [3] : i32 into vector<16xi32>
    %88 = vector.insert %c4095_i32_86, %87 [4] : i32 into vector<16xi32>
    %89 = arith.index_castui %13 : index to i32
    %c1_i32_87 = arith.constant 1 : i32
    %90 = arith.index_castui %71 : index to i32
    %c1_i32_88 = arith.constant 1 : i32
    %91 = vector.insert %89, %88 [5] : i32 into vector<16xi32>
    %92 = vector.insert %90, %91 [6] : i32 into vector<16xi32>
    %c1807_i32_89 = arith.constant 1807 : i32
    %93 = vector.insert %c1807_i32_89, %92 [7] : i32 into vector<16xi32>
    %intptr_90 = memref.extract_aligned_pointer_as_index %arg2 : memref<1024x1024xf32> -> index
    %c1024_91 = arith.constant 1024 : index
    %c1_92 = arith.constant 1 : index
    %c32_93 = arith.constant 32 : index
    %c4_94 = arith.constant 4 : index
    %94 = arith.index_castui %intptr_90 : index to i64
    %cst_95 = arith.constant dense<0> : vector<8xi64>
    %95 = vector.insert %94, %cst_95 [0] : i64 into vector<8xi64>
    %96 = vector.bitcast %95 : vector<8xi64> to vector<16xi32>
    %c4095_i32_96 = arith.constant 4095 : i32
    %c1023_i32_97 = arith.constant 1023 : i32
    %c4095_i32_98 = arith.constant 4095 : i32
    %97 = vector.insert %c4095_i32_96, %96 [2] : i32 into vector<16xi32>
    %98 = vector.insert %c1023_i32_97, %97 [3] : i32 into vector<16xi32>
    %99 = vector.insert %c4095_i32_98, %98 [4] : i32 into vector<16xi32>
    %100 = arith.index_castui %1 : index to i32
    %c1_i32_99 = arith.constant 1 : i32
    %101 = arith.index_castui %0 : index to i32
    %c1_i32_100 = arith.constant 1 : i32
    %102 = vector.insert %100, %99 [5] : i32 into vector<16xi32>
    %103 = vector.insert %101, %102 [6] : i32 into vector<16xi32>
    %c7951_i32 = arith.constant 7951 : i32
    %104 = vector.insert %c7951_i32, %103 [7] : i32 into vector<16xi32>
    %intptr_101 = memref.extract_aligned_pointer_as_index %arg2 : memref<1024x1024xf32> -> index
    %c1024_102 = arith.constant 1024 : index
    %c1_103 = arith.constant 1 : index
    %c32_104 = arith.constant 32 : index
    %c4_105 = arith.constant 4 : index
    %105 = arith.index_castui %intptr_101 : index to i64
    %cst_106 = arith.constant dense<0> : vector<8xi64>
    %106 = vector.insert %105, %cst_106 [0] : i64 into vector<8xi64>
    %107 = vector.bitcast %106 : vector<8xi64> to vector<16xi32>
    %c4095_i32_107 = arith.constant 4095 : i32
    %c1023_i32_108 = arith.constant 1023 : i32
    %c4095_i32_109 = arith.constant 4095 : i32
    %108 = vector.insert %c4095_i32_107, %107 [2] : i32 into vector<16xi32>
    %109 = vector.insert %c1023_i32_108, %108 [3] : i32 into vector<16xi32>
    %110 = vector.insert %c4095_i32_109, %109 [4] : i32 into vector<16xi32>
    %111 = arith.index_castui %13 : index to i32
    %c1_i32_110 = arith.constant 1 : i32
    %112 = arith.index_castui %0 : index to i32
    %c1_i32_111 = arith.constant 1 : i32
    %113 = vector.insert %111, %110 [5] : i32 into vector<16xi32>
    %114 = vector.insert %112, %113 [6] : i32 into vector<16xi32>
    %c7951_i32_112 = arith.constant 7951 : i32
    %115 = vector.insert %c7951_i32_112, %114 [7] : i32 into vector<16xi32>
    %cst_113 = arith.constant dense<0.000000e+00> : vector<512xf32>
    %true = arith.constant true
    %c2_i8 = arith.constant 2 : i8
    %c2_i8_114 = arith.constant 2 : i8
    %116 = vector.from_elements %c2_i8, %c2_i8_114 : vector<2xi8>
    %c1_i8 = arith.constant 1 : i8
    %c16_i16 = arith.constant 16 : i16
    %c32_i16 = arith.constant 32 : i16
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_115 = arith.constant 0 : i32
    %117 = func.call @llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8(%true, %116, %c1_i8, %c16_i16, %c32_i16, %104, %c0_i32, %c0_i32_115, %cst_113) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf32>) -> vector<512xf32>
    %118 = vector.shape_cast %117 : vector<512xf32> to vector<32x16xf32>
    %cst_116 = arith.constant dense<0.000000e+00> : vector<512xf32>
    %true_117 = arith.constant true
    %c2_i8_118 = arith.constant 2 : i8
    %c2_i8_119 = arith.constant 2 : i8
    %119 = vector.from_elements %c2_i8_118, %c2_i8_119 : vector<2xi8>
    %c1_i8_120 = arith.constant 1 : i8
    %c16_i16_121 = arith.constant 16 : i16
    %c32_i16_122 = arith.constant 32 : i16
    %c0_i32_123 = arith.constant 0 : i32
    %c0_i32_124 = arith.constant 0 : i32
    %120 = func.call @llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8(%true_117, %119, %c1_i8_120, %c16_i16_121, %c32_i16_122, %115, %c0_i32_123, %c0_i32_124, %cst_116) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf32>) -> vector<512xf32>
    %121 = vector.shape_cast %120 : vector<512xf32> to vector<32x16xf32>
    %122 = vector.extract_strided_slice %118 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %123 = vector.shape_cast %122 : vector<8x16xf32> to vector<128xf32>
    %124 = vector.extract_strided_slice %118 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %125 = vector.shape_cast %124 : vector<8x16xf32> to vector<128xf32>
    %126 = vector.extract_strided_slice %118 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %127 = vector.shape_cast %126 : vector<8x16xf32> to vector<128xf32>
    %128 = vector.extract_strided_slice %118 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %129 = vector.shape_cast %128 : vector<8x16xf32> to vector<128xf32>
    %130 = vector.extract_strided_slice %121 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %131 = vector.shape_cast %130 : vector<8x16xf32> to vector<128xf32>
    %132 = vector.extract_strided_slice %121 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %133 = vector.shape_cast %132 : vector<8x16xf32> to vector<128xf32>
    %134 = vector.extract_strided_slice %121 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %135 = vector.shape_cast %134 : vector<8x16xf32> to vector<128xf32>
    %136 = vector.extract_strided_slice %121 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %137 = vector.shape_cast %136 : vector<8x16xf32> to vector<128xf32>
    %intptr_125 = memref.extract_aligned_pointer_as_index %arg0 : memref<1024x1024xf16> -> index
    %c1024_126 = arith.constant 1024 : index
    %c1_127 = arith.constant 1 : index
    %c16_128 = arith.constant 16 : index
    %c2 = arith.constant 2 : index
    %138 = arith.index_castui %intptr_125 : index to i64
    %cst_129 = arith.constant dense<0> : vector<8xi64>
    %139 = vector.insert %138, %cst_129 [0] : i64 into vector<8xi64>
    %140 = vector.bitcast %139 : vector<8xi64> to vector<16xi32>
    %c2047_i32 = arith.constant 2047 : i32
    %c1023_i32_130 = arith.constant 1023 : i32
    %c2047_i32_131 = arith.constant 2047 : i32
    %141 = vector.insert %c2047_i32, %140 [2] : i32 into vector<16xi32>
    %142 = vector.insert %c1023_i32_130, %141 [3] : i32 into vector<16xi32>
    %143 = vector.insert %c2047_i32_131, %142 [4] : i32 into vector<16xi32>
    %144 = arith.index_castui %0 : index to i32
    %c1_i32_132 = arith.constant 1 : i32
    %145 = arith.index_castui %c0 : index to i32
    %c1_i32_133 = arith.constant 1 : i32
    %146 = vector.insert %144, %143 [5] : i32 into vector<16xi32>
    %147 = vector.insert %145, %146 [6] : i32 into vector<16xi32>
    %c3847_i32 = arith.constant 3847 : i32
    %148 = vector.insert %c3847_i32, %147 [7] : i32 into vector<16xi32>
    %intptr_134 = memref.extract_aligned_pointer_as_index %arg0 : memref<1024x1024xf16> -> index
    %c1024_135 = arith.constant 1024 : index
    %c1_136 = arith.constant 1 : index
    %c16_137 = arith.constant 16 : index
    %c2_138 = arith.constant 2 : index
    %149 = arith.index_castui %intptr_134 : index to i64
    %cst_139 = arith.constant dense<0> : vector<8xi64>
    %150 = vector.insert %149, %cst_139 [0] : i64 into vector<8xi64>
    %151 = vector.bitcast %150 : vector<8xi64> to vector<16xi32>
    %c2047_i32_140 = arith.constant 2047 : i32
    %c1023_i32_141 = arith.constant 1023 : i32
    %c2047_i32_142 = arith.constant 2047 : i32
    %152 = vector.insert %c2047_i32_140, %151 [2] : i32 into vector<16xi32>
    %153 = vector.insert %c1023_i32_141, %152 [3] : i32 into vector<16xi32>
    %154 = vector.insert %c2047_i32_142, %153 [4] : i32 into vector<16xi32>
    %155 = arith.index_castui %25 : index to i32
    %c1_i32_143 = arith.constant 1 : i32
    %156 = arith.index_castui %c0 : index to i32
    %c1_i32_144 = arith.constant 1 : i32
    %157 = vector.insert %155, %154 [5] : i32 into vector<16xi32>
    %158 = vector.insert %156, %157 [6] : i32 into vector<16xi32>
    %c3847_i32_145 = arith.constant 3847 : i32
    %159 = vector.insert %c3847_i32_145, %158 [7] : i32 into vector<16xi32>
    %intptr_146 = memref.extract_aligned_pointer_as_index %arg0 : memref<1024x1024xf16> -> index
    %c1024_147 = arith.constant 1024 : index
    %c1_148 = arith.constant 1 : index
    %c16_149 = arith.constant 16 : index
    %c2_150 = arith.constant 2 : index
    %160 = arith.index_castui %intptr_146 : index to i64
    %cst_151 = arith.constant dense<0> : vector<8xi64>
    %161 = vector.insert %160, %cst_151 [0] : i64 into vector<8xi64>
    %162 = vector.bitcast %161 : vector<8xi64> to vector<16xi32>
    %c2047_i32_152 = arith.constant 2047 : i32
    %c1023_i32_153 = arith.constant 1023 : i32
    %c2047_i32_154 = arith.constant 2047 : i32
    %163 = vector.insert %c2047_i32_152, %162 [2] : i32 into vector<16xi32>
    %164 = vector.insert %c1023_i32_153, %163 [3] : i32 into vector<16xi32>
    %165 = vector.insert %c2047_i32_154, %164 [4] : i32 into vector<16xi32>
    %166 = arith.index_castui %48 : index to i32
    %c1_i32_155 = arith.constant 1 : i32
    %167 = arith.index_castui %c0 : index to i32
    %c1_i32_156 = arith.constant 1 : i32
    %168 = vector.insert %166, %165 [5] : i32 into vector<16xi32>
    %169 = vector.insert %167, %168 [6] : i32 into vector<16xi32>
    %c3847_i32_157 = arith.constant 3847 : i32
    %170 = vector.insert %c3847_i32_157, %169 [7] : i32 into vector<16xi32>
    %intptr_158 = memref.extract_aligned_pointer_as_index %arg0 : memref<1024x1024xf16> -> index
    %c1024_159 = arith.constant 1024 : index
    %c1_160 = arith.constant 1 : index
    %c16_161 = arith.constant 16 : index
    %c2_162 = arith.constant 2 : index
    %171 = arith.index_castui %intptr_158 : index to i64
    %cst_163 = arith.constant dense<0> : vector<8xi64>
    %172 = vector.insert %171, %cst_163 [0] : i64 into vector<8xi64>
    %173 = vector.bitcast %172 : vector<8xi64> to vector<16xi32>
    %c2047_i32_164 = arith.constant 2047 : i32
    %c1023_i32_165 = arith.constant 1023 : i32
    %c2047_i32_166 = arith.constant 2047 : i32
    %174 = vector.insert %c2047_i32_164, %173 [2] : i32 into vector<16xi32>
    %175 = vector.insert %c1023_i32_165, %174 [3] : i32 into vector<16xi32>
    %176 = vector.insert %c2047_i32_166, %175 [4] : i32 into vector<16xi32>
    %177 = arith.index_castui %71 : index to i32
    %c1_i32_167 = arith.constant 1 : i32
    %178 = arith.index_castui %c0 : index to i32
    %c1_i32_168 = arith.constant 1 : i32
    %179 = vector.insert %177, %176 [5] : i32 into vector<16xi32>
    %180 = vector.insert %178, %179 [6] : i32 into vector<16xi32>
    %c3847_i32_169 = arith.constant 3847 : i32
    %181 = vector.insert %c3847_i32_169, %180 [7] : i32 into vector<16xi32>
    %intptr_170 = memref.extract_aligned_pointer_as_index %arg0 : memref<1024x1024xf16> -> index
    %c1024_171 = arith.constant 1024 : index
    %c1_172 = arith.constant 1 : index
    %c16_173 = arith.constant 16 : index
    %c2_174 = arith.constant 2 : index
    %182 = arith.index_castui %intptr_170 : index to i64
    %cst_175 = arith.constant dense<0> : vector<8xi64>
    %183 = vector.insert %182, %cst_175 [0] : i64 into vector<8xi64>
    %184 = vector.bitcast %183 : vector<8xi64> to vector<16xi32>
    %c2047_i32_176 = arith.constant 2047 : i32
    %c1023_i32_177 = arith.constant 1023 : i32
    %c2047_i32_178 = arith.constant 2047 : i32
    %185 = vector.insert %c2047_i32_176, %184 [2] : i32 into vector<16xi32>
    %186 = vector.insert %c1023_i32_177, %185 [3] : i32 into vector<16xi32>
    %187 = vector.insert %c2047_i32_178, %186 [4] : i32 into vector<16xi32>
    %188 = arith.index_castui %0 : index to i32
    %c1_i32_179 = arith.constant 1 : i32
    %189 = arith.index_castui %c16 : index to i32
    %c1_i32_180 = arith.constant 1 : i32
    %190 = vector.insert %188, %187 [5] : i32 into vector<16xi32>
    %191 = vector.insert %189, %190 [6] : i32 into vector<16xi32>
    %c3847_i32_181 = arith.constant 3847 : i32
    %192 = vector.insert %c3847_i32_181, %191 [7] : i32 into vector<16xi32>
    %intptr_182 = memref.extract_aligned_pointer_as_index %arg0 : memref<1024x1024xf16> -> index
    %c1024_183 = arith.constant 1024 : index
    %c1_184 = arith.constant 1 : index
    %c16_185 = arith.constant 16 : index
    %c2_186 = arith.constant 2 : index
    %193 = arith.index_castui %intptr_182 : index to i64
    %cst_187 = arith.constant dense<0> : vector<8xi64>
    %194 = vector.insert %193, %cst_187 [0] : i64 into vector<8xi64>
    %195 = vector.bitcast %194 : vector<8xi64> to vector<16xi32>
    %c2047_i32_188 = arith.constant 2047 : i32
    %c1023_i32_189 = arith.constant 1023 : i32
    %c2047_i32_190 = arith.constant 2047 : i32
    %196 = vector.insert %c2047_i32_188, %195 [2] : i32 into vector<16xi32>
    %197 = vector.insert %c1023_i32_189, %196 [3] : i32 into vector<16xi32>
    %198 = vector.insert %c2047_i32_190, %197 [4] : i32 into vector<16xi32>
    %199 = arith.index_castui %25 : index to i32
    %c1_i32_191 = arith.constant 1 : i32
    %200 = arith.index_castui %c16 : index to i32
    %c1_i32_192 = arith.constant 1 : i32
    %201 = vector.insert %199, %198 [5] : i32 into vector<16xi32>
    %202 = vector.insert %200, %201 [6] : i32 into vector<16xi32>
    %c3847_i32_193 = arith.constant 3847 : i32
    %203 = vector.insert %c3847_i32_193, %202 [7] : i32 into vector<16xi32>
    %intptr_194 = memref.extract_aligned_pointer_as_index %arg0 : memref<1024x1024xf16> -> index
    %c1024_195 = arith.constant 1024 : index
    %c1_196 = arith.constant 1 : index
    %c16_197 = arith.constant 16 : index
    %c2_198 = arith.constant 2 : index
    %204 = arith.index_castui %intptr_194 : index to i64
    %cst_199 = arith.constant dense<0> : vector<8xi64>
    %205 = vector.insert %204, %cst_199 [0] : i64 into vector<8xi64>
    %206 = vector.bitcast %205 : vector<8xi64> to vector<16xi32>
    %c2047_i32_200 = arith.constant 2047 : i32
    %c1023_i32_201 = arith.constant 1023 : i32
    %c2047_i32_202 = arith.constant 2047 : i32
    %207 = vector.insert %c2047_i32_200, %206 [2] : i32 into vector<16xi32>
    %208 = vector.insert %c1023_i32_201, %207 [3] : i32 into vector<16xi32>
    %209 = vector.insert %c2047_i32_202, %208 [4] : i32 into vector<16xi32>
    %210 = arith.index_castui %48 : index to i32
    %c1_i32_203 = arith.constant 1 : i32
    %211 = arith.index_castui %c16 : index to i32
    %c1_i32_204 = arith.constant 1 : i32
    %212 = vector.insert %210, %209 [5] : i32 into vector<16xi32>
    %213 = vector.insert %211, %212 [6] : i32 into vector<16xi32>
    %c3847_i32_205 = arith.constant 3847 : i32
    %214 = vector.insert %c3847_i32_205, %213 [7] : i32 into vector<16xi32>
    %intptr_206 = memref.extract_aligned_pointer_as_index %arg0 : memref<1024x1024xf16> -> index
    %c1024_207 = arith.constant 1024 : index
    %c1_208 = arith.constant 1 : index
    %c16_209 = arith.constant 16 : index
    %c2_210 = arith.constant 2 : index
    %215 = arith.index_castui %intptr_206 : index to i64
    %cst_211 = arith.constant dense<0> : vector<8xi64>
    %216 = vector.insert %215, %cst_211 [0] : i64 into vector<8xi64>
    %217 = vector.bitcast %216 : vector<8xi64> to vector<16xi32>
    %c2047_i32_212 = arith.constant 2047 : i32
    %c1023_i32_213 = arith.constant 1023 : i32
    %c2047_i32_214 = arith.constant 2047 : i32
    %218 = vector.insert %c2047_i32_212, %217 [2] : i32 into vector<16xi32>
    %219 = vector.insert %c1023_i32_213, %218 [3] : i32 into vector<16xi32>
    %220 = vector.insert %c2047_i32_214, %219 [4] : i32 into vector<16xi32>
    %221 = arith.index_castui %71 : index to i32
    %c1_i32_215 = arith.constant 1 : i32
    %222 = arith.index_castui %c16 : index to i32
    %c1_i32_216 = arith.constant 1 : i32
    %223 = vector.insert %221, %220 [5] : i32 into vector<16xi32>
    %224 = vector.insert %222, %223 [6] : i32 into vector<16xi32>
    %c3847_i32_217 = arith.constant 3847 : i32
    %225 = vector.insert %c3847_i32_217, %224 [7] : i32 into vector<16xi32>
    %intptr_218 = memref.extract_aligned_pointer_as_index %arg1 : memref<1024x1024xf16> -> index
    %c1024_219 = arith.constant 1024 : index
    %c1_220 = arith.constant 1 : index
    %c16_221 = arith.constant 16 : index
    %c2_222 = arith.constant 2 : index
    %226 = arith.index_castui %intptr_218 : index to i64
    %cst_223 = arith.constant dense<0> : vector<8xi64>
    %227 = vector.insert %226, %cst_223 [0] : i64 into vector<8xi64>
    %228 = vector.bitcast %227 : vector<8xi64> to vector<16xi32>
    %c2047_i32_224 = arith.constant 2047 : i32
    %c1023_i32_225 = arith.constant 1023 : i32
    %c2047_i32_226 = arith.constant 2047 : i32
    %229 = vector.insert %c2047_i32_224, %228 [2] : i32 into vector<16xi32>
    %230 = vector.insert %c1023_i32_225, %229 [3] : i32 into vector<16xi32>
    %231 = vector.insert %c2047_i32_226, %230 [4] : i32 into vector<16xi32>
    %232 = arith.index_castui %1 : index to i32
    %c1_i32_227 = arith.constant 1 : i32
    %233 = arith.index_castui %c0 : index to i32
    %c1_i32_228 = arith.constant 1 : i32
    %234 = vector.insert %232, %231 [5] : i32 into vector<16xi32>
    %235 = vector.insert %233, %234 [6] : i32 into vector<16xi32>
    %c7951_i32_229 = arith.constant 7951 : i32
    %236 = vector.insert %c7951_i32_229, %235 [7] : i32 into vector<16xi32>
    %intptr_230 = memref.extract_aligned_pointer_as_index %arg1 : memref<1024x1024xf16> -> index
    %c1024_231 = arith.constant 1024 : index
    %c1_232 = arith.constant 1 : index
    %c16_233 = arith.constant 16 : index
    %c2_234 = arith.constant 2 : index
    %237 = arith.index_castui %intptr_230 : index to i64
    %cst_235 = arith.constant dense<0> : vector<8xi64>
    %238 = vector.insert %237, %cst_235 [0] : i64 into vector<8xi64>
    %239 = vector.bitcast %238 : vector<8xi64> to vector<16xi32>
    %c2047_i32_236 = arith.constant 2047 : i32
    %c1023_i32_237 = arith.constant 1023 : i32
    %c2047_i32_238 = arith.constant 2047 : i32
    %240 = vector.insert %c2047_i32_236, %239 [2] : i32 into vector<16xi32>
    %241 = vector.insert %c1023_i32_237, %240 [3] : i32 into vector<16xi32>
    %242 = vector.insert %c2047_i32_238, %241 [4] : i32 into vector<16xi32>
    %243 = arith.index_castui %13 : index to i32
    %c1_i32_239 = arith.constant 1 : i32
    %244 = arith.index_castui %c0 : index to i32
    %c1_i32_240 = arith.constant 1 : i32
    %245 = vector.insert %243, %242 [5] : i32 into vector<16xi32>
    %246 = vector.insert %244, %245 [6] : i32 into vector<16xi32>
    %c7951_i32_241 = arith.constant 7951 : i32
    %247 = vector.insert %c7951_i32_241, %246 [7] : i32 into vector<16xi32>
    %248:18 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %148, %arg5 = %159, %arg6 = %170, %arg7 = %181, %arg8 = %192, %arg9 = %203, %arg10 = %214, %arg11 = %225, %arg12 = %236, %arg13 = %247, %arg14 = %123, %arg15 = %131, %arg16 = %125, %arg17 = %133, %arg18 = %127, %arg19 = %135, %arg20 = %129, %arg21 = %137) -> (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>) {
      %cst_304 = arith.constant dense<0.000000e+00> : vector<128xf16>
      %true_305 = arith.constant true
      %c2_i8_306 = arith.constant 2 : i8
      %c2_i8_307 = arith.constant 2 : i8
      %257 = vector.from_elements %c2_i8_306, %c2_i8_307 : vector<2xi8>
      %c1_i8_308 = arith.constant 1 : i8
      %c8_i16_309 = arith.constant 8 : i16
      %c16_i16_310 = arith.constant 16 : i16
      %c0_i32_311 = arith.constant 0 : i32
      %c0_i32_312 = arith.constant 0 : i32
      %258 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true_305, %257, %c1_i8_308, %c8_i16_309, %c16_i16_310, %arg4, %c0_i32_311, %c0_i32_312, %cst_304) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
      %259 = vector.shape_cast %258 : vector<128xf16> to vector<8x8x2xf16>
      %cst_313 = arith.constant dense<0.000000e+00> : vector<128xf16>
      %true_314 = arith.constant true
      %c2_i8_315 = arith.constant 2 : i8
      %c2_i8_316 = arith.constant 2 : i8
      %260 = vector.from_elements %c2_i8_315, %c2_i8_316 : vector<2xi8>
      %c1_i8_317 = arith.constant 1 : i8
      %c8_i16_318 = arith.constant 8 : i16
      %c16_i16_319 = arith.constant 16 : i16
      %c0_i32_320 = arith.constant 0 : i32
      %c0_i32_321 = arith.constant 0 : i32
      %261 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true_314, %260, %c1_i8_317, %c8_i16_318, %c16_i16_319, %arg5, %c0_i32_320, %c0_i32_321, %cst_313) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
      %262 = vector.shape_cast %261 : vector<128xf16> to vector<8x8x2xf16>
      %cst_322 = arith.constant dense<0.000000e+00> : vector<128xf16>
      %true_323 = arith.constant true
      %c2_i8_324 = arith.constant 2 : i8
      %c2_i8_325 = arith.constant 2 : i8
      %263 = vector.from_elements %c2_i8_324, %c2_i8_325 : vector<2xi8>
      %c1_i8_326 = arith.constant 1 : i8
      %c8_i16_327 = arith.constant 8 : i16
      %c16_i16_328 = arith.constant 16 : i16
      %c0_i32_329 = arith.constant 0 : i32
      %c0_i32_330 = arith.constant 0 : i32
      %264 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true_323, %263, %c1_i8_326, %c8_i16_327, %c16_i16_328, %arg6, %c0_i32_329, %c0_i32_330, %cst_322) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
      %265 = vector.shape_cast %264 : vector<128xf16> to vector<8x8x2xf16>
      %cst_331 = arith.constant dense<0.000000e+00> : vector<128xf16>
      %true_332 = arith.constant true
      %c2_i8_333 = arith.constant 2 : i8
      %c2_i8_334 = arith.constant 2 : i8
      %266 = vector.from_elements %c2_i8_333, %c2_i8_334 : vector<2xi8>
      %c1_i8_335 = arith.constant 1 : i8
      %c8_i16_336 = arith.constant 8 : i16
      %c16_i16_337 = arith.constant 16 : i16
      %c0_i32_338 = arith.constant 0 : i32
      %c0_i32_339 = arith.constant 0 : i32
      %267 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true_332, %266, %c1_i8_335, %c8_i16_336, %c16_i16_337, %arg7, %c0_i32_338, %c0_i32_339, %cst_331) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
      %268 = vector.shape_cast %267 : vector<128xf16> to vector<8x8x2xf16>
      %cst_340 = arith.constant dense<0.000000e+00> : vector<128xf16>
      %true_341 = arith.constant true
      %c2_i8_342 = arith.constant 2 : i8
      %c2_i8_343 = arith.constant 2 : i8
      %269 = vector.from_elements %c2_i8_342, %c2_i8_343 : vector<2xi8>
      %c1_i8_344 = arith.constant 1 : i8
      %c8_i16_345 = arith.constant 8 : i16
      %c16_i16_346 = arith.constant 16 : i16
      %c0_i32_347 = arith.constant 0 : i32
      %c0_i32_348 = arith.constant 0 : i32
      %270 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true_341, %269, %c1_i8_344, %c8_i16_345, %c16_i16_346, %arg8, %c0_i32_347, %c0_i32_348, %cst_340) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
      %271 = vector.shape_cast %270 : vector<128xf16> to vector<8x8x2xf16>
      %cst_349 = arith.constant dense<0.000000e+00> : vector<128xf16>
      %true_350 = arith.constant true
      %c2_i8_351 = arith.constant 2 : i8
      %c2_i8_352 = arith.constant 2 : i8
      %272 = vector.from_elements %c2_i8_351, %c2_i8_352 : vector<2xi8>
      %c1_i8_353 = arith.constant 1 : i8
      %c8_i16_354 = arith.constant 8 : i16
      %c16_i16_355 = arith.constant 16 : i16
      %c0_i32_356 = arith.constant 0 : i32
      %c0_i32_357 = arith.constant 0 : i32
      %273 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true_350, %272, %c1_i8_353, %c8_i16_354, %c16_i16_355, %arg9, %c0_i32_356, %c0_i32_357, %cst_349) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
      %274 = vector.shape_cast %273 : vector<128xf16> to vector<8x8x2xf16>
      %cst_358 = arith.constant dense<0.000000e+00> : vector<128xf16>
      %true_359 = arith.constant true
      %c2_i8_360 = arith.constant 2 : i8
      %c2_i8_361 = arith.constant 2 : i8
      %275 = vector.from_elements %c2_i8_360, %c2_i8_361 : vector<2xi8>
      %c1_i8_362 = arith.constant 1 : i8
      %c8_i16_363 = arith.constant 8 : i16
      %c16_i16_364 = arith.constant 16 : i16
      %c0_i32_365 = arith.constant 0 : i32
      %c0_i32_366 = arith.constant 0 : i32
      %276 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true_359, %275, %c1_i8_362, %c8_i16_363, %c16_i16_364, %arg10, %c0_i32_365, %c0_i32_366, %cst_358) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
      %277 = vector.shape_cast %276 : vector<128xf16> to vector<8x8x2xf16>
      %cst_367 = arith.constant dense<0.000000e+00> : vector<128xf16>
      %true_368 = arith.constant true
      %c2_i8_369 = arith.constant 2 : i8
      %c2_i8_370 = arith.constant 2 : i8
      %278 = vector.from_elements %c2_i8_369, %c2_i8_370 : vector<2xi8>
      %c1_i8_371 = arith.constant 1 : i8
      %c8_i16_372 = arith.constant 8 : i16
      %c16_i16_373 = arith.constant 16 : i16
      %c0_i32_374 = arith.constant 0 : i32
      %c0_i32_375 = arith.constant 0 : i32
      %279 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true_368, %278, %c1_i8_371, %c8_i16_372, %c16_i16_373, %arg11, %c0_i32_374, %c0_i32_375, %cst_367) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
      %280 = vector.shape_cast %279 : vector<128xf16> to vector<8x8x2xf16>
      %cst_376 = arith.constant dense<0.000000e+00> : vector<512xf16>
      %true_377 = arith.constant true
      %c2_i8_378 = arith.constant 2 : i8
      %c2_i8_379 = arith.constant 2 : i8
      %281 = vector.from_elements %c2_i8_378, %c2_i8_379 : vector<2xi8>
      %c1_i8_380 = arith.constant 1 : i8
      %c16_i16_381 = arith.constant 16 : i16
      %c32_i16_382 = arith.constant 32 : i16
      %c0_i32_383 = arith.constant 0 : i32
      %c0_i32_384 = arith.constant 0 : i32
      %282 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8(%true_377, %281, %c1_i8_380, %c16_i16_381, %c32_i16_382, %arg12, %c0_i32_383, %c0_i32_384, %cst_376) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf16>) -> vector<512xf16>
      %283 = vector.shape_cast %282 : vector<512xf16> to vector<16x16x2xf16>
      %cst_385 = arith.constant dense<0.000000e+00> : vector<512xf16>
      %true_386 = arith.constant true
      %c2_i8_387 = arith.constant 2 : i8
      %c2_i8_388 = arith.constant 2 : i8
      %284 = vector.from_elements %c2_i8_387, %c2_i8_388 : vector<2xi8>
      %c1_i8_389 = arith.constant 1 : i8
      %c16_i16_390 = arith.constant 16 : i16
      %c32_i16_391 = arith.constant 32 : i16
      %c0_i32_392 = arith.constant 0 : i32
      %c0_i32_393 = arith.constant 0 : i32
      %285 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8(%true_386, %284, %c1_i8_389, %c16_i16_390, %c32_i16_391, %arg13, %c0_i32_392, %c0_i32_393, %cst_385) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf16>) -> vector<512xf16>
      %286 = vector.shape_cast %285 : vector<512xf16> to vector<16x16x2xf16>
      %287 = vector.extract_strided_slice %283 {offsets = [0, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
      %288 = vector.shape_cast %287 : vector<8x16x2xf16> to vector<256xf16>
      %289 = vector.extract_strided_slice %283 {offsets = [8, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
      %290 = vector.shape_cast %289 : vector<8x16x2xf16> to vector<256xf16>
      %291 = vector.extract_strided_slice %286 {offsets = [0, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
      %292 = vector.shape_cast %291 : vector<8x16x2xf16> to vector<256xf16>
      %293 = vector.extract_strided_slice %286 {offsets = [8, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
      %294 = vector.shape_cast %293 : vector<8x16x2xf16> to vector<256xf16>
      %295 = vector.shape_cast %259 : vector<8x8x2xf16> to vector<128xf16>
      %296 = vector.bitcast %295 : vector<128xf16> to vector<64xf32>
      %297 = vector.shape_cast %296 : vector<64xf32> to vector<8x8xf32>
      %298 = vector.shape_cast %297 : vector<8x8xf32> to vector<64xf32>
      %alloc = memref.alloc() : memref<4096xf32, 3>
      %299 = gpu.subgroup_id : index
      %300 = arith.muli %299, %c64 : index
      %301 = vector.broadcast %300 : index to vector<8xindex>
      %302 = arith.addi %301, %cst_0 : vector<8xindex>
      %intptr_394 = memref.extract_aligned_pointer_as_index %alloc : memref<4096xf32, 3> -> index
      %303 = arith.index_castui %intptr_394 : index to i32
      %304 = arith.index_castui %302 : vector<8xindex> to vector<8xi32>
      %cst_395 = arith.constant dense<4> : vector<8xi32>
      %305 = arith.muli %304, %cst_395 : vector<8xi32>
      %306 = vector.broadcast %303 : i32 to vector<8xi32>
      %307 = arith.addi %306, %305 : vector<8xi32>
      %c4_i8 = arith.constant 4 : i8
      %c0_i8 = arith.constant 0 : i8
      %c0_i8_396 = arith.constant 0 : i8
      %c1_i16 = arith.constant 1 : i16
      %c0_i32_397 = arith.constant 0 : i32
      %c3_i8_398 = arith.constant 3 : i8
      %c5_i8 = arith.constant 5 : i8
      %c1_i8_399 = arith.constant 1 : i8
      %c0_i8_400 = arith.constant 0 : i8
      %c0_i32_401 = arith.constant 0 : i32
      func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst, %c4_i8, %c0_i8, %c0_i8_396, %c1_i16, %c0_i32_397, %c3_i8_398, %c5_i8, %c1_i8_399, %c0_i8_400, %307, %298, %c0_i32_401) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
      %intptr_402 = memref.extract_aligned_pointer_as_index %alloc : memref<4096xf32, 3> -> index
      %c1_403 = arith.constant 1 : index
      %c32_404 = arith.constant 32 : index
      %c4_405 = arith.constant 4 : index
      %308 = arith.index_castui %intptr_402 : index to i32
      %309 = arith.index_castui %300 : index to i32
      %c4_i32 = arith.constant 4 : i32
      %310 = arith.muli %309, %c4_i32 : i32
      %311 = arith.addi %308, %310 : i32
      %312 = vector.broadcast %311 : i32 to vector<1xi32>
      %cst_406 = arith.constant dense<true> : vector<1xi1>
      %c0_i8_407 = arith.constant 0 : i8
      %c0_i8_408 = arith.constant 0 : i8
      %c0_i8_409 = arith.constant 0 : i8
      %c1_i16_410 = arith.constant 1 : i16
      %c0_i32_411 = arith.constant 0 : i32
      %c3_i8_412 = arith.constant 3 : i8
      %c8_i8 = arith.constant 8 : i8
      %c1_i8_413 = arith.constant 1 : i8
      %c0_i8_414 = arith.constant 0 : i8
      %c0_i32_415 = arith.constant 0 : i32
      %313 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_406, %c0_i8_407, %c0_i8_408, %c0_i8_409, %c1_i16_410, %c0_i32_411, %c3_i8_412, %c8_i8, %c1_i8_413, %c0_i8_414, %312, %c0_i32_415) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
      %314 = vector.bitcast %313 : vector<64xf32> to vector<128xf16>
      %315 = vector.shape_cast %314 : vector<128xf16> to vector<8x16xf16>
      %316 = vector.shape_cast %315 : vector<8x16xf16> to vector<128xf16>
      %317 = vector.shape_cast %271 : vector<8x8x2xf16> to vector<128xf16>
      %318 = vector.bitcast %317 : vector<128xf16> to vector<64xf32>
      %319 = vector.shape_cast %318 : vector<64xf32> to vector<8x8xf32>
      %320 = vector.shape_cast %319 : vector<8x8xf32> to vector<64xf32>
      %alloc_416 = memref.alloc() : memref<4096xf32, 3>
      %intptr_417 = memref.extract_aligned_pointer_as_index %alloc_416 : memref<4096xf32, 3> -> index
      %321 = arith.index_castui %intptr_417 : index to i32
      %322 = arith.index_castui %302 : vector<8xindex> to vector<8xi32>
      %cst_418 = arith.constant dense<4> : vector<8xi32>
      %323 = arith.muli %322, %cst_418 : vector<8xi32>
      %324 = vector.broadcast %321 : i32 to vector<8xi32>
      %325 = arith.addi %324, %323 : vector<8xi32>
      %c4_i8_419 = arith.constant 4 : i8
      %c0_i8_420 = arith.constant 0 : i8
      %c0_i8_421 = arith.constant 0 : i8
      %c1_i16_422 = arith.constant 1 : i16
      %c0_i32_423 = arith.constant 0 : i32
      %c3_i8_424 = arith.constant 3 : i8
      %c5_i8_425 = arith.constant 5 : i8
      %c1_i8_426 = arith.constant 1 : i8
      %c0_i8_427 = arith.constant 0 : i8
      %c0_i32_428 = arith.constant 0 : i32
      func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst, %c4_i8_419, %c0_i8_420, %c0_i8_421, %c1_i16_422, %c0_i32_423, %c3_i8_424, %c5_i8_425, %c1_i8_426, %c0_i8_427, %325, %320, %c0_i32_428) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
      %intptr_429 = memref.extract_aligned_pointer_as_index %alloc_416 : memref<4096xf32, 3> -> index
      %c1_430 = arith.constant 1 : index
      %c32_431 = arith.constant 32 : index
      %c4_432 = arith.constant 4 : index
      %326 = arith.index_castui %intptr_429 : index to i32
      %327 = arith.index_castui %300 : index to i32
      %c4_i32_433 = arith.constant 4 : i32
      %328 = arith.muli %327, %c4_i32_433 : i32
      %329 = arith.addi %326, %328 : i32
      %330 = vector.broadcast %329 : i32 to vector<1xi32>
      %cst_434 = arith.constant dense<true> : vector<1xi1>
      %c0_i8_435 = arith.constant 0 : i8
      %c0_i8_436 = arith.constant 0 : i8
      %c0_i8_437 = arith.constant 0 : i8
      %c1_i16_438 = arith.constant 1 : i16
      %c0_i32_439 = arith.constant 0 : i32
      %c3_i8_440 = arith.constant 3 : i8
      %c8_i8_441 = arith.constant 8 : i8
      %c1_i8_442 = arith.constant 1 : i8
      %c0_i8_443 = arith.constant 0 : i8
      %c0_i32_444 = arith.constant 0 : i32
      %331 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_434, %c0_i8_435, %c0_i8_436, %c0_i8_437, %c1_i16_438, %c0_i32_439, %c3_i8_440, %c8_i8_441, %c1_i8_442, %c0_i8_443, %330, %c0_i32_444) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
      %332 = vector.bitcast %331 : vector<64xf32> to vector<128xf16>
      %333 = vector.shape_cast %332 : vector<128xf16> to vector<8x16xf16>
      %334 = vector.shape_cast %333 : vector<8x16xf16> to vector<128xf16>
      %335 = vector.shape_cast %262 : vector<8x8x2xf16> to vector<128xf16>
      %336 = vector.bitcast %335 : vector<128xf16> to vector<64xf32>
      %337 = vector.shape_cast %336 : vector<64xf32> to vector<8x8xf32>
      %338 = vector.shape_cast %337 : vector<8x8xf32> to vector<64xf32>
      %alloc_445 = memref.alloc() : memref<4096xf32, 3>
      %intptr_446 = memref.extract_aligned_pointer_as_index %alloc_445 : memref<4096xf32, 3> -> index
      %339 = arith.index_castui %intptr_446 : index to i32
      %340 = arith.index_castui %302 : vector<8xindex> to vector<8xi32>
      %cst_447 = arith.constant dense<4> : vector<8xi32>
      %341 = arith.muli %340, %cst_447 : vector<8xi32>
      %342 = vector.broadcast %339 : i32 to vector<8xi32>
      %343 = arith.addi %342, %341 : vector<8xi32>
      %c4_i8_448 = arith.constant 4 : i8
      %c0_i8_449 = arith.constant 0 : i8
      %c0_i8_450 = arith.constant 0 : i8
      %c1_i16_451 = arith.constant 1 : i16
      %c0_i32_452 = arith.constant 0 : i32
      %c3_i8_453 = arith.constant 3 : i8
      %c5_i8_454 = arith.constant 5 : i8
      %c1_i8_455 = arith.constant 1 : i8
      %c0_i8_456 = arith.constant 0 : i8
      %c0_i32_457 = arith.constant 0 : i32
      func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst, %c4_i8_448, %c0_i8_449, %c0_i8_450, %c1_i16_451, %c0_i32_452, %c3_i8_453, %c5_i8_454, %c1_i8_455, %c0_i8_456, %343, %338, %c0_i32_457) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
      %intptr_458 = memref.extract_aligned_pointer_as_index %alloc_445 : memref<4096xf32, 3> -> index
      %c1_459 = arith.constant 1 : index
      %c32_460 = arith.constant 32 : index
      %c4_461 = arith.constant 4 : index
      %344 = arith.index_castui %intptr_458 : index to i32
      %345 = arith.index_castui %300 : index to i32
      %c4_i32_462 = arith.constant 4 : i32
      %346 = arith.muli %345, %c4_i32_462 : i32
      %347 = arith.addi %344, %346 : i32
      %348 = vector.broadcast %347 : i32 to vector<1xi32>
      %cst_463 = arith.constant dense<true> : vector<1xi1>
      %c0_i8_464 = arith.constant 0 : i8
      %c0_i8_465 = arith.constant 0 : i8
      %c0_i8_466 = arith.constant 0 : i8
      %c1_i16_467 = arith.constant 1 : i16
      %c0_i32_468 = arith.constant 0 : i32
      %c3_i8_469 = arith.constant 3 : i8
      %c8_i8_470 = arith.constant 8 : i8
      %c1_i8_471 = arith.constant 1 : i8
      %c0_i8_472 = arith.constant 0 : i8
      %c0_i32_473 = arith.constant 0 : i32
      %349 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_463, %c0_i8_464, %c0_i8_465, %c0_i8_466, %c1_i16_467, %c0_i32_468, %c3_i8_469, %c8_i8_470, %c1_i8_471, %c0_i8_472, %348, %c0_i32_473) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
      %350 = vector.bitcast %349 : vector<64xf32> to vector<128xf16>
      %351 = vector.shape_cast %350 : vector<128xf16> to vector<8x16xf16>
      %352 = vector.shape_cast %351 : vector<8x16xf16> to vector<128xf16>
      %353 = vector.shape_cast %274 : vector<8x8x2xf16> to vector<128xf16>
      %354 = vector.bitcast %353 : vector<128xf16> to vector<64xf32>
      %355 = vector.shape_cast %354 : vector<64xf32> to vector<8x8xf32>
      %356 = vector.shape_cast %355 : vector<8x8xf32> to vector<64xf32>
      %alloc_474 = memref.alloc() : memref<4096xf32, 3>
      %intptr_475 = memref.extract_aligned_pointer_as_index %alloc_474 : memref<4096xf32, 3> -> index
      %357 = arith.index_castui %intptr_475 : index to i32
      %358 = arith.index_castui %302 : vector<8xindex> to vector<8xi32>
      %cst_476 = arith.constant dense<4> : vector<8xi32>
      %359 = arith.muli %358, %cst_476 : vector<8xi32>
      %360 = vector.broadcast %357 : i32 to vector<8xi32>
      %361 = arith.addi %360, %359 : vector<8xi32>
      %c4_i8_477 = arith.constant 4 : i8
      %c0_i8_478 = arith.constant 0 : i8
      %c0_i8_479 = arith.constant 0 : i8
      %c1_i16_480 = arith.constant 1 : i16
      %c0_i32_481 = arith.constant 0 : i32
      %c3_i8_482 = arith.constant 3 : i8
      %c5_i8_483 = arith.constant 5 : i8
      %c1_i8_484 = arith.constant 1 : i8
      %c0_i8_485 = arith.constant 0 : i8
      %c0_i32_486 = arith.constant 0 : i32
      func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst, %c4_i8_477, %c0_i8_478, %c0_i8_479, %c1_i16_480, %c0_i32_481, %c3_i8_482, %c5_i8_483, %c1_i8_484, %c0_i8_485, %361, %356, %c0_i32_486) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
      %intptr_487 = memref.extract_aligned_pointer_as_index %alloc_474 : memref<4096xf32, 3> -> index
      %c1_488 = arith.constant 1 : index
      %c32_489 = arith.constant 32 : index
      %c4_490 = arith.constant 4 : index
      %362 = arith.index_castui %intptr_487 : index to i32
      %363 = arith.index_castui %300 : index to i32
      %c4_i32_491 = arith.constant 4 : i32
      %364 = arith.muli %363, %c4_i32_491 : i32
      %365 = arith.addi %362, %364 : i32
      %366 = vector.broadcast %365 : i32 to vector<1xi32>
      %cst_492 = arith.constant dense<true> : vector<1xi1>
      %c0_i8_493 = arith.constant 0 : i8
      %c0_i8_494 = arith.constant 0 : i8
      %c0_i8_495 = arith.constant 0 : i8
      %c1_i16_496 = arith.constant 1 : i16
      %c0_i32_497 = arith.constant 0 : i32
      %c3_i8_498 = arith.constant 3 : i8
      %c8_i8_499 = arith.constant 8 : i8
      %c1_i8_500 = arith.constant 1 : i8
      %c0_i8_501 = arith.constant 0 : i8
      %c0_i32_502 = arith.constant 0 : i32
      %367 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_492, %c0_i8_493, %c0_i8_494, %c0_i8_495, %c1_i16_496, %c0_i32_497, %c3_i8_498, %c8_i8_499, %c1_i8_500, %c0_i8_501, %366, %c0_i32_502) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
      %368 = vector.bitcast %367 : vector<64xf32> to vector<128xf16>
      %369 = vector.shape_cast %368 : vector<128xf16> to vector<8x16xf16>
      %370 = vector.shape_cast %369 : vector<8x16xf16> to vector<128xf16>
      %371 = vector.shape_cast %265 : vector<8x8x2xf16> to vector<128xf16>
      %372 = vector.bitcast %371 : vector<128xf16> to vector<64xf32>
      %373 = vector.shape_cast %372 : vector<64xf32> to vector<8x8xf32>
      %374 = vector.shape_cast %373 : vector<8x8xf32> to vector<64xf32>
      %alloc_503 = memref.alloc() : memref<4096xf32, 3>
      %intptr_504 = memref.extract_aligned_pointer_as_index %alloc_503 : memref<4096xf32, 3> -> index
      %375 = arith.index_castui %intptr_504 : index to i32
      %376 = arith.index_castui %302 : vector<8xindex> to vector<8xi32>
      %cst_505 = arith.constant dense<4> : vector<8xi32>
      %377 = arith.muli %376, %cst_505 : vector<8xi32>
      %378 = vector.broadcast %375 : i32 to vector<8xi32>
      %379 = arith.addi %378, %377 : vector<8xi32>
      %c4_i8_506 = arith.constant 4 : i8
      %c0_i8_507 = arith.constant 0 : i8
      %c0_i8_508 = arith.constant 0 : i8
      %c1_i16_509 = arith.constant 1 : i16
      %c0_i32_510 = arith.constant 0 : i32
      %c3_i8_511 = arith.constant 3 : i8
      %c5_i8_512 = arith.constant 5 : i8
      %c1_i8_513 = arith.constant 1 : i8
      %c0_i8_514 = arith.constant 0 : i8
      %c0_i32_515 = arith.constant 0 : i32
      func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst, %c4_i8_506, %c0_i8_507, %c0_i8_508, %c1_i16_509, %c0_i32_510, %c3_i8_511, %c5_i8_512, %c1_i8_513, %c0_i8_514, %379, %374, %c0_i32_515) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
      %intptr_516 = memref.extract_aligned_pointer_as_index %alloc_503 : memref<4096xf32, 3> -> index
      %c1_517 = arith.constant 1 : index
      %c32_518 = arith.constant 32 : index
      %c4_519 = arith.constant 4 : index
      %380 = arith.index_castui %intptr_516 : index to i32
      %381 = arith.index_castui %300 : index to i32
      %c4_i32_520 = arith.constant 4 : i32
      %382 = arith.muli %381, %c4_i32_520 : i32
      %383 = arith.addi %380, %382 : i32
      %384 = vector.broadcast %383 : i32 to vector<1xi32>
      %cst_521 = arith.constant dense<true> : vector<1xi1>
      %c0_i8_522 = arith.constant 0 : i8
      %c0_i8_523 = arith.constant 0 : i8
      %c0_i8_524 = arith.constant 0 : i8
      %c1_i16_525 = arith.constant 1 : i16
      %c0_i32_526 = arith.constant 0 : i32
      %c3_i8_527 = arith.constant 3 : i8
      %c8_i8_528 = arith.constant 8 : i8
      %c1_i8_529 = arith.constant 1 : i8
      %c0_i8_530 = arith.constant 0 : i8
      %c0_i32_531 = arith.constant 0 : i32
      %385 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_521, %c0_i8_522, %c0_i8_523, %c0_i8_524, %c1_i16_525, %c0_i32_526, %c3_i8_527, %c8_i8_528, %c1_i8_529, %c0_i8_530, %384, %c0_i32_531) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
      %386 = vector.bitcast %385 : vector<64xf32> to vector<128xf16>
      %387 = vector.shape_cast %386 : vector<128xf16> to vector<8x16xf16>
      %388 = vector.shape_cast %387 : vector<8x16xf16> to vector<128xf16>
      %389 = vector.shape_cast %277 : vector<8x8x2xf16> to vector<128xf16>
      %390 = vector.bitcast %389 : vector<128xf16> to vector<64xf32>
      %391 = vector.shape_cast %390 : vector<64xf32> to vector<8x8xf32>
      %392 = vector.shape_cast %391 : vector<8x8xf32> to vector<64xf32>
      %alloc_532 = memref.alloc() : memref<4096xf32, 3>
      %intptr_533 = memref.extract_aligned_pointer_as_index %alloc_532 : memref<4096xf32, 3> -> index
      %393 = arith.index_castui %intptr_533 : index to i32
      %394 = arith.index_castui %302 : vector<8xindex> to vector<8xi32>
      %cst_534 = arith.constant dense<4> : vector<8xi32>
      %395 = arith.muli %394, %cst_534 : vector<8xi32>
      %396 = vector.broadcast %393 : i32 to vector<8xi32>
      %397 = arith.addi %396, %395 : vector<8xi32>
      %c4_i8_535 = arith.constant 4 : i8
      %c0_i8_536 = arith.constant 0 : i8
      %c0_i8_537 = arith.constant 0 : i8
      %c1_i16_538 = arith.constant 1 : i16
      %c0_i32_539 = arith.constant 0 : i32
      %c3_i8_540 = arith.constant 3 : i8
      %c5_i8_541 = arith.constant 5 : i8
      %c1_i8_542 = arith.constant 1 : i8
      %c0_i8_543 = arith.constant 0 : i8
      %c0_i32_544 = arith.constant 0 : i32
      func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst, %c4_i8_535, %c0_i8_536, %c0_i8_537, %c1_i16_538, %c0_i32_539, %c3_i8_540, %c5_i8_541, %c1_i8_542, %c0_i8_543, %397, %392, %c0_i32_544) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
      %intptr_545 = memref.extract_aligned_pointer_as_index %alloc_532 : memref<4096xf32, 3> -> index
      %c1_546 = arith.constant 1 : index
      %c32_547 = arith.constant 32 : index
      %c4_548 = arith.constant 4 : index
      %398 = arith.index_castui %intptr_545 : index to i32
      %399 = arith.index_castui %300 : index to i32
      %c4_i32_549 = arith.constant 4 : i32
      %400 = arith.muli %399, %c4_i32_549 : i32
      %401 = arith.addi %398, %400 : i32
      %402 = vector.broadcast %401 : i32 to vector<1xi32>
      %cst_550 = arith.constant dense<true> : vector<1xi1>
      %c0_i8_551 = arith.constant 0 : i8
      %c0_i8_552 = arith.constant 0 : i8
      %c0_i8_553 = arith.constant 0 : i8
      %c1_i16_554 = arith.constant 1 : i16
      %c0_i32_555 = arith.constant 0 : i32
      %c3_i8_556 = arith.constant 3 : i8
      %c8_i8_557 = arith.constant 8 : i8
      %c1_i8_558 = arith.constant 1 : i8
      %c0_i8_559 = arith.constant 0 : i8
      %c0_i32_560 = arith.constant 0 : i32
      %403 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_550, %c0_i8_551, %c0_i8_552, %c0_i8_553, %c1_i16_554, %c0_i32_555, %c3_i8_556, %c8_i8_557, %c1_i8_558, %c0_i8_559, %402, %c0_i32_560) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
      %404 = vector.bitcast %403 : vector<64xf32> to vector<128xf16>
      %405 = vector.shape_cast %404 : vector<128xf16> to vector<8x16xf16>
      %406 = vector.shape_cast %405 : vector<8x16xf16> to vector<128xf16>
      %407 = vector.shape_cast %268 : vector<8x8x2xf16> to vector<128xf16>
      %408 = vector.bitcast %407 : vector<128xf16> to vector<64xf32>
      %409 = vector.shape_cast %408 : vector<64xf32> to vector<8x8xf32>
      %410 = vector.shape_cast %409 : vector<8x8xf32> to vector<64xf32>
      %alloc_561 = memref.alloc() : memref<4096xf32, 3>
      %intptr_562 = memref.extract_aligned_pointer_as_index %alloc_561 : memref<4096xf32, 3> -> index
      %411 = arith.index_castui %intptr_562 : index to i32
      %412 = arith.index_castui %302 : vector<8xindex> to vector<8xi32>
      %cst_563 = arith.constant dense<4> : vector<8xi32>
      %413 = arith.muli %412, %cst_563 : vector<8xi32>
      %414 = vector.broadcast %411 : i32 to vector<8xi32>
      %415 = arith.addi %414, %413 : vector<8xi32>
      %c4_i8_564 = arith.constant 4 : i8
      %c0_i8_565 = arith.constant 0 : i8
      %c0_i8_566 = arith.constant 0 : i8
      %c1_i16_567 = arith.constant 1 : i16
      %c0_i32_568 = arith.constant 0 : i32
      %c3_i8_569 = arith.constant 3 : i8
      %c5_i8_570 = arith.constant 5 : i8
      %c1_i8_571 = arith.constant 1 : i8
      %c0_i8_572 = arith.constant 0 : i8
      %c0_i32_573 = arith.constant 0 : i32
      func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst, %c4_i8_564, %c0_i8_565, %c0_i8_566, %c1_i16_567, %c0_i32_568, %c3_i8_569, %c5_i8_570, %c1_i8_571, %c0_i8_572, %415, %410, %c0_i32_573) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
      %intptr_574 = memref.extract_aligned_pointer_as_index %alloc_561 : memref<4096xf32, 3> -> index
      %c1_575 = arith.constant 1 : index
      %c32_576 = arith.constant 32 : index
      %c4_577 = arith.constant 4 : index
      %416 = arith.index_castui %intptr_574 : index to i32
      %417 = arith.index_castui %300 : index to i32
      %c4_i32_578 = arith.constant 4 : i32
      %418 = arith.muli %417, %c4_i32_578 : i32
      %419 = arith.addi %416, %418 : i32
      %420 = vector.broadcast %419 : i32 to vector<1xi32>
      %cst_579 = arith.constant dense<true> : vector<1xi1>
      %c0_i8_580 = arith.constant 0 : i8
      %c0_i8_581 = arith.constant 0 : i8
      %c0_i8_582 = arith.constant 0 : i8
      %c1_i16_583 = arith.constant 1 : i16
      %c0_i32_584 = arith.constant 0 : i32
      %c3_i8_585 = arith.constant 3 : i8
      %c8_i8_586 = arith.constant 8 : i8
      %c1_i8_587 = arith.constant 1 : i8
      %c0_i8_588 = arith.constant 0 : i8
      %c0_i32_589 = arith.constant 0 : i32
      %421 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_579, %c0_i8_580, %c0_i8_581, %c0_i8_582, %c1_i16_583, %c0_i32_584, %c3_i8_585, %c8_i8_586, %c1_i8_587, %c0_i8_588, %420, %c0_i32_589) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
      %422 = vector.bitcast %421 : vector<64xf32> to vector<128xf16>
      %423 = vector.shape_cast %422 : vector<128xf16> to vector<8x16xf16>
      %424 = vector.shape_cast %423 : vector<8x16xf16> to vector<128xf16>
      %425 = vector.shape_cast %280 : vector<8x8x2xf16> to vector<128xf16>
      %426 = vector.bitcast %425 : vector<128xf16> to vector<64xf32>
      %427 = vector.shape_cast %426 : vector<64xf32> to vector<8x8xf32>
      %428 = vector.shape_cast %427 : vector<8x8xf32> to vector<64xf32>
      %alloc_590 = memref.alloc() : memref<4096xf32, 3>
      %intptr_591 = memref.extract_aligned_pointer_as_index %alloc_590 : memref<4096xf32, 3> -> index
      %429 = arith.index_castui %intptr_591 : index to i32
      %430 = arith.index_castui %302 : vector<8xindex> to vector<8xi32>
      %cst_592 = arith.constant dense<4> : vector<8xi32>
      %431 = arith.muli %430, %cst_592 : vector<8xi32>
      %432 = vector.broadcast %429 : i32 to vector<8xi32>
      %433 = arith.addi %432, %431 : vector<8xi32>
      %c4_i8_593 = arith.constant 4 : i8
      %c0_i8_594 = arith.constant 0 : i8
      %c0_i8_595 = arith.constant 0 : i8
      %c1_i16_596 = arith.constant 1 : i16
      %c0_i32_597 = arith.constant 0 : i32
      %c3_i8_598 = arith.constant 3 : i8
      %c5_i8_599 = arith.constant 5 : i8
      %c1_i8_600 = arith.constant 1 : i8
      %c0_i8_601 = arith.constant 0 : i8
      %c0_i32_602 = arith.constant 0 : i32
      func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst, %c4_i8_593, %c0_i8_594, %c0_i8_595, %c1_i16_596, %c0_i32_597, %c3_i8_598, %c5_i8_599, %c1_i8_600, %c0_i8_601, %433, %428, %c0_i32_602) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
      %intptr_603 = memref.extract_aligned_pointer_as_index %alloc_590 : memref<4096xf32, 3> -> index
      %c1_604 = arith.constant 1 : index
      %c32_605 = arith.constant 32 : index
      %c4_606 = arith.constant 4 : index
      %434 = arith.index_castui %intptr_603 : index to i32
      %435 = arith.index_castui %300 : index to i32
      %c4_i32_607 = arith.constant 4 : i32
      %436 = arith.muli %435, %c4_i32_607 : i32
      %437 = arith.addi %434, %436 : i32
      %438 = vector.broadcast %437 : i32 to vector<1xi32>
      %cst_608 = arith.constant dense<true> : vector<1xi1>
      %c0_i8_609 = arith.constant 0 : i8
      %c0_i8_610 = arith.constant 0 : i8
      %c0_i8_611 = arith.constant 0 : i8
      %c1_i16_612 = arith.constant 1 : i16
      %c0_i32_613 = arith.constant 0 : i32
      %c3_i8_614 = arith.constant 3 : i8
      %c8_i8_615 = arith.constant 8 : i8
      %c1_i8_616 = arith.constant 1 : i8
      %c0_i8_617 = arith.constant 0 : i8
      %c0_i32_618 = arith.constant 0 : i32
      %439 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_608, %c0_i8_609, %c0_i8_610, %c0_i8_611, %c1_i16_612, %c0_i32_613, %c3_i8_614, %c8_i8_615, %c1_i8_616, %c0_i8_617, %438, %c0_i32_618) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
      %440 = vector.bitcast %439 : vector<64xf32> to vector<128xf16>
      %441 = vector.shape_cast %440 : vector<128xf16> to vector<8x16xf16>
      %442 = vector.shape_cast %441 : vector<8x16xf16> to vector<128xf16>
      %c134744586_i32 = arith.constant 134744586 : i32
      %443 = vector.bitcast %316 : vector<128xf16> to vector<64xi32>
      %444 = vector.bitcast %288 : vector<256xf16> to vector<128xi32>
      %c10_i32 = arith.constant 10 : i32
      %c10_i32_619 = arith.constant 10 : i32
      %c8_i32 = arith.constant 8 : i32
      %c8_i32_620 = arith.constant 8 : i32
      %c0_i32_621 = arith.constant 0 : i32
      %445 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg14, %444, %443, %c10_i32, %c10_i32_619, %c8_i32, %c8_i32_620, %c0_i32_621, %c0_i32_621) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %c134744586_i32_622 = arith.constant 134744586 : i32
      %446 = vector.bitcast %334 : vector<128xf16> to vector<64xi32>
      %447 = vector.bitcast %290 : vector<256xf16> to vector<128xi32>
      %c10_i32_623 = arith.constant 10 : i32
      %c10_i32_624 = arith.constant 10 : i32
      %c8_i32_625 = arith.constant 8 : i32
      %c8_i32_626 = arith.constant 8 : i32
      %c0_i32_627 = arith.constant 0 : i32
      %448 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%445, %447, %446, %c10_i32_623, %c10_i32_624, %c8_i32_625, %c8_i32_626, %c0_i32_627, %c0_i32_627) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %c134744586_i32_628 = arith.constant 134744586 : i32
      %449 = vector.bitcast %316 : vector<128xf16> to vector<64xi32>
      %450 = vector.bitcast %292 : vector<256xf16> to vector<128xi32>
      %c10_i32_629 = arith.constant 10 : i32
      %c10_i32_630 = arith.constant 10 : i32
      %c8_i32_631 = arith.constant 8 : i32
      %c8_i32_632 = arith.constant 8 : i32
      %c0_i32_633 = arith.constant 0 : i32
      %451 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg15, %450, %449, %c10_i32_629, %c10_i32_630, %c8_i32_631, %c8_i32_632, %c0_i32_633, %c0_i32_633) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %c134744586_i32_634 = arith.constant 134744586 : i32
      %452 = vector.bitcast %334 : vector<128xf16> to vector<64xi32>
      %453 = vector.bitcast %294 : vector<256xf16> to vector<128xi32>
      %c10_i32_635 = arith.constant 10 : i32
      %c10_i32_636 = arith.constant 10 : i32
      %c8_i32_637 = arith.constant 8 : i32
      %c8_i32_638 = arith.constant 8 : i32
      %c0_i32_639 = arith.constant 0 : i32
      %454 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%451, %453, %452, %c10_i32_635, %c10_i32_636, %c8_i32_637, %c8_i32_638, %c0_i32_639, %c0_i32_639) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %c134744586_i32_640 = arith.constant 134744586 : i32
      %455 = vector.bitcast %352 : vector<128xf16> to vector<64xi32>
      %456 = vector.bitcast %288 : vector<256xf16> to vector<128xi32>
      %c10_i32_641 = arith.constant 10 : i32
      %c10_i32_642 = arith.constant 10 : i32
      %c8_i32_643 = arith.constant 8 : i32
      %c8_i32_644 = arith.constant 8 : i32
      %c0_i32_645 = arith.constant 0 : i32
      %457 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg16, %456, %455, %c10_i32_641, %c10_i32_642, %c8_i32_643, %c8_i32_644, %c0_i32_645, %c0_i32_645) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %c134744586_i32_646 = arith.constant 134744586 : i32
      %458 = vector.bitcast %370 : vector<128xf16> to vector<64xi32>
      %459 = vector.bitcast %290 : vector<256xf16> to vector<128xi32>
      %c10_i32_647 = arith.constant 10 : i32
      %c10_i32_648 = arith.constant 10 : i32
      %c8_i32_649 = arith.constant 8 : i32
      %c8_i32_650 = arith.constant 8 : i32
      %c0_i32_651 = arith.constant 0 : i32
      %460 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%457, %459, %458, %c10_i32_647, %c10_i32_648, %c8_i32_649, %c8_i32_650, %c0_i32_651, %c0_i32_651) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %c134744586_i32_652 = arith.constant 134744586 : i32
      %461 = vector.bitcast %352 : vector<128xf16> to vector<64xi32>
      %462 = vector.bitcast %292 : vector<256xf16> to vector<128xi32>
      %c10_i32_653 = arith.constant 10 : i32
      %c10_i32_654 = arith.constant 10 : i32
      %c8_i32_655 = arith.constant 8 : i32
      %c8_i32_656 = arith.constant 8 : i32
      %c0_i32_657 = arith.constant 0 : i32
      %463 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg17, %462, %461, %c10_i32_653, %c10_i32_654, %c8_i32_655, %c8_i32_656, %c0_i32_657, %c0_i32_657) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %c134744586_i32_658 = arith.constant 134744586 : i32
      %464 = vector.bitcast %370 : vector<128xf16> to vector<64xi32>
      %465 = vector.bitcast %294 : vector<256xf16> to vector<128xi32>
      %c10_i32_659 = arith.constant 10 : i32
      %c10_i32_660 = arith.constant 10 : i32
      %c8_i32_661 = arith.constant 8 : i32
      %c8_i32_662 = arith.constant 8 : i32
      %c0_i32_663 = arith.constant 0 : i32
      %466 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%463, %465, %464, %c10_i32_659, %c10_i32_660, %c8_i32_661, %c8_i32_662, %c0_i32_663, %c0_i32_663) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %c134744586_i32_664 = arith.constant 134744586 : i32
      %467 = vector.bitcast %388 : vector<128xf16> to vector<64xi32>
      %468 = vector.bitcast %288 : vector<256xf16> to vector<128xi32>
      %c10_i32_665 = arith.constant 10 : i32
      %c10_i32_666 = arith.constant 10 : i32
      %c8_i32_667 = arith.constant 8 : i32
      %c8_i32_668 = arith.constant 8 : i32
      %c0_i32_669 = arith.constant 0 : i32
      %469 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg18, %468, %467, %c10_i32_665, %c10_i32_666, %c8_i32_667, %c8_i32_668, %c0_i32_669, %c0_i32_669) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %c134744586_i32_670 = arith.constant 134744586 : i32
      %470 = vector.bitcast %406 : vector<128xf16> to vector<64xi32>
      %471 = vector.bitcast %290 : vector<256xf16> to vector<128xi32>
      %c10_i32_671 = arith.constant 10 : i32
      %c10_i32_672 = arith.constant 10 : i32
      %c8_i32_673 = arith.constant 8 : i32
      %c8_i32_674 = arith.constant 8 : i32
      %c0_i32_675 = arith.constant 0 : i32
      %472 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%469, %471, %470, %c10_i32_671, %c10_i32_672, %c8_i32_673, %c8_i32_674, %c0_i32_675, %c0_i32_675) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %c134744586_i32_676 = arith.constant 134744586 : i32
      %473 = vector.bitcast %388 : vector<128xf16> to vector<64xi32>
      %474 = vector.bitcast %292 : vector<256xf16> to vector<128xi32>
      %c10_i32_677 = arith.constant 10 : i32
      %c10_i32_678 = arith.constant 10 : i32
      %c8_i32_679 = arith.constant 8 : i32
      %c8_i32_680 = arith.constant 8 : i32
      %c0_i32_681 = arith.constant 0 : i32
      %475 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg19, %474, %473, %c10_i32_677, %c10_i32_678, %c8_i32_679, %c8_i32_680, %c0_i32_681, %c0_i32_681) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %c134744586_i32_682 = arith.constant 134744586 : i32
      %476 = vector.bitcast %406 : vector<128xf16> to vector<64xi32>
      %477 = vector.bitcast %294 : vector<256xf16> to vector<128xi32>
      %c10_i32_683 = arith.constant 10 : i32
      %c10_i32_684 = arith.constant 10 : i32
      %c8_i32_685 = arith.constant 8 : i32
      %c8_i32_686 = arith.constant 8 : i32
      %c0_i32_687 = arith.constant 0 : i32
      %478 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%475, %477, %476, %c10_i32_683, %c10_i32_684, %c8_i32_685, %c8_i32_686, %c0_i32_687, %c0_i32_687) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %c134744586_i32_688 = arith.constant 134744586 : i32
      %479 = vector.bitcast %424 : vector<128xf16> to vector<64xi32>
      %480 = vector.bitcast %288 : vector<256xf16> to vector<128xi32>
      %c10_i32_689 = arith.constant 10 : i32
      %c10_i32_690 = arith.constant 10 : i32
      %c8_i32_691 = arith.constant 8 : i32
      %c8_i32_692 = arith.constant 8 : i32
      %c0_i32_693 = arith.constant 0 : i32
      %481 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg20, %480, %479, %c10_i32_689, %c10_i32_690, %c8_i32_691, %c8_i32_692, %c0_i32_693, %c0_i32_693) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %c134744586_i32_694 = arith.constant 134744586 : i32
      %482 = vector.bitcast %442 : vector<128xf16> to vector<64xi32>
      %483 = vector.bitcast %290 : vector<256xf16> to vector<128xi32>
      %c10_i32_695 = arith.constant 10 : i32
      %c10_i32_696 = arith.constant 10 : i32
      %c8_i32_697 = arith.constant 8 : i32
      %c8_i32_698 = arith.constant 8 : i32
      %c0_i32_699 = arith.constant 0 : i32
      %484 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%481, %483, %482, %c10_i32_695, %c10_i32_696, %c8_i32_697, %c8_i32_698, %c0_i32_699, %c0_i32_699) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %c134744586_i32_700 = arith.constant 134744586 : i32
      %485 = vector.bitcast %424 : vector<128xf16> to vector<64xi32>
      %486 = vector.bitcast %292 : vector<256xf16> to vector<128xi32>
      %c10_i32_701 = arith.constant 10 : i32
      %c10_i32_702 = arith.constant 10 : i32
      %c8_i32_703 = arith.constant 8 : i32
      %c8_i32_704 = arith.constant 8 : i32
      %c0_i32_705 = arith.constant 0 : i32
      %487 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg21, %486, %485, %c10_i32_701, %c10_i32_702, %c8_i32_703, %c8_i32_704, %c0_i32_705, %c0_i32_705) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %c134744586_i32_706 = arith.constant 134744586 : i32
      %488 = vector.bitcast %442 : vector<128xf16> to vector<64xi32>
      %489 = vector.bitcast %294 : vector<256xf16> to vector<128xi32>
      %c10_i32_707 = arith.constant 10 : i32
      %c10_i32_708 = arith.constant 10 : i32
      %c8_i32_709 = arith.constant 8 : i32
      %c8_i32_710 = arith.constant 8 : i32
      %c0_i32_711 = arith.constant 0 : i32
      %490 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%487, %489, %488, %c10_i32_707, %c10_i32_708, %c8_i32_709, %c8_i32_710, %c0_i32_711, %c0_i32_711) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %c32_i32 = arith.constant 32 : i32
      %491 = vector.extract %arg4[6] : i32 from vector<16xi32>
      %492 = arith.addi %491, %c32_i32 : i32
      %493 = vector.insert %492, %arg4 [6] : i32 into vector<16xi32>
      %c32_i32_712 = arith.constant 32 : i32
      %494 = vector.extract %arg5[6] : i32 from vector<16xi32>
      %495 = arith.addi %494, %c32_i32_712 : i32
      %496 = vector.insert %495, %arg5 [6] : i32 into vector<16xi32>
      %c32_i32_713 = arith.constant 32 : i32
      %497 = vector.extract %arg6[6] : i32 from vector<16xi32>
      %498 = arith.addi %497, %c32_i32_713 : i32
      %499 = vector.insert %498, %arg6 [6] : i32 into vector<16xi32>
      %c32_i32_714 = arith.constant 32 : i32
      %500 = vector.extract %arg7[6] : i32 from vector<16xi32>
      %501 = arith.addi %500, %c32_i32_714 : i32
      %502 = vector.insert %501, %arg7 [6] : i32 into vector<16xi32>
      %c32_i32_715 = arith.constant 32 : i32
      %503 = vector.extract %arg8[6] : i32 from vector<16xi32>
      %504 = arith.addi %503, %c32_i32_715 : i32
      %505 = vector.insert %504, %arg8 [6] : i32 into vector<16xi32>
      %c32_i32_716 = arith.constant 32 : i32
      %506 = vector.extract %arg9[6] : i32 from vector<16xi32>
      %507 = arith.addi %506, %c32_i32_716 : i32
      %508 = vector.insert %507, %arg9 [6] : i32 into vector<16xi32>
      %c32_i32_717 = arith.constant 32 : i32
      %509 = vector.extract %arg10[6] : i32 from vector<16xi32>
      %510 = arith.addi %509, %c32_i32_717 : i32
      %511 = vector.insert %510, %arg10 [6] : i32 into vector<16xi32>
      %c32_i32_718 = arith.constant 32 : i32
      %512 = vector.extract %arg11[6] : i32 from vector<16xi32>
      %513 = arith.addi %512, %c32_i32_718 : i32
      %514 = vector.insert %513, %arg11 [6] : i32 into vector<16xi32>
      %c32_i32_719 = arith.constant 32 : i32
      %515 = vector.extract %arg12[6] : i32 from vector<16xi32>
      %516 = arith.addi %515, %c32_i32_719 : i32
      %517 = vector.insert %516, %arg12 [6] : i32 into vector<16xi32>
      %c32_i32_720 = arith.constant 32 : i32
      %518 = vector.extract %arg13[6] : i32 from vector<16xi32>
      %519 = arith.addi %518, %c32_i32_720 : i32
      %520 = vector.insert %519, %arg13 [6] : i32 into vector<16xi32>
      scf.yield %493, %496, %499, %502, %505, %508, %511, %514, %517, %520, %448, %454, %460, %466, %472, %478, %484, %490 : vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>
    }
    %true_242 = arith.constant true
    %c3_i8 = arith.constant 3 : i8
    %c3_i8_243 = arith.constant 3 : i8
    %249 = vector.from_elements %c3_i8, %c3_i8_243 : vector<2xi8>
    %c1_i8_244 = arith.constant 1 : i8
    %c16_i16_245 = arith.constant 16 : i16
    %c8_i16 = arith.constant 8 : i16
    %c0_i32_246 = arith.constant 0 : i32
    %c0_i32_247 = arith.constant 0 : i32
    func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true_242, %249, %c1_i8_244, %c16_i16_245, %c8_i16, %12, %c0_i32_246, %c0_i32_247, %248#10) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
    %true_248 = arith.constant true
    %c3_i8_249 = arith.constant 3 : i8
    %c3_i8_250 = arith.constant 3 : i8
    %250 = vector.from_elements %c3_i8_249, %c3_i8_250 : vector<2xi8>
    %c1_i8_251 = arith.constant 1 : i8
    %c16_i16_252 = arith.constant 16 : i16
    %c8_i16_253 = arith.constant 8 : i16
    %c0_i32_254 = arith.constant 0 : i32
    %c0_i32_255 = arith.constant 0 : i32
    func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true_248, %250, %c1_i8_251, %c16_i16_252, %c8_i16_253, %24, %c0_i32_254, %c0_i32_255, %248#11) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
    %true_256 = arith.constant true
    %c3_i8_257 = arith.constant 3 : i8
    %c3_i8_258 = arith.constant 3 : i8
    %251 = vector.from_elements %c3_i8_257, %c3_i8_258 : vector<2xi8>
    %c1_i8_259 = arith.constant 1 : i8
    %c16_i16_260 = arith.constant 16 : i16
    %c8_i16_261 = arith.constant 8 : i16
    %c0_i32_262 = arith.constant 0 : i32
    %c0_i32_263 = arith.constant 0 : i32
    func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true_256, %251, %c1_i8_259, %c16_i16_260, %c8_i16_261, %36, %c0_i32_262, %c0_i32_263, %248#12) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
    %true_264 = arith.constant true
    %c3_i8_265 = arith.constant 3 : i8
    %c3_i8_266 = arith.constant 3 : i8
    %252 = vector.from_elements %c3_i8_265, %c3_i8_266 : vector<2xi8>
    %c1_i8_267 = arith.constant 1 : i8
    %c16_i16_268 = arith.constant 16 : i16
    %c8_i16_269 = arith.constant 8 : i16
    %c0_i32_270 = arith.constant 0 : i32
    %c0_i32_271 = arith.constant 0 : i32
    func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true_264, %252, %c1_i8_267, %c16_i16_268, %c8_i16_269, %47, %c0_i32_270, %c0_i32_271, %248#13) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
    %true_272 = arith.constant true
    %c3_i8_273 = arith.constant 3 : i8
    %c3_i8_274 = arith.constant 3 : i8
    %253 = vector.from_elements %c3_i8_273, %c3_i8_274 : vector<2xi8>
    %c1_i8_275 = arith.constant 1 : i8
    %c16_i16_276 = arith.constant 16 : i16
    %c8_i16_277 = arith.constant 8 : i16
    %c0_i32_278 = arith.constant 0 : i32
    %c0_i32_279 = arith.constant 0 : i32
    func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true_272, %253, %c1_i8_275, %c16_i16_276, %c8_i16_277, %59, %c0_i32_278, %c0_i32_279, %248#14) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
    %true_280 = arith.constant true
    %c3_i8_281 = arith.constant 3 : i8
    %c3_i8_282 = arith.constant 3 : i8
    %254 = vector.from_elements %c3_i8_281, %c3_i8_282 : vector<2xi8>
    %c1_i8_283 = arith.constant 1 : i8
    %c16_i16_284 = arith.constant 16 : i16
    %c8_i16_285 = arith.constant 8 : i16
    %c0_i32_286 = arith.constant 0 : i32
    %c0_i32_287 = arith.constant 0 : i32
    func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true_280, %254, %c1_i8_283, %c16_i16_284, %c8_i16_285, %70, %c0_i32_286, %c0_i32_287, %248#15) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
    %true_288 = arith.constant true
    %c3_i8_289 = arith.constant 3 : i8
    %c3_i8_290 = arith.constant 3 : i8
    %255 = vector.from_elements %c3_i8_289, %c3_i8_290 : vector<2xi8>
    %c1_i8_291 = arith.constant 1 : i8
    %c16_i16_292 = arith.constant 16 : i16
    %c8_i16_293 = arith.constant 8 : i16
    %c0_i32_294 = arith.constant 0 : i32
    %c0_i32_295 = arith.constant 0 : i32
    func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true_288, %255, %c1_i8_291, %c16_i16_292, %c8_i16_293, %82, %c0_i32_294, %c0_i32_295, %248#16) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
    %true_296 = arith.constant true
    %c3_i8_297 = arith.constant 3 : i8
    %c3_i8_298 = arith.constant 3 : i8
    %256 = vector.from_elements %c3_i8_297, %c3_i8_298 : vector<2xi8>
    %c1_i8_299 = arith.constant 1 : i8
    %c16_i16_300 = arith.constant 16 : i16
    %c8_i16_301 = arith.constant 8 : i16
    %c0_i32_302 = arith.constant 0 : i32
    %c0_i32_303 = arith.constant 0 : i32
    func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true_296, %256, %c1_i8_299, %c16_i16_300, %c8_i16_301, %93, %c0_i32_302, %c0_i32_303, %248#17) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
    gpu.return
  }
}

// -----// IR Dump After CSE (cse) //----- //
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %memref = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %arg0, %memref : memref<1024x1024xf16> to memref<1024x1024xf16>
    %memref_0 = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %arg1, %memref_0 : memref<1024x1024xf16> to memref<1024x1024xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<1024x1024xf32>
    memref.copy %arg2, %memref_1 : memref<1024x1024xf32> to memref<1024x1024xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c32, %c32, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<1024x1024xf16>, %memref_0 : memref<1024x1024xf16>, %memref_1 : memref<1024x1024xf32>)
    gpu.dealloc  %memref : memref<1024x1024xf16>
    gpu.dealloc  %memref_0 : memref<1024x1024xf16>
    return %memref_1 : memref<1024x1024xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    func.func private @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32", linkage_type = <Import>>}
    func.func private @llvm.genx.dpas2.v128f32.v128i32.v64i32(vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas2.v128f32.v128i32.v64i32", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf16>) -> vector<512xf16> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf32>) -> vector<512xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8", linkage_type = <Import>>}
    gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %cst = arith.constant dense<true> : vector<8xi1>
      %cst_0 = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56]> : vector<8xindex>
      %c64 = arith.constant 64 : index
      %c24 = arith.constant 24 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c32 : index
      %1 = arith.muli %block_id_y, %c32 : index
      %intptr = memref.extract_aligned_pointer_as_index %arg2 : memref<1024x1024xf32> -> index
      %2 = arith.index_castui %intptr : index to i64
      %cst_1 = arith.constant dense<0> : vector<8xi64>
      %3 = vector.insert %2, %cst_1 [0] : i64 into vector<8xi64>
      %4 = vector.bitcast %3 : vector<8xi64> to vector<16xi32>
      %c4095_i32 = arith.constant 4095 : i32
      %c1023_i32 = arith.constant 1023 : i32
      %5 = vector.insert %c4095_i32, %4 [2] : i32 into vector<16xi32>
      %6 = vector.insert %c1023_i32, %5 [3] : i32 into vector<16xi32>
      %7 = vector.insert %c4095_i32, %6 [4] : i32 into vector<16xi32>
      %8 = arith.index_castui %1 : index to i32
      %9 = arith.index_castui %0 : index to i32
      %10 = vector.insert %8, %7 [5] : i32 into vector<16xi32>
      %11 = vector.insert %9, %10 [6] : i32 into vector<16xi32>
      %c1807_i32 = arith.constant 1807 : i32
      %12 = vector.insert %c1807_i32, %11 [7] : i32 into vector<16xi32>
      %13 = arith.addi %1, %c16 : index
      %14 = arith.index_castui %13 : index to i32
      %15 = vector.insert %14, %7 [5] : i32 into vector<16xi32>
      %16 = vector.insert %9, %15 [6] : i32 into vector<16xi32>
      %17 = vector.insert %c1807_i32, %16 [7] : i32 into vector<16xi32>
      %18 = arith.addi %0, %c8 : index
      %19 = arith.index_castui %18 : index to i32
      %20 = vector.insert %19, %10 [6] : i32 into vector<16xi32>
      %21 = vector.insert %c1807_i32, %20 [7] : i32 into vector<16xi32>
      %22 = vector.insert %19, %15 [6] : i32 into vector<16xi32>
      %23 = vector.insert %c1807_i32, %22 [7] : i32 into vector<16xi32>
      %24 = arith.addi %0, %c16 : index
      %25 = arith.index_castui %24 : index to i32
      %26 = vector.insert %25, %10 [6] : i32 into vector<16xi32>
      %27 = vector.insert %c1807_i32, %26 [7] : i32 into vector<16xi32>
      %28 = vector.insert %25, %15 [6] : i32 into vector<16xi32>
      %29 = vector.insert %c1807_i32, %28 [7] : i32 into vector<16xi32>
      %30 = arith.addi %0, %c24 : index
      %31 = arith.index_castui %30 : index to i32
      %32 = vector.insert %31, %10 [6] : i32 into vector<16xi32>
      %33 = vector.insert %c1807_i32, %32 [7] : i32 into vector<16xi32>
      %34 = vector.insert %31, %15 [6] : i32 into vector<16xi32>
      %35 = vector.insert %c1807_i32, %34 [7] : i32 into vector<16xi32>
      %c7951_i32 = arith.constant 7951 : i32
      %36 = vector.insert %c7951_i32, %11 [7] : i32 into vector<16xi32>
      %37 = vector.insert %c7951_i32, %16 [7] : i32 into vector<16xi32>
      %cst_2 = arith.constant dense<0.000000e+00> : vector<512xf32>
      %true = arith.constant true
      %c2_i8 = arith.constant 2 : i8
      %38 = vector.from_elements %c2_i8, %c2_i8 : vector<2xi8>
      %c1_i8 = arith.constant 1 : i8
      %c16_i16 = arith.constant 16 : i16
      %c32_i16 = arith.constant 32 : i16
      %c0_i32 = arith.constant 0 : i32
      %39 = func.call @llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8(%true, %38, %c1_i8, %c16_i16, %c32_i16, %36, %c0_i32, %c0_i32, %cst_2) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf32>) -> vector<512xf32>
      %40 = vector.shape_cast %39 : vector<512xf32> to vector<32x16xf32>
      %41 = func.call @llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8(%true, %38, %c1_i8, %c16_i16, %c32_i16, %37, %c0_i32, %c0_i32, %cst_2) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf32>) -> vector<512xf32>
      %42 = vector.shape_cast %41 : vector<512xf32> to vector<32x16xf32>
      %43 = vector.extract_strided_slice %40 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %44 = vector.shape_cast %43 : vector<8x16xf32> to vector<128xf32>
      %45 = vector.extract_strided_slice %40 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %46 = vector.shape_cast %45 : vector<8x16xf32> to vector<128xf32>
      %47 = vector.extract_strided_slice %40 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %48 = vector.shape_cast %47 : vector<8x16xf32> to vector<128xf32>
      %49 = vector.extract_strided_slice %40 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %50 = vector.shape_cast %49 : vector<8x16xf32> to vector<128xf32>
      %51 = vector.extract_strided_slice %42 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %52 = vector.shape_cast %51 : vector<8x16xf32> to vector<128xf32>
      %53 = vector.extract_strided_slice %42 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %54 = vector.shape_cast %53 : vector<8x16xf32> to vector<128xf32>
      %55 = vector.extract_strided_slice %42 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %56 = vector.shape_cast %55 : vector<8x16xf32> to vector<128xf32>
      %57 = vector.extract_strided_slice %42 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %58 = vector.shape_cast %57 : vector<8x16xf32> to vector<128xf32>
      %intptr_3 = memref.extract_aligned_pointer_as_index %arg0 : memref<1024x1024xf16> -> index
      %59 = arith.index_castui %intptr_3 : index to i64
      %60 = vector.insert %59, %cst_1 [0] : i64 into vector<8xi64>
      %61 = vector.bitcast %60 : vector<8xi64> to vector<16xi32>
      %c2047_i32 = arith.constant 2047 : i32
      %62 = vector.insert %c2047_i32, %61 [2] : i32 into vector<16xi32>
      %63 = vector.insert %c1023_i32, %62 [3] : i32 into vector<16xi32>
      %64 = vector.insert %c2047_i32, %63 [4] : i32 into vector<16xi32>
      %65 = arith.index_castui %c0 : index to i32
      %66 = vector.insert %9, %64 [5] : i32 into vector<16xi32>
      %67 = vector.insert %65, %66 [6] : i32 into vector<16xi32>
      %c3847_i32 = arith.constant 3847 : i32
      %68 = vector.insert %c3847_i32, %67 [7] : i32 into vector<16xi32>
      %69 = vector.insert %19, %64 [5] : i32 into vector<16xi32>
      %70 = vector.insert %65, %69 [6] : i32 into vector<16xi32>
      %71 = vector.insert %c3847_i32, %70 [7] : i32 into vector<16xi32>
      %72 = vector.insert %25, %64 [5] : i32 into vector<16xi32>
      %73 = vector.insert %65, %72 [6] : i32 into vector<16xi32>
      %74 = vector.insert %c3847_i32, %73 [7] : i32 into vector<16xi32>
      %75 = vector.insert %31, %64 [5] : i32 into vector<16xi32>
      %76 = vector.insert %65, %75 [6] : i32 into vector<16xi32>
      %77 = vector.insert %c3847_i32, %76 [7] : i32 into vector<16xi32>
      %78 = arith.index_castui %c16 : index to i32
      %79 = vector.insert %78, %66 [6] : i32 into vector<16xi32>
      %80 = vector.insert %c3847_i32, %79 [7] : i32 into vector<16xi32>
      %81 = vector.insert %78, %69 [6] : i32 into vector<16xi32>
      %82 = vector.insert %c3847_i32, %81 [7] : i32 into vector<16xi32>
      %83 = vector.insert %78, %72 [6] : i32 into vector<16xi32>
      %84 = vector.insert %c3847_i32, %83 [7] : i32 into vector<16xi32>
      %85 = vector.insert %78, %75 [6] : i32 into vector<16xi32>
      %86 = vector.insert %c3847_i32, %85 [7] : i32 into vector<16xi32>
      %intptr_4 = memref.extract_aligned_pointer_as_index %arg1 : memref<1024x1024xf16> -> index
      %87 = arith.index_castui %intptr_4 : index to i64
      %88 = vector.insert %87, %cst_1 [0] : i64 into vector<8xi64>
      %89 = vector.bitcast %88 : vector<8xi64> to vector<16xi32>
      %90 = vector.insert %c2047_i32, %89 [2] : i32 into vector<16xi32>
      %91 = vector.insert %c1023_i32, %90 [3] : i32 into vector<16xi32>
      %92 = vector.insert %c2047_i32, %91 [4] : i32 into vector<16xi32>
      %93 = vector.insert %8, %92 [5] : i32 into vector<16xi32>
      %94 = vector.insert %65, %93 [6] : i32 into vector<16xi32>
      %95 = vector.insert %c7951_i32, %94 [7] : i32 into vector<16xi32>
      %96 = vector.insert %14, %92 [5] : i32 into vector<16xi32>
      %97 = vector.insert %65, %96 [6] : i32 into vector<16xi32>
      %98 = vector.insert %c7951_i32, %97 [7] : i32 into vector<16xi32>
      %99:18 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %68, %arg5 = %71, %arg6 = %74, %arg7 = %77, %arg8 = %80, %arg9 = %82, %arg10 = %84, %arg11 = %86, %arg12 = %95, %arg13 = %98, %arg14 = %44, %arg15 = %52, %arg16 = %46, %arg17 = %54, %arg18 = %48, %arg19 = %56, %arg20 = %50, %arg21 = %58) -> (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>) {
        %cst_5 = arith.constant dense<0.000000e+00> : vector<128xf16>
        %c8_i16_6 = arith.constant 8 : i16
        %101 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %38, %c1_i8, %c8_i16_6, %c16_i16, %arg4, %c0_i32, %c0_i32, %cst_5) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %102 = vector.shape_cast %101 : vector<128xf16> to vector<8x8x2xf16>
        %103 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %38, %c1_i8, %c8_i16_6, %c16_i16, %arg5, %c0_i32, %c0_i32, %cst_5) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %104 = vector.shape_cast %103 : vector<128xf16> to vector<8x8x2xf16>
        %105 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %38, %c1_i8, %c8_i16_6, %c16_i16, %arg6, %c0_i32, %c0_i32, %cst_5) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %106 = vector.shape_cast %105 : vector<128xf16> to vector<8x8x2xf16>
        %107 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %38, %c1_i8, %c8_i16_6, %c16_i16, %arg7, %c0_i32, %c0_i32, %cst_5) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %108 = vector.shape_cast %107 : vector<128xf16> to vector<8x8x2xf16>
        %109 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %38, %c1_i8, %c8_i16_6, %c16_i16, %arg8, %c0_i32, %c0_i32, %cst_5) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %110 = vector.shape_cast %109 : vector<128xf16> to vector<8x8x2xf16>
        %111 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %38, %c1_i8, %c8_i16_6, %c16_i16, %arg9, %c0_i32, %c0_i32, %cst_5) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %112 = vector.shape_cast %111 : vector<128xf16> to vector<8x8x2xf16>
        %113 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %38, %c1_i8, %c8_i16_6, %c16_i16, %arg10, %c0_i32, %c0_i32, %cst_5) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %114 = vector.shape_cast %113 : vector<128xf16> to vector<8x8x2xf16>
        %115 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %38, %c1_i8, %c8_i16_6, %c16_i16, %arg11, %c0_i32, %c0_i32, %cst_5) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %116 = vector.shape_cast %115 : vector<128xf16> to vector<8x8x2xf16>
        %cst_7 = arith.constant dense<0.000000e+00> : vector<512xf16>
        %117 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8(%true, %38, %c1_i8, %c16_i16, %c32_i16, %arg12, %c0_i32, %c0_i32, %cst_7) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf16>) -> vector<512xf16>
        %118 = vector.shape_cast %117 : vector<512xf16> to vector<16x16x2xf16>
        %119 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8(%true, %38, %c1_i8, %c16_i16, %c32_i16, %arg13, %c0_i32, %c0_i32, %cst_7) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf16>) -> vector<512xf16>
        %120 = vector.shape_cast %119 : vector<512xf16> to vector<16x16x2xf16>
        %121 = vector.extract_strided_slice %118 {offsets = [0, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
        %122 = vector.shape_cast %121 : vector<8x16x2xf16> to vector<256xf16>
        %123 = vector.extract_strided_slice %118 {offsets = [8, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
        %124 = vector.shape_cast %123 : vector<8x16x2xf16> to vector<256xf16>
        %125 = vector.extract_strided_slice %120 {offsets = [0, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
        %126 = vector.shape_cast %125 : vector<8x16x2xf16> to vector<256xf16>
        %127 = vector.extract_strided_slice %120 {offsets = [8, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<16x16x2xf16> to vector<8x16x2xf16>
        %128 = vector.shape_cast %127 : vector<8x16x2xf16> to vector<256xf16>
        %129 = vector.shape_cast %102 : vector<8x8x2xf16> to vector<128xf16>
        %130 = vector.bitcast %129 : vector<128xf16> to vector<64xf32>
        %131 = vector.shape_cast %130 : vector<64xf32> to vector<8x8xf32>
        %132 = vector.shape_cast %131 : vector<8x8xf32> to vector<64xf32>
        %alloc = memref.alloc() : memref<4096xf32, 3>
        %133 = gpu.subgroup_id : index
        %134 = arith.muli %133, %c64 : index
        %135 = vector.broadcast %134 : index to vector<8xindex>
        %136 = arith.addi %135, %cst_0 : vector<8xindex>
        %intptr_8 = memref.extract_aligned_pointer_as_index %alloc : memref<4096xf32, 3> -> index
        %137 = arith.index_castui %intptr_8 : index to i32
        %138 = arith.index_castui %136 : vector<8xindex> to vector<8xi32>
        %cst_9 = arith.constant dense<4> : vector<8xi32>
        %139 = arith.muli %138, %cst_9 : vector<8xi32>
        %140 = vector.broadcast %137 : i32 to vector<8xi32>
        %141 = arith.addi %140, %139 : vector<8xi32>
        %c4_i8 = arith.constant 4 : i8
        %c0_i8 = arith.constant 0 : i8
        %c1_i16 = arith.constant 1 : i16
        %c3_i8_10 = arith.constant 3 : i8
        %c5_i8 = arith.constant 5 : i8
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8_10, %c5_i8, %c1_i8, %c0_i8, %141, %132, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %142 = arith.index_castui %134 : index to i32
        %c4_i32 = arith.constant 4 : i32
        %143 = arith.muli %142, %c4_i32 : i32
        %144 = arith.addi %137, %143 : i32
        %145 = vector.broadcast %144 : i32 to vector<1xi32>
        %cst_11 = arith.constant dense<true> : vector<1xi1>
        %c8_i8 = arith.constant 8 : i8
        %146 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_11, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8_10, %c8_i8, %c1_i8, %c0_i8, %145, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %147 = vector.bitcast %146 : vector<64xf32> to vector<128xf16>
        %148 = vector.shape_cast %147 : vector<128xf16> to vector<8x16xf16>
        %149 = vector.shape_cast %148 : vector<8x16xf16> to vector<128xf16>
        %150 = vector.shape_cast %110 : vector<8x8x2xf16> to vector<128xf16>
        %151 = vector.bitcast %150 : vector<128xf16> to vector<64xf32>
        %152 = vector.shape_cast %151 : vector<64xf32> to vector<8x8xf32>
        %153 = vector.shape_cast %152 : vector<8x8xf32> to vector<64xf32>
        %alloc_12 = memref.alloc() : memref<4096xf32, 3>
        %intptr_13 = memref.extract_aligned_pointer_as_index %alloc_12 : memref<4096xf32, 3> -> index
        %154 = arith.index_castui %intptr_13 : index to i32
        %155 = vector.broadcast %154 : i32 to vector<8xi32>
        %156 = arith.addi %155, %139 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8_10, %c5_i8, %c1_i8, %c0_i8, %156, %153, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %157 = arith.addi %154, %143 : i32
        %158 = vector.broadcast %157 : i32 to vector<1xi32>
        %159 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_11, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8_10, %c8_i8, %c1_i8, %c0_i8, %158, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %160 = vector.bitcast %159 : vector<64xf32> to vector<128xf16>
        %161 = vector.shape_cast %160 : vector<128xf16> to vector<8x16xf16>
        %162 = vector.shape_cast %161 : vector<8x16xf16> to vector<128xf16>
        %163 = vector.shape_cast %104 : vector<8x8x2xf16> to vector<128xf16>
        %164 = vector.bitcast %163 : vector<128xf16> to vector<64xf32>
        %165 = vector.shape_cast %164 : vector<64xf32> to vector<8x8xf32>
        %166 = vector.shape_cast %165 : vector<8x8xf32> to vector<64xf32>
        %alloc_14 = memref.alloc() : memref<4096xf32, 3>
        %intptr_15 = memref.extract_aligned_pointer_as_index %alloc_14 : memref<4096xf32, 3> -> index
        %167 = arith.index_castui %intptr_15 : index to i32
        %168 = vector.broadcast %167 : i32 to vector<8xi32>
        %169 = arith.addi %168, %139 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8_10, %c5_i8, %c1_i8, %c0_i8, %169, %166, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %170 = arith.addi %167, %143 : i32
        %171 = vector.broadcast %170 : i32 to vector<1xi32>
        %172 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_11, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8_10, %c8_i8, %c1_i8, %c0_i8, %171, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %173 = vector.bitcast %172 : vector<64xf32> to vector<128xf16>
        %174 = vector.shape_cast %173 : vector<128xf16> to vector<8x16xf16>
        %175 = vector.shape_cast %174 : vector<8x16xf16> to vector<128xf16>
        %176 = vector.shape_cast %112 : vector<8x8x2xf16> to vector<128xf16>
        %177 = vector.bitcast %176 : vector<128xf16> to vector<64xf32>
        %178 = vector.shape_cast %177 : vector<64xf32> to vector<8x8xf32>
        %179 = vector.shape_cast %178 : vector<8x8xf32> to vector<64xf32>
        %alloc_16 = memref.alloc() : memref<4096xf32, 3>
        %intptr_17 = memref.extract_aligned_pointer_as_index %alloc_16 : memref<4096xf32, 3> -> index
        %180 = arith.index_castui %intptr_17 : index to i32
        %181 = vector.broadcast %180 : i32 to vector<8xi32>
        %182 = arith.addi %181, %139 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8_10, %c5_i8, %c1_i8, %c0_i8, %182, %179, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %183 = arith.addi %180, %143 : i32
        %184 = vector.broadcast %183 : i32 to vector<1xi32>
        %185 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_11, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8_10, %c8_i8, %c1_i8, %c0_i8, %184, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %186 = vector.bitcast %185 : vector<64xf32> to vector<128xf16>
        %187 = vector.shape_cast %186 : vector<128xf16> to vector<8x16xf16>
        %188 = vector.shape_cast %187 : vector<8x16xf16> to vector<128xf16>
        %189 = vector.shape_cast %106 : vector<8x8x2xf16> to vector<128xf16>
        %190 = vector.bitcast %189 : vector<128xf16> to vector<64xf32>
        %191 = vector.shape_cast %190 : vector<64xf32> to vector<8x8xf32>
        %192 = vector.shape_cast %191 : vector<8x8xf32> to vector<64xf32>
        %alloc_18 = memref.alloc() : memref<4096xf32, 3>
        %intptr_19 = memref.extract_aligned_pointer_as_index %alloc_18 : memref<4096xf32, 3> -> index
        %193 = arith.index_castui %intptr_19 : index to i32
        %194 = vector.broadcast %193 : i32 to vector<8xi32>
        %195 = arith.addi %194, %139 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8_10, %c5_i8, %c1_i8, %c0_i8, %195, %192, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %196 = arith.addi %193, %143 : i32
        %197 = vector.broadcast %196 : i32 to vector<1xi32>
        %198 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_11, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8_10, %c8_i8, %c1_i8, %c0_i8, %197, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %199 = vector.bitcast %198 : vector<64xf32> to vector<128xf16>
        %200 = vector.shape_cast %199 : vector<128xf16> to vector<8x16xf16>
        %201 = vector.shape_cast %200 : vector<8x16xf16> to vector<128xf16>
        %202 = vector.shape_cast %114 : vector<8x8x2xf16> to vector<128xf16>
        %203 = vector.bitcast %202 : vector<128xf16> to vector<64xf32>
        %204 = vector.shape_cast %203 : vector<64xf32> to vector<8x8xf32>
        %205 = vector.shape_cast %204 : vector<8x8xf32> to vector<64xf32>
        %alloc_20 = memref.alloc() : memref<4096xf32, 3>
        %intptr_21 = memref.extract_aligned_pointer_as_index %alloc_20 : memref<4096xf32, 3> -> index
        %206 = arith.index_castui %intptr_21 : index to i32
        %207 = vector.broadcast %206 : i32 to vector<8xi32>
        %208 = arith.addi %207, %139 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8_10, %c5_i8, %c1_i8, %c0_i8, %208, %205, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %209 = arith.addi %206, %143 : i32
        %210 = vector.broadcast %209 : i32 to vector<1xi32>
        %211 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_11, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8_10, %c8_i8, %c1_i8, %c0_i8, %210, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %212 = vector.bitcast %211 : vector<64xf32> to vector<128xf16>
        %213 = vector.shape_cast %212 : vector<128xf16> to vector<8x16xf16>
        %214 = vector.shape_cast %213 : vector<8x16xf16> to vector<128xf16>
        %215 = vector.shape_cast %108 : vector<8x8x2xf16> to vector<128xf16>
        %216 = vector.bitcast %215 : vector<128xf16> to vector<64xf32>
        %217 = vector.shape_cast %216 : vector<64xf32> to vector<8x8xf32>
        %218 = vector.shape_cast %217 : vector<8x8xf32> to vector<64xf32>
        %alloc_22 = memref.alloc() : memref<4096xf32, 3>
        %intptr_23 = memref.extract_aligned_pointer_as_index %alloc_22 : memref<4096xf32, 3> -> index
        %219 = arith.index_castui %intptr_23 : index to i32
        %220 = vector.broadcast %219 : i32 to vector<8xi32>
        %221 = arith.addi %220, %139 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8_10, %c5_i8, %c1_i8, %c0_i8, %221, %218, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %222 = arith.addi %219, %143 : i32
        %223 = vector.broadcast %222 : i32 to vector<1xi32>
        %224 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_11, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8_10, %c8_i8, %c1_i8, %c0_i8, %223, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %225 = vector.bitcast %224 : vector<64xf32> to vector<128xf16>
        %226 = vector.shape_cast %225 : vector<128xf16> to vector<8x16xf16>
        %227 = vector.shape_cast %226 : vector<8x16xf16> to vector<128xf16>
        %228 = vector.shape_cast %116 : vector<8x8x2xf16> to vector<128xf16>
        %229 = vector.bitcast %228 : vector<128xf16> to vector<64xf32>
        %230 = vector.shape_cast %229 : vector<64xf32> to vector<8x8xf32>
        %231 = vector.shape_cast %230 : vector<8x8xf32> to vector<64xf32>
        %alloc_24 = memref.alloc() : memref<4096xf32, 3>
        %intptr_25 = memref.extract_aligned_pointer_as_index %alloc_24 : memref<4096xf32, 3> -> index
        %232 = arith.index_castui %intptr_25 : index to i32
        %233 = vector.broadcast %232 : i32 to vector<8xi32>
        %234 = arith.addi %233, %139 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8_10, %c5_i8, %c1_i8, %c0_i8, %234, %231, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %235 = arith.addi %232, %143 : i32
        %236 = vector.broadcast %235 : i32 to vector<1xi32>
        %237 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_11, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8_10, %c8_i8, %c1_i8, %c0_i8, %236, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %238 = vector.bitcast %237 : vector<64xf32> to vector<128xf16>
        %239 = vector.shape_cast %238 : vector<128xf16> to vector<8x16xf16>
        %240 = vector.shape_cast %239 : vector<8x16xf16> to vector<128xf16>
        %241 = vector.bitcast %149 : vector<128xf16> to vector<64xi32>
        %242 = vector.bitcast %122 : vector<256xf16> to vector<128xi32>
        %c10_i32 = arith.constant 10 : i32
        %c8_i32 = arith.constant 8 : i32
        %243 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg14, %242, %241, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %244 = vector.bitcast %162 : vector<128xf16> to vector<64xi32>
        %245 = vector.bitcast %124 : vector<256xf16> to vector<128xi32>
        %246 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%243, %245, %244, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %247 = vector.bitcast %126 : vector<256xf16> to vector<128xi32>
        %248 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg15, %247, %241, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %249 = vector.bitcast %128 : vector<256xf16> to vector<128xi32>
        %250 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%248, %249, %244, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %251 = vector.bitcast %175 : vector<128xf16> to vector<64xi32>
        %252 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg16, %242, %251, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %253 = vector.bitcast %188 : vector<128xf16> to vector<64xi32>
        %254 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%252, %245, %253, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %255 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg17, %247, %251, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %256 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%255, %249, %253, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %257 = vector.bitcast %201 : vector<128xf16> to vector<64xi32>
        %258 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg18, %242, %257, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %259 = vector.bitcast %214 : vector<128xf16> to vector<64xi32>
        %260 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%258, %245, %259, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %261 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg19, %247, %257, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %262 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%261, %249, %259, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %263 = vector.bitcast %227 : vector<128xf16> to vector<64xi32>
        %264 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg20, %242, %263, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %265 = vector.bitcast %240 : vector<128xf16> to vector<64xi32>
        %266 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%264, %245, %265, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %267 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg21, %247, %263, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %268 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%267, %249, %265, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %c32_i32 = arith.constant 32 : i32
        %269 = vector.extract %arg4[6] : i32 from vector<16xi32>
        %270 = arith.addi %269, %c32_i32 : i32
        %271 = vector.insert %270, %arg4 [6] : i32 into vector<16xi32>
        %272 = vector.extract %arg5[6] : i32 from vector<16xi32>
        %273 = arith.addi %272, %c32_i32 : i32
        %274 = vector.insert %273, %arg5 [6] : i32 into vector<16xi32>
        %275 = vector.extract %arg6[6] : i32 from vector<16xi32>
        %276 = arith.addi %275, %c32_i32 : i32
        %277 = vector.insert %276, %arg6 [6] : i32 into vector<16xi32>
        %278 = vector.extract %arg7[6] : i32 from vector<16xi32>
        %279 = arith.addi %278, %c32_i32 : i32
        %280 = vector.insert %279, %arg7 [6] : i32 into vector<16xi32>
        %281 = vector.extract %arg8[6] : i32 from vector<16xi32>
        %282 = arith.addi %281, %c32_i32 : i32
        %283 = vector.insert %282, %arg8 [6] : i32 into vector<16xi32>
        %284 = vector.extract %arg9[6] : i32 from vector<16xi32>
        %285 = arith.addi %284, %c32_i32 : i32
        %286 = vector.insert %285, %arg9 [6] : i32 into vector<16xi32>
        %287 = vector.extract %arg10[6] : i32 from vector<16xi32>
        %288 = arith.addi %287, %c32_i32 : i32
        %289 = vector.insert %288, %arg10 [6] : i32 into vector<16xi32>
        %290 = vector.extract %arg11[6] : i32 from vector<16xi32>
        %291 = arith.addi %290, %c32_i32 : i32
        %292 = vector.insert %291, %arg11 [6] : i32 into vector<16xi32>
        %293 = vector.extract %arg12[6] : i32 from vector<16xi32>
        %294 = arith.addi %293, %c32_i32 : i32
        %295 = vector.insert %294, %arg12 [6] : i32 into vector<16xi32>
        %296 = vector.extract %arg13[6] : i32 from vector<16xi32>
        %297 = arith.addi %296, %c32_i32 : i32
        %298 = vector.insert %297, %arg13 [6] : i32 into vector<16xi32>
        scf.yield %271, %274, %277, %280, %283, %286, %289, %292, %295, %298, %246, %250, %254, %256, %260, %262, %266, %268 : vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>
      }
      %c3_i8 = arith.constant 3 : i8
      %100 = vector.from_elements %c3_i8, %c3_i8 : vector<2xi8>
      %c8_i16 = arith.constant 8 : i16
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %100, %c1_i8, %c16_i16, %c8_i16, %12, %c0_i32, %c0_i32, %99#10) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %100, %c1_i8, %c16_i16, %c8_i16, %17, %c0_i32, %c0_i32, %99#11) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %100, %c1_i8, %c16_i16, %c8_i16, %21, %c0_i32, %c0_i32, %99#12) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %100, %c1_i8, %c16_i16, %c8_i16, %23, %c0_i32, %c0_i32, %99#13) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %100, %c1_i8, %c16_i16, %c8_i16, %27, %c0_i32, %c0_i32, %99#14) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %100, %c1_i8, %c16_i16, %c8_i16, %29, %c0_i32, %c0_i32, %99#15) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %100, %c1_i8, %c16_i16, %c8_i16, %33, %c0_i32, %c0_i32, %99#16) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %100, %c1_i8, %c16_i16, %c8_i16, %35, %c0_i32, %c0_i32, %99#17) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 1.000000e+00 : f16
    %alloc = memref.alloc() : memref<1024x1024xf16>
    %alloc_1 = memref.alloc() : memref<1024x1024xf16>
    %alloc_2 = memref.alloc() : memref<1024x1024xf32>
    %alloc_3 = memref.alloc() : memref<1024x1024xf32>
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = index.castu %arg1 : index to i16
        %2 = arith.uitofp %1 : i16 to f16
        memref.store %2, %alloc[%arg0, %arg1] : memref<1024x1024xf16>
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = index.castu %arg0 : index to i32
        %2 = index.castu %arg1 : index to i32
        %3 = arith.cmpi eq, %1, %2 : i32
        scf.if %3 {
          memref.store %cst_0, %alloc_1[%arg0, %arg1] : memref<1024x1024xf16>
        } else {
          memref.store %cst, %alloc_1[%arg0, %arg1] : memref<1024x1024xf16>
        }
      }
    }
    %cst_4 = arith.constant 0.000000e+00 : f32
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        memref.store %cst_4, %alloc_2[%arg0, %arg1] : memref<1024x1024xf32>
        memref.store %cst_4, %alloc_3[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = memref.load %alloc_3[%arg0, %arg1] : memref<1024x1024xf32>
        %2 = scf.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %1) -> (f32) {
          %3 = memref.load %alloc[%arg2, %arg0] : memref<1024x1024xf16>
          %4 = memref.load %alloc_1[%arg2, %arg1] : memref<1024x1024xf16>
          %5 = arith.mulf %3, %4 : f16
          %6 = arith.extf %5 : f16 to f32
          %7 = arith.addf %6, %arg3 : f32
          scf.yield %7 : f32
        }
        memref.store %2, %alloc_3[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    %0 = call @test(%alloc, %alloc_1, %alloc_2) : (memref<1024x1024xf16>, memref<1024x1024xf16>, memref<1024x1024xf32>) -> memref<1024x1024xf32>
    %cast = memref.cast %0 : memref<1024x1024xf32> to memref<*xf32>
    %cast_5 = memref.cast %alloc_3 : memref<1024x1024xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_5) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<1024x1024xf16>
    memref.dealloc %alloc_1 : memref<1024x1024xf16>
    memref.dealloc %alloc_2 : memref<1024x1024xf32>
    memref.dealloc %alloc_3 : memref<1024x1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}


// -----// IR Dump After VectorLinearize (imex-vector-linearize) //----- //
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %memref = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %arg0, %memref : memref<1024x1024xf16> to memref<1024x1024xf16>
    %memref_0 = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %arg1, %memref_0 : memref<1024x1024xf16> to memref<1024x1024xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<1024x1024xf32>
    memref.copy %arg2, %memref_1 : memref<1024x1024xf32> to memref<1024x1024xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c32, %c32, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<1024x1024xf16>, %memref_0 : memref<1024x1024xf16>, %memref_1 : memref<1024x1024xf32>)
    gpu.dealloc  %memref : memref<1024x1024xf16>
    gpu.dealloc  %memref_0 : memref<1024x1024xf16>
    return %memref_1 : memref<1024x1024xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    func.func private @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32", linkage_type = <Import>>}
    func.func private @llvm.genx.dpas2.v128f32.v128i32.v64i32(vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas2.v128f32.v128i32.v64i32", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf16>) -> vector<512xf16> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf32>) -> vector<512xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8", linkage_type = <Import>>}
    gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c16_i32 = arith.constant 16 : i32
      %c32_i32 = arith.constant 32 : i32
      %c8_i32 = arith.constant 8 : i32
      %c10_i32 = arith.constant 10 : i32
      %c8_i8 = arith.constant 8 : i8
      %cst = arith.constant dense<true> : vector<1xi1>
      %c4_i32 = arith.constant 4 : i32
      %c5_i8 = arith.constant 5 : i8
      %c3_i8 = arith.constant 3 : i8
      %c1_i16 = arith.constant 1 : i16
      %c0_i8 = arith.constant 0 : i8
      %c4_i8 = arith.constant 4 : i8
      %cst_0 = arith.constant dense<4> : vector<8xi32>
      %cst_1 = arith.constant dense<0.000000e+00> : vector<512xf16>
      %c8_i16 = arith.constant 8 : i16
      %cst_2 = arith.constant dense<0.000000e+00> : vector<128xf16>
      %c3847_i32 = arith.constant 3847 : i32
      %c2047_i32 = arith.constant 2047 : i32
      %c0_i32 = arith.constant 0 : i32
      %c32_i16 = arith.constant 32 : i16
      %c16_i16 = arith.constant 16 : i16
      %c1_i8 = arith.constant 1 : i8
      %c2_i8 = arith.constant 2 : i8
      %true = arith.constant true
      %cst_3 = arith.constant dense<0.000000e+00> : vector<512xf32>
      %c7951_i32 = arith.constant 7951 : i32
      %c1807_i32 = arith.constant 1807 : i32
      %c1023_i32 = arith.constant 1023 : i32
      %c4095_i32 = arith.constant 4095 : i32
      %cst_4 = arith.constant dense<0> : vector<8xi64>
      %cst_5 = arith.constant dense<true> : vector<8xi1>
      %cst_6 = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56]> : vector<8xindex>
      %c64 = arith.constant 64 : index
      %c24 = arith.constant 24 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c32 : index
      %1 = arith.muli %block_id_y, %c32 : index
      %intptr = memref.extract_aligned_pointer_as_index %arg2 : memref<1024x1024xf32> -> index
      %2 = arith.index_castui %intptr : index to i64
      %3 = vector.insert %2, %cst_4 [0] : i64 into vector<8xi64>
      %4 = vector.bitcast %3 : vector<8xi64> to vector<16xi32>
      %5 = vector.insert %c4095_i32, %4 [2] : i32 into vector<16xi32>
      %6 = vector.insert %c1023_i32, %5 [3] : i32 into vector<16xi32>
      %7 = vector.insert %c4095_i32, %6 [4] : i32 into vector<16xi32>
      %8 = arith.index_castui %1 : index to i32
      %9 = arith.index_castui %0 : index to i32
      %10 = vector.insert %8, %7 [5] : i32 into vector<16xi32>
      %11 = vector.insert %9, %10 [6] : i32 into vector<16xi32>
      %12 = vector.insert %c1807_i32, %11 [7] : i32 into vector<16xi32>
      %13 = arith.addi %1, %c16 : index
      %14 = arith.index_castui %13 : index to i32
      %15 = vector.insert %14, %7 [5] : i32 into vector<16xi32>
      %16 = vector.insert %9, %15 [6] : i32 into vector<16xi32>
      %17 = vector.insert %c1807_i32, %16 [7] : i32 into vector<16xi32>
      %18 = arith.addi %0, %c8 : index
      %19 = arith.index_castui %18 : index to i32
      %20 = vector.insert %19, %10 [6] : i32 into vector<16xi32>
      %21 = vector.insert %c1807_i32, %20 [7] : i32 into vector<16xi32>
      %22 = vector.insert %19, %15 [6] : i32 into vector<16xi32>
      %23 = vector.insert %c1807_i32, %22 [7] : i32 into vector<16xi32>
      %24 = arith.addi %0, %c16 : index
      %25 = arith.index_castui %24 : index to i32
      %26 = vector.insert %25, %10 [6] : i32 into vector<16xi32>
      %27 = vector.insert %c1807_i32, %26 [7] : i32 into vector<16xi32>
      %28 = vector.insert %25, %15 [6] : i32 into vector<16xi32>
      %29 = vector.insert %c1807_i32, %28 [7] : i32 into vector<16xi32>
      %30 = arith.addi %0, %c24 : index
      %31 = arith.index_castui %30 : index to i32
      %32 = vector.insert %31, %10 [6] : i32 into vector<16xi32>
      %33 = vector.insert %c1807_i32, %32 [7] : i32 into vector<16xi32>
      %34 = vector.insert %31, %15 [6] : i32 into vector<16xi32>
      %35 = vector.insert %c1807_i32, %34 [7] : i32 into vector<16xi32>
      %36 = vector.insert %c7951_i32, %11 [7] : i32 into vector<16xi32>
      %37 = vector.insert %c7951_i32, %16 [7] : i32 into vector<16xi32>
      %38 = vector.from_elements %c2_i8, %c2_i8 : vector<2xi8>
      %39 = func.call @llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8(%true, %38, %c1_i8, %c16_i16, %c32_i16, %36, %c0_i32, %c0_i32, %cst_3) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf32>) -> vector<512xf32>
      %40 = vector.shape_cast %39 : vector<512xf32> to vector<32x16xf32>
      %41 = vector.shape_cast %40 : vector<32x16xf32> to vector<512xf32>
      %42 = func.call @llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8(%true, %38, %c1_i8, %c16_i16, %c32_i16, %37, %c0_i32, %c0_i32, %cst_3) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf32>) -> vector<512xf32>
      %43 = vector.shape_cast %42 : vector<512xf32> to vector<32x16xf32>
      %44 = vector.shape_cast %43 : vector<32x16xf32> to vector<512xf32>
      %45 = vector.shuffle %41, %41 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<512xf32>, vector<512xf32>
      %46 = vector.shape_cast %45 : vector<128xf32> to vector<8x16xf32>
      %47 = vector.shape_cast %46 : vector<8x16xf32> to vector<128xf32>
      %48 = vector.shuffle %41, %41 [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<512xf32>, vector<512xf32>
      %49 = vector.shape_cast %48 : vector<128xf32> to vector<8x16xf32>
      %50 = vector.shape_cast %49 : vector<8x16xf32> to vector<128xf32>
      %51 = vector.shuffle %41, %41 [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383] : vector<512xf32>, vector<512xf32>
      %52 = vector.shape_cast %51 : vector<128xf32> to vector<8x16xf32>
      %53 = vector.shape_cast %52 : vector<8x16xf32> to vector<128xf32>
      %54 = vector.shuffle %41, %41 [384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xf32>, vector<512xf32>
      %55 = vector.shape_cast %54 : vector<128xf32> to vector<8x16xf32>
      %56 = vector.shape_cast %55 : vector<8x16xf32> to vector<128xf32>
      %57 = vector.shuffle %44, %44 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<512xf32>, vector<512xf32>
      %58 = vector.shape_cast %57 : vector<128xf32> to vector<8x16xf32>
      %59 = vector.shape_cast %58 : vector<8x16xf32> to vector<128xf32>
      %60 = vector.shuffle %44, %44 [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<512xf32>, vector<512xf32>
      %61 = vector.shape_cast %60 : vector<128xf32> to vector<8x16xf32>
      %62 = vector.shape_cast %61 : vector<8x16xf32> to vector<128xf32>
      %63 = vector.shuffle %44, %44 [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383] : vector<512xf32>, vector<512xf32>
      %64 = vector.shape_cast %63 : vector<128xf32> to vector<8x16xf32>
      %65 = vector.shape_cast %64 : vector<8x16xf32> to vector<128xf32>
      %66 = vector.shuffle %44, %44 [384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xf32>, vector<512xf32>
      %67 = vector.shape_cast %66 : vector<128xf32> to vector<8x16xf32>
      %68 = vector.shape_cast %67 : vector<8x16xf32> to vector<128xf32>
      %intptr_7 = memref.extract_aligned_pointer_as_index %arg0 : memref<1024x1024xf16> -> index
      %69 = arith.index_castui %intptr_7 : index to i64
      %70 = vector.insert %69, %cst_4 [0] : i64 into vector<8xi64>
      %71 = vector.bitcast %70 : vector<8xi64> to vector<16xi32>
      %72 = vector.insert %c2047_i32, %71 [2] : i32 into vector<16xi32>
      %73 = vector.insert %c1023_i32, %72 [3] : i32 into vector<16xi32>
      %74 = vector.insert %c2047_i32, %73 [4] : i32 into vector<16xi32>
      %75 = vector.insert %9, %74 [5] : i32 into vector<16xi32>
      %76 = vector.insert %c0_i32, %75 [6] : i32 into vector<16xi32>
      %77 = vector.insert %c3847_i32, %76 [7] : i32 into vector<16xi32>
      %78 = vector.insert %19, %74 [5] : i32 into vector<16xi32>
      %79 = vector.insert %c0_i32, %78 [6] : i32 into vector<16xi32>
      %80 = vector.insert %c3847_i32, %79 [7] : i32 into vector<16xi32>
      %81 = vector.insert %25, %74 [5] : i32 into vector<16xi32>
      %82 = vector.insert %c0_i32, %81 [6] : i32 into vector<16xi32>
      %83 = vector.insert %c3847_i32, %82 [7] : i32 into vector<16xi32>
      %84 = vector.insert %31, %74 [5] : i32 into vector<16xi32>
      %85 = vector.insert %c0_i32, %84 [6] : i32 into vector<16xi32>
      %86 = vector.insert %c3847_i32, %85 [7] : i32 into vector<16xi32>
      %87 = vector.insert %c16_i32, %75 [6] : i32 into vector<16xi32>
      %88 = vector.insert %c3847_i32, %87 [7] : i32 into vector<16xi32>
      %89 = vector.insert %c16_i32, %78 [6] : i32 into vector<16xi32>
      %90 = vector.insert %c3847_i32, %89 [7] : i32 into vector<16xi32>
      %91 = vector.insert %c16_i32, %81 [6] : i32 into vector<16xi32>
      %92 = vector.insert %c3847_i32, %91 [7] : i32 into vector<16xi32>
      %93 = vector.insert %c16_i32, %84 [6] : i32 into vector<16xi32>
      %94 = vector.insert %c3847_i32, %93 [7] : i32 into vector<16xi32>
      %intptr_8 = memref.extract_aligned_pointer_as_index %arg1 : memref<1024x1024xf16> -> index
      %95 = arith.index_castui %intptr_8 : index to i64
      %96 = vector.insert %95, %cst_4 [0] : i64 into vector<8xi64>
      %97 = vector.bitcast %96 : vector<8xi64> to vector<16xi32>
      %98 = vector.insert %c2047_i32, %97 [2] : i32 into vector<16xi32>
      %99 = vector.insert %c1023_i32, %98 [3] : i32 into vector<16xi32>
      %100 = vector.insert %c2047_i32, %99 [4] : i32 into vector<16xi32>
      %101 = vector.insert %8, %100 [5] : i32 into vector<16xi32>
      %102 = vector.insert %c0_i32, %101 [6] : i32 into vector<16xi32>
      %103 = vector.insert %c7951_i32, %102 [7] : i32 into vector<16xi32>
      %104 = vector.insert %14, %100 [5] : i32 into vector<16xi32>
      %105 = vector.insert %c0_i32, %104 [6] : i32 into vector<16xi32>
      %106 = vector.insert %c7951_i32, %105 [7] : i32 into vector<16xi32>
      %107:18 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %77, %arg5 = %80, %arg6 = %83, %arg7 = %86, %arg8 = %88, %arg9 = %90, %arg10 = %92, %arg11 = %94, %arg12 = %103, %arg13 = %106, %arg14 = %47, %arg15 = %59, %arg16 = %50, %arg17 = %62, %arg18 = %53, %arg19 = %65, %arg20 = %56, %arg21 = %68) -> (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>) {
        %109 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %38, %c1_i8, %c8_i16, %c16_i16, %arg4, %c0_i32, %c0_i32, %cst_2) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %110 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %38, %c1_i8, %c8_i16, %c16_i16, %arg5, %c0_i32, %c0_i32, %cst_2) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %111 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %38, %c1_i8, %c8_i16, %c16_i16, %arg6, %c0_i32, %c0_i32, %cst_2) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %112 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %38, %c1_i8, %c8_i16, %c16_i16, %arg7, %c0_i32, %c0_i32, %cst_2) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %113 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %38, %c1_i8, %c8_i16, %c16_i16, %arg8, %c0_i32, %c0_i32, %cst_2) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %114 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %38, %c1_i8, %c8_i16, %c16_i16, %arg9, %c0_i32, %c0_i32, %cst_2) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %115 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %38, %c1_i8, %c8_i16, %c16_i16, %arg10, %c0_i32, %c0_i32, %cst_2) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %116 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %38, %c1_i8, %c8_i16, %c16_i16, %arg11, %c0_i32, %c0_i32, %cst_2) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %117 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8(%true, %38, %c1_i8, %c16_i16, %c32_i16, %arg12, %c0_i32, %c0_i32, %cst_1) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf16>) -> vector<512xf16>
        %118 = vector.shape_cast %117 : vector<512xf16> to vector<16x16x2xf16>
        %119 = vector.shape_cast %118 : vector<16x16x2xf16> to vector<512xf16>
        %120 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8(%true, %38, %c1_i8, %c16_i16, %c32_i16, %arg13, %c0_i32, %c0_i32, %cst_1) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf16>) -> vector<512xf16>
        %121 = vector.shape_cast %120 : vector<512xf16> to vector<16x16x2xf16>
        %122 = vector.shape_cast %121 : vector<16x16x2xf16> to vector<512xf16>
        %123 = vector.shuffle %119, %119 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<512xf16>, vector<512xf16>
        %124 = vector.shape_cast %123 : vector<256xf16> to vector<8x16x2xf16>
        %125 = vector.shape_cast %124 : vector<8x16x2xf16> to vector<256xf16>
        %126 = vector.shuffle %119, %119 [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xf16>, vector<512xf16>
        %127 = vector.shape_cast %126 : vector<256xf16> to vector<8x16x2xf16>
        %128 = vector.shape_cast %127 : vector<8x16x2xf16> to vector<256xf16>
        %129 = vector.shuffle %122, %122 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<512xf16>, vector<512xf16>
        %130 = vector.shape_cast %129 : vector<256xf16> to vector<8x16x2xf16>
        %131 = vector.shape_cast %130 : vector<8x16x2xf16> to vector<256xf16>
        %132 = vector.shuffle %122, %122 [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xf16>, vector<512xf16>
        %133 = vector.shape_cast %132 : vector<256xf16> to vector<8x16x2xf16>
        %134 = vector.shape_cast %133 : vector<8x16x2xf16> to vector<256xf16>
        %135 = vector.bitcast %109 : vector<128xf16> to vector<64xf32>
        %alloc = memref.alloc() : memref<4096xf32, 3>
        %136 = gpu.subgroup_id : index
        %137 = arith.muli %136, %c64 : index
        %138 = vector.splat %137 : vector<8xindex>
        %139 = arith.addi %138, %cst_6 : vector<8xindex>
        %intptr_9 = memref.extract_aligned_pointer_as_index %alloc : memref<4096xf32, 3> -> index
        %140 = arith.index_castui %intptr_9 : index to i32
        %141 = arith.index_castui %139 : vector<8xindex> to vector<8xi32>
        %142 = arith.muli %141, %cst_0 : vector<8xi32>
        %143 = vector.splat %140 : vector<8xi32>
        %144 = arith.addi %143, %142 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_5, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %144, %135, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %145 = arith.index_castui %137 : index to i32
        %146 = arith.muli %145, %c4_i32 : i32
        %147 = arith.addi %140, %146 : i32
        %148 = vector.splat %147 : vector<1xi32>
        %149 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %148, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %150 = vector.bitcast %113 : vector<128xf16> to vector<64xf32>
        %alloc_10 = memref.alloc() : memref<4096xf32, 3>
        %intptr_11 = memref.extract_aligned_pointer_as_index %alloc_10 : memref<4096xf32, 3> -> index
        %151 = arith.index_castui %intptr_11 : index to i32
        %152 = vector.splat %151 : vector<8xi32>
        %153 = arith.addi %152, %142 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_5, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %153, %150, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %154 = arith.addi %151, %146 : i32
        %155 = vector.splat %154 : vector<1xi32>
        %156 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %155, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %157 = vector.bitcast %110 : vector<128xf16> to vector<64xf32>
        %alloc_12 = memref.alloc() : memref<4096xf32, 3>
        %intptr_13 = memref.extract_aligned_pointer_as_index %alloc_12 : memref<4096xf32, 3> -> index
        %158 = arith.index_castui %intptr_13 : index to i32
        %159 = vector.splat %158 : vector<8xi32>
        %160 = arith.addi %159, %142 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_5, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %160, %157, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %161 = arith.addi %158, %146 : i32
        %162 = vector.splat %161 : vector<1xi32>
        %163 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %162, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %164 = vector.bitcast %114 : vector<128xf16> to vector<64xf32>
        %alloc_14 = memref.alloc() : memref<4096xf32, 3>
        %intptr_15 = memref.extract_aligned_pointer_as_index %alloc_14 : memref<4096xf32, 3> -> index
        %165 = arith.index_castui %intptr_15 : index to i32
        %166 = vector.splat %165 : vector<8xi32>
        %167 = arith.addi %166, %142 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_5, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %167, %164, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %168 = arith.addi %165, %146 : i32
        %169 = vector.splat %168 : vector<1xi32>
        %170 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %169, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %171 = vector.bitcast %111 : vector<128xf16> to vector<64xf32>
        %alloc_16 = memref.alloc() : memref<4096xf32, 3>
        %intptr_17 = memref.extract_aligned_pointer_as_index %alloc_16 : memref<4096xf32, 3> -> index
        %172 = arith.index_castui %intptr_17 : index to i32
        %173 = vector.splat %172 : vector<8xi32>
        %174 = arith.addi %173, %142 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_5, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %174, %171, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %175 = arith.addi %172, %146 : i32
        %176 = vector.splat %175 : vector<1xi32>
        %177 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %176, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %178 = vector.bitcast %115 : vector<128xf16> to vector<64xf32>
        %alloc_18 = memref.alloc() : memref<4096xf32, 3>
        %intptr_19 = memref.extract_aligned_pointer_as_index %alloc_18 : memref<4096xf32, 3> -> index
        %179 = arith.index_castui %intptr_19 : index to i32
        %180 = vector.splat %179 : vector<8xi32>
        %181 = arith.addi %180, %142 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_5, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %181, %178, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %182 = arith.addi %179, %146 : i32
        %183 = vector.splat %182 : vector<1xi32>
        %184 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %183, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %185 = vector.bitcast %112 : vector<128xf16> to vector<64xf32>
        %alloc_20 = memref.alloc() : memref<4096xf32, 3>
        %intptr_21 = memref.extract_aligned_pointer_as_index %alloc_20 : memref<4096xf32, 3> -> index
        %186 = arith.index_castui %intptr_21 : index to i32
        %187 = vector.splat %186 : vector<8xi32>
        %188 = arith.addi %187, %142 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_5, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %188, %185, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %189 = arith.addi %186, %146 : i32
        %190 = vector.splat %189 : vector<1xi32>
        %191 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %190, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %192 = vector.bitcast %116 : vector<128xf16> to vector<64xf32>
        %alloc_22 = memref.alloc() : memref<4096xf32, 3>
        %intptr_23 = memref.extract_aligned_pointer_as_index %alloc_22 : memref<4096xf32, 3> -> index
        %193 = arith.index_castui %intptr_23 : index to i32
        %194 = vector.splat %193 : vector<8xi32>
        %195 = arith.addi %194, %142 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_5, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %195, %192, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %196 = arith.addi %193, %146 : i32
        %197 = vector.splat %196 : vector<1xi32>
        %198 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %197, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %199 = vector.bitcast %149 : vector<64xf32> to vector<64xi32>
        %200 = vector.bitcast %125 : vector<256xf16> to vector<128xi32>
        %201 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg14, %200, %199, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %202 = vector.bitcast %156 : vector<64xf32> to vector<64xi32>
        %203 = vector.bitcast %128 : vector<256xf16> to vector<128xi32>
        %204 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%201, %203, %202, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %205 = vector.bitcast %131 : vector<256xf16> to vector<128xi32>
        %206 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg15, %205, %199, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %207 = vector.bitcast %134 : vector<256xf16> to vector<128xi32>
        %208 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%206, %207, %202, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %209 = vector.bitcast %163 : vector<64xf32> to vector<64xi32>
        %210 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg16, %200, %209, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %211 = vector.bitcast %170 : vector<64xf32> to vector<64xi32>
        %212 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%210, %203, %211, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %213 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg17, %205, %209, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %214 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%213, %207, %211, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %215 = vector.bitcast %177 : vector<64xf32> to vector<64xi32>
        %216 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg18, %200, %215, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %217 = vector.bitcast %184 : vector<64xf32> to vector<64xi32>
        %218 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%216, %203, %217, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %219 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg19, %205, %215, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %220 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%219, %207, %217, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %221 = vector.bitcast %191 : vector<64xf32> to vector<64xi32>
        %222 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg20, %200, %221, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %223 = vector.bitcast %198 : vector<64xf32> to vector<64xi32>
        %224 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%222, %203, %223, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %225 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg21, %205, %221, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %226 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%225, %207, %223, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %c6_i32 = arith.constant 6 : i32
        %227 = vector.extractelement %arg4[%c6_i32 : i32] : vector<16xi32>
        %228 = arith.addi %227, %c32_i32 : i32
        %229 = vector.insert %228, %arg4 [6] : i32 into vector<16xi32>
        %c6_i32_24 = arith.constant 6 : i32
        %230 = vector.extractelement %arg5[%c6_i32_24 : i32] : vector<16xi32>
        %231 = arith.addi %230, %c32_i32 : i32
        %232 = vector.insert %231, %arg5 [6] : i32 into vector<16xi32>
        %c6_i32_25 = arith.constant 6 : i32
        %233 = vector.extractelement %arg6[%c6_i32_25 : i32] : vector<16xi32>
        %234 = arith.addi %233, %c32_i32 : i32
        %235 = vector.insert %234, %arg6 [6] : i32 into vector<16xi32>
        %c6_i32_26 = arith.constant 6 : i32
        %236 = vector.extractelement %arg7[%c6_i32_26 : i32] : vector<16xi32>
        %237 = arith.addi %236, %c32_i32 : i32
        %238 = vector.insert %237, %arg7 [6] : i32 into vector<16xi32>
        %c6_i32_27 = arith.constant 6 : i32
        %239 = vector.extractelement %arg8[%c6_i32_27 : i32] : vector<16xi32>
        %240 = arith.addi %239, %c32_i32 : i32
        %241 = vector.insert %240, %arg8 [6] : i32 into vector<16xi32>
        %c6_i32_28 = arith.constant 6 : i32
        %242 = vector.extractelement %arg9[%c6_i32_28 : i32] : vector<16xi32>
        %243 = arith.addi %242, %c32_i32 : i32
        %244 = vector.insert %243, %arg9 [6] : i32 into vector<16xi32>
        %c6_i32_29 = arith.constant 6 : i32
        %245 = vector.extractelement %arg10[%c6_i32_29 : i32] : vector<16xi32>
        %246 = arith.addi %245, %c32_i32 : i32
        %247 = vector.insert %246, %arg10 [6] : i32 into vector<16xi32>
        %c6_i32_30 = arith.constant 6 : i32
        %248 = vector.extractelement %arg11[%c6_i32_30 : i32] : vector<16xi32>
        %249 = arith.addi %248, %c32_i32 : i32
        %250 = vector.insert %249, %arg11 [6] : i32 into vector<16xi32>
        %c6_i32_31 = arith.constant 6 : i32
        %251 = vector.extractelement %arg12[%c6_i32_31 : i32] : vector<16xi32>
        %252 = arith.addi %251, %c32_i32 : i32
        %253 = vector.insert %252, %arg12 [6] : i32 into vector<16xi32>
        %c6_i32_32 = arith.constant 6 : i32
        %254 = vector.extractelement %arg13[%c6_i32_32 : i32] : vector<16xi32>
        %255 = arith.addi %254, %c32_i32 : i32
        %256 = vector.insert %255, %arg13 [6] : i32 into vector<16xi32>
        scf.yield %229, %232, %235, %238, %241, %244, %247, %250, %253, %256, %204, %208, %212, %214, %218, %220, %224, %226 : vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>
      }
      %108 = vector.from_elements %c3_i8, %c3_i8 : vector<2xi8>
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %108, %c1_i8, %c16_i16, %c8_i16, %12, %c0_i32, %c0_i32, %107#10) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %108, %c1_i8, %c16_i16, %c8_i16, %17, %c0_i32, %c0_i32, %107#11) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %108, %c1_i8, %c16_i16, %c8_i16, %21, %c0_i32, %c0_i32, %107#12) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %108, %c1_i8, %c16_i16, %c8_i16, %23, %c0_i32, %c0_i32, %107#13) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %108, %c1_i8, %c16_i16, %c8_i16, %27, %c0_i32, %c0_i32, %107#14) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %108, %c1_i8, %c16_i16, %c8_i16, %29, %c0_i32, %c0_i32, %107#15) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %108, %c1_i8, %c16_i16, %c8_i16, %33, %c0_i32, %c0_i32, %107#16) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %108, %c1_i8, %c16_i16, %c8_i16, %35, %c0_i32, %c0_i32, %107#17) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %cst_0 = arith.constant 0.000000e+00 : f16
    %cst_1 = arith.constant 1.000000e+00 : f16
    %alloc = memref.alloc() : memref<1024x1024xf16>
    %alloc_2 = memref.alloc() : memref<1024x1024xf16>
    %alloc_3 = memref.alloc() : memref<1024x1024xf32>
    %alloc_4 = memref.alloc() : memref<1024x1024xf32>
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = index.castu %arg1 : index to i16
        %2 = arith.uitofp %1 : i16 to f16
        memref.store %2, %alloc[%arg0, %arg1] : memref<1024x1024xf16>
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = index.castu %arg0 : index to i32
        %2 = index.castu %arg1 : index to i32
        %3 = arith.cmpi eq, %1, %2 : i32
        scf.if %3 {
          memref.store %cst_1, %alloc_2[%arg0, %arg1] : memref<1024x1024xf16>
        } else {
          memref.store %cst_0, %alloc_2[%arg0, %arg1] : memref<1024x1024xf16>
        }
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        memref.store %cst, %alloc_3[%arg0, %arg1] : memref<1024x1024xf32>
        memref.store %cst, %alloc_4[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = memref.load %alloc_4[%arg0, %arg1] : memref<1024x1024xf32>
        %2 = scf.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %1) -> (f32) {
          %3 = memref.load %alloc[%arg2, %arg0] : memref<1024x1024xf16>
          %4 = memref.load %alloc_2[%arg2, %arg1] : memref<1024x1024xf16>
          %5 = arith.mulf %3, %4 : f16
          %6 = arith.extf %5 : f16 to f32
          %7 = arith.addf %6, %arg3 : f32
          scf.yield %7 : f32
        }
        memref.store %2, %alloc_4[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    %0 = call @test(%alloc, %alloc_2, %alloc_3) : (memref<1024x1024xf16>, memref<1024x1024xf16>, memref<1024x1024xf32>) -> memref<1024x1024xf32>
    %cast = memref.cast %0 : memref<1024x1024xf32> to memref<*xf32>
    %cast_5 = memref.cast %alloc_4 : memref<1024x1024xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_5) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<1024x1024xf16>
    memref.dealloc %alloc_2 : memref<1024x1024xf16>
    memref.dealloc %alloc_3 : memref<1024x1024xf32>
    memref.dealloc %alloc_4 : memref<1024x1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %memref = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %arg0, %memref : memref<1024x1024xf16> to memref<1024x1024xf16>
    %memref_0 = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %arg1, %memref_0 : memref<1024x1024xf16> to memref<1024x1024xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<1024x1024xf32>
    memref.copy %arg2, %memref_1 : memref<1024x1024xf32> to memref<1024x1024xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c32, %c32, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<1024x1024xf16>, %memref_0 : memref<1024x1024xf16>, %memref_1 : memref<1024x1024xf32>)
    gpu.dealloc  %memref : memref<1024x1024xf16>
    gpu.dealloc  %memref_0 : memref<1024x1024xf16>
    return %memref_1 : memref<1024x1024xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    func.func private @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32", linkage_type = <Import>>}
    func.func private @llvm.genx.dpas2.v128f32.v128i32.v64i32(vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas2.v128f32.v128i32.v64i32", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf16>) -> vector<512xf16> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf32>) -> vector<512xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8", linkage_type = <Import>>}
    gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %cst = arith.constant dense<3> : vector<2xi8>
      %cst_0 = arith.constant dense<2> : vector<2xi8>
      %c6_i32 = arith.constant 6 : i32
      %c16_i32 = arith.constant 16 : i32
      %c32_i32 = arith.constant 32 : i32
      %c8_i32 = arith.constant 8 : i32
      %c10_i32 = arith.constant 10 : i32
      %c8_i8 = arith.constant 8 : i8
      %cst_1 = arith.constant dense<true> : vector<1xi1>
      %c4_i32 = arith.constant 4 : i32
      %c5_i8 = arith.constant 5 : i8
      %c3_i8 = arith.constant 3 : i8
      %c1_i16 = arith.constant 1 : i16
      %c0_i8 = arith.constant 0 : i8
      %c4_i8 = arith.constant 4 : i8
      %cst_2 = arith.constant dense<4> : vector<8xi32>
      %cst_3 = arith.constant dense<0.000000e+00> : vector<512xf16>
      %c8_i16 = arith.constant 8 : i16
      %cst_4 = arith.constant dense<0.000000e+00> : vector<128xf16>
      %c3847_i32 = arith.constant 3847 : i32
      %c2047_i32 = arith.constant 2047 : i32
      %c0_i32 = arith.constant 0 : i32
      %c32_i16 = arith.constant 32 : i16
      %c16_i16 = arith.constant 16 : i16
      %c1_i8 = arith.constant 1 : i8
      %true = arith.constant true
      %cst_5 = arith.constant dense<0.000000e+00> : vector<512xf32>
      %c7951_i32 = arith.constant 7951 : i32
      %c1807_i32 = arith.constant 1807 : i32
      %c1023_i32 = arith.constant 1023 : i32
      %c4095_i32 = arith.constant 4095 : i32
      %cst_6 = arith.constant dense<0> : vector<8xi64>
      %cst_7 = arith.constant dense<true> : vector<8xi1>
      %cst_8 = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56]> : vector<8xindex>
      %c64 = arith.constant 64 : index
      %c24 = arith.constant 24 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c32 : index
      %1 = arith.muli %block_id_y, %c32 : index
      %intptr = memref.extract_aligned_pointer_as_index %arg2 : memref<1024x1024xf32> -> index
      %2 = arith.index_castui %intptr : index to i64
      %3 = vector.insert %2, %cst_6 [0] : i64 into vector<8xi64>
      %4 = vector.bitcast %3 : vector<8xi64> to vector<16xi32>
      %5 = vector.insert %c4095_i32, %4 [2] : i32 into vector<16xi32>
      %6 = vector.insert %c1023_i32, %5 [3] : i32 into vector<16xi32>
      %7 = vector.insert %c4095_i32, %6 [4] : i32 into vector<16xi32>
      %8 = arith.index_castui %1 : index to i32
      %9 = arith.index_castui %0 : index to i32
      %10 = vector.insert %8, %7 [5] : i32 into vector<16xi32>
      %11 = vector.insert %9, %10 [6] : i32 into vector<16xi32>
      %12 = vector.insert %c1807_i32, %11 [7] : i32 into vector<16xi32>
      %13 = arith.addi %1, %c16 : index
      %14 = arith.index_castui %13 : index to i32
      %15 = vector.insert %14, %7 [5] : i32 into vector<16xi32>
      %16 = vector.insert %9, %15 [6] : i32 into vector<16xi32>
      %17 = vector.insert %c1807_i32, %16 [7] : i32 into vector<16xi32>
      %18 = arith.addi %0, %c8 : index
      %19 = arith.index_castui %18 : index to i32
      %20 = vector.insert %19, %10 [6] : i32 into vector<16xi32>
      %21 = vector.insert %c1807_i32, %20 [7] : i32 into vector<16xi32>
      %22 = vector.insert %19, %15 [6] : i32 into vector<16xi32>
      %23 = vector.insert %c1807_i32, %22 [7] : i32 into vector<16xi32>
      %24 = arith.addi %0, %c16 : index
      %25 = arith.index_castui %24 : index to i32
      %26 = vector.insert %25, %10 [6] : i32 into vector<16xi32>
      %27 = vector.insert %c1807_i32, %26 [7] : i32 into vector<16xi32>
      %28 = vector.insert %25, %15 [6] : i32 into vector<16xi32>
      %29 = vector.insert %c1807_i32, %28 [7] : i32 into vector<16xi32>
      %30 = arith.addi %0, %c24 : index
      %31 = arith.index_castui %30 : index to i32
      %32 = vector.insert %31, %10 [6] : i32 into vector<16xi32>
      %33 = vector.insert %c1807_i32, %32 [7] : i32 into vector<16xi32>
      %34 = vector.insert %31, %15 [6] : i32 into vector<16xi32>
      %35 = vector.insert %c1807_i32, %34 [7] : i32 into vector<16xi32>
      %36 = vector.insert %c7951_i32, %11 [7] : i32 into vector<16xi32>
      %37 = vector.insert %c7951_i32, %16 [7] : i32 into vector<16xi32>
      %38 = func.call @llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8(%true, %cst_0, %c1_i8, %c16_i16, %c32_i16, %36, %c0_i32, %c0_i32, %cst_5) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf32>) -> vector<512xf32>
      %39 = func.call @llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8(%true, %cst_0, %c1_i8, %c16_i16, %c32_i16, %37, %c0_i32, %c0_i32, %cst_5) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf32>) -> vector<512xf32>
      %40 = vector.shuffle %38, %38 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<512xf32>, vector<512xf32>
      %41 = vector.shuffle %38, %38 [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<512xf32>, vector<512xf32>
      %42 = vector.shuffle %38, %38 [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383] : vector<512xf32>, vector<512xf32>
      %43 = vector.shuffle %38, %38 [384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xf32>, vector<512xf32>
      %44 = vector.shuffle %39, %39 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<512xf32>, vector<512xf32>
      %45 = vector.shuffle %39, %39 [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<512xf32>, vector<512xf32>
      %46 = vector.shuffle %39, %39 [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383] : vector<512xf32>, vector<512xf32>
      %47 = vector.shuffle %39, %39 [384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xf32>, vector<512xf32>
      %intptr_9 = memref.extract_aligned_pointer_as_index %arg0 : memref<1024x1024xf16> -> index
      %48 = arith.index_castui %intptr_9 : index to i64
      %49 = vector.insert %48, %cst_6 [0] : i64 into vector<8xi64>
      %50 = vector.bitcast %49 : vector<8xi64> to vector<16xi32>
      %51 = vector.insert %c2047_i32, %50 [2] : i32 into vector<16xi32>
      %52 = vector.insert %c1023_i32, %51 [3] : i32 into vector<16xi32>
      %53 = vector.insert %c2047_i32, %52 [4] : i32 into vector<16xi32>
      %54 = vector.insert %9, %53 [5] : i32 into vector<16xi32>
      %55 = vector.insert %c0_i32, %54 [6] : i32 into vector<16xi32>
      %56 = vector.insert %c3847_i32, %55 [7] : i32 into vector<16xi32>
      %57 = vector.insert %19, %53 [5] : i32 into vector<16xi32>
      %58 = vector.insert %c0_i32, %57 [6] : i32 into vector<16xi32>
      %59 = vector.insert %c3847_i32, %58 [7] : i32 into vector<16xi32>
      %60 = vector.insert %25, %53 [5] : i32 into vector<16xi32>
      %61 = vector.insert %c0_i32, %60 [6] : i32 into vector<16xi32>
      %62 = vector.insert %c3847_i32, %61 [7] : i32 into vector<16xi32>
      %63 = vector.insert %31, %53 [5] : i32 into vector<16xi32>
      %64 = vector.insert %c0_i32, %63 [6] : i32 into vector<16xi32>
      %65 = vector.insert %c3847_i32, %64 [7] : i32 into vector<16xi32>
      %66 = vector.insert %c16_i32, %54 [6] : i32 into vector<16xi32>
      %67 = vector.insert %c3847_i32, %66 [7] : i32 into vector<16xi32>
      %68 = vector.insert %c16_i32, %57 [6] : i32 into vector<16xi32>
      %69 = vector.insert %c3847_i32, %68 [7] : i32 into vector<16xi32>
      %70 = vector.insert %c16_i32, %60 [6] : i32 into vector<16xi32>
      %71 = vector.insert %c3847_i32, %70 [7] : i32 into vector<16xi32>
      %72 = vector.insert %c16_i32, %63 [6] : i32 into vector<16xi32>
      %73 = vector.insert %c3847_i32, %72 [7] : i32 into vector<16xi32>
      %intptr_10 = memref.extract_aligned_pointer_as_index %arg1 : memref<1024x1024xf16> -> index
      %74 = arith.index_castui %intptr_10 : index to i64
      %75 = vector.insert %74, %cst_6 [0] : i64 into vector<8xi64>
      %76 = vector.bitcast %75 : vector<8xi64> to vector<16xi32>
      %77 = vector.insert %c2047_i32, %76 [2] : i32 into vector<16xi32>
      %78 = vector.insert %c1023_i32, %77 [3] : i32 into vector<16xi32>
      %79 = vector.insert %c2047_i32, %78 [4] : i32 into vector<16xi32>
      %80 = vector.insert %8, %79 [5] : i32 into vector<16xi32>
      %81 = vector.insert %c0_i32, %80 [6] : i32 into vector<16xi32>
      %82 = vector.insert %c7951_i32, %81 [7] : i32 into vector<16xi32>
      %83 = vector.insert %14, %79 [5] : i32 into vector<16xi32>
      %84 = vector.insert %c0_i32, %83 [6] : i32 into vector<16xi32>
      %85 = vector.insert %c7951_i32, %84 [7] : i32 into vector<16xi32>
      %86:18 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %56, %arg5 = %59, %arg6 = %62, %arg7 = %65, %arg8 = %67, %arg9 = %69, %arg10 = %71, %arg11 = %73, %arg12 = %82, %arg13 = %85, %arg14 = %40, %arg15 = %44, %arg16 = %41, %arg17 = %45, %arg18 = %42, %arg19 = %46, %arg20 = %43, %arg21 = %47) -> (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>) {
        %87 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg4, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %88 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg5, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %89 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg6, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %90 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg7, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %91 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg8, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %92 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg9, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %93 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg10, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %94 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg11, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %95 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8(%true, %cst_0, %c1_i8, %c16_i16, %c32_i16, %arg12, %c0_i32, %c0_i32, %cst_3) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf16>) -> vector<512xf16>
        %96 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8(%true, %cst_0, %c1_i8, %c16_i16, %c32_i16, %arg13, %c0_i32, %c0_i32, %cst_3) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf16>) -> vector<512xf16>
        %97 = vector.shuffle %95, %95 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<512xf16>, vector<512xf16>
        %98 = vector.shuffle %95, %95 [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xf16>, vector<512xf16>
        %99 = vector.shuffle %96, %96 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<512xf16>, vector<512xf16>
        %100 = vector.shuffle %96, %96 [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xf16>, vector<512xf16>
        %101 = vector.bitcast %87 : vector<128xf16> to vector<64xf32>
        %alloc = memref.alloc() : memref<4096xf32, 3>
        %102 = gpu.subgroup_id : index
        %103 = arith.muli %102, %c64 : index
        %104 = vector.splat %103 : vector<8xindex>
        %105 = arith.addi %104, %cst_8 : vector<8xindex>
        %intptr_11 = memref.extract_aligned_pointer_as_index %alloc : memref<4096xf32, 3> -> index
        %106 = arith.index_castui %intptr_11 : index to i32
        %107 = arith.index_castui %105 : vector<8xindex> to vector<8xi32>
        %108 = arith.muli %107, %cst_2 : vector<8xi32>
        %109 = vector.splat %106 : vector<8xi32>
        %110 = arith.addi %109, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %110, %101, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %111 = arith.index_castui %103 : index to i32
        %112 = arith.muli %111, %c4_i32 : i32
        %113 = arith.addi %106, %112 : i32
        %114 = vector.splat %113 : vector<1xi32>
        %115 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %114, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %116 = vector.bitcast %91 : vector<128xf16> to vector<64xf32>
        %alloc_12 = memref.alloc() : memref<4096xf32, 3>
        %intptr_13 = memref.extract_aligned_pointer_as_index %alloc_12 : memref<4096xf32, 3> -> index
        %117 = arith.index_castui %intptr_13 : index to i32
        %118 = vector.splat %117 : vector<8xi32>
        %119 = arith.addi %118, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %119, %116, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %120 = arith.addi %117, %112 : i32
        %121 = vector.splat %120 : vector<1xi32>
        %122 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %121, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %123 = vector.bitcast %88 : vector<128xf16> to vector<64xf32>
        %alloc_14 = memref.alloc() : memref<4096xf32, 3>
        %intptr_15 = memref.extract_aligned_pointer_as_index %alloc_14 : memref<4096xf32, 3> -> index
        %124 = arith.index_castui %intptr_15 : index to i32
        %125 = vector.splat %124 : vector<8xi32>
        %126 = arith.addi %125, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %126, %123, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %127 = arith.addi %124, %112 : i32
        %128 = vector.splat %127 : vector<1xi32>
        %129 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %128, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %130 = vector.bitcast %92 : vector<128xf16> to vector<64xf32>
        %alloc_16 = memref.alloc() : memref<4096xf32, 3>
        %intptr_17 = memref.extract_aligned_pointer_as_index %alloc_16 : memref<4096xf32, 3> -> index
        %131 = arith.index_castui %intptr_17 : index to i32
        %132 = vector.splat %131 : vector<8xi32>
        %133 = arith.addi %132, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %133, %130, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %134 = arith.addi %131, %112 : i32
        %135 = vector.splat %134 : vector<1xi32>
        %136 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %135, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %137 = vector.bitcast %89 : vector<128xf16> to vector<64xf32>
        %alloc_18 = memref.alloc() : memref<4096xf32, 3>
        %intptr_19 = memref.extract_aligned_pointer_as_index %alloc_18 : memref<4096xf32, 3> -> index
        %138 = arith.index_castui %intptr_19 : index to i32
        %139 = vector.splat %138 : vector<8xi32>
        %140 = arith.addi %139, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %140, %137, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %141 = arith.addi %138, %112 : i32
        %142 = vector.splat %141 : vector<1xi32>
        %143 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %142, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %144 = vector.bitcast %93 : vector<128xf16> to vector<64xf32>
        %alloc_20 = memref.alloc() : memref<4096xf32, 3>
        %intptr_21 = memref.extract_aligned_pointer_as_index %alloc_20 : memref<4096xf32, 3> -> index
        %145 = arith.index_castui %intptr_21 : index to i32
        %146 = vector.splat %145 : vector<8xi32>
        %147 = arith.addi %146, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %147, %144, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %148 = arith.addi %145, %112 : i32
        %149 = vector.splat %148 : vector<1xi32>
        %150 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %149, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %151 = vector.bitcast %90 : vector<128xf16> to vector<64xf32>
        %alloc_22 = memref.alloc() : memref<4096xf32, 3>
        %intptr_23 = memref.extract_aligned_pointer_as_index %alloc_22 : memref<4096xf32, 3> -> index
        %152 = arith.index_castui %intptr_23 : index to i32
        %153 = vector.splat %152 : vector<8xi32>
        %154 = arith.addi %153, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %154, %151, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %155 = arith.addi %152, %112 : i32
        %156 = vector.splat %155 : vector<1xi32>
        %157 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %156, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %158 = vector.bitcast %94 : vector<128xf16> to vector<64xf32>
        %alloc_24 = memref.alloc() : memref<4096xf32, 3>
        %intptr_25 = memref.extract_aligned_pointer_as_index %alloc_24 : memref<4096xf32, 3> -> index
        %159 = arith.index_castui %intptr_25 : index to i32
        %160 = vector.splat %159 : vector<8xi32>
        %161 = arith.addi %160, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %161, %158, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %162 = arith.addi %159, %112 : i32
        %163 = vector.splat %162 : vector<1xi32>
        %164 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %163, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %165 = vector.bitcast %115 : vector<64xf32> to vector<64xi32>
        %166 = vector.bitcast %97 : vector<256xf16> to vector<128xi32>
        %167 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg14, %166, %165, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %168 = vector.bitcast %122 : vector<64xf32> to vector<64xi32>
        %169 = vector.bitcast %98 : vector<256xf16> to vector<128xi32>
        %170 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%167, %169, %168, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %171 = vector.bitcast %99 : vector<256xf16> to vector<128xi32>
        %172 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg15, %171, %165, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %173 = vector.bitcast %100 : vector<256xf16> to vector<128xi32>
        %174 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%172, %173, %168, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %175 = vector.bitcast %129 : vector<64xf32> to vector<64xi32>
        %176 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg16, %166, %175, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %177 = vector.bitcast %136 : vector<64xf32> to vector<64xi32>
        %178 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%176, %169, %177, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %179 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg17, %171, %175, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %180 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%179, %173, %177, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %181 = vector.bitcast %143 : vector<64xf32> to vector<64xi32>
        %182 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg18, %166, %181, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %183 = vector.bitcast %150 : vector<64xf32> to vector<64xi32>
        %184 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%182, %169, %183, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %185 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg19, %171, %181, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %186 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%185, %173, %183, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %187 = vector.bitcast %157 : vector<64xf32> to vector<64xi32>
        %188 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg20, %166, %187, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %189 = vector.bitcast %164 : vector<64xf32> to vector<64xi32>
        %190 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%188, %169, %189, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %191 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg21, %171, %187, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %192 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%191, %173, %189, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %193 = vector.extractelement %arg4[%c6_i32 : i32] : vector<16xi32>
        %194 = arith.addi %193, %c32_i32 : i32
        %195 = vector.insert %194, %arg4 [6] : i32 into vector<16xi32>
        %196 = vector.extractelement %arg5[%c6_i32 : i32] : vector<16xi32>
        %197 = arith.addi %196, %c32_i32 : i32
        %198 = vector.insert %197, %arg5 [6] : i32 into vector<16xi32>
        %199 = vector.extractelement %arg6[%c6_i32 : i32] : vector<16xi32>
        %200 = arith.addi %199, %c32_i32 : i32
        %201 = vector.insert %200, %arg6 [6] : i32 into vector<16xi32>
        %202 = vector.extractelement %arg7[%c6_i32 : i32] : vector<16xi32>
        %203 = arith.addi %202, %c32_i32 : i32
        %204 = vector.insert %203, %arg7 [6] : i32 into vector<16xi32>
        %205 = vector.extractelement %arg8[%c6_i32 : i32] : vector<16xi32>
        %206 = arith.addi %205, %c32_i32 : i32
        %207 = vector.insert %206, %arg8 [6] : i32 into vector<16xi32>
        %208 = vector.extractelement %arg9[%c6_i32 : i32] : vector<16xi32>
        %209 = arith.addi %208, %c32_i32 : i32
        %210 = vector.insert %209, %arg9 [6] : i32 into vector<16xi32>
        %211 = vector.extractelement %arg10[%c6_i32 : i32] : vector<16xi32>
        %212 = arith.addi %211, %c32_i32 : i32
        %213 = vector.insert %212, %arg10 [6] : i32 into vector<16xi32>
        %214 = vector.extractelement %arg11[%c6_i32 : i32] : vector<16xi32>
        %215 = arith.addi %214, %c32_i32 : i32
        %216 = vector.insert %215, %arg11 [6] : i32 into vector<16xi32>
        %217 = vector.extractelement %arg12[%c6_i32 : i32] : vector<16xi32>
        %218 = arith.addi %217, %c32_i32 : i32
        %219 = vector.insert %218, %arg12 [6] : i32 into vector<16xi32>
        %220 = vector.extractelement %arg13[%c6_i32 : i32] : vector<16xi32>
        %221 = arith.addi %220, %c32_i32 : i32
        %222 = vector.insert %221, %arg13 [6] : i32 into vector<16xi32>
        scf.yield %195, %198, %201, %204, %207, %210, %213, %216, %219, %222, %170, %174, %178, %180, %184, %186, %190, %192 : vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>
      }
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %12, %c0_i32, %c0_i32, %86#10) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %17, %c0_i32, %c0_i32, %86#11) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %21, %c0_i32, %c0_i32, %86#12) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %23, %c0_i32, %c0_i32, %86#13) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %27, %c0_i32, %c0_i32, %86#14) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %29, %c0_i32, %c0_i32, %86#15) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %33, %c0_i32, %c0_i32, %86#16) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %35, %c0_i32, %c0_i32, %86#17) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %cst_0 = arith.constant 0.000000e+00 : f16
    %cst_1 = arith.constant 1.000000e+00 : f16
    %alloc = memref.alloc() : memref<1024x1024xf16>
    %alloc_2 = memref.alloc() : memref<1024x1024xf16>
    %alloc_3 = memref.alloc() : memref<1024x1024xf32>
    %alloc_4 = memref.alloc() : memref<1024x1024xf32>
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = index.castu %arg1 : index to i16
        %2 = arith.uitofp %1 : i16 to f16
        memref.store %2, %alloc[%arg0, %arg1] : memref<1024x1024xf16>
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = index.castu %arg0 : index to i32
        %2 = index.castu %arg1 : index to i32
        %3 = arith.cmpi eq, %1, %2 : i32
        scf.if %3 {
          memref.store %cst_1, %alloc_2[%arg0, %arg1] : memref<1024x1024xf16>
        } else {
          memref.store %cst_0, %alloc_2[%arg0, %arg1] : memref<1024x1024xf16>
        }
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        memref.store %cst, %alloc_3[%arg0, %arg1] : memref<1024x1024xf32>
        memref.store %cst, %alloc_4[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = memref.load %alloc_4[%arg0, %arg1] : memref<1024x1024xf32>
        %2 = scf.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %1) -> (f32) {
          %3 = memref.load %alloc[%arg2, %arg0] : memref<1024x1024xf16>
          %4 = memref.load %alloc_2[%arg2, %arg1] : memref<1024x1024xf16>
          %5 = arith.mulf %3, %4 : f16
          %6 = arith.extf %5 : f16 to f32
          %7 = arith.addf %6, %arg3 : f32
          scf.yield %7 : f32
        }
        memref.store %2, %alloc_4[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    %0 = call @test(%alloc, %alloc_2, %alloc_3) : (memref<1024x1024xf16>, memref<1024x1024xf16>, memref<1024x1024xf32>) -> memref<1024x1024xf32>
    %cast = memref.cast %0 : memref<1024x1024xf32> to memref<*xf32>
    %cast_5 = memref.cast %alloc_4 : memref<1024x1024xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_5) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<1024x1024xf16>
    memref.dealloc %alloc_2 : memref<1024x1024xf16>
    memref.dealloc %alloc_3 : memref<1024x1024xf32>
    memref.dealloc %alloc_4 : memref<1024x1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}


// -----// IR Dump After CSE (cse) //----- //
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %memref = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %arg0, %memref : memref<1024x1024xf16> to memref<1024x1024xf16>
    %memref_0 = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %arg1, %memref_0 : memref<1024x1024xf16> to memref<1024x1024xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<1024x1024xf32>
    memref.copy %arg2, %memref_1 : memref<1024x1024xf32> to memref<1024x1024xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c32, %c32, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<1024x1024xf16>, %memref_0 : memref<1024x1024xf16>, %memref_1 : memref<1024x1024xf32>)
    gpu.dealloc  %memref : memref<1024x1024xf16>
    gpu.dealloc  %memref_0 : memref<1024x1024xf16>
    return %memref_1 : memref<1024x1024xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    func.func private @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32", linkage_type = <Import>>}
    func.func private @llvm.genx.dpas2.v128f32.v128i32.v64i32(vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas2.v128f32.v128i32.v64i32", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf16>) -> vector<512xf16> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf32>) -> vector<512xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8", linkage_type = <Import>>}
    gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %cst = arith.constant dense<3> : vector<2xi8>
      %cst_0 = arith.constant dense<2> : vector<2xi8>
      %c6_i32 = arith.constant 6 : i32
      %c16_i32 = arith.constant 16 : i32
      %c32_i32 = arith.constant 32 : i32
      %c8_i32 = arith.constant 8 : i32
      %c10_i32 = arith.constant 10 : i32
      %c8_i8 = arith.constant 8 : i8
      %cst_1 = arith.constant dense<true> : vector<1xi1>
      %c4_i32 = arith.constant 4 : i32
      %c5_i8 = arith.constant 5 : i8
      %c3_i8 = arith.constant 3 : i8
      %c1_i16 = arith.constant 1 : i16
      %c0_i8 = arith.constant 0 : i8
      %c4_i8 = arith.constant 4 : i8
      %cst_2 = arith.constant dense<4> : vector<8xi32>
      %cst_3 = arith.constant dense<0.000000e+00> : vector<512xf16>
      %c8_i16 = arith.constant 8 : i16
      %cst_4 = arith.constant dense<0.000000e+00> : vector<128xf16>
      %c3847_i32 = arith.constant 3847 : i32
      %c2047_i32 = arith.constant 2047 : i32
      %c0_i32 = arith.constant 0 : i32
      %c32_i16 = arith.constant 32 : i16
      %c16_i16 = arith.constant 16 : i16
      %c1_i8 = arith.constant 1 : i8
      %true = arith.constant true
      %cst_5 = arith.constant dense<0.000000e+00> : vector<512xf32>
      %c7951_i32 = arith.constant 7951 : i32
      %c1807_i32 = arith.constant 1807 : i32
      %c1023_i32 = arith.constant 1023 : i32
      %c4095_i32 = arith.constant 4095 : i32
      %cst_6 = arith.constant dense<0> : vector<8xi64>
      %cst_7 = arith.constant dense<true> : vector<8xi1>
      %cst_8 = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56]> : vector<8xindex>
      %c64 = arith.constant 64 : index
      %c24 = arith.constant 24 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c32 : index
      %1 = arith.muli %block_id_y, %c32 : index
      %intptr = memref.extract_aligned_pointer_as_index %arg2 : memref<1024x1024xf32> -> index
      %2 = arith.index_castui %intptr : index to i64
      %3 = vector.insert %2, %cst_6 [0] : i64 into vector<8xi64>
      %4 = vector.bitcast %3 : vector<8xi64> to vector<16xi32>
      %5 = vector.insert %c4095_i32, %4 [2] : i32 into vector<16xi32>
      %6 = vector.insert %c1023_i32, %5 [3] : i32 into vector<16xi32>
      %7 = vector.insert %c4095_i32, %6 [4] : i32 into vector<16xi32>
      %8 = arith.index_castui %1 : index to i32
      %9 = arith.index_castui %0 : index to i32
      %10 = vector.insert %8, %7 [5] : i32 into vector<16xi32>
      %11 = vector.insert %9, %10 [6] : i32 into vector<16xi32>
      %12 = vector.insert %c1807_i32, %11 [7] : i32 into vector<16xi32>
      %13 = arith.addi %1, %c16 : index
      %14 = arith.index_castui %13 : index to i32
      %15 = vector.insert %14, %7 [5] : i32 into vector<16xi32>
      %16 = vector.insert %9, %15 [6] : i32 into vector<16xi32>
      %17 = vector.insert %c1807_i32, %16 [7] : i32 into vector<16xi32>
      %18 = arith.addi %0, %c8 : index
      %19 = arith.index_castui %18 : index to i32
      %20 = vector.insert %19, %10 [6] : i32 into vector<16xi32>
      %21 = vector.insert %c1807_i32, %20 [7] : i32 into vector<16xi32>
      %22 = vector.insert %19, %15 [6] : i32 into vector<16xi32>
      %23 = vector.insert %c1807_i32, %22 [7] : i32 into vector<16xi32>
      %24 = arith.addi %0, %c16 : index
      %25 = arith.index_castui %24 : index to i32
      %26 = vector.insert %25, %10 [6] : i32 into vector<16xi32>
      %27 = vector.insert %c1807_i32, %26 [7] : i32 into vector<16xi32>
      %28 = vector.insert %25, %15 [6] : i32 into vector<16xi32>
      %29 = vector.insert %c1807_i32, %28 [7] : i32 into vector<16xi32>
      %30 = arith.addi %0, %c24 : index
      %31 = arith.index_castui %30 : index to i32
      %32 = vector.insert %31, %10 [6] : i32 into vector<16xi32>
      %33 = vector.insert %c1807_i32, %32 [7] : i32 into vector<16xi32>
      %34 = vector.insert %31, %15 [6] : i32 into vector<16xi32>
      %35 = vector.insert %c1807_i32, %34 [7] : i32 into vector<16xi32>
      %36 = vector.insert %c7951_i32, %11 [7] : i32 into vector<16xi32>
      %37 = vector.insert %c7951_i32, %16 [7] : i32 into vector<16xi32>
      %38 = func.call @llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8(%true, %cst_0, %c1_i8, %c16_i16, %c32_i16, %36, %c0_i32, %c0_i32, %cst_5) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf32>) -> vector<512xf32>
      %39 = func.call @llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8(%true, %cst_0, %c1_i8, %c16_i16, %c32_i16, %37, %c0_i32, %c0_i32, %cst_5) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf32>) -> vector<512xf32>
      %40 = vector.shuffle %38, %38 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<512xf32>, vector<512xf32>
      %41 = vector.shuffle %38, %38 [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<512xf32>, vector<512xf32>
      %42 = vector.shuffle %38, %38 [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383] : vector<512xf32>, vector<512xf32>
      %43 = vector.shuffle %38, %38 [384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xf32>, vector<512xf32>
      %44 = vector.shuffle %39, %39 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<512xf32>, vector<512xf32>
      %45 = vector.shuffle %39, %39 [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<512xf32>, vector<512xf32>
      %46 = vector.shuffle %39, %39 [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383] : vector<512xf32>, vector<512xf32>
      %47 = vector.shuffle %39, %39 [384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xf32>, vector<512xf32>
      %intptr_9 = memref.extract_aligned_pointer_as_index %arg0 : memref<1024x1024xf16> -> index
      %48 = arith.index_castui %intptr_9 : index to i64
      %49 = vector.insert %48, %cst_6 [0] : i64 into vector<8xi64>
      %50 = vector.bitcast %49 : vector<8xi64> to vector<16xi32>
      %51 = vector.insert %c2047_i32, %50 [2] : i32 into vector<16xi32>
      %52 = vector.insert %c1023_i32, %51 [3] : i32 into vector<16xi32>
      %53 = vector.insert %c2047_i32, %52 [4] : i32 into vector<16xi32>
      %54 = vector.insert %9, %53 [5] : i32 into vector<16xi32>
      %55 = vector.insert %c0_i32, %54 [6] : i32 into vector<16xi32>
      %56 = vector.insert %c3847_i32, %55 [7] : i32 into vector<16xi32>
      %57 = vector.insert %19, %53 [5] : i32 into vector<16xi32>
      %58 = vector.insert %c0_i32, %57 [6] : i32 into vector<16xi32>
      %59 = vector.insert %c3847_i32, %58 [7] : i32 into vector<16xi32>
      %60 = vector.insert %25, %53 [5] : i32 into vector<16xi32>
      %61 = vector.insert %c0_i32, %60 [6] : i32 into vector<16xi32>
      %62 = vector.insert %c3847_i32, %61 [7] : i32 into vector<16xi32>
      %63 = vector.insert %31, %53 [5] : i32 into vector<16xi32>
      %64 = vector.insert %c0_i32, %63 [6] : i32 into vector<16xi32>
      %65 = vector.insert %c3847_i32, %64 [7] : i32 into vector<16xi32>
      %66 = vector.insert %c16_i32, %54 [6] : i32 into vector<16xi32>
      %67 = vector.insert %c3847_i32, %66 [7] : i32 into vector<16xi32>
      %68 = vector.insert %c16_i32, %57 [6] : i32 into vector<16xi32>
      %69 = vector.insert %c3847_i32, %68 [7] : i32 into vector<16xi32>
      %70 = vector.insert %c16_i32, %60 [6] : i32 into vector<16xi32>
      %71 = vector.insert %c3847_i32, %70 [7] : i32 into vector<16xi32>
      %72 = vector.insert %c16_i32, %63 [6] : i32 into vector<16xi32>
      %73 = vector.insert %c3847_i32, %72 [7] : i32 into vector<16xi32>
      %intptr_10 = memref.extract_aligned_pointer_as_index %arg1 : memref<1024x1024xf16> -> index
      %74 = arith.index_castui %intptr_10 : index to i64
      %75 = vector.insert %74, %cst_6 [0] : i64 into vector<8xi64>
      %76 = vector.bitcast %75 : vector<8xi64> to vector<16xi32>
      %77 = vector.insert %c2047_i32, %76 [2] : i32 into vector<16xi32>
      %78 = vector.insert %c1023_i32, %77 [3] : i32 into vector<16xi32>
      %79 = vector.insert %c2047_i32, %78 [4] : i32 into vector<16xi32>
      %80 = vector.insert %8, %79 [5] : i32 into vector<16xi32>
      %81 = vector.insert %c0_i32, %80 [6] : i32 into vector<16xi32>
      %82 = vector.insert %c7951_i32, %81 [7] : i32 into vector<16xi32>
      %83 = vector.insert %14, %79 [5] : i32 into vector<16xi32>
      %84 = vector.insert %c0_i32, %83 [6] : i32 into vector<16xi32>
      %85 = vector.insert %c7951_i32, %84 [7] : i32 into vector<16xi32>
      %86:18 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %56, %arg5 = %59, %arg6 = %62, %arg7 = %65, %arg8 = %67, %arg9 = %69, %arg10 = %71, %arg11 = %73, %arg12 = %82, %arg13 = %85, %arg14 = %40, %arg15 = %44, %arg16 = %41, %arg17 = %45, %arg18 = %42, %arg19 = %46, %arg20 = %43, %arg21 = %47) -> (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>) {
        %87 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg4, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %88 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg5, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %89 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg6, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %90 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg7, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %91 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg8, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %92 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg9, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %93 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg10, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %94 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg11, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %95 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8(%true, %cst_0, %c1_i8, %c16_i16, %c32_i16, %arg12, %c0_i32, %c0_i32, %cst_3) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf16>) -> vector<512xf16>
        %96 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8(%true, %cst_0, %c1_i8, %c16_i16, %c32_i16, %arg13, %c0_i32, %c0_i32, %cst_3) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf16>) -> vector<512xf16>
        %97 = vector.shuffle %95, %95 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<512xf16>, vector<512xf16>
        %98 = vector.shuffle %95, %95 [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xf16>, vector<512xf16>
        %99 = vector.shuffle %96, %96 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<512xf16>, vector<512xf16>
        %100 = vector.shuffle %96, %96 [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xf16>, vector<512xf16>
        %101 = vector.bitcast %87 : vector<128xf16> to vector<64xf32>
        %alloc = memref.alloc() : memref<4096xf32, 3>
        %102 = gpu.subgroup_id : index
        %103 = arith.muli %102, %c64 : index
        %104 = vector.splat %103 : vector<8xindex>
        %105 = arith.addi %104, %cst_8 : vector<8xindex>
        %intptr_11 = memref.extract_aligned_pointer_as_index %alloc : memref<4096xf32, 3> -> index
        %106 = arith.index_castui %intptr_11 : index to i32
        %107 = arith.index_castui %105 : vector<8xindex> to vector<8xi32>
        %108 = arith.muli %107, %cst_2 : vector<8xi32>
        %109 = vector.splat %106 : vector<8xi32>
        %110 = arith.addi %109, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %110, %101, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %111 = arith.index_castui %103 : index to i32
        %112 = arith.muli %111, %c4_i32 : i32
        %113 = arith.addi %106, %112 : i32
        %114 = vector.splat %113 : vector<1xi32>
        %115 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %114, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %116 = vector.bitcast %91 : vector<128xf16> to vector<64xf32>
        %alloc_12 = memref.alloc() : memref<4096xf32, 3>
        %intptr_13 = memref.extract_aligned_pointer_as_index %alloc_12 : memref<4096xf32, 3> -> index
        %117 = arith.index_castui %intptr_13 : index to i32
        %118 = vector.splat %117 : vector<8xi32>
        %119 = arith.addi %118, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %119, %116, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %120 = arith.addi %117, %112 : i32
        %121 = vector.splat %120 : vector<1xi32>
        %122 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %121, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %123 = vector.bitcast %88 : vector<128xf16> to vector<64xf32>
        %alloc_14 = memref.alloc() : memref<4096xf32, 3>
        %intptr_15 = memref.extract_aligned_pointer_as_index %alloc_14 : memref<4096xf32, 3> -> index
        %124 = arith.index_castui %intptr_15 : index to i32
        %125 = vector.splat %124 : vector<8xi32>
        %126 = arith.addi %125, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %126, %123, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %127 = arith.addi %124, %112 : i32
        %128 = vector.splat %127 : vector<1xi32>
        %129 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %128, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %130 = vector.bitcast %92 : vector<128xf16> to vector<64xf32>
        %alloc_16 = memref.alloc() : memref<4096xf32, 3>
        %intptr_17 = memref.extract_aligned_pointer_as_index %alloc_16 : memref<4096xf32, 3> -> index
        %131 = arith.index_castui %intptr_17 : index to i32
        %132 = vector.splat %131 : vector<8xi32>
        %133 = arith.addi %132, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %133, %130, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %134 = arith.addi %131, %112 : i32
        %135 = vector.splat %134 : vector<1xi32>
        %136 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %135, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %137 = vector.bitcast %89 : vector<128xf16> to vector<64xf32>
        %alloc_18 = memref.alloc() : memref<4096xf32, 3>
        %intptr_19 = memref.extract_aligned_pointer_as_index %alloc_18 : memref<4096xf32, 3> -> index
        %138 = arith.index_castui %intptr_19 : index to i32
        %139 = vector.splat %138 : vector<8xi32>
        %140 = arith.addi %139, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %140, %137, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %141 = arith.addi %138, %112 : i32
        %142 = vector.splat %141 : vector<1xi32>
        %143 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %142, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %144 = vector.bitcast %93 : vector<128xf16> to vector<64xf32>
        %alloc_20 = memref.alloc() : memref<4096xf32, 3>
        %intptr_21 = memref.extract_aligned_pointer_as_index %alloc_20 : memref<4096xf32, 3> -> index
        %145 = arith.index_castui %intptr_21 : index to i32
        %146 = vector.splat %145 : vector<8xi32>
        %147 = arith.addi %146, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %147, %144, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %148 = arith.addi %145, %112 : i32
        %149 = vector.splat %148 : vector<1xi32>
        %150 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %149, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %151 = vector.bitcast %90 : vector<128xf16> to vector<64xf32>
        %alloc_22 = memref.alloc() : memref<4096xf32, 3>
        %intptr_23 = memref.extract_aligned_pointer_as_index %alloc_22 : memref<4096xf32, 3> -> index
        %152 = arith.index_castui %intptr_23 : index to i32
        %153 = vector.splat %152 : vector<8xi32>
        %154 = arith.addi %153, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %154, %151, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %155 = arith.addi %152, %112 : i32
        %156 = vector.splat %155 : vector<1xi32>
        %157 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %156, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %158 = vector.bitcast %94 : vector<128xf16> to vector<64xf32>
        %alloc_24 = memref.alloc() : memref<4096xf32, 3>
        %intptr_25 = memref.extract_aligned_pointer_as_index %alloc_24 : memref<4096xf32, 3> -> index
        %159 = arith.index_castui %intptr_25 : index to i32
        %160 = vector.splat %159 : vector<8xi32>
        %161 = arith.addi %160, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %161, %158, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %162 = arith.addi %159, %112 : i32
        %163 = vector.splat %162 : vector<1xi32>
        %164 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %163, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %165 = vector.bitcast %115 : vector<64xf32> to vector<64xi32>
        %166 = vector.bitcast %97 : vector<256xf16> to vector<128xi32>
        %167 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg14, %166, %165, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %168 = vector.bitcast %122 : vector<64xf32> to vector<64xi32>
        %169 = vector.bitcast %98 : vector<256xf16> to vector<128xi32>
        %170 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%167, %169, %168, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %171 = vector.bitcast %99 : vector<256xf16> to vector<128xi32>
        %172 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg15, %171, %165, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %173 = vector.bitcast %100 : vector<256xf16> to vector<128xi32>
        %174 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%172, %173, %168, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %175 = vector.bitcast %129 : vector<64xf32> to vector<64xi32>
        %176 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg16, %166, %175, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %177 = vector.bitcast %136 : vector<64xf32> to vector<64xi32>
        %178 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%176, %169, %177, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %179 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg17, %171, %175, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %180 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%179, %173, %177, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %181 = vector.bitcast %143 : vector<64xf32> to vector<64xi32>
        %182 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg18, %166, %181, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %183 = vector.bitcast %150 : vector<64xf32> to vector<64xi32>
        %184 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%182, %169, %183, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %185 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg19, %171, %181, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %186 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%185, %173, %183, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %187 = vector.bitcast %157 : vector<64xf32> to vector<64xi32>
        %188 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg20, %166, %187, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %189 = vector.bitcast %164 : vector<64xf32> to vector<64xi32>
        %190 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%188, %169, %189, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %191 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg21, %171, %187, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %192 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%191, %173, %189, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %193 = vector.extractelement %arg4[%c6_i32 : i32] : vector<16xi32>
        %194 = arith.addi %193, %c32_i32 : i32
        %195 = vector.insert %194, %arg4 [6] : i32 into vector<16xi32>
        %196 = vector.extractelement %arg5[%c6_i32 : i32] : vector<16xi32>
        %197 = arith.addi %196, %c32_i32 : i32
        %198 = vector.insert %197, %arg5 [6] : i32 into vector<16xi32>
        %199 = vector.extractelement %arg6[%c6_i32 : i32] : vector<16xi32>
        %200 = arith.addi %199, %c32_i32 : i32
        %201 = vector.insert %200, %arg6 [6] : i32 into vector<16xi32>
        %202 = vector.extractelement %arg7[%c6_i32 : i32] : vector<16xi32>
        %203 = arith.addi %202, %c32_i32 : i32
        %204 = vector.insert %203, %arg7 [6] : i32 into vector<16xi32>
        %205 = vector.extractelement %arg8[%c6_i32 : i32] : vector<16xi32>
        %206 = arith.addi %205, %c32_i32 : i32
        %207 = vector.insert %206, %arg8 [6] : i32 into vector<16xi32>
        %208 = vector.extractelement %arg9[%c6_i32 : i32] : vector<16xi32>
        %209 = arith.addi %208, %c32_i32 : i32
        %210 = vector.insert %209, %arg9 [6] : i32 into vector<16xi32>
        %211 = vector.extractelement %arg10[%c6_i32 : i32] : vector<16xi32>
        %212 = arith.addi %211, %c32_i32 : i32
        %213 = vector.insert %212, %arg10 [6] : i32 into vector<16xi32>
        %214 = vector.extractelement %arg11[%c6_i32 : i32] : vector<16xi32>
        %215 = arith.addi %214, %c32_i32 : i32
        %216 = vector.insert %215, %arg11 [6] : i32 into vector<16xi32>
        %217 = vector.extractelement %arg12[%c6_i32 : i32] : vector<16xi32>
        %218 = arith.addi %217, %c32_i32 : i32
        %219 = vector.insert %218, %arg12 [6] : i32 into vector<16xi32>
        %220 = vector.extractelement %arg13[%c6_i32 : i32] : vector<16xi32>
        %221 = arith.addi %220, %c32_i32 : i32
        %222 = vector.insert %221, %arg13 [6] : i32 into vector<16xi32>
        scf.yield %195, %198, %201, %204, %207, %210, %213, %216, %219, %222, %170, %174, %178, %180, %184, %186, %190, %192 : vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>
      }
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %12, %c0_i32, %c0_i32, %86#10) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %17, %c0_i32, %c0_i32, %86#11) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %21, %c0_i32, %c0_i32, %86#12) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %23, %c0_i32, %c0_i32, %86#13) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %27, %c0_i32, %c0_i32, %86#14) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %29, %c0_i32, %c0_i32, %86#15) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %33, %c0_i32, %c0_i32, %86#16) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %35, %c0_i32, %c0_i32, %86#17) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %cst_0 = arith.constant 0.000000e+00 : f16
    %cst_1 = arith.constant 1.000000e+00 : f16
    %alloc = memref.alloc() : memref<1024x1024xf16>
    %alloc_2 = memref.alloc() : memref<1024x1024xf16>
    %alloc_3 = memref.alloc() : memref<1024x1024xf32>
    %alloc_4 = memref.alloc() : memref<1024x1024xf32>
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = index.castu %arg1 : index to i16
        %2 = arith.uitofp %1 : i16 to f16
        memref.store %2, %alloc[%arg0, %arg1] : memref<1024x1024xf16>
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = index.castu %arg0 : index to i32
        %2 = index.castu %arg1 : index to i32
        %3 = arith.cmpi eq, %1, %2 : i32
        scf.if %3 {
          memref.store %cst_1, %alloc_2[%arg0, %arg1] : memref<1024x1024xf16>
        } else {
          memref.store %cst_0, %alloc_2[%arg0, %arg1] : memref<1024x1024xf16>
        }
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        memref.store %cst, %alloc_3[%arg0, %arg1] : memref<1024x1024xf32>
        memref.store %cst, %alloc_4[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = memref.load %alloc_4[%arg0, %arg1] : memref<1024x1024xf32>
        %2 = scf.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %1) -> (f32) {
          %3 = memref.load %alloc[%arg2, %arg0] : memref<1024x1024xf16>
          %4 = memref.load %alloc_2[%arg2, %arg1] : memref<1024x1024xf16>
          %5 = arith.mulf %3, %4 : f16
          %6 = arith.extf %5 : f16 to f32
          %7 = arith.addf %6, %arg3 : f32
          scf.yield %7 : f32
        }
        memref.store %2, %alloc_4[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    %0 = call @test(%alloc, %alloc_2, %alloc_3) : (memref<1024x1024xf16>, memref<1024x1024xf16>, memref<1024x1024xf32>) -> memref<1024x1024xf32>
    %cast = memref.cast %0 : memref<1024x1024xf32> to memref<*xf32>
    %cast_5 = memref.cast %alloc_4 : memref<1024x1024xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_5) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<1024x1024xf16>
    memref.dealloc %alloc_2 : memref<1024x1024xf16>
    memref.dealloc %alloc_3 : memref<1024x1024xf32>
    memref.dealloc %alloc_4 : memref<1024x1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}


sg_gemm_1kx1kx1k_f16_f16_f32_transpose_a.mlir:52:20: error: failed to legalize unresolved materialization from ('i32') to ('vector<1xi32>') that remained live after conversion
        %a_value = xetile.load_tile %a_tile  : !xetile.tile<32x32xf16> -> vector<32x32xf16>
                   ^
sg_gemm_1kx1kx1k_f16_f16_f32_transpose_a.mlir:52:20: note: see current operation: %216 = "builtin.unrealized_conversion_cast"(%215) : (i32) -> vector<1xi32>
sg_gemm_1kx1kx1k_f16_f16_f32_transpose_a.mlir:52:20: note: see existing live user here: %157 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %156, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
// -----// IR Dump After RemoveSingleElemVector Failed (imex-remove-single-elem-vector) //----- //
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %memref = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %arg0, %memref : memref<1024x1024xf16> to memref<1024x1024xf16>
    %memref_0 = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %arg1, %memref_0 : memref<1024x1024xf16> to memref<1024x1024xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<1024x1024xf32>
    memref.copy %arg2, %memref_1 : memref<1024x1024xf32> to memref<1024x1024xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c32, %c32, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<1024x1024xf16>, %memref_0 : memref<1024x1024xf16>, %memref_1 : memref<1024x1024xf32>)
    gpu.dealloc  %memref : memref<1024x1024xf16>
    gpu.dealloc  %memref_0 : memref<1024x1024xf16>
    return %memref_1 : memref<1024x1024xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    func.func private @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32", linkage_type = <Import>>}
    func.func private @llvm.genx.dpas2.v128f32.v128i32.v64i32(vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas2.v128f32.v128i32.v64i32", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf16>) -> vector<512xf16> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8", linkage_type = <Import>>}
    func.func private @llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8(i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf32>) -> vector<512xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8", linkage_type = <Import>>}
    gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %cst = arith.constant dense<3> : vector<2xi8>
      %cst_0 = arith.constant dense<2> : vector<2xi8>
      %c6_i32 = arith.constant 6 : i32
      %c16_i32 = arith.constant 16 : i32
      %c32_i32 = arith.constant 32 : i32
      %c8_i32 = arith.constant 8 : i32
      %c10_i32 = arith.constant 10 : i32
      %c8_i8 = arith.constant 8 : i8
      %cst_1 = arith.constant dense<true> : vector<1xi1>
      %c4_i32 = arith.constant 4 : i32
      %c5_i8 = arith.constant 5 : i8
      %c3_i8 = arith.constant 3 : i8
      %c1_i16 = arith.constant 1 : i16
      %c0_i8 = arith.constant 0 : i8
      %c4_i8 = arith.constant 4 : i8
      %cst_2 = arith.constant dense<4> : vector<8xi32>
      %cst_3 = arith.constant dense<0.000000e+00> : vector<512xf16>
      %c8_i16 = arith.constant 8 : i16
      %cst_4 = arith.constant dense<0.000000e+00> : vector<128xf16>
      %c3847_i32 = arith.constant 3847 : i32
      %c2047_i32 = arith.constant 2047 : i32
      %c0_i32 = arith.constant 0 : i32
      %c32_i16 = arith.constant 32 : i16
      %c16_i16 = arith.constant 16 : i16
      %c1_i8 = arith.constant 1 : i8
      %true = arith.constant true
      %cst_5 = arith.constant dense<0.000000e+00> : vector<512xf32>
      %c7951_i32 = arith.constant 7951 : i32
      %c1807_i32 = arith.constant 1807 : i32
      %c1023_i32 = arith.constant 1023 : i32
      %c4095_i32 = arith.constant 4095 : i32
      %cst_6 = arith.constant dense<0> : vector<8xi64>
      %cst_7 = arith.constant dense<true> : vector<8xi1>
      %cst_8 = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56]> : vector<8xindex>
      %c64 = arith.constant 64 : index
      %c24 = arith.constant 24 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c32 : index
      %1 = arith.muli %block_id_y, %c32 : index
      %intptr = memref.extract_aligned_pointer_as_index %arg2 : memref<1024x1024xf32> -> index
      %2 = arith.index_castui %intptr : index to i64
      %3 = vector.insert %2, %cst_6 [0] : i64 into vector<8xi64>
      %4 = vector.bitcast %3 : vector<8xi64> to vector<16xi32>
      %5 = vector.insert %c4095_i32, %4 [2] : i32 into vector<16xi32>
      %6 = vector.insert %c1023_i32, %5 [3] : i32 into vector<16xi32>
      %7 = vector.insert %c4095_i32, %6 [4] : i32 into vector<16xi32>
      %8 = arith.index_castui %1 : index to i32
      %9 = arith.index_castui %0 : index to i32
      %10 = vector.insert %8, %7 [5] : i32 into vector<16xi32>
      %11 = vector.insert %9, %10 [6] : i32 into vector<16xi32>
      %12 = vector.insert %c1807_i32, %11 [7] : i32 into vector<16xi32>
      %13 = arith.addi %1, %c16 : index
      %14 = arith.index_castui %13 : index to i32
      %15 = vector.insert %14, %7 [5] : i32 into vector<16xi32>
      %16 = vector.insert %9, %15 [6] : i32 into vector<16xi32>
      %17 = vector.insert %c1807_i32, %16 [7] : i32 into vector<16xi32>
      %18 = arith.addi %0, %c8 : index
      %19 = arith.index_castui %18 : index to i32
      %20 = vector.insert %19, %10 [6] : i32 into vector<16xi32>
      %21 = vector.insert %c1807_i32, %20 [7] : i32 into vector<16xi32>
      %22 = vector.insert %19, %15 [6] : i32 into vector<16xi32>
      %23 = vector.insert %c1807_i32, %22 [7] : i32 into vector<16xi32>
      %24 = arith.addi %0, %c16 : index
      %25 = arith.index_castui %24 : index to i32
      %26 = vector.insert %25, %10 [6] : i32 into vector<16xi32>
      %27 = vector.insert %c1807_i32, %26 [7] : i32 into vector<16xi32>
      %28 = vector.insert %25, %15 [6] : i32 into vector<16xi32>
      %29 = vector.insert %c1807_i32, %28 [7] : i32 into vector<16xi32>
      %30 = arith.addi %0, %c24 : index
      %31 = arith.index_castui %30 : index to i32
      %32 = vector.insert %31, %10 [6] : i32 into vector<16xi32>
      %33 = vector.insert %c1807_i32, %32 [7] : i32 into vector<16xi32>
      %34 = vector.insert %31, %15 [6] : i32 into vector<16xi32>
      %35 = vector.insert %c1807_i32, %34 [7] : i32 into vector<16xi32>
      %36 = vector.insert %c7951_i32, %11 [7] : i32 into vector<16xi32>
      %37 = vector.insert %c7951_i32, %16 [7] : i32 into vector<16xi32>
      %38 = func.call @llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8(%true, %cst_0, %c1_i8, %c16_i16, %c32_i16, %36, %c0_i32, %c0_i32, %cst_5) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf32>) -> vector<512xf32>
      %39 = func.call @llvm.genx.lsc.load.2d.ugm.desc.v512f32.v2i8(%true, %cst_0, %c1_i8, %c16_i16, %c32_i16, %37, %c0_i32, %c0_i32, %cst_5) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf32>) -> vector<512xf32>
      %40 = vector.shuffle %38, %38 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<512xf32>, vector<512xf32>
      %41 = vector.shuffle %38, %38 [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<512xf32>, vector<512xf32>
      %42 = vector.shuffle %38, %38 [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383] : vector<512xf32>, vector<512xf32>
      %43 = vector.shuffle %38, %38 [384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xf32>, vector<512xf32>
      %44 = vector.shuffle %39, %39 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<512xf32>, vector<512xf32>
      %45 = vector.shuffle %39, %39 [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<512xf32>, vector<512xf32>
      %46 = vector.shuffle %39, %39 [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383] : vector<512xf32>, vector<512xf32>
      %47 = vector.shuffle %39, %39 [384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xf32>, vector<512xf32>
      %intptr_9 = memref.extract_aligned_pointer_as_index %arg0 : memref<1024x1024xf16> -> index
      %48 = arith.index_castui %intptr_9 : index to i64
      %49 = vector.insert %48, %cst_6 [0] : i64 into vector<8xi64>
      %50 = vector.bitcast %49 : vector<8xi64> to vector<16xi32>
      %51 = vector.insert %c2047_i32, %50 [2] : i32 into vector<16xi32>
      %52 = vector.insert %c1023_i32, %51 [3] : i32 into vector<16xi32>
      %53 = vector.insert %c2047_i32, %52 [4] : i32 into vector<16xi32>
      %54 = vector.insert %9, %53 [5] : i32 into vector<16xi32>
      %55 = vector.insert %c0_i32, %54 [6] : i32 into vector<16xi32>
      %56 = vector.insert %c3847_i32, %55 [7] : i32 into vector<16xi32>
      %57 = vector.insert %19, %53 [5] : i32 into vector<16xi32>
      %58 = vector.insert %c0_i32, %57 [6] : i32 into vector<16xi32>
      %59 = vector.insert %c3847_i32, %58 [7] : i32 into vector<16xi32>
      %60 = vector.insert %25, %53 [5] : i32 into vector<16xi32>
      %61 = vector.insert %c0_i32, %60 [6] : i32 into vector<16xi32>
      %62 = vector.insert %c3847_i32, %61 [7] : i32 into vector<16xi32>
      %63 = vector.insert %31, %53 [5] : i32 into vector<16xi32>
      %64 = vector.insert %c0_i32, %63 [6] : i32 into vector<16xi32>
      %65 = vector.insert %c3847_i32, %64 [7] : i32 into vector<16xi32>
      %66 = vector.insert %c16_i32, %54 [6] : i32 into vector<16xi32>
      %67 = vector.insert %c3847_i32, %66 [7] : i32 into vector<16xi32>
      %68 = vector.insert %c16_i32, %57 [6] : i32 into vector<16xi32>
      %69 = vector.insert %c3847_i32, %68 [7] : i32 into vector<16xi32>
      %70 = vector.insert %c16_i32, %60 [6] : i32 into vector<16xi32>
      %71 = vector.insert %c3847_i32, %70 [7] : i32 into vector<16xi32>
      %72 = vector.insert %c16_i32, %63 [6] : i32 into vector<16xi32>
      %73 = vector.insert %c3847_i32, %72 [7] : i32 into vector<16xi32>
      %intptr_10 = memref.extract_aligned_pointer_as_index %arg1 : memref<1024x1024xf16> -> index
      %74 = arith.index_castui %intptr_10 : index to i64
      %75 = vector.insert %74, %cst_6 [0] : i64 into vector<8xi64>
      %76 = vector.bitcast %75 : vector<8xi64> to vector<16xi32>
      %77 = vector.insert %c2047_i32, %76 [2] : i32 into vector<16xi32>
      %78 = vector.insert %c1023_i32, %77 [3] : i32 into vector<16xi32>
      %79 = vector.insert %c2047_i32, %78 [4] : i32 into vector<16xi32>
      %80 = vector.insert %8, %79 [5] : i32 into vector<16xi32>
      %81 = vector.insert %c0_i32, %80 [6] : i32 into vector<16xi32>
      %82 = vector.insert %c7951_i32, %81 [7] : i32 into vector<16xi32>
      %83 = vector.insert %14, %79 [5] : i32 into vector<16xi32>
      %84 = vector.insert %c0_i32, %83 [6] : i32 into vector<16xi32>
      %85 = vector.insert %c7951_i32, %84 [7] : i32 into vector<16xi32>
      %86:18 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %56, %arg5 = %59, %arg6 = %62, %arg7 = %65, %arg8 = %67, %arg9 = %69, %arg10 = %71, %arg11 = %73, %arg12 = %82, %arg13 = %85, %arg14 = %40, %arg15 = %44, %arg16 = %41, %arg17 = %45, %arg18 = %42, %arg19 = %46, %arg20 = %43, %arg21 = %47) -> (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>) {
        %87 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg4, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %88 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg5, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %89 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg6, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %90 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg7, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %91 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg8, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %92 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg9, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %93 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg10, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %94 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%true, %cst_0, %c1_i8, %c8_i16, %c16_i16, %arg11, %c0_i32, %c0_i32, %cst_4) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %95 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8(%true, %cst_0, %c1_i8, %c16_i16, %c32_i16, %arg12, %c0_i32, %c0_i32, %cst_3) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf16>) -> vector<512xf16>
        %96 = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v512f16.v2i8(%true, %cst_0, %c1_i8, %c16_i16, %c32_i16, %arg13, %c0_i32, %c0_i32, %cst_3) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<512xf16>) -> vector<512xf16>
        %97 = vector.shuffle %95, %95 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<512xf16>, vector<512xf16>
        %98 = vector.shuffle %95, %95 [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xf16>, vector<512xf16>
        %99 = vector.shuffle %96, %96 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<512xf16>, vector<512xf16>
        %100 = vector.shuffle %96, %96 [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xf16>, vector<512xf16>
        %101 = vector.bitcast %87 : vector<128xf16> to vector<64xf32>
        %alloc = memref.alloc() : memref<4096xf32, 3>
        %102 = gpu.subgroup_id : index
        %103 = arith.muli %102, %c64 : index
        %104 = vector.splat %103 : vector<8xindex>
        %105 = arith.addi %104, %cst_8 : vector<8xindex>
        %intptr_11 = memref.extract_aligned_pointer_as_index %alloc : memref<4096xf32, 3> -> index
        %106 = arith.index_castui %intptr_11 : index to i32
        %107 = arith.index_castui %105 : vector<8xindex> to vector<8xi32>
        %108 = arith.muli %107, %cst_2 : vector<8xi32>
        %109 = vector.splat %106 : vector<8xi32>
        %110 = arith.addi %109, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %110, %101, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %111 = arith.index_castui %103 : index to i32
        %112 = arith.muli %111, %c4_i32 : i32
        %113 = arith.addi %106, %112 : i32
        %114 = builtin.unrealized_conversion_cast %113 : i32 to vector<1xi32>
        %115 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %114, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %116 = vector.bitcast %91 : vector<128xf16> to vector<64xf32>
        %alloc_12 = memref.alloc() : memref<4096xf32, 3>
        %intptr_13 = memref.extract_aligned_pointer_as_index %alloc_12 : memref<4096xf32, 3> -> index
        %117 = arith.index_castui %intptr_13 : index to i32
        %118 = vector.splat %117 : vector<8xi32>
        %119 = arith.addi %118, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %119, %116, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %120 = arith.addi %117, %112 : i32
        %121 = builtin.unrealized_conversion_cast %120 : i32 to vector<1xi32>
        %122 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %121, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %123 = vector.bitcast %88 : vector<128xf16> to vector<64xf32>
        %alloc_14 = memref.alloc() : memref<4096xf32, 3>
        %intptr_15 = memref.extract_aligned_pointer_as_index %alloc_14 : memref<4096xf32, 3> -> index
        %124 = arith.index_castui %intptr_15 : index to i32
        %125 = vector.splat %124 : vector<8xi32>
        %126 = arith.addi %125, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %126, %123, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %127 = arith.addi %124, %112 : i32
        %128 = builtin.unrealized_conversion_cast %127 : i32 to vector<1xi32>
        %129 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %128, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %130 = vector.bitcast %92 : vector<128xf16> to vector<64xf32>
        %alloc_16 = memref.alloc() : memref<4096xf32, 3>
        %intptr_17 = memref.extract_aligned_pointer_as_index %alloc_16 : memref<4096xf32, 3> -> index
        %131 = arith.index_castui %intptr_17 : index to i32
        %132 = vector.splat %131 : vector<8xi32>
        %133 = arith.addi %132, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %133, %130, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %134 = arith.addi %131, %112 : i32
        %135 = builtin.unrealized_conversion_cast %134 : i32 to vector<1xi32>
        %136 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %135, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %137 = vector.bitcast %89 : vector<128xf16> to vector<64xf32>
        %alloc_18 = memref.alloc() : memref<4096xf32, 3>
        %intptr_19 = memref.extract_aligned_pointer_as_index %alloc_18 : memref<4096xf32, 3> -> index
        %138 = arith.index_castui %intptr_19 : index to i32
        %139 = vector.splat %138 : vector<8xi32>
        %140 = arith.addi %139, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %140, %137, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %141 = arith.addi %138, %112 : i32
        %142 = builtin.unrealized_conversion_cast %141 : i32 to vector<1xi32>
        %143 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %142, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %144 = vector.bitcast %93 : vector<128xf16> to vector<64xf32>
        %alloc_20 = memref.alloc() : memref<4096xf32, 3>
        %intptr_21 = memref.extract_aligned_pointer_as_index %alloc_20 : memref<4096xf32, 3> -> index
        %145 = arith.index_castui %intptr_21 : index to i32
        %146 = vector.splat %145 : vector<8xi32>
        %147 = arith.addi %146, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %147, %144, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %148 = arith.addi %145, %112 : i32
        %149 = builtin.unrealized_conversion_cast %148 : i32 to vector<1xi32>
        %150 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %149, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %151 = vector.bitcast %90 : vector<128xf16> to vector<64xf32>
        %alloc_22 = memref.alloc() : memref<4096xf32, 3>
        %intptr_23 = memref.extract_aligned_pointer_as_index %alloc_22 : memref<4096xf32, 3> -> index
        %152 = arith.index_castui %intptr_23 : index to i32
        %153 = vector.splat %152 : vector<8xi32>
        %154 = arith.addi %153, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %154, %151, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %155 = arith.addi %152, %112 : i32
        %156 = builtin.unrealized_conversion_cast %155 : i32 to vector<1xi32>
        %157 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %156, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %158 = vector.bitcast %94 : vector<128xf16> to vector<64xf32>
        %alloc_24 = memref.alloc() : memref<4096xf32, 3>
        %intptr_25 = memref.extract_aligned_pointer_as_index %alloc_24 : memref<4096xf32, 3> -> index
        %159 = arith.index_castui %intptr_25 : index to i32
        %160 = vector.splat %159 : vector<8xi32>
        %161 = arith.addi %160, %108 : vector<8xi32>
        func.call @llvm.genx.lsc.store.slm.v8i1.v8i32.v64f32(%cst_7, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c5_i8, %c1_i8, %c0_i8, %161, %158, %c0_i32) : (vector<8xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<8xi32>, vector<64xf32>, i32) -> ()
        %162 = arith.addi %159, %112 : i32
        %163 = builtin.unrealized_conversion_cast %162 : i32 to vector<1xi32>
        %164 = func.call @llvm.genx.lsc.load.slm.v64f32.v1i1.v1i32(%cst_1, %c0_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c8_i8, %c1_i8, %c0_i8, %163, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi32>, i32) -> vector<64xf32>
        %165 = vector.bitcast %115 : vector<64xf32> to vector<64xi32>
        %166 = vector.bitcast %97 : vector<256xf16> to vector<128xi32>
        %167 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg14, %166, %165, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %168 = vector.bitcast %122 : vector<64xf32> to vector<64xi32>
        %169 = vector.bitcast %98 : vector<256xf16> to vector<128xi32>
        %170 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%167, %169, %168, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %171 = vector.bitcast %99 : vector<256xf16> to vector<128xi32>
        %172 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg15, %171, %165, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %173 = vector.bitcast %100 : vector<256xf16> to vector<128xi32>
        %174 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%172, %173, %168, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %175 = vector.bitcast %129 : vector<64xf32> to vector<64xi32>
        %176 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg16, %166, %175, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %177 = vector.bitcast %136 : vector<64xf32> to vector<64xi32>
        %178 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%176, %169, %177, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %179 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg17, %171, %175, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %180 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%179, %173, %177, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %181 = vector.bitcast %143 : vector<64xf32> to vector<64xi32>
        %182 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg18, %166, %181, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %183 = vector.bitcast %150 : vector<64xf32> to vector<64xi32>
        %184 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%182, %169, %183, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %185 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg19, %171, %181, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %186 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%185, %173, %183, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %187 = vector.bitcast %157 : vector<64xf32> to vector<64xi32>
        %188 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg20, %166, %187, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %189 = vector.bitcast %164 : vector<64xf32> to vector<64xi32>
        %190 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%188, %169, %189, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %191 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%arg21, %171, %187, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %192 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%191, %173, %189, %c10_i32, %c10_i32, %c8_i32, %c8_i32, %c0_i32, %c0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %193 = vector.extractelement %arg4[%c6_i32 : i32] : vector<16xi32>
        %194 = arith.addi %193, %c32_i32 : i32
        %195 = vector.insert %194, %arg4 [6] : i32 into vector<16xi32>
        %196 = vector.extractelement %arg5[%c6_i32 : i32] : vector<16xi32>
        %197 = arith.addi %196, %c32_i32 : i32
        %198 = vector.insert %197, %arg5 [6] : i32 into vector<16xi32>
        %199 = vector.extractelement %arg6[%c6_i32 : i32] : vector<16xi32>
        %200 = arith.addi %199, %c32_i32 : i32
        %201 = vector.insert %200, %arg6 [6] : i32 into vector<16xi32>
        %202 = vector.extractelement %arg7[%c6_i32 : i32] : vector<16xi32>
        %203 = arith.addi %202, %c32_i32 : i32
        %204 = vector.insert %203, %arg7 [6] : i32 into vector<16xi32>
        %205 = vector.extractelement %arg8[%c6_i32 : i32] : vector<16xi32>
        %206 = arith.addi %205, %c32_i32 : i32
        %207 = vector.insert %206, %arg8 [6] : i32 into vector<16xi32>
        %208 = vector.extractelement %arg9[%c6_i32 : i32] : vector<16xi32>
        %209 = arith.addi %208, %c32_i32 : i32
        %210 = vector.insert %209, %arg9 [6] : i32 into vector<16xi32>
        %211 = vector.extractelement %arg10[%c6_i32 : i32] : vector<16xi32>
        %212 = arith.addi %211, %c32_i32 : i32
        %213 = vector.insert %212, %arg10 [6] : i32 into vector<16xi32>
        %214 = vector.extractelement %arg11[%c6_i32 : i32] : vector<16xi32>
        %215 = arith.addi %214, %c32_i32 : i32
        %216 = vector.insert %215, %arg11 [6] : i32 into vector<16xi32>
        %217 = vector.extractelement %arg12[%c6_i32 : i32] : vector<16xi32>
        %218 = arith.addi %217, %c32_i32 : i32
        %219 = vector.insert %218, %arg12 [6] : i32 into vector<16xi32>
        %220 = vector.extractelement %arg13[%c6_i32 : i32] : vector<16xi32>
        %221 = arith.addi %220, %c32_i32 : i32
        %222 = vector.insert %221, %arg13 [6] : i32 into vector<16xi32>
        scf.yield %195, %198, %201, %204, %207, %210, %213, %216, %219, %222, %170, %174, %178, %180, %184, %186, %190, %192 : vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>
      }
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %12, %c0_i32, %c0_i32, %86#10) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %17, %c0_i32, %c0_i32, %86#11) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %21, %c0_i32, %c0_i32, %86#12) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %23, %c0_i32, %c0_i32, %86#13) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %27, %c0_i32, %c0_i32, %86#14) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %29, %c0_i32, %c0_i32, %86#15) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %33, %c0_i32, %c0_i32, %86#16) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%true, %cst, %c1_i8, %c16_i16, %c8_i16, %35, %c0_i32, %c0_i32, %86#17) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %cst_0 = arith.constant 0.000000e+00 : f16
    %cst_1 = arith.constant 1.000000e+00 : f16
    %alloc = memref.alloc() : memref<1024x1024xf16>
    %alloc_2 = memref.alloc() : memref<1024x1024xf16>
    %alloc_3 = memref.alloc() : memref<1024x1024xf32>
    %alloc_4 = memref.alloc() : memref<1024x1024xf32>
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = index.castu %arg1 : index to i16
        %2 = arith.uitofp %1 : i16 to f16
        memref.store %2, %alloc[%arg0, %arg1] : memref<1024x1024xf16>
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = index.castu %arg0 : index to i32
        %2 = index.castu %arg1 : index to i32
        %3 = arith.cmpi eq, %1, %2 : i32
        scf.if %3 {
          memref.store %cst_1, %alloc_2[%arg0, %arg1] : memref<1024x1024xf16>
        } else {
          memref.store %cst_0, %alloc_2[%arg0, %arg1] : memref<1024x1024xf16>
        }
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        memref.store %cst, %alloc_3[%arg0, %arg1] : memref<1024x1024xf32>
        memref.store %cst, %alloc_4[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = memref.load %alloc_4[%arg0, %arg1] : memref<1024x1024xf32>
        %2 = scf.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %1) -> (f32) {
          %3 = memref.load %alloc[%arg2, %arg0] : memref<1024x1024xf16>
          %4 = memref.load %alloc_2[%arg2, %arg1] : memref<1024x1024xf16>
          %5 = arith.mulf %3, %4 : f16
          %6 = arith.extf %5 : f16 to f32
          %7 = arith.addf %6, %arg3 : f32
          scf.yield %7 : f32
        }
        memref.store %2, %alloc_4[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    %0 = call @test(%alloc, %alloc_2, %alloc_3) : (memref<1024x1024xf16>, memref<1024x1024xf16>, memref<1024x1024xf32>) -> memref<1024x1024xf32>
    %cast = memref.cast %0 : memref<1024x1024xf32> to memref<*xf32>
    %cast_5 = memref.cast %alloc_4 : memref<1024x1024xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_5) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<1024x1024xf16>
    memref.dealloc %alloc_2 : memref<1024x1024xf16>
    memref.dealloc %alloc_3 : memref<1024x1024xf32>
    memref.dealloc %alloc_4 : memref<1024x1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}


Error: entry point not found
