// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

// TODO: Add run commands
// NOTES:
// This example assumes 2x2 subgroups per one workgroup and the kernel specifies the computation
// done by a single subgroup. This shows the result of lowering wg_gemm_1kx1kx1k_f16_f16_f32 example
// assuming the following layout maps.
//
// #wg_map_a = #xetile.wg_map<sg_layout = [2, 2], sg_data = [32, 128]>
// #xe_map_a = #xetile.xe_map<wg = #wg_map_a>
//
// #wg_map_b = #xetile.wg_map<sg_layout = [2, 2], sg_data = [128, 32]>
// #xe_map_b = #xetile.xe_map<wg = #wg_map_b>
//
// #wg_map_c = #xetile.wg_map<sg_layout = [2, 2], sg_data = [32, 32]>
// #xe_map_c = #xetile.xe_map<wg = #wg_map_c>
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %memref = gpu.alloc  () : memref<1024x1024xf16>
    gpu.memcpy  %memref, %arg0 : memref<1024x1024xf16>, memref<1024x1024xf16>
    %memref_0 = gpu.alloc  () : memref<1024x1024xf16>
    gpu.memcpy  %memref_0, %arg1 : memref<1024x1024xf16>, memref<1024x1024xf16>
    %memref_1 = gpu.alloc  () : memref<1024x1024xf32>
    gpu.memcpy  %memref_1, %arg2 : memref<1024x1024xf32>, memref<1024x1024xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c8, %c8, %c1) threads in (%c2, %c2, %c1)  args(%memref : memref<1024x1024xf16>, %memref_0 : memref<1024x1024xf16>, %memref_1 : memref<1024x1024xf32>)
    gpu.dealloc  %memref : memref<1024x1024xf16>
    gpu.dealloc  %memref_0 : memref<1024x1024xf16>
    %alloc = memref.alloc() : memref<1024x1024xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<1024x1024xf32>, memref<1024x1024xf32>
    gpu.dealloc  %memref_1 : memref<1024x1024xf32>
    return %alloc : memref<1024x1024xf32>
  }
  gpu.module @test_kernel  {
        // %c8 = arith.constant 8 : index
        // %c16 = arith.constant 16 : index
        // get linear sub group id
        // get the x, y cordinate of this linear id assuming [2, 2] coord system
        // each subgroup in the [2, 2] subgroups needs to update four 32x32 C sub-tiles
        // that are arranged in round robin fashin according to SG coords
        // | (0,0) | (0,1) | (0,0) | (0,1) |
        // | (1,0) | (1,1) | (1,0) | (1,1) |
        // | (0,0) | (0,1) | (0,0) | (0,1) |
        // | (1,0) | (1,1) | (1,0) | (1,1) |
        // first calculate the offset into the first SG sub-tile
        // C sub tiles
        // global offset for sub tile 1 for this SG
        // global offset for sub tile 2 for this SG (shift 64 in x)
        // global offset for sub tile 3 for this SG (shift 64 in y)
        // global offset for sub tile 4 for this SG (shift 64 in x and y)
        // intialize C sub tiles and load them
        // for A, each subgroup need to load two 32x128 subtiles. The access arrangement is as follows
        // | (0,0), (0,1)|
        // | (1,0), (1,1)|
        // | (0,0), (0,1)|
        // | (1,0), (1,1)|
        // calculate the initial offset in x dim for this sg
        // x offsets for A subtiles
        // init A subtiles
        // for B, each subgroup need to load two 128x32 subtiles. The access arrangement is as follows
        // | (0,0) | (0,1) | (0,0) | (0, 1) |
        // | (1,0) | (1,1) | (1,0) | (1, 1) |
        // calculate the initial offset along y dim for this sg
        // y offsets for B subtiles
        // init B subtiles
        // compute the value of C subtiles by iterating over subtiles in k-dimension and doing dpas
          // load A subtiles
          // load B subtiles
          // perform 4 dpas ops and update the C subtiles
          // update offsets for A subtiles
          // update offsets for B subtiles
          // yield subtiles and partial C results
        // store the C final subtiles into memory
    gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c1024 = arith.constant 1024 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c128 : index
      %1 = arith.muli %block_id_y, %c128 : index
      %2 = gpu.subgroup_id : index
      %c2 = arith.constant 2 : index
      %3 = index.floordivs %2, %c2
      %4 = index.and %2, %c1
      %5 = index.mul %c32, %3
      %6 = index.mul %c32, %4
      %7 = index.add %0, %5
      %8 = index.add %1, %6
      %9 = index.add %7, %c64
      %10 = index.add %8, %c0
      %11 = index.add %7, %c0
      %12 = index.add %8, %c64
      %13 = index.add %7, %c64
      %14 = index.add %8, %c64
      %15 = xetile.init_tile %arg2[%7, %8] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
      %16 = xetile.load_tile %15 : !xetile.tile<32x32xf32> -> vector<32x32xf32>
      %17 = xetile.init_tile %arg2[%9, %10] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
      %18 = xetile.load_tile %17 : !xetile.tile<32x32xf32> -> vector<32x32xf32>
      %19 = xetile.init_tile %arg2[%11, %12] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
      %20 = xetile.load_tile %19 : !xetile.tile<32x32xf32> -> vector<32x32xf32>
      %21 = xetile.init_tile %arg2[%13, %14] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
      %22 = xetile.load_tile %19 : !xetile.tile<32x32xf32> -> vector<32x32xf32>
      %23 = index.mul %3, %c32
      %24 = index.add %0, %23
      %25 = index.add %24, %c64
      %26 = xetile.init_tile %arg0[%24, %c0] : memref<1024x1024xf16> -> !xetile.tile<32x128xf16>
      %27 = xetile.init_tile %arg0[%25, %c0] : memref<1024x1024xf16> -> !xetile.tile<32x128xf16>
      %28 = index.mul %4, %c32
      %29 = index.add %1, %28
      %30 = index.add %29, %c64
      %31 = xetile.init_tile %arg1[%c0, %29] : memref<1024x1024xf16> -> !xetile.tile<128x32xf16>
      %32 = xetile.init_tile %arg1[%c0, %30] : memref<1024x1024xf16> -> !xetile.tile<128x32xf16>
      %33:8 = scf.for %arg3 = %c0 to %c1024 step %c128 iter_args(%arg4 = %26, %arg5 = %27, %arg6 = %31, %arg7 = %32, %arg8 = %16, %arg9 = %20, %arg10 = %20, %arg11 = %22) -> (!xetile.tile<32x128xf16>, !xetile.tile<32x128xf16>, !xetile.tile<128x32xf16>, !xetile.tile<128x32xf16>, vector<32x32xf32>, vector<32x32xf32>, vector<32x32xf32>, vector<32x32xf32>) {
        %34 = xetile.load_tile %arg4 : !xetile.tile<32x128xf16> -> vector<32x128xf16>
        %35 = xetile.load_tile %arg5 : !xetile.tile<32x128xf16> -> vector<32x128xf16>
        %36 = xetile.load_tile %arg6 : !xetile.tile<128x32xf16> -> vector<128x32xf16>
        %37 = xetile.load_tile %arg7 : !xetile.tile<128x32xf16> -> vector<128x32xf16>
        %38 = xetile.tile_mma %34, %36, %arg8 : vector<32x128xf16>, vector<128x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
        %39 = xetile.tile_mma %34, %37, %arg9 : vector<32x128xf16>, vector<128x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
        %40 = xetile.tile_mma %35, %36, %arg10 : vector<32x128xf16>, vector<128x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
        %41 = xetile.tile_mma %35, %37, %arg11 : vector<32x128xf16>, vector<128x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
        %42 = xetile.update_tile_offset %arg4, [%c0, %c128] : !xetile.tile<32x128xf16>
        %43 = xetile.update_tile_offset %arg5, [%c0, %c128] : !xetile.tile<32x128xf16>
        %44 = xetile.update_tile_offset %arg6, [%c128, %c0] : !xetile.tile<128x32xf16>
        %45 = xetile.update_tile_offset %arg7, [%c128, %c0] : !xetile.tile<128x32xf16>
        scf.yield %42, %43, %44, %45, %38, %39, %40, %40 : !xetile.tile<32x128xf16>, !xetile.tile<32x128xf16>, !xetile.tile<128x32xf16>, !xetile.tile<128x32xf16>, vector<32x32xf32>, vector<32x32xf32>, vector<32x32xf32>, vector<32x32xf32>
      }
      xetile.store_tile %33#4,  %15 : vector<32x32xf32>, !xetile.tile<32x32xf32>
      xetile.store_tile %33#5,  %17 : vector<32x32xf32>, !xetile.tile<32x32xf32>
      xetile.store_tile %33#6,  %19 : vector<32x32xf32>, !xetile.tile<32x32xf32>
      xetile.store_tile %33#7,  %21 : vector<32x32xf32>, !xetile.tile<32x32xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    // intialize matrix A ; A[i, j] = j
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
    // make matrix B an identity matrix
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
    // intialize matrix C and C_ref ; C[i, j] = 0
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        memref.store %cst, %alloc_3[%arg0, %arg1] : memref<1024x1024xf32>
        memref.store %cst, %alloc_4[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    // compute C for reference
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = memref.load %alloc_4[%arg0, %arg1] : memref<1024x1024xf32>
        %2 = scf.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %1) -> (f32) {
          %3 = memref.load %alloc[%arg0, %arg2] : memref<1024x1024xf16>
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
