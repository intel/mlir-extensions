// TODO: Add run commands
// RUN:

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
  func.func @test(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %A_gpu = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %A, %A_gpu : memref<1024x1024xf16> to memref<1024x1024xf16>
    %B_gpu = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %B, %B_gpu : memref<1024x1024xf16> to memref<1024x1024xf16>
    %C_gpu = gpu.alloc  host_shared () : memref<1024x1024xf32>
    memref.copy %C, %C_gpu : memref<1024x1024xf32> to memref<1024x1024xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c8, %c8, %c1) threads in (%c2, %c2, %c1) args(%A_gpu : memref<1024x1024xf16>, %B_gpu : memref<1024x1024xf16>, %C_gpu : memref<1024x1024xf32>)
    gpu.dealloc  %A_gpu : memref<1024x1024xf16>
    gpu.dealloc  %B_gpu : memref<1024x1024xf16>
    return %C_gpu : memref<1024x1024xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        // %c8 = arith.constant 8 : index
        // %c16 = arith.constant 16 : index
        %c32 = arith.constant 32 : index
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %c1024 = arith.constant 1024 : index
        %block_id_x = gpu.block_id x
        %block_id_y = gpu.block_id y
        %m = arith.muli %block_id_x, %c128 : index
        %n = arith.muli %block_id_y, %c128 : index

        // get linear sub group id
        %sg_id = gpu.subgroup_id : index
        // get the x, y cordinate of this linear id assuming [2, 2] coord system
        %c2 = arith.constant 2 : index
        %sg_coord_x = index.floordivs %sg_id, %c2
        %sg_coord_y = index.and %sg_id, %c1

        // each subgroup in the [2, 2] subgroups needs to update four 32x32 C sub-tiles
        // that are arranged in round robin fashin according to SG coords
        // | (0,0) | (0,1) | (0,0) | (0,1) |
        // | (1,0) | (1,1) | (1,0) | (1,1) |
        // | (0,0) | (0,1) | (0,0) | (0,1) |
        // | (1,0) | (1,1) | (1,0) | (1,1) |
        // first calculate the offset into the first SG sub-tile
        %C_sg_tile_offset_x = index.mul %c32, %sg_coord_x
        %C_sg_tile_offset_y = index.mul %c32, %sg_coord_y

        // C sub tiles
        // global offset for sub tile 1 for this SG
        %global_offset_slice0_x = index.add %m, %C_sg_tile_offset_x
        %global_offset_slice0_y = index.add %n, %C_sg_tile_offset_y
        // global offset for sub tile 2 for this SG (shift 64 in x)
        %global_offset_slice1_x = index.add %global_offset_slice0_x, %c64
        %global_offset_slice1_y = index.add %global_offset_slice0_y, %c0
        // global offset for sub tile 3 for this SG (shift 64 in y)
        %global_offset_slice2_x = index.add %global_offset_slice0_x, %c0
        %global_offset_slice2_y = index.add %global_offset_slice0_y, %c64
        // global offset for sub tile 4 for this SG (shift 64 in x and y)
        %global_offset_slice3_x = index.add %global_offset_slice0_x, %c64
        %global_offset_slice3_y = index.add %global_offset_slice0_y, %c64

        // intialize C sub tiles and load them
        %c_init_subtile0 = xetile.init_tile %C[%global_offset_slice0_x, %global_offset_slice0_y] : memref<1024x1024xf32>
          -> !xetile.tile<32x32xf32>
        %c_init_value0 = xetile.load_tile %c_init_subtile0 : !xetile.tile<32x32xf32>
          -> vector<32x32xf32>
        %c_init_subtile1 = xetile.init_tile %C[%global_offset_slice1_x, %global_offset_slice1_y] : memref<1024x1024xf32>
          -> !xetile.tile<32x32xf32>
        %c_init_value1 = xetile.load_tile %c_init_subtile1 : !xetile.tile<32x32xf32>
          -> vector<32x32xf32>
        %c_init_subtile2 = xetile.init_tile %C[%global_offset_slice2_x, %global_offset_slice2_y] : memref<1024x1024xf32>
          -> !xetile.tile<32x32xf32>
        %c_init_value2 = xetile.load_tile %c_init_subtile2 : !xetile.tile<32x32xf32>
          -> vector<32x32xf32>
        %c_init_subtile3 = xetile.init_tile %C[%global_offset_slice3_x, %global_offset_slice3_y] : memref<1024x1024xf32>
          -> !xetile.tile<32x32xf32>
        %c_init_value3 = xetile.load_tile %c_init_subtile2 : !xetile.tile<32x32xf32>
          -> vector<32x32xf32>

        // for A, each subgroup need to load two 32x128 subtiles. The access arrangement is as follows
        // | (0,0), (0,1)|
        // | (1,0), (1,1)|
        // | (0,0), (0,1)|
        // | (1,0), (1,1)|

        // calculate the initial offset in x dim for this sg
        %a_init_offset = index.mul %sg_coord_x, %c32

        // x offsets for A subtiles
        %a_subtile0_x = index.add %m, %a_init_offset
        %a_subtile1_x = index.add %a_subtile0_x, %c64

        // init A subtiles
        %a_init_subtile0 = xetile.init_tile %A[%a_subtile0_x, %c0] : memref<1024x1024xf16>
          -> !xetile.tile<32x128xf16>
        %a_init_subtile1 = xetile.init_tile %A[%a_subtile1_x, %c0] : memref<1024x1024xf16>
          -> !xetile.tile<32x128xf16>

        // for B, each subgroup need to load two 128x32 subtiles. The access arrangement is as follows
        // | (0,0) | (0,1) | (0,0) | (0, 1) |
        // | (1,0) | (1,1) | (1,0) | (1, 1) |

        // calculate the initial offset along y dim for this sg
        %b_init_offset = index.mul %sg_coord_y, %c32

        // y offsets for B subtiles
        %b_subtile0_y = index.add %n, %b_init_offset
        %b_subtile1_y = index.add %b_subtile0_y, %c64

        // init B subtiles
        %b_init_subtile0 = xetile.init_tile %B[%c0, %b_subtile0_y] : memref<1024x1024xf16>
          -> !xetile.tile<128x32xf16>
        %b_init_subtile1 = xetile.init_tile %B[%c0, %b_subtile1_y] : memref<1024x1024xf16>
          -> !xetile.tile<128x32xf16>

        // compute the value of C subtiles by iterating over subtiles in k-dimension and doing dpas
        %out:8 = scf.for %k = %c0 to %c1024 step %c128
          iter_args(%a_subtile0 = %a_init_subtile0,  %a_subtile1 = %a_init_subtile1,
                     %b_subtile0 = %b_init_subtile0, %b_subtile1 = %b_init_subtile1,
                    %c_value0 = %c_init_value0, %c_value1 = %c_init_value2,
                    %c_value2 = %c_init_value2, %c_value3 = %c_init_value3)
          -> (!xetile.tile<32x128xf16>,
              !xetile.tile<32x128xf16>,
              !xetile.tile<128x32xf16>,
              !xetile.tile<128x32xf16>,
              vector<32x32xf32>, vector<32x32xf32>, vector<32x32xf32>, vector<32x32xf32>) {

          // load A subtiles
          %a_value0 = xetile.load_tile %a_subtile0 : !xetile.tile<32x128xf16>
            -> vector<32x128xf16>
          %a_value1 = xetile.load_tile %a_subtile1 : !xetile.tile<32x128xf16>
            -> vector<32x128xf16>

          // load B subtiles
          %b_value0 = xetile.load_tile %b_subtile0 : !xetile.tile<128x32xf16>
            -> vector<128x32xf16>
          %b_value1 = xetile.load_tile %b_subtile1 : !xetile.tile<128x32xf16>
            -> vector<128x32xf16>

          // perform 4 dpas ops and update the C subtiles
          %c_new_value0 = xetile.tile_mma %a_value0, %b_value0, %c_value0
            : vector<32x128xf16>, vector<128x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
          %c_new_value1 = xetile.tile_mma %a_value0, %b_value1, %c_value1
            : vector<32x128xf16>, vector<128x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
          %c_new_value2 = xetile.tile_mma %a_value1, %b_value0, %c_value2
            : vector<32x128xf16>, vector<128x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
          %c_new_value3 = xetile.tile_mma %a_value1, %b_value1, %c_value3
            : vector<32x128xf16>, vector<128x32xf16>, vector<32x32xf32> -> vector<32x32xf32>

          // update offsets for A subtiles
          %a_next_subtile0 = xetile.update_tile_offset %a_subtile0, [%c0, %c128] : !xetile.tile<32x128xf16>
          %a_next_subtile1 = xetile.update_tile_offset %a_subtile1, [%c0, %c128] : !xetile.tile<32x128xf16>
          // update offsets for B subtiles
          %b_next_subtile0 = xetile.update_tile_offset %b_subtile0, [%c128, %c0] : !xetile.tile<128x32xf16>
          %b_next_subtile1 = xetile.update_tile_offset %b_subtile1, [%c128, %c0] : !xetile.tile<128x32xf16>

          // yield subtiles and partial C results
          scf.yield %a_next_subtile0, %a_next_subtile1, %b_next_subtile0, %b_next_subtile1,
            %c_new_value0, %c_new_value1, %c_new_value2, %c_new_value2
            : !xetile.tile<32x128xf16>,
              !xetile.tile<32x128xf16>,
              !xetile.tile<128x32xf16>,
              !xetile.tile<128x32xf16>,
             vector<32x32xf32>, vector<32x32xf32>, vector<32x32xf32>, vector<32x32xf32>
        }
        // store the C final subtiles into memory
        xetile.store_tile %out#4, %c_init_subtile0 : vector<32x32xf32>,
          !xetile.tile<32x32xf32>
        xetile.store_tile %out#5, %c_init_subtile1 : vector<32x32xf32>,
          !xetile.tile<32x32xf32>
        xetile.store_tile %out#6, %c_init_subtile2 : vector<32x32xf32>,
          !xetile.tile<32x32xf32>
        xetile.store_tile %out#7, %c_init_subtile3 : vector<32x32xf32>,
          !xetile.tile<32x32xf32>

        gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %cf_0 = arith.constant 0.0 : f16
    %cf_1 = arith.constant 1.0 : f16
    %A = memref.alloc() : memref<1024x1024xf16>
    %B = memref.alloc() : memref<1024x1024xf16>
    %C = memref.alloc() : memref<1024x1024xf32>
    %C_ref = memref.alloc() : memref<1024x1024xf32>
    // intialize matrix A ; A[i, j] = j
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        %t = index.castu %j : index to i16
        %val = arith.uitofp %t : i16 to f16
        memref.store %val, %A[%i, %j] : memref<1024x1024xf16>
      }
    }
    // make matrix B an identity matrix
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        %i_i32 = index.castu %i : index to i32
        %j_i32 = index.castu %j : index to i32
        %i_j_same = arith.cmpi eq, %i_i32, %j_i32 : i32

        scf.if %i_j_same {
          memref.store %cf_1, %B[%i, %j] : memref<1024x1024xf16>
        } else {
          memref.store %cf_0, %B[%i, %j] : memref<1024x1024xf16>
        }
      }
    }
    // intialize matrix C and C_ref ; C[i, j] = 0
    %c0_f32 = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        memref.store %c0_f32, %C[%i, %j] : memref<1024x1024xf32>
        memref.store %c0_f32, %C_ref[%i, %j] : memref<1024x1024xf32>
      }
    }
    // compute C for reference
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        %c_curr = memref.load %C_ref[%i, %j] : memref<1024x1024xf32>
        %c_val = scf.for %k = %c0 to %c1024 step %c1 iter_args(%c_partial = %c_curr) -> f32 {
          %a_val = memref.load %A[%i, %k] : memref<1024x1024xf16>
          %b_val = memref.load %B[%k, %j] : memref<1024x1024xf16>
          %t = arith.mulf %a_val, %b_val : f16
          %t_cast = arith.extf %t : f16 to f32
          %c_sum = arith.addf %t_cast, %c_partial : f32
          scf.yield %c_sum : f32
        }
        memref.store %c_val , %C_ref[%i, %j] : memref<1024x1024xf32>
      }
    }
    %2 = call @test(%A, %B, %C) : (memref<1024x1024xf16>, memref<1024x1024xf16>, memref<1024x1024xf32>) -> memref<1024x1024xf32>
    %cast_C = memref.cast %2 : memref<1024x1024xf32> to memref<*xf32>
    %cast_C_ref = memref.cast %C_ref : memref<1024x1024xf32> to memref<*xf32>

    call @printAllcloseF32(%cast_C, %cast_C_ref) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %A : memref<1024x1024xf16>
    memref.dealloc %B : memref<1024x1024xf16>
    memref.dealloc %C : memref<1024x1024xf32>
    memref.dealloc %C_ref : memref<1024x1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
