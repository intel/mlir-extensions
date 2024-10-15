// RUN:  imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking --canonicalize %s -verify-diagnostics -o -| FileCheck %s
module @gemm attributes {gpu.container_module} {
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {

    //CHECK: gpu.func @test_kernel(%[[arg0:.*]]: memref<4096x4096xf16>, %[[arg1:.*]]: memref<4096x4096xf16>, %[[arg2:.*]]: memref<4096x4096xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_kernel(%A: memref<4096x4096xf16>, %B: memref<4096x4096xf16>, %C: memref<4096x4096xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      //CHECK: %[[c0_i8:.*]] = arith.constant 0 : i8
      //CHECK: %[[c1_i8:.*]] = arith.constant 1 : i8
      //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<4x4x8x16xf32>
      //CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
      //CHECK: %[[c64:.*]] = arith.constant 64 : index
      //CHECK: %[[c8:.*]] = arith.constant 8 : index
      //CHECK: %[[c4:.*]] = arith.constant 4 : index
      //CHECK: %[[c4096:.*]] = arith.constant 4096 : index
      //CHECK: %[[c32:.*]] = arith.constant 32 : index
      //CHECK: %[[c256:.*]] = arith.constant 256 : index
      //CHECK: %[[c0:.*]] = arith.constant 0 : index
      //CHECK: %[[c3:.*]] = arith.constant 3 : index
      //CHECK: %[[c1:.*]] = arith.constant 1 : index
      %c0_23 = arith.constant 0 : index
      %c3 = arith.constant 3 : index
      %c1_24 = arith.constant 1 : index
      //CHECK: scf.for %[[arg3:.*]] = %[[c0]] to %[[c3]] step %[[c1]] {
      scf.for %arg17 = %c0_23 to %c3 step %c1_24 {
        %c256 = arith.constant 256 : index
        %c512 = arith.constant 512 : index
        %c128 = arith.constant 128 : index
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
        %c0_i32 = arith.constant 0 : i32
        //CHECK: %[[R0:.*]] = gpu.block_id  x
        //CHECK: %[[R1:.*]] = gpu.block_id  y
        //CHECK: %[[R2:.*]] = gpu.global_id  x
        //CHECK: %[[R3:.*]] = gpu.global_id  y
        //CHECK: %[[R4:.*]] = arith.remui %[[R2]], %[[c8]] : index
        //CHECK: %[[R5:.*]] = arith.remui %[[R3]], %[[c4]] : index
        //CHECK: %[[R6:.*]] = arith.muli %[[R2:.*]], %[[c32]] : index
        //CHECK: %[[R7:.*]] = arith.muli %[[R3:.*]], %[[c64]] : index
        //CHECK: %[[R8:.*]] = arith.muli %[[R0:.*]], %[[c256]] : index
        //CHECK: %[[R9:.*]] = arith.muli %[[R1:.*]], %[[c256]] : index
        //CHECK: %[[R10:.*]] = arith.muli %[[R4]], %[[c4]] : index
        //CHECK: %[[R11:.*]] = arith.addi %[[R10]], %[[R5]] : index
        //CHECK: %[[R12:.*]] = arith.muli %[[R11]], %[[c8]] : index
        //CHECK: %[[R13:.*]] = arith.addi %[[R12]], %[[R8]] : index
        %wg_id_x = gpu.block_id x
        %wg_id_y = gpu.block_id y
        %global_sg_id_x = gpu.global_id x
        %global_sg_id_y = gpu.global_id y
        %local_sg_id_x = arith.remui %global_sg_id_x, %c8 : index
        %local_sg_id_y = arith.remui %global_sg_id_y, %c4 : index
        %C_sg_tile_offset_x = arith.muli %global_sg_id_x, %c32 : index
        %C_sg_tile_offset_y = arith.muli %global_sg_id_y, %c64 : index
        %wg_tile_offset_x = arith.muli %wg_id_x, %c256 : index
        %wg_tile_offset_y = arith.muli %wg_id_y, %c256 : index
        %local_sg_id_temp = arith.muli %local_sg_id_x, %c4 : index
        %local_sg_id = arith.addi %local_sg_id_temp, %local_sg_id_y : index
        %A_sg_prefetch_offset_x_temp = arith.muli %local_sg_id, %c8 : index
        %A_sg_prefetch_offset_x = arith.addi %A_sg_prefetch_offset_x_temp, %wg_tile_offset_x : index

        //CHECK: %[[R14:.*]] = xetile.init_tile %[[arg0]][%[[R13]], %[[c0]]] : memref<4096x4096xf16> -> !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
        //CHECK: xetile.prefetch_tile %[[R14]] : !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
        //CHECK: %[[R15:.*]] = xetile.update_tile_offset %[[R14]], [%[[c0]],  %[[c32]]] : !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>, index, index -> !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
        //CHECK: xetile.prefetch_tile %[[R15]] : !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
        //CHECK: %[[R16:.*]] = xetile.update_tile_offset %[[R15]], [%[[c0]],  %[[c32]]] : !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>, index, index -> !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
        //CHECK: xetile.prefetch_tile %[[R16]] : !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
        //CHECK: %[[R17:.*]] = xetile.update_tile_offset %[[R16]], [%[[c0]],  %[[c32]]] : !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>, index, index -> !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
        //CHECK: %[[R18:.*]] = arith.remui %[[R4]], %[[c4]] : index
        //CHECK: %[[R19:.*]] = arith.muli %[[R18]], %[[c8]] : index
        //CHECK: %[[R20:.*]] = arith.muli %[[R5]], %[[c64]] : index
        //CHECK: %[[R21:.*]] = arith.divui %[[R4]], %[[c4]] : index
        //CHECK: %[[R22:.*]] = arith.muli %[[R21]], %[[c32]] : index
        //CHECK: %[[R23:.*]] = arith.addi %[[R20]], %[[R22]] : index
        //CHECK: %[[R24:.*]] = arith.addi %[[R9]], %[[R23]] : index


        %A_sg_prefetch_tile_iter0 = xetile.init_tile %A[%A_sg_prefetch_offset_x, %c0] : memref<4096x4096xf16> -> !xetile.tile<8x32xf16>
        xetile.prefetch_tile %A_sg_prefetch_tile_iter0 : !xetile.tile<8x32xf16>
        %A_sg_prefetch_tile_iter1 = xetile.update_tile_offset %A_sg_prefetch_tile_iter0, [%c0, %c32] : !xetile.tile<8x32xf16>, index, index -> !xetile.tile<8x32xf16>
        xetile.prefetch_tile %A_sg_prefetch_tile_iter1 : !xetile.tile<8x32xf16>
        %A_sg_prefetch_tile_iter2 = xetile.update_tile_offset %A_sg_prefetch_tile_iter1, [%c0, %c32] : !xetile.tile<8x32xf16>, index, index -> !xetile.tile<8x32xf16>
        xetile.prefetch_tile %A_sg_prefetch_tile_iter2 : !xetile.tile<8x32xf16>
        %A_sg_prefetch_tile_iter3 = xetile.update_tile_offset %A_sg_prefetch_tile_iter2, [%c0, %c32] : !xetile.tile<8x32xf16>, index, index -> !xetile.tile<8x32xf16>
        %B_sg_prefetch_offset_x_temp0 = arith.remui %local_sg_id_x, %c4 : index
        %B_sg_prefetch_offset_x = arith.muli %B_sg_prefetch_offset_x_temp0, %c8 : index
        %B_sg_prefetch_offset_y_temp0 = arith.muli %local_sg_id_y, %c64 : index
        %B_sg_prefetch_offset_y_temp1 = arith.divui %local_sg_id_x, %c4 : index
        %B_sg_prefetch_offset_y_temp2 = arith.muli %B_sg_prefetch_offset_y_temp1, %c32 : index
        %B_sg_prefetch_offset_y_temp3 = arith.addi %B_sg_prefetch_offset_y_temp0, %B_sg_prefetch_offset_y_temp2 : index
        %B_sg_prefetch_offset_y = arith.addi %wg_tile_offset_y, %B_sg_prefetch_offset_y_temp3 : index

        //CHECK: %[[R25:.*]] = xetile.init_tile %[[arg1]][%[[R19]], %[[R24]]] : memref<4096x4096xf16> -> !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
        //CHECK: xetile.prefetch_tile %[[R25]] : !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
        //CHECK: %[[R26:.*]] = xetile.update_tile_offset %[[R25]], [%[[c32]],  %[[c0]]] : !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>, index, index -> !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
        //CHECK: xetile.prefetch_tile %[[R26]] : !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
        //CHECK: %[[R27:.*]] = xetile.update_tile_offset %[[R26]], [%[[c32]],  %[[c0]]] : !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>, index, index -> !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
        //CHECK: xetile.prefetch_tile %[[R27]] : !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
        //CHECK: %[[R28:.*]] = xetile.update_tile_offset %[[R27]], [%[[c32]],  %[[c0]]] : !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>, index, index -> !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
        //CHECK: %[[R29:.*]] = xetile.init_tile %[[arg0]][%[[R6]], %[[c0]]] : memref<4096x4096xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
        //CHECK: %[[R30:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[R7]]] : memref<4096x4096xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
        //CHECK: xegpu.alloc_nbarrier 16
        %B_sg_prefetch_tile_iter0 = xetile.init_tile %B[%B_sg_prefetch_offset_x, %B_sg_prefetch_offset_y]  : memref<4096x4096xf16> -> !xetile.tile<8x32xf16>
        xetile.prefetch_tile %B_sg_prefetch_tile_iter0 : !xetile.tile<8x32xf16>
        %B_sg_prefetch_tile_iter1 = xetile.update_tile_offset %B_sg_prefetch_tile_iter0, [%c32, %c0] : !xetile.tile<8x32xf16>, index, index -> !xetile.tile<8x32xf16>
        xetile.prefetch_tile %B_sg_prefetch_tile_iter1 : !xetile.tile<8x32xf16>
        %B_sg_prefetch_tile_iter2 = xetile.update_tile_offset %B_sg_prefetch_tile_iter1, [%c32, %c0] : !xetile.tile<8x32xf16>, index, index -> !xetile.tile<8x32xf16>
        xetile.prefetch_tile %B_sg_prefetch_tile_iter2 : !xetile.tile<8x32xf16>
        %B_sg_prefetch_tile_iter3 = xetile.update_tile_offset %B_sg_prefetch_tile_iter2, [%c32, %c0] : !xetile.tile<8x32xf16>, index, index -> !xetile.tile<8x32xf16>
        %A_sg_init_tile = xetile.init_tile %A[%C_sg_tile_offset_x, %c0] : memref<4096x4096xf16> -> !xetile.tile<32x32xf16>
        %B_sg_init_tile = xetile.init_tile %B[%c0, %C_sg_tile_offset_y] : memref<4096x4096xf16> -> !xetile.tile<32x64xf16>
        %c_init_val = arith.constant dense<0.0> : vector<32x64xf32>
        xegpu.alloc_nbarrier 16

        %nbarrier_id = arith.constant 1 : i8
        %nbarrier_role = arith.constant 0 : i8
        //CHECK: %[[R31:.*]] = xegpu.init_nbarrier %[[c1_i8]], %[[c0_i8]] : i8, i8 -> !xegpu.nbarrier
        %nbarrier = xegpu.init_nbarrier %nbarrier_id, %nbarrier_role : i8, i8 -> !xegpu.nbarrier
        //CHECK: %[[R32:.*]]:5 = scf.for %[[arg4:.*]] = %[[c0]] to %[[c4096]] step %[[c32]] iter_args(%[[arg5:.*]] = %[[R29]], %[[arg6:.*]] = %[[R30]], %[[arg7:.*]] = %[[cst]], %[[arg8:.*]] = %[[R17]], %[[arg9:.*]] = %[[R28]]) -> (!xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, vector<4x4x8x16xf32>, !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>, !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>) {
        %k_loop_result:5 = scf.for %k = %c0 to %c4096 step %c32 iter_args (
            %A_tile = %A_sg_init_tile,
            %B_tile = %B_sg_init_tile,
            %c_val = %c_init_val,

            %A_prefetch_tile = %A_sg_prefetch_tile_iter3,
            %B_prefetch_tile = %B_sg_prefetch_tile_iter3
            ) ->
            (!xetile.tile<32x32xf16>, !xetile.tile<32x64xf16>,
            vector<32x64xf32>,
            !xetile.tile<8x32xf16>, !xetile.tile<8x32xf16>
            )
            {
          //CHECK: %[[R36:.*]] = arith.remui %[[arg4]], %[[c256]] : index
          //CHECK: %[[R37:.*]] = arith.index_cast %[[R36]] : index to i32
          //CHECK: %[[R38:.*]] = arith.cmpi eq, %[[R37]], %[[c0_i32]] : i32
          %every_8th_iter = arith.remui %k, %c256 : index
          %every_8th_iter_i32 = arith.index_cast %every_8th_iter : index to i32
          %every_8th_iter_cond = arith.cmpi eq, %every_8th_iter_i32, %c0_i32 : i32
          //CHECK: scf.if %[[R38]]
          scf.if %every_8th_iter_cond  {
            //CHECK: xegpu.nbarrier_arrive %[[R31]] : !xegpu.nbarrier
            xegpu.nbarrier_arrive %nbarrier : !xegpu.nbarrier
          }
          //CHECK: %[[R39:.*]] = xetile.load_tile %[[arg5]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<1x2x32x16xf16>
          //CHECK: %[[R40:.*]] = xetile.tile_unpack %[[R39]] {inner_blocks = array<i64: 32, 16>}  : vector<1x2x32x16xf16> -> vector<32x32xf16>
          %a_val = xetile.load_tile %A_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>
          //CHECK: %[[R41:.*]] = xetile.load_tile %[[arg6]] : !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<1x4x32x16xf16>
          //CHECK: %[[R42:.*]] = xetile.tile_unpack %[[R41]] {inner_blocks = array<i64: 32, 16>}  : vector<1x4x32x16xf16> -> vector<32x64xf16>
          %b_val = xetile.load_tile %B_tile  : !xetile.tile<32x64xf16> -> vector<32x64xf16>
          //CHECK: xegpu.compile_hint
          xegpu.compile_hint
          //CHECK: xetile.prefetch_tile %[[arg8]] : !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
          //CHECK: xetile.prefetch_tile %[[arg9]] : !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
          xetile.prefetch_tile %A_prefetch_tile : !xetile.tile<8x32xf16>
          xetile.prefetch_tile %B_prefetch_tile : !xetile.tile<8x32xf16>
          //CHECK: xegpu.compile_hint
          xegpu.compile_hint
          //CHECK: %[[R43:.*]] = xetile.update_tile_offset %[[arg8]], [%[[c0]],  %[[c32]]] : !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>, index, index -> !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
          //CHECK: %[[R44:.*]] = xetile.update_tile_offset %[[arg9]], [%[[c32]],  %[[c0]]] : !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>, index, index -> !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
          //CHECK: %[[R45:.*]] = xetile.update_tile_offset %[[arg5]], [%[[c0]],  %[[c32]]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, index, index -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
          //CHECK: %[[R46:.*]] = xetile.update_tile_offset %[[arg6]], [%[[c32]],  %[[c0]]] : !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, index, index -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
          %next_A_prefetch_tile = xetile.update_tile_offset %A_prefetch_tile, [%c0, %c32] : !xetile.tile<8x32xf16>, index, index -> !xetile.tile<8x32xf16>
          %next_B_prefetch_tile = xetile.update_tile_offset %B_prefetch_tile, [%c32, %c0] : !xetile.tile<8x32xf16>, index, index -> !xetile.tile<8x32xf16>
          %next_A_tile = xetile.update_tile_offset %A_tile, [%c0, %c32]  : !xetile.tile<32x32xf16>, index, index -> !xetile.tile<32x32xf16>
          %next_B_tile = xetile.update_tile_offset %B_tile, [%c32, %c0]  : !xetile.tile<32x64xf16>, index, index -> !xetile.tile<32x64xf16>
          //CHECK: xegpu.compile_hint
          xegpu.compile_hint
          //CHECK: %[[R47:.*]] = xetile.tile_pack %[[R40]] {inner_blocks = array<i64: 8, 16>}  : vector<32x32xf16> -> vector<4x2x8x16xf16>
          //CHECK: %[[R48:.*]] = xetile.tile_pack %[[R42]] {inner_blocks = array<i64: 16, 16>}  : vector<32x64xf16> -> vector<2x4x16x16xf16>
          //CHECK: %[[R51:.*]] = xetile.tile_mma %[[R47]], %[[R48]], %[[arg7]] : vector<4x2x8x16xf16>, vector<2x4x16x16xf16>, vector<4x4x8x16xf32> -> vector<4x4x8x16xf32>
          %new_c_val = xetile.tile_mma %a_val, %b_val, %c_val : vector<32x32xf16>, vector<32x64xf16>, vector<32x64xf32> -> vector<32x64xf32>
          //CHECK: xegpu.compile_hint
          xegpu.compile_hint
          //CHECK: scf.if %[[R38]]
          scf.if %every_8th_iter_cond {
            //CHECK: xegpu.nbarrier_wait %[[R31]] : !xegpu.nbarrier
            xegpu.nbarrier_wait %nbarrier : !xegpu.nbarrier
          }

          //CHECK: scf.yield %[[R45]], %[[R46]], %[[R51]], %[[R43]], %[[R44]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, vector<4x4x8x16xf32>, !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>, !xetile.tile<8x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
          scf.yield %next_A_tile, %next_B_tile, %new_c_val,
                    %next_A_prefetch_tile, %next_B_prefetch_tile
                    : !xetile.tile<32x32xf16>, !xetile.tile<32x64xf16>, vector<32x64xf32>,
                    !xetile.tile<8x32xf16>, !xetile.tile<8x32xf16>
        }
        //CHECK: %[[R34:.*]] = xetile.init_tile %[[arg2]][%[[R6]], %[[R7]]] : memref<4096x4096xf32> -> !xetile.tile<32x64xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
        %c_sg_tile = xetile.init_tile %C[%C_sg_tile_offset_x, %C_sg_tile_offset_y] : memref<4096x4096xf32> -> !xetile.tile<32x64xf32>
        //CHECK: xetile.store_tile %[[R32]]#2,  %[[R34]] : vector<4x4x8x16xf32>, !xetile.tile<32x64xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
        xetile.store_tile %k_loop_result#2 , %c_sg_tile : vector<32x64xf32>, !xetile.tile<32x64xf32>
      }
      //CHECK: gpu.return
      gpu.return
    }
  }
}
