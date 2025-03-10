// RUN:  imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking --canonicalize %s -verify-diagnostics -o -| FileCheck %s
module @gemm attributes {gpu.container_module} {
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {

    //CHECK: gpu.func @test_kernel(%[[arg0:.*]]: memref<4096x4096xf16>, %[[arg1:.*]]: memref<4096x4096xf16>, %[[arg2:.*]]: memref<4096x4096xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_kernel(%A: memref<4096x4096xf16>, %B: memref<4096x4096xf16>, %C: memref<4096x4096xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
      //CHECK: %[[c24:.*]] = arith.constant 24 : index
      //CHECK: %[[c48:.*]] = arith.constant 48 : index
      //CHECK: %[[c16:.*]] = arith.constant 16 : index
      //CHECK: %[[c0_i8:.*]] = arith.constant 0 : i8
      //CHECK: %[[c1_i8:.*]] = arith.constant 1 : i8
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
        //CHECK: %[[block_id_x:.*]] = gpu.block_id  x
        //CHECK: %[[block_id_y:.*]] = gpu.block_id  y
        //CHECK: %[[global_id_x:.*]] = gpu.global_id  x
        //CHECK: %[[global_id_y:.*]] = gpu.global_id  y
        //CHECK: %[[r0:.*]] = arith.remui %[[global_id_x]], %[[c8]] : index
        //CHECK: %[[r1:.*]] = arith.remui %[[global_id_y]], %[[c4]] : index
        //CHECK: %[[r2:.*]] = arith.muli %[[global_id_x]], %[[c32]] : index
        //CHECK: %[[r3:.*]] = arith.muli %[[global_id_y]], %[[c64]] : index
        //CHECK: %[[r4:.*]] = arith.muli %[[block_id_x]], %[[c256]] : index
        //CHECK: %[[r5:.*]] = arith.muli %[[block_id_y]], %[[c256]] : index
        //CHECK: %[[r6:.*]] = arith.muli %[[r0]], %[[c4]] : index
        //CHECK: %[[r7:.*]] = arith.addi %[[r6]], %[[r1]] : index
        //CHECK: %[[r8:.*]] = arith.muli %[[r7]], %[[c8]] : index
        //CHECK: %[[r9:.*]] = arith.addi %[[r8]], %[[r4]] : index

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

        //CHECK: %[[r10:.*]] = xetile.init_tile %[[arg0]][%[[r9]], %[[c0]]] : memref<4096x4096xf16> -> !xetile.tile<8x32xf16>
        //CHECK: xetile.prefetch_tile %[[r10]] : !xetile.tile<8x32xf16>
        //CHECK: %[[r11:.*]] = xetile.update_tile_offset %[[r10]], [%[[c0]], %[[c32]]] : !xetile.tile<8x32xf16>
        //CHECK: xetile.prefetch_tile %[[r11]] : !xetile.tile<8x32xf16>
        //CHECK: %[[r12:.*]] = xetile.update_tile_offset %[[r11]], [%[[c0]], %[[c32]]] : !xetile.tile<8x32xf16>
        //CHECK: xetile.prefetch_tile %[[r12]] : !xetile.tile<8x32xf16>
        //CHECK: %[[r13:.*]] = xetile.update_tile_offset %[[r12]], [%[[c0]], %[[c32]]] : !xetile.tile<8x32xf16>
        //CHECK: %[[r14:.*]] = arith.remui %[[r0]], %[[c4]] : index
        //CHECK: %[[r15:.*]] = arith.muli %[[r14]], %[[c8]] : index
        //CHECK: %[[r16:.*]] = arith.muli %[[r1]], %[[c64]] : index
        //CHECK: %[[r17:.*]] = arith.divui %[[r0]], %[[c4]] : index
        //CHECK: %[[r18:.*]] = arith.muli %[[r17]], %[[c32]] : index
        //CHECK: %[[r19:.*]] = arith.addi %[[r16]], %[[r18]] : index
        //CHECK: %[[r20:.*]] = arith.addi %[[r5]], %[[r19]] : index


        %A_sg_prefetch_tile_iter0 = xetile.init_tile %A[%A_sg_prefetch_offset_x, %c0] : memref<4096x4096xf16> -> !xetile.tile<8x32xf16>
        xetile.prefetch_tile %A_sg_prefetch_tile_iter0 : !xetile.tile<8x32xf16>
        %A_sg_prefetch_tile_iter1 = xetile.update_tile_offset %A_sg_prefetch_tile_iter0, [%c0, %c32] : !xetile.tile<8x32xf16>
        xetile.prefetch_tile %A_sg_prefetch_tile_iter1 : !xetile.tile<8x32xf16>
        %A_sg_prefetch_tile_iter2 = xetile.update_tile_offset %A_sg_prefetch_tile_iter1, [%c0, %c32] : !xetile.tile<8x32xf16>
        xetile.prefetch_tile %A_sg_prefetch_tile_iter2 : !xetile.tile<8x32xf16>
        %A_sg_prefetch_tile_iter3 = xetile.update_tile_offset %A_sg_prefetch_tile_iter2, [%c0, %c32] : !xetile.tile<8x32xf16>
        %B_sg_prefetch_offset_x_temp0 = arith.remui %local_sg_id_x, %c4 : index
        %B_sg_prefetch_offset_x = arith.muli %B_sg_prefetch_offset_x_temp0, %c8 : index
        %B_sg_prefetch_offset_y_temp0 = arith.muli %local_sg_id_y, %c64 : index
        %B_sg_prefetch_offset_y_temp1 = arith.divui %local_sg_id_x, %c4 : index
        %B_sg_prefetch_offset_y_temp2 = arith.muli %B_sg_prefetch_offset_y_temp1, %c32 : index
        %B_sg_prefetch_offset_y_temp3 = arith.addi %B_sg_prefetch_offset_y_temp0, %B_sg_prefetch_offset_y_temp2 : index
        %B_sg_prefetch_offset_y = arith.addi %wg_tile_offset_y, %B_sg_prefetch_offset_y_temp3 : index

        //CHECK: %[[r21:.*]] = xetile.init_tile %[[arg1]][%[[r15]], %[[r20]]] : memref<4096x4096xf16> -> !xetile.tile<8x32xf16>
        //CHECK: xetile.prefetch_tile %[[r21]] : !xetile.tile<8x32xf16>
        //CHECK: %[[r22:.*]] = xetile.update_tile_offset %[[r21]], [%[[c32]], %[[c0]]] : !xetile.tile<8x32xf16>
        //CHECK: xetile.prefetch_tile %[[r22]] : !xetile.tile<8x32xf16>
        //CHECK: %[[r23:.*]] = xetile.update_tile_offset %[[r22]], [%[[c32]], %[[c0]]] : !xetile.tile<8x32xf16>
        //CHECK: xetile.prefetch_tile %[[r23]] : !xetile.tile<8x32xf16>
        //CHECK: %[[r24:.*]] = xetile.update_tile_offset %[[r23]], [%[[c32]], %[[c0]]] : !xetile.tile<8x32xf16>
        //CHECK: %[[r25:.*]] = xetile.init_tile %[[arg0]][%[[r2]], %[[c0]]] : memref<4096x4096xf16> -> !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
        //CHECK: %[[r26:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[r3]]] : memref<4096x4096xf16> -> !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
        //CHECK: %[[r27:.*]] = arith.addi %[[r3]], %[[c32]] : index
        //CHECK: %[[r28:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[r27]]] : memref<4096x4096xf16> -> !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
        //CHECK: xegpu.alloc_nbarrier 16

        %B_sg_prefetch_tile_iter0 = xetile.init_tile %B[%B_sg_prefetch_offset_x, %B_sg_prefetch_offset_y]  : memref<4096x4096xf16> -> !xetile.tile<8x32xf16>
        xetile.prefetch_tile %B_sg_prefetch_tile_iter0 : !xetile.tile<8x32xf16>
        %B_sg_prefetch_tile_iter1 = xetile.update_tile_offset %B_sg_prefetch_tile_iter0, [%c32, %c0] : !xetile.tile<8x32xf16>
        xetile.prefetch_tile %B_sg_prefetch_tile_iter1 : !xetile.tile<8x32xf16>
        %B_sg_prefetch_tile_iter2 = xetile.update_tile_offset %B_sg_prefetch_tile_iter1, [%c32, %c0] : !xetile.tile<8x32xf16>
        xetile.prefetch_tile %B_sg_prefetch_tile_iter2 : !xetile.tile<8x32xf16>
        %B_sg_prefetch_tile_iter3 = xetile.update_tile_offset %B_sg_prefetch_tile_iter2, [%c32, %c0] : !xetile.tile<8x32xf16>
        %A_sg_init_tile = xetile.init_tile %A[%C_sg_tile_offset_x, %c0] : memref<4096x4096xf16> -> !xetile.tile<32x32xf16>
        %B_sg_init_tile = xetile.init_tile %B[%c0, %C_sg_tile_offset_y] : memref<4096x4096xf16> -> !xetile.tile<32x64xf16>
        %c_init_val = arith.constant dense<0.0> : vector<32x64xf32>
        xegpu.alloc_nbarrier 16

        %nbarrier_id = arith.constant 1 : i8
        %nbarrier_role = arith.constant 0 : i8
        //CHECK: %[[r29:.*]] = xegpu.init_nbarrier %[[c1_i8]], %[[c0_i8]] : i8, i8 -> !xegpu.nbarrier
        %nbarrier = xegpu.init_nbarrier %nbarrier_id, %nbarrier_role : i8, i8 -> !xegpu.nbarrier

        //CHECK: %[[r30:.*]]:21 = scf.for %[[arg4:.*]] = %[[c0]] to %[[c4096]] step %[[c32]] iter_args(%[[arg5:.*]] = %[[r25]], %[[arg6:.*]] = %[[r26]], %[[arg7:.*]] = %[[r28]], %[[arg8:.*]] = %[[cst]], %[[arg9:.*]] = %cst, %[[arg10:.*]] = %cst, %[[arg11:.*]] = %cst, %[[arg12:.*]] = %cst, %[[arg13:.*]] = %cst, %[[arg14:.*]] = %cst, %[[arg15:.*]] = %cst, %[[arg16:.*]] = %cst, %[[arg17:.*]] = %cst, %[[arg18:.*]] = %cst, %[[arg19:.*]] = %cst, %[[arg20:.*]] = %cst, %[[arg21:.*]] = %cst, %[[arg22:.*]] = %cst, %[[arg23:.*]] = %cst, %[[arg24:.*]] = %[[r13]], %[[arg25:.*]] = %[[r24]]) -> (!xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>, !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>, !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, !xetile.tile<8x32xf16>, !xetile.tile<8x32xf16>) {
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
          //CHECK: %[[r71:.*]] = arith.remui %[[arg4]], %[[c256]] : index
          //CHECK: %[[r72:.*]] = arith.index_cast %[[r71]] : index to i32
          //CHECK: %[[r73:.*]] = arith.cmpi eq, %[[r72]], %[[c0_i32]] : i32
          %every_8th_iter = arith.remui %k, %c256 : index
          %every_8th_iter_i32 = arith.index_cast %every_8th_iter : index to i32
          %every_8th_iter_cond = arith.cmpi eq, %every_8th_iter_i32, %c0_i32 : i32
          //CHECK: scf.if %[[r73]]
          scf.if %every_8th_iter_cond  {
            //CHECK: xegpu.nbarrier_arrive %[[r29]] : !xegpu.nbarrier
            xegpu.nbarrier_arrive %nbarrier : !xegpu.nbarrier
          }

          //CHECK: %[[r74:.*]]:2 = xetile.load_tile %arg5 : !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>> -> vector<32x16xf16>, vector<32x16xf16>
          %a_val = xetile.load_tile %A_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>

          //CHECK: %[[r75:.*]]:2 = xetile.load_tile %arg6 : !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>> -> vector<32x16xf16>, vector<32x16xf16>
          //CHECK: %[[r76:.*]]:2 = xetile.load_tile %arg7 : !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>> -> vector<32x16xf16>, vector<32x16xf16>
          %b_val = xetile.load_tile %B_tile  : !xetile.tile<32x64xf16> -> vector<32x64xf16>

          //CHECK: xegpu.compile_hint
          xegpu.compile_hint
          //CHECK-COUNT-2: xetile.prefetch_tile %{{.*}} : !xetile.tile<8x32xf16>
          xetile.prefetch_tile %A_prefetch_tile : !xetile.tile<8x32xf16>
          xetile.prefetch_tile %B_prefetch_tile : !xetile.tile<8x32xf16>
          //CHECK: xegpu.compile_hint
          xegpu.compile_hint
          //CHECK-COUNT-2: %{{.*}} = xetile.update_tile_offset %{{.*}}, [%{{.*}},  %{{.*}}] : !xetile.tile<8x32xf16>
          %next_A_prefetch_tile = xetile.update_tile_offset %A_prefetch_tile, [%c0, %c32] : !xetile.tile<8x32xf16>
          //CHECK: %[[r79:.*]] = xetile.update_tile_offset %arg5, [%[[c0]], %[[c32]]] : !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
          %next_B_prefetch_tile = xetile.update_tile_offset %B_prefetch_tile, [%c32, %c0] : !xetile.tile<8x32xf16>
          %next_A_tile = xetile.update_tile_offset %A_tile, [%c0, %c32]  :  !xetile.tile<32x32xf16>
          //CHECK-COUNT-2: %{{.*}} = xetile.update_tile_offset %{{.*}}, [%[[c32]], %[[c0]]] : !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
          %next_B_tile = xetile.update_tile_offset %B_tile, [%c32, %c0]  : !xetile.tile<32x64xf16>
          //CHECK: xegpu.compile_hint
          xegpu.compile_hint
          //CHECK-COUNT-8: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
          //CHECK-COUNT-8: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
          //CHECK-COUNT-32: %{{.*}} = xetile.tile_mma %{{.*}} : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          %new_c_val = xetile.tile_mma %a_val, %b_val, %c_val : vector<32x32xf16>, vector<32x64xf16>, vector<32x64xf32> -> vector<32x64xf32>
          //CHECK: xegpu.compile_hint
          xegpu.compile_hint
          //CHECK: scf.if %[[r73]]
          scf.if %every_8th_iter_cond {
            //CHECK: xegpu.nbarrier_wait %[[r29]] : !xegpu.nbarrier
            xegpu.nbarrier_wait %nbarrier : !xegpu.nbarrier
          }

          //CHECK: scf.yield %{{.*}} : !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>, !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>, !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, !xetile.tile<8x32xf16>, !xetile.tile<8x32xf16>
          scf.yield %next_A_tile, %next_B_tile, %new_c_val,
                    %next_A_prefetch_tile, %next_B_prefetch_tile
                    : !xetile.tile<32x32xf16>, !xetile.tile<32x64xf16>, vector<32x64xf32>,
                    !xetile.tile<8x32xf16>, !xetile.tile<8x32xf16>
        }
        //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
        //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
        //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
        //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
        //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
        //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
        //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
        //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
        //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
        //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
        //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
        //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
        //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
        //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
        //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
        //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
        %c_sg_tile = xetile.init_tile %C[%C_sg_tile_offset_x, %C_sg_tile_offset_y] : memref<4096x4096xf32> -> !xetile.tile<32x64xf32>
        //CHECK-COUNT-16: xetile.store_tile %{{.*}},  %{{.*}} : vector<8x16xf32>, !xetile.tile<8x16xf32>
        xetile.store_tile %k_loop_result#2 , %c_sg_tile : vector<32x64xf32>, !xetile.tile<32x64xf32>
      }
      gpu.return
    }
  }
}
