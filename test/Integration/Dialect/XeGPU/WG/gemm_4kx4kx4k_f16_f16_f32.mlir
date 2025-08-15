// RUN: %python_executable %imex_runner --requires=mlir-sycl-runtime,spirv-backend -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_sycl_runtime --filecheck
#a = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [8, 16]>
#b = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [16, 16]>
#c = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>
#a_prefetch = #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 32], inst_data = [8, 16]>
#b_prefetch = #xegpu.layout<sg_layout = [4, 8], sg_data = [8, 32], inst_data = [8, 16]>
module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<4096x4096xf16>, %B: memref<4096x4096xf16>, %C: memref<4096x4096xf32>) -> memref<4096x4096xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %t0 = gpu.wait async
    %A_gpu, %t1 = gpu.alloc async [%t0] () : memref<4096x4096xf16>
    %t2 = gpu.memcpy async [%t1] %A_gpu, %A : memref<4096x4096xf16>, memref<4096x4096xf16>
    %B_gpu, %t3 = gpu.alloc  async [%t2] () : memref<4096x4096xf16>
    %t4 = gpu.memcpy async [%t3] %B_gpu, %B : memref<4096x4096xf16>, memref<4096x4096xf16>
    %C_gpu, %t5 = gpu.alloc async [%t4]  () : memref<4096x4096xf32>
    %t6 = gpu.memcpy async [%t5] %C_gpu, %C : memref<4096x4096xf32>, memref<4096x4096xf32>
    // NOTE: Here we can't use [8, 64] wi threads following the SG thread layout of [8, 4]. Because runtime will linearize the x dimension first (we need y dimension to be linearized first).
    // So just use linearized thread layout of [512, 1] wi threads.
    %t7 = gpu.launch_func async [%t6]  @test_kernel::@test_kernel blocks in (%c16, %c16, %c1) threads in (%c512, %c1, %c1) args(%A_gpu : memref<4096x4096xf16>, %B_gpu : memref<4096x4096xf16>, %C_gpu : memref<4096x4096xf32>)
    gpu.wait [%t7] // Wait for the kernel to finish.
    %t12 = gpu.wait async
    %t8 = gpu.memcpy async [%t12] %C, %C_gpu : memref<4096x4096xf32>, memref<4096x4096xf32>
    %t9 = gpu.dealloc async [%t8]  %A_gpu : memref<4096x4096xf16>
    %t10 = gpu.dealloc async [%t9] %B_gpu : memref<4096x4096xf16>
    %t11 = gpu.dealloc async [%t10] %C_gpu : memref<4096x4096xf32>
    gpu.wait [%t11]
    return %C : memref<4096x4096xf32>
  }

  gpu.module @test_kernel   {
    gpu.func @test_kernel(%A: memref<4096x4096xf16>, %B: memref<4096x4096xf16>, %C: memref<4096x4096xf32>) kernel  {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c4096 = arith.constant 4096 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %m = arith.muli %block_id_x, %c256 : index
      %n = arith.muli %block_id_y, %c256 : index
      %c_tdesc = xegpu.create_nd_tdesc %C[%m, %n] : memref<4096x4096xf32> -> !xegpu.tensor_desc<256x256xf32, #c>
      %c_init_value = xegpu.load_nd %c_tdesc : !xegpu.tensor_desc<256x256xf32, #c> -> vector<256x256xf32>
      %a_tdesc = xegpu.create_nd_tdesc %A[%m, %c0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16, #a>
      %b_tdesc = xegpu.create_nd_tdesc %B[%c0, %n] : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16, #b>
      // Prefetch A 3 times.
      %a_prefetch_tdesc = xegpu.create_nd_tdesc %A[%m, %c0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16, #a_prefetch>
      xegpu.prefetch_nd %a_prefetch_tdesc : !xegpu.tensor_desc<256x32xf16, #a_prefetch>
      %a_prefetch_tdesc_2 = xegpu.update_nd_offset %a_prefetch_tdesc, [%c0, %c32] : !xegpu.tensor_desc<256x32xf16, #a_prefetch>
      xegpu.prefetch_nd %a_prefetch_tdesc_2 : !xegpu.tensor_desc<256x32xf16, #a_prefetch>
      %a_prefetch_tdesc_3 = xegpu.update_nd_offset %a_prefetch_tdesc_2, [%c0, %c32] : !xegpu.tensor_desc<256x32xf16, #a_prefetch>
      xegpu.prefetch_nd %a_prefetch_tdesc_3 : !xegpu.tensor_desc<256x32xf16, #a_prefetch>
      %a_prefetch_tdesc_4 = xegpu.update_nd_offset %a_prefetch_tdesc_3, [%c0, %c32] : !xegpu.tensor_desc<256x32xf16, #a_prefetch>
      // Prefetch B 3 times.
      %b_prefetch_tdesc = xegpu.create_nd_tdesc %B[%c0, %n] : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16, #b_prefetch>
      xegpu.prefetch_nd %b_prefetch_tdesc : !xegpu.tensor_desc<32x256xf16, #b_prefetch>
      %b_prefetch_tdesc_2 = xegpu.update_nd_offset %b_prefetch_tdesc, [%c32, %c0] : !xegpu.tensor_desc<32x256xf16, #b_prefetch>
      xegpu.prefetch_nd %b_prefetch_tdesc_2 : !xegpu.tensor_desc<32x256xf16, #b_prefetch>
      %b_prefetch_tdesc_3 = xegpu.update_nd_offset %b_prefetch_tdesc_2, [%c32, %c0] : !xegpu.tensor_desc<32x256xf16, #b_prefetch>
      xegpu.prefetch_nd %b_prefetch_tdesc_3 : !xegpu.tensor_desc<32x256xf16, #b_prefetch>
      %b_prefetch_tdesc_4 = xegpu.update_nd_offset %b_prefetch_tdesc_3, [%c32, %c0] : !xegpu.tensor_desc<32x256xf16, #b_prefetch>

      %out:5 = scf.for %k = %c0 to %c4096 step %c32
        iter_args(%a_tile = %a_tdesc, %b_tile = %b_tdesc,
                  %c_value = %c_init_value,
                  %a_prefetch_tile = %a_prefetch_tdesc_4,
                  %b_prefetch_tile = %b_prefetch_tdesc_4)
        -> (!xegpu.tensor_desc<256x32xf16, #a>, !xegpu.tensor_desc<32x256xf16, #b>, vector<256x256xf32>,
              !xegpu.tensor_desc<256x32xf16, #a_prefetch>, !xegpu.tensor_desc<32x256xf16, #b_prefetch>) {
        %a_value = xegpu.load_nd %a_tile  : !xegpu.tensor_desc<256x32xf16, #a> -> vector<256x32xf16>
        %b_value = xegpu.load_nd %b_tile : !xegpu.tensor_desc<32x256xf16, #b> -> vector<32x256xf16>
        // Prefetch next tiles.
        xegpu.prefetch_nd %a_prefetch_tile : !xegpu.tensor_desc<256x32xf16, #a_prefetch>
        xegpu.prefetch_nd %b_prefetch_tile : !xegpu.tensor_desc<32x256xf16, #b_prefetch>
        // Update offsets for next tiles.
        %a_next_tile = xegpu.update_nd_offset %a_tile, [%c0, %c32] : !xegpu.tensor_desc<256x32xf16, #a>
        %b_next_tile = xegpu.update_nd_offset %b_tile, [%c32, %c0] : !xegpu.tensor_desc<32x256xf16, #b>
        %a_prefetch_next_tile = xegpu.update_nd_offset %a_prefetch_tile, [%c0, %c32] : !xegpu.tensor_desc<256x32xf16, #a_prefetch>
        %b_prefetch_next_tile = xegpu.update_nd_offset %b_prefetch_tile, [%c32, %c0] : !xegpu.tensor_desc<32x256xf16, #b_prefetch>
        %c_new_value = xegpu.dpas %a_value, %b_value, %c_value {layout_result_0 = #c}
          : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf32> -> vector<256x256xf32>
        scf.yield %a_next_tile, %b_next_tile, %c_new_value, %a_prefetch_next_tile, %b_prefetch_next_tile :
          !xegpu.tensor_desc<256x32xf16, #a>, !xegpu.tensor_desc<32x256xf16, #b>, vector<256x256xf32>, !xegpu.tensor_desc<256x32xf16, #a_prefetch>, !xegpu.tensor_desc<32x256xf16, #b_prefetch>
      }
      xegpu.store_nd %out#2, %c_tdesc : vector<256x256xf32>, !xegpu.tensor_desc<256x256xf32, #c>
      gpu.return
    }
  }

  // compute CPU reference (takes minutes)
  func.func @cpu_reference(%A : memref<4096x4096xf16>, %B : memref<4096x4096xf16>, %C : memref<4096x4096xf32>) {
    %c4096 = arith.constant 4096 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    scf.for %i = %c0 to %c4096 step %c1 {
      scf.for %j = %c0 to %c4096 step %c1 {
        %c_curr = memref.load %C[%i, %j] : memref<4096x4096xf32>
        %c_val = scf.for %k_tile = %c0 to %c4096 step %c16 iter_args(%c_partial = %c_curr) -> f32 {
          %c_val_dpas = scf.for %k = %c0 to %c16 step %c1 iter_args(%c_dpas_partial = %c_partial) -> f32 {
            %k_dpas = arith.addi %k_tile, %k : index
            %a_val = memref.load %A[%i, %k_dpas] : memref<4096x4096xf16>
            %b_val = memref.load %B[%k_dpas, %j] : memref<4096x4096xf16>
            %a_cast = arith.extf %a_val : f16 to f32
            %b_cast = arith.extf %b_val : f16 to f32
            %t = arith.mulf %a_cast, %b_cast : f32
            // %t_cast = arith.extf %t : f16 to f16
            %c_sum = arith.addf %t, %c_dpas_partial : f32
            scf.yield %c_sum : f32
          }
          scf.yield %c_val_dpas : f32
        }
        %c_val_f16 = arith.truncf %c_val : f32 to f16
        %c_val_ = arith.extf %c_val_f16 : f16 to f32
        memref.store %c_val_ , %C[%i, %j] : memref<4096x4096xf32>
      }
    }
    return
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_f16 = arith.constant 1.0 : f16
    %c2_f16 = arith.constant 2.0 : f16
    %c4096 = arith.constant 4096 : index
    %cf_0 = arith.constant 0.0 : f16
    %cf_1 = arith.constant 1.0 : f16
    %A = memref.alloc() : memref<4096x4096xf16>
    %B = memref.alloc() : memref<4096x4096xf16>
    %C = memref.alloc() : memref<4096x4096xf32>
    %C_ref = memref.alloc() : memref<4096x4096xf32>
    %c_gen_int = arith.constant 0 : i1
    %cf_lower = arith.constant -0.5 : f32
    %cf_upper = arith.constant 0.5 : f32
    // Use one of the two options to initialize the A matrix
    // Option 1: intialize matrix A ; A[i, j] = j
    // scf.for %i = %c0 to %c4096 step %c1 {
    //   scf.for %j = %c0 to %c4096 step %c1 {
    //     %t = index.castu %j : index to i16
    //     %val = arith.uitofp %t : i16 to f16
    //     memref.store %val, %A[%i, %j] : memref<4096x4096xf16>
    //     // memref.store %c1_f16, %A[%i, %j] : memref<4096x4096xf16>
    //     // memref.store %c2_f16, %B[%i, %j] : memref<4096x4096xf16>
    //   }
    // }
    // Option 2:  convert the memref to 1D and fill with random values in (-0.5, 0.5)
    %A_random = memref.cast %A : memref<4096x4096xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%A_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf16>, f32, f32, i1) -> ()


    // Use one of the two options below to initialize the B matrix
    // Option 1: make matrix B an identity matrix
    // scf.for %i = %c0 to %c4096 step %c1 {
    //   scf.for %j = %c0 to %c4096 step %c1 {
    //     %i_i32 = index.castu %i : index to i32
    //     %j_i32 = index.castu %j : index to i32
    //     %i_j_same = arith.cmpi eq, %i_i32, %j_i32 : i32

    //     scf.if %i_j_same {
    //       memref.store %cf_1, %B[%i, %j] : memref<4096x4096xf16>
    //     } else {
    //       memref.store %cf_0, %B[%i, %j] : memref<4096x4096xf16>
    //     }
    //   }
    // }
    // Option 2:  convert the memref to 1D and fill with random values in (-0.5, 0.5)
    %B_random = memref.cast %B : memref<4096x4096xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%B_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf16>, f32, f32, i1) -> ()


    // intialize matrix C and C_ref ; C[i, j] = 0
    %c0_f32 = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c4096 step %c1 {
      scf.for %j = %c0 to %c4096 step %c1 {
        memref.store %c0_f32, %C[%i, %j] : memref<4096x4096xf32>
        memref.store %c0_f32, %C_ref[%i, %j] : memref<4096x4096xf32>
      }
    }
    // %A_row_0 = memref.subview %A[1, 0][1, 4096][1, 1] : memref<4096x4096xf16> to memref<1x4096xf16, strided<[4096, 1], offset: 4096>>
    // %A_row_0_cast = memref.cast %A_row_0 : memref<1x4096xf16, strided<[4096, 1], offset: 4096>> to memref<*xf16>
    // call @printMemrefF16(%A_row_0_cast) : (memref<*xf16>) -> ()

    // run GPU
    %2 = call @test(%A, %B, %C) : (memref<4096x4096xf16>, memref<4096x4096xf16>, memref<4096x4096xf32>) -> memref<4096x4096xf32>

    call @cpu_reference(%A, %B, %C_ref) : (memref<4096x4096xf16>, memref<4096x4096xf16>, memref<4096x4096xf32>) -> ()

    // %cast = memref.cast %A : memref<4096x4096xf16> to memref<*xf16>
    // call @printMemrefF16(%cast) : (memref<*xf16>) -> ()
    %cast_C = memref.cast %2 : memref<4096x4096xf32> to memref<*xf32>
    %cast_C_ref = memref.cast %C_ref : memref<4096x4096xf32> to memref<*xf32>
    // call @printMemrefF16(%cast_C) : (memref<*xf16>) -> ()
    // call @printMemrefF32(%cast_C_ref) : (memref<*xf32>) -> ()

    %C_row_0 = memref.subview %C_ref[0, 0][1, 4096][1, 1] : memref<4096x4096xf32> to memref<1x4096xf32, strided<[4096, 1], offset:0>>
    %C_row_0_cast = memref.cast %C_row_0 : memref<1x4096xf32, strided<[4096, 1], offset: 0>> to memref<*xf32>
    // call @printMemrefF32(%C_row_0_cast) : (memref<*xf32>) -> ()

    %C_row_0_gpu  = memref.subview %2[0, 0][1, 4096][1, 1] : memref<4096x4096xf32> to memref<1x4096xf32, strided<[4096, 1], offset:0>>
    %C_row_0_cast_gpu = memref.cast %C_row_0_gpu : memref<1x4096xf32, strided<[4096, 1], offset: 0>> to memref<*xf32>
    // call @printMemrefF32(%C_row_0_cast_gpu) : (memref<*xf32>) -> ()

    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast_C, %cast_C_ref) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %A : memref<4096x4096xf16>
    memref.dealloc %B : memref<4096x4096xf16>
    memref.dealloc %C : memref<4096x4096xf32>
    memref.dealloc %C_ref : memref<4096x4096xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMaxErrorF16(memref<*xf16>, memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}

}
