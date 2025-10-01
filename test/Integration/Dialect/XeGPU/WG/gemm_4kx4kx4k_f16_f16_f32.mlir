// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime,spirv-backend -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck
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
    %A_gpu = gpu.alloc () : memref<4096x4096xf16>
    gpu.memcpy %A_gpu, %A : memref<4096x4096xf16>, memref<4096x4096xf16>
    %B_gpu = gpu.alloc () : memref<4096x4096xf16>
    gpu.memcpy %B_gpu, %B : memref<4096x4096xf16>, memref<4096x4096xf16>
    %C_gpu = gpu.alloc () : memref<4096x4096xf32>
    gpu.memcpy %C_gpu, %C : memref<4096x4096xf32>, memref<4096x4096xf32>
    // NOTE: Here we can't use [8, 64] wi threads following the SG thread layout of [8, 4]. Because runtime will linearize the x dimension first (we need y dimension to be linearized first).
    // So just use linearized thread layout of [512, 1] wi threads.
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c16, %c16, %c1) threads in (%c512, %c1, %c1) args(%A_gpu : memref<4096x4096xf16>, %B_gpu : memref<4096x4096xf16>, %C_gpu : memref<4096x4096xf32>)
    gpu.wait // Wait for the kernel to finish.
    gpu.memcpy %C, %C_gpu : memref<4096x4096xf32>, memref<4096x4096xf32>
    gpu.dealloc %A_gpu : memref<4096x4096xf16>
    gpu.dealloc %B_gpu : memref<4096x4096xf16>
    gpu.dealloc %C_gpu : memref<4096x4096xf32>
    return %C : memref<4096x4096xf32>
  }

  gpu.module @test_kernel   {
    gpu.func @test_kernel(%A: memref<4096x4096xf16>, %B: memref<4096x4096xf16>, %C: memref<4096x4096xf32>) kernel  {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c96 = arith.constant 96 : index
      %c256 = arith.constant 256 : index
      %c4096 = arith.constant 4096 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %m = arith.muli %block_id_x, %c256 : index
      %n = arith.muli %block_id_y, %c256 : index
      %c_tdesc = xegpu.create_nd_tdesc %C : memref<4096x4096xf32> -> !xegpu.tensor_desc<256x256xf32, #c>
      %c_init_value = xegpu.load_nd %c_tdesc[%m, %n] : !xegpu.tensor_desc<256x256xf32, #c> -> vector<256x256xf32>
      %a_tdesc = xegpu.create_nd_tdesc %A : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16, #a>
      %b_tdesc = xegpu.create_nd_tdesc %B : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16, #b>
      // Prefetch A 3 times.
      %a_prefetch_tdesc = xegpu.create_nd_tdesc %A : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16, #a_prefetch>
      xegpu.prefetch_nd %a_prefetch_tdesc[%m, %c0] : !xegpu.tensor_desc<256x32xf16, #a_prefetch>
      xegpu.prefetch_nd %a_prefetch_tdesc[%m, %c32] : !xegpu.tensor_desc<256x32xf16, #a_prefetch>
      xegpu.prefetch_nd %a_prefetch_tdesc[%m, %c64] : !xegpu.tensor_desc<256x32xf16, #a_prefetch>
       // Prefetch B 3 times.
      %b_prefetch_tdesc = xegpu.create_nd_tdesc %B : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16, #b_prefetch>
      xegpu.prefetch_nd %b_prefetch_tdesc[%c0, %n] : !xegpu.tensor_desc<32x256xf16, #b_prefetch>
      xegpu.prefetch_nd %b_prefetch_tdesc[%c32, %n] : !xegpu.tensor_desc<32x256xf16, #b_prefetch>
      xegpu.prefetch_nd %b_prefetch_tdesc[%c64, %n] : !xegpu.tensor_desc<32x256xf16, #b_prefetch>

      %out = scf.for %k = %c0 to %c4096 step %c32
        iter_args(%c_value = %c_init_value)
        -> (vector<256x256xf32>) {
        %a_value = xegpu.load_nd %a_tdesc[%m, %k]  : !xegpu.tensor_desc<256x32xf16, #a> -> vector<256x32xf16>
        %b_value = xegpu.load_nd %b_tdesc[%k, %n] : !xegpu.tensor_desc<32x256xf16, #b> -> vector<32x256xf16>
        // Prefetch next tiles.
        %prefetch_offset = arith.addi %k, %c96 : index
        xegpu.prefetch_nd %a_prefetch_tdesc[%m, %prefetch_offset] : !xegpu.tensor_desc<256x32xf16, #a_prefetch>
        xegpu.prefetch_nd %b_prefetch_tdesc[%prefetch_offset, %n] : !xegpu.tensor_desc<32x256xf16, #b_prefetch>
        %c_new_value = xegpu.dpas %a_value, %b_value, %c_value {layout_result_0 = #c}
          : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf32> -> vector<256x256xf32>
        scf.yield %c_new_value : vector<256x256xf32>
      }
      xegpu.store_nd %out, %c_tdesc[%m, %n] : vector<256x256xf32>, !xegpu.tensor_desc<256x256xf32, #c>
      gpu.return
    }
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


    // Initialize matrix C and C_ref ; C[i, j] = 0
    %c0_f32 = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c4096 step %c1 {
      scf.for %j = %c0 to %c4096 step %c1 {
        memref.store %c0_f32, %C[%i, %j] : memref<4096x4096xf32>
        memref.store %c0_f32, %C_ref[%i, %j] : memref<4096x4096xf32>
      }
    }

    // Run GPU version.
    %2 = call @test(%A, %B, %C) : (memref<4096x4096xf16>, memref<4096x4096xf16>, memref<4096x4096xf32>) -> memref<4096x4096xf32>
    %gpu_result_cast = memref.cast %2 : memref<4096x4096xf32> to memref<*xf32>

    // Run CPU version.
    %A_cast = memref.cast %A : memref<4096x4096xf16> to memref<*xf16>
    %B_cast = memref.cast %B : memref<4096x4096xf16> to memref<*xf16>
    %C_cast = memref.cast %C_ref : memref<4096x4096xf32> to memref<*xf32>
    call @gemmF16F16F32(%A_cast, %B_cast, %C_cast) : (memref<*xf16>, memref<*xf16>, memref<*xf32>) -> ()

    %C_row_0 = memref.subview %C_ref[0, 0][1, 4096][1, 1] : memref<4096x4096xf32> to memref<1x4096xf32, strided<[4096, 1], offset:0>>
    %C_row_0_cast = memref.cast %C_row_0 : memref<1x4096xf32, strided<[4096, 1], offset: 0>> to memref<*xf32>
    call @printMemrefF32(%C_row_0_cast) : (memref<*xf32>) -> ()

    %C_row_0_gpu  = memref.subview %2[0, 0][1, 4096][1, 1] : memref<4096x4096xf32> to memref<1x4096xf32, strided<[4096, 1], offset:0>>
    %C_row_0_cast_gpu = memref.cast %C_row_0_gpu : memref<1x4096xf32, strided<[4096, 1], offset: 0>> to memref<*xf32>
    call @printMemrefF32(%C_row_0_cast_gpu) : (memref<*xf32>) -> ()

    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%gpu_result_cast, %C_cast) : (memref<*xf32>, memref<*xf32>) -> ()
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
  func.func private @gemmF16F16F32(memref<*xf16>, memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}

}
