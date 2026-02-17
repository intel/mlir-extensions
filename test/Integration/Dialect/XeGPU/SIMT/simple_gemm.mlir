// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=lane" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --shared-libs=%irunner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// Example of pass pipeline usage:
// %python_executable %imex_runner --requires=mlir-levelzero-runtime,spirv-backend -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
//                                        --runner mlir-runner -e main \
//                                        --entry-point-result=void \
//                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @simple_gemm(%a: memref<256x256xf16>, %b: memref<256x256xf16>, %c: memref<256x256xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      %block_x = gpu.block_id x
      %block_y = gpu.block_id y
      %x_block_offset = arith.muli %block_x, %c8 : index
      %y_block_offset = arith.muli %block_y, %c16 : index

      %c_tdesc = xegpu.create_nd_tdesc %c : memref<256x256xf32> -> !xegpu.tensor_desc<8x16xf32>
      %c_init_value = xegpu.load_nd %c_tdesc[%x_block_offset, %y_block_offset] : !xegpu.tensor_desc<8x16xf32> -> vector<8xf32>
      %a_tdesc = xegpu.create_nd_tdesc %a : memref<256x256xf16> -> !xegpu.tensor_desc<8x16xf16>
      %b_tdesc = xegpu.create_nd_tdesc %b : memref<256x256xf16> -> !xegpu.tensor_desc<16x16xf16>

      %r = scf.for %k = %c0 to %c256 step %c16 iter_args(%arg_c = %c_init_value) -> (vector<8xf32>) {

        %a_val = xegpu.load_nd %a_tdesc[%x_block_offset, %k] : !xegpu.tensor_desc<8x16xf16> -> vector<8xf16>
        %b_val = xegpu.load_nd %b_tdesc[%k, %y_block_offset] : !xegpu.tensor_desc<16x16xf16> -> vector<16xf16>
        %dpas = xegpu.dpas %a_val, %b_val, %arg_c : vector<8xf16>, vector<16xf16>, vector<8xf32> -> vector<8xf32>
        scf.yield %dpas : vector<8xf32>
      }
      xegpu.store_nd %r, %c_tdesc[%x_block_offset, %y_block_offset] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>: vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }

  func.func @test(%a : memref<256x256xf16>, %b : memref<256x256xf16>, %c : memref<256x256xf32>) -> memref<256x256xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %memref_a = gpu.alloc  () : memref<256x256xf16>
    gpu.memcpy %memref_a, %a : memref<256x256xf16>, memref<256x256xf16>
    %memref_b = gpu.alloc  () : memref<256x256xf16>
    gpu.memcpy %memref_b, %b : memref<256x256xf16>, memref<256x256xf16>
    %memref_c = gpu.alloc  () : memref<256x256xf32>
    gpu.memcpy %memref_c, %c : memref<256x256xf32>, memref<256x256xf32>
    gpu.launch_func @kernel::@simple_gemm blocks in (%c32, %c16, %c1) threads in (%c16, %c1, %c1) args(%memref_a : memref<256x256xf16>, %memref_b : memref<256x256xf16>, %memref_c : memref<256x256xf32>)
    gpu.wait // Wait for the kernel to finish.
    gpu.memcpy %c, %memref_c : memref<256x256xf32>, memref<256x256xf32>
    gpu.dealloc %memref_a : memref<256x256xf16>
    gpu.dealloc %memref_b : memref<256x256xf16>
    gpu.dealloc %memref_c : memref<256x256xf32>
    return %c : memref<256x256xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_f16 = arith.constant 1.0 : f16
    %c2_f16 = arith.constant 2.0 : f16
    %c256 = arith.constant 256 : index
    %cf_0 = arith.constant 0.0 : f16
    %cf_1 = arith.constant 1.0 : f16
    %A = memref.alloc() : memref<256x256xf16>
    %B = memref.alloc() : memref<256x256xf16>
    %C = memref.alloc() : memref<256x256xf32>
    %C_ref = memref.alloc() : memref<256x256xf32>
    %c_gen_int = arith.constant 0 : i1
    %cf_lower = arith.constant -0.5 : f32
    %cf_upper = arith.constant 0.5 : f32
    // Use one of the two options to initialize the A matrix
    // Option 1: intialize matrix A ; A[i, j] = j
    // scf.for %i = %c0 to %c256 step %c1 {
    //   scf.for %j = %c0 to %c256 step %c1 {
    //     %t = index.castu %j : index to i16
    //     %val = arith.uitofp %t : i16 to f16
    //     memref.store %val, %A[%i, %j] : memref<256x256xf16>
    //     // memref.store %c1_f16, %A[%i, %j] : memref<256x256xf16>
    //     // memref.store %c2_f16, %B[%i, %j] : memref<256x256xf16>
    //   }
    // }
    // Option 2:  convert the memref to 1D and fill with random values in (-0.5, 0.5)
    %A_random = memref.cast %A : memref<256x256xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%A_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf16>, f32, f32, i1) -> ()


    // Use one of the two options below to initialize the B matrix
    // Option 1: make matrix B an identity matrix
    // scf.for %i = %c0 to %c256 step %c1 {
    //   scf.for %j = %c0 to %c256 step %c1 {
    //     %i_i32 = index.castu %i : index to i32
    //     %j_i32 = index.castu %j : index to i32
    //     %i_j_same = arith.cmpi eq, %i_i32, %j_i32 : i32

    //     scf.if %i_j_same {
    //       memref.store %cf_1, %B[%i, %j] : memref<256x256xf16>
    //     } else {
    //       memref.store %cf_0, %B[%i, %j] : memref<256x256xf16>
    //     }
    //   }
    // }
    // Option 2:  convert the memref to 1D and fill with random values in (-0.5, 0.5)
    %B_random = memref.cast %B : memref<256x256xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%B_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf16>, f32, f32, i1) -> ()


    // Initialize matrix C and C_ref ; C[i, j] = 0
    %c0_f32 = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        memref.store %c0_f32, %C[%i, %j] : memref<256x256xf32>
        memref.store %c0_f32, %C_ref[%i, %j] : memref<256x256xf32>
      }
    }

    // Run GPU version.
    %2 = call @test(%A, %B, %C) : (memref<256x256xf16>, memref<256x256xf16>, memref<256x256xf32>) -> memref<256x256xf32>
    %gpu_result_cast = memref.cast %2 : memref<256x256xf32> to memref<*xf32>

    // Run CPU version.
    %A_cast = memref.cast %A : memref<256x256xf16> to memref<*xf16>
    %B_cast = memref.cast %B : memref<256x256xf16> to memref<*xf16>
    %C_cast = memref.cast %C_ref : memref<256x256xf32> to memref<*xf32>
    call @gemmF16F16F32(%A_cast, %B_cast, %C_cast) : (memref<*xf16>, memref<*xf16>, memref<*xf32>) -> ()

    %C_row_0 = memref.subview %C_ref[0, 0][1, 256][1, 1] : memref<256x256xf32> to memref<1x256xf32, strided<[256, 1], offset:0>>
    %C_row_0_cast = memref.cast %C_row_0 : memref<1x256xf32, strided<[256, 1], offset: 0>> to memref<*xf32>
    // call @printMemrefF32(%C_row_0_cast) : (memref<*xf32>) -> ()

    %C_row_0_gpu  = memref.subview %2[0, 0][1, 256][1, 1] : memref<256x256xf32> to memref<1x256xf32, strided<[256, 1], offset:0>>
    %C_row_0_cast_gpu = memref.cast %C_row_0_gpu : memref<1x256xf32, strided<[256, 1], offset: 0>> to memref<*xf32>
    // call @printMemrefF32(%C_row_0_cast_gpu) : (memref<*xf32>) -> ()

    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%gpu_result_cast, %C_cast) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %A : memref<256x256xf16>
    memref.dealloc %B : memref<256x256xf16>
    memref.dealloc %C : memref<256x256xf32>
    memref.dealloc %C_ref : memref<256x256xf32>
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @gemmF16F16F32(memref<*xf16>, memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
