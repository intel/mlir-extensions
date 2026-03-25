// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --shared-libs=%irunner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

module attributes {gpu.container_module} {
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c4096 = arith.constant 4096 : index
    %c0_f16 = arith.constant 0.0 : f16
    %c0_f32 = arith.constant 0.0 : f32
    %c2_f32 = arith.constant 2.0 : f32
    %c3_f32 = arith.constant 3.0 : f32

    // Allocate host matrices.
    %A = memref.alloc() : memref<4096x4096xf16>
    %B = memref.alloc() : memref<4096x4096xf16>
    %C = memref.alloc() : memref<4096x4096xf16>
    %C_ref = memref.alloc() : memref<4096x4096xf32>

    // Fill A with 2.0 and B with 3.0 so that C[i,j] = 2*3*4096 = 24576,
    // which is exactly representable in f16 (= 1.5 * 2^14).
    %A_cast = memref.cast %A : memref<4096x4096xf16> to memref<*xf16>
    call @fillResource1DF16(%A_cast, %c2_f32) : (memref<*xf16>, f32) -> ()
    %B_cast = memref.cast %B : memref<4096x4096xf16> to memref<*xf16>
    call @fillResource1DF16(%B_cast, %c3_f32) : (memref<*xf16>, f32) -> ()

    // Initialize C and C_ref to 0.
    scf.for %i = %c0 to %c4096 step %c1 {
      scf.for %j = %c0 to %c4096 step %c1 {
        memref.store %c0_f16, %C[%i, %j] : memref<4096x4096xf16>
        memref.store %c0_f32, %C_ref[%i, %j] : memref<4096x4096xf32>
      }
    }

    // Allocate GPU buffers and copy inputs.
    %A_gpu = gpu.alloc () : memref<4096x4096xf16>
    gpu.memcpy %A_gpu, %A : memref<4096x4096xf16>, memref<4096x4096xf16>
    %B_gpu = gpu.alloc () : memref<4096x4096xf16>
    gpu.memcpy %B_gpu, %B : memref<4096x4096xf16>, memref<4096x4096xf16>
    %C_gpu = gpu.alloc () : memref<4096x4096xf16>

    // Launch kernel and wait for completion.
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c64, %c32, %c1) threads in (%c16, %c8, %c1)
      args(%A_gpu : memref<4096x4096xf16>, %B_gpu : memref<4096x4096xf16>, %C_gpu : memref<4096x4096xf16>)
    gpu.wait

    // Copy result back to host.
    gpu.memcpy %C, %C_gpu : memref<4096x4096xf16>, memref<4096x4096xf16>
    gpu.dealloc %A_gpu : memref<4096x4096xf16>
    gpu.dealloc %B_gpu : memref<4096x4096xf16>
    gpu.dealloc %C_gpu : memref<4096x4096xf16>

    // Run CPU reference in f32 precision.
    %C_ref_cast = memref.cast %C_ref : memref<4096x4096xf32> to memref<*xf32>
    call @gemmF16F16F32(%A_cast, %B_cast, %C_ref_cast) : (memref<*xf16>, memref<*xf16>, memref<*xf32>) -> ()

    // CHECK: [ALLCLOSE: TRUE]
    %C_cast = memref.cast %C : memref<4096x4096xf16> to memref<*xf16>
    call @printAllcloseF16(%C_cast, %C_ref_cast) : (memref<*xf16>, memref<*xf32>) -> ()

    memref.dealloc %A : memref<4096x4096xf16>
    memref.dealloc %B : memref<4096x4096xf16>
    memref.dealloc %C : memref<4096x4096xf16>
    memref.dealloc %C_ref : memref<4096x4096xf32>
    return
  }
  gpu.module @main_kernel [#xevm.target<O = 3, chip = "pvc">] {
    gpu.func @main_kernel(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) kernel attributes {known_block_size = array<i32: 16, 8, 1>} {
      %cst = arith.constant dense<0.000000e+00> : vector<64x128xf16>
      %c16 = arith.constant 16 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %0 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%block_id_x]
      %1 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%block_id_y]
      %2 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> ->
        !xegpu.tensor_desc<64x16xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [2, 1], sg_data = [32, 16]>>
      %3 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf16> ->
        !xegpu.tensor_desc<16x128xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [1, 4], sg_data = [16, 32]>>
      %4 = scf.for %arg3 = %c0 to %c4096 step %c16 iter_args(%arg4 = %cst) -> (vector<64x128xf16>) {
        %6 = xegpu.load_nd %2[%0, %arg3] <{layout = #xegpu.layout<sg_layout = [2, 1], sg_data = [32, 16]>}> :
          !xegpu.tensor_desc<64x16xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [2, 1], sg_data = [32, 16]>>
          -> vector<64x16xf16>
        %7 = xegpu.load_nd %3[%arg3, %1] <{layout = #xegpu.layout<sg_layout = [1, 4], sg_data = [16, 32]>}> :
          !xegpu.tensor_desc<16x128xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [1, 4], sg_data = [16, 32]>>
          -> vector<16x128xf16>
        %8 = xegpu.dpas %6, %7, %arg4
          {
            layout_a = #xegpu.layout<sg_layout = [2, 1], sg_data = [32, 16]>,
            layout_b = #xegpu.layout<sg_layout = [1, 4], sg_data = [16, 32]>,
            layout_cd = #xegpu.layout<sg_layout = [2, 4], sg_data = [32, 32]>
          } : vector<64x16xf16>, vector<16x128xf16>, vector<64x128xf16> -> vector<64x128xf16>
        scf.yield %8 : vector<64x128xf16>
      }
      %5 = xegpu.create_nd_tdesc %arg2 : memref<4096x4096xf16> ->
        !xegpu.tensor_desc<64x128xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [2, 4], sg_data = [32, 32]>>
      xegpu.store_nd %4, %5[%0, %1] <{layout = #xegpu.layout<sg_layout = [2, 4], sg_data = [32, 32]>}> :
        vector<64x128xf16>, !xegpu.tensor_desc<64x128xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [2, 4], sg_data = [32, 32]>>
      gpu.return
    }
  }
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF16(memref<*xf16>, f32) attributes {llvm.emit_c_interface}
  func.func private @gemmF16F16F32(memref<*xf16>, memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
