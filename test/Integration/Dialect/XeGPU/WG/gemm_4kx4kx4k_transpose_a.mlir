// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --shared-libs=%irunner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

module attributes {gpu.container_module} {
  func.func @test(%A: memref<4096x4096xf16>, %B: memref<4096x4096xf16>, %C: memref<4096x4096xf32>) -> memref<4096x4096xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c512 = arith.constant 512 : index
    %A_gpu = gpu.alloc () : memref<4096x4096xf16>
    gpu.memcpy %A_gpu, %A : memref<4096x4096xf16>, memref<4096x4096xf16>
    %B_gpu = gpu.alloc () : memref<4096x4096xf16>
    gpu.memcpy %B_gpu, %B : memref<4096x4096xf16>, memref<4096x4096xf16>
    %C_gpu = gpu.alloc () : memref<4096x4096xf32>
    gpu.memcpy %C_gpu, %C : memref<4096x4096xf32>, memref<4096x4096xf32>
    gpu.launch_func @main_kernel::@main_kernel blocks in (%c16, %c16, %c1) threads in (%c512, %c1, %c1) args(%A_gpu : memref<4096x4096xf16>, %B_gpu : memref<4096x4096xf16>, %C_gpu : memref<4096x4096xf32>)
    gpu.wait
    gpu.memcpy %C, %C_gpu : memref<4096x4096xf32>, memref<4096x4096xf32>
    gpu.dealloc %A_gpu : memref<4096x4096xf16>
    gpu.dealloc %B_gpu : memref<4096x4096xf16>
    gpu.dealloc %C_gpu : memref<4096x4096xf32>
    return %C : memref<4096x4096xf32>
  }

  gpu.module @main_kernel [#xevm.target<O = 3>] {
    gpu.func @main_kernel(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf32>) kernel attributes {known_block_size = array<i32: 512, 1, 1>, known_grid_size = array<i32: 16, 16, 1>} {
      %cst = arith.constant dense<0.000000e+00> : vector<256x256xf32>
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %0 = arith.muli %block_id_x, %c256 overflow<nsw> : index
      %1 = arith.muli %block_id_y, %c256 overflow<nsw> : index
      %2 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %3 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %4 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %cst) -> (vector<256x256xf32>) {
        %6 = xegpu.load_nd %2[%arg3, %1] <{layout = #xegpu.layout<sg_layout = [4, 8], sg_data = [32, 32], inst_data = [16, 16]>}> : !xegpu.tensor_desc<32x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<32x256xf16>
        %7 = xegpu.load_nd %3[%arg3, %0] <{layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [16, 16], order = [0, 1]>}> : !xegpu.tensor_desc<32x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<32x256xf16>
        %8 = vector.transpose %7, [1, 0] : vector<32x256xf16> to vector<256x32xf16>
        %9 = xegpu.dpas %8, %6, %arg4 {layout_a = #xegpu.layout<sg_layout = [4, 8], sg_data = [64, 32], inst_data = [8, 16]>, layout_b = #xegpu.layout<sg_layout = [4, 8], sg_data = [32, 32], inst_data = [16, 16]>, layout_cd = #xegpu.layout<sg_layout = [4, 8], sg_data = [64, 32], inst_data = [8, 16]>} : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf32> -> vector<256x256xf32>
        scf.yield %9 : vector<256x256xf32>
      }
      %5 = xegpu.create_nd_tdesc %arg2 : memref<4096x4096xf32> -> !xegpu.tensor_desc<256x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %4, %5[%0, %1] <{layout = #xegpu.layout<sg_layout = [4, 8], sg_data = [64, 32], inst_data = [8, 16]>}> : vector<256x256xf32>, !xegpu.tensor_desc<256x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4096 = arith.constant 4096 : index
    %c_gen_int = arith.constant 0 : i1
    %cf_lower = arith.constant -0.5 : f32
    %cf_upper = arith.constant 0.5 : f32
    %c0_f32 = arith.constant 0.0 : f32

    // A is stored K-major: shape <4096 x 4096>, indexed as A[k, m] (i.e. A^T physical).
    %A = memref.alloc() : memref<4096x4096xf16>
    // B is stored K-major: shape <4096 x 4096>, indexed as B[k, n].
    %B = memref.alloc() : memref<4096x4096xf16>
    // C is the GPU output: shape <4096 x 4096> f32, indexed as C[m, n].
    %C = memref.alloc() : memref<4096x4096xf32>
    %C_ref = memref.alloc() : memref<4096x4096xf32>

    // Fill A and B with random values in (-0.5, 0.5).
    %A_random = memref.cast %A : memref<4096x4096xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%A_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf16>, f32, f32, i1) -> ()
    %B_random = memref.cast %B : memref<4096x4096xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%B_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf16>, f32, f32, i1) -> ()

    // Initialize C and C_ref with zeros.
    scf.for %i = %c0 to %c4096 step %c1 {
      scf.for %j = %c0 to %c4096 step %c1 {
        memref.store %c0_f32, %C[%i, %j] : memref<4096x4096xf32>
        memref.store %c0_f32, %C_ref[%i, %j] : memref<4096x4096xf32>
      }
    }

    // Run GPU version.
    %gpu_out = call @test(%A, %B, %C) : (memref<4096x4096xf16>, memref<4096x4096xf16>, memref<4096x4096xf32>) -> memref<4096x4096xf32>

    // Build a non-transposed view of A for the CPU reference: A_mk[m, k] = A[k, m].
    %A_mk = memref.alloc() : memref<4096x4096xf16>
    scf.for %m = %c0 to %c4096 step %c1 {
      scf.for %k = %c0 to %c4096 step %c1 {
        %a_val = memref.load %A[%k, %m] : memref<4096x4096xf16>
        memref.store %a_val, %A_mk[%m, %k] : memref<4096x4096xf16>
      }
    }

    // CPU reference: C_ref = A_mk @ B (gemmF16F16F32 computes C = A @ B with no transposes).
    %A_mk_cast = memref.cast %A_mk : memref<4096x4096xf16> to memref<*xf16>
    %B_cast = memref.cast %B : memref<4096x4096xf16> to memref<*xf16>
    %C_ref_cast = memref.cast %C_ref : memref<4096x4096xf32> to memref<*xf32>
    call @gemmF16F16F32(%A_mk_cast, %B_cast, %C_ref_cast) : (memref<*xf16>, memref<*xf16>, memref<*xf32>) -> ()

    %gpu_out_cast = memref.cast %gpu_out : memref<4096x4096xf32> to memref<*xf32>

    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%gpu_out_cast, %C_ref_cast) : (memref<*xf32>, memref<*xf32>) -> ()

    memref.dealloc %A : memref<4096x4096xf16>
    memref.dealloc %B : memref<4096x4096xf16>
    memref.dealloc %C : memref<4096x4096xf32>
    memref.dealloc %C_ref : memref<4096x4096xf32>
    memref.dealloc %A_mk : memref<4096x4096xf16>
    return
  }

  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @gemmF16F16F32(memref<*xf16>, memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
