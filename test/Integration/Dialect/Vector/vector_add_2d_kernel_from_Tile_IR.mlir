// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup enable-vector-to-xegpu=true igc-cmd-options=-ze-opt-large-register-file" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --shared-libs=%irunner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

//
// Elementwise 2D vector add written in the vector dialect and lowered to XeVM
// through the workgroup pipeline.
//
// This is the ORIGINAL, unmodified kernel as emitted by the cuTile / Tile-IR
// front end (fully dynamic strided 2D memrefs, unmasked rank-2 nd transfers,
// `ub.poison` padding value, and the "tir-dropped-*" hint attributes
// preserved). It is kept verbatim ON PURPOSE.
//
// The interesting part is the workgroup shape rather than the arithmetic: the
// kernel declares 512 work items (32 subgroups of 16 lanes) and each workgroup
// handles a 64x16xf32 tile, so one subgroup only owns a 2x16xf32 slice. That is
// smaller than the largest 8x16 hardware block, which used to make subgroup
// layout propagation give up with
//
//   error: Failed to determine required layout for store_nd.
//
// The upstream fix is llvm/llvm-project#239888 (merged as 00f4ec3c3a66), which
// retries progressively smaller inst_data until a subgroup layout fits.
//
// Problem shape (fixed by the kernel's index math / vector types):
//   M = 256   (block_id_x covers 64 rows each, so blocks_x = M / 64 = 4)
//   N = 64    (block_id_y covers 16 cols each, so blocks_y = N / 16 = 4)
//
// Grid mapping:
//   block_id_x  -> row tile   (64 rows)
//   block_id_y  -> column tile (16 columns)
//   block dims  -> match known_block_size = [512, 1, 1]
//
module @vector_add attributes {gpu.container_module} {
  gpu.module @kernels [#xevm.target<flags = {"cmd-options" = ["-ze-opt-large-register-file"]}>] {
    gpu.func @vec_add_kernel_2d_Kt1_A2f32_1v4l0_2i16t1_p16_A2f32_1v4l0_2i16t1_p16_A2f32_1v4l0_2i16t1_p16_I64_I16(%arg0: memref<?x?xf32, strided<[?, 1]>>, %arg1: memref<?x?xf32, strided<[?, 1]>>, %arg2: memref<?x?xf32, strided<[?, 1]>>) kernel attributes {"tir-dropped-optimization-hints" = {default = {}}, known_block_size = array<i32: 512, 1, 1>} {
      %0 = ub.poison : f32
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %block_id_x = gpu.block_id x
      %1 = arith.index_cast %block_id_x : index to i32
      %block_id_y = gpu.block_id y
      %2 = arith.index_cast %block_id_y : index to i32
      %3 = arith.index_castui %1 nneg : i32 to index
      %4 = arith.muli %3, %c64 overflow<nsw> : index
      %5 = arith.index_castui %2 nneg : i32 to index
      %6 = arith.muli %5, %c16 overflow<nsw> : index
      %7 = vector.transfer_read %arg0[%4, %6], %0 : memref<?x?xf32, strided<[?, 1]>>, vector<64x16xf32>
      %8 = vector.transfer_read %arg1[%4, %6], %0 : memref<?x?xf32, strided<[?, 1]>>, vector<64x16xf32>
      %9 = arith.addf %7, %8 : vector<64x16xf32>
      vector.transfer_write %9, %arg2[%4, %6] : vector<64x16xf32>, memref<?x?xf32, strided<[?, 1]>>
      gpu.return
    }
  }

  // Host wrapper: copy A/B to device, launch the kernel, copy C back.
  func.func @gpu_impl(%A: memref<256x64xf32>, %B: memref<256x64xf32>,
                      %C: memref<256x64xf32>) -> memref<256x64xf32> {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index     // blocks_x = M / 64 = 256 / 64
                                       // blocks_y = N / 16 = 64 / 16 (same value)
    // block dims match known_block_size = [512, 1, 1]
    %bx = arith.constant 512 : index

    %A_gpu = gpu.alloc () : memref<256x64xf32>
    gpu.memcpy %A_gpu, %A : memref<256x64xf32>, memref<256x64xf32>
    %B_gpu = gpu.alloc () : memref<256x64xf32>
    gpu.memcpy %B_gpu, %B : memref<256x64xf32>, memref<256x64xf32>
    %C_gpu = gpu.alloc () : memref<256x64xf32>
    gpu.memcpy %C_gpu, %C : memref<256x64xf32>, memref<256x64xf32>

    // The kernel expects fully-dynamic strided 2D memrefs. A contiguous static
    // memref conforms to strided<[?, 1]>, so cast.
    %A_arg = memref.cast %A_gpu : memref<256x64xf32> to memref<?x?xf32, strided<[?, 1]>>
    %B_arg = memref.cast %B_gpu : memref<256x64xf32> to memref<?x?xf32, strided<[?, 1]>>
    %C_arg = memref.cast %C_gpu : memref<256x64xf32> to memref<?x?xf32, strided<[?, 1]>>

    gpu.launch_func @kernels::@vec_add_kernel_2d_Kt1_A2f32_1v4l0_2i16t1_p16_A2f32_1v4l0_2i16t1_p16_A2f32_1v4l0_2i16t1_p16_I64_I16
      blocks in (%c4, %c4, %c1) threads in (%bx, %c1, %c1)
      args(%A_arg : memref<?x?xf32, strided<[?, 1]>>,
           %B_arg : memref<?x?xf32, strided<[?, 1]>>,
           %C_arg : memref<?x?xf32, strided<[?, 1]>>)
    gpu.wait

    gpu.memcpy %C, %C_gpu : memref<256x64xf32>, memref<256x64xf32>
    gpu.dealloc %A_gpu : memref<256x64xf32>
    gpu.dealloc %B_gpu : memref<256x64xf32>
    gpu.dealloc %C_gpu : memref<256x64xf32>
    return %C : memref<256x64xf32>
  }

  // Scalar reference: C[i, j] = A[i, j] + B[i, j].
  func.func @cpu_impl(%A: memref<256x64xf32>, %B: memref<256x64xf32>,
                      %C: memref<256x64xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c64 step %c1 {
        %a = memref.load %A[%i, %j] : memref<256x64xf32>
        %b = memref.load %B[%i, %j] : memref<256x64xf32>
        %c = arith.addf %a, %b : f32
        memref.store %c, %C[%i, %j] : memref<256x64xf32>
      }
    }
    return
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    // random fill params
    %rand_low = arith.constant -1.0 : f32
    %rand_high = arith.constant 1.0 : f32
    %gen_int = arith.constant 0 : i1

    %A = memref.alloc() : memref<256x64xf32>
    %B = memref.alloc() : memref<256x64xf32>
    %C = memref.alloc() : memref<256x64xf32>
    %C_ref = memref.alloc() : memref<256x64xf32>

    %A_u = memref.cast %A : memref<256x64xf32> to memref<*xf32>
    %B_u = memref.cast %B : memref<256x64xf32> to memref<*xf32>

    call @fillResource1DRandomF32(%A_u, %rand_low, %rand_high, %gen_int) : (memref<*xf32>, f32, f32, i1) -> ()
    call @fillResource1DRandomF32(%B_u, %rand_low, %rand_high, %gen_int) : (memref<*xf32>, f32, f32, i1) -> ()

    // Note: both outputs are fully overwritten (the GPU kernel writes all of C,
    // and @cpu_impl writes all of C_ref), so no explicit zero-init is needed.

    // GPU run
    %gpu_out = call @gpu_impl(%A, %B, %C)
      : (memref<256x64xf32>, memref<256x64xf32>, memref<256x64xf32>) -> memref<256x64xf32>

    // CPU reference
    call @cpu_impl(%A, %B, %C_ref)
      : (memref<256x64xf32>, memref<256x64xf32>, memref<256x64xf32>) -> ()

    %gpu_out_u = memref.cast %gpu_out : memref<256x64xf32> to memref<*xf32>
    %ref_u = memref.cast %C_ref : memref<256x64xf32> to memref<*xf32>

    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%gpu_out_u, %ref_u) : (memref<*xf32>, memref<*xf32>) -> ()

    memref.dealloc %A : memref<256x64xf32>
    memref.dealloc %B : memref<256x64xf32>
    memref.dealloc %C : memref<256x64xf32>
    memref.dealloc %C_ref : memref<256x64xf32>
    return
  }

  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF32(memref<*xf32>, f32, f32, i1) attributes {llvm.emit_c_interface}
}
