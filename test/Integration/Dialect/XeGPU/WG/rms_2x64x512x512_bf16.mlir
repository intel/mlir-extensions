// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup" \
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

// Workgroup-level RMSNorm over dim 1, on the NCHW shape of the KernelBench
// level1/36_RMSNorm benchmark:
//
//   out[b, c, h, w] = in[b, c, h, w] / sqrt(mean_c(in[b, :, h, w]^2) + 1e-5)
//
// The kernel is a generated one, kept as emitted. Its arguments are positional:
// %arg0 is the *output*, written through %subview / %subview_3 / %5; %arg1 is
// the *input*, only ever read through %4. The host wrapper below therefore
// launches with `args(%out_gpu, %in_gpu)`.
//
// Grid mapping, fixed by the kernel's own constants and attributes:
//   block_id_x -> batch, block_id_y * 128 -> H tile, block_id_z * 128 -> W tile
//   known_block_size = [512, 1, 1]  -> threads in (512, 1, 1)
//   sg_layout = [32, 1, 1], sg_data = [1, 16, 32] over 512 work items =
//   32 subgroups x 16 lanes, i.e. 32 threads distribute the 32-deep reduction
//   chunk, 1 channel each.
// H = W = 512 tiles exactly (512 = 4 * 128), so the grid is (batch, 4, 4) and
// every (block_id_y, block_id_z) pair is exercised.
//
// Only edit vs. the generated kernel: the batch dim is 2 instead of 112, i.e.
// `known_grid_size = [2, 4, 4]` and the memref types, so the test fits the
// suite's memory budget (112 batches is 3.75 GiB per buffer -- 7.5 GiB on the
// device and 15 GiB on the host once the f32 reference is added). Nothing else
// changes: batch is only ever `block_id_x`, and every stride in the kernel
// (16777216 = 64*512*512, 262144 = 512*512, 512) is independent of the batch
// extent, so the bytes of every subview and descriptor are untouched. Two
// batches keep block_id_x non-degenerate.
//
// Fill: every element is +-1, sign taken from (b + 2c + h + 2w) mod 3 (mod 3
// rather than a parity so the pattern is not symmetric in h and w, and so it
// varies along c). This makes the whole kernel exact in bf16 and the check
// independent of reduction order: every square is exactly 1, so sum_c(x^2) = 64
// and every partial sum is an integer < 256, all exactly representable in bf16;
// 64/64 = 1; 1 + 1e-5 rounds to 1.0 in bf16 (bf16 ulp at 1.0 is 2^-8); sqrt is
// 1.0; and x/1.0 = +-1 exactly. The f32 reference computes the unrounded
// +-1/sqrt(1 + 1e-5) = +-0.999995, within the runner's atol=1e-4.
// This matters because the kernel accumulates the sum of squares in bf16 (an
// artifact of the conversion that produced it), and bf16's ~4e-3 worst-case
// rounding is *larger* than the runner's rtol=1e-3, so with arbitrary inputs
// neither the accumulation nor the final `arith.divf ... : bf16` could be
// compared against an unrounded f32 reference at this tolerance at all. Landing
// on exactly +-1 sidesteps that while still checking the real addressing: the
// sign makes every element's (b, c, h, w) mapping observable, and any channel
// the reduction drops or double-counts moves sum_c away from 64 and the ratio
// measurably off 1.0. Do not switch this to `fillResource1DRandomBF16`.
//
// The output buffer is zero-filled before the launch, so a channel the kernel
// never writes shows up as 0.0 against a +-0.999995 reference rather than
// silently inheriting whatever was on the device. That is what makes this test
// a regression guard for the base_height bug described next: the output
// descriptor is a 3D view of a strided subview
// (memref<64x128x128xbf16, strided<[262144, 512, 1]>>), whose planes are not
// packed back to back. Before llvm/llvm-project#225556 the XeGPU->XeVM lowering
// sized the flattened 2D surface as the product of the leading sizes
// (128 * 64 = 8192 rows) instead of spanning the plane stride (512 rows per
// plane), so channels 16..63 fell outside base_height and their stores were
// silently dropped by the hardware boundary check.
module @rms attributes {gpu.container_module} {
  func.func @test(%in: memref<2x64x512x512xbf16>, %out: memref<2x64x512x512xbf16>) -> memref<2x64x512x512xbf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c512 = arith.constant 512 : index
    %in_gpu = gpu.alloc () : memref<2x64x512x512xbf16>
    gpu.memcpy %in_gpu, %in : memref<2x64x512x512xbf16>, memref<2x64x512x512xbf16>
    %out_gpu = gpu.alloc () : memref<2x64x512x512xbf16>
    gpu.memcpy %out_gpu, %out : memref<2x64x512x512xbf16>, memref<2x64x512x512xbf16>
    // payload_kernel(%arg0 = output, %arg1 = input).
    gpu.launch_func @payload_kernel::@payload_kernel blocks in (%c2, %c4, %c4) threads in (%c512, %c1, %c1) args(%out_gpu : memref<2x64x512x512xbf16>, %in_gpu : memref<2x64x512x512xbf16>)
    gpu.wait
    gpu.memcpy %out, %out_gpu : memref<2x64x512x512xbf16>, memref<2x64x512x512xbf16>
    gpu.dealloc %in_gpu : memref<2x64x512x512xbf16>
    gpu.dealloc %out_gpu : memref<2x64x512x512xbf16>
    return %out : memref<2x64x512x512xbf16>
  }

  gpu.module @payload_kernel [#xevm.target<O = 3>] {
    gpu.func @payload_kernel(%arg0: memref<2x64x512x512xbf16>, %arg1: memref<2x64x512x512xbf16>) kernel attributes {known_block_size = array<i32: 512, 1, 1>, known_grid_size = array<i32: 2, 4, 4>} {
      %c16 = arith.constant 16 : index
      %cst = arith.constant dense<6.400000e+01> : vector<16x32xbf16>
      %cst_0 = arith.constant dense<0.000000e+00> : vector<16x32xbf16>
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst_1 = arith.constant 1.000000e-05 : f64
      %c128 = arith.constant 128 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %block_id_z = gpu.block_id z
      %0 = arith.muli %block_id_y, %c128 overflow<nsw> : index
      %1 = arith.muli %block_id_z, %c128 overflow<nsw> : index
      %subview = memref.subview %arg0[%block_id_x, 0, %0, %1] [1, 64, 128, 128] [1, 1, 1, 1] : memref<2x64x512x512xbf16> to memref<1x64x128x128xbf16, strided<[16777216, 262144, 512, 1], offset: ?>>
      %2 = arith.truncf %cst_1 : f64 to bf16
      %3 = vector.broadcast %2 : bf16 to vector<16x32xbf16>
      %subview_2 = memref.subview %arg1[%block_id_x, 0, 0, 0] [1, 64, 512, 512] [1, 1, 1, 1] : memref<2x64x512x512xbf16> to memref<64x512x512xbf16, strided<[262144, 512, 1], offset: ?>>
      %4 = xegpu.create_nd_tdesc %subview_2 : memref<64x512x512xbf16, strided<[262144, 512, 1], offset: ?>> -> !xegpu.tensor_desc<32x16x32xbf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %subview_3 = memref.subview %subview[0, 0, 0, 0] [1, 64, 128, 128] [1, 1, 1, 1] : memref<1x64x128x128xbf16, strided<[16777216, 262144, 512, 1], offset: ?>> to memref<64x128x128xbf16, strided<[262144, 512, 1], offset: ?>>
      %5 = xegpu.create_nd_tdesc %subview_3 : memref<64x128x128xbf16, strided<[262144, 512, 1], offset: ?>> -> !xegpu.tensor_desc<32x16x32xbf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      scf.for %arg2 = %c0 to %c128 step %c16 {
        %6 = arith.addi %0, %arg2 : index
        scf.for %arg3 = %c0 to %c128 step %c32 {
          %7 = arith.addi %1, %arg3 : index
          %8 = scf.for %arg4 = %c0 to %c64 step %c32 iter_args(%arg5 = %cst_0) -> (vector<16x32xbf16>) {
            %13 = xegpu.load_nd %4[%arg4, %6, %7] <{layout = #xegpu.layout<sg_layout = [32, 1, 1], sg_data = [1, 16, 32]>}> : !xegpu.tensor_desc<32x16x32xbf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<32x16x32xbf16>
            %14 = arith.mulf %13, %13 : vector<32x16x32xbf16>
            %15 = vector.multi_reduction <add>, %14, %arg5 [0] : vector<32x16x32xbf16> to vector<16x32xbf16>
            scf.yield %15 : vector<16x32xbf16>
          }
          %9 = arith.divf %8, %cst : vector<16x32xbf16>
          %10 = arith.addf %9, %3 : vector<16x32xbf16>
          %11 = math.sqrt %10 : vector<16x32xbf16>
          %12 = vector.broadcast %11 : vector<16x32xbf16> to vector<32x16x32xbf16>
          scf.for %arg4 = %c0 to %c64 step %c32 {
            %13 = xegpu.load_nd %4[%arg4, %6, %7] : !xegpu.tensor_desc<32x16x32xbf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<32x16x32xbf16>
            %14 = arith.divf %13, %12 : vector<32x16x32xbf16>
            xegpu.store_nd %14, %5[%arg4, %arg2, %arg3] <{layout = #xegpu.layout<sg_layout = [32, 1, 1], sg_data = [1, 16, 32]>}> : vector<32x16x32xbf16>, !xegpu.tensor_desc<32x16x32xbf16, #xegpu.block_tdesc_attr<boundary_check = false>>
          }
        }
      }
      gpu.return
    }
  }

  // Mirrors the kernel's own index space: iterate (b, h, w), reduce over c.
  // Computed in f32; see the header for why the fill makes the kernel's bf16
  // rounding irrelevant to the comparison.
  func.func @rms_reference(%in: memref<2x64x512x512xbf16>, %out_ref: memref<2x64x512x512xf32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    %zero = arith.constant 0.0 : f32
    %c64_f32 = arith.constant 6.400000e+01 : f32
    %eps = arith.constant 1.000000e-05 : f32
    scf.for %b = %c0 to %c2 step %c1 {
      scf.for %h = %c0 to %c512 step %c1 {
        scf.for %w = %c0 to %c512 step %c1 {
          %ssq = scf.for %c = %c0 to %c64 step %c1 iter_args(%acc = %zero) -> f32 {
            %v = memref.load %in[%b, %c, %h, %w] : memref<2x64x512x512xbf16>
            %vf = arith.extf %v : bf16 to f32
            %sq = arith.mulf %vf, %vf : f32
            %acc_new = arith.addf %acc, %sq : f32
            scf.yield %acc_new : f32
          }
          %mean = arith.divf %ssq, %c64_f32 : f32
          %mean_eps = arith.addf %mean, %eps : f32
          %rms = math.sqrt %mean_eps : f32
          scf.for %c = %c0 to %c64 step %c1 {
            %v = memref.load %in[%b, %c, %h, %w] : memref<2x64x512x512xbf16>
            %vf = arith.extf %v : bf16 to f32
            %o = arith.divf %vf, %rms : f32
            memref.store %o, %out_ref[%b, %c, %h, %w] : memref<2x64x512x512xf32>
          }
        }
      }
    }
    return
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    %pos1_bf16 = arith.constant 1.000000e+00 : bf16
    %neg1_bf16 = arith.constant -1.000000e+00 : bf16
    %zero_bf16 = arith.constant 0.000000e+00 : bf16
    %zero_f32 = arith.constant 0.0 : f32

    %in = memref.alloc() : memref<2x64x512x512xbf16>
    %out = memref.alloc() : memref<2x64x512x512xbf16>
    %out_ref = memref.alloc() : memref<2x64x512x512xf32>

    // in[b, c, h, w] = +-1, sign from (b + 2c + h + 2w) mod 3; see header.
    scf.for %b = %c0 to %c2 step %c1 {
      scf.for %c = %c0 to %c64 step %c1 {
        scf.for %h = %c0 to %c512 step %c1 {
          scf.for %w = %c0 to %c512 step %c1 {
            %c2x = arith.muli %c, %c2 : index
            %w2x = arith.muli %w, %c2 : index
            %t0 = arith.addi %b, %c2x : index
            %t1 = arith.addi %t0, %h : index
            %t2 = arith.addi %t1, %w2x : index
            %r = arith.remui %t2, %c3 : index
            %is_neg = arith.cmpi eq, %r, %c0 : index
            %sign = arith.select %is_neg, %neg1_bf16, %pos1_bf16 : bf16
            memref.store %sign, %in[%b, %c, %h, %w] : memref<2x64x512x512xbf16>
            memref.store %zero_bf16, %out[%b, %c, %h, %w] : memref<2x64x512x512xbf16>
            memref.store %zero_f32, %out_ref[%b, %c, %h, %w] : memref<2x64x512x512xf32>
          }
        }
      }
    }

    %gpu_result = call @test(%in, %out) : (memref<2x64x512x512xbf16>, memref<2x64x512x512xbf16>) -> memref<2x64x512x512xbf16>
    %gpu_result_cast = memref.cast %gpu_result : memref<2x64x512x512xbf16> to memref<*xbf16>

    call @rms_reference(%in, %out_ref) : (memref<2x64x512x512xbf16>, memref<2x64x512x512xf32>) -> ()
    %ref_cast = memref.cast %out_ref : memref<2x64x512x512xf32> to memref<*xf32>

    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseBF16(%gpu_result_cast, %ref_cast) : (memref<*xbf16>, memref<*xf32>) -> ()

    memref.dealloc %in : memref<2x64x512x512xbf16>
    memref.dealloc %out : memref<2x64x512x512xbf16>
    memref.dealloc %out_ref : memref<2x64x512x512xf32>
    return
  }
  func.func private @printAllcloseBF16(memref<*xbf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
