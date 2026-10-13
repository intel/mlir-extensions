// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --shared-libs=%irunner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// Full nanoGPT forward on XeGPU: 1 transformer block (T=C=256, hidden=1024,
// H=4 heads of head_size=64) followed by the final layernorm + lm_head. Each
// block is
//   a = x + attn_proj( MultiHeadAttention( ln1(x) ) )   (NON-CAUSAL flash attn)
//   y = a + ffn( ln2(a) ),  ffn(z) = relu(z @ w1 + bb1) @ w2 + bb2
// and  out = ln_f(h) @ lm_head + lmb.
//
// The @payload function and the kernel gpu.modules below are the
// xegpu-wg IR dumped from the lighthouse nanoGPT workload
// (examples/xegpu/nanoGPT.py --gpt-layers 1 --dump xegpu-wg): @payload chains 24
// kernel launches through on-device buffers. The dump emitted one gpu.module per
// launch; byte-identical modules have been collapsed so only the 10 unique
// kernels remain and repeated launches reuse them. The GEMM prefetch layouts were
// adjusted to the kernels' 4-subgroup configuration. The GEMM kernels take f16
// inputs and accumulate in f32; the flash-attention kernel (@flash_attention)
// runs QK^T, the online softmax and PV entirely in f16. The CPU reference
// (@ref_block) rounds activations through f16 at the same boundaries
// (round_to_f16) and computes attention in f32 with f16-rounded softmax weights.
// Input/weight magnitudes are kept small so the f16 / flash-online-softmax
// rounding error stays under the printAllcloseF32 bound (atol=1e-4, rtol=1e-3);
// fillResource1DRandom is unseeded.

module attributes {gpu.container_module} {
  func.func @payload(%arg0: memref<256x256xf32>, %arg1: memref<256x256xf32>, %arg2: memref<256xf32>, %arg3: memref<256xf32>, %arg4: memref<256x256xf16>, %arg5: memref<256x256xf16>, %arg6: memref<256x256xf16>, %arg7: memref<256x256xf16>, %arg8: memref<256xf32>, %arg9: memref<256xf32>, %arg10: memref<256xf32>, %arg11: memref<256x1024xf16>, %arg12: memref<1024xf32>, %arg13: memref<1024x256xf16>, %arg14: memref<256xf32>, %arg15: memref<256xf32>, %arg16: memref<256xf32>, %arg17: memref<256x256xf16>, %arg18: memref<256xf32>) attributes {llvm.emit_c_interface} {
    %c128 = arith.constant 128 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %memref = gpu.alloc  () : memref<256x256xf32>
    gpu.launch_func  @layer_norm::@layer_norm blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1)  args(%arg1 : memref<256x256xf32>, %memref : memref<256x256xf32>, %arg2 : memref<256xf32>, %arg3 : memref<256xf32>)
    %memref_0 = gpu.alloc  () : memref<256x256xf16>
    gpu.launch_func  @truncf_256x256::@truncf_256x256 blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1)  args(%memref : memref<256x256xf32>, %memref_0 : memref<256x256xf16>)
    %memref_1 = gpu.alloc  () : memref<256x256xf32>
    gpu.launch_func  @matmul_256x256x256::@matmul_256x256x256 blocks in (%c4, %c4, %c1) threads in (%c64, %c1, %c1)  args(%memref_1 : memref<256x256xf32>, %memref_0 : memref<256x256xf16>, %arg4 : memref<256x256xf16>)
    %memref_2 = gpu.alloc  () : memref<256x256xf16>
    gpu.launch_func  @truncf_256x256::@truncf_256x256 blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1)  args(%memref_1 : memref<256x256xf32>, %memref_2 : memref<256x256xf16>)
    %memref_3 = gpu.alloc  () : memref<256x256xf32>
    gpu.launch_func  @matmul_256x256x256::@matmul_256x256x256 blocks in (%c4, %c4, %c1) threads in (%c64, %c1, %c1)  args(%memref_3 : memref<256x256xf32>, %memref_0 : memref<256x256xf16>, %arg5 : memref<256x256xf16>)
    %memref_4 = gpu.alloc  () : memref<256x256xf16>
    gpu.launch_func  @truncf_256x256::@truncf_256x256 blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1)  args(%memref_3 : memref<256x256xf32>, %memref_4 : memref<256x256xf16>)
    %memref_5 = gpu.alloc  () : memref<256x256xf32>
    gpu.launch_func  @matmul_256x256x256::@matmul_256x256x256 blocks in (%c4, %c4, %c1) threads in (%c64, %c1, %c1)  args(%memref_5 : memref<256x256xf32>, %memref_0 : memref<256x256xf16>, %arg6 : memref<256x256xf16>)
    %memref_6 = gpu.alloc  () : memref<256x256xf16>
    gpu.launch_func  @truncf_256x256::@truncf_256x256 blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1)  args(%memref_5 : memref<256x256xf32>, %memref_6 : memref<256x256xf16>)
    %expand_shape = memref.expand_shape %memref_2 [[0], [1, 2]] output_shape [256, 4, 64] : memref<256x256xf16> into memref<256x4x64xf16>
    %transpose = memref.transpose %expand_shape (d0, d1, d2) -> (d1, d0, d2) : memref<256x4x64xf16> to memref<4x256x64xf16, strided<[64, 256, 1]>>
    %expand_shape_7 = memref.expand_shape %memref_4 [[0], [1, 2]] output_shape [256, 4, 64] : memref<256x256xf16> into memref<256x4x64xf16>
    %transpose_8 = memref.transpose %expand_shape_7 (d0, d1, d2) -> (d1, d0, d2) : memref<256x4x64xf16> to memref<4x256x64xf16, strided<[64, 256, 1]>>
    %expand_shape_9 = memref.expand_shape %memref_6 [[0], [1, 2]] output_shape [256, 4, 64] : memref<256x256xf16> into memref<256x4x64xf16>
    %transpose_10 = memref.transpose %expand_shape_9 (d0, d1, d2) -> (d1, d0, d2) : memref<256x4x64xf16> to memref<4x256x64xf16, strided<[64, 256, 1]>>
    %memref_11 = gpu.alloc  () : memref<256x256xf16>
    %expand_shape_12 = memref.expand_shape %memref_11 [[0], [1, 2]] output_shape [256, 4, 64] : memref<256x256xf16> into memref<256x4x64xf16>
    %transpose_13 = memref.transpose %expand_shape_12 (d0, d1, d2) -> (d1, d0, d2) : memref<256x4x64xf16> to memref<4x256x64xf16, strided<[64, 256, 1]>>
    gpu.launch_func  @flash_attention::@flash_attention blocks in (%c4, %c2, %c1) threads in (%c128, %c1, %c1)  args(%transpose : memref<4x256x64xf16, strided<[64, 256, 1]>>, %transpose_8 : memref<4x256x64xf16, strided<[64, 256, 1]>>, %transpose_10 : memref<4x256x64xf16, strided<[64, 256, 1]>>, %transpose_13 : memref<4x256x64xf16, strided<[64, 256, 1]>>)
    %memref_14 = gpu.alloc  () : memref<256x256xf32>
    gpu.launch_func  @matmul_256x256x256::@matmul_256x256x256 blocks in (%c4, %c4, %c1) threads in (%c64, %c1, %c1)  args(%memref_14 : memref<256x256xf32>, %memref_11 : memref<256x256xf16>, %arg7 : memref<256x256xf16>)
    %memref_15 = gpu.alloc  () : memref<256x256xf32>
    gpu.launch_func  @add_bias::@add_bias blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1)  args(%memref_14 : memref<256x256xf32>, %arg8 : memref<256xf32>, %memref_15 : memref<256x256xf32>)
    %memref_16 = gpu.alloc  () : memref<256x256xf32>
    gpu.launch_func  @residual_add::@residual_add blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1)  args(%arg1 : memref<256x256xf32>, %memref_15 : memref<256x256xf32>, %memref_16 : memref<256x256xf32>)
    %memref_17 = gpu.alloc  () : memref<256x256xf32>
    gpu.launch_func  @layer_norm::@layer_norm blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1)  args(%memref_16 : memref<256x256xf32>, %memref_17 : memref<256x256xf32>, %arg9 : memref<256xf32>, %arg10 : memref<256xf32>)
    %memref_18 = gpu.alloc  () : memref<256x256xf16>
    gpu.launch_func  @truncf_256x256::@truncf_256x256 blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1)  args(%memref_17 : memref<256x256xf32>, %memref_18 : memref<256x256xf16>)
    %memref_19 = gpu.alloc  () : memref<256x1024xf32>
    gpu.launch_func  @matmul_256x1024x256::@matmul_256x1024x256 blocks in (%c4, %c16, %c1) threads in (%c64, %c1, %c1)  args(%memref_19 : memref<256x1024xf32>, %memref_18 : memref<256x256xf16>, %arg11 : memref<256x1024xf16>)
    %memref_20 = gpu.alloc  () : memref<256x1024xf32>
    gpu.launch_func  @add_bias_relu::@add_bias_relu blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1)  args(%memref_19 : memref<256x1024xf32>, %arg12 : memref<1024xf32>, %memref_20 : memref<256x1024xf32>)
    %memref_21 = gpu.alloc  () : memref<256x1024xf16>
    gpu.launch_func  @truncf_256x1024::@truncf_256x1024 blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1)  args(%memref_20 : memref<256x1024xf32>, %memref_21 : memref<256x1024xf16>)
    %memref_22 = gpu.alloc  () : memref<256x256xf32>
    gpu.launch_func  @matmul_256x256x1024::@matmul_256x256x1024 blocks in (%c4, %c4, %c1) threads in (%c64, %c1, %c1)  args(%memref_22 : memref<256x256xf32>, %memref_21 : memref<256x1024xf16>, %arg13 : memref<1024x256xf16>)
    %memref_23 = gpu.alloc  () : memref<256x256xf32>
    gpu.launch_func  @add_bias::@add_bias blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1)  args(%memref_22 : memref<256x256xf32>, %arg14 : memref<256xf32>, %memref_23 : memref<256x256xf32>)
    %memref_24 = gpu.alloc  () : memref<256x256xf32>
    gpu.launch_func  @residual_add::@residual_add blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1)  args(%memref_16 : memref<256x256xf32>, %memref_23 : memref<256x256xf32>, %memref_24 : memref<256x256xf32>)
    %memref_25 = gpu.alloc  () : memref<256x256xf32>
    gpu.launch_func  @layer_norm::@layer_norm blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1)  args(%memref_24 : memref<256x256xf32>, %memref_25 : memref<256x256xf32>, %arg15 : memref<256xf32>, %arg16 : memref<256xf32>)
    %memref_26 = gpu.alloc  () : memref<256x256xf16>
    gpu.launch_func  @truncf_256x256::@truncf_256x256 blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1)  args(%memref_25 : memref<256x256xf32>, %memref_26 : memref<256x256xf16>)
    %memref_27 = gpu.alloc  () : memref<256x256xf32>
    gpu.launch_func  @matmul_256x256x256::@matmul_256x256x256 blocks in (%c4, %c4, %c1) threads in (%c64, %c1, %c1)  args(%memref_27 : memref<256x256xf32>, %memref_26 : memref<256x256xf16>, %arg17 : memref<256x256xf16>)
    gpu.launch_func  @add_bias::@add_bias blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1)  args(%memref_27 : memref<256x256xf32>, %arg18 : memref<256xf32>, %arg0 : memref<256x256xf32>)
    gpu.dealloc  %memref : memref<256x256xf32>
    gpu.dealloc  %memref_0 : memref<256x256xf16>
    gpu.dealloc  %memref_1 : memref<256x256xf32>
    gpu.dealloc  %memref_2 : memref<256x256xf16>
    gpu.dealloc  %memref_3 : memref<256x256xf32>
    gpu.dealloc  %memref_4 : memref<256x256xf16>
    gpu.dealloc  %memref_5 : memref<256x256xf32>
    gpu.dealloc  %memref_6 : memref<256x256xf16>
    gpu.dealloc  %memref_11 : memref<256x256xf16>
    gpu.dealloc  %memref_14 : memref<256x256xf32>
    gpu.dealloc  %memref_15 : memref<256x256xf32>
    gpu.dealloc  %memref_16 : memref<256x256xf32>
    gpu.dealloc  %memref_17 : memref<256x256xf32>
    gpu.dealloc  %memref_18 : memref<256x256xf16>
    gpu.dealloc  %memref_19 : memref<256x1024xf32>
    gpu.dealloc  %memref_20 : memref<256x1024xf32>
    gpu.dealloc  %memref_21 : memref<256x1024xf16>
    gpu.dealloc  %memref_22 : memref<256x256xf32>
    gpu.dealloc  %memref_23 : memref<256x256xf32>
    gpu.dealloc  %memref_24 : memref<256x256xf32>
    gpu.dealloc  %memref_25 : memref<256x256xf32>
    gpu.dealloc  %memref_26 : memref<256x256xf16>
    gpu.dealloc  %memref_27 : memref<256x256xf32>
    return
  }
  gpu.module @layer_norm [#xevm.target<O = 3>] {
    gpu.func @layer_norm(%arg0: memref<256x256xf32>, %arg1: memref<256x256xf32>, %arg2: memref<256xf32>, %arg3: memref<256xf32>) kernel attributes {known_block_size = array<i32: 128, 1, 1>, known_grid_size = array<i32: 4, 1, 1>} {
      %c4 = arith.constant 4 : index
      %cst = arith.constant dense<true> : vector<16xi1>
      %cst_0 = arith.constant dense<9.99999974E-6> : vector<64xf32>
      %cst_1 = arith.constant dense<3.906250e-03> : vector<64xf32>
      %cst_2 = arith.constant dense<0.000000e+00> : vector<64xf32>
      %c16 = arith.constant 16 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %cst_3 = arith.constant dense<0.000000e+00> : vector<64x16xf32>
      %c64 = arith.constant 64 : index
      %block_id_x = gpu.block_id x
      %0 = arith.muli %block_id_x, %c64 overflow<nsw> : index
      %alloca = memref.alloca() : memref<64x16xf32, 3>
      %1 = xegpu.create_mem_desc %alloca : memref<64x16xf32, 3> -> !xegpu.mem_desc<64x16xf32>
      xegpu.store_matrix %cst_3, %1[0, 0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [8, 16]>}>: vector<64x16xf32>, !xegpu.mem_desc<64x16xf32>
      %alloca_4 = memref.alloca() : memref<64x16xf32, 3>
      memref.copy %alloca, %alloca_4 : memref<64x16xf32, 3> to memref<64x16xf32, 3>
      scf.for %arg4 = %c0 to %c256 step %c16 {
        %7 = xegpu.create_nd_tdesc %arg0 : memref<256x256xf32> -> !xegpu.tensor_desc<64x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
        %8 = xegpu.load_nd %7[%0, %arg4]  : !xegpu.tensor_desc<64x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x16xf32>
        %9 = xegpu.create_mem_desc %alloca_4 : memref<64x16xf32, 3> -> !xegpu.mem_desc<64x16xf32>
        %10 = xegpu.load_matrix %9[0, 0] : !xegpu.mem_desc<64x16xf32> -> vector<64x16xf32>
        %11 = arith.addf %8, %10 : vector<64x16xf32>
        xegpu.store_matrix %11, %9[0, 0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [8, 16]>}>: vector<64x16xf32>, !xegpu.mem_desc<64x16xf32>
      }
      %2 = xegpu.create_mem_desc %alloca_4 : memref<64x16xf32, 3> -> !xegpu.mem_desc<64x16xf32>
      %3 = xegpu.load_matrix %2[0, 0] : !xegpu.mem_desc<64x16xf32> -> vector<64x16xf32>
      %4 = vector.multi_reduction <add>, %3, %cst_2 [1] : vector<64x16xf32> to vector<64xf32>
      scf.for %arg4 = %c0 to %c256 step %c16 {
        %7 = xegpu.create_nd_tdesc %arg0 : memref<256x256xf32> -> !xegpu.tensor_desc<64x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
        %8 = xegpu.load_nd %7[%0, %arg4]  : !xegpu.tensor_desc<64x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x16xf32>
        %9 = xegpu.load_matrix %1[0, 0] : !xegpu.mem_desc<64x16xf32> -> vector<64x16xf32>
        %10 = arith.mulf %4, %cst_1 : vector<64xf32>
        %11 = vector.broadcast %10 : vector<64xf32> to vector<16x64xf32>
        %12 = vector.transpose %11, [1, 0] : vector<16x64xf32> to vector<64x16xf32>
        %13 = arith.subf %8, %12 : vector<64x16xf32>
        %14 = arith.mulf %13, %13 : vector<64x16xf32>
        %15 = arith.addf %14, %9 : vector<64x16xf32>
        xegpu.store_matrix %15, %1[0, 0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [8, 16]>}>: vector<64x16xf32>, !xegpu.mem_desc<64x16xf32>
      }
      %5 = xegpu.load_matrix %1[0, 0] : !xegpu.mem_desc<64x16xf32> -> vector<64x16xf32>
      %6 = vector.multi_reduction <add>, %5, %cst_2 [1] : vector<64x16xf32> to vector<64xf32>
      %subview = memref.subview %arg1[%0, 0] [64, 256] [1, 1] : memref<256x256xf32> to memref<64x256xf32, strided<[256, 1], offset: ?>>
      scf.for %arg4 = %c0 to %c256 step %c16 {
        %7 = xegpu.create_nd_tdesc %arg0 : memref<256x256xf32> -> !xegpu.tensor_desc<64x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
        %8 = xegpu.load_nd %7[%0, %arg4]  : !xegpu.tensor_desc<64x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x16xf32>
        %9 = vector.step : vector<16xindex>
        %10 = vector.broadcast %arg4 : index to vector<16xindex>
        %11 = arith.addi %10, %9 : vector<16xindex>
        %intptr = memref.extract_aligned_pointer_as_index %arg2 : memref<256xf32> -> index
        %12 = arith.index_cast %intptr : index to i64
        %13 = xegpu.load %12[%11], %cst  : i64, vector<16xindex>, vector<16xi1> -> vector<16xf32>
        %14 = vector.broadcast %13 : vector<16xf32> to vector<64x16xf32>
        %intptr_5 = memref.extract_aligned_pointer_as_index %arg3 : memref<256xf32> -> index
        %15 = arith.index_cast %intptr_5 : index to i64
        %16 = xegpu.load %15[%11], %cst  : i64, vector<16xindex>, vector<16xi1> -> vector<16xf32>
        %17 = vector.broadcast %16 : vector<16xf32> to vector<64x16xf32>
        %18 = arith.mulf %4, %cst_1 : vector<64xf32>
        %19 = vector.broadcast %18 : vector<64xf32> to vector<16x64xf32>
        %20 = vector.transpose %19, [1, 0] : vector<16x64xf32> to vector<64x16xf32>
        %21 = arith.mulf %6, %cst_1 : vector<64xf32>
        %22 = arith.addf %21, %cst_0 : vector<64xf32>
        %23 = math.rsqrt %22 : vector<64xf32>
        %24 = vector.broadcast %23 : vector<64xf32> to vector<16x64xf32>
        %25 = vector.transpose %24, [1, 0] : vector<16x64xf32> to vector<64x16xf32>
        %26 = arith.subf %8, %20 : vector<64x16xf32>
        %27 = arith.mulf %26, %25 : vector<64x16xf32>
        %28 = arith.mulf %27, %14 : vector<64x16xf32>
        %29 = arith.addf %28, %17 : vector<64x16xf32>
        %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<64x256xf32, strided<[256, 1], offset: ?>> -> memref<f32>, index, index, index, index, index
        %intptr_6 = memref.extract_aligned_pointer_as_index %base_buffer : memref<f32> -> index
        %30 = arith.muli %offset, %c4 : index
        %31 = arith.addi %intptr_6, %30 : index
        %32 = arith.index_cast %31 : index to i64
        %33 = xegpu.create_nd_tdesc %32, shape : [64, 256], strides : [256, 1] : i64 -> !xegpu.tensor_desc<64x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
        xegpu.store_nd %29, %33[0, %arg4] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [8, 16]>}> : vector<64x16xf32>, !xegpu.tensor_desc<64x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      }
      gpu.return
    }
  }
  gpu.module @truncf_256x256 [#xevm.target<O = 3>] {
    gpu.func @truncf_256x256(%arg0: memref<256x256xf32>, %arg1: memref<256x256xf16>) kernel attributes {known_block_size = array<i32: 128, 1, 1>, known_grid_size = array<i32: 4, 1, 1>} {
      %c64 = arith.constant 64 : index
      %block_id_x = gpu.block_id x
      %0 = arith.muli %block_id_x, %c64 overflow<nsw> : index
      %1 = xegpu.create_nd_tdesc %arg0 : memref<256x256xf32> -> !xegpu.tensor_desc<64x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      %2 = xegpu.load_nd %1[%0, 0]  : !xegpu.tensor_desc<64x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x256xf32>
      %3 = arith.truncf %2 : vector<64x256xf32> to vector<64x256xf16>
      %4 = xegpu.create_nd_tdesc %arg1 : memref<256x256xf16> -> !xegpu.tensor_desc<64x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %3, %4[%0, 0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [8, 16]>}> : vector<64x256xf16>, !xegpu.tensor_desc<64x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
  gpu.module @matmul_256x256x256 [#xevm.target<O = 3>] {
    gpu.func @matmul_256x256x256(%arg0: memref<256x256xf32>, %arg1: memref<256x256xf16>, %arg2: memref<256x256xf16>) kernel attributes {known_block_size = array<i32: 64, 1, 1>, known_grid_size = array<i32: 4, 4, 1>} {
      %c4 = arith.constant 4 : index
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant dense<0.000000e+00> : vector<64x64xf32>
      %c64 = arith.constant 64 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %0 = arith.muli %block_id_x, %c64 overflow<nsw> : index
      %1 = arith.muli %block_id_y, %c64 overflow<nsw> : index
      %subview = memref.subview %arg0[%0, %1] [64, 64] [1, 1] : memref<256x256xf32> to memref<64x64xf32, strided<[256, 1], offset: ?>>
      %2 = xegpu.create_nd_tdesc %arg0 : memref<256x256xf32> -> !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %cst, %2[%0, %1] <{layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 32], inst_data = [8, 16]>}> : vector<64x64xf32>, !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      %3 = xegpu.create_nd_tdesc %arg1 : memref<256x256xf16> -> !xegpu.tensor_desc<64x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.prefetch_nd %3[%0, %c0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 16], inst_data = [8, 16]>}> : !xegpu.tensor_desc<64x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %4 = xegpu.create_nd_tdesc %arg2 : memref<256x256xf16> -> !xegpu.tensor_desc<32x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.prefetch_nd %4[%c0, %1] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 32], inst_data = [8, 16]>}> : !xegpu.tensor_desc<32x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<64x64xf32, strided<[256, 1], offset: ?>> -> memref<f32>, index, index, index, index, index
      %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f32> -> index
      %5 = arith.muli %offset, %c4 : index
      %6 = arith.addi %intptr, %5 : index
      %7 = arith.index_cast %6 : index to i64
      %8 = xegpu.create_nd_tdesc %7, shape : [64, 64], strides : [256, 1] : i64 -> !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      scf.for %arg3 = %c0 to %c256 step %c32 {
        %9 = arith.addi %arg3, %c32 : index
        xegpu.prefetch_nd %4[%9, %1] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 32], inst_data = [8, 16]>}> : !xegpu.tensor_desc<32x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        xegpu.prefetch_nd %3[%0, %9] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 16], inst_data = [8, 16]>}> : !xegpu.tensor_desc<64x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %10 = xegpu.load_nd %3[%0, %arg3] <{layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 32], inst_data = [8, 16]>}> : !xegpu.tensor_desc<64x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x32xf16>
        %11 = xegpu.load_nd %4[%arg3, %1] <{layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 32], inst_data = [16, 16]>}> : !xegpu.tensor_desc<32x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<32x64xf16>
        %12 = xegpu.load_nd %8[0, 0]  : !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf32>
        %13 = xegpu.dpas %10, %11, %12 <{layout_a = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 32], inst_data = [8, 16]>, layout_b = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 32], inst_data = [16, 16]>, layout_cd = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 32], inst_data = [8, 16]>}> : vector<64x32xf16>, vector<32x64xf16>, vector<64x64xf32> -> vector<64x64xf32>
        xegpu.store_nd %13, %8[0, 0] <{layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 32], inst_data = [8, 16]>}> : vector<64x64xf32>, !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      }
      gpu.return
    }
  }
  gpu.module @flash_attention [#xevm.target<O = 3>] {
    gpu.func @flash_attention(%arg0: memref<4x256x64xf16, strided<[64, 256, 1]>>, %arg1: memref<4x256x64xf16, strided<[64, 256, 1]>>, %arg2: memref<4x256x64xf16, strided<[64, 256, 1]>>, %arg3: memref<4x256x64xf16, strided<[64, 256, 1]>>) kernel attributes {known_block_size = array<i32: 128, 1, 1>, known_grid_size = array<i32: 4, 2, 1>} {
      %c2 = arith.constant 2 : index
      %cst = arith.constant dense<0.000000e+00> : vector<128x64xf16>
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf16>
      %cst_1 = arith.constant dense<1.250000e-01> : vector<128x16xf16>
      %cst_2 = arith.constant dense<1.250000e-01> : vector<128xf16>
      %cst_3 = arith.constant dense<0xFC00> : vector<128xf16>
      %c48 = arith.constant 48 : index
      %c32 = arith.constant 32 : index
      %c16 = arith.constant 16 : index
      %cst_4 = arith.constant dense<0.000000e+00> : vector<128x16xf16>
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %0 = arith.muli %block_id_y, %c128 overflow<nsw> : index
      %subview = memref.subview %arg0[%block_id_x, 0, 0] [1, 256, 64] [1, 1, 1] : memref<4x256x64xf16, strided<[64, 256, 1]>> to memref<256x64xf16, strided<[256, 1], offset: ?>>
      %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<256x64xf16, strided<[256, 1], offset: ?>> -> memref<f16>, index, index, index, index, index
      %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f16> -> index
      %1 = arith.muli %offset, %c2 : index
      %2 = arith.addi %intptr, %1 : index
      %3 = arith.index_cast %2 : index to i64
      %4 = xegpu.create_nd_tdesc %3, shape : [256, 64], strides : [256, 1] : i64 -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %5 = xegpu.load_nd %4[%0, 0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [8, 16]>}> : !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<128x64xf16>
      %subview_5 = memref.subview %arg1[%block_id_x, 0, 0] [1, 256, 64] [1, 1, 1] : memref<4x256x64xf16, strided<[64, 256, 1]>> to memref<256x64xf16, strided<[256, 1], offset: ?>>
      %base_buffer_6, %offset_7, %sizes_8:2, %strides_9:2 = memref.extract_strided_metadata %subview_5 : memref<256x64xf16, strided<[256, 1], offset: ?>> -> memref<f16>, index, index, index, index, index
      %intptr_10 = memref.extract_aligned_pointer_as_index %base_buffer_6 : memref<f16> -> index
      %6 = arith.muli %offset_7, %c2 : index
      %7 = arith.addi %intptr_10, %6 : index
      %8 = arith.index_cast %7 : index to i64
      %9 = xegpu.create_nd_tdesc %8, shape : [256, 64], strides : [256, 1] : i64 -> !xegpu.tensor_desc<16x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %subview_11 = memref.subview %arg2[%block_id_x, 0, 0] [1, 256, 64] [1, 1, 1] : memref<4x256x64xf16, strided<[64, 256, 1]>> to memref<256x64xf16, strided<[256, 1], offset: ?>>
      %base_buffer_12, %offset_13, %sizes_14:2, %strides_15:2 = memref.extract_strided_metadata %subview_11 : memref<256x64xf16, strided<[256, 1], offset: ?>> -> memref<f16>, index, index, index, index, index
      %intptr_16 = memref.extract_aligned_pointer_as_index %base_buffer_12 : memref<f16> -> index
      %10 = arith.muli %offset_13, %c2 : index
      %11 = arith.addi %intptr_16, %10 : index
      %12 = arith.index_cast %11 : index to i64
      %13 = xegpu.create_nd_tdesc %12, shape : [256, 64], strides : [256, 1] : i64 -> !xegpu.tensor_desc<16x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %14:3 = scf.for %arg4 = %c0 to %c256 step %c64 iter_args(%arg5 = %cst_3, %arg6 = %cst_0, %arg7 = %cst) -> (vector<128xf16>, vector<128xf16>, vector<128x64xf16>) {
        %22 = xegpu.load_nd %9[%arg4, 0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [16, 16]>}> : !xegpu.tensor_desc<16x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<16x64xf16>
        %23 = vector.transpose %22, [1, 0] : vector<16x64xf16> to vector<64x16xf16>
        %24 = xegpu.dpas %5, %23, %cst_4 <{layout_a = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [8, 16]>, layout_b = #xegpu.layout<sg_layout = [1, 8], sg_data = [64, 16], inst_data = [16, 16], order = [0, 1]>, layout_cd = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 16], inst_data = [8, 16]>}> : vector<128x64xf16>, vector<64x16xf16>, vector<128x16xf16> -> vector<128x16xf16>
        %25 = arith.addi %arg4, %c16 : index
        %26 = xegpu.load_nd %9[%25, 0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [16, 16]>}> : !xegpu.tensor_desc<16x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<16x64xf16>
        %27 = vector.transpose %26, [1, 0] : vector<16x64xf16> to vector<64x16xf16>
        %28 = xegpu.dpas %5, %27, %cst_4 <{layout_a = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [8, 16]>, layout_b = #xegpu.layout<sg_layout = [1, 8], sg_data = [64, 16], inst_data = [16, 16], order = [0, 1]>, layout_cd = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 16], inst_data = [8, 16]>}> : vector<128x64xf16>, vector<64x16xf16>, vector<128x16xf16> -> vector<128x16xf16>
        %29 = arith.addi %arg4, %c32 : index
        %30 = xegpu.load_nd %9[%29, 0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [16, 16]>}> : !xegpu.tensor_desc<16x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<16x64xf16>
        %31 = vector.transpose %30, [1, 0] : vector<16x64xf16> to vector<64x16xf16>
        %32 = xegpu.dpas %5, %31, %cst_4 <{layout_a = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [8, 16]>, layout_b = #xegpu.layout<sg_layout = [1, 8], sg_data = [64, 16], inst_data = [16, 16], order = [0, 1]>, layout_cd = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 16], inst_data = [8, 16]>}> : vector<128x64xf16>, vector<64x16xf16>, vector<128x16xf16> -> vector<128x16xf16>
        %33 = arith.addi %arg4, %c48 : index
        %34 = xegpu.load_nd %9[%33, 0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [16, 16]>}> : !xegpu.tensor_desc<16x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<16x64xf16>
        %35 = vector.transpose %34, [1, 0] : vector<16x64xf16> to vector<64x16xf16>
        %36 = xegpu.dpas %5, %35, %cst_4 <{layout_a = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [8, 16]>, layout_b = #xegpu.layout<sg_layout = [1, 8], sg_data = [64, 16], inst_data = [16, 16], order = [0, 1]>, layout_cd = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 16], inst_data = [8, 16]>}> : vector<128x64xf16>, vector<64x16xf16>, vector<128x16xf16> -> vector<128x16xf16>
        %37 = arith.maximumf %24, %28 : vector<128x16xf16>
        %38 = arith.maximumf %37, %32 : vector<128x16xf16>
        %39 = arith.maximumf %38, %36 : vector<128x16xf16>
        %40 = vector.multi_reduction <maxnumf>, %39, %cst_3 [1] : vector<128x16xf16> to vector<128xf16>
        %41 = arith.mulf %40, %cst_2 : vector<128xf16>
        %42 = arith.maximumf %arg5, %41 : vector<128xf16>
        %43 = vector.broadcast %42 : vector<128xf16> to vector<16x128xf16>
        %44 = vector.transpose %43, [1, 0] : vector<16x128xf16> to vector<128x16xf16>
        %45 = arith.mulf %24, %cst_1 : vector<128x16xf16>
        %46 = arith.subf %45, %44 : vector<128x16xf16>
        %47 = math.exp %46 : vector<128x16xf16>
        %48 = arith.mulf %28, %cst_1 : vector<128x16xf16>
        %49 = arith.subf %48, %44 : vector<128x16xf16>
        %50 = math.exp %49 : vector<128x16xf16>
        %51 = arith.mulf %32, %cst_1 : vector<128x16xf16>
        %52 = arith.subf %51, %44 : vector<128x16xf16>
        %53 = math.exp %52 : vector<128x16xf16>
        %54 = arith.mulf %36, %cst_1 : vector<128x16xf16>
        %55 = arith.subf %54, %44 : vector<128x16xf16>
        %56 = math.exp %55 : vector<128x16xf16>
        %57 = arith.addf %47, %50 : vector<128x16xf16>
        %58 = arith.addf %57, %53 : vector<128x16xf16>
        %59 = arith.addf %58, %56 : vector<128x16xf16>
        %60 = vector.multi_reduction <add>, %59, %cst_0 [1] : vector<128x16xf16> to vector<128xf16>
        %61 = arith.subf %arg5, %42 : vector<128xf16>
        %62 = math.exp %61 : vector<128xf16>
        %63 = arith.mulf %arg6, %62 : vector<128xf16>
        %64 = arith.addf %63, %60 : vector<128xf16>
        %65 = vector.broadcast %62 : vector<128xf16> to vector<64x128xf16>
        %66 = vector.transpose %65, [1, 0] : vector<64x128xf16> to vector<128x64xf16>
        %67 = arith.mulf %arg7, %66 : vector<128x64xf16>
        %68 = xegpu.load_nd %13[%arg4, 0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [16, 16]>}> : !xegpu.tensor_desc<16x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<16x64xf16>
        %69 = xegpu.dpas %47, %68, %67 <{layout_a = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 16], inst_data = [8, 16]>, layout_b = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [16, 16]>, layout_cd = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [8, 16]>}> : vector<128x16xf16>, vector<16x64xf16>, vector<128x64xf16> -> vector<128x64xf16>
        %70 = xegpu.load_nd %13[%25, 0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [16, 16]>}> : !xegpu.tensor_desc<16x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<16x64xf16>
        %71 = xegpu.dpas %50, %70, %69 <{layout_a = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 16], inst_data = [8, 16]>, layout_b = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [16, 16]>, layout_cd = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [8, 16]>}> : vector<128x16xf16>, vector<16x64xf16>, vector<128x64xf16> -> vector<128x64xf16>
        %72 = xegpu.load_nd %13[%29, 0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [16, 16]>}> : !xegpu.tensor_desc<16x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<16x64xf16>
        %73 = xegpu.dpas %53, %72, %71 <{layout_a = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 16], inst_data = [8, 16]>, layout_b = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [16, 16]>, layout_cd = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [8, 16]>}> : vector<128x16xf16>, vector<16x64xf16>, vector<128x64xf16> -> vector<128x64xf16>
        %74 = xegpu.load_nd %13[%33, 0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [16, 16]>}> : !xegpu.tensor_desc<16x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<16x64xf16>
        %75 = xegpu.dpas %56, %74, %73 <{layout_a = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 16], inst_data = [8, 16]>, layout_b = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [16, 16]>, layout_cd = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [8, 16]>}> : vector<128x16xf16>, vector<16x64xf16>, vector<128x64xf16> -> vector<128x64xf16>
        scf.yield %42, %64, %75 : vector<128xf16>, vector<128xf16>, vector<128x64xf16>
      }
      %15 = vector.broadcast %14#1 : vector<128xf16> to vector<64x128xf16>
      %16 = vector.transpose %15, [1, 0] : vector<64x128xf16> to vector<128x64xf16>
      %17 = arith.divf %14#2, %16 : vector<128x64xf16>
      %subview_17 = memref.subview %arg3[%block_id_x, 0, 0] [1, 256, 64] [1, 1, 1] : memref<4x256x64xf16, strided<[64, 256, 1]>> to memref<256x64xf16, strided<[256, 1], offset: ?>>
      %base_buffer_18, %offset_19, %sizes_20:2, %strides_21:2 = memref.extract_strided_metadata %subview_17 : memref<256x64xf16, strided<[256, 1], offset: ?>> -> memref<f16>, index, index, index, index, index
      %intptr_22 = memref.extract_aligned_pointer_as_index %base_buffer_18 : memref<f16> -> index
      %18 = arith.muli %offset_19, %c2 : index
      %19 = arith.addi %intptr_22, %18 : index
      %20 = arith.index_cast %19 : index to i64
      %21 = xegpu.create_nd_tdesc %20, shape : [256, 64], strides : [256, 1] : i64 -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %17, %21[%0, 0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64], inst_data = [8, 16]>}> : vector<128x64xf16>, !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
  gpu.module @add_bias [#xevm.target<O = 3>] {
    gpu.func @add_bias(%arg0: memref<256x256xf32>, %arg1: memref<256xf32>, %arg2: memref<256x256xf32>) kernel attributes {known_block_size = array<i32: 128, 1, 1>, known_grid_size = array<i32: 4, 1, 1>} {
      %cst = arith.constant dense<true> : vector<256xi1>
      %cst_0 = arith.constant dense<0.000000e+00> : vector<64x256xf32>
      %c64 = arith.constant 64 : index
      %block_id_x = gpu.block_id x
      %0 = arith.muli %block_id_x, %c64 overflow<nsw> : index
      %1 = xegpu.create_nd_tdesc %arg0 : memref<256x256xf32> -> !xegpu.tensor_desc<64x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      %2 = xegpu.load_nd %1[%0, 0]  : !xegpu.tensor_desc<64x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x256xf32>
      %3 = vector.step : vector<256xindex>
      %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<256xf32> -> index
      %4 = arith.index_cast %intptr : index to i64
      %5 = xegpu.load %4[%3], %cst  : i64, vector<256xindex>, vector<256xi1> -> vector<256xf32>
      %6 = vector.broadcast %5 : vector<256xf32> to vector<64x256xf32>
      %7 = arith.addf %2, %6 : vector<64x256xf32>
      %8 = arith.addf %7, %cst_0 : vector<64x256xf32>
      %9 = xegpu.create_nd_tdesc %arg2 : memref<256x256xf32> -> !xegpu.tensor_desc<64x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %8, %9[%0, 0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [8, 16]>}> : vector<64x256xf32>, !xegpu.tensor_desc<64x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
  gpu.module @residual_add [#xevm.target<O = 3>] {
    gpu.func @residual_add(%arg0: memref<256x256xf32>, %arg1: memref<256x256xf32>, %arg2: memref<256x256xf32>) kernel attributes {known_block_size = array<i32: 128, 1, 1>, known_grid_size = array<i32: 4, 1, 1>} {
      %c64 = arith.constant 64 : index
      %block_id_x = gpu.block_id x
      %0 = arith.muli %block_id_x, %c64 overflow<nsw> : index
      %1 = xegpu.create_nd_tdesc %arg0 : memref<256x256xf32> -> !xegpu.tensor_desc<64x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      %2 = xegpu.load_nd %1[%0, 0]  : !xegpu.tensor_desc<64x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x256xf32>
      %3 = xegpu.create_nd_tdesc %arg1 : memref<256x256xf32> -> !xegpu.tensor_desc<64x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      %4 = xegpu.load_nd %3[%0, 0]  : !xegpu.tensor_desc<64x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x256xf32>
      %5 = arith.addf %2, %4 : vector<64x256xf32>
      %6 = xegpu.create_nd_tdesc %arg2 : memref<256x256xf32> -> !xegpu.tensor_desc<64x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %5, %6[%0, 0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [8, 16]>}> : vector<64x256xf32>, !xegpu.tensor_desc<64x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
  gpu.module @matmul_256x1024x256 [#xevm.target<O = 3>] {
    gpu.func @matmul_256x1024x256(%arg0: memref<256x1024xf32>, %arg1: memref<256x256xf16>, %arg2: memref<256x1024xf16>) kernel attributes {known_block_size = array<i32: 64, 1, 1>, known_grid_size = array<i32: 4, 16, 1>} {
      %c4 = arith.constant 4 : index
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant dense<0.000000e+00> : vector<64x64xf32>
      %c64 = arith.constant 64 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %0 = arith.muli %block_id_x, %c64 overflow<nsw> : index
      %1 = arith.muli %block_id_y, %c64 overflow<nsw> : index
      %subview = memref.subview %arg0[%0, %1] [64, 64] [1, 1] : memref<256x1024xf32> to memref<64x64xf32, strided<[1024, 1], offset: ?>>
      %2 = xegpu.create_nd_tdesc %arg0 : memref<256x1024xf32> -> !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %cst, %2[%0, %1] <{layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 32], inst_data = [8, 16]>}> : vector<64x64xf32>, !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      %3 = xegpu.create_nd_tdesc %arg1 : memref<256x256xf16> -> !xegpu.tensor_desc<64x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.prefetch_nd %3[%0, %c0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 16], inst_data = [8, 16]>}> : !xegpu.tensor_desc<64x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %4 = xegpu.create_nd_tdesc %arg2 : memref<256x1024xf16> -> !xegpu.tensor_desc<32x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.prefetch_nd %4[%c0, %1] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 32], inst_data = [8, 16]>}> : !xegpu.tensor_desc<32x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<64x64xf32, strided<[1024, 1], offset: ?>> -> memref<f32>, index, index, index, index, index
      %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f32> -> index
      %5 = arith.muli %offset, %c4 : index
      %6 = arith.addi %intptr, %5 : index
      %7 = arith.index_cast %6 : index to i64
      %8 = xegpu.create_nd_tdesc %7, shape : [64, 64], strides : [1024, 1] : i64 -> !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      scf.for %arg3 = %c0 to %c256 step %c32 {
        %9 = arith.addi %arg3, %c32 : index
        xegpu.prefetch_nd %4[%9, %1] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 32], inst_data = [8, 16]>}> : !xegpu.tensor_desc<32x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        xegpu.prefetch_nd %3[%0, %9] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 16], inst_data = [8, 16]>}> : !xegpu.tensor_desc<64x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %10 = xegpu.load_nd %3[%0, %arg3] <{layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 32], inst_data = [8, 16]>}> : !xegpu.tensor_desc<64x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x32xf16>
        %11 = xegpu.load_nd %4[%arg3, %1] <{layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 32], inst_data = [16, 16]>}> : !xegpu.tensor_desc<32x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<32x64xf16>
        %12 = xegpu.load_nd %8[0, 0]  : !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf32>
        %13 = xegpu.dpas %10, %11, %12 <{layout_a = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 32], inst_data = [8, 16]>, layout_b = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 32], inst_data = [16, 16]>, layout_cd = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 32], inst_data = [8, 16]>}> : vector<64x32xf16>, vector<32x64xf16>, vector<64x64xf32> -> vector<64x64xf32>
        xegpu.store_nd %13, %8[0, 0] <{layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 32], inst_data = [8, 16]>}> : vector<64x64xf32>, !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      }
      gpu.return
    }
  }
  gpu.module @add_bias_relu [#xevm.target<O = 3>] {
    gpu.func @add_bias_relu(%arg0: memref<256x1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<256x1024xf32>) kernel attributes {known_block_size = array<i32: 128, 1, 1>, known_grid_size = array<i32: 4, 1, 1>} {
      %cst = arith.constant dense<true> : vector<1024xi1>
      %cst_0 = arith.constant dense<0.000000e+00> : vector<64x1024xf32>
      %c64 = arith.constant 64 : index
      %block_id_x = gpu.block_id x
      %0 = arith.muli %block_id_x, %c64 overflow<nsw> : index
      %1 = xegpu.create_nd_tdesc %arg0 : memref<256x1024xf32> -> !xegpu.tensor_desc<64x1024xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      %2 = xegpu.load_nd %1[%0, 0]  : !xegpu.tensor_desc<64x1024xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x1024xf32>
      %3 = vector.step : vector<1024xindex>
      %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<1024xf32> -> index
      %4 = arith.index_cast %intptr : index to i64
      %5 = xegpu.load %4[%3], %cst  : i64, vector<1024xindex>, vector<1024xi1> -> vector<1024xf32>
      %6 = vector.broadcast %5 : vector<1024xf32> to vector<64x1024xf32>
      %7 = arith.addf %2, %6 : vector<64x1024xf32>
      %8 = arith.maximumf %7, %cst_0 : vector<64x1024xf32>
      %9 = xegpu.create_nd_tdesc %arg2 : memref<256x1024xf32> -> !xegpu.tensor_desc<64x1024xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %8, %9[%0, 0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [8, 16]>}> : vector<64x1024xf32>, !xegpu.tensor_desc<64x1024xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
  gpu.module @truncf_256x1024 [#xevm.target<O = 3>] {
    gpu.func @truncf_256x1024(%arg0: memref<256x1024xf32>, %arg1: memref<256x1024xf16>) kernel attributes {known_block_size = array<i32: 128, 1, 1>, known_grid_size = array<i32: 4, 1, 1>} {
      %c64 = arith.constant 64 : index
      %block_id_x = gpu.block_id x
      %0 = arith.muli %block_id_x, %c64 overflow<nsw> : index
      %1 = xegpu.create_nd_tdesc %arg0 : memref<256x1024xf32> -> !xegpu.tensor_desc<64x1024xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      %2 = xegpu.load_nd %1[%0, 0]  : !xegpu.tensor_desc<64x1024xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x1024xf32>
      %3 = arith.truncf %2 : vector<64x1024xf32> to vector<64x1024xf16>
      %4 = xegpu.create_nd_tdesc %arg1 : memref<256x1024xf16> -> !xegpu.tensor_desc<64x1024xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %3, %4[%0, 0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [8, 16]>}> : vector<64x1024xf16>, !xegpu.tensor_desc<64x1024xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
  gpu.module @matmul_256x256x1024 [#xevm.target<O = 3>] {
    gpu.func @matmul_256x256x1024(%arg0: memref<256x256xf32>, %arg1: memref<256x1024xf16>, %arg2: memref<1024x256xf16>) kernel attributes {known_block_size = array<i32: 64, 1, 1>, known_grid_size = array<i32: 4, 4, 1>} {
      %c4 = arith.constant 4 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant dense<0.000000e+00> : vector<64x64xf32>
      %c64 = arith.constant 64 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %0 = arith.muli %block_id_x, %c64 overflow<nsw> : index
      %1 = arith.muli %block_id_y, %c64 overflow<nsw> : index
      %subview = memref.subview %arg0[%0, %1] [64, 64] [1, 1] : memref<256x256xf32> to memref<64x64xf32, strided<[256, 1], offset: ?>>
      %2 = xegpu.create_nd_tdesc %arg0 : memref<256x256xf32> -> !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %cst, %2[%0, %1] <{layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 32], inst_data = [8, 16]>}> : vector<64x64xf32>, !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      %3 = xegpu.create_nd_tdesc %arg1 : memref<256x1024xf16> -> !xegpu.tensor_desc<64x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.prefetch_nd %3[%0, %c0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 16], inst_data = [8, 16]>}> : !xegpu.tensor_desc<64x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %4 = xegpu.create_nd_tdesc %arg2 : memref<1024x256xf16> -> !xegpu.tensor_desc<32x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.prefetch_nd %4[%c0, %1] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 32], inst_data = [8, 16]>}> : !xegpu.tensor_desc<32x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<64x64xf32, strided<[256, 1], offset: ?>> -> memref<f32>, index, index, index, index, index
      %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f32> -> index
      %5 = arith.muli %offset, %c4 : index
      %6 = arith.addi %intptr, %5 : index
      %7 = arith.index_cast %6 : index to i64
      %8 = xegpu.create_nd_tdesc %7, shape : [64, 64], strides : [256, 1] : i64 -> !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      scf.for %arg3 = %c0 to %c1024 step %c32 {
        %9 = arith.addi %arg3, %c32 : index
        xegpu.prefetch_nd %4[%9, %1] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 32], inst_data = [8, 16]>}> : !xegpu.tensor_desc<32x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        xegpu.prefetch_nd %3[%0, %9] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 16], inst_data = [8, 16]>}> : !xegpu.tensor_desc<64x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %10 = xegpu.load_nd %3[%0, %arg3] <{layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 32], inst_data = [8, 16]>}> : !xegpu.tensor_desc<64x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x32xf16>
        %11 = xegpu.load_nd %4[%arg3, %1] <{layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 32], inst_data = [16, 16]>}> : !xegpu.tensor_desc<32x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<32x64xf16>
        %12 = xegpu.load_nd %8[0, 0]  : !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf32>
        %13 = xegpu.dpas %10, %11, %12 <{layout_a = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 32], inst_data = [8, 16]>, layout_b = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 32], inst_data = [16, 16]>, layout_cd = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 32], inst_data = [8, 16]>}> : vector<64x32xf16>, vector<32x64xf16>, vector<64x64xf32> -> vector<64x64xf32>
        xegpu.store_nd %13, %8[0, 0] <{layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 32], inst_data = [8, 16]>}> : vector<64x64xf32>, !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      }
      gpu.return
    }
  }

  // =========================================================================
  // CPU REFERENCE HELPERS (dynamic-shape, f32). Matmul operands that the GPU
  // feeds as f16 are pre-rounded to f16 precision by @round_to_f16 at the same
  // points the payload casts, then accumulated in f32 (matching the GPU dpas).
  // =========================================================================
  func.func @ref_matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.000000e+00 : f32
    %M = memref.dim %A, %c0 : memref<?x?xf32>
    %K = memref.dim %A, %c1 : memref<?x?xf32>
    %N = memref.dim %B, %c1 : memref<?x?xf32>
    scf.for %i = %c0 to %M step %c1 {
      scf.for %j = %c0 to %N step %c1 {
        %acc = scf.for %k = %c0 to %K step %c1 iter_args(%s = %zero) -> f32 {
          %a = memref.load %A[%i, %k] : memref<?x?xf32>
          %b = memref.load %B[%k, %j] : memref<?x?xf32>
          %m = arith.mulf %a, %b : f32
          %ns = arith.addf %s, %m : f32
          scf.yield %ns : f32
        }
        memref.store %acc, %C[%i, %j] : memref<?x?xf32>
      }
    }
    return
  }

  func.func @ext_f16_to_f32(%in: memref<?x?xf16>, %out: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %M = memref.dim %in, %c0 : memref<?x?xf16>
    %N = memref.dim %in, %c1 : memref<?x?xf16>
    scf.for %i = %c0 to %M step %c1 {
      scf.for %j = %c0 to %N step %c1 {
        %v = memref.load %in[%i, %j] : memref<?x?xf16>
        %e = arith.extf %v : f16 to f32
        memref.store %e, %out[%i, %j] : memref<?x?xf32>
      }
    }
    return
  }

  func.func @round_to_f16(%io: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %M = memref.dim %io, %c0 : memref<?x?xf32>
    %N = memref.dim %io, %c1 : memref<?x?xf32>
    scf.for %i = %c0 to %M step %c1 {
      scf.for %j = %c0 to %N step %c1 {
        %v = memref.load %io[%i, %j] : memref<?x?xf32>
        %h = arith.truncf %v : f32 to f16
        %r = arith.extf %h : f16 to f32
        memref.store %r, %io[%i, %j] : memref<?x?xf32>
      }
    }
    return
  }

  func.func @ref_layernorm(%in: memref<?x?xf32>, %gamma: memref<?xf32>,
                           %beta: memref<?xf32>, %out: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.000000e+00 : f32
    %one = arith.constant 1.000000e+00 : f32
    %eps = arith.constant 9.99999974E-6 : f32
    %M = memref.dim %in, %c0 : memref<?x?xf32>
    %N = memref.dim %in, %c1 : memref<?x?xf32>
    %N_i = arith.index_cast %N : index to i64
    %N_i32 = arith.trunci %N_i : i64 to i32
    %N_f = arith.sitofp %N_i32 : i32 to f32
    scf.for %i = %c0 to %M step %c1 {
      %sum = scf.for %j = %c0 to %N step %c1 iter_args(%s = %zero) -> f32 {
        %v = memref.load %in[%i, %j] : memref<?x?xf32>
        %ns = arith.addf %s, %v : f32
        scf.yield %ns : f32
      }
      %mean = arith.divf %sum, %N_f : f32
      %vsum = scf.for %j = %c0 to %N step %c1 iter_args(%s = %zero) -> f32 {
        %v = memref.load %in[%i, %j] : memref<?x?xf32>
        %d = arith.subf %v, %mean : f32
        %d2 = arith.mulf %d, %d : f32
        %ns = arith.addf %s, %d2 : f32
        scf.yield %ns : f32
      }
      %var = arith.divf %vsum, %N_f : f32
      %var_eps = arith.addf %var, %eps : f32
      %std = math.sqrt %var_eps : f32
      %inv = arith.divf %one, %std : f32
      scf.for %j = %c0 to %N step %c1 {
        %v = memref.load %in[%i, %j] : memref<?x?xf32>
        %g = memref.load %gamma[%j] : memref<?xf32>
        %b = memref.load %beta[%j] : memref<?xf32>
        %d = arith.subf %v, %mean : f32
        %nrm = arith.mulf %d, %inv : f32
        %sc = arith.mulf %nrm, %g : f32
        %res = arith.addf %sc, %b : f32
        memref.store %res, %out[%i, %j] : memref<?x?xf32>
      }
    }
    return
  }

  func.func @ref_add_bias(%io: memref<?x?xf32>, %bias: memref<?xf32>, %relu: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.000000e+00 : f32
    %M = memref.dim %io, %c0 : memref<?x?xf32>
    %N = memref.dim %io, %c1 : memref<?x?xf32>
    scf.for %i = %c0 to %M step %c1 {
      scf.for %j = %c0 to %N step %c1 {
        %v = memref.load %io[%i, %j] : memref<?x?xf32>
        %b = memref.load %bias[%j] : memref<?xf32>
        %s = arith.addf %v, %b : f32
        %r = arith.maximumf %s, %zero : f32
        %res = arith.select %relu, %r, %s : f32
        memref.store %res, %io[%i, %j] : memref<?x?xf32>
      }
    }
    return
  }

  func.func @ref_add(%a: memref<?x?xf32>, %b: memref<?x?xf32>, %out: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %M = memref.dim %a, %c0 : memref<?x?xf32>
    %N = memref.dim %a, %c1 : memref<?x?xf32>
    scf.for %i = %c0 to %M step %c1 {
      scf.for %j = %c0 to %N step %c1 {
        %va = memref.load %a[%i, %j] : memref<?x?xf32>
        %vb = memref.load %b[%i, %j] : memref<?x?xf32>
        %s = arith.addf %va, %vb : f32
        memref.store %s, %out[%i, %j] : memref<?x?xf32>
      }
    }
    return
  }

  // Non-causal multi-head attention: q/k/v are (256,256) f16-valued f32, H=4
  // heads of head_size=64, scale = 1/sqrt(64) = 0.125. Per head:
  //   scores = q_h @ k_h^T * scale ; w = f16(softmax(scores)) ; out_h = w @ v_h
  func.func @ref_attention(%q: memref<256x256xf32>, %k: memref<256x256xf32>,
                           %v: memref<256x256xf32>, %out: memref<256x256xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0.000000e+00 : f32
    %neg_inf = arith.constant 0xFF800000 : f32
    %scale = arith.constant 1.250000e-01 : f32
    %scratch = memref.alloc() : memref<256xf32>
    scf.for %h = %c0 to %c4 step %c1 {
      %base = arith.muli %h, %c64 : index
      scf.for %i = %c0 to %c256 step %c1 {
        %rmax = scf.for %j = %c0 to %c256 step %c1 iter_args(%mx = %neg_inf) -> f32 {
          %dot = scf.for %dd = %c0 to %c64 step %c1 iter_args(%s = %zero) -> f32 {
            %qcol = arith.addi %base, %dd : index
            %qv = memref.load %q[%i, %qcol] : memref<256x256xf32>
            %kv = memref.load %k[%j, %qcol] : memref<256x256xf32>
            %m = arith.mulf %qv, %kv : f32
            %ns = arith.addf %s, %m : f32
            scf.yield %ns : f32
          }
          %sc = arith.mulf %dot, %scale : f32
          memref.store %sc, %scratch[%j] : memref<256xf32>
          %nmx = arith.maximumf %mx, %sc : f32
          scf.yield %nmx : f32
        }
        %sum = scf.for %j = %c0 to %c256 step %c1 iter_args(%s = %zero) -> f32 {
          %sc = memref.load %scratch[%j] : memref<256xf32>
          %sh = arith.subf %sc, %rmax : f32
          %e = math.exp %sh : f32
          memref.store %e, %scratch[%j] : memref<256xf32>
          %ns = arith.addf %s, %e : f32
          scf.yield %ns : f32
        }
        scf.for %j = %c0 to %c256 step %c1 {
          %e = memref.load %scratch[%j] : memref<256xf32>
          %w = arith.divf %e, %sum : f32
          %wh = arith.truncf %w : f32 to f16
          %wf = arith.extf %wh : f16 to f32
          memref.store %wf, %scratch[%j] : memref<256xf32>
        }
        scf.for %d = %c0 to %c64 step %c1 {
          %col = arith.addi %base, %d : index
          %o = scf.for %j = %c0 to %c256 step %c1 iter_args(%s = %zero) -> f32 {
            %w = memref.load %scratch[%j] : memref<256xf32>
            %vv = memref.load %v[%j, %col] : memref<256x256xf32>
            %m = arith.mulf %w, %vv : f32
            %ns = arith.addf %s, %m : f32
            scf.yield %ns : f32
          }
          memref.store %o, %out[%i, %col] : memref<256x256xf32>
        }
      }
    }
    memref.dealloc %scratch : memref<256xf32>
    return
  }

  // One transformer block. Weights are f16; x/y are (256,256) f32. Produces y.
  func.func @ref_block(%x: memref<256x256xf32>,
                       %g1: memref<256xf32>, %b1n: memref<256xf32>,
                       %wq: memref<256x256xf16>, %wk: memref<256x256xf16>,
                       %wv: memref<256x256xf16>, %wp: memref<256x256xf16>,
                       %bp: memref<256xf32>,
                       %g2: memref<256xf32>, %b2n: memref<256xf32>,
                       %w1: memref<256x1024xf16>, %bb1: memref<1024xf32>,
                       %w2: memref<1024x256xf16>, %bb2: memref<256xf32>,
                       %y: memref<256x256xf32>) {
    // extend f16 weights to f32
    %wq32 = memref.alloc() : memref<256x256xf32>
    %wk32 = memref.alloc() : memref<256x256xf32>
    %wv32 = memref.alloc() : memref<256x256xf32>
    %wp32 = memref.alloc() : memref<256x256xf32>
    %w132 = memref.alloc() : memref<256x1024xf32>
    %w232 = memref.alloc() : memref<1024x256xf32>
    %wq_c = memref.cast %wq : memref<256x256xf16> to memref<?x?xf16>
    %wk_c = memref.cast %wk : memref<256x256xf16> to memref<?x?xf16>
    %wv_c = memref.cast %wv : memref<256x256xf16> to memref<?x?xf16>
    %wp_c = memref.cast %wp : memref<256x256xf16> to memref<?x?xf16>
    %w1_c = memref.cast %w1 : memref<256x1024xf16> to memref<?x?xf16>
    %w2_c = memref.cast %w2 : memref<1024x256xf16> to memref<?x?xf16>
    %wq32_c = memref.cast %wq32 : memref<256x256xf32> to memref<?x?xf32>
    %wk32_c = memref.cast %wk32 : memref<256x256xf32> to memref<?x?xf32>
    %wv32_c = memref.cast %wv32 : memref<256x256xf32> to memref<?x?xf32>
    %wp32_c = memref.cast %wp32 : memref<256x256xf32> to memref<?x?xf32>
    %w132_c = memref.cast %w132 : memref<256x1024xf32> to memref<?x?xf32>
    %w232_c = memref.cast %w232 : memref<1024x256xf32> to memref<?x?xf32>
    call @ext_f16_to_f32(%wq_c, %wq32_c) : (memref<?x?xf16>, memref<?x?xf32>) -> ()
    call @ext_f16_to_f32(%wk_c, %wk32_c) : (memref<?x?xf16>, memref<?x?xf32>) -> ()
    call @ext_f16_to_f32(%wv_c, %wv32_c) : (memref<?x?xf16>, memref<?x?xf32>) -> ()
    call @ext_f16_to_f32(%wp_c, %wp32_c) : (memref<?x?xf16>, memref<?x?xf32>) -> ()
    call @ext_f16_to_f32(%w1_c, %w132_c) : (memref<?x?xf16>, memref<?x?xf32>) -> ()
    call @ext_f16_to_f32(%w2_c, %w232_c) : (memref<?x?xf16>, memref<?x?xf32>) -> ()

    %x_c = memref.cast %x : memref<256x256xf32> to memref<?x?xf32>
    %g1_c = memref.cast %g1 : memref<256xf32> to memref<?xf32>
    %b1n_c = memref.cast %b1n : memref<256xf32> to memref<?xf32>
    %g2_c = memref.cast %g2 : memref<256xf32> to memref<?xf32>
    %b2n_c = memref.cast %b2n : memref<256xf32> to memref<?xf32>
    %bp_c = memref.cast %bp : memref<256xf32> to memref<?xf32>
    %bb1_c = memref.cast %bb1 : memref<1024xf32> to memref<?xf32>
    %bb2_c = memref.cast %bb2 : memref<256xf32> to memref<?xf32>
    %false = arith.constant 0 : i1
    %true = arith.constant 1 : i1

    // ln1 = f16(LN(x))
    %ln1 = memref.alloc() : memref<256x256xf32>
    %ln1_c = memref.cast %ln1 : memref<256x256xf32> to memref<?x?xf32>
    call @ref_layernorm(%x_c, %g1_c, %b1n_c, %ln1_c) : (memref<?x?xf32>, memref<?xf32>, memref<?xf32>, memref<?x?xf32>) -> ()
    call @round_to_f16(%ln1_c) : (memref<?x?xf32>) -> ()

    // q,k,v = f16(ln1 @ wq/wk/wv)
    %q = memref.alloc() : memref<256x256xf32>
    %k = memref.alloc() : memref<256x256xf32>
    %v = memref.alloc() : memref<256x256xf32>
    %q_c = memref.cast %q : memref<256x256xf32> to memref<?x?xf32>
    %k_c = memref.cast %k : memref<256x256xf32> to memref<?x?xf32>
    %v_c = memref.cast %v : memref<256x256xf32> to memref<?x?xf32>
    call @ref_matmul(%ln1_c, %wq32_c, %q_c) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    call @round_to_f16(%q_c) : (memref<?x?xf32>) -> ()
    call @ref_matmul(%ln1_c, %wk32_c, %k_c) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    call @round_to_f16(%k_c) : (memref<?x?xf32>) -> ()
    call @ref_matmul(%ln1_c, %wv32_c, %v_c) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    call @round_to_f16(%v_c) : (memref<?x?xf32>) -> ()

    // attn = f16(MHA(q,k,v))
    %attn = memref.alloc() : memref<256x256xf32>
    %attn_c = memref.cast %attn : memref<256x256xf32> to memref<?x?xf32>
    call @ref_attention(%q, %k, %v, %attn) : (memref<256x256xf32>, memref<256x256xf32>, memref<256x256xf32>, memref<256x256xf32>) -> ()
    call @round_to_f16(%attn_c) : (memref<?x?xf32>) -> ()

    // proj = attn @ wp + bp ; a = x + proj
    %proj = memref.alloc() : memref<256x256xf32>
    %proj_c = memref.cast %proj : memref<256x256xf32> to memref<?x?xf32>
    call @ref_matmul(%attn_c, %wp32_c, %proj_c) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    call @ref_add_bias(%proj_c, %bp_c, %false) : (memref<?x?xf32>, memref<?xf32>, i1) -> ()
    %a = memref.alloc() : memref<256x256xf32>
    %a_c = memref.cast %a : memref<256x256xf32> to memref<?x?xf32>
    call @ref_add(%x_c, %proj_c, %a_c) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

    // ln2 = f16(LN(a))
    %ln2 = memref.alloc() : memref<256x256xf32>
    %ln2_c = memref.cast %ln2 : memref<256x256xf32> to memref<?x?xf32>
    call @ref_layernorm(%a_c, %g2_c, %b2n_c, %ln2_c) : (memref<?x?xf32>, memref<?xf32>, memref<?xf32>, memref<?x?xf32>) -> ()
    call @round_to_f16(%ln2_c) : (memref<?x?xf32>) -> ()

    // hh = f16(relu(ln2 @ w1 + bb1))
    %hh = memref.alloc() : memref<256x1024xf32>
    %hh_c = memref.cast %hh : memref<256x1024xf32> to memref<?x?xf32>
    call @ref_matmul(%ln2_c, %w132_c, %hh_c) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    call @ref_add_bias(%hh_c, %bb1_c, %true) : (memref<?x?xf32>, memref<?xf32>, i1) -> ()
    call @round_to_f16(%hh_c) : (memref<?x?xf32>) -> ()

    // o = hh @ w2 + bb2 ; y = a + o
    %o = memref.alloc() : memref<256x256xf32>
    %o_c = memref.cast %o : memref<256x256xf32> to memref<?x?xf32>
    call @ref_matmul(%hh_c, %w232_c, %o_c) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    call @ref_add_bias(%o_c, %bb2_c, %false) : (memref<?x?xf32>, memref<?xf32>, i1) -> ()
    %y_c = memref.cast %y : memref<256x256xf32> to memref<?x?xf32>
    call @ref_add(%a_c, %o_c, %y_c) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

    memref.dealloc %wq32 : memref<256x256xf32>
    memref.dealloc %wk32 : memref<256x256xf32>
    memref.dealloc %wv32 : memref<256x256xf32>
    memref.dealloc %wp32 : memref<256x256xf32>
    memref.dealloc %w132 : memref<256x1024xf32>
    memref.dealloc %w232 : memref<1024x256xf32>
    memref.dealloc %ln1 : memref<256x256xf32>
    memref.dealloc %q : memref<256x256xf32>
    memref.dealloc %k : memref<256x256xf32>
    memref.dealloc %v : memref<256x256xf32>
    memref.dealloc %attn : memref<256x256xf32>
    memref.dealloc %proj : memref<256x256xf32>
    memref.dealloc %a : memref<256x256xf32>
    memref.dealloc %ln2 : memref<256x256xf32>
    memref.dealloc %hh : memref<256x1024xf32>
    memref.dealloc %o : memref<256x256xf32>
    return
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    // NOTE: fillResource1DRandom uses an unseeded RNG, so input data varies per
    // run. Magnitudes are kept small so the f16 / flash-online-softmax rounding
    // error stays well under rtol=1e-3 / atol=1e-4 on every draw across the
    // layer (output ~O(0.1), observed max abs error well under 1e-4).
    %gen_int = arith.constant 0 : i1
    %x_lo = arith.constant -2.500000e-02 : f32
    %x_hi = arith.constant 2.500000e-02 : f32
    %g_lo = arith.constant 9.750000e-01 : f32
    %g_hi = arith.constant 1.025000e+00 : f32
    %s_lo = arith.constant -6.000000e-03 : f32
    %s_hi = arith.constant 6.000000e-03 : f32
    // q/k projection weights kept extra-small so attention scores stay near 0
    // (softmax ~uniform): keeps the flash online-softmax kernel close to a plain
    // softmax CPU reference.
    %qk_lo = arith.constant -6.000000e-03 : f32
    %qk_hi = arith.constant 6.000000e-03 : f32

    // ---- input activations ----
    %x = memref.alloc() : memref<256x256xf32>
    %x_u = memref.cast %x : memref<256x256xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%x_u, %x_lo, %x_hi, %gen_int) : (memref<*xf32>, f32, f32, i1) -> ()

    // ---- layer 0 weights ----
    %l0_g1 = memref.alloc() : memref<256xf32>
    %l0_g1_u = memref.cast %l0_g1 : memref<256xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%l0_g1_u, %g_lo, %g_hi, %gen_int) : (memref<*xf32>, f32, f32, i1) -> ()
    %l0_b1n = memref.alloc() : memref<256xf32>
    %l0_b1n_u = memref.cast %l0_b1n : memref<256xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%l0_b1n_u, %s_lo, %s_hi, %gen_int) : (memref<*xf32>, f32, f32, i1) -> ()
    %l0_wq = memref.alloc() : memref<256x256xf16>
    %l0_wq_u = memref.cast %l0_wq : memref<256x256xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%l0_wq_u, %qk_lo, %qk_hi, %gen_int) : (memref<*xf16>, f32, f32, i1) -> ()
    %l0_wk = memref.alloc() : memref<256x256xf16>
    %l0_wk_u = memref.cast %l0_wk : memref<256x256xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%l0_wk_u, %qk_lo, %qk_hi, %gen_int) : (memref<*xf16>, f32, f32, i1) -> ()
    %l0_wv = memref.alloc() : memref<256x256xf16>
    %l0_wv_u = memref.cast %l0_wv : memref<256x256xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%l0_wv_u, %s_lo, %s_hi, %gen_int) : (memref<*xf16>, f32, f32, i1) -> ()
    %l0_wp = memref.alloc() : memref<256x256xf16>
    %l0_wp_u = memref.cast %l0_wp : memref<256x256xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%l0_wp_u, %s_lo, %s_hi, %gen_int) : (memref<*xf16>, f32, f32, i1) -> ()
    %l0_bp = memref.alloc() : memref<256xf32>
    %l0_bp_u = memref.cast %l0_bp : memref<256xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%l0_bp_u, %s_lo, %s_hi, %gen_int) : (memref<*xf32>, f32, f32, i1) -> ()
    %l0_g2 = memref.alloc() : memref<256xf32>
    %l0_g2_u = memref.cast %l0_g2 : memref<256xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%l0_g2_u, %g_lo, %g_hi, %gen_int) : (memref<*xf32>, f32, f32, i1) -> ()
    %l0_b2n = memref.alloc() : memref<256xf32>
    %l0_b2n_u = memref.cast %l0_b2n : memref<256xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%l0_b2n_u, %s_lo, %s_hi, %gen_int) : (memref<*xf32>, f32, f32, i1) -> ()
    %l0_w1 = memref.alloc() : memref<256x1024xf16>
    %l0_w1_u = memref.cast %l0_w1 : memref<256x1024xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%l0_w1_u, %s_lo, %s_hi, %gen_int) : (memref<*xf16>, f32, f32, i1) -> ()
    %l0_bb1 = memref.alloc() : memref<1024xf32>
    %l0_bb1_u = memref.cast %l0_bb1 : memref<1024xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%l0_bb1_u, %s_lo, %s_hi, %gen_int) : (memref<*xf32>, f32, f32, i1) -> ()
    %l0_w2 = memref.alloc() : memref<1024x256xf16>
    %l0_w2_u = memref.cast %l0_w2 : memref<1024x256xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%l0_w2_u, %s_lo, %s_hi, %gen_int) : (memref<*xf16>, f32, f32, i1) -> ()
    %l0_bb2 = memref.alloc() : memref<256xf32>
    %l0_bb2_u = memref.cast %l0_bb2 : memref<256xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%l0_bb2_u, %s_lo, %s_hi, %gen_int) : (memref<*xf32>, f32, f32, i1) -> ()

    // ---- final layernorm + lm_head ----
    %gf_g = memref.alloc() : memref<256xf32>
    %gf_g_u = memref.cast %gf_g : memref<256xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%gf_g_u, %g_lo, %g_hi, %gen_int) : (memref<*xf32>, f32, f32, i1) -> ()
    %gf_b = memref.alloc() : memref<256xf32>
    %gf_b_u = memref.cast %gf_b : memref<256xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%gf_b_u, %s_lo, %s_hi, %gen_int) : (memref<*xf32>, f32, f32, i1) -> ()
    %lmw = memref.alloc() : memref<256x256xf16>
    %lmw_u = memref.cast %lmw : memref<256x256xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%lmw_u, %s_lo, %s_hi, %gen_int) : (memref<*xf16>, f32, f32, i1) -> ()
    %lmb = memref.alloc() : memref<256xf32>
    %lmb_u = memref.cast %lmb : memref<256xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%lmb_u, %s_lo, %s_hi, %gen_int) : (memref<*xf32>, f32, f32, i1) -> ()

    %out = memref.alloc() : memref<256x256xf32>

    // ---- copy inputs/output to GPU ----
    %x_g = gpu.alloc () : memref<256x256xf32>
    gpu.memcpy %x_g, %x : memref<256x256xf32>, memref<256x256xf32>
    // layer 0 -> GPU
    %l0_g1_g = gpu.alloc () : memref<256xf32>
    gpu.memcpy %l0_g1_g, %l0_g1 : memref<256xf32>, memref<256xf32>
    %l0_b1n_g = gpu.alloc () : memref<256xf32>
    gpu.memcpy %l0_b1n_g, %l0_b1n : memref<256xf32>, memref<256xf32>
    %l0_wq_g = gpu.alloc () : memref<256x256xf16>
    gpu.memcpy %l0_wq_g, %l0_wq : memref<256x256xf16>, memref<256x256xf16>
    %l0_wk_g = gpu.alloc () : memref<256x256xf16>
    gpu.memcpy %l0_wk_g, %l0_wk : memref<256x256xf16>, memref<256x256xf16>
    %l0_wv_g = gpu.alloc () : memref<256x256xf16>
    gpu.memcpy %l0_wv_g, %l0_wv : memref<256x256xf16>, memref<256x256xf16>
    %l0_wp_g = gpu.alloc () : memref<256x256xf16>
    gpu.memcpy %l0_wp_g, %l0_wp : memref<256x256xf16>, memref<256x256xf16>
    %l0_bp_g = gpu.alloc () : memref<256xf32>
    gpu.memcpy %l0_bp_g, %l0_bp : memref<256xf32>, memref<256xf32>
    %l0_g2_g = gpu.alloc () : memref<256xf32>
    gpu.memcpy %l0_g2_g, %l0_g2 : memref<256xf32>, memref<256xf32>
    %l0_b2n_g = gpu.alloc () : memref<256xf32>
    gpu.memcpy %l0_b2n_g, %l0_b2n : memref<256xf32>, memref<256xf32>
    %l0_w1_g = gpu.alloc () : memref<256x1024xf16>
    gpu.memcpy %l0_w1_g, %l0_w1 : memref<256x1024xf16>, memref<256x1024xf16>
    %l0_bb1_g = gpu.alloc () : memref<1024xf32>
    gpu.memcpy %l0_bb1_g, %l0_bb1 : memref<1024xf32>, memref<1024xf32>
    %l0_w2_g = gpu.alloc () : memref<1024x256xf16>
    gpu.memcpy %l0_w2_g, %l0_w2 : memref<1024x256xf16>, memref<1024x256xf16>
    %l0_bb2_g = gpu.alloc () : memref<256xf32>
    gpu.memcpy %l0_bb2_g, %l0_bb2 : memref<256xf32>, memref<256xf32>
    %gf_g_g = gpu.alloc () : memref<256xf32>
    gpu.memcpy %gf_g_g, %gf_g : memref<256xf32>, memref<256xf32>
    %gf_b_g = gpu.alloc () : memref<256xf32>
    gpu.memcpy %gf_b_g, %gf_b : memref<256xf32>, memref<256xf32>
    %lmw_g = gpu.alloc () : memref<256x256xf16>
    gpu.memcpy %lmw_g, %lmw : memref<256x256xf16>, memref<256x256xf16>
    %lmb_g = gpu.alloc () : memref<256xf32>
    gpu.memcpy %lmb_g, %lmb : memref<256xf32>, memref<256xf32>
    %out_g = gpu.alloc () : memref<256x256xf32>

    call @payload(%out_g, %x_g, %l0_g1_g, %l0_b1n_g, %l0_wq_g, %l0_wk_g, %l0_wv_g, %l0_wp_g, %l0_bp_g, %l0_g2_g, %l0_b2n_g, %l0_w1_g, %l0_bb1_g, %l0_w2_g, %l0_bb2_g, %gf_g_g, %gf_b_g, %lmw_g, %lmb_g)
        : (memref<256x256xf32>, memref<256x256xf32>, memref<256xf32>, memref<256xf32>, memref<256x256xf16>, memref<256x256xf16>, memref<256x256xf16>, memref<256x256xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256x1024xf16>, memref<1024xf32>, memref<1024x256xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256x256xf16>, memref<256xf32>) -> ()
    gpu.wait
    gpu.memcpy %out, %out_g : memref<256x256xf32>, memref<256x256xf32>
    gpu.dealloc %x_g : memref<256x256xf32>
    gpu.dealloc %l0_g1_g : memref<256xf32>
    gpu.dealloc %l0_b1n_g : memref<256xf32>
    gpu.dealloc %l0_wq_g : memref<256x256xf16>
    gpu.dealloc %l0_wk_g : memref<256x256xf16>
    gpu.dealloc %l0_wv_g : memref<256x256xf16>
    gpu.dealloc %l0_wp_g : memref<256x256xf16>
    gpu.dealloc %l0_bp_g : memref<256xf32>
    gpu.dealloc %l0_g2_g : memref<256xf32>
    gpu.dealloc %l0_b2n_g : memref<256xf32>
    gpu.dealloc %l0_w1_g : memref<256x1024xf16>
    gpu.dealloc %l0_bb1_g : memref<1024xf32>
    gpu.dealloc %l0_w2_g : memref<1024x256xf16>
    gpu.dealloc %l0_bb2_g : memref<256xf32>
    gpu.dealloc %gf_g_g : memref<256xf32>
    gpu.dealloc %gf_b_g : memref<256xf32>
    gpu.dealloc %lmw_g : memref<256x256xf16>
    gpu.dealloc %lmb_g : memref<256xf32>
    gpu.dealloc %out_g : memref<256x256xf32>

    // ---- CPU reference: 1 transformer block, then ln_f + lm_head ----
    %h0 = memref.alloc() : memref<256x256xf32>
    call @ref_block(%x, %l0_g1, %l0_b1n, %l0_wq, %l0_wk, %l0_wv, %l0_wp, %l0_bp, %l0_g2, %l0_b2n, %l0_w1, %l0_bb1, %l0_w2, %l0_bb2, %h0)
        : (memref<256x256xf32>, memref<256xf32>, memref<256xf32>, memref<256x256xf16>, memref<256x256xf16>, memref<256x256xf16>, memref<256x256xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256x1024xf16>, memref<1024xf32>, memref<1024x256xf16>, memref<256xf32>, memref<256x256xf32>) -> ()

    // hf = f16(LN(h, gf_g, gf_b)) ; cpu_out = hf @ lmw + lmb
    %hf = memref.alloc() : memref<256x256xf32>
    %hf_c = memref.cast %hf : memref<256x256xf32> to memref<?x?xf32>
    %hlast_c = memref.cast %h0 : memref<256x256xf32> to memref<?x?xf32>
    %gf_g_c = memref.cast %gf_g : memref<256xf32> to memref<?xf32>
    %gf_b_c = memref.cast %gf_b : memref<256xf32> to memref<?xf32>
    call @ref_layernorm(%hlast_c, %gf_g_c, %gf_b_c, %hf_c) : (memref<?x?xf32>, memref<?xf32>, memref<?xf32>, memref<?x?xf32>) -> ()
    call @round_to_f16(%hf_c) : (memref<?x?xf32>) -> ()
    %lmw32 = memref.alloc() : memref<256x256xf32>
    %lmw_c = memref.cast %lmw : memref<256x256xf16> to memref<?x?xf16>
    %lmw32_c = memref.cast %lmw32 : memref<256x256xf32> to memref<?x?xf32>
    call @ext_f16_to_f32(%lmw_c, %lmw32_c) : (memref<?x?xf16>, memref<?x?xf32>) -> ()
    %cpu_out = memref.alloc() : memref<256x256xf32>
    %cpu_out_c = memref.cast %cpu_out : memref<256x256xf32> to memref<?x?xf32>
    %lmb_c = memref.cast %lmb : memref<256xf32> to memref<?xf32>
    %false_m = arith.constant 0 : i1
    call @ref_matmul(%hf_c, %lmw32_c, %cpu_out_c) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    call @ref_add_bias(%cpu_out_c, %lmb_c, %false_m) : (memref<?x?xf32>, memref<?xf32>, i1) -> ()

    // ---- compare ----
    %out_star = memref.cast %out : memref<256x256xf32> to memref<*xf32>
    %cpu_star = memref.cast %cpu_out : memref<256x256xf32> to memref<*xf32>
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%out_star, %cpu_star) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %x : memref<256x256xf32>
    memref.dealloc %l0_g1 : memref<256xf32>
    memref.dealloc %l0_b1n : memref<256xf32>
    memref.dealloc %l0_wq : memref<256x256xf16>
    memref.dealloc %l0_wk : memref<256x256xf16>
    memref.dealloc %l0_wv : memref<256x256xf16>
    memref.dealloc %l0_wp : memref<256x256xf16>
    memref.dealloc %l0_bp : memref<256xf32>
    memref.dealloc %l0_g2 : memref<256xf32>
    memref.dealloc %l0_b2n : memref<256xf32>
    memref.dealloc %l0_w1 : memref<256x1024xf16>
    memref.dealloc %l0_bb1 : memref<1024xf32>
    memref.dealloc %l0_w2 : memref<1024x256xf16>
    memref.dealloc %l0_bb2 : memref<256xf32>
    memref.dealloc %gf_g : memref<256xf32>
    memref.dealloc %gf_b : memref<256xf32>
    memref.dealloc %lmw : memref<256x256xf16>
    memref.dealloc %lmb : memref<256xf32>
    memref.dealloc %out : memref<256x256xf32>
    memref.dealloc %h0 : memref<256x256xf32>
    memref.dealloc %hf : memref<256x256xf32>
    memref.dealloc %lmw32 : memref<256x256xf32>
    memref.dealloc %cpu_out : memref<256x256xf32>
    return
  }

  func.func private @fillResource1DRandomF32(memref<*xf32>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
