// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --shared-libs=%irunner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// Workgroup-level layer normalization over the last dimension of a 1024x512
// f32 tensor:
//   mean_i    = (1/N) * sum_j x[i, j]
//   var_i     = (1/N) * sum_j (x[i, j] - mean_i)^2
//   out[i, j] = (x[i, j] - mean_i) / sqrt(var_i + eps) * gamma[j] + beta[j]
//
// The @layer_norm kernel below is the xegpu-wg IR dumped from the lighthouse
// layer_norm workload (examples/xegpu/layer_norm.py --dump-kernel=xegpu-wg):
// each of the 16 workgroups normalizes 64 rows; the per-row mean/variance are
// accumulated in SLM (xegpu.mem_desc) across 16-column tiles, then the rows are
// normalized and scaled/shifted by gamma/beta.

module attributes {gpu.container_module} {
  gpu.module @layer_norm [#xevm.target<O = 3>] {
    gpu.func @layer_norm(%arg0: memref<1024x512xf32>, %arg1: memref<1024x512xf32>, %arg2: memref<512xf32>, %arg3: memref<512xf32>) kernel attributes {known_block_size = array<i32: 128, 1, 1>, known_grid_size = array<i32: 16, 1, 1>} {
      %c4 = arith.constant 4 : index
      %cst = arith.constant dense<true> : vector<16xi1>
      %cst_0 = arith.constant dense<9.99999974E-6> : vector<64xf32>
      %cst_1 = arith.constant dense<0.001953125> : vector<64xf32>
      %cst_2 = arith.constant dense<0.000000e+00> : vector<64xf32>
      %c16 = arith.constant 16 : index
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %cst_3 = arith.constant dense<0.000000e+00> : vector<64x16xf32>
      %c64 = arith.constant 64 : index
      %block_id_x = gpu.block_id x
      %0 = arith.muli %block_id_x, %c64 overflow<nsw> : index
      %alloca = memref.alloca() : memref<64x16xf32, 3>
      %1 = xegpu.create_mem_desc %alloca : memref<64x16xf32, 3> -> !xegpu.mem_desc<64x16xf32>
      xegpu.store_matrix %cst_3, %1[0, 0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [8, 16]>}>: vector<64x16xf32>, !xegpu.mem_desc<64x16xf32>
      %alloca_4 = memref.alloca() : memref<64x16xf32, 3>
      %mdesc_4 = xegpu.create_mem_desc %alloca_4 : memref<64x16xf32, 3> -> !xegpu.mem_desc<64x16xf32>
      xegpu.store_matrix %cst_3, %mdesc_4[0, 0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [8, 16]>}>: vector<64x16xf32>, !xegpu.mem_desc<64x16xf32>
      scf.for %arg4 = %c0 to %c512 step %c16 {
        %7 = xegpu.create_nd_tdesc %arg0 : memref<1024x512xf32> -> !xegpu.tensor_desc<64x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
        %8 = xegpu.load_nd %7[%0, %arg4]  : !xegpu.tensor_desc<64x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x16xf32>
        %9 = xegpu.create_mem_desc %alloca_4 : memref<64x16xf32, 3> -> !xegpu.mem_desc<64x16xf32>
        %10 = xegpu.load_matrix %9[0, 0] : !xegpu.mem_desc<64x16xf32> -> vector<64x16xf32>
        %11 = arith.addf %8, %10 : vector<64x16xf32>
        xegpu.store_matrix %11, %9[0, 0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [8, 16]>}>: vector<64x16xf32>, !xegpu.mem_desc<64x16xf32>
      }
      %2 = xegpu.create_mem_desc %alloca_4 : memref<64x16xf32, 3> -> !xegpu.mem_desc<64x16xf32>
      %3 = xegpu.load_matrix %2[0, 0] : !xegpu.mem_desc<64x16xf32> -> vector<64x16xf32>
      %4 = vector.multi_reduction <add>, %3, %cst_2 [1] : vector<64x16xf32> to vector<64xf32>
      scf.for %arg4 = %c0 to %c512 step %c16 {
        %7 = xegpu.create_nd_tdesc %arg0 : memref<1024x512xf32> -> !xegpu.tensor_desc<64x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
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
      %subview = memref.subview %arg1[%0, 0] [64, 512] [1, 1] : memref<1024x512xf32> to memref<64x512xf32, strided<[512, 1], offset: ?>>
      scf.for %arg4 = %c0 to %c512 step %c16 {
        %7 = xegpu.create_nd_tdesc %arg0 : memref<1024x512xf32> -> !xegpu.tensor_desc<64x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
        %8 = xegpu.load_nd %7[%0, %arg4]  : !xegpu.tensor_desc<64x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x16xf32>
        %9 = vector.step : vector<16xindex>
        %10 = vector.broadcast %arg4 : index to vector<16xindex>
        %11 = arith.addi %10, %9 : vector<16xindex>
        %intptr = memref.extract_aligned_pointer_as_index %arg2 : memref<512xf32> -> index
        %12 = arith.index_cast %intptr : index to i64
        %13 = xegpu.load %12[%11], %cst  : i64, vector<16xindex>, vector<16xi1> -> vector<16xf32>
        %14 = vector.broadcast %13 : vector<16xf32> to vector<64x16xf32>
        %intptr_5 = memref.extract_aligned_pointer_as_index %arg3 : memref<512xf32> -> index
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
        %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<64x512xf32, strided<[512, 1], offset: ?>> -> memref<f32>, index, index, index, index, index
        %intptr_6 = memref.extract_aligned_pointer_as_index %base_buffer : memref<f32> -> index
        %30 = arith.muli %offset, %c4 : index
        %31 = arith.addi %intptr_6, %30 : index
        %32 = arith.index_cast %31 : index to i64
        %33 = xegpu.create_nd_tdesc %32, shape : [64, 512], strides : [512, 1] : i64 -> !xegpu.tensor_desc<64x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
        xegpu.store_nd %29, %33[0, %arg4] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [8, 16]>}> : vector<64x16xf32>, !xegpu.tensor_desc<64x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      }
      gpu.return
    }
  }

  // CPU reference: layer normalization over each row of length N=512.
  func.func @cpu_reference(%input: memref<1024x512xf32>, %gamma: memref<512xf32>,
                           %beta: memref<512xf32>, %output: memref<1024x512xf32>)
      attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %zero = arith.constant 0.000000e+00 : f32
    %one = arith.constant 1.000000e+00 : f32
    %eps = arith.constant 9.99999974E-6 : f32
    %n_f32 = arith.constant 5.120000e+02 : f32
    scf.for %row = %c0 to %c1024 step %c1 {
      // mean = (1/N) * sum_j x[row, j]
      %sum = scf.for %col = %c0 to %c512 step %c1 iter_args(%acc = %zero) -> f32 {
        %v = memref.load %input[%row, %col] : memref<1024x512xf32>
        %a = arith.addf %acc, %v : f32
        scf.yield %a : f32
      }
      %mean = arith.divf %sum, %n_f32 : f32
      // var = (1/N) * sum_j (x[row, j] - mean)^2
      %var_sum = scf.for %col = %c0 to %c512 step %c1 iter_args(%acc = %zero) -> f32 {
        %v = memref.load %input[%row, %col] : memref<1024x512xf32>
        %d = arith.subf %v, %mean : f32
        %d2 = arith.mulf %d, %d : f32
        %a = arith.addf %acc, %d2 : f32
        scf.yield %a : f32
      }
      %var = arith.divf %var_sum, %n_f32 : f32
      %var_eps = arith.addf %var, %eps : f32
      %std = math.sqrt %var_eps : f32
      %inv_std = arith.divf %one, %std : f32
      // out[row, j] = (x[row, j] - mean) * inv_std * gamma[j] + beta[j]
      scf.for %col = %c0 to %c512 step %c1 {
        %v = memref.load %input[%row, %col] : memref<1024x512xf32>
        %g = memref.load %gamma[%col] : memref<512xf32>
        %b = memref.load %beta[%col] : memref<512xf32>
        %d = arith.subf %v, %mean : f32
        %norm = arith.mulf %d, %inv_std : f32
        %scaled = arith.mulf %norm, %g : f32
        %res = arith.addf %scaled, %b : f32
        memref.store %res, %output[%row, %col] : memref<1024x512xf32>
      }
    }
    return
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c0_f32 = arith.constant 0.000000e+00 : f32
    %cf_lower = arith.constant -1.000000e+00 : f32
    %cf_upper = arith.constant 1.000000e+00 : f32
    %g_lower = arith.constant 5.000000e-01 : f32
    %g_upper = arith.constant 1.500000e+00 : f32
    %b_lower = arith.constant -1.000000e-01 : f32
    %b_upper = arith.constant 1.000000e-01 : f32
    %gen_int = arith.constant 0 : i1

    // Random input in [-1, 1], gamma in [0.5, 1.5], beta in [-0.1, 0.1].
    %input = memref.alloc() : memref<1024x512xf32>
    %input_cast = memref.cast %input : memref<1024x512xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%input_cast, %cf_lower, %cf_upper, %gen_int) : (memref<*xf32>, f32, f32, i1) -> ()

    %gamma = memref.alloc() : memref<512xf32>
    %gamma_cast = memref.cast %gamma : memref<512xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%gamma_cast, %g_lower, %g_upper, %gen_int) : (memref<*xf32>, f32, f32, i1) -> ()

    %beta = memref.alloc() : memref<512xf32>
    %beta_cast = memref.cast %beta : memref<512xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%beta_cast, %b_lower, %b_upper, %gen_int) : (memref<*xf32>, f32, f32, i1) -> ()

    // Zero-initialized GPU output buffer (host side).
    %output = memref.alloc() : memref<1024x512xf32>
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c512 step %c1 {
        memref.store %c0_f32, %output[%i, %j] : memref<1024x512xf32>
      }
    }

    // Copy inputs/output to the GPU.
    %input_gpu = gpu.alloc () : memref<1024x512xf32>
    gpu.memcpy %input_gpu, %input : memref<1024x512xf32>, memref<1024x512xf32>
    %gamma_gpu = gpu.alloc () : memref<512xf32>
    gpu.memcpy %gamma_gpu, %gamma : memref<512xf32>, memref<512xf32>
    %beta_gpu = gpu.alloc () : memref<512xf32>
    gpu.memcpy %beta_gpu, %beta : memref<512xf32>, memref<512xf32>
    %output_gpu = gpu.alloc () : memref<1024x512xf32>
    gpu.memcpy %output_gpu, %output : memref<1024x512xf32>, memref<1024x512xf32>

    // Launch: 16 workgroups of 128 threads, each normalizing 64 rows.
    gpu.launch_func @layer_norm::@layer_norm
      blocks in (%c16, %c1, %c1) threads in (%c128, %c1, %c1)
      args(%input_gpu : memref<1024x512xf32>, %output_gpu : memref<1024x512xf32>,
           %gamma_gpu : memref<512xf32>, %beta_gpu : memref<512xf32>)
    gpu.wait

    gpu.memcpy %output, %output_gpu : memref<1024x512xf32>, memref<1024x512xf32>
    gpu.dealloc %input_gpu : memref<1024x512xf32>
    gpu.dealloc %gamma_gpu : memref<512xf32>
    gpu.dealloc %beta_gpu : memref<512xf32>
    gpu.dealloc %output_gpu : memref<1024x512xf32>

    // CPU reference.
    %cpu_out = memref.alloc() : memref<1024x512xf32>
    call @cpu_reference(%input, %gamma, %beta, %cpu_out) : (memref<1024x512xf32>, memref<512xf32>, memref<512xf32>, memref<1024x512xf32>) -> ()

    %output_star = memref.cast %output : memref<1024x512xf32> to memref<*xf32>
    %cpu_out_star = memref.cast %cpu_out : memref<1024x512xf32> to memref<*xf32>
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%output_star, %cpu_out_star) : (memref<*xf32>, memref<*xf32>) -> ()

    memref.dealloc %input : memref<1024x512xf32>
    memref.dealloc %gamma : memref<512xf32>
    memref.dealloc %beta : memref<512xf32>
    memref.dealloc %output : memref<1024x512xf32>
    memref.dealloc %cpu_out : memref<1024x512xf32>
    return
  }

  func.func private @fillResource1DRandomF32(memref<*xf32>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
