// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --shared-libs=%irunner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

#lo_sg_8x1_data_8x64 = #xegpu.layout<sg_layout = [8, 1], sg_data = [8, 64], order = [1, 0]>
#lo_sg_1x8_data_64x8 = #xegpu.layout<sg_layout = [1, 8], sg_data = [64, 8], order = [0, 1]>
module attributes {gpu.container_module} {
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %c0_f32 = arith.constant 0.0 : f32
    %cf_lower = arith.constant -0.5 : f32
    %cf_upper = arith.constant 0.5 : f32
    %c_gen_int = arith.constant 0 : i1

    // Allocate and randomly initialize input in [-0.5, 0.5]
    %arg0 = memref.alloc() : memref<1024x64xf32>
    %arg0_cast = memref.cast %arg0 : memref<1024x64xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%arg0_cast, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf32>, f32, f32, i1) -> ()

    // Allocate and zero-initialize GPU output buffer (CPU side)
    %arg1 = memref.alloc() : memref<1024x64xf32>
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c64 step %c1 {
        memref.store %c0_f32, %arg1[%i, %j] : memref<1024x64xf32>
      }
    }

    // Allocate GPU buffers and copy input/output to GPU
    %arg0_gpu = gpu.alloc () : memref<1024x64xf32>
    gpu.memcpy %arg0_gpu, %arg0 : memref<1024x64xf32>, memref<1024x64xf32>
    %arg1_gpu = gpu.alloc () : memref<1024x64xf32>
    gpu.memcpy %arg1_gpu, %arg1 : memref<1024x64xf32>, memref<1024x64xf32>

    // Launch kernel and wait for completion
    gpu.launch_func  @main_kernel::@main_kernel
      blocks in (%c16, %c1, %c1) threads in (%c128, %c1, %c1)
      args(%arg0_gpu : memref<1024x64xf32>, %arg1_gpu : memref<1024x64xf32>)
    gpu.wait

    // Copy result back to host
    gpu.memcpy %arg1, %arg1_gpu : memref<1024x64xf32>, memref<1024x64xf32>
    gpu.dealloc %arg0_gpu : memref<1024x64xf32>
    gpu.dealloc %arg1_gpu : memref<1024x64xf32>

    // Compute CPU reference
    %cpu_out = memref.alloc() : memref<1024x64xf32>
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c64 step %c1 {
        memref.store %c0_f32, %cpu_out[%i, %j] : memref<1024x64xf32>
      }
    }
    call @cpu_reference(%arg0, %cpu_out) : (memref<1024x64xf32>, memref<1024x64xf32>) -> ()

    // Compare GPU and CPU results
    %arg1_star = memref.cast %arg1 : memref<1024x64xf32> to memref<*xf32>
    %cpu_out_star = memref.cast %cpu_out : memref<1024x64xf32> to memref<*xf32>
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%arg1_star, %cpu_out_star) : (memref<*xf32>, memref<*xf32>) -> ()
    // Debug print the first row of GPU and CPU output
    // %gpu_row_0 = memref.subview %arg1[%c0, %c0] [%c1, %c64] [%c1, %c1]
    //   : memref<1024x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    // %gpu_row_0_star = memref.cast %gpu_row_0
    //   : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<*xf32>
    // call @printMemrefF32(%gpu_row_0_star) : (memref<*xf32>) -> ()

    // %cpu_row_0 = memref.subview %cpu_out[%c0, %c0] [%c1, %c64] [%c1, %c1]
    //   : memref<1024x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    // %cpu_row_0_star = memref.cast %cpu_row_0
    //   : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<*xf32>
    // call @printMemrefF32(%cpu_row_0_star) : (memref<*xf32>) -> ()

    memref.dealloc %arg0 : memref<1024x64xf32>
    memref.dealloc %arg1 : memref<1024x64xf32>
    memref.dealloc %cpu_out : memref<1024x64xf32>
    return
  }
  func.func @cpu_reference(%arg0: memref<1024x64xf32>, %arg1: memref<1024x64xf32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c64 = arith.constant 64 : index
    %neg_inf = arith.constant 0xFF800000 : f32
    %zero = arith.constant 0.0 : f32
    // Iterate over each row
    scf.for %row = %c0 to %c1024 step %c1 {
      // Step 1: find row max
      %max = scf.for %col = %c0 to %c64 step %c1 iter_args(%cur_max = %neg_inf) -> f32 {
        %val = memref.load %arg0[%row, %col] : memref<1024x64xf32>
        %new_max = arith.maximumf %cur_max, %val : f32
        scf.yield %new_max : f32
      }
      // Step 2: compute exp(x - max) and accumulate sum
      %sum = scf.for %col = %c0 to %c64 step %c1 iter_args(%cur_sum = %zero) -> f32 {
        %val = memref.load %arg0[%row, %col] : memref<1024x64xf32>
        %shifted = arith.subf %val, %max : f32
        %exp_val = math.exp %shifted : f32
        memref.store %exp_val, %arg1[%row, %col] : memref<1024x64xf32>
        %new_sum = arith.addf %cur_sum, %exp_val : f32
        scf.yield %new_sum : f32
      }
      // Step 3: divide by sum
      scf.for %col = %c0 to %c64 step %c1 {
        %exp_val = memref.load %arg1[%row, %col] : memref<1024x64xf32>
        %result = arith.divf %exp_val, %sum : f32
        memref.store %result, %arg1[%row, %col] : memref<1024x64xf32>
      }
    }
    return
  }
  gpu.module @main_kernel [#xevm.target<chip = "pvc">] {
    gpu.func @main_kernel(%arg0: memref<1024x64xf32>, %arg1: memref<1024x64xf32>) kernel attributes {intel_reqd_sub_group_size = 16 : i32, known_block_size = array<i32: 128, 1, 1>, known_grid_size = array<i32: 16, 1, 1>} {
      %cst = arith.constant {layout_result_0 = #xegpu.slice<#lo_sg_8x1_data_8x64, dims = [1]>} dense<0.000000e+00> : vector<64xf32>
      %cst_0 = arith.constant {layout_result_0 = #xegpu.slice<#lo_sg_8x1_data_8x64, dims = [1]>} dense<0xFF800000> : vector<64xf32>
      %c64 = arith.constant 64 : index
      %block_id_x = gpu.block_id  x
      %0 = arith.muli %block_id_x, %c64 overflow<nsw> : index
      %1 = xegpu.create_nd_tdesc %arg0 : memref<1024x64xf32>
        -> !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #lo_sg_8x1_data_8x64>
      %2 = xegpu.load_nd %1[%0, 0] <{layout = #lo_sg_8x1_data_8x64}> :
        !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #lo_sg_8x1_data_8x64> -> vector<64x64xf32>
      %4 = vector.multi_reduction <maximumf>, %2, %cst_0
        {layout_result_0 = #xegpu.slice<#lo_sg_8x1_data_8x64, dims = [1]>} [1] : vector<64x64xf32> to vector<64xf32>
      %5 = xegpu.convert_layout %4 <
        {input_layout = #xegpu.slice<#lo_sg_8x1_data_8x64, dims = [1]>,
        target_layout = #xegpu.slice<#lo_sg_1x8_data_64x8, dims = [0]>}> : vector<64xf32>
      %6 = vector.broadcast %5 {layout_result_0 = #lo_sg_1x8_data_64x8} : vector<64xf32> to vector<64x64xf32>
      %7 = vector.transpose %6, [1, 0] {layout_result_0 = #lo_sg_8x1_data_8x64} : vector<64x64xf32> to vector<64x64xf32>
      %8 = arith.subf %2, %7 {layout_result_0 = #lo_sg_8x1_data_8x64} : vector<64x64xf32>
      %9 = math.exp %8 {layout_result_0 = #lo_sg_8x1_data_8x64} : vector<64x64xf32>
      %11 = vector.multi_reduction <add>, %9, %cst
        {layout_result_0 = #xegpu.slice<#lo_sg_8x1_data_8x64, dims = [1]>} [1] : vector<64x64xf32> to vector<64xf32>
      %12 = xegpu.convert_layout %11 <
        {input_layout = #xegpu.slice<#lo_sg_8x1_data_8x64, dims = [1]>,
         target_layout = #xegpu.slice<#lo_sg_1x8_data_64x8, dims = [0]>}> : vector<64xf32>
      %13 = vector.broadcast %12
        {layout_result_0 = #lo_sg_1x8_data_64x8} : vector<64xf32> to vector<64x64xf32>
      %14 = vector.transpose %13, [1, 0] {layout_result_0 = #lo_sg_8x1_data_8x64} : vector<64x64xf32> to vector<64x64xf32>
      %15 = arith.divf %9, %14 {layout_result_0 = #lo_sg_8x1_data_8x64} : vector<64x64xf32>
      %16 = xegpu.create_nd_tdesc %arg1 : memref<1024x64xf32> ->
        !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #lo_sg_8x1_data_8x64>
      xegpu.store_nd %15, %16[%0, 0] <{layout = #lo_sg_8x1_data_8x64}>
        : vector<64x64xf32>, !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #lo_sg_8x1_data_8x64>
      gpu.return
    }
  }
  func.func private @fillResource1DRandomF32(memref<*xf32>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
