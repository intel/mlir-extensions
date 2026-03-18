// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup"

module attributes {gpu.container_module} {
  func.func @multiple_ops_entry(%arg0: memref<2x8x256xf16>, %arg1: memref<2x8x256xf16>, %arg2: memref<2x8x256xf32>, %arg3: memref<2x8x256xf32>, %arg4: memref<2x8x256xf32>, %arg5: memref<2x8x256xf32>, %arg6: memref<2x8x256xf32>) attributes {L2Mem = 0 : i64, gemm_tiles_x = dense<1> : vector<4xi64>, gemm_tiles_y = dense<[1, -1, 4, 32]> : vector<4xi64>, habana_runner.tests = [{inputs = [2.400000e+01 : f16, 6.101560e+00 : f16, 5.000000e+00 : f32, 2.000000e+00 : f32], outputs = [3.010000e+01 : f32, 7.000000e+00 : f32, 3.710000e+01 : f32], tolerance = [1.000000e-01]}], linear_block_size = array<i32: 1024, 1, 1>, linear_grid_size = array<i32: 4, 1, 1>, region_partition = 1 : i64, region_size = 4 : i64, syn.fusion_successful, syn.gemm_pipeline, syn.tensor_signature = (tensor<2x8x256xf16>, tensor<2x8x256xf16>, tensor<2x8x256xf32>, tensor<2x8x256xf32>) -> (tensor<2x8x256xf32>, tensor<2x8x256xf32>, tensor<2x8x256xf32>), synFusionGenOps = 2 : i64, synFusionRequiredBeamSize = 1 : i64, synFusionTotalCost = 1.315600e+03 : f64} {
    %c2 = arith.constant 2 : index
    %c2048 = arith.constant 2048 : index
    %cst = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 32]>} dense<true> : vector<4x8x32xi1>
    %cst_0 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 1]>, dims = [0, 2]>} dense<32> : vector<8xindex>
    %cst_1 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 1]>, dims = [1, 2]>} dense<256> : vector<4xindex>
    %c4 = arith.constant 4 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @multiple_ops::@multiple_ops blocks in (%c4, %c1, %c1) threads in (%c1024, %c1, %c1)  args(%arg0 : memref<2x8x256xf16>, %arg1 : memref<2x8x256xf16>, %arg2 : memref<2x8x256xf32>, %arg3 : memref<2x8x256xf32>, %arg4 : memref<2x8x256xf32>, %arg5 : memref<2x8x256xf32>, %arg6 : memref<2x8x256xf32>)
    return
  }
  gpu.module @multiple_ops {
    gpu.func @multiple_ops(%arg0: memref<2x8x256xf16>, %arg1: memref<2x8x256xf16>, %arg2: memref<2x8x256xf32>, %arg3: memref<2x8x256xf32>, %arg4: memref<2x8x256xf32>, %arg5: memref<2x8x256xf32>, %arg6: memref<2x8x256xf32>) kernel attributes {known_block_size = array<i32: 1024, 1, 1>, known_grid_size = array<i32: 4, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 1]>, dims = [0, 2]>} dense<32> : vector<8xindex>
      %c2048 = arith.constant 2048 : index
      %cst_0 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 1]>, dims = [1, 2]>} dense<256> : vector<4xindex>
      %cst_1 = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 32]>} dense<true> : vector<4x8x32xi1>
      %block_id_x_2 = gpu.block_id  x
      %0 = arith.shrsi %block_id_x_2, %c2 : index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.subi %block_id_x_2, %1 : index
      %3 = arith.shrsi %2, %c1 : index
      %4 = arith.muli %3, %c2 : index
      %5 = arith.subi %2, %4 : index
      %6 = arith.shrsi %3, %c1 : index
      %7 = arith.muli %6, %c2 : index
      %8 = arith.subi %3, %7 : index
      %9 = arith.muli %5, %c4 overflow<nsw> : index
      %10 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 1]>, dims = [1, 2]>} : vector<4xindex>
      %11 = vector.broadcast %9 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 1]>, dims = [1, 2]>} : index to vector<4xindex>
      %12 = arith.addi %11, %10 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 1]>, dims = [1, 2]>} : vector<4xindex>
      %13 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 1]>, dims = [0, 2]>} : vector<8xindex>
      %14 = arith.muli %13, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 1]>, dims = [0, 2]>} : vector<8xindex>
      %15 = vector.shape_cast %14 {layout_result_0 = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 1]>} : vector<8xindex> to vector<1x8x1xindex>
      %16 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 32]>, dims = [0, 1]>} : vector<32xindex>
      %17 = vector.broadcast %15 {layout_result_0 = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 32]>} : vector<1x8x1xindex> to vector<4x8x32xindex>
      %18 = vector.broadcast %16 {layout_result_0 = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 32]>} : vector<32xindex> to vector<4x8x32xindex>
      %19 = arith.addi %17, %18 {layout_result_0 = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 32]>} : vector<4x8x32xindex>
      %20 = arith.muli %8, %c2048 : index
      %21 = vector.broadcast %20 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 1]>, dims = [1, 2]>} : index to vector<4xindex>
      %22 = arith.muli %12, %cst_0 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 1]>, dims = [1, 2]>} : vector<4xindex>
      %23 = arith.addi %21, %22 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 1]>, dims = [1, 2]>} : vector<4xindex>
      %24 = vector.shape_cast %23 {layout_result_0 = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 1]>} : vector<4xindex> to vector<4x1x1xindex>
      %25 = vector.broadcast %24 {layout_result_0 = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 32]>} : vector<4x1x1xindex> to vector<4x8x32xindex>
      %26 = arith.addi %25, %19 {layout_result_0 = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 32]>} : vector<4x8x32xindex>
      %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<2x8x256xf16> -> index
      %27 = arith.index_cast %intptr : index to i64
      %28 = xegpu.load %27[%26], %cst_1  {layout = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 32]>} : i64, vector<4x8x32xindex>, vector<4x8x32xi1> -> vector<4x8x32xf16>
      %intptr_3 = memref.extract_aligned_pointer_as_index %arg1 : memref<2x8x256xf16> -> index
      %29 = arith.index_cast %intptr_3 : index to i64
      %30 = xegpu.load %29[%26], %cst_1  {layout = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 32]>} : i64, vector<4x8x32xindex>, vector<4x8x32xi1> -> vector<4x8x32xf16>
      %31 = arith.addf %28, %30 {layout_result_0 = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 32]>} : vector<4x8x32xf16>
      %32 = arith.extf %31 {layout_result_0 = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 32]>} : vector<4x8x32xf16> to vector<4x8x32xf32>
      %intptr_4 = memref.extract_aligned_pointer_as_index %arg4 : memref<2x8x256xf32> -> index
      %33 = arith.index_cast %intptr_4 : index to i64
      xegpu.store %32, %33[%26], %cst_1  {layout = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 32]>} : vector<4x8x32xf32>, i64, vector<4x8x32xindex>, vector<4x8x32xi1>
      %intptr_5 = memref.extract_aligned_pointer_as_index %arg2 : memref<2x8x256xf32> -> index
      %34 = arith.index_cast %intptr_5 : index to i64
      %35 = xegpu.load %34[%26], %cst_1  {layout = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 32]>} : i64, vector<4x8x32xindex>, vector<4x8x32xi1> -> vector<4x8x32xf32>
      %intptr_6 = memref.extract_aligned_pointer_as_index %arg3 : memref<2x8x256xf32> -> index
      %36 = arith.index_cast %intptr_6 : index to i64
      %37 = xegpu.load %36[%26], %cst_1  {layout = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 32]>} : i64, vector<4x8x32xindex>, vector<4x8x32xi1> -> vector<4x8x32xf32>
      %38 = arith.addf %35, %37 {layout_result_0 = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 32]>} : vector<4x8x32xf32>
      %intptr_7 = memref.extract_aligned_pointer_as_index %arg5 : memref<2x8x256xf32> -> index
      %39 = arith.index_cast %intptr_7 : index to i64
      xegpu.store %38, %39[%26], %cst_1  {layout = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 32]>} : vector<4x8x32xf32>, i64, vector<4x8x32xindex>, vector<4x8x32xi1>
      %40 = arith.addf %32, %38 {layout_result_0 = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 32]>} : vector<4x8x32xf32>
      %intptr_8 = memref.extract_aligned_pointer_as_index %arg6 : memref<2x8x256xf32> -> index
      %41 = arith.index_cast %intptr_8 : index to i64
      xegpu.store %40, %41[%26], %cst_1  {layout = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 32]>} : vector<4x8x32xf32>, i64, vector<4x8x32xindex>, vector<4x8x32xi1>
      gpu.return
    }
  }
}