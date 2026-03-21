// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup"

module attributes {gpu.container_module} {
  func.func @broadcast_entry(%arg0: memref<192xf32>, %arg1: memref<32x2x192xf32>, %arg2: memref<32x2x192xf32>) attributes {L2Mem = 0 : i64, gemm_tiles_x = dense<1> : vector<4xi64>, gemm_tiles_y = dense<[1, -1, 16, 24]> : vector<4xi64>, habana_runner.num_inputs = 2 : i64, habana_runner.tests = [{inputs = [dense<8.000000e+00> : tensor<192xf32>, dense<2.000000e+00> : tensor<32x2x192xf32>], outputs = [dense<1.000000e+01> : tensor<32x2x192xf32>]}], linear_block_size = array<i32: 1024, 1, 1>, linear_grid_size = array<i32: 16, 1, 1>, region_partition = 1 : i64, region_size = 16 : i64, syn.fusion_successful, syn.gemm_pipeline, syn.tensor_signature = (tensor<192xf32>, tensor<32x2x192xf32>) -> tensor<32x2x192xf32>, synFusionGenOps = 1 : i64, synFusionRequiredBeamSize = 1 : i64, synFusionTotalCost = 1.240800e+03 : f64} {
    %c2 = arith.constant 2 : index
    %cst = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} dense<true> : vector<2x2x6x32xi1>
    %cst_0 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>, dims = [0, 1, 3]>} dense<32> : vector<6xindex>
    %cst_1 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>, dims = [0, 1]>} dense<true> : vector<6x32xi1>
    %cst_2 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [1, 2, 3]>} dense<384> : vector<2xindex>
    %cst_3 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [0, 2, 3]>} dense<192> : vector<2xindex>
    %c16 = arith.constant 16 : index
    %c768 = arith.constant 768 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @broadcast::@broadcast blocks in (%c16, %c1, %c1) threads in (%c768, %c1, %c1)  args(%arg0 : memref<192xf32>, %arg1 : memref<32x2x192xf32>, %arg2 : memref<32x2x192xf32>)
    return
  }
  gpu.module @broadcast {
    gpu.func @broadcast(%arg0: memref<192xf32>, %arg1: memref<32x2x192xf32>, %arg2: memref<32x2x192xf32>) kernel attributes {known_block_size = array<i32: 768, 1, 1>, known_grid_size = array<i32: 16, 1, 1>} {
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
      %c16 = arith.constant 16 : index
      %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [0, 1, 3]>} dense<32> : vector<6xindex>
      %cst_0 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>, dims = [0, 1]>} dense<true> : vector<6x32xi1>
      %c2 = arith.constant 2 : index
      %cst_1 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [1, 2, 3]>} dense<384> : vector<2xindex>
      %cst_2 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [0, 2, 3]>} dense<192> : vector<2xindex>
      %cst_3 = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} dense<true> : vector<2x2x6x32xi1>
      %block_id_x_4 = gpu.block_id  x
      %c4 = arith.constant 4 : index
      %0 = arith.shrsi %block_id_x_4, %c4 : index
      %1 = arith.muli %0, %c16 : index
      %2 = arith.subi %block_id_x_4, %1 : index
      %3 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [0, 1, 3]>} : vector<6xindex>
      %4 = arith.muli %3, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [0, 1, 3]>} : vector<6xindex>
      %5 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>, dims = [0, 1, 2]>} : vector<32xindex>
      %6 = vector.shape_cast %4 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [0, 1]>} : vector<6xindex> to vector<6x1xindex>
      %7 = vector.broadcast %6 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>, dims = [0, 1]>} : vector<6x1xindex> to vector<6x32xindex>
      %8 = vector.broadcast %5 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>, dims = [0, 1]>} : vector<32xindex> to vector<6x32xindex>
      %9 = arith.addi %7, %8 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>, dims = [0, 1]>} : vector<6x32xindex>
      %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<192xf32> -> index
      %10 = arith.index_cast %intptr : index to i64
      %11 = xegpu.load %10[%9], %cst_0  {layout = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>, dims = [0, 1]>} : i64, vector<6x32xindex>, vector<6x32xi1> -> vector<6x32xf32>
      %12 = vector.broadcast %11 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : vector<6x32xf32> to vector<2x2x6x32xf32>
      %13 = arith.muli %2, %c2 overflow<nsw> : index
      %14 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [1, 2, 3]>} : vector<2xindex>
      %15 = vector.broadcast %13 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [1, 2, 3]>} : index to vector<2xindex>
      %16 = arith.addi %15, %14 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [1, 2, 3]>} : vector<2xindex>
      %17 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [0, 2, 3]>} : vector<2xindex>
      %18 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [0, 1, 3]>} : vector<6xindex>
      %19 = arith.muli %18, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [0, 1, 3]>} : vector<6xindex>
      %20 = vector.shape_cast %19 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>} : vector<6xindex> to vector<1x1x6x1xindex>
      %21 = vector.broadcast %20 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : vector<1x1x6x1xindex> to vector<2x2x6x32xindex>
      %22 = vector.broadcast %5 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : vector<32xindex> to vector<2x2x6x32xindex>
      %23 = arith.addi %21, %22 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : vector<2x2x6x32xindex>
      %24 = arith.muli %16, %cst_1 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [1, 2, 3]>} : vector<2xindex>
      %25 = vector.shape_cast %24 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>} : vector<2xindex> to vector<2x1x1x1xindex>
      %26 = vector.broadcast %25 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : vector<2x1x1x1xindex> to vector<2x2x6x32xindex>
      %27 = arith.muli %17, %cst_2 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [0, 2, 3]>} : vector<2xindex>
      %28 = vector.shape_cast %27 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>} : vector<2xindex> to vector<1x2x1x1xindex>
      %29 = vector.broadcast %28 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : vector<1x2x1x1xindex> to vector<2x2x6x32xindex>
      %30 = arith.addi %26, %29 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : vector<2x2x6x32xindex>
      %31 = arith.addi %30, %23 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : vector<2x2x6x32xindex>
      %intptr_5 = memref.extract_aligned_pointer_as_index %arg1 : memref<32x2x192xf32> -> index
      %32 = arith.index_cast %intptr_5 : index to i64
      %33 = xegpu.load %32[%31], %cst_3  {layout = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : i64, vector<2x2x6x32xindex>, vector<2x2x6x32xi1> -> vector<2x2x6x32xf32>
      %34 = arith.addf %12, %33 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : vector<2x2x6x32xf32>
      %intptr_6 = memref.extract_aligned_pointer_as_index %arg2 : memref<32x2x192xf32> -> index
      %35 = arith.index_cast %intptr_6 : index to i64
      xegpu.store %34, %35[%31], %cst_3  {layout = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>, layout_operand_2 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>, layout_operand_3 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : vector<2x2x6x32xf32>, i64, vector<2x2x6x32xindex>, vector<2x2x6x32xi1>
      gpu.return
    }
  }
}
