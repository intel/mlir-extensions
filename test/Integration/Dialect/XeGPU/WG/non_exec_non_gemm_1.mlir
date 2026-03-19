// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup"

module attributes {gpu.container_module} {
  func.func @main(%arg0: memref<32x2x192xf32>, %arg1: memref<32x2x192xf32>, %arg2: memref<32x2x192xf32>) attributes {L2Mem = 0 : i64, gemm_tiles_x = dense<1> : vector<4xi64>, gemm_tiles_y = dense<[1, -1, 16, 24]> : vector<4xi64>, habana_runner.num_inputs = 2 : i64, habana_runner.tests = [{inputs = [dense<2.000000e+00> : tensor<32x2x192xf32>, dense<3.000000e+00> : tensor<32x2x192xf32>], outputs = [dense<2.000000e+00> : tensor<32x2x192xf32>]}], linear_block_size = array<i32: 1024, 1, 1>, linear_grid_size = array<i32: 16, 1, 1>, region_partition = 1 : i64, region_size = 16 : i64, syn.fusion_successful, syn.gemm_pipeline, syn.tensor_signature = (tensor<32x2x192xf32>, tensor<32x2x192xf32>) -> tensor<32x2x192xf32>, synFusionGenOps = 1 : i64, synFusionRequiredBeamSize = 1 : i64, synFusionTotalCost = 1.628800e+03 : f64} {
    %c2 = arith.constant 2 : index
    %cst = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} dense<true> : vector<2x2x6x32xi1>
    %cst_0 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [0, 1, 3]>} dense<32> : vector<6xindex>
    %cst_1 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [1, 2, 3]>} dense<384> : vector<2xindex>
    %cst_2 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [0, 2, 3]>} dense<192> : vector<2xindex>
    %c16 = arith.constant 16 : index
    %c768 = arith.constant 768 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c16, %c1, %c1) threads in (%c768, %c1, %c1)  args(%arg0 : memref<32x2x192xf32>, %arg1 : memref<32x2x192xf32>, %arg2 : memref<32x2x192xf32>)
    return
  }
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: memref<32x2x192xf32>, %arg1: memref<32x2x192xf32>, %arg2: memref<32x2x192xf32>) kernel attributes {known_block_size = array<i32: 768, 1, 1>, known_grid_size = array<i32: 16, 1, 1>} {
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
      %c2 = arith.constant 2 : index
      %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [0, 1, 3]>} dense<32> : vector<6xindex>
      %cst_0 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [1, 2, 3]>} dense<384> : vector<2xindex>
      %cst_1 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [0, 2, 3]>} dense<192> : vector<2xindex>
      %cst_2 = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} dense<true> : vector<2x2x6x32xi1>
      %block_id_x_3 = gpu.block_id  x
      %c4 = arith.constant 4 : index
      %0 = arith.shrsi %block_id_x_3, %c4 : index
      %1 = arith.muli %0, %c16 : index
      %2 = arith.subi %block_id_x_3, %1 : index
      %3 = arith.muli %2, %c2 overflow<nsw> : index
      %4 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [1, 2, 3]>} : vector<2xindex>
      %5 = vector.broadcast %3 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [1, 2, 3]>} : index to vector<2xindex>
      %6 = arith.addi %5, %4 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [1, 2, 3]>} : vector<2xindex>
      %7 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [0, 2, 3]>} : vector<2xindex>
      %8 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [0, 1, 3]>} : vector<6xindex>
      %9 = arith.muli %8, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [0, 1, 3]>} : vector<6xindex>
      %10 = vector.shape_cast %9 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>} : vector<6xindex> to vector<1x1x6x1xindex>
      %11 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>, dims = [0, 1, 2]>} : vector<32xindex>
      %12 = vector.broadcast %10 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : vector<1x1x6x1xindex> to vector<2x2x6x32xindex>
      %13 = vector.broadcast %11 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : vector<32xindex> to vector<2x2x6x32xindex>
      %14 = arith.addi %12, %13 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : vector<2x2x6x32xindex>
      %15 = arith.muli %6, %cst_0 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [1, 2, 3]>} : vector<2xindex>
      %16 = vector.shape_cast %15 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>} : vector<2xindex> to vector<2x1x1x1xindex>
      %17 = vector.broadcast %16 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : vector<2x1x1x1xindex> to vector<2x2x6x32xindex>
      %18 = arith.muli %7, %cst_1 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [0, 2, 3]>} : vector<2xindex>
      %19 = vector.shape_cast %18 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 1]>} : vector<2xindex> to vector<1x2x1x1xindex>
      %20 = vector.broadcast %19 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : vector<1x2x1x1xindex> to vector<2x2x6x32xindex>
      %21 = arith.addi %17, %20 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : vector<2x2x6x32xindex>
      %22 = arith.addi %21, %14 {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : vector<2x2x6x32xindex>
      %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<32x2x192xf32> -> index
      %23 = arith.index_cast %intptr : index to i64
      %24 = xegpu.load %23[%22], %cst_2  {layout = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : i64, vector<2x2x6x32xindex>, vector<2x2x6x32xi1> -> vector<2x2x6x32xf32>
      %intptr_4 = memref.extract_aligned_pointer_as_index %arg1 : memref<32x2x192xf32> -> index
      %25 = arith.index_cast %intptr_4 : index to i64
      %26 = xegpu.load %25[%22], %cst_2  {layout = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : i64, vector<2x2x6x32xindex>, vector<2x2x6x32xi1> -> vector<2x2x6x32xf32>
      %27 = arith.minimumf %24, %26 fastmath<nnan> {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : vector<2x2x6x32xf32>
      %intptr_5 = memref.extract_aligned_pointer_as_index %arg2 : memref<32x2x192xf32> -> index
      %28 = arith.index_cast %intptr_5 : index to i64
      xegpu.store %27, %28[%22], %cst_2  {layout = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>, layout_operand_2 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>, layout_operand_3 = #xegpu.layout<sg_layout = [2, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : vector<2x2x6x32xf32>, i64, vector<2x2x6x32xindex>, vector<2x2x6x32xi1>
      gpu.return
    }
  }
}
