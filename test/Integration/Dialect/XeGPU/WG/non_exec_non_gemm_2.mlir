// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup"

module attributes {gpu.container_module} {
  func.func @unaligned_entry(%arg0: memref<50xf32>, %arg1: memref<50xf32>, %arg2: memref<50xf32>) attributes {L2Mem = 1024 : i64, gemm_tiles_x = dense<1> : vector<4xi64>, gemm_tiles_y = dense<[1, -1, 1, 2]> : vector<4xi64>, habana_runner.tests = [{inputs = [-2.000000e+00 : f32, 2.000000e+00 : f32], outputs = [1.000000e+00 : f32], tolerance = [1.000000e-02]}], linear_block_size = array<i32: 1024, 1, 1>, linear_grid_size = array<i32: 1, 1, 1>, region_partition = 1 : i64, region_size = 1 : i64, syn.fusion_successful, syn.gemm_pipeline, syn.tensor_signature = (tensor<50xf32>, tensor<50xf32>) -> tensor<50xf32>, synFusionGenOps = 1 : i64, synFusionRequiredBeamSize = 1 : i64, synFusionTotalCost = 144.66249999999999 : f64} {
    %cst = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [2, 1], sg_data = [1, 32]>} dense<50> : vector<2x32xindex>
    %cst_0 = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [2, 1], sg_data = [1, 32]>} dense<0.000000e+00> : vector<2x32xf32>
    %cst_1 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 1], sg_data = [1, 1]>, dims = [1]>} dense<32> : vector<2xindex>
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    gpu.launch_func  @unaligned::@unaligned blocks in (%c1, %c1, %c1) threads in (%c64, %c1, %c1)  args(%arg0 : memref<50xf32>, %arg1 : memref<50xf32>, %arg2 : memref<50xf32>)
    return
  }
  gpu.module @unaligned {
    gpu.func @unaligned(%arg0: memref<50xf32>, %arg1: memref<50xf32>, %arg2: memref<50xf32>) kernel attributes {known_block_size = array<i32: 64, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
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
      %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 1], sg_data = [1, 1]>, dims = [1]>} dense<32> : vector<2xindex>
      %cst_0 = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [2, 1], sg_data = [1, 32]>} dense<50> : vector<2x32xindex>
      %cst_1 = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [2, 1], sg_data = [1, 32]>} dense<0.000000e+00> : vector<2x32xf32>
      %0 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 1], sg_data = [1, 1]>, dims = [1]>} : vector<2xindex>
      %1 = arith.muli %0, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 1], sg_data = [1, 1]>, dims = [1]>} : vector<2xindex>
      %2 = vector.shape_cast %1 {layout_result_0 = #xegpu.layout<sg_layout = [2, 1], sg_data = [1, 1]>} : vector<2xindex> to vector<2x1xindex>
      %3 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 1], sg_data = [1, 32]>, dims = [0]>} : vector<32xindex>
      %4 = vector.broadcast %2 {layout_result_0 = #xegpu.layout<sg_layout = [2, 1], sg_data = [1, 32]>} : vector<2x1xindex> to vector<2x32xindex>
      %5 = vector.broadcast %3 {layout_result_0 = #xegpu.layout<sg_layout = [2, 1], sg_data = [1, 32]>} : vector<32xindex> to vector<2x32xindex>
      %6 = arith.addi %4, %5 {layout_result_0 = #xegpu.layout<sg_layout = [2, 1], sg_data = [1, 32]>} : vector<2x32xindex>
      %7 = arith.cmpi slt, %6, %cst_0 {layout_result_0 = #xegpu.layout<sg_layout = [2, 1], sg_data = [1, 32]>} : vector<2x32xindex>
      %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<50xf32> -> index
      %8 = arith.index_cast %intptr : index to i64
      %9 = xegpu.load %8[%6], %7  {layout = #xegpu.layout<sg_layout = [2, 1], sg_data = [1, 32]>} : i64, vector<2x32xindex>, vector<2x32xi1> -> vector<2x32xf32>
      %10 = arith.select %7, %9, %cst_1 {layout_result_0 = #xegpu.layout<sg_layout = [2, 1], sg_data = [1, 32]>} : vector<2x32xi1>, vector<2x32xf32>
      %intptr_2 = memref.extract_aligned_pointer_as_index %arg1 : memref<50xf32> -> index
      %11 = arith.index_cast %intptr_2 : index to i64
      %12 = xegpu.load %11[%6], %7  {layout = #xegpu.layout<sg_layout = [2, 1], sg_data = [1, 32]>} : i64, vector<2x32xindex>, vector<2x32xi1> -> vector<2x32xf32>
      %13 = arith.select %7, %12, %cst_1 {layout_result_0 = #xegpu.layout<sg_layout = [2, 1], sg_data = [1, 32]>} : vector<2x32xi1>, vector<2x32xf32>
      %14 = arith.addf %10, %13 {layout_result_0 = #xegpu.layout<sg_layout = [2, 1], sg_data = [1, 32]>} : vector<2x32xf32>
      %15 = math.cos %14 {layout_result_0 = #xegpu.layout<sg_layout = [2, 1], sg_data = [1, 32]>} : vector<2x32xf32>
      %intptr_3 = memref.extract_aligned_pointer_as_index %arg2 : memref<50xf32> -> index
      %16 = arith.index_cast %intptr_3 : index to i64
      xegpu.store %15, %16[%6], %7  {layout = #xegpu.layout<sg_layout = [2, 1], sg_data = [1, 32]>, layout_operand_2 = #xegpu.layout<sg_layout = [2, 1], sg_data = [1, 32]>, layout_operand_3 = #xegpu.layout<sg_layout = [2, 1], sg_data = [1, 32]>} : vector<2x32xf32>, i64, vector<2x32xindex>, vector<2x32xi1>
      gpu.return
    }
  }
}
