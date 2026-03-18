// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup"

// XFAIL: *

module attributes {gpu.container_module} {
  func.func @main(%arg0: memref<16x16xbf16>, %arg1: memref<16x16xbf16>, %arg2: memref<16x16xbf16>) attributes {L2Mem = 512 : i64, gemm_tiles_x = dense<1> : vector<4xi64>, gemm_tiles_y = dense<[1, -1, 1, 16]> : vector<4xi64>, habana_runner.num_inputs = 2 : i64, habana_runner.tests = [{inputs = [1.000000e+00 : bf16, 2.000000e+00 : bf16]}], linear_block_size = array<i32: 1024, 1, 1>, linear_grid_size = array<i32: 1, 1, 1>, region_partition = 1 : i64, region_size = 1 : i64, syn.fusion_successful, syn.gemm_pipeline, syn.tensor_signature = (tensor<16x16xbf16>, tensor<16x16xbf16>) -> tensor<16x16xbf16>, synFusionGenOps = 1 : i64, synFusionRequiredBeamSize = 1 : i64, synFusionTotalCost = 1.398000e+02 : f64} {
    %cst = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [16, 1], sg_data = [1, 32]>} dense<0.000000e+00> : vector<16x32xbf16>
    %cst_0 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [16, 1], sg_data = [1, 1]>, dims = [1]>} dense<16> : vector<16xindex>
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c1, %c1, %c1) threads in (%c512, %c1, %c1)  args(%arg0 : memref<16x16xbf16>, %arg1 : memref<16x16xbf16>, %arg2 : memref<16x16xbf16>)
    return
  }
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: memref<16x16xbf16>, %arg1: memref<16x16xbf16>, %arg2: memref<16x16xbf16>) kernel attributes {known_block_size = array<i32: 512, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
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
      %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [16, 1], sg_data = [1, 1]>, dims = [1]>} dense<16> : vector<16xindex>
      %cst_0 = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [16, 1], sg_data = [1, 32]>} dense<0.000000e+00> : vector<16x32xbf16>
      %0 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [16, 1], sg_data = [1, 1]>, dims = [1]>} : vector<16xindex>
      %1 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [16, 1], sg_data = [1, 32]>, dims = [0]>} : vector<32xindex>
      %2 = vector.broadcast %1 {layout_result_0 = #xegpu.layout<sg_layout = [16, 1], sg_data = [1, 32]>} : vector<32xindex> to vector<16x32xindex>
      %3 = arith.muli %0, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [16, 1], sg_data = [1, 1]>, dims = [1]>} : vector<16xindex>
      %4 = vector.shape_cast %3 {layout_result_0 = #xegpu.layout<sg_layout = [16, 1], sg_data = [1, 1]>} : vector<16xindex> to vector<16x1xindex>
      %5 = vector.broadcast %4 {layout_result_0 = #xegpu.layout<sg_layout = [16, 1], sg_data = [1, 32]>} : vector<16x1xindex> to vector<16x32xindex>
      %6 = arith.addi %5, %2 {layout_result_0 = #xegpu.layout<sg_layout = [16, 1], sg_data = [1, 32]>} : vector<16x32xindex>
      %7 = vector.constant_mask [16] {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [16, 1], sg_data = [1, 32]>, dims = [0]>} : vector<32xi1>
      %8 = vector.broadcast %7 {layout_result_0 = #xegpu.layout<sg_layout = [16, 1], sg_data = [1, 32]>} : vector<32xi1> to vector<16x32xi1>
      %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<16x16xbf16> -> index
      %9 = arith.index_cast %intptr : index to i64
      %10 = xegpu.load %9[%6], %8  {layout = #xegpu.layout<sg_layout = [16, 1], sg_data = [1, 32]>} : i64, vector<16x32xindex>, vector<16x32xi1> -> vector<16x32xbf16>
      %11 = arith.select %8, %10, %cst_0 {layout_result_0 = #xegpu.layout<sg_layout = [16, 1], sg_data = [1, 32]>} : vector<16x32xi1>, vector<16x32xbf16>
      %intptr_1 = memref.extract_aligned_pointer_as_index %arg1 : memref<16x16xbf16> -> index
      %12 = arith.index_cast %intptr_1 : index to i64
      %13 = xegpu.load %12[%6], %8  {layout = #xegpu.layout<sg_layout = [16, 1], sg_data = [1, 32]>} : i64, vector<16x32xindex>, vector<16x32xi1> -> vector<16x32xbf16>
      %14 = arith.select %8, %13, %cst_0 {layout_result_0 = #xegpu.layout<sg_layout = [16, 1], sg_data = [1, 32]>} : vector<16x32xi1>, vector<16x32xbf16>
      %15 = arith.addf %11, %14 {layout_result_0 = #xegpu.layout<sg_layout = [16, 1], sg_data = [1, 32]>} : vector<16x32xbf16>
      %intptr_2 = memref.extract_aligned_pointer_as_index %arg2 : memref<16x16xbf16> -> index
      %16 = arith.index_cast %intptr_2 : index to i64
      xegpu.store %15, %16[%6], %8  {layout = #xegpu.layout<sg_layout = [16, 1], sg_data = [1, 32]>, layout_operand_2 = #xegpu.layout<sg_layout = [16, 1], sg_data = [1, 32]>, layout_operand_3 = #xegpu.layout<sg_layout = [16, 1], sg_data = [1, 32]>} : vector<16x32xbf16>, i64, vector<16x32xindex>, vector<16x32xi1>
      gpu.return
    }
  }
}