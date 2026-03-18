// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup"

module attributes {gpu.container_module} {
  func.func @single_op_entry(%arg0: memref<256xf16>, %arg1: memref<256xf16>, %arg2: memref<256xf16>) attributes {L2Mem = 512 : i64, gemm_tiles_x = dense<1> : vector<4xi64>, gemm_tiles_y = dense<[1, -1, 1, 8]> : vector<4xi64>, habana_runner.tests = [{inputs = [2.400000e+01 : f16, 6.101560e+00 : f16], outputs = [3.009380e+01 : f16]}], linear_block_size = array<i32: 1024, 1, 1>, linear_grid_size = array<i32: 1, 1, 1>, region_partition = 1 : i64, region_size = 1 : i64, syn.fusion_successful, syn.gemm_pipeline, syn.tensor_signature = (tensor<256xf16>, tensor<256xf16>) -> tensor<256xf16>, synFusionGenOps = 1 : i64, synFusionRequiredBeamSize = 1 : i64, synFusionTotalCost = 7.320000e+01 : f64} {
    %cst = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [8, 1], sg_data = [1, 32]>} dense<true> : vector<8x32xi1>
    %cst_0 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [8, 1], sg_data = [1, 1]>, dims = [1]>} dense<32> : vector<8xindex>
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    gpu.launch_func  @single_op::@single_op blocks in (%c1, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<256xf16>, %arg1 : memref<256xf16>, %arg2 : memref<256xf16>)
    return
  }
  gpu.module @single_op {
    gpu.func @single_op(%arg0: memref<256xf16>, %arg1: memref<256xf16>, %arg2: memref<256xf16>) kernel attributes {known_block_size = array<i32: 256, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
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
      %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [8, 1], sg_data = [1, 1]>, dims = [1]>} dense<32> : vector<8xindex>
      %cst_0 = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [8, 1], sg_data = [1, 32]>} dense<true> : vector<8x32xi1>
      %0 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [8, 1], sg_data = [1, 1]>, dims = [1]>} : vector<8xindex>
      %1 = arith.muli %0, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [8, 1], sg_data = [1, 1]>, dims = [1]>} : vector<8xindex>
      %2 = vector.shape_cast %1 {layout_result_0 = #xegpu.layout<sg_layout = [8, 1], sg_data = [1, 1]>} : vector<8xindex> to vector<8x1xindex>
      %3 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [8, 1], sg_data = [1, 32]>, dims = [0]>} : vector<32xindex>
      %4 = vector.broadcast %2 {layout_result_0 = #xegpu.layout<sg_layout = [8, 1], sg_data = [1, 32]>} : vector<8x1xindex> to vector<8x32xindex>
      %5 = vector.broadcast %3 {layout_result_0 = #xegpu.layout<sg_layout = [8, 1], sg_data = [1, 32]>} : vector<32xindex> to vector<8x32xindex>
      %6 = arith.addi %4, %5 {layout_result_0 = #xegpu.layout<sg_layout = [8, 1], sg_data = [1, 32]>} : vector<8x32xindex>
      %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<256xf16> -> index
      %7 = arith.index_cast %intptr : index to i64
      %8 = xegpu.load %7[%6], %cst_0  {layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [1, 32]>} : i64, vector<8x32xindex>, vector<8x32xi1> -> vector<8x32xf16>
      %intptr_1 = memref.extract_aligned_pointer_as_index %arg1 : memref<256xf16> -> index
      %9 = arith.index_cast %intptr_1 : index to i64
      %10 = xegpu.load %9[%6], %cst_0  {layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [1, 32]>} : i64, vector<8x32xindex>, vector<8x32xi1> -> vector<8x32xf16>
      %11 = arith.addf %8, %10 {layout_result_0 = #xegpu.layout<sg_layout = [8, 1], sg_data = [1, 32]>} : vector<8x32xf16>
      %intptr_2 = memref.extract_aligned_pointer_as_index %arg2 : memref<256xf16> -> index
      %12 = arith.index_cast %intptr_2 : index to i64
      xegpu.store %11, %12[%6], %cst_0  {layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [1, 32]>, layout_operand_2 = #xegpu.layout<sg_layout = [8, 1], sg_data = [1, 32]>, layout_operand_3 = #xegpu.layout<sg_layout = [8, 1], sg_data = [1, 32]>} : vector<8x32xf16>, i64, vector<8x32xindex>, vector<8x32xi1>
      gpu.return
    }
  }
}