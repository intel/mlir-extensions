// RUN: imex-opt --split-input-file -convert-region-parallel-loops-to-gpu %s -verify-diagnostics -o -| FileCheck %s

// 2-d parallel loop mapped to block.y and block.x

// -----
func.func @test_convert_region_parloop_gpu(%arg0 : index, %arg1 : index, %arg2 : index,
                              %arg3 : index, %arg4 : index,
                              %buf : memref<?x?xf32>,
                              %res : memref<?x?xf32>) {
  %step = arith.constant 2 : index
  region.env_region #region.gpu_env<device = "test"> {
    scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                            step (%arg4, %step)  {
      %val = memref.load %buf[%i0, %i1] : memref<?x?xf32>
      memref.store %val, %res[%i1, %i0] : memref<?x?xf32>
    } { mapping = [#gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>, #gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>] }
    region.env_region_yield
  }
  return
}
// CHECK: test_convert_region_parloop_gpu
// CHECK: region.env_region #region.gpu_env<device = "test">
// CHECK: gpu.launch

// -----
func.func @test_convert_region_parloop_cpu(%arg0 : index, %arg1 : index, %arg2 : index,
                              %arg3 : index, %arg4 : index,
                              %buf : memref<?x?xf32>,
                              %res : memref<?x?xf32>) {
  %step = arith.constant 2 : index
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                          step (%arg4, %step)  {
    %val = memref.load %buf[%i0, %i1] : memref<?x?xf32>
    memref.store %val, %res[%i1, %i0] : memref<?x?xf32>
  } { mapping = [#gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>, #gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>] }
  return
}
// CHECK: test_convert_region_parloop_cpu
// CHECK: scf.parallel

// -----
func.func @test_convert_region_parloop_combined(%arg0 : index, %arg1 : index, %arg2 : index,
                              %arg3 : index, %arg4 : index,
                              %buf : memref<?x?xf32>,
                              %res : memref<?x?xf32>) {
  %step = arith.constant 2 : index
  region.env_region "something" {
    region.env_region #region.gpu_env<device = "test"> {
      scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                              step (%arg4, %step)  {
        %val = memref.load %buf[%i0, %i1] : memref<?x?xf32>
        memref.store %val, %res[%i1, %i0] : memref<?x?xf32>
      } { mapping = [#gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>, #gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>] }
      region.env_region_yield
    }

    scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                          step (%arg4, %step)  {
    %val = memref.load %buf[%i0, %i1] : memref<?x?xf32>
    memref.store %val, %res[%i1, %i0] : memref<?x?xf32>
  } { mapping = [#gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>, #gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>] }
  }
  return
}
// CHECK: test_convert_region_parloop_combined
// CHECK: region.env_region "something" {
// CHECK: region.env_region #region.gpu_env<device = "test">
// CHECK: gpu.launch
// CHECK: scf.parallel
