// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking %s -verify-diagnostics -o -| FileCheck %s

gpu.module @test_kernel {
    gpu.func @sg_atomic_rmw(%value: vector<32x64xf32>, %arg2: memref<65536xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<true> : vector<32x64xi1>
    %cst_0 = arith.constant dense<1> : vector<32x64xindex>
    %tile = xetile.init_tile %arg2, %cst_0 : memref<65536xf32>, vector<32x64xindex> -> !xetile.tile<32x64xf32, #xetile.tile_attr<scattered = true>>
    //CHECK-COUNT-128: {{.*}} = xetile.atomic_rmw addf {{.*}}, {{.*}} : vector<1x16xf32>, !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>> -> vector<1x16xf32>
    %rmw = xetile.atomic_rmw addf %value, %tile : vector<32x64xf32>, !xetile.tile<32x64xf32, #xetile.tile_attr<scattered = true>> -> vector<32x64xf32>
    xetile.store %rmw, %tile, %cst : vector<32x64xf32>, !xetile.tile<32x64xf32, #xetile.tile_attr<scattered = true>>, vector<32x64xi1>
  	gpu.return
  }
}
