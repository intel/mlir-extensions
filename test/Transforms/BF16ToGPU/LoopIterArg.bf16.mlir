// RUN: imex-opt %s --bf16-to-gpu | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @sum_dim_list_bf16_5973807D_E60426A5 attributes {} {
    gpu.func @sum_dim_list_bf16_5973807D_E60426A5(%arg0: memref<2x3x2x10xbf16>, %arg1: memref<3x2x10xbf16>) kernel attributes {} {
      // CHECK: %[[CONST:.*]] = arith.constant {{.*}} : i16
      %cst = arith.constant 0.000000e+00 : bf16
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      // CHECK: %[[BITCAST:.*]] = arith.bitcast %[[CONST]] : i16 to bf16
      // CHECK: %[[EXTF:.*]] = arith.extf %[[BITCAST]] : bf16 to f32
      // CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[EXTF]]) -> (f32) {
      %0 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %cst) -> (bf16) {
        %1 = memref.load %arg0[%arg2, %thread_id_x, %thread_id_y, %thread_id_z] : memref<2x3x2x10xbf16>
        %2 = arith.addf %arg3, %1 : bf16

        // CHECK: scf.yield {{.*}} : f32
        scf.yield %2 : bf16
      } {lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 1>, step = 1 : index, upperBoundMap = affine_map<() -> (2)>}
      // CHECK: arith.truncf %{{.*}} : f32 to bf16
      // CHECK: arith.bitcast %{{.*}} : bf16 to i16
      memref.store %0, %arg1[%thread_id_x, %thread_id_y, %thread_id_z] : memref<3x2x10xbf16>
      gpu.return
    }
  }
}
