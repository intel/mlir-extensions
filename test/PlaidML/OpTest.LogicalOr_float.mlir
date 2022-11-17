// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp \
// RUN:                                       --runner mlir-cpu-runner -e main \
// RUN:                                       --shared-libs=%mlir_runner_utils \
// RUN:                                       --entry-point-result=void | FileCheck %s
// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN:                                       --runner mlir-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%mlir_runner_utils,%levelzero_runtime | FileCheck %s
#map = affine_map<(d0, d1) -> (d0, d1)>
module @logical_or {
func.func @main() {
    %0= arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 0.0, 6.0], [7.0, 0.0, 9.0]]>:tensor<3x3xf32>
    %1= arith.constant dense<[[10.0, 11.0, 12.0], [0.0, 0.0, 0.0], [16.0, 17.0, 18.0]]>:tensor<3x3xf32>
    %2 = call @test(%0,%1) : (tensor<3x3xf32>,tensor<3x3xf32>) -> tensor<3x3xi1>
    %unranked = tensor.cast %2 : tensor<3x3xi1>to tensor<*xi1>
    call @printMemrefI32(%unranked) : (tensor<*xi1>) -> ()
    return
}
func.func private @printMemrefI32(tensor<*xi1>)
func.func @test(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>)->tensor<3x3xi1>{
    %0 = tensor.empty() : tensor<3x3xi1>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<3x3xf32>, tensor<3x3xf32>) outs(%0 : tensor<3x3xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %cst = arith.constant 0.000000e+00 : f32
      %2 = arith.cmpf one, %arg2, %cst : f32
      %cst_0 = arith.constant 0.000000e+00 : f32
      %3 = arith.cmpf one, %arg3, %cst_0 : f32
      %4 = arith.ori %2, %3 : i1
      linalg.yield %4 : i1
    } -> tensor<3x3xi1>
    return %1 : tensor<3x3xi1>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [3, 3] strides = {{.*}} data =
// CHECK:   1
// CHECK:   1
// CHECK:   1
// CHECK:   1
// CHECK:   0
// CHECK:   1
// CHECK:   1
// CHECK:   1
// CHECK:   1
