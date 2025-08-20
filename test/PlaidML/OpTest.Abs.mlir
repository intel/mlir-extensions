// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils \
// RUN:                                       --entry-point-result=void --filecheck
// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> ()>
module @abs {
func.func @main() {
    %0= arith.constant dense<[[[[-9.0, 8.0, 0.0], [1.0, 5.0, 0.0], [1.0, 1.0, -7.0], [8.0, 2.0, 2.0]], [[8.0, 0.0, 4.0], [7.0, 5.0, 5.0], [8.0, -2.0, 0.0], [0.0, 9.0, -5.0]], [[4.0, 7.0, 2.0], [4.0, 5.0, 1.0], [-3.0, 3.0, 6.0], [8.0, 0.0, 1.0]], [[2.0, 8.0, 4.0], [0.0, 5.0, 5.0], [-6.0, -1.0, 1.0], [3.0, 3.0, 1.0]]]]>:tensor<1x4x4x3xf32>
    %1 = call @test(%0) : (tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
    %unranked = tensor.cast %1 : tensor<1x4x4x3xf32>to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    return
}
func.func private @printMemrefF32(tensor<*xf32>)
func.func @test(%arg0: tensor<1x4x4x3xf32>)->tensor<1x4x4x3xf32>{
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x4x4x3xi1>
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %cst : tensor<1x4x4x3xf32>, f32) outs(%0 : tensor<1x4x4x3xi1>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: i1):
      %6 = arith.cmpf olt, %arg1, %arg2 : f32
      linalg.yield %6 : i1
    } -> tensor<1x4x4x3xi1>
    %2 = tensor.empty() : tensor<1x4x4x3xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x4x4x3xf32>) outs(%2 : tensor<1x4x4x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %6 = arith.negf %arg1 : f32
      linalg.yield %6 : f32
    } -> tensor<1x4x4x3xf32>
    %4 = tensor.empty() : tensor<1x4x4x3xf32>
    %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %3, %arg0 : tensor<1x4x4x3xi1>, tensor<1x4x4x3xf32>, tensor<1x4x4x3xf32>) outs(%4 : tensor<1x4x4x3xf32>) {
    ^bb0(%arg1: i1, %arg2: f32, %arg3: f32, %arg4: f32):
      %6 = arith.select %arg1, %arg2, %arg3 : f32
      linalg.yield %6 : f32
    } -> tensor<1x4x4x3xf32>
    return %5 : tensor<1x4x4x3xf32>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [1, 4, 4, 3] strides = {{.*}} data =
// CHECK:   9
// CHECK:   8
// CHECK:   0
// CHECK:   1
// CHECK:   5
// CHECK:   0
// CHECK:   1
// CHECK:   1
// CHECK:   7
// CHECK:   8
// CHECK:   2
// CHECK:   2
// CHECK:   8
// CHECK:   0
// CHECK:   4
// CHECK:   7
// CHECK:   5
// CHECK:   5
// CHECK:   8
// CHECK:   2
// CHECK:   0
// CHECK:   0
// CHECK:   9
// CHECK:   5
// CHECK:   4
// CHECK:   7
// CHECK:   2
// CHECK:   4
// CHECK:   5
// CHECK:   1
// CHECK:   3
// CHECK:   3
// CHECK:   6
// CHECK:   8
// CHECK:   0
// CHECK:   1
// CHECK:   2
// CHECK:   8
// CHECK:   4
// CHECK:   0
// CHECK:   5
// CHECK:   5
// CHECK:   6
// CHECK:   1
// CHECK:   1
// CHECK:   3
// CHECK:   3
// CHECK:   1
