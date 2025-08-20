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
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
module @relu {
func.func @main() {
    %0= arith.constant dense<[[-0.1, -0.2, -0.3, 0.4, 0.5], [0.1, -0.2, 0.3, -0.4, 0.5], [0.1, 0.2, 0.3, -0.4, -0.5], [0.1, 0.2, 0.3, 0.4, 0.5]]>:tensor<4x5xf32>
    %1 = call @test(%0) : (tensor<4x5xf32>) -> tensor<4x5xf32>
    %unranked = tensor.cast %1 : tensor<4x5xf32>to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    return
}
func.func private @printMemrefF32(tensor<*xf32>)
func.func @test(%arg0: tensor<4x5xf32>)->tensor<4x5xf32>{
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x5xi1>
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst : tensor<4x5xf32>, f32) outs(%0 : tensor<4x5xi1>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: i1):
      %4 = arith.cmpf olt, %arg1, %arg2 : f32
      linalg.yield %4 : i1
    } -> tensor<4x5xi1>
    %2 = tensor.empty() : tensor<4x5xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map1, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%1, %cst, %arg0 : tensor<4x5xi1>, f32, tensor<4x5xf32>) outs(%2 : tensor<4x5xf32>) {
    ^bb0(%arg1: i1, %arg2: f32, %arg3: f32, %arg4: f32):
      %4 = arith.select %arg1, %arg2, %arg3 : f32
      linalg.yield %4 : f32
    } -> tensor<4x5xf32>
    return %3 : tensor<4x5xf32>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [4, 5] strides = {{.*}} data =
// CHECK:   0
// CHECK:   0
// CHECK:   0
// CHECK:   0.4
// CHECK:   0.5
// CHECK:   0.1
// CHECK:   0
// CHECK:   0.3
// CHECK:   0
// CHECK:   0.5
// CHECK:   0.1
// CHECK:   0.2
// CHECK:   0.3
// CHECK:   0
// CHECK:   0
// CHECK:   0.1
// CHECK:   0.2
// CHECK:   0.3
// CHECK:   0.4
// CHECK:   0.5
