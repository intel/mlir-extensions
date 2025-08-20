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
// RUN:                                        --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
#map = affine_map<(d0) -> (d0)>
module @unique_names {
func.func @main() {
    %0= arith.constant dense<[0.1]>:tensor<1xf32>
    %1= arith.constant dense<[0.2]>:tensor<1xf32>
    %2= arith.constant dense<[0.3]>:tensor<1xf32>
    %3= arith.constant dense<[0.3]>:tensor<1xf32>
    %4 = call @test(%0,%1,%2,%3) : (tensor<1xf32>,tensor<1xf32>,tensor<1xf32>,tensor<1xf32>) -> tensor<1xf32>
    %unranked = tensor.cast %4 : tensor<1xf32>to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    return
}
func.func private @printMemrefF32(tensor<*xf32>)
func.func @test(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1xf32>)->tensor<1xf32>{
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<1xf32>, tensor<1xf32>) outs(%0 : tensor<1xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %6 = arith.addf %arg4, %arg5 : f32
      linalg.yield %6 : f32
    } -> tensor<1xf32>
    %2 = tensor.empty() : tensor<1xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%1, %arg2 : tensor<1xf32>, tensor<1xf32>) outs(%2 : tensor<1xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %6 = arith.addf %arg4, %arg5 : f32
      linalg.yield %6 : f32
    } -> tensor<1xf32>
    %4 = tensor.empty() : tensor<1xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3, %arg3 : tensor<1xf32>, tensor<1xf32>) outs(%4 : tensor<1xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %6 = arith.addf %arg4, %arg5 : f32
      linalg.yield %6 : f32
    } -> tensor<1xf32>
    return %5 : tensor<1xf32>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [1] strides = {{.*}} data =
// CHECK:   0.9
