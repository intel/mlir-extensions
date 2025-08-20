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
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module @gemv {
func.func @test(%arg0: tensor<3x3xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>)->tensor<3xf32>{
    %0 = tensor.empty() : tensor<3xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<3x3xf32>, tensor<3xf32>) outs(%arg2 : tensor<3xf32>) attrs =  {iterator_ranges = [3, 3]} {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %2 = arith.mulf %arg3, %arg4 : f32
      %3 = arith.addf %arg5, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<3xf32>
    return %1 : tensor<3xf32>
  }
func.func @main() {
    %0= arith.constant dense<[[1.0, 0.5, 1.0], [1.0, 1.0, 1.0], [1.0, 0.6, 1.0]]>:tensor<3x3xf32>
    %1= arith.constant dense<[1.0, 1.0, 1.0]>:tensor<3xf32>
    %2= arith.constant dense<[1.0, 1.0, 1.0]>:tensor<3xf32>
    %3 = call @test(%0,%1,%2) : (tensor<3x3xf32>,tensor<3xf32>,tensor<3xf32>) -> tensor<3xf32>
    %unranked = tensor.cast %3 : tensor<3xf32>to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    return
}
func.func private @printMemrefF32(tensor<*xf32>)
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [3] strides = {{.*}} data =
// CHECK:   3.5
// CHECK:   4
// CHECK:   3.6
