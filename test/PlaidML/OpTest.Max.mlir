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
#map1 = affine_map<(d0, d1) -> (d0)>
module @max {
func.func @main() {
    %0= arith.constant dense<[[-5.0, -6.0, -7.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]>:tensor<3x3xf32>
    %1 = call @test(%0) : (tensor<3x3xf32>) -> tensor<3xf32>
    %unranked = tensor.cast %1 : tensor<3xf32>to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    return
}
func.func private @printMemrefF32(tensor<*xf32>)
func.func @test(%arg0: tensor<3x3xf32>)->tensor<3xf32>{
    %cst = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<3xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<3xf32>) -> tensor<3xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<3x3xf32>) outs(%1 : tensor<3xf32>) attrs =  {iterator_ranges = [3, 3]} {
    ^bb0(%arg1: f32, %arg2: f32):
      %3 = arith.cmpf ogt, %arg2, %arg1 : f32
      %4 = arith.select %3, %arg2, %arg1 : f32
      linalg.yield %4 : f32
    } -> tensor<3xf32>
    return %2 : tensor<3xf32>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [3] strides = {{.*}} data =
// CHECK:   -5
// CHECK:   6
// CHECK:   9
