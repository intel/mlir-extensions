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
#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> ()>
#map2 = affine_map<() -> ()>
module @global_min {
func.func @main() {
    %0 = arith.constant dense<0.5> : tensor<10x10x10xf32>
    %1 = call @test(%0) : (tensor<10x10x10xf32>) -> tensor<f32>
    %unranked = tensor.cast %1 : tensor<f32> to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    // CHECK: [0.5]
    return
  }
func.func private @printMemrefF32(tensor<*xf32>)
func.func @test(%arg0: tensor<10x10x10xf32>) -> tensor<f32> {
    %cst = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<10x10x10xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<10x10x10xf32>) outs(%0 : tensor<10x10x10xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %7 = arith.negf %arg1 : f32
      linalg.yield %7 : f32
    } -> tensor<10x10x10xf32>
    %2 = tensor.empty() : tensor<f32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<f32>) -> tensor<f32>
    %4 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction", "reduction", "reduction"]} ins(%1 : tensor<10x10x10xf32>) outs(%3 : tensor<f32>) attrs =  {iterator_ranges = [10, 10, 10]} {
    ^bb0(%arg1: f32, %arg2: f32):
      %7 = arith.cmpf ogt, %arg2, %arg1 : f32
      %8 = arith.select %7, %arg2, %arg1 : f32
      linalg.yield %8 : f32
    } -> tensor<f32>
    %5 = tensor.empty() : tensor<f32>
    %6 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%4 : tensor<f32>) outs(%5 : tensor<f32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %7 = arith.negf %arg1 : f32
      linalg.yield %7 : f32
    } -> tensor<f32>
    return %6 : tensor<f32>
  }
}
