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
#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @double_dot {
func.func @test(%arg0: tensor<10x20xf32>, %arg1: tensor<20x30xf32>, %arg2: tensor<30x40xf32>) -> tensor<10x40xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<10x30xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<10x30xf32>) -> tensor<10x30xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<10x20xf32>, tensor<20x30xf32>) outs(%1 : tensor<10x30xf32>) attrs =  {iterator_ranges = [10, 30, 20]} {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %6 = arith.mulf %arg3, %arg4 : f32
      %7 = arith.addf %arg5, %6 : f32
      linalg.yield %7 : f32
    } -> tensor<10x30xf32>
    %3 = tensor.empty() : tensor<10x40xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<10x40xf32>) -> tensor<10x40xf32>
    %5 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2, %arg2 : tensor<10x30xf32>, tensor<30x40xf32>) outs(%4 : tensor<10x40xf32>) attrs =  {iterator_ranges = [10, 40, 30]} {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %6 = arith.mulf %arg3, %arg4 : f32
      %7 = arith.addf %arg5, %6 : f32
      linalg.yield %7 : f32
    } -> tensor<10x40xf32>
    return %5 : tensor<10x40xf32>
  }
  func.func @main() {
    %0 = arith.constant dense<1.0> : tensor<10x20xf32>
    %1 = arith.constant dense<2.0> : tensor<20x30xf32>
    %2 = arith.constant dense<3.0> : tensor<30x40xf32>
    %3 = call @test(%0, %1, %2) : (tensor<10x20xf32>, tensor<20x30xf32>, tensor<30x40xf32>) -> tensor<10x40xf32>
    %unranked = tensor.cast %3 : tensor<10x40xf32> to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-NEXT: [3600,   3600,   3600,   3600,   3600,   3600,   3600,   3600,   3600,   3600,   3600,   3600,   3600,   3600,   3600,   3600,   3600
    return
  }

  func.func private @printMemrefF32(%ptr : tensor<*xf32>)
}
