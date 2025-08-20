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
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module @dot_f16 {
  func.func @test(%arg0: tensor<8x16xf16>, %arg1: tensor<16x32xf16>) -> tensor<8x32xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<8x32xf16>
    %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<8x32xf16>) -> tensor<8x32xf16>
    %2 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x16xf16>, tensor<16x32xf16>) outs(%1 : tensor<8x32xf16>) attrs =  {iterator_ranges = [8, 32, 16]} {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
      %3 = arith.mulf %arg2, %arg3 : f16
      %4 = arith.addf %arg4, %3 : f16
      linalg.yield %4 : f16
    } -> tensor<8x32xf16>
    return %2 : tensor<8x32xf16>
  }
  func.func @main() {
    %0 = arith.constant dense<1.0> : tensor<8x16xf16>
    %1 = arith.constant dense<2.0> : tensor<16x32xf16>
    %2 = call @test(%0, %1) : (tensor<8x16xf16>, tensor<16x32xf16>) -> tensor<8x32xf16>
    %3 = call @convertf16(%2) : (tensor<8x32xf16>) -> (tensor<8x32xf32>)
    %unranked = tensor.cast %3 : tensor<8x32xf32> to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-NEXT: [32,   32,   32,   32,   32,   32,   32,   32,   32,   32,   32,   32,   32,   32,   32,   32,   32,   32,   32,   32,   32,   32,
    return
  }
  func.func private @printMemrefF32(%ptr : tensor<*xf32>)
  func.func @convertf16(%arg0: tensor<8x32xf16>) -> tensor<8x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<8x32xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8x32xf32>) -> tensor<8x32xf32>
    %2 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<8x32xf16>) outs(%1 : tensor<8x32xf32>) attrs =  {iterator_ranges = [8, 32]} {
    ^bb0(%arg1: f16, %arg2: f32):
      %3 = arith.extf %arg1 : f16 to f32
      linalg.yield %3 : f32
    } -> tensor<8x32xf32>
    return %2 : tensor<8x32xf32>
  }
}
