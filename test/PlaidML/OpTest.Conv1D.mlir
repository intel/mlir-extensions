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
#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 + d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
module @conv_1d {
  func.func @test(%arg0: tensor<1x14x3xf32>, %arg1: tensor<3x3x1xf32>) -> tensor<1x12x1xf32> {
    %c0 = arith.constant 0.0 : f32
    %0 = tensor.empty() : tensor<1x12x1xf32>
    %1 = linalg.fill ins(%c0 : f32) outs(%0 : tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2 = linalg.generic {
            indexing_maps = [#map0, #map1, #map2],
            iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
          }
          ins(%arg0, %arg1 : tensor<1x14x3xf32>, tensor<3x3x1xf32>)
          outs(%1 : tensor<1x12x1xf32>)
          attrs =  {iterator_ranges = [1, 12, 1, 3, 3]} {
            ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
              %3 = arith.mulf %arg2, %arg3 : f32
              %4 = arith.addf %arg4, %3 : f32
              linalg.yield %4 : f32
          } -> tensor<1x12x1xf32>
    return %2 : tensor<1x12x1xf32>
    }

  func.func @main() {
    %0 = arith.constant dense<[[[1.0, 2.0, 3.0],
                                [2.0, 3.0, 4.0],
                                [3.0, 4.0, 5.0],
                                [4.0, 5.0, 6.0],
                                [5.0, 6.0, 7.0],
                                [6.0, 7.0, 8.0],
                                [7.0, 8.0, 9.0],
                                [8.0, 9.0, 0.0],
                                [9.0, 0.0, 1.0],
                                [0.0, 1.0, 2.0],
                                [1.0, 2.0, 3.0],
                                [2.0, 3.0, 4.0],
                                [3.0, 4.0, 5.0],
                                [4.0, 5.0, 6.0]]]> : tensor<1x14x3xf32>
    %1 = arith.constant dense<[[[1.0], [2.0], [1.0]], [[2.0], [4.0], [2.0]], [[3.0], [5.0], [3.0]]]> : tensor<3x3x1xf32>
    %2 = call @test(%0, %1) : (tensor<1x14x3xf32>, tensor<3x3x1xf32>) -> tensor<1x12x1xf32>
    %unranked = tensor.cast %2 : tensor<1x12x1xf32> to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    return
  }

  func.func private @printMemrefF32(%ptr : tensor<*xf32>)
}
//      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
// CHECK-NEXT: [76
// CHECK-NEXT: 99
// CHECK-NEXT: 122
// CHECK-NEXT: 145
// CHECK-NEXT: 168
// CHECK-NEXT: 161
// CHECK-NEXT: 114
// CHECK-NEXT: 57
// CHECK-NEXT: 40
// CHECK-NEXT: 53
// CHECK-NEXT: 76
// CHECK-NEXT: 99
