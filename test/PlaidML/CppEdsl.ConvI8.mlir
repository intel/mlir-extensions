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
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
module @convolution {
  func.func @test(%arg0: tensor<1x224x224x3xi8>, %arg1: tensor<3x3x3x32xi8>) -> tensor<1x224x224x32xi8> {
    %c0_i8 = arith.constant 0 : i8
    %0 = tensor.empty() : tensor<1x224x224x3xi8>
    %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x224x224x3xi8>) outs(%0 : tensor<1x224x224x3xi8>) {
    ^bb0(%arg2: i8, %arg3: i8):
      linalg.yield %arg2 : i8
    } -> tensor<1x224x224x3xi8>
    %c0_i8_0 = arith.constant 0 : i8
    %2 = tensor.pad %1 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
      tensor.yield %c0_i8_0 : i8
    } : tensor<1x224x224x3xi8> to tensor<1x226x226x3xi8>
    %3 = tensor.empty() : tensor<1x224x224x32xi8>
    %4 = linalg.fill ins(%c0_i8 : i8) outs(%3 : tensor<1x224x224x32xi8>) -> tensor<1x224x224x32xi8>
    %5 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%2, %arg1 : tensor<1x226x226x3xi8>, tensor<3x3x3x32xi8>) outs(%4 : tensor<1x224x224x32xi8>) attrs =  {iterator_ranges = [1, 224, 224, 32, 3, 3, 3]} {
    ^bb0(%arg2: i8, %arg3: i8, %arg4: i8):
      %6 = arith.muli %arg2, %arg3 : i8
      %7 = arith.addi %arg4, %6 : i8
      linalg.yield %7 : i8
    } -> tensor<1x224x224x32xi8>
    return %5 : tensor<1x224x224x32xi8>
  }

  func.func @main() {
    %0 = arith.constant dense<1> : tensor<1x224x224x3xi8>
    %1 = arith.constant dense<1> : tensor<3x3x3x32xi8>
    %2 = call @test(%0, %1) : (tensor<1x224x224x3xi8>, tensor<3x3x3x32xi8>) -> tensor<1x224x224x32xi8>
    %3 = call @castI8toI32(%2): (tensor<1x224x224x32xi8>) -> tensor<1x224x224x32xi32>
    %unranked = tensor.cast %3 : tensor<1x224x224x32xi32> to tensor<*xi32>
    call @printMemrefI32(%unranked) : (tensor<*xi32>) -> ()
    return
  }

  func.func @castI8toI32(%arg0: tensor<1x224x224x32xi8>) -> tensor<1x224x224x32xi32> {
  %1 = tensor.empty() : tensor<1x224x224x32xi32>
  %2 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
       ins(%arg0: tensor<1x224x224x32xi8>)
       outs(%1 : tensor<1x224x224x32xi32>)
       attrs =  {iterator_ranges = [1, 224, 224, 32]} {
  ^bb0(%arg1: i8, %arg2: i32):
    %3 = arith.extui %arg1: i8 to i32
    linalg.yield %3 : i32
  } -> tensor<1x224x224x32xi32>
  return %2: tensor<1x224x224x32xi32>
}

  //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
  // CHECK-NEXT: [12, 12]
  func.func private @printMemrefI32(tensor<*xi32>)
}
