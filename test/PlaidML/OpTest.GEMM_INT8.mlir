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
module @gemm {
func.func @main() {
    %0= arith.constant dense<[[1, 1, 1], [1, 1, 2], [3, 3, 3]]>:tensor<3x3xi8>
    %1 = arith.constant dense<[[10, 11, 12], [13, 14, 15], [16, 17, 18]]>:tensor<3x3xi8>
    %2= arith.constant dense<[[1, 1, 1], [1, 1, 1], [1, 2, 3]]>:tensor<3x3xi8>
    %3 = call @test(%0,%1,%2) : (tensor<3x3xi8>,tensor<3x3xi8>,tensor<3x3xi8>) -> tensor<3x3xi8>
    %4 = call @castI8toI32(%3): (tensor<3x3xi8>) -> tensor<3x3xi32>
    %unranked = tensor.cast %4 : tensor<3x3xi32>to tensor<*xi32>
    call @printMemrefI32(%unranked) : (tensor<*xi32>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-NEXT: [40,   43,   46]
    // CHECK-NEXT: [56,   60,   64]
    // CHECK-NEXT: [118,   128,   138]
    return
}

func.func @castI8toI32(%arg0: tensor<3x3xi8>) -> tensor<3x3xi32> {
  %1 = tensor.empty() : tensor<3x3xi32>
  %2 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]}
       ins(%arg0: tensor<3x3xi8>)
       outs(%1 : tensor<3x3xi32>)
       attrs =  {iterator_ranges = [3, 3]} {
  ^bb0(%arg1: i8, %arg2: i32):
    %3 = arith.extui %arg1: i8 to i32
    linalg.yield %3 : i32
  } -> tensor<3x3xi32>
  return %2: tensor<3x3xi32>
}

func.func private @printMemrefI32(tensor<*xi32>) attributes { llvm.emit_c_interface }
func.func @test(%arg0: tensor<3x3xi8>, %arg1: tensor<3x3xi8>, %arg2: tensor<3x3xi8>) -> tensor<3x3xi8> {
    %c0_i8 = arith.constant 0 : i8
    %0 = tensor.empty() : tensor<3x3xi8>
    %1 = linalg.fill ins(%c0_i8 : i8) outs(%0 : tensor<3x3xi8>) -> tensor<3x3xi8>
    %2 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<3x3xi8>, tensor<3x3xi8>) outs(%1 : tensor<3x3xi8>) attrs =  {iterator_ranges = [3, 3, 3]} {
    ^bb0(%arg3: i8, %arg4: i8, %arg5: i8):
      %5 = arith.muli %arg3, %arg4 : i8
      %6 = arith.addi %arg5, %5 : i8
      linalg.yield %6 : i8
    } -> tensor<3x3xi8>
    %3 = tensor.empty() : tensor<3x3xi8>
    %4 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%2, %arg2 : tensor<3x3xi8>, tensor<3x3xi8>) outs(%3 : tensor<3x3xi8>) {
    ^bb0(%arg3: i8, %arg4: i8, %arg5: i8):
      %5 = arith.addi %arg3, %arg4 : i8
      linalg.yield %5 : i8
    } -> tensor<3x3xi8>
    return %4 : tensor<3x3xi8>
  }
}
