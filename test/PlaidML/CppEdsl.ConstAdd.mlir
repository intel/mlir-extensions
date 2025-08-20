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
#map = affine_map<(d0) -> (d0)>
module @const_add {
func.func @test(%arg0: tensor<4xi32> {stdx.const}, %arg1: tensor<4xi32> {stdx.const}) -> tensor<4xi32> {
    %0 = tensor.empty() : tensor<4xi32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<4xi32>, tensor<4xi32>) outs(%0 : tensor<4xi32>) {
    ^bb0(%arg2: i32, %arg3: i32, %arg4: i32):
      %2 = arith.addi %arg2, %arg3 : i32
      linalg.yield %2 : i32
    } -> tensor<4xi32>
    return %1 : tensor<4xi32>
  }
  func.func @main() {
    %0 = arith.constant dense<1> : tensor<4xi32>
    %1 = arith.constant dense<2> : tensor<4xi32>
    %2 = call @test(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    %unranked = tensor.cast %2 : tensor<4xi32> to tensor<*xi32>
    call @printMemrefI32(%unranked) : (tensor<*xi32>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-NEXT: [3,  3,  3,  3
    return
  }

  func.func private @printMemrefI32(%ptr : tensor<*xi32>)
}
