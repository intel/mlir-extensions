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
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module @jit__reduce_sum.357 {

  func.func private @printMemrefI32(tensor<*xi32>)

  func.func private @callee(%arg0: tensor<1xi32>) -> tensor<i32> {
    %cst = arith.constant dense<0> : tensor<i32>
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<i32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<i32>) -> tensor<i32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction"]} ins(%arg0 : tensor<1xi32>) outs(%1 : tensor<i32>) {
    ^bb0(%arg1: i32, %arg2: i32):
      %3 = arith.addi %arg1, %arg2 : i32
      linalg.yield %3 : i32
    } -> tensor<i32>
    return %2 : tensor<i32>
  }
  func.func @main() {
    %0 = arith.constant dense<[-10]> : tensor<1xi32>
    %3 = func.call @callee(%0) : (tensor<1xi32>) -> tensor<i32>
    %unranked = tensor.cast %3 : tensor<i32> to tensor<*xi32>
    func.call @printMemrefI32(%unranked) : (tensor<*xi32>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 0 offset = 0 sizes = [] strides = [] data =
    //      CHECK: [-10]
    return
  }
}
