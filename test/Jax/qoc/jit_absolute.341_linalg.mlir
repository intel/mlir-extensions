// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp --runner mlir-cpu-runner -e main --shared-libs=%mlir_runner_utils --entry-point-result=void | FileCheck %s
// RUN-GPU: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN-GPU:                                        --runner mlir-cpu-runner -e main \
// RUN-GPU:                                        --entry-point-result=void \
// RUN-GPU:                                        --shared-libs=%mlir_runner_utils,%sycl_runtime | FileCheck %s

#map = affine_map<() -> ()>
module @jit_absolute.341 {

  func.func private @printMemrefI32(tensor<*xi32>)

  func.func private @callee(%arg0: tensor<i32>) -> tensor<i32> {
    %0 = tensor.empty() : tensor<i32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%arg0 : tensor<i32>) outs(%0 : tensor<i32>) {
    ^bb0(%arg1: i32, %arg2: i32):
      %c0_i32 = arith.constant 0 : i32
      %2 = arith.cmpi sge, %arg1, %c0_i32 : i32
      %3 = arith.subi %c0_i32, %arg1 : i32
      %4 = arith.select %2, %arg1, %3 : i32
      linalg.yield %4 : i32
    } -> tensor<i32>
    return %1 : tensor<i32>
  }
  func.func @main() {
    %0 = arith.constant dense<-17> : tensor<i32>
    %3 = func.call @callee(%0) : (tensor<i32>) -> tensor<i32>
    %unranked = tensor.cast %3 : tensor<i32> to tensor<*xi32>
    func.call @printMemrefI32(%unranked) : (tensor<*xi32>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 0 offset = 0 sizes = [] strides = [] data =
    //      CHECK: [17]
    return
  }
}
