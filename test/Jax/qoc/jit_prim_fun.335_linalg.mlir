// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils \
// RUN:                                       --entry-point-result=void --filecheck
// RUN-GPU: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN-GPU:                                       --runner mlir-runner -e main \
// RUN-GPU:                                       --entry-point-result=void \
// RUN-GPU:                                       --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN-GPU: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN-GPU:                                        --runner mlir-runner -e main \
// RUN-GPU:                                        --entry-point-result=void \
// RUN-GPU:                                        --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
#map = affine_map<(d0) -> (d0)>
module @jit_prim_fun.335 {

  func.func private @printMemrefI32(tensor<*xi32>)

  func.func private @callee(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<2xi32> {
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<2xi32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<2xi32>) {
    ^bb0(%arg2: i32):
      %2 = linalg.index 0 : index
      %3 = linalg.index 0 : index
      %c0_0 = arith.constant 0 : index
      %4 = tensor.dim %arg0, %c0_0 : tensor<1xi32>
      %5 = arith.addi %c0, %4 : index
      %6 = arith.cmpi ult, %3, %5 : index
      %7 = scf.if %6 -> (i32) {
        %8 = arith.subi %3, %c0 : index
        %9 = tensor.extract %arg0[%8] : tensor<1xi32>
        scf.yield %9 : i32
      } else {
        %8 = arith.subi %3, %5 : index
        %9 = tensor.extract %arg1[%8] : tensor<1xi32>
        scf.yield %9 : i32
      }
      linalg.yield %7 : i32
    } -> tensor<2xi32>
    return %1 : tensor<2xi32>
  }
  func.func @main() {
    %0 = arith.constant dense<11> : tensor<1xi32>
    %1 = arith.constant dense<41> : tensor<1xi32>
    %3 = func.call @callee(%0, %1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %unranked = tensor.cast %3 : tensor<2xi32> to tensor<*xi32>
    func.call @printMemrefI32(%unranked) : (tensor<*xi32>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [2] strides = [1] data =
    //      CHECK: [11, 41]
    return
  }
}
