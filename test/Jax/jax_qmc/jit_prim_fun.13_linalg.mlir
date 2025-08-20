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
module @jit_prim_fun.13 {

  func.func private @printMemrefI32(tensor<*xi32>)

  func.func private @callee(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<2xi32> {
    %c0 = arith.constant 0 : index
    %2 = tensor.empty() : tensor<2xi32>
    %3 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%2 : tensor<2xi32>) {
    ^bb0(%arg2: i32):
      %5 = linalg.index 0 : index
      %6 = linalg.index 0 : index
      %c0_0 = arith.constant 0 : index
      %7 = tensor.dim %arg0, %c0_0 : tensor<1xi32>
      %8 = arith.addi %c0, %7 : index
      %9 = arith.cmpi ult, %6, %8 : index
      %10 = scf.if %9 -> (i32) {
        %11 = arith.subi %6, %c0 : index
        %12 = tensor.extract %arg0[%11] : tensor<1xi32>
        scf.yield %12 : i32
      } else {
        %11 = arith.subi %6, %8 : index
        %12 = tensor.extract %arg1[%11] : tensor<1xi32>
        scf.yield %12 : i32
      }
      linalg.yield %10 : i32
    } -> tensor<2xi32>
    return %3 : tensor<2xi32>
  }
  func.func @main() {
    %0 = arith.constant dense<1> : tensor<1xi32>
    %1 = arith.constant dense<3> : tensor<1xi32>
    %3 = func.call @callee(%0, %1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %unranked = tensor.cast %3 : tensor<2xi32> to tensor<*xi32>
    func.call @printMemrefI32(%unranked) : (tensor<*xi32>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [2] strides = [1] data =
    //      CHECK: [1, 3]
    return
  }
}
