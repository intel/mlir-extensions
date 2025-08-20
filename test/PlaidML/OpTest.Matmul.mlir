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
module @linalg_matmul{
func.func @matmul(%arg0: tensor<5x3xf32>, %arg1: tensor<3x2xf32>) -> (tensor<5x2xf32>) {
  %cst = arith.constant 0.0 : f32
  %arg2 = tensor.empty() : tensor<5x2xf32>
  %output_tensor = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<5x2xf32>) -> tensor<5x2xf32>
  %result = linalg.matmul ins(%arg0, %arg1: tensor<5x3xf32>, tensor<3x2xf32>)
                outs(%output_tensor: tensor<5x2xf32>) -> tensor<5x2xf32>
  return %result : tensor<5x2xf32>
}

func.func private @printMemrefF32(tensor<*xf32>)
func.func @main() {
  %0 = arith.constant dense<13.0> :  tensor<5x3xf32>
  %1 = arith.constant dense<17.0> :  tensor<3x2xf32>
  %2 = call @matmul(%0, %1) : (tensor<5x3xf32>, tensor<3x2xf32>) -> tensor<5x2xf32>
  %unranked = tensor.cast %2 : tensor<5x2xf32> to tensor<*xf32>
  call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
  // CHECK:       [663,   663],
  // CHECK-NEXT:  [663,   663],
  // CHECK-NEXT:  [663,   663],
  // CHECK-NEXT:  [663,   663],
  // CHECK-NEXT:  [663,   663]
  return
}
}
