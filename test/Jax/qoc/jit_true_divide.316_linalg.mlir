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
#map = affine_map<() -> ()>
module @jit_true_divide.316 {
  func.func private @callee(%arg0: tensor<complex<f32>>, %arg1: tensor<i32>) -> tensor<complex<f32>> {
    %0 = tensor.empty() : tensor<complex<f32>>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%arg1 : tensor<i32>) outs(%0 : tensor<complex<f32>>) {
    ^bb0(%arg2: i32, %arg3: complex<f32>):
      %4 = arith.sitofp %arg2 : i32 to f32
      %cst = arith.constant 0.000000e+00 : f32
      %5 = complex.create %4, %cst : complex<f32>
      linalg.yield %5 : complex<f32>
    } -> tensor<complex<f32>>
    %2 = tensor.empty() : tensor<complex<f32>>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%arg0, %1 : tensor<complex<f32>>, tensor<complex<f32>>) outs(%2 : tensor<complex<f32>>) {
    ^bb0(%arg2: complex<f32>, %arg3: complex<f32>, %arg4: complex<f32>):
      %4 = complex.div %arg2, %arg3 : complex<f32>
      linalg.yield %4 : complex<f32>
    } -> tensor<complex<f32>>
    return %3 : tensor<complex<f32>>
  }
  func.func @main() {
    %0 = arith.constant dense<(0.4, -0.4)> : tensor<complex<f32>>
    %1 = arith.constant dense<2> : tensor<i32>
    %3 = func.call @callee(%0, %1) : (tensor<complex<f32>>, tensor<i32>) -> tensor<complex<f32>>
    %cplx0 = tensor.extract %3[] : tensor<complex<f32>>
    %re0 = complex.re %cplx0 : complex<f32>
    %im0 = complex.im %cplx0 : complex<f32>
    vector.print %re0 : f32
    vector.print %im0 : f32
    //      CHECK: 0.2
    // CHECK-NEXT: -0.2
    return
  }
}
