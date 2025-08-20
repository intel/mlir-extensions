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
#map = affine_map<(d0, d1) -> (d0, d1)>
module @jit_conjugate.307 {
  func.func private @callee(%arg0: tensor<1x2xcomplex<f32>>) -> tensor<1x2xcomplex<f32>> {
    %0 = tensor.empty() : tensor<1x2xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x2xcomplex<f32>>) outs(%0 : tensor<1x2xf32>) {
    ^bb0(%arg1: complex<f32>, %arg2: f32):
      %8 = complex.re %arg1 : complex<f32>
      linalg.yield %8 : f32
    } -> tensor<1x2xf32>
    %2 = tensor.empty() : tensor<1x2xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x2xcomplex<f32>>) outs(%2 : tensor<1x2xf32>) {
    ^bb0(%arg1: complex<f32>, %arg2: f32):
      %8 = complex.im %arg1 : complex<f32>
      linalg.yield %8 : f32
    } -> tensor<1x2xf32>
    %4 = tensor.empty() : tensor<1x2xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<1x2xf32>) outs(%4 : tensor<1x2xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %8 = arith.negf %arg1 : f32
      linalg.yield %8 : f32
    } -> tensor<1x2xf32>
    %6 = tensor.empty() : tensor<1x2xcomplex<f32>>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %5 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%6 : tensor<1x2xcomplex<f32>>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: complex<f32>):
      %8 = complex.create %arg1, %arg2 : complex<f32>
      linalg.yield %8 : complex<f32>
    } -> tensor<1x2xcomplex<f32>>
    return %7 : tensor<1x2xcomplex<f32>>
  }
  func.func @main() {
    %cst0 = arith.constant 0 : index
    %cst1 = arith.constant 1 : index
    %0 = arith.constant dense<[[(0.1, -0.1), (0.2, -0.2)]]> : tensor<1x2xcomplex<f32>>
    %3 = func.call @callee(%0) : (tensor<1x2xcomplex<f32>>) -> tensor<1x2xcomplex<f32>>
    %cplx0 = tensor.extract %3[%cst0, %cst0] : tensor<1x2xcomplex<f32>>
    %cplx1 = tensor.extract %3[%cst0, %cst1] : tensor<1x2xcomplex<f32>>
    %re0 = complex.re %cplx0 : complex<f32>
    %im0 = complex.im %cplx0 : complex<f32>
    %re1 = complex.re %cplx1 : complex<f32>
    %im1 = complex.im %cplx1 : complex<f32>
    vector.print %re0 : f32
    vector.print %im0 : f32
    vector.print %re1 : f32
    vector.print %im1 : f32
    // CHECK: 0.1
    // CHECK: 0.1
    // CHECK: 0.2
    // CHECK: 0.2
    return
  }
}
