// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp --runner imex-cpu-runner -e main --shared-libs=%mlir_runner_utils --entry-point-result=void | FileCheck %s
// RUN-GPU: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN-GPU:                                        --runner imex-cpu-runner -e main \
// RUN-GPU:                                        --entry-point-result=void \
// RUN-GPU:                                        --shared-libs=%mlir_runner_utils,%sycl_runtime | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module @jit_real.364 {

  func.func private @printMemrefF32(tensor<*xf32>)

  func.func private @callee(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xf32> {
    %0 = tensor.empty() : tensor<2xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<2xcomplex<f32>>) outs(%0 : tensor<2xf32>) {
    ^bb0(%arg1: complex<f32>, %arg2: f32):
      %2 = complex.re %arg1 : complex<f32>
      linalg.yield %2 : f32
    } -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }
  func.func @main() {
    %0 = arith.constant dense<[(0.1, -0.1), (0.2, -0.2)]> : tensor<2xcomplex<f32>>
    %3 = func.call @callee(%0) : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
    %unranked = tensor.cast %3 : tensor<2xf32> to tensor<*xf32>
    func.call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [2] strides = [1] data =
    //      CHECK: [0.1, 0.2]
    return
  }
}
