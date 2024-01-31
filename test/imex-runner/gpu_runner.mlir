// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/fullgpu.pp -e main --entry-point-result=void --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck

module {

  func.func private @printMemrefI32(tensor<*xi32>)
  func.func @main() {
    call @test_linspace() : () -> ()
    return
  }

  func.func @test_linspace() {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.600000e+01 : f64
    %c16 = arith.constant 16 : index
    %ar = ndarray.linspace %cst %cst_0 %c16 false : (f64, f64, index) -> !ndarray.ndarray<16xi32, #region.gpu_env<device = "gpu">>
    // %host = ndarray.copy %ar: !ndarray.ndarray<16xi32, #region.gpu_env<device = "gpu">> -> !ndarray.ndarray<16xi32>
    %t = "ndarray.to_tensor"(%ar) : (!ndarray.ndarray<16xi32, #region.gpu_env<device = "gpu">>) -> tensor<16xi32>
    %cast = tensor.cast %t : tensor<16xi32> to tensor<*xi32>
    call @printMemrefI32(%cast) : (tensor<*xi32>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [16] strides = [1] data =
    // CHECK-NEXT{LITERAL}: [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15]
    return
  }
}
