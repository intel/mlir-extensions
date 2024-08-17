// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils \
// RUN:                                       --entry-point-result=void --filecheck
// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
module @relu {
func.func @main() {
    %0= arith.constant dense<[[-0.1, -0.2, -0.3, 0.4, 0.5], [0.1, -0.2, 0.3, -0.4, 0.5], [0.1, 0.2, 0.3, -0.4, -0.5], [0.1, 0.2, 0.3, 0.4, 0.5]]>:tensor<4x5xbf16>
    %1 = call @test(%0) : (tensor<4x5xbf16>) -> tensor<4x5xbf16>
    %unranked = tensor.cast %1 : tensor<4x5xbf16>to tensor<*xbf16>
    call @printMemrefBF16(%unranked) : (tensor<*xbf16>) -> ()
    return
}
func.func private @printMemrefBF16(tensor<*xbf16>)
func.func @test(%arg0: tensor<4x5xbf16>)->tensor<4x5xbf16>{
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<4x5xi1>
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst : tensor<4x5xbf16>, bf16) outs(%0 : tensor<4x5xi1>) {
    ^bb0(%arg1: bf16, %arg2: bf16, %arg3: i1):
      %arg1_f32 = arith.extf %arg1 : bf16 to f32
      %arg2_f32 = arith.extf %arg2 : bf16 to f32
      %4 = arith.cmpf olt, %arg1_f32, %arg2_f32 : f32
      linalg.yield %4 : i1
    } -> tensor<4x5xi1>
    %2 = tensor.empty() : tensor<4x5xbf16>
    %3 = linalg.generic {indexing_maps = [#map0, #map1, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%1, %cst, %arg0 : tensor<4x5xi1>, bf16, tensor<4x5xbf16>) outs(%2 : tensor<4x5xbf16>) {
    ^bb0(%arg1: i1, %arg2: bf16, %arg3: bf16, %arg4: bf16):
      %4 = arith.select %arg1, %arg2, %arg3 : bf16
      linalg.yield %4 : bf16
    } -> tensor<4x5xbf16>
    return %3 : tensor<4x5xbf16>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [4, 5] strides = {{.*}} data =
// CHECK:   0
// CHECK:   0
// CHECK:   0
// CHECK:   0.4
// CHECK:   0.5
// CHECK:   0.1
// CHECK:   0
// CHECK:   0.3
// CHECK:   0
// CHECK:   0.5
// CHECK:   0.1
// CHECK:   0.2
// CHECK:   0.3
// CHECK:   0
// CHECK:   0
// CHECK:   0.1
// CHECK:   0.2
// CHECK:   0.3
// CHECK:   0.4
// CHECK:   0.5
