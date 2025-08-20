// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils \
// RUN:                                       --entry-point-result=void --filecheck
// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/linalg-to-llvm-caching.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/linalg-to-llvm-caching.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @gemm {
func.func @main() {
    %0= arith.constant dense<[[0.5, 0.2, 4.0], [1.0, 1.0, 2.0], [3.0, 3.0, 0.3]]>:tensor<3x3xf32>
    %1= arith.constant dense<[[1.0, 2.0, 3.0], [3.0, 4.0, 0.5], [3.0, 3.0, 3.0]]>:tensor<3x3xf32>
    %2= arith.constant dense<[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]>:tensor<3x3xf32>
    %lb = arith.constant 0 : index
    %ub = arith.constant 100 : index
    %step = arith.constant 1 : index
    scf.for %temp = %lb to %ub step %step {
      %3 = func.call @test(%0,%1,%2) : (tensor<3x3xf32>,tensor<3x3xf32>,tensor<3x3xf32>) -> tensor<3x3xf32>
      %unranked = tensor.cast %3 : tensor<3x3xf32> to tensor<*xf32>
      func.call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    }
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-NEXT: [14.1,   14.8,   14.6]
    // CHECK-NEXT: [11,   13,   10.5]
    // CHECK-NEXT: [13.9,   19.9,   12.4]
    return
}
func.func private @printMemrefF32(tensor<*xf32>)
func.func @test(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>, %arg2: tensor<3x3xf32>)->tensor<3x3xf32>{
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<3x3xf32>, tensor<3x3xf32>) outs(%arg2 : tensor<3x3xf32>) attrs =  {iterator_ranges = [3, 3, 3]} {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %2 = arith.mulf %arg3, %arg4 : f32
      %3 = arith.addf %arg5, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<3x3xf32>
    return %1 : tensor<3x3xf32>
  }
}
