// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp \
// RUN:                                        --runner mlir-cpu-runner -e main \
// RUN:                                        --shared-libs=%mlir_runner_utils \
// RUN:                                        --entry-point-result=void | FileCheck %s
// RUN-GPU: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN-GPU:                                        --runner mlir-cpu-runner -e main \
// RUN-GPU:                                        --entry-point-result=void \
// RUN-GPU:                                        --shared-libs=%mlir_runner_utils,%sycl_runtime | FileCheck %s
#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module @gemm {
func.func @main() {
    %0= arith.constant dense<[[1, 1, 1], [1, 1, 2], [3, 3, 3]]>:tensor<3x3xi32>
    %1= arith.constant dense<[[10, 11, 12], [13, 14, 15], [16, 17, 18]]>:tensor<3x3xi32>
    %2= arith.constant dense<[[1, 1, 1], [1, 1, 1], [1, 1, 1]]>:tensor<3x3xi32>
    %3 = call @test(%0,%1,%2) : (tensor<3x3xi32>,tensor<3x3xi32>,tensor<3x3xi32>) -> tensor<3x3xi32>
    %unranked = tensor.cast %3 : tensor<3x3xi32>to tensor<*xi32>
    call @printMemrefI32(%unranked) : (tensor<*xi32>) -> ()
     // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-NEXT: [40,   43,   46]
    // CHECK-NEXT: [56,   60,   64]
    // CHECK-NEXT: [118,   127,   136]
    return
}
func.func private @printMemrefI32(tensor<*xi32>)
func.func @test(%arg0: tensor<3x3xi32>, %arg1: tensor<3x3xi32>, %arg2: tensor<3x3xi32>)->tensor<3x3xi32>{
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<3x3xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<3x3xi32>) -> tensor<3x3xi32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<3x3xi32>, tensor<3x3xi32>) outs(%1 : tensor<3x3xi32>) attrs =  {iterator_ranges = [3, 3, 3]} {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %5 = arith.muli %arg3, %arg4 : i32
      %6 = arith.addi %arg5, %5 : i32
      linalg.yield %6 : i32
    } -> tensor<3x3xi32>
    %3 = tensor.empty() : tensor<3x3xi32>
    %4 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%2, %arg2 : tensor<3x3xi32>, tensor<3x3xi32>) outs(%3 : tensor<3x3xi32>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %5 = arith.addi %arg3, %arg4 : i32
      linalg.yield %5 : i32
    } -> tensor<3x3xi32>
    return %4 : tensor<3x3xi32>
  }
}
