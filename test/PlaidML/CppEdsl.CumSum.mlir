
// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp \
// RUN:                                       --runner mlir-cpu-runner -e main \
// RUN:                                       --shared-libs=%mlir_runner_utils \
// RUN:                                       --entry-point-result=void | FileCheck %s
// RUN-GPU: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN-GPU:                                        --runner mlir-cpu-runner -e main \
// RUN-GPU:                                        --entry-point-result=void \
// RUN-GPU:                                        --shared-libs=%mlir_runner_utils,%sycl_runtime | FileCheck %s
#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#set = affine_set<(d0, d1) : (d0 - d1 >= 0)>
module @cumsum {
func.func @test(%arg0: tensor<10xf32>) -> tensor<10xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<10xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<10xf32>) -> tensor<10xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<10xf32>) outs(%1 : tensor<10xf32>) attrs =  {constraints = #set, iterator_ranges = [10, 10]} {
    ^bb0(%arg1: f32, %arg2: f32):
      %3 = arith.addf %arg2, %arg1 : f32
      linalg.yield %3 : f32
    } -> tensor<10xf32>
    return %2 : tensor<10xf32>
  }
func.func @main() {
    %0= arith.constant dense<1.0>:tensor<10xf32>
    %1 = call @test(%0) : (tensor<10xf32>) -> tensor<10xf32>
    %unranked = tensor.cast %1 : tensor<10xf32>to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    return
}
func.func private @printMemrefF32(tensor<*xf32>)
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-NEXT: [10,  10,  10,  10,  10,  10,  10,  10,  10,  10
}
