// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp \
//                                            --runner mlir-cpu-runner -e main \
//                                            --shared-libs=%mlir_runner_utils \
//                                            --entry-point-result=void | FileCheck %s
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0 + 2, d1 + 1)>
module @explicit_padding {
func.func @test() {
    %0= arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>:tensor<2x3xf32>
    %1 = call @main(%0) : (tensor<2x3xf32>) -> tensor<6x5xf32>
    %unranked = tensor.cast %1 : tensor<6x5xf32>to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    return
}
func.func private @printMemrefF32(tensor<*xf32>)
func.func @main(%arg0: tensor<2x3xf32>)->tensor<6x5xf32>{
    %cst = arith.constant -1.000000e+00 : f32
    %0 = tensor.empty() : tensor<6x5xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<6x5xf32>) -> tensor<6x5xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<2x3xf32>) outs(%1 : tensor<6x5xf32>) attrs =  {iterator_ranges = [2, 3], name = "explicit_padding"} {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<6x5xf32>
    return %2 : tensor<6x5xf32>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [6, 5] strides = {{.*}} data =
// CHECK:   -1
// CHECK:   -1
// CHECK:   -1
// CHECK:   -1
// CHECK:   -1
// CHECK:   -1
// CHECK:   -1
// CHECK:   -1
// CHECK:   -1
// CHECK:   -1
// CHECK:   -1
// CHECK:   1
// CHECK:   2
// CHECK:   3
// CHECK:   -1
// CHECK:   -1
// CHECK:   4
// CHECK:   5
// CHECK:   6
// CHECK:   -1
// CHECK:   -1
// CHECK:   -1
// CHECK:   -1
// CHECK:   -1
// CHECK:   -1
// CHECK:   -1
// CHECK:   -1
// CHECK:   -1
// CHECK:   -1
// CHECK:   -1
