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
#map = affine_map<(d0, d1) -> (d0, d1)>
module @logical_or {
func.func @main() {
    %0= arith.constant dense<[[1, 2, 3], [4, 0, 6], [7, 0, 9]]>:tensor<3x3xi32>
    %1= arith.constant dense<[[10, 11, 12], [0, 0, 0], [16, 17, 18]]>:tensor<3x3xi32>
    %2 = call @test(%0,%1) : (tensor<3x3xi32>,tensor<3x3xi32>) -> tensor<3x3xi1>
    %3 = call @castI1toI32(%2): (tensor<3x3xi1>) -> tensor<3x3xi32>
    %unranked = tensor.cast %3 : tensor<3x3xi32>to tensor<*xi32>
    call @printMemrefI32(%unranked) : (tensor<*xi32>) -> ()
    return
}
func.func @castI1toI32(%arg0: tensor<3x3xi1>) -> tensor<3x3xi32> {
  %1 = tensor.empty() : tensor<3x3xi32>
  %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
       ins(%arg0: tensor<3x3xi1>)
       outs(%1 : tensor<3x3xi32>)
       attrs =  {iterator_ranges = [3, 3]} {
  ^bb0(%arg1: i1, %arg2: i32):
    %3 = arith.extui %arg1: i1 to i32
    linalg.yield %3 : i32
  } -> tensor<3x3xi32>
  return %2: tensor<3x3xi32>
}
func.func private @printMemrefI32(tensor<*xi32>)
func.func @test(%arg0: tensor<3x3xi32>, %arg1: tensor<3x3xi32>)->tensor<3x3xi1>{
    %0 = tensor.empty() : tensor<3x3xi1>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<3x3xi32>, tensor<3x3xi32>) outs(%0 : tensor<3x3xi1>) {
    ^bb0(%arg2: i32, %arg3: i32, %arg4: i1):
      %c0_i32 = arith.constant 0 : i32
      %2 = arith.cmpi ne, %arg2, %c0_i32 : i32
      %c0_i32_0 = arith.constant 0 : i32
      %3 = arith.cmpi ne, %arg3, %c0_i32_0 : i32
      %4 = arith.ori %2, %3 : i1
      linalg.yield %4 : i1
    } -> tensor<3x3xi1>
    return %1 : tensor<3x3xi1>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [3, 3] strides = {{.*}} data =
// CHECK:   1
// CHECK:   1
// CHECK:   1
// CHECK:   1
// CHECK:   0
// CHECK:   1
// CHECK:   1
// CHECK:   1
// CHECK:   1
