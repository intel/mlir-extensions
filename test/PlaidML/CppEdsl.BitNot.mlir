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
module @bit_not {
func.func @main() {
    %0= arith.constant dense<[[0, 1, 2], [16, 17, 34], [240, 15, 255]]>:tensor<3x3xi8>
    %1 = call @test(%0) : (tensor<3x3xi8>) -> tensor<3x3xi8>
    %2 = call @castI8toI32(%1): (tensor<3x3xi8>) -> tensor<3x3xi32>
    %unranked = tensor.cast %2 : tensor<3x3xi32> to tensor<*xi32>
    call @printMemrefI32(%unranked) : (tensor<*xi32>) -> ()
    return
}

func.func @castI8toI32(%arg0: tensor<3x3xi8>) -> tensor<3x3xi32> {
  %1 = tensor.empty() : tensor<3x3xi32>
  %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
       ins(%arg0: tensor<3x3xi8>)
       outs(%1 : tensor<3x3xi32>)
       attrs =  {iterator_ranges = [3, 3]} {
  ^bb0(%arg1: i8, %arg2: i32):
    %3 = arith.extui %arg1: i8 to i32
    linalg.yield %3 : i32
  } -> tensor<3x3xi32>
  return %2: tensor<3x3xi32>
}

func.func private @printMemrefI32(tensor<*xi32>)
func.func @test(%arg0: tensor<3x3xi8>)->tensor<3x3xi8>{
    %0 = tensor.empty() : tensor<3x3xi8>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<3x3xi8>) outs(%0 : tensor<3x3xi8>) {
    ^bb0(%arg1: i8, %arg2: i8):
      %c-1_i8 = arith.constant -1 : i8
      %2 = arith.subi %c-1_i8, %arg1 : i8
      linalg.yield %2 : i8
    } -> tensor<3x3xi8>
    return %1 : tensor<3x3xi8>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [3, 3] strides = {{.*}} data =
// CHECK:   255
// CHECK:   254
// CHECK:   253
// CHECK:   239
// CHECK:   238
// CHECK:   221
// CHECK:   15
// CHECK:   240
// CHECK:   0
