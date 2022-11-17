// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp \
// RUN:                                            --runner mlir-cpu-runner -e main \
// RUN:                                            --shared-libs=%mlir_runner_utils \
// RUN:                                            --entry-point-result=void | FileCheck %s
// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN:                                            --runner mlir-cpu-runner -e main \
// RUN:                                            --entry-point-result=void \
// RUN:                                            --shared-libs=%mlir_runner_utils,%levelzero_runtime | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0) -> (d0)>
module @gemv {
func.func @main() {
    %0= arith.constant dense<[[1, 2, 3], [1, 1, 1], [1, 1, 1]]>:tensor<3x3xi8>
    %1= arith.constant dense<[1, 1, 1]>:tensor<3xi8>
    %2= arith.constant dense<[1, 1, 1]>:tensor<3xi8>
    %3 = call @test(%0,%1,%2) : (tensor<3x3xi8>,tensor<3xi8>,tensor<3xi8>) -> tensor<3xi8>
    %unranked = tensor.cast %3 : tensor<3xi8>to tensor<*xi8>
    call @printMemrefI8(%unranked) : (tensor<*xi8>) -> ()
    // CHECK:
    return
}
func.func private @printMemrefI8(tensor<*xi8>)
func.func @test(%arg0: tensor<3x3xi8>, %arg1: tensor<3xi8>, %arg2: tensor<3xi8>) -> tensor<3xi8> {
    %c0_i8 = arith.constant 0 : i8
    %0 = tensor.empty() : tensor<3xi8>
    %1 = linalg.fill ins(%c0_i8 : i8) outs(%0 : tensor<3xi8>) -> tensor<3xi8>
    %2 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<3x3xi8>, tensor<3xi8>) outs(%1 : tensor<3xi8>) attrs =  {iterator_ranges = [3, 3]} {
    ^bb0(%arg3: i8, %arg4: i8, %arg5: i8):
      %5 = arith.muli %arg3, %arg4 : i8
      %6 = arith.addi %arg5, %5 : i8
      linalg.yield %6 : i8
    } -> tensor<3xi8>
    %3 = tensor.empty() : tensor<3xi8>
    %4 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%2, %arg2 : tensor<3xi8>, tensor<3xi8>) outs(%3 : tensor<3xi8>) {
    ^bb0(%arg3: i8, %arg4: i8, %arg5: i8):
      %5 = arith.addi %arg3, %arg4 : i8
      linalg.yield %5 : i8
    } -> tensor<3xi8>
    return %4 : tensor<3xi8>
  }
}
