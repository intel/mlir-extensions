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
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1) -> ()>
module @jit_trace.340 {

  func.func private @printMemrefI32(tensor<*xi32>)

  func.func private @callee(%arg0: tensor<2x2xi32>) -> tensor<i32> {
    %0 = tensor.empty() : tensor<2xi32>
    %1 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel"]} outs(%0 : tensor<2xi32>) {
    ^bb0(%arg1: i32):
      %14 = linalg.index 0 : index
      %15 = arith.index_cast %14 : index to i32
      linalg.yield %15 : i32
    } -> tensor<2xi32>
    %2 = tensor.empty() : tensor<2x2xi32>
    %3 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<2xi32>) outs(%2 : tensor<2x2xi32>) {
    ^bb0(%arg1: i32, %arg2: i32):
      linalg.yield %arg1 : i32
    } -> tensor<2x2xi32>
    %4 = tensor.empty() : tensor<2xi32>
    %5 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel"]} outs(%4 : tensor<2xi32>) {
    ^bb0(%arg1: i32):
      %14 = linalg.index 0 : index
      %15 = arith.index_cast %14 : index to i32
      linalg.yield %15 : i32
    } -> tensor<2xi32>
    %6 = tensor.empty() : tensor<2x2xi32>
    %7 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<2xi32>) outs(%6 : tensor<2x2xi32>) {
    ^bb0(%arg1: i32, %arg2: i32):
      linalg.yield %arg1 : i32
    } -> tensor<2x2xi32>
    %8 = tensor.empty() : tensor<2x2xi1>
    %9 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%3, %7 : tensor<2x2xi32>, tensor<2x2xi32>) outs(%8 : tensor<2x2xi1>) {
    ^bb0(%arg1: i32, %arg2: i32, %arg3: i1):
      %14 = arith.cmpi eq, %arg1, %arg2 : i32
      linalg.yield %14 : i1
    } -> tensor<2x2xi1>
    %cst = arith.constant dense<0> : tensor<i32>
    %cst_0 = arith.constant dense<0> : tensor<2x2xi32>
    %10 = call @_where.10(%9, %arg0, %cst_0) : (tensor<2x2xi1>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    %cst_1 = arith.constant dense<0> : tensor<i32>
    %c0_i32 = arith.constant 0 : i32
    %11 = tensor.empty() : tensor<i32>
    %12 = linalg.fill ins(%c0_i32 : i32) outs(%11 : tensor<i32>) -> tensor<i32>
    %13 = linalg.generic {indexing_maps = [#map2, #map4], iterator_types = ["reduction", "reduction"]} ins(%10 : tensor<2x2xi32>) outs(%12 : tensor<i32>) {
    ^bb0(%arg1: i32, %arg2: i32):
      %14 = arith.addi %arg1, %arg2 : i32
      linalg.yield %14 : i32
    } -> tensor<i32>
    return %13 : tensor<i32>
  }
  func.func private @_where.10(%arg0: tensor<2x2xi1>, %arg1: tensor<2x2xi32>, %arg2: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = tensor.empty() : tensor<2x2xi32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<2x2xi1>, tensor<2x2xi32>, tensor<2x2xi32>) outs(%0 : tensor<2x2xi32>) {
    ^bb0(%arg3: i1, %arg4: i32, %arg5: i32, %arg6: i32):
      %2 = arith.select %arg3, %arg4, %arg5 : i32
      linalg.yield %2 : i32
    } -> tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
  func.func @main() {
    %0 = arith.constant dense<[[-10, 10], [20, -20]]> : tensor<2x2xi32>
    %3 = func.call @callee(%0) : (tensor<2x2xi32>) -> tensor<i32>
    %unranked = tensor.cast %3 : tensor<i32> to tensor<*xi32>
    func.call @printMemrefI32(%unranked) : (tensor<*xi32>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 0 offset = 0 sizes = [] strides = [] data =
    //      CHECK: [-30]
    return
  }
}
