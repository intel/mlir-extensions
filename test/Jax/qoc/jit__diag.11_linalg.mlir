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
module @jit__diag.11 {

  func.func private @printMemrefF32(tensor<*xf32>)

  func.func private @callee(%arg0: tensor<1xf32>) -> tensor<2x2xf32> {
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
    %cst = arith.constant dense<-1> : tensor<i32>
    %cst_0 = arith.constant dense<-1> : tensor<2x2xi32>
    %4 = tensor.empty() : tensor<2x2xi32>
    %5 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%3, %cst_0 : tensor<2x2xi32>, tensor<2x2xi32>) outs(%4 : tensor<2x2xi32>) {
    ^bb0(%arg1: i32, %arg2: i32, %arg3: i32):
      %14 = arith.addi %arg1, %arg2 : i32
      linalg.yield %14 : i32
    } -> tensor<2x2xi32>
    %6 = tensor.empty() : tensor<2xi32>
    %7 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel"]} outs(%6 : tensor<2xi32>) {
    ^bb0(%arg1: i32):
      %14 = linalg.index 0 : index
      %15 = arith.index_cast %14 : index to i32
      linalg.yield %15 : i32
    } -> tensor<2xi32>
    %8 = tensor.empty() : tensor<2x2xi32>
    %9 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<2xi32>) outs(%8 : tensor<2x2xi32>) {
    ^bb0(%arg1: i32, %arg2: i32):
      linalg.yield %arg1 : i32
    } -> tensor<2x2xi32>
    %10 = tensor.empty() : tensor<2x2xi1>
    %11 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%5, %9 : tensor<2x2xi32>, tensor<2x2xi32>) outs(%10 : tensor<2x2xi1>) {
    ^bb0(%arg1: i32, %arg2: i32, %arg3: i1):
      %14 = arith.cmpi eq, %arg1, %arg2 : i32
      linalg.yield %14 : i1
    } -> tensor<2x2xi1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %12 = tensor.pad %arg0 low[0] high[1] {
    ^bb0(%arg1: index):
      tensor.yield %cst_2 : f32
    } : tensor<1xf32> to tensor<2xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<2xf32>
    %13 = call @_where.14(%11, %12, %cst_4) : (tensor<2x2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
    return %13 : tensor<2x2xf32>
  }
  func.func private @_where.14(%arg0: tensor<2x2xi1>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>) -> tensor<2x2xf32> {
    %0 = tensor.empty() : tensor<2x2xf32>
    %1 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<2xf32>) outs(%0 : tensor<2x2xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<2x2xf32>
    %2 = tensor.empty() : tensor<2x2xf32>
    %3 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<2xf32>) outs(%2 : tensor<2x2xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<2x2xf32>
    %4 = tensor.empty() : tensor<2x2xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg0, %1, %3 : tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) outs(%4 : tensor<2x2xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %6 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %6 : f32
    } -> tensor<2x2xf32>
    return %5 : tensor<2x2xf32>
  }
  func.func @main() {
    %0 = arith.constant dense<[-1.23]> : tensor<1xf32>
    %3 = func.call @callee(%0) : (tensor<1xf32>) -> tensor<2x2xf32>
    %unranked = tensor.cast %3 : tensor<2x2xf32> to tensor<*xf32>
    func.call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [2, 2] strides = [2, 1] data =
    //      CHECK: [0, 0]
    // CHECK-NEXT: [-1.23, 0]
    return
  }
}
