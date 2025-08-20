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
#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> ()>
#map6 = affine_map<(d0, d1) -> (d0, 0)>
module @mnist_mlp {
  func.func @main() {
    %0 = arith.constant dense<1.0> : tensor<1x784xf32>
    %1 = arith.constant dense<2.0> : tensor<784x512xf32>
    %2 = arith.constant dense<0.5> : tensor<512xf32>
    %3 = arith.constant dense<0.8> : tensor<512x512xf32>
    %4 = arith.constant dense<0.5> : tensor<512xf32>
    %5 = arith.constant dense<1.5> : tensor<512x10xf32>
    %6 = arith.constant dense<0.3> : tensor<10xf32>
    %7 = call @test(%0,%1,%2,%3,%4,%5,%6) : (tensor<1x784xf32>, tensor<784x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>) -> tensor<1x10xf32>
    %unranked = tensor.cast %7 : tensor<1x10xf32>to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @test(%arg0: tensor<1x784xf32>, %arg1: tensor<784x512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x512xf32>, %arg4: tensor<512xf32>, %arg5: tensor<512x10xf32>, %arg6: tensor<10xf32>) -> tensor<1x10xf32> {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x512xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xf32>) outs(%0 : tensor<1x512xf32>) {
    ^bb0(%arg7: f32, %arg8: f32):
      linalg.yield %arg7 : f32
    } -> tensor<1x512xf32>
    %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x784xf32>, tensor<784x512xf32>) outs(%1 : tensor<1x512xf32>) attrs =  {iterator_ranges = [1, 512, 784]} {
    ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):
      %29 = arith.mulf %arg7, %arg8 : f32
      %30 = arith.addf %arg9, %29 : f32
      linalg.yield %30 : f32
    } -> tensor<1x512xf32>
    %3 = tensor.empty() : tensor<1x512xi1>
    %4 = linalg.generic {indexing_maps = [#map1, #map5, #map1], iterator_types = ["parallel", "parallel"]} ins(%2, %cst_0 : tensor<1x512xf32>, f32) outs(%3 : tensor<1x512xi1>) {
    ^bb0(%arg7: f32, %arg8: f32, %arg9: i1):
      %29 = arith.cmpf olt, %arg7, %arg8 : f32
      linalg.yield %29 : i1
    } -> tensor<1x512xi1>
    %5 = tensor.empty() : tensor<1x512xf32>
    %6 = linalg.generic {indexing_maps = [#map1, #map5, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%4, %cst_0, %2 : tensor<1x512xi1>, f32, tensor<1x512xf32>) outs(%5 : tensor<1x512xf32>) {
    ^bb0(%arg7: i1, %arg8: f32, %arg9: f32, %arg10: f32):
      %29 = arith.select %arg7, %arg8, %arg9 : f32
      linalg.yield %29 : f32
    } -> tensor<1x512xf32>
    %7 = tensor.empty() : tensor<1x512xf32>
    %8 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg4 : tensor<512xf32>) outs(%7 : tensor<1x512xf32>) {
    ^bb0(%arg7: f32, %arg8: f32):
      linalg.yield %arg7 : f32
    } -> tensor<1x512xf32>
    %9 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%6, %arg3 : tensor<1x512xf32>, tensor<512x512xf32>) outs(%8 : tensor<1x512xf32>) attrs =  {iterator_ranges = [1, 512, 512]} {
    ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):
      %29 = arith.mulf %arg7, %arg8 : f32
      %30 = arith.addf %arg9, %29 : f32
      linalg.yield %30 : f32
    } -> tensor<1x512xf32>
    %10 = tensor.empty() : tensor<1x512xi1>
    %11 = linalg.generic {indexing_maps = [#map1, #map5, #map1], iterator_types = ["parallel", "parallel"]} ins(%9, %cst_0 : tensor<1x512xf32>, f32) outs(%10 : tensor<1x512xi1>) {
    ^bb0(%arg7: f32, %arg8: f32, %arg9: i1):
      %29 = arith.cmpf olt, %arg7, %arg8 : f32
      linalg.yield %29 : i1
    } -> tensor<1x512xi1>
    %12 = tensor.empty() : tensor<1x512xf32>
    %13 = linalg.generic {indexing_maps = [#map1, #map5, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%11, %cst_0, %9 : tensor<1x512xi1>, f32, tensor<1x512xf32>) outs(%12 : tensor<1x512xf32>) {
    ^bb0(%arg7: i1, %arg8: f32, %arg9: f32, %arg10: f32):
      %29 = arith.select %arg7, %arg8, %arg9 : f32
      linalg.yield %29 : f32
    } -> tensor<1x512xf32>
    %14 = tensor.empty() : tensor<1x10xf32>
    %15 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg6 : tensor<10xf32>) outs(%14 : tensor<1x10xf32>) {
    ^bb0(%arg7: f32, %arg8: f32):
      linalg.yield %arg7 : f32
    } -> tensor<1x10xf32>
    %16 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13, %arg5 : tensor<1x512xf32>, tensor<512x10xf32>) outs(%15 : tensor<1x10xf32>) attrs =  {iterator_ranges = [1, 10, 512]} {
    ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):
      %29 = arith.mulf %arg7, %arg8 : f32
      %30 = arith.addf %arg9, %29 : f32
      linalg.yield %30 : f32
    } -> tensor<1x10xf32>
    %17 = tensor.empty() : tensor<1x1xf32>
    %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %19 = linalg.generic {indexing_maps = [#map1, #map6], iterator_types = ["parallel", "reduction"]} ins(%16 : tensor<1x10xf32>) outs(%18 : tensor<1x1xf32>) attrs =  {iterator_ranges = [1, 10]} {
    ^bb0(%arg7: f32, %arg8: f32):
      %29 = arith.cmpf ogt, %arg8, %arg7 : f32
      %30 = arith.select %29, %arg8, %arg7 : f32
      linalg.yield %30 : f32
    } -> tensor<1x1xf32>
    %20 = tensor.empty() : tensor<1x10xf32>
    %21 = linalg.generic {indexing_maps = [#map1, #map6, #map1], iterator_types = ["parallel", "parallel"]} ins(%16, %19 : tensor<1x10xf32>, tensor<1x1xf32>) outs(%20 : tensor<1x10xf32>) {
    ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):
      %29 = arith.subf %arg7, %arg8 : f32
      linalg.yield %29 : f32
    } -> tensor<1x10xf32>
    %22 = tensor.empty() : tensor<1x10xf32>
    %23 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%21 : tensor<1x10xf32>) outs(%22 : tensor<1x10xf32>) {
    ^bb0(%arg7: f32, %arg8: f32):
      %29 = math.exp %arg7 : f32
      linalg.yield %29 : f32
    } -> tensor<1x10xf32>
    %24 = tensor.empty() : tensor<1x1xf32>
    %25 = linalg.fill ins(%cst_0 : f32) outs(%24 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %26 = linalg.generic {indexing_maps = [#map1, #map6], iterator_types = ["parallel", "reduction"]} ins(%23 : tensor<1x10xf32>) outs(%25 : tensor<1x1xf32>) attrs =  {iterator_ranges = [1, 10]} {
    ^bb0(%arg7: f32, %arg8: f32):
      %29 = arith.addf %arg8, %arg7 : f32
      linalg.yield %29 : f32
    } -> tensor<1x1xf32>
    %27 = tensor.empty() : tensor<1x10xf32>
    %28 = linalg.generic {indexing_maps = [#map1, #map6, #map1], iterator_types = ["parallel", "parallel"]} ins(%23, %26 : tensor<1x10xf32>, tensor<1x1xf32>) outs(%27 : tensor<1x10xf32>) {
    ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):
      %29 = arith.divf %arg7, %arg8 : f32
      linalg.yield %29 : f32
    } -> tensor<1x10xf32>
    return %28 : tensor<1x10xf32>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [1, 10] strides = {{.*}} data =
// CHECK: 0.1
// CHECK: 0.1
// CHECK: 0.1
// CHECK: 0.1
// CHECK: 0.1
// CHECK: 0.1
// CHECK: 0.1
// CHECK: 0.1
// CHECK: 0.1
// CHECK: 0.1
