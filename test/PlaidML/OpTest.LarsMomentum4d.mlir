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
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> ()>
#map2 = affine_map<() -> ()>
module @lars_momentum4d {
func.func @main() {
    %0 = arith.constant dense<1.5> : tensor<4x7x3x9xf32>
    %1 = arith.constant dense<2.5> : tensor<4x7x3x9xf32>
    %2 = arith.constant dense<3.5> : tensor<4x7x3x9xf32>
    %3 = arith.constant dense<0.5> : tensor<f32>
    %4, %5 = call @test(%0, %1, %2, %3) : (tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>, tensor<f32>) -> (tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>)
    %unranked_1 = tensor.cast %4 : tensor<4x7x3x9xf32> to tensor<*xf32>
    %unranked_2 = tensor.cast %5 : tensor<4x7x3x9xf32> to tensor<*xf32>
    call @printMemrefF32(%unranked_1) : (tensor<*xf32>) -> ()
    call @printMemrefF32(%unranked_2) : (tensor<*xf32>) -> ()
    // CHECK: [1.061{{.*}}, 1.061{{.*}}, 1.061{{.*}}, 1.061{{.*}}, 1.061{{.*}}, 1.061{{.*}}, 1.061{{.*}}, 1.061{{.*}}, 1.061{{.*}}]
    // CHECK: [0.438{{.*}} 0.438{{.*}}, 0.438{{.*}}, 0.438{{.*}}, 0.438{{.*}}, 0.438{{.*}}, 0.438{{.*}}, 0.438{{.*}}, 0.438{{.*}}]
    return
  }
func.func private @printMemrefF32(tensor<*xf32>)
func.func @test(%arg0: tensor<4x7x3x9xf32>, %arg1: tensor<4x7x3x9xf32>, %arg2: tensor<4x7x3x9xf32>, %arg3: tensor<f32>) -> (tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.250000e-01 : f32
    %cst_1 = arith.constant 9.765625E-4 : f32
    %cst_2 = arith.constant 4.8828125E-4 : f32
    %0 = tensor.empty() : tensor<4x7x3x9xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %cst_0 : tensor<4x7x3x9xf32>, f32) outs(%0 : tensor<4x7x3x9xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %36 = arith.mulf %arg4, %arg5 : f32
      linalg.yield %36 : f32
    } -> tensor<4x7x3x9xf32>
    %2 = tensor.empty() : tensor<f32>
    %3 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%arg3, %cst_1 : tensor<f32>, f32) outs(%2 : tensor<f32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %36 = arith.mulf %arg4, %arg5 : f32
      linalg.yield %36 : f32
    } -> tensor<f32>
    %4 = tensor.empty() : tensor<4x7x3x9xf32>
    %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg0 : tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>) outs(%4 : tensor<4x7x3x9xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %36 = arith.mulf %arg4, %arg5 : f32
      linalg.yield %36 : f32
    } -> tensor<4x7x3x9xf32>
    %6 = tensor.empty() : tensor<f32>
    %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<f32>) -> tensor<f32>
    %8 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction", "reduction", "reduction", "reduction"]} ins(%5 : tensor<4x7x3x9xf32>) outs(%7 : tensor<f32>) attrs =  {iterator_ranges = [4, 7, 3, 9]} {
    ^bb0(%arg4: f32, %arg5: f32):
      %36 = arith.addf %arg5, %arg4 : f32
      linalg.yield %36 : f32
    } -> tensor<f32>
    %9 = tensor.empty() : tensor<f32>
    %10 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%8 : tensor<f32>) outs(%9 : tensor<f32>) {
    ^bb0(%arg4: f32, %arg5: f32):
      %36 = math.sqrt %arg4 : f32
      linalg.yield %36 : f32
    } -> tensor<f32>
    %11 = tensor.empty() : tensor<f32>
    %12 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%3, %10 : tensor<f32>, tensor<f32>) outs(%11 : tensor<f32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %36 = arith.mulf %arg4, %arg5 : f32
      linalg.yield %36 : f32
    } -> tensor<f32>
    %13 = tensor.empty() : tensor<4x7x3x9xf32>
    %14 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1, %arg1 : tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>) outs(%13 : tensor<4x7x3x9xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %36 = arith.mulf %arg4, %arg5 : f32
      linalg.yield %36 : f32
    } -> tensor<4x7x3x9xf32>
    %15 = tensor.empty() : tensor<f32>
    %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<f32>) -> tensor<f32>
    %17 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction", "reduction", "reduction", "reduction"]} ins(%14 : tensor<4x7x3x9xf32>) outs(%16 : tensor<f32>) attrs =  {iterator_ranges = [4, 7, 3, 9]} {
    ^bb0(%arg4: f32, %arg5: f32):
      %36 = arith.addf %arg5, %arg4 : f32
      linalg.yield %36 : f32
    } -> tensor<f32>
    %18 = tensor.empty() : tensor<f32>
    %19 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%17 : tensor<f32>) outs(%18 : tensor<f32>) {
    ^bb0(%arg4: f32, %arg5: f32):
      %36 = math.sqrt %arg4 : f32
      linalg.yield %36 : f32
    } -> tensor<f32>
    %20 = tensor.empty() : tensor<f32>
    %21 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%10, %cst_2 : tensor<f32>, f32) outs(%20 : tensor<f32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %36 = arith.mulf %arg4, %arg5 : f32
      linalg.yield %36 : f32
    } -> tensor<f32>
    %22 = tensor.empty() : tensor<f32>
    %23 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%19, %21 : tensor<f32>, tensor<f32>) outs(%22 : tensor<f32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %36 = arith.addf %arg4, %arg5 : f32
      linalg.yield %36 : f32
    } -> tensor<f32>
    %24 = tensor.empty() : tensor<f32>
    %25 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%12, %23 : tensor<f32>, tensor<f32>) outs(%24 : tensor<f32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %36 = arith.divf %arg4, %arg5 : f32
      linalg.yield %36 : f32
    } -> tensor<f32>
    %26 = tensor.empty() : tensor<4x7x3x9xf32>
    %27 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %cst_2 : tensor<4x7x3x9xf32>, f32) outs(%26 : tensor<4x7x3x9xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %36 = arith.mulf %arg4, %arg5 : f32
      linalg.yield %36 : f32
    } -> tensor<4x7x3x9xf32>
    %28 = tensor.empty() : tensor<4x7x3x9xf32>
    %29 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1, %27 : tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>) outs(%28 : tensor<4x7x3x9xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %36 = arith.addf %arg4, %arg5 : f32
      linalg.yield %36 : f32
    } -> tensor<4x7x3x9xf32>
    %30 = tensor.empty() : tensor<4x7x3x9xf32>
    %31 = linalg.generic {indexing_maps = [#map1, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%25, %29 : tensor<f32>, tensor<4x7x3x9xf32>) outs(%30 : tensor<4x7x3x9xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %36 = arith.mulf %arg4, %arg5 : f32
      linalg.yield %36 : f32
    } -> tensor<4x7x3x9xf32>
    %32 = tensor.empty() : tensor<4x7x3x9xf32>
    %33 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %31 : tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>) outs(%32 : tensor<4x7x3x9xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %36 = arith.addf %arg4, %arg5 : f32
      linalg.yield %36 : f32
    } -> tensor<4x7x3x9xf32>
    %34 = tensor.empty() : tensor<4x7x3x9xf32>
    %35 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %33 : tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>) outs(%34 : tensor<4x7x3x9xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %36 = arith.subf %arg4, %arg5 : f32
      linalg.yield %36 : f32
    } -> tensor<4x7x3x9xf32>
    return %35, %33 : tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>
  }
}
