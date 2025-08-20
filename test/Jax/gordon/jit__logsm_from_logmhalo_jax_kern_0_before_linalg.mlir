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
#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0) -> (d0)>
module @jit__logsm_from_logmhalo_jax_kern.0 {

  func.func private @printMemrefF32(tensor<*xf32>)

  func.func private @callee(%arg0: tensor<500xf32>, %arg1: tensor<5xf32>) -> tensor<500xf32> {
    %0 = tensor.extract_slice %arg1[0] [1] [1] : tensor<5xf32> to tensor<1xf32>
    %1 = tensor.collapse_shape %0 [] : tensor<1xf32> into tensor<f32>
    %2 = tensor.extract_slice %arg1[1] [1] [1] : tensor<5xf32> to tensor<1xf32>
    %3 = tensor.collapse_shape %2 [] : tensor<1xf32> into tensor<f32>
    %4 = tensor.empty() : tensor<f32>
    %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%1, %3 : tensor<f32>, tensor<f32>) outs(%4 : tensor<f32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %46 = arith.addf %arg2, %arg3 : f32
      linalg.yield %46 : f32
    } -> tensor<f32>
    %6 = tensor.empty() : tensor<500xf32>
    %7 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%5 : tensor<f32>) outs(%6 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<500xf32>
    %8 = tensor.extract_slice %arg1[3] [1] [1] : tensor<5xf32> to tensor<1xf32>
    %9 = tensor.collapse_shape %8 [] : tensor<1xf32> into tensor<f32>
    %10 = tensor.empty() : tensor<500xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%9 : tensor<f32>) outs(%10 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<500xf32>
    %12 = tensor.extract_slice %arg1[4] [1] [1] : tensor<5xf32> to tensor<1xf32>
    %13 = tensor.collapse_shape %12 [] : tensor<1xf32> into tensor<f32>
    %14 = tensor.empty() : tensor<f32>
    %15 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%13, %9 : tensor<f32>, tensor<f32>) outs(%14 : tensor<f32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %46 = arith.subf %arg2, %arg3 : f32
      linalg.yield %46 : f32
    } -> tensor<f32>
    %16 = tensor.empty() : tensor<500xf32>
    %17 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%15 : tensor<f32>) outs(%16 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<500xf32>
    %18 = tensor.extract_slice %arg1[2] [1] [1] : tensor<5xf32> to tensor<1xf32>
    %19 = tensor.collapse_shape %18 [] : tensor<1xf32> into tensor<f32>
    %20 = tensor.empty() : tensor<f32>
    %21 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = []} ins(%19 : tensor<f32>) outs(%20 : tensor<f32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %46 = arith.negf %arg2 : f32
      linalg.yield %46 : f32
    } -> tensor<f32>
    %22 = tensor.empty() : tensor<500xf32>
    %23 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%21 : tensor<f32>) outs(%22 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<500xf32>
    %24 = tensor.empty() : tensor<500xf32>
    %25 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%1 : tensor<f32>) outs(%24 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<500xf32>
    %26 = tensor.empty() : tensor<500xf32>
    %27 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %25 : tensor<500xf32>, tensor<500xf32>) outs(%26 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %46 = arith.subf %arg2, %arg3 : f32
      linalg.yield %46 : f32
    } -> tensor<500xf32>
    %28 = tensor.empty() : tensor<500xf32>
    %29 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%23, %27 : tensor<500xf32>, tensor<500xf32>) outs(%28 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %46 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %46 : f32
    } -> tensor<500xf32>
    %30 = tensor.empty() : tensor<500xf32>
    %31 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%29 : tensor<500xf32>) outs(%30 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %46 = math.exp %arg2 : f32
      linalg.yield %46 : f32
    } -> tensor<500xf32>
    %cst = arith.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<500xf32>
    %32 = tensor.empty() : tensor<500xf32>
    %33 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%31, %cst_0 : tensor<500xf32>, tensor<500xf32>) outs(%32 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %46 = arith.addf %arg2, %arg3 : f32
      linalg.yield %46 : f32
    } -> tensor<500xf32>
    %34 = tensor.empty() : tensor<500xf32>
    %35 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%17, %33 : tensor<500xf32>, tensor<500xf32>) outs(%34 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %46 = arith.divf %arg2, %arg3 : f32
      linalg.yield %46 : f32
    } -> tensor<500xf32>
    %36 = tensor.empty() : tensor<500xf32>
    %37 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%11, %35 : tensor<500xf32>, tensor<500xf32>) outs(%36 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %46 = arith.addf %arg2, %arg3 : f32
      linalg.yield %46 : f32
    } -> tensor<500xf32>
    %38 = tensor.empty() : tensor<500xf32>
    %39 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%1 : tensor<f32>) outs(%38 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<500xf32>
    %40 = tensor.empty() : tensor<500xf32>
    %41 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %39 : tensor<500xf32>, tensor<500xf32>) outs(%40 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %46 = arith.subf %arg2, %arg3 : f32
      linalg.yield %46 : f32
    } -> tensor<500xf32>
    %42 = tensor.empty() : tensor<500xf32>
    %43 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%37, %41 : tensor<500xf32>, tensor<500xf32>) outs(%42 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %46 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %46 : f32
    } -> tensor<500xf32>
    %44 = tensor.empty() : tensor<500xf32>
    %45 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%7, %43 : tensor<500xf32>, tensor<500xf32>) outs(%44 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %46 = arith.addf %arg2, %arg3 : f32
      linalg.yield %46 : f32
    } -> tensor<500xf32>
    return %45 : tensor<500xf32>
  }
  func.func @main() {
    %0 = arith.constant dense<1.0> : tensor<500xf32>
    %1 = arith.constant dense<[5.0, 4.0, 3.0, 2.0, 1.0]> : tensor<5xf32>
    %2 = func.call @callee(%0, %1) : (tensor<500xf32>, tensor<5xf32>) -> tensor<500xf32>
    %unranked = tensor.cast %2 : tensor<500xf32> to tensor<*xf32>
    func.call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    //          CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    //     CHECK-SAME: rank = 1 offset = 0 sizes = [500] strides = [1] data =
    //          CHECK: [1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-COUNT-48:  1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    //          CHECK:  1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002]
    return
  }
}
