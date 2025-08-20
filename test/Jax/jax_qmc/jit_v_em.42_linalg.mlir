// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils \
// RUN:                                       --entry-point-result=void --filecheck
// RUN: %python_executable %imex_runner --requires=l0-runtime,igpu-fp64 %igpu_fp64 -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime,igpu-fp64 %igpu_fp64 -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
#map = affine_map<() -> ()>
module @jit_v_em.42 {

  func.func private @printMemrefF64(tensor<*xf64>)

  func.func private @callee(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = arith.constant dense<1.000000e+00> : tensor<f64>
    %cst_0 = arith.constant dense<1.000000e-04> : tensor<f64>
    %0 = tensor.empty() : tensor<f64>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%arg0, %cst_0 : tensor<f64>, tensor<f64>) outs(%0 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.maximumf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %cst_1 = arith.constant dense<4.270000e+00> : tensor<f64>
    %2 = tensor.empty() : tensor<f64>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%1, %cst_1 : tensor<f64>, tensor<f64>) outs(%2 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %cst_2 = arith.constant dense<1.100000e+01> : tensor<f64>
    %4 = tensor.empty() : tensor<f64>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%3, %cst_2 : tensor<f64>, tensor<f64>) outs(%4 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %cst_3 = arith.constant dense<1.600000e+01> : tensor<f64>
    %6 = tensor.empty() : tensor<f64>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%5, %cst_3 : tensor<f64>, tensor<f64>) outs(%6 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.divf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %8 = tensor.empty() : tensor<f64>
    %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%7, %cst : tensor<f64>, tensor<f64>) outs(%8 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.addf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %10 = tensor.empty() : tensor<f64>
    %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%3, %3 : tensor<f64>, tensor<f64>) outs(%10 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %cst_4 = arith.constant dense<3.000000e+00> : tensor<f64>
    %12 = tensor.empty() : tensor<f64>
    %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%11, %cst_4 : tensor<f64>, tensor<f64>) outs(%12 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %14 = tensor.empty() : tensor<f64>
    %15 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%13, %cst_3 : tensor<f64>, tensor<f64>) outs(%14 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.divf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %16 = tensor.empty() : tensor<f64>
    %17 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%9, %15 : tensor<f64>, tensor<f64>) outs(%16 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.addf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %18 = tensor.empty() : tensor<f64>
    %19 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%3, %3 : tensor<f64>, tensor<f64>) outs(%18 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %20 = tensor.empty() : tensor<f64>
    %21 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%3, %19 : tensor<f64>, tensor<f64>) outs(%20 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %cst_5 = arith.constant dense<4.800000e+01> : tensor<f64>
    %22 = tensor.empty() : tensor<f64>
    %23 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%21, %cst_5 : tensor<f64>, tensor<f64>) outs(%22 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.divf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %24 = tensor.empty() : tensor<f64>
    %25 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%17, %23 : tensor<f64>, tensor<f64>) outs(%24 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.addf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %26 = tensor.empty() : tensor<f64>
    %27 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%3 : tensor<f64>) outs(%26 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64):
      %38 = arith.negf %arg1 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %28 = tensor.empty() : tensor<f64>
    %29 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%27 : tensor<f64>) outs(%28 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64):
      %38 = math.exp %arg1 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %30 = tensor.empty() : tensor<f64>
    %31 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%25, %29 : tensor<f64>, tensor<f64>) outs(%30 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %32 = tensor.empty() : tensor<f64>
    %33 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%cst, %31 : tensor<f64>, tensor<f64>) outs(%32 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.subf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %cst_6 = arith.constant dense<1.4399651726528193> : tensor<f64>
    %34 = tensor.empty() : tensor<f64>
    %35 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%33, %cst_6 : tensor<f64>, tensor<f64>) outs(%34 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %36 = tensor.empty() : tensor<f64>
    %37 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%35, %1 : tensor<f64>, tensor<f64>) outs(%36 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.divf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    return %37 : tensor<f64>
  }
  func.func @main() {
    %0 = arith.constant dense<1.99> : tensor<f64>
    %3 = func.call @callee(%0) : (tensor<f64>) -> tensor<f64>
    %unranked = tensor.cast %3 : tensor<f64> to tensor<*xf64>
    func.call @printMemrefF64(%unranked) : (tensor<*xf64>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 0 offset = 0 sizes = [] strides = [] data =
    //      CHECK: [0.718705]
    return
  }
}
