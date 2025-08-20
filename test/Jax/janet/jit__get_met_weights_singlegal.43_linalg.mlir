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
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<() -> ()>
#map3 = affine_map<(d0) -> (0)>
module @jit__get_met_weights_singlegal.43 {

  func.func private @printMemrefF32(tensor<*xf32>)

  func.func private @callee(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<23xf32>) -> tensor<22xf32> {
    %0 = func.call @triweighted_histogram.170(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<23xf32>) -> tensor<22xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %1 = tensor.empty() : tensor<f32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<f32>) -> tensor<f32>
    %3 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction"]} ins(%0 : tensor<22xf32>) outs(%2 : tensor<f32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %12 = arith.addf %arg3, %arg4 : f32
      linalg.yield %12 : f32
    } -> tensor<f32>
    %4 = tensor.empty() : tensor<i1>
    %5 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%3, %cst : tensor<f32>, tensor<f32>) outs(%4 : tensor<i1>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: i1):
      %12 = arith.cmpf oeq, %arg3, %arg4 : f32
      linalg.yield %12 : i1
    } -> tensor<i1>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<f32>
    %6 = func.call @_where.184(%5, %cst_1, %3) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %7 = tensor.empty() : tensor<22xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel"]} ins(%6 : tensor<f32>) outs(%7 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<22xf32>
    %9 = tensor.empty() : tensor<22xf32>
    %10 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%0, %8 : tensor<22xf32>, tensor<22xf32>) outs(%9 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %12 = arith.divf %arg3, %arg4 : f32
      linalg.yield %12 : f32
    } -> tensor<22xf32>
    %11 = func.call @_fill_empty_weights_singlegal.214(%arg0, %arg2, %10) : (tensor<f32>, tensor<23xf32>, tensor<22xf32>) -> tensor<22xf32>
    return %11 : tensor<22xf32>
  }
  func.func private @triweighted_histogram.170(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<23xf32>) -> tensor<22xf32> {
    %0 = tensor.extract_slice %arg2[0] [22] [1] : tensor<23xf32> to tensor<22xf32>
    %1 = tensor.extract_slice %arg2[1] [22] [1] : tensor<23xf32> to tensor<22xf32>
    %2 = func.call @_triweighted_histogram_kernel.164(%arg0, %arg1, %0, %1) : (tensor<f32>, tensor<f32>, tensor<22xf32>, tensor<22xf32>) -> tensor<22xf32>
    return %2 : tensor<22xf32>
  }
  func.func private @_triweighted_histogram_kernel.164(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<22xf32>, %arg3: tensor<22xf32>) -> tensor<22xf32> {
    %0 = func.call @vmap__triweighted_histogram_kernel_.156(%arg0, %arg1, %arg2, %arg3) : (tensor<f32>, tensor<f32>, tensor<22xf32>, tensor<22xf32>) -> tensor<22xf32>
    return %0 : tensor<22xf32>
  }
  func.func private @vmap__triweighted_histogram_kernel_.156(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<22xf32>, %arg3: tensor<22xf32>) -> tensor<22xf32> {
    %0 = func.call @vmap__tw_cuml_kern_.31(%arg0, %arg2, %arg1) : (tensor<f32>, tensor<22xf32>, tensor<f32>) -> tensor<22xf32>
    %1 = func.call @vmap__tw_cuml_kern__2.106(%arg0, %arg3, %arg1) : (tensor<f32>, tensor<22xf32>, tensor<f32>) -> tensor<22xf32>
    %2 = tensor.empty() : tensor<22xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%0, %1 : tensor<22xf32>, tensor<22xf32>) outs(%2 : tensor<22xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %4 = arith.subf %arg4, %arg5 : f32
      linalg.yield %4 : f32
    } -> tensor<22xf32>
    return %3 : tensor<22xf32>
  }
  func.func private @vmap__tw_cuml_kern_.31(%arg0: tensor<f32>, %arg1: tensor<22xf32>, %arg2: tensor<f32>) -> tensor<22xf32> {
    %0 = tensor.empty() : tensor<22xf32>
    %1 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel"]} ins(%arg0 : tensor<f32>) outs(%0 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<22xf32>
    %2 = tensor.empty() : tensor<22xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%1, %arg1 : tensor<22xf32>, tensor<22xf32>) outs(%2 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.subf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %4 = tensor.empty() : tensor<22xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel"]} ins(%arg2 : tensor<f32>) outs(%4 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<22xf32>
    %6 = tensor.empty() : tensor<22xf32>
    %7 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%3, %5 : tensor<22xf32>, tensor<22xf32>) outs(%6 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.divf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %cst = arith.constant dense<3.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<3.000000e+00> : tensor<22xf32>
    %8 = tensor.empty() : tensor<22xi1>
    %9 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%7, %cst_0 : tensor<22xf32>, tensor<22xf32>) outs(%8 : tensor<22xi1>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: i1):
      %44 = arith.cmpf ogt, %arg3, %arg4 : f32
      linalg.yield %44 : i1
    } -> tensor<22xi1>
    %cst_1 = arith.constant dense<1> : tensor<i32>
    %cst_2 = arith.constant dense<-3.000000e+00> : tensor<f32>
    %cst_3 = arith.constant dense<-3.000000e+00> : tensor<22xf32>
    %10 = tensor.empty() : tensor<22xi1>
    %11 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%7, %cst_3 : tensor<22xf32>, tensor<22xf32>) outs(%10 : tensor<22xi1>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: i1):
      %44 = arith.cmpf olt, %arg3, %arg4 : f32
      linalg.yield %44 : i1
    } -> tensor<22xi1>
    %cst_4 = arith.constant dense<0> : tensor<i32>
    %12 = func.call @integer_pow.6(%7) : (tensor<22xf32>) -> tensor<22xf32>
    %cst_5 = arith.constant dense<-5.000000e+00> : tensor<f32>
    %cst_6 = arith.constant dense<-5.000000e+00> : tensor<22xf32>
    %13 = tensor.empty() : tensor<22xf32>
    %14 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%12, %cst_6 : tensor<22xf32>, tensor<22xf32>) outs(%13 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.mulf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %cst_7 = arith.constant dense<6.998400e+04> : tensor<f32>
    %cst_8 = arith.constant dense<6.998400e+04> : tensor<22xf32>
    %15 = tensor.empty() : tensor<22xf32>
    %16 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%14, %cst_8 : tensor<22xf32>, tensor<22xf32>) outs(%15 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.divf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %17 = func.call @integer_pow_0.12(%7) : (tensor<22xf32>) -> tensor<22xf32>
    %cst_9 = arith.constant dense<7.000000e+00> : tensor<f32>
    %cst_10 = arith.constant dense<7.000000e+00> : tensor<22xf32>
    %18 = tensor.empty() : tensor<22xf32>
    %19 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%17, %cst_10 : tensor<22xf32>, tensor<22xf32>) outs(%18 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.mulf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %cst_11 = arith.constant dense<2.592000e+03> : tensor<f32>
    %cst_12 = arith.constant dense<2.592000e+03> : tensor<22xf32>
    %20 = tensor.empty() : tensor<22xf32>
    %21 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%19, %cst_12 : tensor<22xf32>, tensor<22xf32>) outs(%20 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.divf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %22 = tensor.empty() : tensor<22xf32>
    %23 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%16, %21 : tensor<22xf32>, tensor<22xf32>) outs(%22 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.addf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %24 = tensor.empty() : tensor<22xf32>
    %25 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%7, %7 : tensor<22xf32>, tensor<22xf32>) outs(%24 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.mulf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %26 = tensor.empty() : tensor<22xf32>
    %27 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%7, %25 : tensor<22xf32>, tensor<22xf32>) outs(%26 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.mulf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %cst_13 = arith.constant dense<3.500000e+01> : tensor<f32>
    %cst_14 = arith.constant dense<3.500000e+01> : tensor<22xf32>
    %28 = tensor.empty() : tensor<22xf32>
    %29 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%27, %cst_14 : tensor<22xf32>, tensor<22xf32>) outs(%28 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.mulf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %cst_15 = arith.constant dense<8.640000e+02> : tensor<f32>
    %cst_16 = arith.constant dense<8.640000e+02> : tensor<22xf32>
    %30 = tensor.empty() : tensor<22xf32>
    %31 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%29, %cst_16 : tensor<22xf32>, tensor<22xf32>) outs(%30 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.divf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %32 = tensor.empty() : tensor<22xf32>
    %33 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%23, %31 : tensor<22xf32>, tensor<22xf32>) outs(%32 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.subf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %34 = tensor.empty() : tensor<22xf32>
    %35 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%7, %cst_14 : tensor<22xf32>, tensor<22xf32>) outs(%34 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.mulf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %cst_17 = arith.constant dense<9.600000e+01> : tensor<f32>
    %cst_18 = arith.constant dense<9.600000e+01> : tensor<22xf32>
    %36 = tensor.empty() : tensor<22xf32>
    %37 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%35, %cst_18 : tensor<22xf32>, tensor<22xf32>) outs(%36 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.divf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %38 = tensor.empty() : tensor<22xf32>
    %39 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%33, %37 : tensor<22xf32>, tensor<22xf32>) outs(%38 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.addf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %cst_19 = arith.constant dense<5.000000e-01> : tensor<f32>
    %cst_20 = arith.constant dense<5.000000e-01> : tensor<22xf32>
    %40 = tensor.empty() : tensor<22xf32>
    %41 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%39, %cst_20 : tensor<22xf32>, tensor<22xf32>) outs(%40 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.addf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %42 = func.call @vmap__where_.17(%11, %cst_4, %41) : (tensor<22xi1>, tensor<i32>, tensor<22xf32>) -> tensor<22xf32>
    %43 = func.call @vmap__where__1.24(%9, %cst_1, %42) : (tensor<22xi1>, tensor<i32>, tensor<22xf32>) -> tensor<22xf32>
    return %43 : tensor<22xf32>
  }
  func.func private @integer_pow.6(%arg0: tensor<22xf32>) -> tensor<22xf32> {
    %0 = tensor.empty() : tensor<22xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %arg0 : tensor<22xf32>, tensor<22xf32>) outs(%0 : tensor<22xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %8 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %8 : f32
    } -> tensor<22xf32>
    %2 = tensor.empty() : tensor<22xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %1 : tensor<22xf32>, tensor<22xf32>) outs(%2 : tensor<22xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %8 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %8 : f32
    } -> tensor<22xf32>
    %4 = tensor.empty() : tensor<22xf32>
    %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%1, %1 : tensor<22xf32>, tensor<22xf32>) outs(%4 : tensor<22xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %8 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %8 : f32
    } -> tensor<22xf32>
    %6 = tensor.empty() : tensor<22xf32>
    %7 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%3, %5 : tensor<22xf32>, tensor<22xf32>) outs(%6 : tensor<22xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %8 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %8 : f32
    } -> tensor<22xf32>
    return %7 : tensor<22xf32>
  }
  func.func private @integer_pow_0.12(%arg0: tensor<22xf32>) -> tensor<22xf32> {
    %0 = tensor.empty() : tensor<22xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %arg0 : tensor<22xf32>, tensor<22xf32>) outs(%0 : tensor<22xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %6 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %6 : f32
    } -> tensor<22xf32>
    %2 = tensor.empty() : tensor<22xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%1, %1 : tensor<22xf32>, tensor<22xf32>) outs(%2 : tensor<22xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %6 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %6 : f32
    } -> tensor<22xf32>
    %4 = tensor.empty() : tensor<22xf32>
    %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %3 : tensor<22xf32>, tensor<22xf32>) outs(%4 : tensor<22xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %6 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %6 : f32
    } -> tensor<22xf32>
    return %5 : tensor<22xf32>
  }
  func.func private @vmap__where_.17(%arg0: tensor<22xi1>, %arg1: tensor<i32>, %arg2: tensor<22xf32>) -> tensor<22xf32> {
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%arg1 : tensor<i32>) outs(%0 : tensor<f32>) {
    ^bb0(%arg3: i32, %arg4: f32):
      %6 = arith.sitofp %arg3 : i32 to f32
      linalg.yield %6 : f32
    } -> tensor<f32>
    %2 = tensor.empty() : tensor<22xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel"]} ins(%1 : tensor<f32>) outs(%2 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<22xf32>
    %4 = tensor.empty() : tensor<22xf32>
    %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %3, %arg2 : tensor<22xi1>, tensor<22xf32>, tensor<22xf32>) outs(%4 : tensor<22xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %6 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %6 : f32
    } -> tensor<22xf32>
    return %5 : tensor<22xf32>
  }
  func.func private @vmap__where__1.24(%arg0: tensor<22xi1>, %arg1: tensor<i32>, %arg2: tensor<22xf32>) -> tensor<22xf32> {
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%arg1 : tensor<i32>) outs(%0 : tensor<f32>) {
    ^bb0(%arg3: i32, %arg4: f32):
      %6 = arith.sitofp %arg3 : i32 to f32
      linalg.yield %6 : f32
    } -> tensor<f32>
    %2 = tensor.empty() : tensor<22xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel"]} ins(%1 : tensor<f32>) outs(%2 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<22xf32>
    %4 = tensor.empty() : tensor<22xf32>
    %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %3, %arg2 : tensor<22xi1>, tensor<22xf32>, tensor<22xf32>) outs(%4 : tensor<22xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %6 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %6 : f32
    } -> tensor<22xf32>
    return %5 : tensor<22xf32>
  }
  func.func private @vmap__tw_cuml_kern__2.106(%arg0: tensor<f32>, %arg1: tensor<22xf32>, %arg2: tensor<f32>) -> tensor<22xf32> {
    %0 = tensor.empty() : tensor<22xf32>
    %1 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel"]} ins(%arg0 : tensor<f32>) outs(%0 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<22xf32>
    %2 = tensor.empty() : tensor<22xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%1, %arg1 : tensor<22xf32>, tensor<22xf32>) outs(%2 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.subf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %4 = tensor.empty() : tensor<22xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel"]} ins(%arg2 : tensor<f32>) outs(%4 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<22xf32>
    %6 = tensor.empty() : tensor<22xf32>
    %7 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%3, %5 : tensor<22xf32>, tensor<22xf32>) outs(%6 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.divf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %cst = arith.constant dense<3.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<3.000000e+00> : tensor<22xf32>
    %8 = tensor.empty() : tensor<22xi1>
    %9 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%7, %cst_0 : tensor<22xf32>, tensor<22xf32>) outs(%8 : tensor<22xi1>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: i1):
      %44 = arith.cmpf ogt, %arg3, %arg4 : f32
      linalg.yield %44 : i1
    } -> tensor<22xi1>
    %cst_1 = arith.constant dense<1> : tensor<i32>
    %cst_2 = arith.constant dense<-3.000000e+00> : tensor<f32>
    %cst_3 = arith.constant dense<-3.000000e+00> : tensor<22xf32>
    %10 = tensor.empty() : tensor<22xi1>
    %11 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%7, %cst_3 : tensor<22xf32>, tensor<22xf32>) outs(%10 : tensor<22xi1>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: i1):
      %44 = arith.cmpf olt, %arg3, %arg4 : f32
      linalg.yield %44 : i1
    } -> tensor<22xi1>
    %cst_4 = arith.constant dense<0> : tensor<i32>
    %12 = func.call @integer_pow.81(%7) : (tensor<22xf32>) -> tensor<22xf32>
    %cst_5 = arith.constant dense<-5.000000e+00> : tensor<f32>
    %cst_6 = arith.constant dense<-5.000000e+00> : tensor<22xf32>
    %13 = tensor.empty() : tensor<22xf32>
    %14 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%12, %cst_6 : tensor<22xf32>, tensor<22xf32>) outs(%13 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.mulf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %cst_7 = arith.constant dense<6.998400e+04> : tensor<f32>
    %cst_8 = arith.constant dense<6.998400e+04> : tensor<22xf32>
    %15 = tensor.empty() : tensor<22xf32>
    %16 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%14, %cst_8 : tensor<22xf32>, tensor<22xf32>) outs(%15 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.divf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %17 = func.call @integer_pow_0.87(%7) : (tensor<22xf32>) -> tensor<22xf32>
    %cst_9 = arith.constant dense<7.000000e+00> : tensor<f32>
    %cst_10 = arith.constant dense<7.000000e+00> : tensor<22xf32>
    %18 = tensor.empty() : tensor<22xf32>
    %19 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%17, %cst_10 : tensor<22xf32>, tensor<22xf32>) outs(%18 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.mulf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %cst_11 = arith.constant dense<2.592000e+03> : tensor<f32>
    %cst_12 = arith.constant dense<2.592000e+03> : tensor<22xf32>
    %20 = tensor.empty() : tensor<22xf32>
    %21 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%19, %cst_12 : tensor<22xf32>, tensor<22xf32>) outs(%20 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.divf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %22 = tensor.empty() : tensor<22xf32>
    %23 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%16, %21 : tensor<22xf32>, tensor<22xf32>) outs(%22 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.addf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %24 = tensor.empty() : tensor<22xf32>
    %25 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%7, %7 : tensor<22xf32>, tensor<22xf32>) outs(%24 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.mulf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %26 = tensor.empty() : tensor<22xf32>
    %27 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%7, %25 : tensor<22xf32>, tensor<22xf32>) outs(%26 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.mulf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %cst_13 = arith.constant dense<3.500000e+01> : tensor<f32>
    %cst_14 = arith.constant dense<3.500000e+01> : tensor<22xf32>
    %28 = tensor.empty() : tensor<22xf32>
    %29 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%27, %cst_14 : tensor<22xf32>, tensor<22xf32>) outs(%28 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.mulf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %cst_15 = arith.constant dense<8.640000e+02> : tensor<f32>
    %cst_16 = arith.constant dense<8.640000e+02> : tensor<22xf32>
    %30 = tensor.empty() : tensor<22xf32>
    %31 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%29, %cst_16 : tensor<22xf32>, tensor<22xf32>) outs(%30 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.divf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %32 = tensor.empty() : tensor<22xf32>
    %33 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%23, %31 : tensor<22xf32>, tensor<22xf32>) outs(%32 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.subf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %34 = tensor.empty() : tensor<22xf32>
    %35 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%7, %cst_14 : tensor<22xf32>, tensor<22xf32>) outs(%34 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.mulf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %cst_17 = arith.constant dense<9.600000e+01> : tensor<f32>
    %cst_18 = arith.constant dense<9.600000e+01> : tensor<22xf32>
    %36 = tensor.empty() : tensor<22xf32>
    %37 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%35, %cst_18 : tensor<22xf32>, tensor<22xf32>) outs(%36 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.divf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %38 = tensor.empty() : tensor<22xf32>
    %39 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%33, %37 : tensor<22xf32>, tensor<22xf32>) outs(%38 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.addf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %cst_19 = arith.constant dense<5.000000e-01> : tensor<f32>
    %cst_20 = arith.constant dense<5.000000e-01> : tensor<22xf32>
    %40 = tensor.empty() : tensor<22xf32>
    %41 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%39, %cst_20 : tensor<22xf32>, tensor<22xf32>) outs(%40 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %44 = arith.addf %arg3, %arg4 : f32
      linalg.yield %44 : f32
    } -> tensor<22xf32>
    %42 = func.call @vmap__where__3.92(%11, %cst_4, %41) : (tensor<22xi1>, tensor<i32>, tensor<22xf32>) -> tensor<22xf32>
    %43 = func.call @vmap__where__4.99(%9, %cst_1, %42) : (tensor<22xi1>, tensor<i32>, tensor<22xf32>) -> tensor<22xf32>
    return %43 : tensor<22xf32>
  }
  func.func private @integer_pow.81(%arg0: tensor<22xf32>) -> tensor<22xf32> {
    %0 = tensor.empty() : tensor<22xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %arg0 : tensor<22xf32>, tensor<22xf32>) outs(%0 : tensor<22xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %8 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %8 : f32
    } -> tensor<22xf32>
    %2 = tensor.empty() : tensor<22xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %1 : tensor<22xf32>, tensor<22xf32>) outs(%2 : tensor<22xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %8 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %8 : f32
    } -> tensor<22xf32>
    %4 = tensor.empty() : tensor<22xf32>
    %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%1, %1 : tensor<22xf32>, tensor<22xf32>) outs(%4 : tensor<22xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %8 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %8 : f32
    } -> tensor<22xf32>
    %6 = tensor.empty() : tensor<22xf32>
    %7 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%3, %5 : tensor<22xf32>, tensor<22xf32>) outs(%6 : tensor<22xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %8 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %8 : f32
    } -> tensor<22xf32>
    return %7 : tensor<22xf32>
  }
  func.func private @integer_pow_0.87(%arg0: tensor<22xf32>) -> tensor<22xf32> {
    %0 = tensor.empty() : tensor<22xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %arg0 : tensor<22xf32>, tensor<22xf32>) outs(%0 : tensor<22xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %6 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %6 : f32
    } -> tensor<22xf32>
    %2 = tensor.empty() : tensor<22xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%1, %1 : tensor<22xf32>, tensor<22xf32>) outs(%2 : tensor<22xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %6 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %6 : f32
    } -> tensor<22xf32>
    %4 = tensor.empty() : tensor<22xf32>
    %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %3 : tensor<22xf32>, tensor<22xf32>) outs(%4 : tensor<22xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %6 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %6 : f32
    } -> tensor<22xf32>
    return %5 : tensor<22xf32>
  }
  func.func private @vmap__where__3.92(%arg0: tensor<22xi1>, %arg1: tensor<i32>, %arg2: tensor<22xf32>) -> tensor<22xf32> {
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%arg1 : tensor<i32>) outs(%0 : tensor<f32>) {
    ^bb0(%arg3: i32, %arg4: f32):
      %6 = arith.sitofp %arg3 : i32 to f32
      linalg.yield %6 : f32
    } -> tensor<f32>
    %2 = tensor.empty() : tensor<22xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel"]} ins(%1 : tensor<f32>) outs(%2 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<22xf32>
    %4 = tensor.empty() : tensor<22xf32>
    %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %3, %arg2 : tensor<22xi1>, tensor<22xf32>, tensor<22xf32>) outs(%4 : tensor<22xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %6 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %6 : f32
    } -> tensor<22xf32>
    return %5 : tensor<22xf32>
  }
  func.func private @vmap__where__4.99(%arg0: tensor<22xi1>, %arg1: tensor<i32>, %arg2: tensor<22xf32>) -> tensor<22xf32> {
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%arg1 : tensor<i32>) outs(%0 : tensor<f32>) {
    ^bb0(%arg3: i32, %arg4: f32):
      %6 = arith.sitofp %arg3 : i32 to f32
      linalg.yield %6 : f32
    } -> tensor<f32>
    %2 = tensor.empty() : tensor<22xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel"]} ins(%1 : tensor<f32>) outs(%2 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<22xf32>
    %4 = tensor.empty() : tensor<22xf32>
    %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %3, %arg2 : tensor<22xi1>, tensor<22xf32>, tensor<22xf32>) outs(%4 : tensor<22xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %6 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %6 : f32
    } -> tensor<22xf32>
    return %5 : tensor<22xf32>
  }
  func.func private @_where.184(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2], iterator_types = []} ins(%arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>, tensor<f32>) outs(%0 : tensor<f32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %2 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %2 : f32
    } -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @_fill_empty_weights_singlegal.214(%arg0: tensor<f32>, %arg1: tensor<23xf32>, %arg2: tensor<22xf32>) -> tensor<22xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<22xf32>
    %0 = tensor.empty() : tensor<22xi1>
    %1 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg2, %cst_0 : tensor<22xf32>, tensor<22xf32>) outs(%0 : tensor<22xi1>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: i1):
      %21 = arith.cmpf oeq, %arg3, %arg4 : f32
      linalg.yield %21 : i1
    } -> tensor<22xi1>
    %cst_1 = arith.constant dense<true> : tensor<i1>
    %true = arith.constant true
    %2 = tensor.empty() : tensor<i1>
    %3 = linalg.fill ins(%true : i1) outs(%2 : tensor<i1>) -> tensor<i1>
    %4 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction"]} ins(%1 : tensor<22xi1>) outs(%3 : tensor<i1>) {
    ^bb0(%arg3: i1, %arg4: i1):
      %21 = arith.andi %arg3, %arg4 : i1
      linalg.yield %21 : i1
    } -> tensor<i1>
    %5 = tensor.extract_slice %arg1[22] [1] [1] : tensor<23xf32> to tensor<1xf32>
    %6 = tensor.collapse_shape %5 [] : tensor<1xf32> into tensor<f32>
    %7 = tensor.empty() : tensor<i1>
    %8 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%arg0, %6 : tensor<f32>, tensor<f32>) outs(%7 : tensor<i1>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: i1):
      %21 = arith.cmpf ogt, %arg3, %arg4 : f32
      linalg.yield %21 : i1
    } -> tensor<i1>
    %9 = tensor.empty() : tensor<i1>
    %10 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%4, %8 : tensor<i1>, tensor<i1>) outs(%9 : tensor<i1>) {
    ^bb0(%arg3: i1, %arg4: i1, %arg5: i1):
      %21 = arith.andi %arg3, %arg4 : i1
      linalg.yield %21 : i1
    } -> tensor<i1>
    %cst_2 = arith.constant dense<21> : tensor<1xi32>
    %cst_3 = arith.constant dense<1.000000e+00> : tensor<f32>
    %11 = linalg.generic {indexing_maps = [#map0, #map3, #map1, #map0], iterator_types = ["parallel"]} ins(%cst_0, %cst_2, %cst_3 : tensor<22xf32>, tensor<1xi32>, tensor<f32>) outs(%cst_0 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: i32, %arg5: f32, %arg6: f32):
      %21 = linalg.index 0 : index
      %22 = arith.index_cast %arg4 : i32 to index
      %23 = arith.cmpi eq, %21, %22 : index
      %24 = arith.select %23, %arg5, %arg6 : f32
      linalg.yield %24 : f32
    } -> tensor<22xf32>
    %12 = tensor.extract_slice %arg1[0] [1] [1] : tensor<23xf32> to tensor<1xf32>
    %13 = tensor.collapse_shape %12 [] : tensor<1xf32> into tensor<f32>
    %14 = tensor.empty() : tensor<i1>
    %15 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%arg0, %13 : tensor<f32>, tensor<f32>) outs(%14 : tensor<i1>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: i1):
      %21 = arith.cmpf olt, %arg3, %arg4 : f32
      linalg.yield %21 : i1
    } -> tensor<i1>
    %16 = tensor.empty() : tensor<i1>
    %17 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%4, %15 : tensor<i1>, tensor<i1>) outs(%16 : tensor<i1>) {
    ^bb0(%arg3: i1, %arg4: i1, %arg5: i1):
      %21 = arith.andi %arg3, %arg4 : i1
      linalg.yield %21 : i1
    } -> tensor<i1>
    %cst_4 = arith.constant dense<0> : tensor<1xi32>
    %18 = linalg.generic {indexing_maps = [#map0, #map3, #map1, #map0], iterator_types = ["parallel"]} ins(%cst_0, %cst_4, %cst_3 : tensor<22xf32>, tensor<1xi32>, tensor<f32>) outs(%cst_0 : tensor<22xf32>) {
    ^bb0(%arg3: f32, %arg4: i32, %arg5: f32, %arg6: f32):
      %21 = linalg.index 0 : index
      %22 = arith.index_cast %arg4 : i32 to index
      %23 = arith.cmpi eq, %21, %22 : index
      %24 = arith.select %23, %arg5, %arg6 : f32
      linalg.yield %24 : f32
    } -> tensor<22xf32>
    %19 = func.call @_where_5.202(%17, %18, %arg2) : (tensor<i1>, tensor<22xf32>, tensor<22xf32>) -> tensor<22xf32>
    %20 = func.call @_where_6.208(%10, %11, %19) : (tensor<i1>, tensor<22xf32>, tensor<22xf32>) -> tensor<22xf32>
    return %20 : tensor<22xf32>
  }
  func.func private @_where_5.202(%arg0: tensor<i1>, %arg1: tensor<22xf32>, %arg2: tensor<22xf32>) -> tensor<22xf32> {
    %0 = tensor.empty() : tensor<22xi1>
    %1 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel"]} ins(%arg0 : tensor<i1>) outs(%0 : tensor<22xi1>) {
    ^bb0(%arg3: i1, %arg4: i1):
      linalg.yield %arg3 : i1
    } -> tensor<22xi1>
    %2 = tensor.empty() : tensor<22xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%1, %arg1, %arg2 : tensor<22xi1>, tensor<22xf32>, tensor<22xf32>) outs(%2 : tensor<22xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %4 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %4 : f32
    } -> tensor<22xf32>
    return %3 : tensor<22xf32>
  }
  func.func private @_where_6.208(%arg0: tensor<i1>, %arg1: tensor<22xf32>, %arg2: tensor<22xf32>) -> tensor<22xf32> {
    %0 = tensor.empty() : tensor<22xi1>
    %1 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel"]} ins(%arg0 : tensor<i1>) outs(%0 : tensor<22xi1>) {
    ^bb0(%arg3: i1, %arg4: i1):
      linalg.yield %arg3 : i1
    } -> tensor<22xi1>
    %2 = tensor.empty() : tensor<22xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%1, %arg1, %arg2 : tensor<22xi1>, tensor<22xf32>, tensor<22xf32>) outs(%2 : tensor<22xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %4 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %4 : f32
    } -> tensor<22xf32>
    return %3 : tensor<22xf32>
  }
  func.func @main() {
    %0 = arith.constant dense<0.9> : tensor<f32>
    %1 = arith.constant dense<1.1> : tensor<f32>
    %2 = arith.constant dense<0.8> : tensor<23xf32>
    %3 = func.call @callee(%0, %1, %2) : (tensor<f32>, tensor<f32>, tensor<23xf32>) -> tensor<22xf32>
    %unranked = tensor.cast %3 : tensor<22xf32> to tensor<*xf32>
    func.call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [22] strides = [1] data =
    //      CHECK: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //      CHECK:  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //      CHECK:  0, 1]
    return
  }
}
