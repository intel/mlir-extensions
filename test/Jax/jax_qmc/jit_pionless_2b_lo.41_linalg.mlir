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
#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> (0)>
#map3 = affine_map<(d0) -> ()>
module @jit_pionless_2b_lo.41 {

  func.func private @printMemrefF64(tensor<*xf64>)

  func.func private @callee(%arg0: tensor<f64>) -> tensor<6xf64> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<6xf64>
    %cst_1 = arith.constant dense<0> : tensor<1xi32>
    %cst_2 = arith.constant dense<1.83039397> : tensor<f64>
    %0 = tensor.empty() : tensor<f64>
    %1 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%arg0, %cst_2 : tensor<f64>, tensor<f64>) outs(%0 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.divf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %2 = tensor.empty() : tensor<f64>
    %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%1, %1 : tensor<f64>, tensor<f64>) outs(%2 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %4 = tensor.empty() : tensor<f64>
    %5 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = []} ins(%3 : tensor<f64>) outs(%4 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64):
      %56 = arith.negf %arg1 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %6 = tensor.empty() : tensor<f64>
    %7 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = []} ins(%5 : tensor<f64>) outs(%6 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64):
      %56 = math.exp %arg1 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %cst_3 = arith.constant dense<0.029284746016929555> : tensor<f64>
    %8 = tensor.empty() : tensor<f64>
    %9 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%7, %cst_3 : tensor<f64>, tensor<f64>) outs(%8 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %cst_4 = arith.constant dense<-5.2751867099999998> : tensor<f64>
    %10 = tensor.empty() : tensor<f64>
    %11 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%9, %cst_4 : tensor<f64>, tensor<f64>) outs(%10 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %cst_5 = arith.constant dense<1.5459298400000001> : tensor<f64>
    %12 = tensor.empty() : tensor<f64>
    %13 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%arg0, %cst_5 : tensor<f64>, tensor<f64>) outs(%12 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.divf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %14 = tensor.empty() : tensor<f64>
    %15 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%13, %13 : tensor<f64>, tensor<f64>) outs(%14 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %16 = tensor.empty() : tensor<f64>
    %17 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = []} ins(%15 : tensor<f64>) outs(%16 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64):
      %56 = arith.negf %arg1 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %18 = tensor.empty() : tensor<f64>
    %19 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = []} ins(%17 : tensor<f64>) outs(%18 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64):
      %56 = math.exp %arg1 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %cst_6 = arith.constant dense<0.048607787159564257> : tensor<f64>
    %20 = tensor.empty() : tensor<f64>
    %21 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%19, %cst_6 : tensor<f64>, tensor<f64>) outs(%20 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %cst_7 = arith.constant dense<-7.0404007999999996> : tensor<f64>
    %22 = tensor.empty() : tensor<f64>
    %23 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%21, %cst_7 : tensor<f64>, tensor<f64>) outs(%22 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %24 = tensor.empty() : tensor<f64>
    %25 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%11, %23 : tensor<f64>, tensor<f64>) outs(%24 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.addf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %cst_8 = arith.constant dense<3.000000e+00> : tensor<f64>
    %26 = tensor.empty() : tensor<f64>
    %27 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%25, %cst_8 : tensor<f64>, tensor<f64>) outs(%26 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %28 = linalg.generic {indexing_maps = [#map1, #map2, #map3, #map1], iterator_types = ["parallel"]} ins(%cst_0, %cst_1, %27 : tensor<6xf64>, tensor<1xi32>, tensor<f64>) outs(%cst_0 : tensor<6xf64>) {
    ^bb0(%arg1: f64, %arg2: i32, %arg3: f64, %arg4: f64):
      %56 = linalg.index 0 : index
      %57 = arith.index_cast %arg2 : i32 to index
      %58 = arith.cmpi eq, %56, %57 : index
      %59 = arith.select %58, %arg3, %arg4 : f64
      linalg.yield %59 : f64
    } -> tensor<6xf64>
    %cst_9 = arith.constant dense<1> : tensor<1xi32>
    %29 = tensor.empty() : tensor<f64>
    %30 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%9, %cst_4 : tensor<f64>, tensor<f64>) outs(%29 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %cst_10 = arith.constant dense<-21.121202399999998> : tensor<f64>
    %31 = tensor.empty() : tensor<f64>
    %32 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%21, %cst_10 : tensor<f64>, tensor<f64>) outs(%31 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %33 = tensor.empty() : tensor<f64>
    %34 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%30, %32 : tensor<f64>, tensor<f64>) outs(%33 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.subf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %35 = linalg.generic {indexing_maps = [#map1, #map2, #map3, #map1], iterator_types = ["parallel"]} ins(%28, %cst_9, %34 : tensor<6xf64>, tensor<1xi32>, tensor<f64>) outs(%28 : tensor<6xf64>) {
    ^bb0(%arg1: f64, %arg2: i32, %arg3: f64, %arg4: f64):
      %56 = linalg.index 0 : index
      %57 = arith.index_cast %arg2 : i32 to index
      %58 = arith.cmpi eq, %56, %57 : index
      %59 = arith.select %58, %arg3, %arg4 : f64
      linalg.yield %59 : f64
    } -> tensor<6xf64>
    %cst_11 = arith.constant dense<2> : tensor<1xi32>
    %cst_12 = arith.constant dense<15.82556013> : tensor<f64>
    %36 = tensor.empty() : tensor<f64>
    %37 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%9, %cst_12 : tensor<f64>, tensor<f64>) outs(%36 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %38 = tensor.empty() : tensor<f64>
    %39 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%21, %cst_7 : tensor<f64>, tensor<f64>) outs(%38 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %40 = tensor.empty() : tensor<f64>
    %41 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%37, %39 : tensor<f64>, tensor<f64>) outs(%40 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.addf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %42 = linalg.generic {indexing_maps = [#map1, #map2, #map3, #map1], iterator_types = ["parallel"]} ins(%35, %cst_11, %41 : tensor<6xf64>, tensor<1xi32>, tensor<f64>) outs(%35 : tensor<6xf64>) {
    ^bb0(%arg1: f64, %arg2: i32, %arg3: f64, %arg4: f64):
      %56 = linalg.index 0 : index
      %57 = arith.index_cast %arg2 : i32 to index
      %58 = arith.cmpi eq, %56, %57 : index
      %59 = arith.select %58, %arg3, %arg4 : f64
      linalg.yield %59 : f64
    } -> tensor<6xf64>
    %cst_13 = arith.constant dense<3> : tensor<1xi32>
    %43 = tensor.empty() : tensor<f64>
    %44 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%9, %cst_4 : tensor<f64>, tensor<f64>) outs(%43 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %45 = tensor.empty() : tensor<f64>
    %46 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%21, %cst_7 : tensor<f64>, tensor<f64>) outs(%45 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %47 = tensor.empty() : tensor<f64>
    %48 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%44, %46 : tensor<f64>, tensor<f64>) outs(%47 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.addf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %cst_14 = arith.constant dense<-1.000000e+00> : tensor<f64>
    %49 = tensor.empty() : tensor<f64>
    %50 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%48, %cst_14 : tensor<f64>, tensor<f64>) outs(%49 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<f64>
    %51 = linalg.generic {indexing_maps = [#map1, #map2, #map3, #map1], iterator_types = ["parallel"]} ins(%42, %cst_13, %50 : tensor<6xf64>, tensor<1xi32>, tensor<f64>) outs(%42 : tensor<6xf64>) {
    ^bb0(%arg1: f64, %arg2: i32, %arg3: f64, %arg4: f64):
      %56 = linalg.index 0 : index
      %57 = arith.index_cast %arg2 : i32 to index
      %58 = arith.cmpi eq, %56, %57 : index
      %59 = arith.select %58, %arg3, %arg4 : f64
      linalg.yield %59 : f64
    } -> tensor<6xf64>
    %cst_15 = arith.constant dense<1.600000e+01> : tensor<f64>
    %cst_16 = arith.constant dense<1.600000e+01> : tensor<6xf64>
    %52 = tensor.empty() : tensor<6xf64>
    %53 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%51, %cst_16 : tensor<6xf64>, tensor<6xf64>) outs(%52 : tensor<6xf64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.divf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<6xf64>
    %cst_17 = arith.constant dense<197.32705300000001> : tensor<f64>
    %cst_18 = arith.constant dense<197.32705300000001> : tensor<6xf64>
    %54 = tensor.empty() : tensor<6xf64>
    %55 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%53, %cst_18 : tensor<6xf64>, tensor<6xf64>) outs(%54 : tensor<6xf64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %56 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %56 : f64
    } -> tensor<6xf64>
    return %55 : tensor<6xf64>
  }
  func.func @main() {
    %0 = arith.constant dense<1.91> : tensor<f64>
    %3 = func.call @callee(%0) : (tensor<f64>) -> tensor<6xf64>
    %unranked = tensor.cast %3 : tensor<6xf64> to tensor<*xf64>
    func.call @printMemrefF64(%unranked) : (tensor<*xf64>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [6] strides = [1] data =
    //      CHECK: [-4.67528, 2.11012, 1.00673, 1.55843, 0, 0]
    return
  }
}
