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
module @jit__get_age_weights_from_tables.8 {

  func.func private @printMemrefF32(tensor<*xf32>)

  func.func private @callee(%arg0: tensor<95xf32>, %arg1: tensor<50000xf32>, %arg2: tensor<50000xf32>) -> tensor<94xf32> {
    %cst = arith.constant dense<1.000000e+01> : tensor<f32>
    %cst_0 = arith.constant dense<1.000000e+01> : tensor<95xf32>
    %0 = func.call @interp.145(%arg0, %arg1, %arg2) : (tensor<95xf32>, tensor<50000xf32>, tensor<50000xf32>) -> tensor<95xf32>
    %1 = tensor.empty(): tensor<95xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%cst_0, %0 : tensor<95xf32>, tensor<95xf32>) outs(%1 : tensor<95xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %13 = math.powf %arg3, %arg4 : f32
      linalg.yield %13 : f32
    } -> tensor<95xf32>
    %3 = func.call @diff.224(%2) : (tensor<95xf32>) -> tensor<94xf32>
    %4 = tensor.empty() : tensor<94xf32>
    %5 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%3 : tensor<94xf32>) outs(%4 : tensor<94xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %13 = arith.negf %arg3 : f32
      linalg.yield %13 : f32
    } -> tensor<94xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %6 = tensor.empty() : tensor<f32>
    %7 = linalg.fill ins(%cst_2 : f32) outs(%6 : tensor<f32>) -> tensor<f32>
    %8 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction"]} ins(%5 : tensor<94xf32>) outs(%7 : tensor<f32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %13 = arith.addf %arg3, %arg4 : f32
      linalg.yield %13 : f32
    } -> tensor<f32>
    %9 = tensor.empty() : tensor<94xf32>
    %10 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel"]} ins(%8 : tensor<f32>) outs(%9 : tensor<94xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<94xf32>
    %11 = tensor.empty() : tensor<94xf32>
    %12 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%5, %10 : tensor<94xf32>, tensor<94xf32>) outs(%11 : tensor<94xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %13 = arith.divf %arg3, %arg4 : f32
      linalg.yield %13 : f32
    } -> tensor<94xf32>
    return %12 : tensor<94xf32>
  }
  func.func private @interp.145(%arg0: tensor<95xf32>, %arg1: tensor<50000xf32>, %arg2: tensor<50000xf32>) -> tensor<95xf32> {
    %0 = tensor.extract_slice %arg1[49999] [1] [1] : tensor<50000xf32> to tensor<1xf32>
    %1 = tensor.collapse_shape %0 [] : tensor<1xf32> into tensor<f32>
    %2 = tensor.empty() : tensor<95xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel"]} ins(%1 : tensor<f32>) outs(%2 : tensor<95xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<95xf32>
    %4 = tensor.empty() : tensor<95xi1>
    %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %3 : tensor<95xf32>, tensor<95xf32>) outs(%4 : tensor<95xi1>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: i1):
      %106 = arith.cmpf ogt, %arg3, %arg4 : f32
      linalg.yield %106 : i1
    } -> tensor<95xi1>
    %6 = tensor.extract_slice %arg2[49999] [1] [1] : tensor<50000xf32> to tensor<1xf32>
    %7 = tensor.collapse_shape %6 [] : tensor<1xf32> into tensor<f32>
    %8 = tensor.extract_slice %arg1[0] [1] [1] : tensor<50000xf32> to tensor<1xf32>
    %9 = tensor.collapse_shape %8 [] : tensor<1xf32> into tensor<f32>
    %10 = tensor.empty() : tensor<95xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel"]} ins(%9 : tensor<f32>) outs(%10 : tensor<95xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<95xf32>
    %12 = tensor.empty() : tensor<95xi1>
    %13 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %11 : tensor<95xf32>, tensor<95xf32>) outs(%12 : tensor<95xi1>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: i1):
      %106 = arith.cmpf olt, %arg3, %arg4 : f32
      linalg.yield %106 : i1
    } -> tensor<95xi1>
    %14 = tensor.extract_slice %arg2[0] [1] [1] : tensor<50000xf32> to tensor<1xf32>
    %15 = tensor.collapse_shape %14 [] : tensor<1xf32> into tensor<f32>
    %16 = func.call @searchsorted.104(%arg1, %arg0) : (tensor<50000xf32>, tensor<95xf32>) -> tensor<95xi32>
    %cst = arith.constant dense<1> : tensor<i32>
    %cst_0 = arith.constant dense<49999> : tensor<i32>
    %17 = func.call @clip.120(%16, %cst, %cst_0) : (tensor<95xi32>, tensor<i32>, tensor<i32>) -> tensor<95xi32>
    %cst_1 = arith.constant dense<0> : tensor<i32>
    %cst_2 = arith.constant dense<0> : tensor<95xi32>
    %18 = tensor.empty() : tensor<95xi1>
    %19 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%17, %cst_2 : tensor<95xi32>, tensor<95xi32>) outs(%18 : tensor<95xi1>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i1):
      %106 = arith.cmpi slt, %arg3, %arg4 : i32
      linalg.yield %106 : i1
    } -> tensor<95xi1>
    %cst_3 = arith.constant dense<50000> : tensor<i32>
    %cst_4 = arith.constant dense<50000> : tensor<95xi32>
    %20 = tensor.empty() : tensor<95xi32>
    %21 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%17, %cst_4 : tensor<95xi32>, tensor<95xi32>) outs(%20 : tensor<95xi32>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %106 = arith.addi %arg3, %arg4 : i32
      linalg.yield %106 : i32
    } -> tensor<95xi32>
    %22 = tensor.empty() : tensor<95xi32>
    %23 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%19, %21, %17 : tensor<95xi1>, tensor<95xi32>, tensor<95xi32>) outs(%22 : tensor<95xi32>) {
    ^bb0(%arg3: i1, %arg4: i32, %arg5: i32, %arg6: i32):
      %106 = arith.select %arg3, %arg4, %arg5 : i32
      linalg.yield %106 : i32
    } -> tensor<95xi32>
    %24 = tensor.expand_shape %23 [[0, 1]] : tensor<95xi32> into tensor<95x1xi32>
    %c0 = arith.constant 0 : index
    %25 = tensor.empty() : tensor<95xf32>
    %26 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel"]} outs(%25 : tensor<95xf32>) {
    ^bb0(%arg3: f32):
      %106 = linalg.index 0 : index
      %107 = tensor.extract %24[%106, %c0] : tensor<95x1xi32>
      %108 = arith.index_cast %107 : i32 to index
      %109 = arith.addi %108, %c0 : index
      %110 = tensor.extract %arg1[%109] : tensor<50000xf32>
      linalg.yield %110 : f32
    } -> tensor<95xf32>
    %cst_5 = arith.constant dense<1> : tensor<i32>
    %cst_6 = arith.constant dense<1> : tensor<95xi32>
    %27 = tensor.empty() : tensor<95xi32>
    %28 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%17, %cst_6 : tensor<95xi32>, tensor<95xi32>) outs(%27 : tensor<95xi32>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %106 = arith.subi %arg3, %arg4 : i32
      linalg.yield %106 : i32
    } -> tensor<95xi32>
    %29 = tensor.empty() : tensor<95xi1>
    %30 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%28, %cst_2 : tensor<95xi32>, tensor<95xi32>) outs(%29 : tensor<95xi1>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i1):
      %106 = arith.cmpi slt, %arg3, %arg4 : i32
      linalg.yield %106 : i1
    } -> tensor<95xi1>
    %31 = tensor.empty() : tensor<95xi32>
    %32 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%28, %cst_4 : tensor<95xi32>, tensor<95xi32>) outs(%31 : tensor<95xi32>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %106 = arith.addi %arg3, %arg4 : i32
      linalg.yield %106 : i32
    } -> tensor<95xi32>
    %33 = tensor.empty() : tensor<95xi32>
    %34 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%30, %32, %28 : tensor<95xi1>, tensor<95xi32>, tensor<95xi32>) outs(%33 : tensor<95xi32>) {
    ^bb0(%arg3: i1, %arg4: i32, %arg5: i32, %arg6: i32):
      %106 = arith.select %arg3, %arg4, %arg5 : i32
      linalg.yield %106 : i32
    } -> tensor<95xi32>
    %35 = tensor.expand_shape %34 [[0, 1]] : tensor<95xi32> into tensor<95x1xi32>
    %c0_7 = arith.constant 0 : index
    %36 = tensor.empty() : tensor<95xf32>
    %37 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel"]} outs(%36 : tensor<95xf32>) {
    ^bb0(%arg3: f32):
      %106 = linalg.index 0 : index
      %107 = tensor.extract %35[%106, %c0_7] : tensor<95x1xi32>
      %108 = arith.index_cast %107 : i32 to index
      %109 = arith.addi %108, %c0_7 : index
      %110 = tensor.extract %arg1[%109] : tensor<50000xf32>
      linalg.yield %110 : f32
    } -> tensor<95xf32>
    %38 = tensor.empty() : tensor<95xf32>
    %39 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%26, %37 : tensor<95xf32>, tensor<95xf32>) outs(%38 : tensor<95xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %106 = arith.subf %arg3, %arg4 : f32
      linalg.yield %106 : f32
    } -> tensor<95xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<95xf32>
    %40 = tensor.empty() : tensor<95xi1>
    %41 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%39, %cst_9 : tensor<95xf32>, tensor<95xf32>) outs(%40 : tensor<95xi1>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: i1):
      %106 = arith.cmpf oeq, %arg3, %arg4 : f32
      linalg.yield %106 : i1
    } -> tensor<95xi1>
    %42 = tensor.empty() : tensor<95xi1>
    %43 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%17, %cst_2 : tensor<95xi32>, tensor<95xi32>) outs(%42 : tensor<95xi1>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i1):
      %106 = arith.cmpi slt, %arg3, %arg4 : i32
      linalg.yield %106 : i1
    } -> tensor<95xi1>
    %44 = tensor.empty() : tensor<95xi32>
    %45 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%17, %cst_4 : tensor<95xi32>, tensor<95xi32>) outs(%44 : tensor<95xi32>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %106 = arith.addi %arg3, %arg4 : i32
      linalg.yield %106 : i32
    } -> tensor<95xi32>
    %46 = tensor.empty() : tensor<95xi32>
    %47 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%43, %45, %17 : tensor<95xi1>, tensor<95xi32>, tensor<95xi32>) outs(%46 : tensor<95xi32>) {
    ^bb0(%arg3: i1, %arg4: i32, %arg5: i32, %arg6: i32):
      %106 = arith.select %arg3, %arg4, %arg5 : i32
      linalg.yield %106 : i32
    } -> tensor<95xi32>
    %48 = tensor.expand_shape %47 [[0, 1]] : tensor<95xi32> into tensor<95x1xi32>
    %c0_10 = arith.constant 0 : index
    %49 = tensor.empty() : tensor<95xf32>
    %50 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel"]} outs(%49 : tensor<95xf32>) {
    ^bb0(%arg3: f32):
      %106 = linalg.index 0 : index
      %107 = tensor.extract %48[%106, %c0_10] : tensor<95x1xi32>
      %108 = arith.index_cast %107 : i32 to index
      %109 = arith.addi %108, %c0_10 : index
      %110 = tensor.extract %arg2[%109] : tensor<50000xf32>
      linalg.yield %110 : f32
    } -> tensor<95xf32>
    %51 = tensor.empty() : tensor<95xi32>
    %52 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%17, %cst_6 : tensor<95xi32>, tensor<95xi32>) outs(%51 : tensor<95xi32>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %106 = arith.subi %arg3, %arg4 : i32
      linalg.yield %106 : i32
    } -> tensor<95xi32>
    %53 = tensor.empty() : tensor<95xi1>
    %54 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%52, %cst_2 : tensor<95xi32>, tensor<95xi32>) outs(%53 : tensor<95xi1>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i1):
      %106 = arith.cmpi slt, %arg3, %arg4 : i32
      linalg.yield %106 : i1
    } -> tensor<95xi1>
    %55 = tensor.empty() : tensor<95xi32>
    %56 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%52, %cst_4 : tensor<95xi32>, tensor<95xi32>) outs(%55 : tensor<95xi32>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %106 = arith.addi %arg3, %arg4 : i32
      linalg.yield %106 : i32
    } -> tensor<95xi32>
    %57 = tensor.empty() : tensor<95xi32>
    %58 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%54, %56, %52 : tensor<95xi1>, tensor<95xi32>, tensor<95xi32>) outs(%57 : tensor<95xi32>) {
    ^bb0(%arg3: i1, %arg4: i32, %arg5: i32, %arg6: i32):
      %106 = arith.select %arg3, %arg4, %arg5 : i32
      linalg.yield %106 : i32
    } -> tensor<95xi32>
    %59 = tensor.expand_shape %58 [[0, 1]] : tensor<95xi32> into tensor<95x1xi32>
    %c0_11 = arith.constant 0 : index
    %60 = tensor.empty() : tensor<95xf32>
    %61 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel"]} outs(%60 : tensor<95xf32>) {
    ^bb0(%arg3: f32):
      %106 = linalg.index 0 : index
      %107 = tensor.extract %59[%106, %c0_11] : tensor<95x1xi32>
      %108 = arith.index_cast %107 : i32 to index
      %109 = arith.addi %108, %c0_11 : index
      %110 = tensor.extract %arg2[%109] : tensor<50000xf32>
      linalg.yield %110 : f32
    } -> tensor<95xf32>
    %62 = tensor.empty() : tensor<95xi32>
    %63 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%17, %cst_6 : tensor<95xi32>, tensor<95xi32>) outs(%62 : tensor<95xi32>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %106 = arith.subi %arg3, %arg4 : i32
      linalg.yield %106 : i32
    } -> tensor<95xi32>
    %64 = tensor.empty() : tensor<95xi1>
    %65 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%63, %cst_2 : tensor<95xi32>, tensor<95xi32>) outs(%64 : tensor<95xi1>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i1):
      %106 = arith.cmpi slt, %arg3, %arg4 : i32
      linalg.yield %106 : i1
    } -> tensor<95xi1>
    %66 = tensor.empty() : tensor<95xi32>
    %67 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%63, %cst_4 : tensor<95xi32>, tensor<95xi32>) outs(%66 : tensor<95xi32>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %106 = arith.addi %arg3, %arg4 : i32
      linalg.yield %106 : i32
    } -> tensor<95xi32>
    %68 = tensor.empty() : tensor<95xi32>
    %69 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%65, %67, %63 : tensor<95xi1>, tensor<95xi32>, tensor<95xi32>) outs(%68 : tensor<95xi32>) {
    ^bb0(%arg3: i1, %arg4: i32, %arg5: i32, %arg6: i32):
      %106 = arith.select %arg3, %arg4, %arg5 : i32
      linalg.yield %106 : i32
    } -> tensor<95xi32>
    %70 = tensor.expand_shape %69 [[0, 1]] : tensor<95xi32> into tensor<95x1xi32>
    %c0_12 = arith.constant 0 : index
    %71 = tensor.empty() : tensor<95xf32>
    %72 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel"]} outs(%71 : tensor<95xf32>) {
    ^bb0(%arg3: f32):
      %106 = linalg.index 0 : index
      %107 = tensor.extract %70[%106, %c0_12] : tensor<95x1xi32>
      %108 = arith.index_cast %107 : i32 to index
      %109 = arith.addi %108, %c0_12 : index
      %110 = tensor.extract %arg1[%109] : tensor<50000xf32>
      linalg.yield %110 : f32
    } -> tensor<95xf32>
    %73 = tensor.empty() : tensor<95xf32>
    %74 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %72 : tensor<95xf32>, tensor<95xf32>) outs(%73 : tensor<95xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %106 = arith.subf %arg3, %arg4 : f32
      linalg.yield %106 : f32
    } -> tensor<95xf32>
    %75 = tensor.empty() : tensor<95xf32>
    %76 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%74, %39 : tensor<95xf32>, tensor<95xf32>) outs(%75 : tensor<95xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %106 = arith.divf %arg3, %arg4 : f32
      linalg.yield %106 : f32
    } -> tensor<95xf32>
    %77 = tensor.empty() : tensor<95xi1>
    %78 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%17, %cst_2 : tensor<95xi32>, tensor<95xi32>) outs(%77 : tensor<95xi1>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i1):
      %106 = arith.cmpi slt, %arg3, %arg4 : i32
      linalg.yield %106 : i1
    } -> tensor<95xi1>
    %79 = tensor.empty() : tensor<95xi32>
    %80 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%17, %cst_4 : tensor<95xi32>, tensor<95xi32>) outs(%79 : tensor<95xi32>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %106 = arith.addi %arg3, %arg4 : i32
      linalg.yield %106 : i32
    } -> tensor<95xi32>
    %81 = tensor.empty() : tensor<95xi32>
    %82 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%78, %80, %17 : tensor<95xi1>, tensor<95xi32>, tensor<95xi32>) outs(%81 : tensor<95xi32>) {
    ^bb0(%arg3: i1, %arg4: i32, %arg5: i32, %arg6: i32):
      %106 = arith.select %arg3, %arg4, %arg5 : i32
      linalg.yield %106 : i32
    } -> tensor<95xi32>
    %83 = tensor.expand_shape %82 [[0, 1]] : tensor<95xi32> into tensor<95x1xi32>
    %c0_13 = arith.constant 0 : index
    %84 = tensor.empty() : tensor<95xf32>
    %85 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel"]} outs(%84 : tensor<95xf32>) {
    ^bb0(%arg3: f32):
      %106 = linalg.index 0 : index
      %107 = tensor.extract %83[%106, %c0_13] : tensor<95x1xi32>
      %108 = arith.index_cast %107 : i32 to index
      %109 = arith.addi %108, %c0_13 : index
      %110 = tensor.extract %arg2[%109] : tensor<50000xf32>
      linalg.yield %110 : f32
    } -> tensor<95xf32>
    %86 = tensor.empty() : tensor<95xi32>
    %87 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%17, %cst_6 : tensor<95xi32>, tensor<95xi32>) outs(%86 : tensor<95xi32>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %106 = arith.subi %arg3, %arg4 : i32
      linalg.yield %106 : i32
    } -> tensor<95xi32>
    %88 = tensor.empty() : tensor<95xi1>
    %89 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%87, %cst_2 : tensor<95xi32>, tensor<95xi32>) outs(%88 : tensor<95xi1>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i1):
      %106 = arith.cmpi slt, %arg3, %arg4 : i32
      linalg.yield %106 : i1
    } -> tensor<95xi1>
    %90 = tensor.empty() : tensor<95xi32>
    %91 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%87, %cst_4 : tensor<95xi32>, tensor<95xi32>) outs(%90 : tensor<95xi32>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %106 = arith.addi %arg3, %arg4 : i32
      linalg.yield %106 : i32
    } -> tensor<95xi32>
    %92 = tensor.empty() : tensor<95xi32>
    %93 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%89, %91, %87 : tensor<95xi1>, tensor<95xi32>, tensor<95xi32>) outs(%92 : tensor<95xi32>) {
    ^bb0(%arg3: i1, %arg4: i32, %arg5: i32, %arg6: i32):
      %106 = arith.select %arg3, %arg4, %arg5 : i32
      linalg.yield %106 : i32
    } -> tensor<95xi32>
    %94 = tensor.expand_shape %93 [[0, 1]] : tensor<95xi32> into tensor<95x1xi32>
    %c0_14 = arith.constant 0 : index
    %95 = tensor.empty() : tensor<95xf32>
    %96 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel"]} outs(%95 : tensor<95xf32>) {
    ^bb0(%arg3: f32):
      %106 = linalg.index 0 : index
      %107 = tensor.extract %94[%106, %c0_14] : tensor<95x1xi32>
      %108 = arith.index_cast %107 : i32 to index
      %109 = arith.addi %108, %c0_14 : index
      %110 = tensor.extract %arg2[%109] : tensor<50000xf32>
      linalg.yield %110 : f32
    } -> tensor<95xf32>
    %97 = tensor.empty() : tensor<95xf32>
    %98 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%85, %96 : tensor<95xf32>, tensor<95xf32>) outs(%97 : tensor<95xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %106 = arith.subf %arg3, %arg4 : f32
      linalg.yield %106 : f32
    } -> tensor<95xf32>
    %99 = tensor.empty() : tensor<95xf32>
    %100 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%76, %98 : tensor<95xf32>, tensor<95xf32>) outs(%99 : tensor<95xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %106 = arith.mulf %arg3, %arg4 : f32
      linalg.yield %106 : f32
    } -> tensor<95xf32>
    %101 = tensor.empty() : tensor<95xf32>
    %102 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%61, %100 : tensor<95xf32>, tensor<95xf32>) outs(%101 : tensor<95xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %106 = arith.addf %arg3, %arg4 : f32
      linalg.yield %106 : f32
    } -> tensor<95xf32>
    %103 = func.call @_where.128(%41, %50, %102) : (tensor<95xi1>, tensor<95xf32>, tensor<95xf32>) -> tensor<95xf32>
    %104 = func.call @_where_2.133(%13, %15, %103) : (tensor<95xi1>, tensor<f32>, tensor<95xf32>) -> tensor<95xf32>
    %105 = func.call @_where_3.139(%5, %7, %104) : (tensor<95xi1>, tensor<f32>, tensor<95xf32>) -> tensor<95xf32>
    return %105 : tensor<95xf32>
  }
  func.func private @searchsorted.104(%arg0: tensor<50000xf32>, %arg1: tensor<95xf32>) -> tensor<95xi32> {
    %cst = arith.constant dense<0> : tensor<i32>
    %cst_0 = arith.constant dense<0> : tensor<i32>
    %cst_1 = arith.constant dense<0> : tensor<95xi32>
    %cst_2 = arith.constant dense<50000> : tensor<i32>
    %cst_3 = arith.constant dense<50000> : tensor<95xi32>
    %0:6 = scf.while (%arg2 = %cst, %arg3 = %cst, %arg4 = %cst_1, %arg5 = %cst_3, %arg6 = %arg0, %arg7 = %arg1) : (tensor<i32>, tensor<i32>, tensor<95xi32>, tensor<95xi32>, tensor<50000xf32>, tensor<95xf32>) -> (tensor<i32>, tensor<i32>, tensor<95xi32>, tensor<95xi32>, tensor<50000xf32>, tensor<95xf32>) {
      %cst_4 = arith.constant dense<16> : tensor<i32>
      %1 = tensor.empty() : tensor<i1>
      %2 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%arg2, %cst_4 : tensor<i32>, tensor<i32>) outs(%1 : tensor<i1>) {
      ^bb0(%arg8: i32, %arg9: i32, %arg10: i1):
        %4 = arith.cmpi slt, %arg8, %arg9 : i32
        linalg.yield %4 : i1
      } -> tensor<i1>
      %3 = tensor.extract %2[] : tensor<i1>
      scf.condition(%3) %arg2, %arg3, %arg4, %arg5, %arg6, %arg7 : tensor<i32>, tensor<i32>, tensor<95xi32>, tensor<95xi32>, tensor<50000xf32>, tensor<95xf32>
    } do {
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<95xi32>, %arg5: tensor<95xi32>, %arg6: tensor<50000xf32>, %arg7: tensor<95xf32>):
      %cst_4 = arith.constant dense<1> : tensor<i32>
      %1 = tensor.empty() : tensor<i32>
      %2 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%arg2, %cst_4 : tensor<i32>, tensor<i32>) outs(%1 : tensor<i32>) {
      ^bb0(%arg8: i32, %arg9: i32, %arg10: i32):
        %61 = arith.addi %arg8, %arg9 : i32
        linalg.yield %61 : i32
      } -> tensor<i32>
      %3 = tensor.empty() : tensor<i32>
      %4 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%arg3, %cst_4 : tensor<i32>, tensor<i32>) outs(%3 : tensor<i32>) {
      ^bb0(%arg8: i32, %arg9: i32, %arg10: i32):
        %61 = arith.addi %arg8, %arg9 : i32
        linalg.yield %61 : i32
      } -> tensor<i32>
      %5 = tensor.empty() : tensor<95xi1>
      %6 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg7, %arg7 : tensor<95xf32>, tensor<95xf32>) outs(%5 : tensor<95xi1>) {
      ^bb0(%arg8: f32, %arg9: f32, %arg10: i1):
        %61 = arith.cmpf une, %arg8, %arg9 : f32
        linalg.yield %61 : i1
      } -> tensor<95xi1>
      %cst_5 = arith.constant dense<2143289344> : tensor<i32>
      %cst_6 = arith.constant dense<2143289344> : tensor<95xi32>
      %cst_7 = arith.constant dense<0.000000e+00> : tensor<f32>
      %cst_8 = arith.constant dense<0.000000e+00> : tensor<95xf32>
      %7 = tensor.empty() : tensor<95xi1>
      %8 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg7, %cst_8 : tensor<95xf32>, tensor<95xf32>) outs(%7 : tensor<95xi1>) {
      ^bb0(%arg8: f32, %arg9: f32, %arg10: i1):
        %61 = arith.cmpf oeq, %arg8, %arg9 : f32
        linalg.yield %61 : i1
      } -> tensor<95xi1>
      %cst_9 = arith.constant dense<0> : tensor<i32>
      %cst_10 = arith.constant dense<0> : tensor<95xi32>
      %9 = tensor.empty() : tensor<95xi32>
      %10 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%arg7 : tensor<95xf32>) outs(%9 : tensor<95xi32>) {
      ^bb0(%arg8: f32, %arg9: i32):
        %61 = arith.bitcast %arg8 : f32 to i32
        linalg.yield %61 : i32
      } -> tensor<95xi32>
      %11 = tensor.empty() : tensor<95xi32>
      %12 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%8, %cst_10, %10 : tensor<95xi1>, tensor<95xi32>, tensor<95xi32>) outs(%11 : tensor<95xi32>) {
      ^bb0(%arg8: i1, %arg9: i32, %arg10: i32, %arg11: i32):
        %61 = arith.select %arg8, %arg9, %arg10 : i32
        linalg.yield %61 : i32
      } -> tensor<95xi32>
      %13 = tensor.empty() : tensor<95xi32>
      %14 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%6, %cst_6, %12 : tensor<95xi1>, tensor<95xi32>, tensor<95xi32>) outs(%13 : tensor<95xi32>) {
      ^bb0(%arg8: i1, %arg9: i32, %arg10: i32, %arg11: i32):
        %61 = arith.select %arg8, %arg9, %arg10 : i32
        linalg.yield %61 : i32
      } -> tensor<95xi32>
      %15 = tensor.empty() : tensor<95xi1>
      %16 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%14, %cst_10 : tensor<95xi32>, tensor<95xi32>) outs(%15 : tensor<95xi1>) {
      ^bb0(%arg8: i32, %arg9: i32, %arg10: i1):
        %61 = arith.cmpi slt, %arg8, %arg9 : i32
        linalg.yield %61 : i1
      } -> tensor<95xi1>
      %cst_11 = arith.constant dense<2147483647> : tensor<i32>
      %cst_12 = arith.constant dense<2147483647> : tensor<95xi32>
      %17 = tensor.empty() : tensor<95xi32>
      %18 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%arg7 : tensor<95xf32>) outs(%17 : tensor<95xi32>) {
      ^bb0(%arg8: f32, %arg9: i32):
        %61 = arith.bitcast %arg8 : f32 to i32
        linalg.yield %61 : i32
      } -> tensor<95xi32>
      %19 = tensor.empty() : tensor<95xi32>
      %20 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%cst_12, %18 : tensor<95xi32>, tensor<95xi32>) outs(%19 : tensor<95xi32>) {
      ^bb0(%arg8: i32, %arg9: i32, %arg10: i32):
        %61 = arith.subi %arg8, %arg9 : i32
        linalg.yield %61 : i32
      } -> tensor<95xi32>
      %21 = tensor.empty() : tensor<95xi32>
      %22 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%20 : tensor<95xi32>) outs(%21 : tensor<95xi32>) {
      ^bb0(%arg8: i32, %arg9: i32):
        %61 = arith.bitcast %arg8 : i32 to i32
        linalg.yield %61 : i32
      } -> tensor<95xi32>
      %23 = tensor.empty() : tensor<95xi32>
      %24 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%16, %22, %14 : tensor<95xi1>, tensor<95xi32>, tensor<95xi32>) outs(%23 : tensor<95xi32>) {
      ^bb0(%arg8: i1, %arg9: i32, %arg10: i32, %arg11: i32):
        %61 = arith.select %arg8, %arg9, %arg10 : i32
        linalg.yield %61 : i32
      } -> tensor<95xi32>
      %25 = tensor.empty() : tensor<95xi32>
      %26 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg4, %arg5 : tensor<95xi32>, tensor<95xi32>) outs(%25 : tensor<95xi32>) {
      ^bb0(%arg8: i32, %arg9: i32, %arg10: i32):
        %61 = arith.addi %arg8, %arg9 : i32
        linalg.yield %61 : i32
      } -> tensor<95xi32>
      %cst_13 = arith.constant dense<2> : tensor<i32>
      %27 = func.call @vmap_floor_divide_.12(%26, %cst_13) : (tensor<95xi32>, tensor<i32>) -> tensor<95xi32>
      %28 = tensor.empty() : tensor<95xi1>
      %29 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%27, %cst_10 : tensor<95xi32>, tensor<95xi32>) outs(%28 : tensor<95xi1>) {
      ^bb0(%arg8: i32, %arg9: i32, %arg10: i1):
        %61 = arith.cmpi slt, %arg8, %arg9 : i32
        linalg.yield %61 : i1
      } -> tensor<95xi1>
      %cst_14 = arith.constant dense<50000> : tensor<i32>
      %cst_15 = arith.constant dense<50000> : tensor<95xi32>
      %30 = tensor.empty() : tensor<95xi32>
      %31 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%27, %cst_15 : tensor<95xi32>, tensor<95xi32>) outs(%30 : tensor<95xi32>) {
      ^bb0(%arg8: i32, %arg9: i32, %arg10: i32):
        %61 = arith.addi %arg8, %arg9 : i32
        linalg.yield %61 : i32
      } -> tensor<95xi32>
      %32 = tensor.empty() : tensor<95xi32>
      %33 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%29, %31, %27 : tensor<95xi1>, tensor<95xi32>, tensor<95xi32>) outs(%32 : tensor<95xi32>) {
      ^bb0(%arg8: i1, %arg9: i32, %arg10: i32, %arg11: i32):
        %61 = arith.select %arg8, %arg9, %arg10 : i32
        linalg.yield %61 : i32
      } -> tensor<95xi32>
      %34 = tensor.expand_shape %33 [[0, 1]] : tensor<95xi32> into tensor<95x1xi32>
      %c0 = arith.constant 0 : index
      %35 = tensor.empty() : tensor<95xf32>
      %36 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel"]} outs(%35 : tensor<95xf32>) {
      ^bb0(%arg8: f32):
        %61 = linalg.index 0 : index
        %62 = tensor.extract %34[%61, %c0] : tensor<95x1xi32>
        %63 = arith.index_cast %62 : i32 to index
        %64 = arith.addi %63, %c0 : index
        %65 = tensor.extract %arg6[%64] : tensor<50000xf32>
        linalg.yield %65 : f32
      } -> tensor<95xf32>
      %37 = tensor.empty() : tensor<95xi1>
      %38 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%36, %36 : tensor<95xf32>, tensor<95xf32>) outs(%37 : tensor<95xi1>) {
      ^bb0(%arg8: f32, %arg9: f32, %arg10: i1):
        %61 = arith.cmpf une, %arg8, %arg9 : f32
        linalg.yield %61 : i1
      } -> tensor<95xi1>
      %39 = tensor.empty() : tensor<95xi1>
      %40 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%36, %cst_8 : tensor<95xf32>, tensor<95xf32>) outs(%39 : tensor<95xi1>) {
      ^bb0(%arg8: f32, %arg9: f32, %arg10: i1):
        %61 = arith.cmpf oeq, %arg8, %arg9 : f32
        linalg.yield %61 : i1
      } -> tensor<95xi1>
      %41 = tensor.empty() : tensor<95xi32>
      %42 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%36 : tensor<95xf32>) outs(%41 : tensor<95xi32>) {
      ^bb0(%arg8: f32, %arg9: i32):
        %61 = arith.bitcast %arg8 : f32 to i32
        linalg.yield %61 : i32
      } -> tensor<95xi32>
      %43 = tensor.empty() : tensor<95xi32>
      %44 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%40, %cst_10, %42 : tensor<95xi1>, tensor<95xi32>, tensor<95xi32>) outs(%43 : tensor<95xi32>) {
      ^bb0(%arg8: i1, %arg9: i32, %arg10: i32, %arg11: i32):
        %61 = arith.select %arg8, %arg9, %arg10 : i32
        linalg.yield %61 : i32
      } -> tensor<95xi32>
      %45 = tensor.empty() : tensor<95xi32>
      %46 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%38, %cst_6, %44 : tensor<95xi1>, tensor<95xi32>, tensor<95xi32>) outs(%45 : tensor<95xi32>) {
      ^bb0(%arg8: i1, %arg9: i32, %arg10: i32, %arg11: i32):
        %61 = arith.select %arg8, %arg9, %arg10 : i32
        linalg.yield %61 : i32
      } -> tensor<95xi32>
      %47 = tensor.empty() : tensor<95xi1>
      %48 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%46, %cst_10 : tensor<95xi32>, tensor<95xi32>) outs(%47 : tensor<95xi1>) {
      ^bb0(%arg8: i32, %arg9: i32, %arg10: i1):
        %61 = arith.cmpi slt, %arg8, %arg9 : i32
        linalg.yield %61 : i1
      } -> tensor<95xi1>
      %49 = tensor.empty() : tensor<95xi32>
      %50 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%36 : tensor<95xf32>) outs(%49 : tensor<95xi32>) {
      ^bb0(%arg8: f32, %arg9: i32):
        %61 = arith.bitcast %arg8 : f32 to i32
        linalg.yield %61 : i32
      } -> tensor<95xi32>
      %51 = tensor.empty() : tensor<95xi32>
      %52 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%cst_12, %50 : tensor<95xi32>, tensor<95xi32>) outs(%51 : tensor<95xi32>) {
      ^bb0(%arg8: i32, %arg9: i32, %arg10: i32):
        %61 = arith.subi %arg8, %arg9 : i32
        linalg.yield %61 : i32
      } -> tensor<95xi32>
      %53 = tensor.empty() : tensor<95xi32>
      %54 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%52 : tensor<95xi32>) outs(%53 : tensor<95xi32>) {
      ^bb0(%arg8: i32, %arg9: i32):
        %61 = arith.bitcast %arg8 : i32 to i32
        linalg.yield %61 : i32
      } -> tensor<95xi32>
      %55 = tensor.empty() : tensor<95xi32>
      %56 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%48, %54, %46 : tensor<95xi1>, tensor<95xi32>, tensor<95xi32>) outs(%55 : tensor<95xi32>) {
      ^bb0(%arg8: i1, %arg9: i32, %arg10: i32, %arg11: i32):
        %61 = arith.select %arg8, %arg9, %arg10 : i32
        linalg.yield %61 : i32
      } -> tensor<95xi32>
      %57 = tensor.empty() : tensor<95xi1>
      %58 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%24, %56 : tensor<95xi32>, tensor<95xi32>) outs(%57 : tensor<95xi1>) {
      ^bb0(%arg8: i32, %arg9: i32, %arg10: i1):
        %61 = arith.cmpi slt, %arg8, %arg9 : i32
        linalg.yield %61 : i1
      } -> tensor<95xi1>
      %59 = func.call @vmap__where__0.31(%58, %arg4, %27) : (tensor<95xi1>, tensor<95xi32>, tensor<95xi32>) -> tensor<95xi32>
      %60 = func.call @vmap__where__1.36(%58, %27, %arg5) : (tensor<95xi1>, tensor<95xi32>, tensor<95xi32>) -> tensor<95xi32>
      scf.yield %2, %4, %59, %60, %arg6, %arg7 : tensor<i32>, tensor<i32>, tensor<95xi32>, tensor<95xi32>, tensor<50000xf32>, tensor<95xf32>
    }
    return %0#3 : tensor<95xi32>
  }
  func.func private @vmap_floor_divide_.12(%arg0: tensor<95xi32>, %arg1: tensor<i32>) -> tensor<95xi32> {
    %0 = tensor.empty() : tensor<95xi32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%arg0 : tensor<95xi32>) outs(%0 : tensor<95xi32>) {
    ^bb0(%arg2: i32, %arg3: i32):
      %c0_i32 = arith.constant 0 : i32
      %c31_i32 = arith.constant 31 : i32
      %c1_i32 = arith.constant 1 : i32
      %23 = arith.cmpi eq, %arg2, %c0_i32 : i32
      %24 = arith.shrsi %arg2, %c31_i32 : i32
      %25 = arith.ori %24, %c1_i32 : i32
      %26 = arith.select %23, %c0_i32, %25 : i32
      linalg.yield %26 : i32
    } -> tensor<95xi32>
    %2 = tensor.empty() : tensor<i32>
    %3 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%arg1 : tensor<i32>) outs(%2 : tensor<i32>) {
    ^bb0(%arg2: i32, %arg3: i32):
      %c0_i32 = arith.constant 0 : i32
      %c31_i32 = arith.constant 31 : i32
      %c1_i32 = arith.constant 1 : i32
      %23 = arith.cmpi eq, %arg2, %c0_i32 : i32
      %24 = arith.shrsi %arg2, %c31_i32 : i32
      %25 = arith.ori %24, %c1_i32 : i32
      %26 = arith.select %23, %c0_i32, %25 : i32
      linalg.yield %26 : i32
    } -> tensor<i32>
    %4 = tensor.empty() : tensor<95xi32>
    %5 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel"]} ins(%3 : tensor<i32>) outs(%4 : tensor<95xi32>) {
    ^bb0(%arg2: i32, %arg3: i32):
      linalg.yield %arg2 : i32
    } -> tensor<95xi32>
    %6 = tensor.empty() : tensor<95xi1>
    %7 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%1, %5 : tensor<95xi32>, tensor<95xi32>) outs(%6 : tensor<95xi1>) {
    ^bb0(%arg2: i32, %arg3: i32, %arg4: i1):
      %23 = arith.cmpi ne, %arg2, %arg3 : i32
      linalg.yield %23 : i1
    } -> tensor<95xi1>
    %8 = tensor.empty() : tensor<95xi32>
    %9 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel"]} ins(%arg1 : tensor<i32>) outs(%8 : tensor<95xi32>) {
    ^bb0(%arg2: i32, %arg3: i32):
      linalg.yield %arg2 : i32
    } -> tensor<95xi32>
    %10 = tensor.empty() : tensor<95xi32>
    %11 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %9 : tensor<95xi32>, tensor<95xi32>) outs(%10 : tensor<95xi32>) {
    ^bb0(%arg2: i32, %arg3: i32, %arg4: i32):
      %23 = arith.remsi %arg2, %arg3 : i32
      linalg.yield %23 : i32
    } -> tensor<95xi32>
    %cst = arith.constant dense<0> : tensor<i32>
    %cst_0 = arith.constant dense<0> : tensor<95xi32>
    %12 = tensor.empty() : tensor<95xi1>
    %13 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%11, %cst_0 : tensor<95xi32>, tensor<95xi32>) outs(%12 : tensor<95xi1>) {
    ^bb0(%arg2: i32, %arg3: i32, %arg4: i1):
      %23 = arith.cmpi ne, %arg2, %arg3 : i32
      linalg.yield %23 : i1
    } -> tensor<95xi1>
    %14 = tensor.empty() : tensor<95xi1>
    %15 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%7, %13 : tensor<95xi1>, tensor<95xi1>) outs(%14 : tensor<95xi1>) {
    ^bb0(%arg2: i1, %arg3: i1, %arg4: i1):
      %23 = arith.andi %arg2, %arg3 : i1
      linalg.yield %23 : i1
    } -> tensor<95xi1>
    %16 = tensor.empty() : tensor<95xi32>
    %17 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel"]} ins(%arg1 : tensor<i32>) outs(%16 : tensor<95xi32>) {
    ^bb0(%arg2: i32, %arg3: i32):
      linalg.yield %arg2 : i32
    } -> tensor<95xi32>
    %18 = tensor.empty() : tensor<95xi32>
    %19 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %17 : tensor<95xi32>, tensor<95xi32>) outs(%18 : tensor<95xi32>) {
    ^bb0(%arg2: i32, %arg3: i32, %arg4: i32):
      %23 = arith.divsi %arg2, %arg3 : i32
      linalg.yield %23 : i32
    } -> tensor<95xi32>
    %cst_1 = arith.constant dense<1> : tensor<i32>
    %cst_2 = arith.constant dense<1> : tensor<95xi32>
    %20 = tensor.empty() : tensor<95xi32>
    %21 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%19, %cst_2 : tensor<95xi32>, tensor<95xi32>) outs(%20 : tensor<95xi32>) {
    ^bb0(%arg2: i32, %arg3: i32, %arg4: i32):
      %23 = arith.subi %arg2, %arg3 : i32
      linalg.yield %23 : i32
    } -> tensor<95xi32>
    %22 = func.call @vmap__where_.7(%15, %21, %19) : (tensor<95xi1>, tensor<95xi32>, tensor<95xi32>) -> tensor<95xi32>
    return %22 : tensor<95xi32>
  }
  func.func private @vmap__where_.7(%arg0: tensor<95xi1>, %arg1: tensor<95xi32>, %arg2: tensor<95xi32>) -> tensor<95xi32> {
    %0 = tensor.empty() : tensor<95xi32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %arg1, %arg2 : tensor<95xi1>, tensor<95xi32>, tensor<95xi32>) outs(%0 : tensor<95xi32>) {
    ^bb0(%arg3: i1, %arg4: i32, %arg5: i32, %arg6: i32):
      %2 = arith.select %arg3, %arg4, %arg5 : i32
      linalg.yield %2 : i32
    } -> tensor<95xi32>
    return %1 : tensor<95xi32>
  }
  func.func private @vmap__where__0.31(%arg0: tensor<95xi1>, %arg1: tensor<95xi32>, %arg2: tensor<95xi32>) -> tensor<95xi32> {
    %0 = tensor.empty() : tensor<95xi32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %arg1, %arg2 : tensor<95xi1>, tensor<95xi32>, tensor<95xi32>) outs(%0 : tensor<95xi32>) {
    ^bb0(%arg3: i1, %arg4: i32, %arg5: i32, %arg6: i32):
      %2 = arith.select %arg3, %arg4, %arg5 : i32
      linalg.yield %2 : i32
    } -> tensor<95xi32>
    return %1 : tensor<95xi32>
  }
  func.func private @vmap__where__1.36(%arg0: tensor<95xi1>, %arg1: tensor<95xi32>, %arg2: tensor<95xi32>) -> tensor<95xi32> {
    %0 = tensor.empty() : tensor<95xi32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %arg1, %arg2 : tensor<95xi1>, tensor<95xi32>, tensor<95xi32>) outs(%0 : tensor<95xi32>) {
    ^bb0(%arg3: i1, %arg4: i32, %arg5: i32, %arg6: i32):
      %2 = arith.select %arg3, %arg4, %arg5 : i32
      linalg.yield %2 : i32
    } -> tensor<95xi32>
    return %1 : tensor<95xi32>
  }
  func.func private @clip.120(%arg0: tensor<95xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<95xi32> {
    %0 = tensor.empty() : tensor<95xi32>
    %1 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel"]} ins(%arg2 : tensor<i32>) outs(%0 : tensor<95xi32>) {
    ^bb0(%arg3: i32, %arg4: i32):
      linalg.yield %arg3 : i32
    } -> tensor<95xi32>
    %2 = tensor.empty() : tensor<95xi32>
    %3 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel"]} ins(%arg1 : tensor<i32>) outs(%2 : tensor<95xi32>) {
    ^bb0(%arg3: i32, %arg4: i32):
      linalg.yield %arg3 : i32
    } -> tensor<95xi32>
    %4 = tensor.empty() : tensor<95xi32>
    %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%3, %arg0 : tensor<95xi32>, tensor<95xi32>) outs(%4 : tensor<95xi32>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %8 = arith.maxsi %arg3, %arg4 : i32
      linalg.yield %8 : i32
    } -> tensor<95xi32>
    %6 = tensor.empty() : tensor<95xi32>
    %7 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%1, %5 : tensor<95xi32>, tensor<95xi32>) outs(%6 : tensor<95xi32>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %8 = arith.minsi %arg3, %arg4 : i32
      linalg.yield %8 : i32
    } -> tensor<95xi32>
    return %7 : tensor<95xi32>
  }
  func.func private @_where.128(%arg0: tensor<95xi1>, %arg1: tensor<95xf32>, %arg2: tensor<95xf32>) -> tensor<95xf32> {
    %0 = tensor.empty() : tensor<95xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %arg1, %arg2 : tensor<95xi1>, tensor<95xf32>, tensor<95xf32>) outs(%0 : tensor<95xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %2 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %2 : f32
    } -> tensor<95xf32>
    return %1 : tensor<95xf32>
  }
  func.func private @_where_2.133(%arg0: tensor<95xi1>, %arg1: tensor<f32>, %arg2: tensor<95xf32>) -> tensor<95xf32> {
    %0 = tensor.empty() : tensor<95xf32>
    %1 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel"]} ins(%arg1 : tensor<f32>) outs(%0 : tensor<95xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<95xf32>
    %2 = tensor.empty() : tensor<95xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %1, %arg2 : tensor<95xi1>, tensor<95xf32>, tensor<95xf32>) outs(%2 : tensor<95xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %4 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %4 : f32
    } -> tensor<95xf32>
    return %3 : tensor<95xf32>
  }
  func.func private @_where_3.139(%arg0: tensor<95xi1>, %arg1: tensor<f32>, %arg2: tensor<95xf32>) -> tensor<95xf32> {
    %0 = tensor.empty() : tensor<95xf32>
    %1 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel"]} ins(%arg1 : tensor<f32>) outs(%0 : tensor<95xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<95xf32>
    %2 = tensor.empty() : tensor<95xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %1, %arg2 : tensor<95xi1>, tensor<95xf32>, tensor<95xf32>) outs(%2 : tensor<95xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %4 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %4 : f32
    } -> tensor<95xf32>
    return %3 : tensor<95xf32>
  }
  func.func private @diff.224(%arg0: tensor<95xf32>) -> tensor<94xf32> {
    %0 = tensor.extract_slice %arg0[1] [94] [1] : tensor<95xf32> to tensor<94xf32>
    %1 = tensor.extract_slice %arg0[0] [94] [1] : tensor<95xf32> to tensor<94xf32>
    %2 = tensor.empty() : tensor<94xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%0, %1 : tensor<94xf32>, tensor<94xf32>) outs(%2 : tensor<94xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %4 = arith.subf %arg1, %arg2 : f32
      linalg.yield %4 : f32
    } -> tensor<94xf32>
    return %3 : tensor<94xf32>
  }
  func.func @main() {
    // FIXME: Output does not converge. Need to update input values.
    %0 = arith.constant dense<1.0> : tensor<95xf32>
    %1 = arith.constant dense<0.1> : tensor<50000xf32>
    %2 = arith.constant dense<0.2> : tensor<50000xf32>
    %3 = func.call @callee(%0, %1, %2) : (tensor<95xf32>, tensor<50000xf32>, tensor<50000xf32>) -> tensor<94xf32>
    %unranked = tensor.cast %3 : tensor<94xf32> to tensor<*xf32>
    func.call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [94] strides = [1] data =
    return
  }
}
