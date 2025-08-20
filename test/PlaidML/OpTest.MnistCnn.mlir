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
#map1 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> ()>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d3)>
#map7 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
#map8 = affine_map<(d0, d1) -> (d1)>
#map9 = affine_map<(d0, d1) -> (d0, d1)>
#map10 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map11 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map12 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map13 = affine_map<(d0, d1) -> ()>
#map14 = affine_map<(d0, d1) -> (d0, 0)>
#map15 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>
module @mnist_cnn {
  func.func @main() {
    %0 = arith.constant dense<1.0> : tensor<1x224x224x1xf32>
    %1 = arith.constant dense<2.0> : tensor<3x3x1x32xf32>
    %2 = arith.constant dense<0.5> : tensor<32xf32>
    %3 = arith.constant dense<0.8> : tensor<3x3x32x64xf32>
    %4 = arith.constant dense<0.5> : tensor<64xf32>
    %5 = arith.constant dense<1.5> : tensor<802816x128xf32>
    %6 = arith.constant dense<0.3> : tensor<128xf32>
    %7 = arith.constant dense<0.2> : tensor<128x100xf32>
    %8 = arith.constant dense<4.0> : tensor<100xf32>
    %9 = call @test(%0,%1,%2,%3,%4,%5,%6,%7,%8) : (tensor<1x224x224x1xf32>, tensor<3x3x1x32xf32>, tensor<32xf32>, tensor<3x3x32x64xf32>, tensor<64xf32>, tensor<802816x128xf32>, tensor<128xf32>, tensor<128x100xf32>, tensor<100xf32>) -> tensor<1x100xf32>
    %unranked = tensor.cast %9 : tensor<1x100xf32>to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @test(%arg0: tensor<1x224x224x1xf32>, %arg1: tensor<3x3x1x32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<3x3x32x64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<802816x128xf32>, %arg6: tensor<128xf32>, %arg7: tensor<128x100xf32>, %arg8: tensor<100xf32>) -> tensor<1x100xf32> {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x224x224x1xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x224x224x1xf32>) outs(%0 : tensor<1x224x224x1xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> tensor<1x224x224x1xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %2 = tensor.pad %1 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg9: index, %arg10: index, %arg11: index, %arg12: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x224x224x1xf32> to tensor<1x226x226x1xf32>
    %3 = tensor.empty(): tensor<1x224x224x32xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<32xf32>) outs(%3 : tensor<1x224x224x32xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> tensor<1x224x224x32xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%2, %arg1 : tensor<1x226x226x1xf32>, tensor<3x3x1x32xf32>) outs(%4 : tensor<1x224x224x32xf32>) attrs =  {iterator_ranges = [1, 224, 224, 32, 3, 3, 1]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %44 = arith.mulf %arg9, %arg10 : f32
      %45 = arith.addf %arg11, %44 : f32
      linalg.yield %45 : f32
    } -> tensor<1x224x224x32xf32>
    %6 = tensor.empty() : tensor<1x224x224x32xi1>
    %7 = linalg.generic {indexing_maps = [#map0, #map5, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5, %cst_0 : tensor<1x224x224x32xf32>, f32) outs(%6 : tensor<1x224x224x32xi1>) {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: i1):
      %44 = arith.cmpf olt, %arg9, %arg10 : f32
      linalg.yield %44 : i1
    } -> tensor<1x224x224x32xi1>
    %8 = tensor.empty() : tensor<1x224x224x32xf32>
    %9 = linalg.generic {indexing_maps = [#map0, #map5, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7, %cst_0, %5 : tensor<1x224x224x32xi1>, f32, tensor<1x224x224x32xf32>) outs(%8 : tensor<1x224x224x32xf32>) {
    ^bb0(%arg9: i1, %arg10: f32, %arg11: f32, %arg12: f32):
      %44 = arith.select %arg9, %arg10, %arg11 : f32
      linalg.yield %44 : f32
    } -> tensor<1x224x224x32xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %10 = tensor.pad %9 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg9: index, %arg10: index, %arg11: index, %arg12: index):
      tensor.yield %cst_2 : f32
    } : tensor<1x224x224x32xf32> to tensor<1x226x226x32xf32>
    %11 = tensor.empty() : tensor<1x224x224x64xf32>
    %12 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg4 : tensor<64xf32>) outs(%11 : tensor<1x224x224x64xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> tensor<1x224x224x64xf32>
    %13 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%10, %arg3 : tensor<1x226x226x32xf32>, tensor<3x3x32x64xf32>) outs(%12 : tensor<1x224x224x64xf32>) attrs =  {iterator_ranges = [1, 224, 224, 64, 3, 3, 32]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %44 = arith.mulf %arg9, %arg10 : f32
      %45 = arith.addf %arg11, %44 : f32
      linalg.yield %45 : f32
    } -> tensor<1x224x224x64xf32>
    %14 = tensor.empty() : tensor<1x224x224x64xi1>
    %15 = linalg.generic {indexing_maps = [#map0, #map5, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13, %cst_0 : tensor<1x224x224x64xf32>, f32) outs(%14 : tensor<1x224x224x64xi1>) {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: i1):
      %44 = arith.cmpf olt, %arg9, %arg10 : f32
      linalg.yield %44 : i1
    } -> tensor<1x224x224x64xi1>
    %16 = tensor.empty() : tensor<1x224x224x64xf32>
    %17 = linalg.generic {indexing_maps = [#map0, #map5, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15, %cst_0, %13 : tensor<1x224x224x64xi1>, f32, tensor<1x224x224x64xf32>) outs(%16 : tensor<1x224x224x64xf32>) {
    ^bb0(%arg9: i1, %arg10: f32, %arg11: f32, %arg12: f32):
      %44 = arith.select %arg9, %arg10, %arg11 : f32
      linalg.yield %44 : f32
    } -> tensor<1x224x224x64xf32>
    %18 = tensor.empty() : tensor<1x112x112x64xf32>
    %fake = tensor.empty(): tensor<2x2xf32>
    %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
    %20 = linalg.generic {indexing_maps = [#map6, #map15, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%17, %fake : tensor<1x224x224x64xf32>, tensor<2x2xf32>) outs(%19 : tensor<1x112x112x64xf32>) attrs =  {iterator_ranges = [1, 112, 112, 64, 2, 2]} {
    ^bb0(%arg9: f32, %arg_f: f32, %arg10: f32):
      %44 = arith.cmpf ogt, %arg10, %arg9 : f32
      %45 = arith.select %44, %arg10, %arg9 : f32
      linalg.yield %45 : f32
    } -> tensor<1x112x112x64xf32>
    %21 = tensor.collapse_shape %20 [[0], [1, 2, 3]] : tensor<1x112x112x64xf32> into tensor<1x802816xf32>
    %22 = tensor.empty() : tensor<1x128xf32>
    %23 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel"]} ins(%arg6 : tensor<128xf32>) outs(%22 : tensor<1x128xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> tensor<1x128xf32>
    %24 = linalg.generic {indexing_maps = [#map10, #map11, #map12], iterator_types = ["parallel", "parallel", "reduction"]} ins(%21, %arg5 : tensor<1x802816xf32>, tensor<802816x128xf32>) outs(%23 : tensor<1x128xf32>) attrs =  {iterator_ranges = [1, 128, 802816]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %44 = arith.mulf %arg9, %arg10 : f32
      %45 = arith.addf %arg11, %44 : f32
      linalg.yield %45 : f32
    } -> tensor<1x128xf32>
    %25 = tensor.empty() : tensor<1x128xi1>
    %26 = linalg.generic {indexing_maps = [#map9, #map13, #map9], iterator_types = ["parallel", "parallel"]} ins(%24, %cst_0 : tensor<1x128xf32>, f32) outs(%25 : tensor<1x128xi1>) {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: i1):
      %44 = arith.cmpf olt, %arg9, %arg10 : f32
      linalg.yield %44 : i1
    } -> tensor<1x128xi1>
    %27 = tensor.empty() : tensor<1x128xf32>
    %28 = linalg.generic {indexing_maps = [#map9, #map13, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%26, %cst_0, %24 : tensor<1x128xi1>, f32, tensor<1x128xf32>) outs(%27 : tensor<1x128xf32>) {
    ^bb0(%arg9: i1, %arg10: f32, %arg11: f32, %arg12: f32):
      %44 = arith.select %arg9, %arg10, %arg11 : f32
      linalg.yield %44 : f32
    } -> tensor<1x128xf32>
    %29 = tensor.empty() : tensor<1x100xf32>
    %30 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel"]} ins(%arg8 : tensor<100xf32>) outs(%29 : tensor<1x100xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> tensor<1x100xf32>
    %31 = linalg.generic {indexing_maps = [#map10, #map11, #map12], iterator_types = ["parallel", "parallel", "reduction"]} ins(%28, %arg7 : tensor<1x128xf32>, tensor<128x100xf32>) outs(%30 : tensor<1x100xf32>) attrs =  {iterator_ranges = [1, 100, 128]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %44 = arith.mulf %arg9, %arg10 : f32
      %45 = arith.addf %arg11, %44 : f32
      linalg.yield %45 : f32
    } -> tensor<1x100xf32>
    %32 = tensor.empty() : tensor<1x1xf32>
    %33 = linalg.fill ins(%cst : f32) outs(%32 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %34 = linalg.generic {indexing_maps = [#map9, #map14], iterator_types = ["parallel", "reduction"]} ins(%31 : tensor<1x100xf32>) outs(%33 : tensor<1x1xf32>) attrs =  {iterator_ranges = [1, 100]} {
    ^bb0(%arg9: f32, %arg10: f32):
      %44 = arith.cmpf ogt, %arg10, %arg9 : f32
      %45 = arith.select %44, %arg10, %arg9 : f32
      linalg.yield %45 : f32
    } -> tensor<1x1xf32>
    %35 = tensor.empty() : tensor<1x100xf32>
    %36 = linalg.generic {indexing_maps = [#map9, #map14, #map9], iterator_types = ["parallel", "parallel"]} ins(%31, %34 : tensor<1x100xf32>, tensor<1x1xf32>) outs(%35 : tensor<1x100xf32>) {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %44 = arith.subf %arg9, %arg10 : f32
      linalg.yield %44 : f32
    } -> tensor<1x100xf32>
    %37 = tensor.empty() : tensor<1x100xf32>
    %38 = linalg.generic {indexing_maps = [#map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%36 : tensor<1x100xf32>) outs(%37 : tensor<1x100xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      %44 = math.exp %arg9 : f32
      linalg.yield %44 : f32
    } -> tensor<1x100xf32>
    %39 = tensor.empty() : tensor<1x1xf32>
    %40 = linalg.fill ins(%cst_0 : f32) outs(%39 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %41 = linalg.generic {indexing_maps = [#map9, #map14], iterator_types = ["parallel", "reduction"]} ins(%38 : tensor<1x100xf32>) outs(%40 : tensor<1x1xf32>) attrs =  {iterator_ranges = [1, 100]} {
    ^bb0(%arg9: f32, %arg10: f32):
      %44 = arith.addf %arg10, %arg9 : f32
      linalg.yield %44 : f32
    } -> tensor<1x1xf32>
    %42 = tensor.empty() : tensor<1x100xf32>
    %43 = linalg.generic {indexing_maps = [#map9, #map14, #map9], iterator_types = ["parallel", "parallel"]} ins(%38, %41 : tensor<1x100xf32>, tensor<1x1xf32>) outs(%42 : tensor<1x100xf32>) {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %44 = arith.divf %arg9, %arg10 : f32
      linalg.yield %44 : f32
    } -> tensor<1x100xf32>
    return %43 : tensor<1x100xf32>
  }
}

// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [1, 100] strides = {{.*}} data =
// CHECK: 0.01
// CHECK: 0.01
// CHECK: 0.01
