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
module @binary_crossentropy {
func.func @main() {
    %0= arith.constant dense<[[[[-1.8268, -0.4094], [0.064, -0.5185], [-0.6317, -0.4328]], [[-0.596, -1.5718], [-0.1279, -0.0748], [0.6378, -0.4637]], [[1.6832, 0.1822], [0.081, -0.3461], [0.5564, -0.0673]]], [[[1.5682, 0.8883], [1.7371, 0.873], [0.9894, -0.028]], [[0.5348, 0.1942], [-0.5626, -0.102], [0.5176, 1.1897]], [[1.1937, 0.542], [-0.0949, 1.9825], [-0.1454, -0.2875]]], [[[0.5359, 1.3199], [2.1894, 0.0785], [0.6801, -0.059]], [[-0.561, 0.3118], [-1.5161, 0.3264], [-0.3428, 1.1087]], [[0.8555, 0.5152], [-0.6066, -0.2265], [0.9426, 0.1842]]]]>:tensor<3x3x3x2xf32>
    %1= arith.constant dense<[[[[0.0, 2.0], [4.0, 5.0], [4.0, 4.0]], [[1.0, 7.0], [7.0, 3.0], [7.0, 3.0]], [[7.0, 0.0], [7.0, 5.0], [7.0, 2.0]]], [[[5.0, 7.0], [3.0, 4.0], [7.0, 0.0]], [[0.0, 2.0], [2.0, 6.0], [0.0, 7.0]], [[4.0, 1.0], [3.0, 6.0], [2.0, 1.0]]], [[[0.0, 1.0], [1.0, 3.0], [6.0, 6.0]], [[2.0, 1.0], [4.0, 7.0], [1.0, 4.0]], [[0.0, 3.0], [3.0, 4.0], [0.0, 2.0]]]]>:tensor<3x3x3x2xf32>
    %2 = call @test(%0,%1) : (tensor<3x3x3x2xf32>,tensor<3x3x3x2xf32>) -> tensor<3x3x3x2xf32>
    %unranked = tensor.cast %2 : tensor<3x3x3x2xf32>to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    return
}
func.func private @printMemrefF32(tensor<*xf32>)
func.func @test(%arg0: tensor<3x3x3x2xf32>, %arg1: tensor<3x3x3x2xf32>)->tensor<3x3x3x2xf32>{
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<3x3x3x2xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<3x3x3x2xf32>) outs(%0 : tensor<3x3x3x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %24 = arith.negf %arg2 : f32
      linalg.yield %24 : f32
    } -> tensor<3x3x3x2xf32>
    %2 = tensor.empty() : tensor<3x3x3x2xi1>
    %3 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1, %cst : tensor<3x3x3x2xf32>, f32) outs(%2 : tensor<3x3x3x2xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %24 = arith.cmpf ogt, %arg2, %arg3 : f32
      linalg.yield %24 : i1
    } -> tensor<3x3x3x2xi1>
    %4 = tensor.empty() : tensor<3x3x3x2xf32>
    %5 = linalg.generic {indexing_maps = [#map0, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %arg1, %cst : tensor<3x3x3x2xi1>, tensor<3x3x3x2xf32>, f32) outs(%4 : tensor<3x3x3x2xf32>) {
    ^bb0(%arg2: i1, %arg3: f32, %arg4: f32, %arg5: f32):
      %24 = arith.select %arg2, %arg3, %arg4 : f32
      linalg.yield %24 : f32
    } -> tensor<3x3x3x2xf32>
    %6 = tensor.empty() : tensor<3x3x3x2xi1>
    %7 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5, %cst_0 : tensor<3x3x3x2xf32>, f32) outs(%6 : tensor<3x3x3x2xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %24 = arith.cmpf olt, %arg2, %arg3 : f32
      linalg.yield %24 : i1
    } -> tensor<3x3x3x2xi1>
    %8 = tensor.empty() : tensor<3x3x3x2xf32>
    %9 = linalg.generic {indexing_maps = [#map0, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7, %5, %cst_0 : tensor<3x3x3x2xi1>, tensor<3x3x3x2xf32>, f32) outs(%8 : tensor<3x3x3x2xf32>) {
    ^bb0(%arg2: i1, %arg3: f32, %arg4: f32, %arg5: f32):
      %24 = arith.select %arg2, %arg3, %arg4 : f32
      linalg.yield %24 : f32
    } -> tensor<3x3x3x2xf32>
    %10 = tensor.empty() : tensor<3x3x3x2xf32>
    %11 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9 : tensor<3x3x3x2xf32>) outs(%10 : tensor<3x3x3x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %24 = math.log %arg2 : f32
      linalg.yield %24 : f32
    } -> tensor<3x3x3x2xf32>
    %12 = tensor.empty() : tensor<3x3x3x2xf32>
    %13 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %11 : tensor<3x3x3x2xf32>, tensor<3x3x3x2xf32>) outs(%12 : tensor<3x3x3x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %24 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %24 : f32
    } -> tensor<3x3x3x2xf32>
    %14 = tensor.empty() : tensor<3x3x3x2xf32>
    %15 = linalg.generic {indexing_maps = [#map1, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_0, %arg0 : f32, tensor<3x3x3x2xf32>) outs(%14 : tensor<3x3x3x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %24 = arith.subf %arg2, %arg3 : f32
      linalg.yield %24 : f32
    } -> tensor<3x3x3x2xf32>
    %16 = tensor.empty() : tensor<3x3x3x2xf32>
    %17 = linalg.generic {indexing_maps = [#map1, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_0, %9 : f32, tensor<3x3x3x2xf32>) outs(%16 : tensor<3x3x3x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %24 = arith.subf %arg2, %arg3 : f32
      linalg.yield %24 : f32
    } -> tensor<3x3x3x2xf32>
    %18 = tensor.empty() : tensor<3x3x3x2xf32>
    %19 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17 : tensor<3x3x3x2xf32>) outs(%18 : tensor<3x3x3x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %24 = math.log %arg2 : f32
      linalg.yield %24 : f32
    } -> tensor<3x3x3x2xf32>
    %20 = tensor.empty() : tensor<3x3x3x2xf32>
    %21 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15, %19 : tensor<3x3x3x2xf32>, tensor<3x3x3x2xf32>) outs(%20 : tensor<3x3x3x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %24 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %24 : f32
    } -> tensor<3x3x3x2xf32>
    %22 = tensor.empty() : tensor<3x3x3x2xf32>
    %23 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13, %21 : tensor<3x3x3x2xf32>, tensor<3x3x3x2xf32>) outs(%22 : tensor<3x3x3x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %24 = arith.subf %arg2, %arg3 : f32
      linalg.yield %24 : f32
    } -> tensor<3x3x3x2xf32>
    return %23 : tensor<3x3x3x2xf32>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [] strides = {{.*}} data =
// CHECK:   -0.1468
