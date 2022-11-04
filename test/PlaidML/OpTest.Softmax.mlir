#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
module @softmax {
func.func @test() {
    %0= arith.constant dense<[[0.9, 0.1, 0.4, 0.9, 0.1, 0.4, 0.9, 0.1, 0.7, 0.4, 0.9, 0.1, 0.4, 0.9, 0.1, 0.4, 0.9, 0.1, 0.7, 0.4],
                              [1.9, 0.7, 1.2, 1.9, 0.7, 1.2, 1.9, 0.6, 1.5, 1.6, 1.9, 0.7, 1.2, 1.9, 0.7, 1.2, 1.9, 0.7, 1.5, 1.6],
                              [0.9, 0.1, 0.4, 0.9, 0.1, 0.4, 0.9, 0.1, 0.7, 0.4, 1.9, 0.7, 1.2, 1.9, 0.6, 1.2, 1.9, 0.6, 1.5, 1.6],
                              [1.9, 0.6, 1.2, 1.9, 0.7, 1.2, 1.9, 0.7, 1.5, 1.6, 1.9, 0.6, 1.2, 1.9, 0.6, 1.2, 1.9, 0.7, 1.5, 1.6],
                              [0.9, 0.1, 0.4, 0.9, 0.1, 0.4, 0.9, 0.1, 0.7, 0.4, 1.9, 0.6, 1.2, 1.9, 0.7, 1.2, 1.9, 0.6, 1.5, 1.6],
                              [1.9, 0.7, 1.2, 1.9, 0.7, 1.2, 1.9, 0.7, 1.5, 1.6, 1.9, 0.7, 1.2, 1.9, 0.7, 1.2, 1.9, 0.7, 1.5, 1.6],
                              [0.9, 0.1, 0.4, 0.9, 0.1, 0.4, 0.9, 0.1, 0.7, 0.4, 1.9, 0.6, 1.2, 1.9, 0.6, 1.2, 1.9, 0.6, 1.5, 1.6],
                              [1.9, 0.6, 1.2, 1.9, 0.7, 1.2, 1.9, 0.7, 1.5, 1.6, 1.9, 0.7, 1.2, 1.9, 0.7, 1.2, 1.9, 0.7, 1.5, 1.6],
                              [0.9, 0.1, 0.4, 0.9, 0.1, 0.4, 0.9, 0.1, 0.7, 0.4, 1.9, 0.6, 1.2, 1.9, 0.6, 1.2, 1.9, 0.6, 1.5, 1.6],
                              [1.9, 0.7, 1.2, 1.9, 0.7, 1.2, 1.9, 0.6, 1.5, 1.6, 1.9, 0.7, 1.2, 1.9, 0.7, 1.2, 1.9, 0.7, 1.5, 1.6]]>:tensor<10x20xf32>
    %1 = call @main(%0) : (tensor<10x20xf32>) -> tensor<10x20xf32>
    %unranked = tensor.cast %1 : tensor<10x20xf32>to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    return
}
func.func private @printMemrefF32(tensor<*xf32>)
func.func @main(%arg0: tensor<10x20xf32>)->tensor<10x20xf32>{
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<10x1xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<10x1xf32>) -> tensor<10x1xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<10x20xf32>) outs(%1 : tensor<10x1xf32>) attrs =  {iterator_ranges = [10, 20], name = "softmax"} {
    ^bb0(%arg1: f32, %arg2: f32):
      %12 = arith.cmpf ogt, %arg2, %arg1 : f32
      %13 = arith.select %12, %arg2, %arg1 : f32
      linalg.yield %13 : f32
    } -> tensor<10x1xf32>
    %3 = tensor.empty() : tensor<10x20xf32>
    %4 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : tensor<10x20xf32>, tensor<10x1xf32>) outs(%3 : tensor<10x20xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %12 = arith.subf %arg1, %arg2 : f32
      linalg.yield %12 : f32
    } -> tensor<10x20xf32>
    %5 = tensor.empty() : tensor<10x20xf32>
    %6 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<10x20xf32>) outs(%5 : tensor<10x20xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %12 = math.exp %arg1 : f32
      linalg.yield %12 : f32
    } -> tensor<10x20xf32>
    %7 = tensor.empty() : tensor<10x1xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<10x1xf32>) -> tensor<10x1xf32>
    %9 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%6 : tensor<10x20xf32>) outs(%8 : tensor<10x1xf32>) attrs =  {iterator_ranges = [10, 20], name = "softmax"} {
    ^bb0(%arg1: f32, %arg2: f32):
      %12 = arith.addf %arg2, %arg1 : f32
      linalg.yield %12 : f32
    } -> tensor<10x1xf32>
    %10 = tensor.empty() : tensor<10x20xf32>
    %11 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%6, %9 : tensor<10x20xf32>, tensor<10x1xf32>) outs(%10 : tensor<10x20xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %12 = arith.divf %arg1, %arg2 : f32
      linalg.yield %12 : f32
    } -> tensor<10x20xf32>
    return %11 : tensor<10x20xf32>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [2, 3] strides = {{.*}} data =
// CHECK:   0.0716
// CHECK:   0.0322
// CHECK:   0.0434
// CHECK:   0.0716
// CHECK:   0.0322
// CHECK:   0.0434
// CHECK:   0.0716
// CHECK:   0.0322
// CHECK:   0.0586
// CHECK:   0.0434
// CHECK:   0.0716
// CHECK:   0.0322
// CHECK:   0.0434
// CHECK:   0.0716
// CHECK:   0.0322
// CHECK:   0.0434
// CHECK:   0.0716
// CHECK:   0.0322
// CHECK:   0.0586
// CHECK:   0.0434
// CHECK:   0.0794
// CHECK:   0.0239
// CHECK:   0.0395
// CHECK:   0.0794
// CHECK:   0.0239
// CHECK:   0.0395
// CHECK:   0.0794
// CHECK:   0.0217
// CHECK:   0.0533
// CHECK:   0.0589
// CHECK:   0.0794
// CHECK:   0.0239
// CHECK:   0.0395
// CHECK:   0.0794
// CHECK:   0.0239
// CHECK:   0.0395
// CHECK:   0.0794
// CHECK:   0.0239
// CHECK:   0.0533
// CHECK:   0.0589
// CHECK:   0.0417
// CHECK:   0.0187
// CHECK:   0.0253
// CHECK:   0.0417
// CHECK:   0.0187
// CHECK:   0.0253
// CHECK:   0.0417
// CHECK:   0.0187
// CHECK:   0.0341
// CHECK:   0.0253
// CHECK:   0.1134
// CHECK:   0.0341
// CHECK:   0.0563
// CHECK:   0.1134
// CHECK:   0.0309
// CHECK:   0.0563
// CHECK:   0.1134
// CHECK:   0.0309
// CHECK:   0.076
// CHECK:   0.084
// CHECK:   0.0798
// CHECK:   0.0218
// CHECK:   0.0396
// CHECK:   0.0798
// CHECK:   0.024
// CHECK:   0.0396
// CHECK:   0.0798
// CHECK:   0.024
// CHECK:   0.0535
// CHECK:   0.0591
// CHECK:   0.0798
// CHECK:   0.0218
// CHECK:   0.0396
// CHECK:   0.0798
// CHECK:   0.0218
// CHECK:   0.0396
// CHECK:   0.0798
// CHECK:   0.0535
// CHECK:   0.0591
// CHECK:   0.0417
// CHECK:   0.0187
// CHECK:   0.0253
// CHECK:   0.0417
// CHECK:   0.0187
// CHECK:   0.0253
// CHECK:   0.0417
// CHECK:   0.0187
// CHECK:   0.0341
// CHECK:   0.0253
// CHECK:   0.1134
// CHECK:   0.0309
// CHECK:   0.0563
// CHECK:   0.1134
// CHECK:   0.0341
// CHECK:   0.0563
// CHECK:   0.1134
// CHECK:   0.0309
// CHECK:   0.076
// CHECK:   0.084
// CHECK:   0.0793
// CHECK:   0.0239
// CHECK:   0.0394
// CHECK:   0.0793
// CHECK:   0.0239
// CHECK:   0.0394
// CHECK:   0.0793
// CHECK:   0.0239
// CHECK:   0.0531
// CHECK:   0.0587
// CHECK:   0.0793
// CHECK:   0.0239
// CHECK:   0.0394
// CHECK:   0.0793
// CHECK:   0.0239
// CHECK:   0.0394
// CHECK:   0.0793
// CHECK:   0.0239
// CHECK:   0.0531
// CHECK:   0.0587
// CHECK:   0.0418
// CHECK:   0.0188
// CHECK:   0.0254
// CHECK:   0.0418
// CHECK:   0.0188
// CHECK:   0.0254
// CHECK:   0.0418
// CHECK:   0.0188
// CHECK:   0.0343
// CHECK:   0.0254
// CHECK:   0.1137
// CHECK:   0.031
// CHECK:   0.0565
// CHECK:   0.1137
// CHECK:   0.031
// CHECK:   0.0565
// CHECK:   0.1137
// CHECK:   0.031
// CHECK:   0.0762
// CHECK:   0.0843
// CHECK:   0.0794
// CHECK:   0.0217
// CHECK:   0.0395
// CHECK:   0.0794
// CHECK:   0.0239
// CHECK:   0.0395
// CHECK:   0.0794
// CHECK:   0.0239
// CHECK:   0.0533
// CHECK:   0.0589
// CHECK:   0.0794
// CHECK:   0.0239
// CHECK:   0.0395
// CHECK:   0.0794
// CHECK:   0.0239
// CHECK:   0.0395
// CHECK:   0.0794
// CHECK:   0.0239
// CHECK:   0.0533
// CHECK:   0.0589
// CHECK:   0.0418
// CHECK:   0.0188
// CHECK:   0.0254
// CHECK:   0.0418
// CHECK:   0.0188
// CHECK:   0.0254
// CHECK:   0.0418
// CHECK:   0.0188
// CHECK:   0.0343
// CHECK:   0.0254
// CHECK:   0.1137
// CHECK:   0.031
// CHECK:   0.0565
// CHECK:   0.1137
// CHECK:   0.031
// CHECK:   0.0565
// CHECK:   0.1137
// CHECK:   0.031
// CHECK:   0.0762
// CHECK:   0.0843
// CHECK:   0.0794
// CHECK:   0.0239
// CHECK:   0.0395
// CHECK:   0.0794
// CHECK:   0.0239
// CHECK:   0.0395
// CHECK:   0.0794
// CHECK:   0.0217
// CHECK:   0.0533
// CHECK:   0.0589
// CHECK:   0.0794
// CHECK:   0.0239
// CHECK:   0.0395
// CHECK:   0.0794
// CHECK:   0.0239
// CHECK:   0.0395
// CHECK:   0.0794
// CHECK:   0.0239
// CHECK:   0.0533
// CHECK:   0.0589
