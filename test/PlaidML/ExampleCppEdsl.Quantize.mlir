#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module @quantize {
func.func @main(%arg0: tensor<3xf32>)->tensor<3xi8>{
    %cst = arith.constant 2.560000e+02 : f32
    %0 = tensor.empty() : tensor<3xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel"]} ins(%arg0, %cst : tensor<3xf32>, f32) outs(%0 : tensor<3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %4 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %4 : f32
    } -> tensor<3xf32>
    %2 = tensor.empty() : tensor<3xi8>
    %3 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%1 : tensor<3xf32>) outs(%2 : tensor<3xi8>) {
    ^bb0(%arg1: f32, %arg2: i8):
      %4 = arith.fptosi %arg1 : f32 to i8
      linalg.yield %4 : i8
    } -> tensor<3xi8>
    return %3 : tensor<3xi8>
  }

func.func @test() {
    %0= arith.constant dense<[0.1, 0.4, 0.3]>:tensor<3xf32>
    %1 = call @main(%0) : (tensor<3xf32>) -> tensor<3xi8>
    %unranked = tensor.cast %1 : tensor<3xi8>to tensor<*xi8>
    call @printMemrefI32(%unranked) : (tensor<*xi8>) -> ()
    return
  }
func.func private @printMemrefI32(tensor<*xi8>)
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [3] strides = {{.*}} data =
// CHECK:   25
// CHECK:   102
// CHECK:   76
