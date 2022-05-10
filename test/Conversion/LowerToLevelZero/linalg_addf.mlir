// XFAIL: *
// RUN:

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
func.func @addt(%arg0: tensor<2x5xf32>, %arg1: tensor<2x5xf32>) -> tensor<2x5xf32> {
%0 = linalg.init_tensor [2, 5] : tensor<2x5xf32>
%1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<2x5xf32>, tensor<2x5x
f32>) outs(%0 : tensor<2x5xf32>) {
^bb0(%arg2: f32, %arg3: f32, %arg4: f32): // no predecessors
%2 = arith.addf %arg2, %arg3 : f32
linalg.yield %2 : f32
} -> tensor<2x5xf32>
return %1 : tensor<2x5xf32>
}


func.func @main() {
%0 = arith.constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]> : tensor<2x5xf32>
%1 = arith.constant dense<[[10.0, 9.0, 8.0, 7.0, 6.0], [5.0, 4.0, 3.0, 2.0, 1.0]]> : tensor<2x5xf32>
%2 = call @addt(%0, %1) : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
%unranked = tensor.cast %2 : tensor<2x5xf32> to tensor<*xf32>
call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
}
