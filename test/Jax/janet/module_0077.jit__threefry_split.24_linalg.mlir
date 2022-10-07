#map0 = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
module @jit__threefry_split.24 {
  func @main(%arg0: tensor<2xui32>) -> tensor<2x2xui32> {
    %0 = builtin.unrealized_conversion_cast %arg0 : tensor<2xui32> to tensor<2xi32>
    %1 = tensor.extract_slice %0[0] [1] [1] : tensor<2xi32> to tensor<1xi32>
    %2 = tensor.collapse_shape %1 [] : tensor<1xi32> into tensor<i32>
    %3 = linalg.init_tensor [2] : tensor<2xi32>
    %4 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel"]} ins(%2 : tensor<i32>) outs(%3 : tensor<2xi32>) {
    ^bb0(%arg1: i32, %arg2: i32):
      linalg.yield %arg1 : i32
    } -> tensor<2xi32>
    %5 = builtin.unrealized_conversion_cast %4 : tensor<2xi32> to tensor<2xui32>
    %6 = tensor.extract_slice %0[1] [1] [1] : tensor<2xi32> to tensor<1xi32>
    %7 = tensor.collapse_shape %6 [] : tensor<1xi32> into tensor<i32>
    %8 = linalg.init_tensor [2] : tensor<2xi32>
    %9 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel"]} ins(%7 : tensor<i32>) outs(%8 : tensor<2xi32>) {
    ^bb0(%arg1: i32, %arg2: i32):
      linalg.yield %arg1 : i32
    } -> tensor<2xi32>
    %10 = builtin.unrealized_conversion_cast %9 : tensor<2xi32> to tensor<2xui32>
    %11 = linalg.init_tensor [4] : tensor<4xi32>
    %12 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%11 : tensor<4xi32>) {
    ^bb0(%arg1: i32):
      %26 = linalg.index 0 : index
      %27 = arith.index_cast %26 : index to i32
      linalg.yield %27 : i32
    } -> tensor<4xi32>
    %13 = tensor.extract_slice %12[0] [2] [1] : tensor<4xi32> to tensor<2xi32>
    %14 = builtin.unrealized_conversion_cast %13 : tensor<2xi32> to tensor<2xui32>
    %15 = tensor.extract_slice %12[2] [2] [1] : tensor<4xi32> to tensor<2xi32>
    %16 = builtin.unrealized_conversion_cast %15 : tensor<2xi32> to tensor<2xui32>
    %17 = "mhlo.custom_call"(%5, %10, %14, %16) {api_version = 2 : i32, backend_config = "\02\00\00\00\00\00\00\00", call_target_name = "cuda_threefry2x32", has_side_effect = false, operand_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>], xla_shape = "(u32[2]{0}, u32[2]{0})"} : (tensor<2xui32>, tensor<2xui32>, tensor<2xui32>, tensor<2xui32>) -> tuple<tensor<2xui32>, tensor<2xui32>>
    %18 = "mhlo.get_tuple_element"(%17) {index = 0 : i32} : (tuple<tensor<2xui32>, tensor<2xui32>>) -> tensor<2xui32>
    %19 = builtin.unrealized_conversion_cast %18 : tensor<2xui32> to tensor<2xi32>
    %20 = "mhlo.get_tuple_element"(%17) {index = 1 : i32} : (tuple<tensor<2xui32>, tensor<2xui32>>) -> tensor<2xui32>
    %21 = builtin.unrealized_conversion_cast %20 : tensor<2xui32> to tensor<2xi32>
    %c0 = arith.constant 0 : index
    %22 = linalg.init_tensor [4] : tensor<4xi32>
    %23 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%22 : tensor<4xi32>) {
    ^bb0(%arg1: i32):
      %26 = linalg.index 0 : index
      %27 = linalg.index 0 : index
      %c0_0 = arith.constant 0 : index
      %28 = tensor.dim %19, %c0_0 : tensor<2xi32>
      %29 = arith.addi %c0, %28 : index
      %30 = arith.cmpi ult, %27, %29 : index
      %31 = scf.if %30 -> (i32) {
        %32 = arith.subi %27, %c0 : index
        %33 = tensor.extract %19[%32] : tensor<2xi32>
        scf.yield %33 : i32
      } else {
        %32 = arith.subi %27, %29 : index
        %33 = tensor.extract %21[%32] : tensor<2xi32>
        scf.yield %33 : i32
      }
      linalg.yield %31 : i32
    } -> tensor<4xi32>
    %24 = tensor.expand_shape %23 [[0, 1]] : tensor<4xi32> into tensor<2x2xi32>
    %25 = builtin.unrealized_conversion_cast %24 : tensor<2x2xi32> to tensor<2x2xui32>
    return %25 : tensor<2x2xui32>
  }
}
