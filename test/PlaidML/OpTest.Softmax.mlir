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
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
module @softmax {
func.func @main() {
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
    %1 = call @test(%0) : (tensor<10x20xf32>) -> tensor<10x20xf32>
    %unranked = tensor.cast %1 : tensor<10x20xf32>to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    return
}
func.func private @printMemrefF32(tensor<*xf32>)
func.func @test(%arg0: tensor<10x20xf32>)->tensor<10x20xf32>{
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<10x1xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<10x1xf32>) -> tensor<10x1xf32>
    %2 = linalg.generic {
            indexing_maps = [#map0, #map1],
            iterator_types = ["parallel", "reduction"]
          }
          ins(%arg0 : tensor<10x20xf32>)
          outs(%1 : tensor<10x1xf32>)
          attrs =  {iterator_ranges = [10, 20], name = "softmax"} {
            ^bb0(%arg1: f32, %arg2: f32):
              %12 = arith.cmpf ogt, %arg2, %arg1 : f32
              %13 = arith.select %12, %arg2, %arg1 : f32
              linalg.yield %13 : f32
         } -> tensor<10x1xf32>
    %3 = tensor.empty() : tensor<10x20xf32>
    %4 = linalg.generic {
            indexing_maps = [#map0, #map1, #map0],
            iterator_types = ["parallel", "parallel"]
          }
          ins(%arg0, %2 : tensor<10x20xf32>, tensor<10x1xf32>)
          outs(%3 : tensor<10x20xf32>) {
            ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
              %12 = arith.subf %arg1, %arg2 : f32
              linalg.yield %12 : f32
          } -> tensor<10x20xf32>
    %5 = tensor.empty() : tensor<10x20xf32>
    %6 = linalg.generic {
            indexing_maps = [#map0, #map0],
            iterator_types = ["parallel", "parallel"]
          }
          ins(%4 : tensor<10x20xf32>) outs(%5 : tensor<10x20xf32>) {
            ^bb0(%arg1: f32, %arg2: f32):
              %12 = math.exp %arg1 : f32
              linalg.yield %12 : f32
          } -> tensor<10x20xf32>
    %7 = tensor.empty() : tensor<10x1xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<10x1xf32>) -> tensor<10x1xf32>
    %9 = linalg.generic {
            indexing_maps = [#map0, #map1],
            iterator_types = ["parallel", "reduction"]
          }
          ins(%6 : tensor<10x20xf32>)
          outs(%8 : tensor<10x1xf32>)
          attrs = {iterator_ranges = [10, 20], name = "softmax"} {
            ^bb0(%arg1: f32, %arg2: f32):
              %12 = arith.addf %arg2, %arg1 : f32
              linalg.yield %12 : f32
          } -> tensor<10x1xf32>
    %10 = tensor.empty() : tensor<10x20xf32>
    %11 = linalg.generic {
            indexing_maps = [#map0, #map1, #map0],
            iterator_types = ["parallel", "parallel"]
          }
          ins(%6, %9 : tensor<10x20xf32>, tensor<10x1xf32>)
          outs(%10 : tensor<10x20xf32>) {
            ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
              %12 = arith.divf %arg1, %arg2 : f32
              linalg.yield %12 : f32
          } -> tensor<10x20xf32>
    return %11 : tensor<10x20xf32>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [10, 20] strides = {{.*}} data =
// CHECK-NEXT:  [0.0715685, 0.0321578, 0.0434085, 0.0715685, 0.0321578, 0.0434085, 0.0715685, 0.0321578, 0.0585954, 0.0434085, 0.0715685, 0.0321578, 0.0434085, 0.0715685, 0.0321578, 0.0434085, 0.0715685, 0.0321578, 0.0585954, 0.0434085]
// CHECK-NEXT:  [0.0794463, 0.0239288, 0.0394519, 0.0794463, 0.0239288, 0.0394519, 0.0794463, 0.0216516, 0.0532544, 0.0588553, 0.0794463, 0.0239288, 0.0394519, 0.0794463, 0.0239288, 0.0394519, 0.0794463, 0.0239288, 0.0532544, 0.0588553]
// CHECK-NEXT:  [0.0417064, 0.0187399, 0.0252962, 0.0417064, 0.0187399, 0.0252962, 0.0417064, 0.0187399, 0.0341463, 0.0252962, 0.11337,   0.0341463, 0.0562978, 0.11337,   0.0308969, 0.0562978, 0.11337,   0.0308969, 0.0759941, 0.0839865]
// CHECK-NEXT:  [0.0798098, 0.0217507, 0.0396324, 0.0798098, 0.0240382, 0.0396324, 0.0798098, 0.0240382, 0.0534981, 0.0591245, 0.0798098, 0.0217507, 0.0396324, 0.0798098, 0.0217507, 0.0396324, 0.0798098, 0.0240382, 0.0534981, 0.0591245]
// CHECK-NEXT:  [0.0417064, 0.0187399, 0.0252962, 0.0417064, 0.0187399, 0.0252962, 0.0417064, 0.0187399, 0.0341463, 0.0252962, 0.11337,   0.0308969, 0.0562978, 0.11337,   0.0341463, 0.0562978, 0.11337,   0.0308969, 0.0759941, 0.0839865]
// CHECK-NEXT:  [0.0792658, 0.0238744, 0.0393622, 0.0792658, 0.0238744, 0.0393622, 0.0792658, 0.0238744, 0.0531335, 0.0587215, 0.0792658, 0.0238744, 0.0393622, 0.0792658, 0.0238744, 0.0393622, 0.0792658, 0.0238744, 0.0531335, 0.0587215]
// CHECK-NEXT:  [0.0418424, 0.018801,  0.0253787, 0.0418424, 0.018801,  0.0253787, 0.0418424, 0.018801,  0.0342577, 0.0253787, 0.113739,  0.0309976, 0.0564813, 0.113739,  0.0309976, 0.0564813, 0.113739,  0.0309976, 0.0762418, 0.0842603]
// CHECK-NEXT:  [0.0794463, 0.0216516, 0.0394519, 0.0794463, 0.0239288, 0.0394519, 0.0794463, 0.0239288, 0.0532544, 0.0588553, 0.0794463, 0.0239288, 0.0394519, 0.0794463, 0.0239288, 0.0394519, 0.0794463, 0.0239288, 0.0532544, 0.0588553]
// CHECK-NEXT:  [0.0418424, 0.018801,  0.0253787, 0.0418424, 0.018801,  0.0253787, 0.0418424, 0.018801,  0.0342577, 0.0253787, 0.113739,  0.0309976, 0.0564813, 0.113739,  0.0309976, 0.0564813, 0.113739,  0.0309976, 0.0762418, 0.0842603]
// CHECK-NEXT:  [0.0794463, 0.0239288, 0.0394519, 0.0794463, 0.0239288, 0.0394519, 0.0794463, 0.0216516, 0.0532544, 0.0588553, 0.0794463, 0.0239288, 0.0394519, 0.0794463, 0.0239288, 0.0394519, 0.0794463, 0.0239288, 0.0532544, 0.0588553]
