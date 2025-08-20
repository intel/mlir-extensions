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
#map1 = affine_map<(d0, d1) -> (d0 + 2, d1 + 1)>
module @explicit_padding {
func.func @main() {
    %0= arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>:tensor<2x3xf32>
    %1 = call @test(%0) : (tensor<2x3xf32>) -> tensor<6x5xf32>
    %unranked = tensor.cast %1 : tensor<6x5xf32>to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    return
}
func.func private @printMemrefF32(tensor<*xf32>)
func.func @test(%arg0: tensor<2x3xf32>)->tensor<6x5xf32>{
    %cst = arith.constant 0.0 : f32
    %0 = tensor.empty() : tensor<6x5xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<6x5xf32>) -> tensor<6x5xf32>

    %2 = linalg.generic {
        indexing_maps = [#map0, #map1],
        iterator_types = ["parallel", "parallel"]
      }
      ins(%arg0 : tensor<2x3xf32>) outs(%1 : tensor<6x5xf32>)
      attrs =  {iterator_ranges = [2, 3], name = "explicit_padding"} {
        ^bb0(%arg1: f32, %arg2: f32):
          %e = arith.addf %arg1, %arg2: f32 // enforce arg2 is used, otherwise a new tensor is allocated
          %o = arith.subf %e, %cst: f32
          linalg.yield %o: f32
      } -> tensor<6x5xf32>
    return %2 : tensor<6x5xf32>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [6, 5] strides = {{.*}} data =
// CHECK-NEXT: [0, 0, 0, 0, 0]
// CHECK-NEXT: [0, 0, 0, 0, 0]
// CHECK-NEXT: [0, 1, 2, 3, 0]
// CHECK-NEXT: [0, 4, 5, 6, 0]
// CHECK-NEXT: [0, 0, 0, 0, 0]
// CHECK-NEXT: [0, 0, 0, 0, 0]
