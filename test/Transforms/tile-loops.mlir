// RUN: imex-opt --split-input-file -tile-loops='tile-sizes=32' -tile-loops='tile-sizes=1' %s -verify-diagnostics -o -| FileCheck %s

#map = affine_map<(d0) -> (d0)>
module {
  func.func @add(%arg0: tensor<129xf32>, %arg1: tensor<129xf32>, %arg2: tensor<129xf32>) -> tensor<129xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<129xf32>, tensor<129xf32>) outs(%arg2 : tensor<129xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.addf %in, %in_0 : f32
      linalg.yield %1 : f32
    } -> tensor<129xf32>
    return %0 : tensor<129xf32>
  }
}
// CHECK-LABEL: func.func @add
// CHECK-NEXT: %[[FORALL:.*]] = scf.forall (%arg3) = (0) to (129) step (32) shared_outs(%arg4 = %arg2) -> (tensor<129xf32>) {
// CHECK-NEXT: %[[C129:.*]] = arith.constant 129 : index
// CHECK-NEXT: %[[MIN:.*]] = affine.min #map(%[[ARG3:.*]])
// CHECK-NEXT: %[[APPLY1:.*]] = affine.apply #map1(%[[MIN]])
// CHECK: %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %arg0[%[[ARG3]]] [%[[MIN]]] [1] : tensor<129xf32> to tensor<?xf32>
// CHECK-NEXT: %[[EXTRACTED_SLICE_0:.*]] = tensor.extract_slice %arg1[%[[ARG3]]] [%[[MIN]]] [1] : tensor<129xf32> to tensor<?xf32>
// CHECK-NEXT: %[[EXTRACTED_SLICE_1:.*]] = tensor.extract_slice %arg4[%[[ARG3]]] [%[[MIN]]] [1] : tensor<129xf32> to tensor<?xf32>
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[FORALL:.*]] = scf.forall (%[[ARG5:.*]]) in (%[[MIN]]) shared_outs(%[[ARG6:.*]] = %[[EXTRACTED_SLICE_1]]) -> (tensor<?xf32>) {
// CHECK-NEXT: %[[EXTRACTED_SLICE_4:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE]][%[[ARG5]]] [1] [1] : tensor<?xf32> to tensor<1xf32>
// CHECK-NEXT: %[[EXTRACTED_SLICE_5:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE_0]][%[[ARG5]]] [1] [1] : tensor<?xf32> to tensor<1xf32>
// CHECK-NEXT: %[[EXTRACTED_SLICE_6:.*]] = tensor.extract_slice %[[ARG6]][%[[ARG5]]] [1] [1] : tensor<?xf32> to tensor<1xf32>
// CHECK-NEXT: %[[GENERIC:.*]] = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%[[EXTRACTED_SLICE_4]], %[[EXTRACTED_SLICE_5]] : tensor<1xf32>, tensor<1xf32>) outs(%[[EXTRACTED_SLICE_6]] : tensor<1xf32>) {
// CHECK-NEXT: ^bb0(%[[IN:.*]]: f32, %[[IN_7:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT: %[[ADDF:.*]] = arith.addf %[[IN]], %[[IN_7]] : f32
// CHECK-NEXT: linalg.yield %[[ADDF]] : f32
// CHECK-NEXT: } -> tensor<1xf32>
// CHECK-NEXT: scf.forall.in_parallel {
// CHECK-NEXT: tensor.parallel_insert_slice %[[GENERIC]] into %[[ARG6]][%[[ARG5]]] [1] [1] : tensor<1xf32> into tensor<?xf32>
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK: scf.forall.in_parallel {
// CHECK-NEXT: tensor.parallel_insert_slice %[[FORALL]] into %arg4[%[[ARG3]]] [%[[MIN]]] [1] : tensor<?xf32> into tensor<129xf32>
// CHECK-NEXT: }
