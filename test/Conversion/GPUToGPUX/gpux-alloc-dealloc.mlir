// RUN: imex-opt --convert-gpu-to-gpux %s | FileCheck %s

func.func @main() attributes {llvm.emit_c_interface} {
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[STREAM:.*]] = "gpux.create_stream"() : () -> !gpux.StreamType
  // CHECK: %[[ALLOC_0:.*]] = "gpux.alloc"(%[[STREAM]]) <{operandSegmentSizes = array<i32: 0, 1, 0, 0>}> : (!gpux.StreamType) -> memref<8xf32>
  %memref = gpu.alloc  () : memref<8xf32>
  // CHECK: %[[ALLOC_1:.*]] = "gpux.alloc"(%[[STREAM]]) <{operandSegmentSizes = array<i32: 0, 1, 0, 0>}> : (!gpux.StreamType) -> memref<8xf32>
  %memref_1 = gpu.alloc  () : memref<8xf32>
  // CHECK: %[[ALLOC_2:.*]] = "gpux.alloc"(%[[STREAM]]) <{operandSegmentSizes = array<i32: 0, 1, 0, 0>}> : (!gpux.StreamType) -> memref<8xf32>
  %memref_2 = gpu.alloc  () : memref<8xf32>
  // CHECK: "gpux.dealloc"(%[[STREAM]], %[[ALLOC_0]]) : (!gpux.StreamType, memref<8xf32>) -> ()
  gpu.dealloc  %memref : memref<8xf32>
  // CHECK: "gpux.dealloc"(%[[STREAM]], %[[ALLOC_1]]) : (!gpux.StreamType, memref<8xf32>) -> ()
  gpu.dealloc  %memref_1 : memref<8xf32>
  // CHECK: "gpux.dealloc"(%[[STREAM]], %[[ALLOC_2]]) : (!gpux.StreamType, memref<8xf32>) -> ()
  gpu.dealloc  %memref_2 : memref<8xf32>
  // CHECK: "gpux.destroy_stream"(%[[STREAM]]) : (!gpux.StreamType) -> ()
  return
}
