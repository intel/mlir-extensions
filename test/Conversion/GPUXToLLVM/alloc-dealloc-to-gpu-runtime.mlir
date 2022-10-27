// RUN: imex-opt -allow-unregistered-dialect -split-input-file -convert-gpux-to-llvm -verify-diagnostics %s -o - | FileCheck %s

module attributes {gpu.container_module}{
  // CHECK-LABEL: llvm.func @main
  func.func @main() attributes {llvm.emit_c_interface} {
    // CHECK: %[[stream:.*]] = llvm.call @gpuCreateStream()
    %0 = "gpux.create_stream"() : () -> !gpux.StreamType
    %memref = "gpux.alloc"(%0) {operand_segment_sizes = array<i32: 0, 1, 0, 0>} : (!gpux.StreamType) -> memref<8xf32>
    %memref_0 = "gpux.alloc"(%0) {operand_segment_sizes = array<i32: 0, 1, 0, 0>} : (!gpux.StreamType) -> memref<8xf32>
    %memref_1 = "gpux.alloc"(%0) {operand_segment_sizes = array<i32: 0, 1, 0, 0>} : (!gpux.StreamType) -> memref<8xf32>
    "gpux.dealloc"(%0, %memref) : (!gpux.StreamType, memref<8xf32>) -> ()
    "gpux.dealloc"(%0, %memref_0) : (!gpux.StreamType, memref<8xf32>) -> ()
    "gpux.dealloc"(%0, %memref_1) : (!gpux.StreamType, memref<8xf32>) -> ()
    "gpux.destroy_stream"(%0) : (!gpux.StreamType) -> ()
    return
  }
}
