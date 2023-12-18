// RUN: imex-opt -convert-func-to-llvm -convert-gpux-to-llvm %s | FileCheck %s

module attributes {gpu.container_module}{
  // CHECK-LABEL: llvm.func @main
  func.func @main() attributes {llvm.emit_c_interface} {
    // CHECK: %[[DEVICE:.*]] =  llvm.mlir.zero : !llvm.ptr
    // CHECK: %[[CONTEXT:.*]] =  llvm.mlir.zero : !llvm.ptr
    // CHECK: %[[STREAM:.*]] = llvm.call @gpuCreateStream(%[[DEVICE]], %[[CONTEXT]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %0 = "gpux.create_stream"() : () -> !gpux.StreamType
    // CHECK: llvm.call @gpuMemAlloc(%[[STREAM]], %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, i64, i64, i32) -> !llvm.ptr
    %memref = "gpux.alloc"(%0) {operandSegmentSizes = array<i32: 0, 1, 0, 0>} : (!gpux.StreamType) -> memref<8xf32>
    // CHECK: llvm.call @gpuMemFree(%[[STREAM]], %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> ()
    "gpux.dealloc"(%0, %memref) : (!gpux.StreamType, memref<8xf32>) -> ()
    "gpux.destroy_stream"(%0) : (!gpux.StreamType) -> ()
    return
  }
}
