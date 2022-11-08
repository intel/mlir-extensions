// RUN: imex-opt -convert-func-to-llvm -convert-gpux-to-llvm %s | FileCheck %s

module attributes {gpu.container_module}{
  // CHECK-LABEL: llvm.func @main
  func.func @main() attributes {llvm.emit_c_interface} {
    // CHECK: %[[DEVICE:.*]] =  llvm.mlir.null : !llvm.ptr<i8>
    // CHECK: %[[CONTEXT:.*]] =  llvm.mlir.null : !llvm.ptr<i8>
    // CHECK: %[[STREAM:.*]] = llvm.call @gpuCreateStream(%[[DEVICE:.*]], %[[CONTEXT:.*]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.ptr<i8>
    %0 = "gpux.create_stream"() : () -> !gpux.StreamType
    // CHECK: llvm.call @gpuMemAlloc(%[[stream:.*]], %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, i64, i64, i32) -> !llvm.ptr<i8>
    %memref = "gpux.alloc"(%0) {operand_segment_sizes = array<i32: 0, 1, 0, 0>} : (!gpux.StreamType) -> memref<8xf32>
    // CHECK: llvm.call @gpuMemFree(%[[stream:.*]], %{{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
    "gpux.dealloc"(%0, %memref) : (!gpux.StreamType, memref<8xf32>) -> ()
    "gpux.destroy_stream"(%0) : (!gpux.StreamType) -> ()
    return
  }
}
