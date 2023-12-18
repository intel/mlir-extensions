// RUN: imex-opt -convert-func-to-llvm -convert-gpux-to-llvm %s | FileCheck %s

module attributes {gpu.container_module}{
  // CHECK-LABEL: llvm.func @main
  func.func @main() attributes {llvm.emit_c_interface} {
    // CHECK: %[[DEVICE:.*]] =  llvm.mlir.zero : !llvm.ptr
    // CHECK: %[[CONTEXT:.*]] =  llvm.mlir.zero : !llvm.ptr
    // CHECK: %[[STREAM:.*]] = llvm.call @gpuCreateStream(%[[DEVICE]], %[[CONTEXT]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %0 = "gpux.create_stream"() : () -> !gpux.StreamType
    // CHECK: llvm.call @gpuStreamDestroy(%[[STREAM]]) : (!llvm.ptr)
    "gpux.destroy_stream"(%0) : (!gpux.StreamType) -> ()
    return
  }
}
