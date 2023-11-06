// RUN: imex-opt -convert-func-to-llvm -convert-gpux-to-llvm %s | FileCheck %s

module attributes {gpu.container_module}{
  // CHECK-LABEL: llvm.func @main
  func.func @main() attributes {llvm.emit_c_interface} {
    // CHECK: %[[DEVICE:.*]] =  llvm.mlir.zero : !llvm.ptr<i8>
    // CHECK: %[[CONTEXT:.*]] =  llvm.mlir.zero : !llvm.ptr<i8>
    // CHECK: %[[STREAM:.*]] = llvm.call @gpuCreateStream(%[[DEVICE]], %[[CONTEXT]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.ptr<i8>
    %0 = "gpux.create_stream"() : () -> !gpux.StreamType
    // CHECK: llvm.call @gpuStreamDestroy(%[[STREAM]]) : (!llvm.ptr<i8>) -> ()
    "gpux.destroy_stream"(%0) : (!gpux.StreamType) -> ()
    return
  }
}
