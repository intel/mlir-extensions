// RUN: imex-opt --convert-gpu-to-gpux %s | FileCheck %s


func.func @memcpy(%dst : memref<3x7xf32>, %src : memref<3x7xf32, 1>) {
    // CHECK-LABEL: func @memcpy
    // CHECK: %[[STREAM:.*]] = "gpux.create_stream"() : () -> !gpux.StreamType
    // CHECK: "gpux.memcpy"(%[[STREAM]], {{.*}}, {{.*}}) : (!gpux.StreamType, memref<3x7xf32>, memref<3x7xf32, 1>) -> ()
    gpu.memcpy %dst, %src : memref<3x7xf32>, memref<3x7xf32, 1>
    // CHECK: "gpux.destroy_stream"(%0) : (!gpux.StreamType) -> ()
    return
  }
