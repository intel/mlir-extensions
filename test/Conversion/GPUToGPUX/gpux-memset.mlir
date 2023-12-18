// RUN: imex-opt --convert-gpu-to-gpux %s | FileCheck %s


func.func @memset(%dst : memref<3x7xf32>, %value : f32) {
    // CHECK-LABEL: func @memset
    // CHECK: %[[STREAM:.*]] = "gpux.create_stream"() : () -> !gpux.StreamType
    // CHECK: "gpux.memset"(%[[STREAM]], {{.*}}, {{.*}}) : (!gpux.StreamType, memref<3x7xf32>, f32) -> ()
    gpu.memset %dst, %value : memref<3x7xf32>, f32
    // CHECK: "gpux.destroy_stream"(%0) : (!gpux.StreamType) -> ()
    return
  }
