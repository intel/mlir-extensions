// RUN: imex-opt %s | FileCheck %s


// CHECK-LABEL: @test_create_stream
func.func @test_create_stream() -> !gpux.StreamType{
    %3 = "gpux.create_stream"() : () -> !gpux.StreamType
    return %3 : !gpux.StreamType
}

// CHECK-LABEL: @test_destroy_stream
func.func @test_destroy_stream() {
    %0 = "gpux.create_stream"() : () -> !gpux.StreamType
    "gpux.destroy_stream"(%0) : (!gpux.StreamType) -> ()
    return
}

// CHECK-LABEL: @test_gpux_alloc
func.func @test_gpux_alloc() {
    %0 = "gpux.create_stream"() : () -> !gpux.StreamType
    %1 = "gpux.alloc"(%0) {operand_segment_sizes = array<i32: 0, 1, 0, 0>} : (!gpux.StreamType) -> memref<13xf32, 1>
    return
}

// CHECK-LABEL: @test_gpux_dealloc
func.func @test_gpux_dealloc() {
    %0 = "gpux.create_stream"() : () -> !gpux.StreamType
    %1 = "gpux.alloc"(%0) {operand_segment_sizes = array<i32: 0, 1, 0, 0>} : (!gpux.StreamType) -> memref<13xf32, 1>
    "gpux.dealloc"(%0, %1) : (!gpux.StreamType, memref<13xf32, 1>) -> ()
    return
}

// CHECK-LABEL: @test_gpux_wait
func.func @test_gpux_wait() {
    %0 = "gpux.create_stream"() : () -> !gpux.StreamType
    %1 = "gpux.alloc"(%0) {operand_segment_sizes = array<i32: 0, 1, 0, 0>} : (!gpux.StreamType) -> memref<13xf32, 1>
    "gpux.wait"(%0) : (!gpux.StreamType) -> ()
    return
}

gpu.module @kernels {
    gpu.func @kernel_1() kernel {
      gpu.return
    }
  }

// CHECK-LABEL: @test_gpux_launch_func
func.func @test_gpux_launch_func() {
    %0 = "gpux.create_stream"() : () -> !gpux.StreamType
    %cst = arith.constant 8 : index
    %1 = "gpux.alloc"(%0) {operand_segment_sizes = array<i32: 0, 1, 0, 0>} : (!gpux.StreamType) -> memref<13xf32, 1>
    %2 = "gpux.alloc"(%0) {operand_segment_sizes = array<i32: 0, 1, 0, 0>} : (!gpux.StreamType) -> memref<13xf32, 1>
    "gpux.launch_func"(%0, %cst, %cst, %cst, %cst, %cst, %cst, %1, %2) {kernel = @kernels::@kernel_1, operand_segment_sizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 1, 0, 2>}
                     : (!gpux.StreamType, index, index, index, index, index, index, memref<13xf32, 1>, memref<13xf32, 1>) -> ()
return
}


// CHECK-LABEL: @test_gpux_memcpy
func.func @test_gpux_memcpy(%dst : memref<3x7xf32>, %src : memref<3x7xf32, 1>) {
    %0 = "gpux.create_stream"() : () -> !gpux.StreamType
    "gpux.memcpy"(%0, %dst, %src) : (!gpux.StreamType, memref<3x7xf32>, memref<3x7xf32, 1>) -> ()
    return
}

// CHECK-LABEL: gpux.memset
func.func @test_gpux_memset(%dst : memref<3x7xf32>, %value : f32) {
    %0 = "gpux.create_stream"() : () -> !gpux.StreamType
    "gpux.memset"(%0, %dst, %value) : (!gpux.StreamType, memref<3x7xf32>, f32) -> ()
    return
}
