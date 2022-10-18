// RUN: imex-opt -allow-unregistered-dialect %s -split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt -allow-unregistered-dialect %s | imex-opt -allow-unregistered-dialect | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -allow-unregistered-dialect -mlir-print-op-generic %s | imex-opt -allow-unregistered-dialect | FileCheck %s


// CHECK-LABEL: gpux.create_stream
func.func @test_create_stream() -> !imex.gpux.StreamType {
    %3 = "imex.gpux.create_stream"() : () -> !imex.gpux.StreamType
    return %3 : !imex.gpux.StreamType
}

// CHECK-LABEL: gpux.destroy_stream
func.func @test_destroy_stream() {
    %0 = "imex.gpux.create_stream"() : () -> !imex.gpux.StreamType
    "imex.gpux.destroy_stream"(%0) : (!imex.gpux.StreamType) -> ()
    return
}

// CHECK-LABEL: gpux.alloc
func.func @test_gpux_alloc() {
    %0 = "imex.gpux.create_stream"() : () -> !imex.gpux.StreamType
    %1 = "imex.gpux.alloc"(%0) : (!imex.gpux.StreamType) -> memref<13xf32, 1>
    return
}

// CHECK-LABEL: gpux.dealloc
func.func @test_gpux_dealloc() {
    %0 = "imex.gpux.create_stream"() : () -> !imex.gpux.StreamType
    %1 = "imex.gpux.alloc"(%0) : (!imex.gpux.StreamType) -> memref<13xf32, 1>
    "imex.gpux.dealloc"(%0, %1) : (!imex.gpux.StreamType, memref<13xf32, 1>) -> ()
    return
}

// CHECK-LABEL: gpux.wait
func.func @test_gpux_wait() {
    %0 = "imex.gpux.create_stream"() : () -> !imex.gpux.StreamType
    %1 = "imex.gpux.alloc"(%0) : (!imex.gpux.StreamType) -> memref<13xf32, 1>
    "imex.gpux.wait"(%0) : (!imex.gpux.StreamType) -> ()
    return
}

gpu.module @kernels {
    gpu.func @kernel_1() kernel {
      gpu.return
    }
  }

// CHECK-LABEL: gpux.launch_func
func.func @test_gpux_launch_func() {
%0 = "imex.gpux.create_stream"() : () -> !imex.gpux.StreamType
%cst = arith.constant 8 : index
%1 = "imex.gpux.alloc"(%0) : (!imex.gpux.StreamType) -> memref<13xf32, 1>
%2 = "imex.gpux.alloc"(%0) : (!imex.gpux.StreamType) -> memref<13xf32, 1>
"imex.gpux.launch_func"(%0, %cst, %cst, %cst, %cst, %cst, %cst, %1, %2) {kernel = @kernels::@kernel_1}
                     : (!imex.gpux.StreamType, index, index, index, index, index, index, memref<13xf32, 1>, memref<13xf32, 1>) -> ()
return
}


// CHECK-LABEL: gpux.memcpy
func.func @test_gpux_memcpy(%dst : memref<3x7xf32>, %src : memref<3x7xf32, 1>) {
    %0 = "imex.gpux.create_stream"() : () -> !imex.gpux.StreamType
    "imex.gpux.memcpy"(%0, %dst, %src) : (!imex.gpux.StreamType, memref<3x7xf32>, memref<3x7xf32, 1>) -> ()
    return
}

// CHECK-LABEL: gpux.memset
func.func @test_gpux_memset(%dst : memref<3x7xf32>, %value : f32) {
    %0 = "imex.gpux.create_stream"() : () -> !imex.gpux.StreamType
    "imex.gpux.memset"(%0, %dst, %value) : (!imex.gpux.StreamType, memref<3x7xf32>, f32) -> ()
    return
}
