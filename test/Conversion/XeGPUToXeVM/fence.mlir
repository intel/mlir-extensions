// RUN: imex-opt -convert-xegpu-to-xevm -split-input-file %s | FileCheck %s

gpu.module @fence_check {
    gpu.func @fence(%dst: memref<8x16xf32, 1>) kernel {
        %tid_x = gpu.thread_id x
        %tid_x_i32 = arith.index_cast %tid_x : index to i32
        %tid_x_f32 = arith.sitofp %tid_x_i32 : i32 to f32
        // CHECK: xevm.memfence addrspace = global, scope = workgroup
        xegpu.fence memory_kind = global, fence_scope = workgroup
        %c0 = arith.constant 0 : index
        memref.store %tid_x_f32, %dst[%c0, %c0] : memref<8x16xf32, 1>
        gpu.return
    }
}
