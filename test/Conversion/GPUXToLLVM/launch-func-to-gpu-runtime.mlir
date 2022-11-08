// RUN: imex-opt -convert-func-to-llvm -convert-gpux-to-llvm %s | FileCheck %s

module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>} {
   // CHECK: llvm.mlir.global internal constant @Kernels_kernel_1_kernel_name("kernel_1\00") {addr_space = 0 : i32}
   // CHECK: llvm.mlir.global internal constant @Kernels_spirv_binary("\03\02#\07\00\00\01\00\16\00\00\00\17\00\00\00\00\00\00\00\11\00\02\00\0B\00\00\00\11\00\02\00\04\00\00\00\11\00\02\00\06\00\00\00\0E\00\03\00\02\00\00\00\02\00\00\00\0F\00\07\00\06\00\00\00\09\00\00\00main_kernel\00\04\00\00\00\05\00\09\00\04\00\00\00__builtin_var_WorkgroupId__\00\05\00\05\00\09\00\00\00main_kernel\00G\00\04\00\04\00\00\00\0B\00\00\00\1A\00\00\00\15\00\04\00\03\00\00\00@\00\00\00\00\00\00\00\17\00\04\00\02\00\00\00\03\00\00\00\03\00\00\00 \00\04\00\01\00\00\00\01\00\00\00\02\00\00\00;\00\04\00\01\00\00\00\04\00\00\00\01\00\00\00\13\00\02\00\06\00\00\00\16\00\03\00\08\00\00\00 \00\00\00 \00\04\00\07\00\00\00\05\00\00\00\08\00\00\00!\00\06\00\05\00\00\00\06\00\00\00\07\00\00\00\07\00\00\00\07\00\00\006\00\05\00\06\00\00\00\09\00\00\00\00\00\00\00\05\00\00\007\00\03\00\07\00\00\00\0A\00\00\007\00\03\00\07\00\00\00\0B\00\00\007\00\03\00\07\00\00\00\0C\00\00\00\F8\00\02\00\0D\00\00\00\F9\00\02\00\0E\00\00\00\F8\00\02\00\0E\00\00\00=\00\04\00\02\00\00\00\0F\00\00\00\04\00\00\00Q\00\05\00\03\00\00\00\10\00\00\00\0F\00\00\00\00\00\00\00F\00\05\00\07\00\00\00\11\00\00\00\0A\00\00\00\10\00\00\00=\00\06\00\08\00\00\00\12\00\00\00\11\00\00\00\02\00\00\00\04\00\00\00F\00\05\00\07\00\00\00\13\00\00\00\0B\00\00\00\10\00\00\00=\00\06\00\08\00\00\00\14\00\00\00\13\00\00\00\02\00\00\00\04\00\00\00\81\00\05\00\08\00\00\00\15\00\00\00\12\00\00\00\14\00\00\00F\00\05\00\07\00\00\00\16\00\00\00\0C\00\00\00\10\00\00\00>\00\05\00\16\00\00\00\15\00\00\00\02\00\00\00\04\00\00\00\FD\00\01\008\00\01\00") {addr_space = 0 : i32}
  func.func @main() attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    // CHECK: %[[DEVICE:.*]] =  llvm.mlir.null : !llvm.ptr<i8>
    // CHECK: %[[CONTEXT:.*]] =  llvm.mlir.null : !llvm.ptr<i8>
    // CHECK: %[[STREAM:.*]] = llvm.call @gpuCreateStream(%[[DEVICE:.*]], %[[CONTEXT:.*]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.ptr<i8>
    %0 = "gpux.create_stream"() : () -> !gpux.StreamType
    %memref = "gpux.alloc"(%0) {operand_segment_sizes = array<i32: 0, 1, 0, 0>} : (!gpux.StreamType) -> memref<8xf32>
    %memref_0 = "gpux.alloc"(%0) {operand_segment_sizes = array<i32: 0, 1, 0, 0>} : (!gpux.StreamType) -> memref<8xf32>
    %memref_1 = "gpux.alloc"(%0) {operand_segment_sizes = array<i32: 0, 1, 0, 0>} : (!gpux.StreamType) -> memref<8xf32>

    // CHECK: llvm.mlir.addressof @Kernels_spirv_binary : !llvm.ptr<array<552 x i8>>
    // CHECK: %[[MODULE:.*]] = llvm.call @gpuModuleLoad(%[[STREAM:.*]], %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
    // CHECK: llvm.mlir.addressof @Kernels_kernel_1_kernel_name : !llvm.ptr<array<9 x i8>>
    // CHECK: %[[KERNEL:.*]] = llvm.call @gpuKernelGet(%[[STREAM:.*]], %[[MODULE:.*]], %{{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.ptr<i8>
    // CHECK: llvm.call @gpuLaunchKernel(%[[STREAM:.*]], %[[KERNEL:.*]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i64, i64, i64, i64, i64, i32, !llvm.ptr<struct<(ptr<i8>, i64)>>) -> ()
    "gpux.launch_func"(%0, %c8, %c1, %c1, %c1, %c1, %c1, %memref, %memref_0, %memref_1) {kernel = @Kernels::@kernel_1, operand_segment_sizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 1, 0, 3>} : (!gpux.StreamType, index, index, index, index, index, index, memref<8xf32>, memref<8xf32>, memref<8xf32>) -> ()
    "gpux.dealloc"(%0, %memref) : (!gpux.StreamType, memref<8xf32>) -> ()
    "gpux.dealloc"(%0, %memref_0) : (!gpux.StreamType, memref<8xf32>) -> ()
    "gpux.dealloc"(%0, %memref_1) : (!gpux.StreamType, memref<8xf32>) -> ()
    "gpux.destroy_stream"(%0) : (!gpux.StreamType) -> ()
    return
  }
  gpu.module @Kernels attributes {gpu.binary = "\03\02#\07\00\00\01\00\16\00\00\00\17\00\00\00\00\00\00\00\11\00\02\00\0B\00\00\00\11\00\02\00\04\00\00\00\11\00\02\00\06\00\00\00\0E\00\03\00\02\00\00\00\02\00\00\00\0F\00\07\00\06\00\00\00\09\00\00\00main_kernel\00\04\00\00\00\05\00\09\00\04\00\00\00__builtin_var_WorkgroupId__\00\05\00\05\00\09\00\00\00main_kernel\00G\00\04\00\04\00\00\00\0B\00\00\00\1A\00\00\00\15\00\04\00\03\00\00\00@\00\00\00\00\00\00\00\17\00\04\00\02\00\00\00\03\00\00\00\03\00\00\00 \00\04\00\01\00\00\00\01\00\00\00\02\00\00\00;\00\04\00\01\00\00\00\04\00\00\00\01\00\00\00\13\00\02\00\06\00\00\00\16\00\03\00\08\00\00\00 \00\00\00 \00\04\00\07\00\00\00\05\00\00\00\08\00\00\00!\00\06\00\05\00\00\00\06\00\00\00\07\00\00\00\07\00\00\00\07\00\00\006\00\05\00\06\00\00\00\09\00\00\00\00\00\00\00\05\00\00\007\00\03\00\07\00\00\00\0A\00\00\007\00\03\00\07\00\00\00\0B\00\00\007\00\03\00\07\00\00\00\0C\00\00\00\F8\00\02\00\0D\00\00\00\F9\00\02\00\0E\00\00\00\F8\00\02\00\0E\00\00\00=\00\04\00\02\00\00\00\0F\00\00\00\04\00\00\00Q\00\05\00\03\00\00\00\10\00\00\00\0F\00\00\00\00\00\00\00F\00\05\00\07\00\00\00\11\00\00\00\0A\00\00\00\10\00\00\00=\00\06\00\08\00\00\00\12\00\00\00\11\00\00\00\02\00\00\00\04\00\00\00F\00\05\00\07\00\00\00\13\00\00\00\0B\00\00\00\10\00\00\00=\00\06\00\08\00\00\00\14\00\00\00\13\00\00\00\02\00\00\00\04\00\00\00\81\00\05\00\08\00\00\00\15\00\00\00\12\00\00\00\14\00\00\00F\00\05\00\07\00\00\00\16\00\00\00\0C\00\00\00\10\00\00\00>\00\05\00\16\00\00\00\15\00\00\00\02\00\00\00\04\00\00\00\FD\00\01\008\00\01\00"} {
    gpu.func @kernel_1(%arg0: memref<8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %0 = gpu.block_id  x
      %1 = memref.load %arg0[%0] : memref<8xf32>
      %2 = memref.load %arg1[%0] : memref<8xf32>
      %3 = arith.addf %1, %2 : f32
      memref.store %3, %arg2[%0] : memref<8xf32>
      gpu.return
    }
  }
}
