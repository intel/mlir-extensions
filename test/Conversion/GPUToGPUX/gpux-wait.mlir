// RUN: imex-opt --convert-gpu-to-gpux %s | FileCheck %s

module attributes {
  gpu.container_module,
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spv.resource_limits<>>
}{
func.func @main() attributes {llvm.emit_c_interface} {
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[STREAM:.*]] = "gpux.create_stream"() : () -> !gpux.StreamType
  // CHECK: %[[WAIT:.*]] = "gpux.wait"(%[[STREAM:.*]]) : (!gpux.StreamType) -> !gpu.async.token
  // CHECK: %[[ALLOC_0:.*]], %[[ASYNC_TOKEN:.*]] = "gpux.alloc"(%[[WAIT:.*]], %[[STREAM:.*]]) {operand_segment_sizes = array<i32: 1, 1, 0, 0>} : (!gpu.async.token, !gpux.StreamType) -> (memref<8xf32>, !gpu.async.token)
  // CHECK: "gpux.wait"(%[[ASYNC_TOKEN:.*]], %[[STREAM:.*]]) : (!gpu.async.token, !gpux.StreamType) -> ()
  %0 = gpu.wait async
  %memref, %token_0 = gpu.alloc async [%0] () : memref<8xf32>
  gpu.wait[%token_0]
  %memref_1 = gpu.alloc () : memref<8xf32>
  %memref_2 = gpu.alloc () : memref<8xf32>
  // CHECK: %[[WAIT_1:.*]] = "gpux.wait"(%[[STREAM:.*]]) : (!gpux.StreamType) -> !gpu.async.token
  %1 = gpu.wait async
  // CHECK: %[[ASYNC_TOKEN_1:.*]] = "gpux.launch_func"(%[[WAIT_1:.*]], %[[STREAM:.*]], %[[C8:.*]], %[[C1:.*]], %[[C1:.*]], %[[C1:.*]], %[[C1:.*]], %[[C1:.*]], %[[ALLOC_0:.*]], %[[ALLOC_1:.*]], %[[ALLOC_2:.*]]) {kernel = @Kernels::@kernel_1, operand_segment_sizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 0, 3>} : (!gpu.async.token, !gpux.StreamType, index, index, index, index, index, index, memref<8xf32>, memref<8xf32>, memref<8xf32>) -> !gpu.async.token
  %token_1 = gpu.launch_func async [%1] @Kernels::@kernel_1 blocks in (%c8, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<8xf32>, %memref_1 : memref<8xf32>, %memref_2 : memref<8xf32>)
  // CHECK: "gpux.wait"(%[[ASYNC_TOKEN_1:.*]], %[[STREAM:.*]]) : (!gpu.async.token, !gpux.StreamType) -> ()
  gpu.wait[%token_1]
  // CHECK: "gpux.destroy_stream"(%[[STREAM:.*]]) : (!gpux.StreamType) -> ()
  return
}

gpu.module @Kernels attributes {gpu.binary = "\03\02#\07\00\00\01\00\16\00\00\00\17\00\00\00\00\00\00\00\11\00\02\00\0B\00\00\00\11\00\02\00\04\00\00\00\11\00\02\00\06\00\00\00\0E\00\03\00\02\00\00\00\02\00\00\00\0F\00\07\00\06\00\00\00\09\00\00\00main_kernel\00\04\00\00\00\05\00\09\00\04\00\00\00__builtin_var_WorkgroupId__\00\05\00\05\00\09\00\00\00main_kernel\00G\00\04\00\04\00\00\00\0B\00\00\00\1A\00\00\00\15\00\04\00\03\00\00\00@\00\00\00\00\00\00\00\17\00\04\00\02\00\00\00\03\00\00\00\03\00\00\00 \00\04\00\01\00\00\00\01\00\00\00\02\00\00\00;\00\04\00\01\00\00\00\04\00\00\00\01\00\00\00\13\00\02\00\06\00\00\00\16\00\03\00\08\00\00\00 \00\00\00 \00\04\00\07\00\00\00\05\00\00\00\08\00\00\00!\00\06\00\05\00\00\00\06\00\00\00\07\00\00\00\07\00\00\00\07\00\00\006\00\05\00\06\00\00\00\09\00\00\00\00\00\00\00\05\00\00\007\00\03\00\07\00\00\00\0A\00\00\007\00\03\00\07\00\00\00\0B\00\00\007\00\03\00\07\00\00\00\0C\00\00\00\F8\00\02\00\0D\00\00\00\F9\00\02\00\0E\00\00\00\F8\00\02\00\0E\00\00\00=\00\04\00\02\00\00\00\0F\00\00\00\04\00\00\00Q\00\05\00\03\00\00\00\10\00\00\00\0F\00\00\00\00\00\00\00F\00\05\00\07\00\00\00\11\00\00\00\0A\00\00\00\10\00\00\00=\00\06\00\08\00\00\00\12\00\00\00\11\00\00\00\02\00\00\00\04\00\00\00F\00\05\00\07\00\00\00\13\00\00\00\0B\00\00\00\10\00\00\00=\00\06\00\08\00\00\00\14\00\00\00\13\00\00\00\02\00\00\00\04\00\00\00\81\00\05\00\08\00\00\00\15\00\00\00\12\00\00\00\14\00\00\00F\00\05\00\07\00\00\00\16\00\00\00\0C\00\00\00\10\00\00\00>\00\05\00\16\00\00\00\15\00\00\00\02\00\00\00\04\00\00\00\FD\00\01\008\00\01\00"} {
    gpu.func @kernel_1(%arg0: memref<8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>) kernel attributes {spv.entry_point_abi = #spv.entry_point_abi<>} {
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
