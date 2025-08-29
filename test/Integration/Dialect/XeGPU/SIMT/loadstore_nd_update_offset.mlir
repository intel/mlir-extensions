// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime,spirv-backend -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @load_store_2d(%src: memref<16x16xf32, 1>, %dst: memref<16x16xf32, 1>) kernel {
        %srcce = memref.memory_space_cast %src : memref<16x16xf32, 1> to memref<16x16xf32>
        %dstte = memref.memory_space_cast %dst : memref<16x16xf32, 1> to memref<16x16xf32>

        %src_tdesc = xegpu.create_nd_tdesc %srcce[0, 0] : memref<16x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>
        %dst_tdesc = xegpu.create_nd_tdesc %dstte[0, 0] : memref<16x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>
        %loaded = xegpu.load_nd %src_tdesc <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>> -> vector<8xf32>

        %tid_x = gpu.thread_id x
        %tid_x_i32 = arith.index_cast %tid_x : index to i32
        %tid_x_f32 = arith.sitofp %tid_x_i32 : i32 to f32

        %loaded_modified = vector.insert %tid_x_f32, %loaded[0] : f32 into vector<8xf32>

        xegpu.store_nd %loaded_modified, %dst_tdesc <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>: vector<8xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>

        %c8 = arith.constant 8 : index
        %c0 = arith.constant 0 : index
        %dst_tdesc_new = xegpu.update_nd_offset %dst_tdesc, [%c8, %c0] : !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>
        xegpu.store_nd %loaded_modified, %dst_tdesc_new <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>: vector<8xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>
        gpu.return
    }
  }

  func.func @test(%src : memref<16x16xf32>) -> memref<16x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %memref_src = gpu.alloc  () : memref<16x16xf32>
    gpu.memcpy %memref_src, %src : memref<16x16xf32>, memref<16x16xf32>
    %memref_dst = gpu.alloc  () : memref<16x16xf32>
    %srcc = memref.memory_space_cast %memref_src : memref<16x16xf32> to memref<16x16xf32, 1>
    %dstt = memref.memory_space_cast %memref_dst : memref<16x16xf32> to memref<16x16xf32, 1>

    gpu.launch_func @kernel::@load_store_2d blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1) args(%srcc : memref<16x16xf32, 1>, %dstt : memref<16x16xf32, 1>)
    gpu.wait
    %out = memref.alloc () : memref<16x16xf32>
    gpu.memcpy %out, %memref_dst : memref<16x16xf32>, memref<16x16xf32>
    gpu.dealloc %memref_src : memref<16x16xf32>
    gpu.dealloc %memref_dst : memref<16x16xf32>
    return %out : memref<16x16xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.alloc() : memref<16x16xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 16 : index
    %c16 = arith.constant 16 : index
    %c11_f32 = arith.constant 11.11 : f32
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %c11_f32, %A[%i, %j] : memref<16x16xf32>
      }
    }
    %B = call @test(%A) : (memref<16x16xf32>) -> memref<16x16xf32>
    %B_cast = memref.cast %B : memref<16x16xf32> to memref<*xf32>
    %A_cast = memref.cast %A : memref<16x16xf32> to memref<*xf32>
    call @printMemrefF32(%A_cast) : (memref<*xf32>) -> ()
    call @printMemrefF32(%B_cast) : (memref<*xf32>) -> ()

    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-NEXT: [11.11{{.*}}]
    // CHECK-COUNT-224: 11.11
    // CHECK-NEXT: [11.11{{.*}}]

    // CHECK-NEXT: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    // CHECK-COUNT-96: 11.11
    // CHECK-NEXT: [11.11{{.*}}]
    // CHECK: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    // CHECK-COUNT-96: 11.11
    // CHECK-NEXT: [11.11{{.*}}]

    memref.dealloc %A : memref<16x16xf32>
    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
}
