// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @load_store_2d(%src: memref<8x16xf16, 1>, %dst: memref<8x16xf16, 1>) kernel {
        %srcce = memref.memory_space_cast %src : memref<8x16xf16, 1> to memref<8x16xf16>
        %dstte = memref.memory_space_cast %dst : memref<8x16xf16, 1> to memref<8x16xf16>

        %src_tdesc = xegpu.create_nd_tdesc %srcce[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space = global>, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
        %dst_tdesc = xegpu.create_nd_tdesc %dstte[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space = global>, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>

        xegpu.prefetch_nd %src_tdesc<{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space = global>, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>

        %loaded = xegpu.load_nd %src_tdesc <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space = global>, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>> -> vector<8x1xf16>

        %tid_x = gpu.thread_id x
        %tid_x_i32 = arith.index_cast %tid_x : index to i32
        %tid_x_f32 = arith.sitofp %tid_x_i32 : i32 to f16

        %loaded_modified = vector.insert %tid_x_f32, %loaded[0, 0] : f16 into vector<8x1xf16>

        xegpu.store_nd %loaded_modified, %dst_tdesc <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>: vector<8x1xf16>, !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space = global>, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
        gpu.return
    }
  }

  func.func @test(%src : memref<8x16xf16>) -> memref<8x16xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %memref_src = gpu.alloc host_shared () : memref<8x16xf16>
    memref.copy %src, %memref_src : memref<8x16xf16> to memref<8x16xf16>
    %memref_dst = gpu.alloc host_shared () : memref<8x16xf16>
    %srcc = memref.memory_space_cast %memref_src : memref<8x16xf16> to memref<8x16xf16, 1>
    %dstt = memref.memory_space_cast %memref_dst : memref<8x16xf16> to memref<8x16xf16, 1>

    gpu.launch_func @kernel::@load_store_2d blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1) args(%srcc : memref<8x16xf16, 1>, %dstt : memref<8x16xf16, 1>)
    return %memref_dst : memref<8x16xf16>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.alloc() : memref<8x16xf16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c11_f32 = arith.constant 11.0 : f16
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %c11_f32, %A[%i, %j] : memref<8x16xf16>
      }
    }
    %B = call @test(%A) : (memref<8x16xf16>) -> memref<8x16xf16>
    %B_cast = memref.cast %B : memref<8x16xf16> to memref<*xf16>
    %A_cast = memref.cast %A : memref<8x16xf16> to memref<*xf16>
    call @printMemrefF16(%A_cast) : (memref<*xf16>) -> ()
    call @printMemrefF16(%B_cast) : (memref<*xf16>) -> ()

    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-NEXT: [11{{.*}}]
    // CHECK-COUNT-96: 11
    // CHECK-NEXT: [11{{.*}}]

    // CHECK-NEXT: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    // CHECK-COUNT-96: 11
    // CHECK-NEXT: [11{{.*}}]

    memref.dealloc %A : memref<8x16xf16>
    return
  }
  func.func private @printMemrefF16(%ptr : memref<*xf16>) attributes {llvm.emit_c_interface}
}
