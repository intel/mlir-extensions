// RUN: %python_executable %imex_runner --requires=mlir-sycl-runtime,spirv-backend -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @load_store_2d(%src: memref<128xf32, 1>, %dst: memref<128xf32, 1>) kernel {
        %srcce = memref.memory_space_cast %src : memref<128xf32, 1> to memref<128xf32>
        %dstte = memref.memory_space_cast %dst : memref<128xf32, 1> to memref<128xf32>

        %offsets = arith.constant dense<[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]> : vector<16xindex>
        %src_tdesc = xegpu.create_tdesc %srcce, %offsets : memref<128xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
        %dst_tdesc = xegpu.create_tdesc %dstte, %offsets : memref<128xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>

        %mask = arith.constant dense<[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]> : vector<16xi1>
        %loaded = xegpu.load %src_tdesc, %mask <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<16xi1> -> vector<2xf32>

        %tid_x = gpu.thread_id x
        %tid_x_i32 = arith.index_cast %tid_x : index to i32
        %tid_x_f32 = arith.sitofp %tid_x_i32 : i32 to f32
        %loaded_modified = vector.insert %tid_x_f32, %loaded[0] : f32 into vector<2xf32>

        xegpu.store %loaded_modified, %dst_tdesc, %mask <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<2xf32>, !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<16xi1>
        gpu.return
    }
  }

  func.func @test(%src : memref<8x16xf32>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %memref_src = gpu.alloc host_shared () : memref<8x16xf32>
    memref.copy %src, %memref_src : memref<8x16xf32> to memref<8x16xf32>
    %memref_dst = gpu.alloc host_shared () : memref<8x16xf32>
    %srcc = memref.memory_space_cast %memref_src : memref<8x16xf32> to memref<8x16xf32, 1>
    %dstt = memref.memory_space_cast %memref_dst : memref<8x16xf32> to memref<8x16xf32, 1>
    %srccc = memref.reinterpret_cast %srcc to offset: [0], sizes: [128],
           strides: [1] : memref<8x16xf32, 1> to memref<128xf32, 1>
    %dstte = memref.reinterpret_cast %dstt to offset: [0], sizes: [128],
           strides: [1] : memref<8x16xf32, 1> to memref<128xf32, 1>
    gpu.launch_func @kernel::@load_store_2d blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1) args(%srccc : memref<128xf32, 1>, %dstte : memref<128xf32, 1>)
    return %memref_dst : memref<8x16xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.alloc() : memref<8x16xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c11_f32 = arith.constant 11.11 : f32
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %c11_f32, %A[%i, %j] : memref<8x16xf32>
      }
    }
    %B = call @test(%A) : (memref<8x16xf32>) -> memref<8x16xf32>
    %B_cast = memref.cast %B : memref<8x16xf32> to memref<*xf32>
    %A_cast = memref.cast %A : memref<8x16xf32> to memref<*xf32>
    call @printMemrefF32(%A_cast) : (memref<*xf32>) -> ()
    call @printMemrefF32(%B_cast) : (memref<*xf32>) -> ()

    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-NEXT: [11.11{{.*}}]
    // CHECK-COUNT-96: 11.11
    // CHECK-NEXT: [11.11{{.*}}]

    // CHECK-NEXT: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK: 0, 11.11, 1, 11.11, 2, 11.11, 3, 11.11, 4, 11.11, 5, 11.11, 6, 11.11, 7, 11.11
    // CHECK: 8, 11.11, 9, 11.11, 10, 11.11, 11, 11.11, 12, 11.11, 13, 11.11, 14, 11.11, 15, 11.11
    // CHECK-COUNT-80: 0
    // CHECK-NEXT: [0{{.*}}]

    memref.dealloc %A : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
}
