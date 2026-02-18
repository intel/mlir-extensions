// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=lane" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --shared-libs=%irunner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// Example of pass pipeline usage:
// %python_executable %imex_runner --requires=mlir-levelzero-runtime,spirv-backend -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
//                                        --runner mlir-runner -e main \
//                                        --entry-point-result=void \
//                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @load_store_2d(%src: memref<128xf32, 1>, %dst: memref<128xf32, 1>) kernel {
        %srcce = memref.memory_space_cast %src : memref<128xf32, 1> to memref<128xf32>
        %dstte = memref.memory_space_cast %dst : memref<128xf32, 1> to memref<128xf32>

        %c2 = arith.constant 2 : index
        %tid_x = gpu.thread_id x
        %offsets = arith.muli %tid_x, %c2 : index

        %mask = arith.constant 1 : i1
        %loaded = xegpu.load %srcce[%offsets], %mask <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, chunk_size = 2}> : memref<128xf32>, index, i1 -> vector<2xf32>

        %tid_x_i32 = arith.index_cast %tid_x : index to i32
        %tid_x_f32 = arith.sitofp %tid_x_i32 : i32 to f32
        %loaded_modified = vector.insert %tid_x_f32, %loaded[0] : f32 into vector<2xf32>

        xegpu.store %loaded_modified, %dstte[%offsets], %mask <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>, chunk_size = 2}> : vector<2xf32>, memref<128xf32>, index, i1
        gpu.return
    }
  }

  func.func @test(%src : memref<8x16xf32>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %memref_src = gpu.alloc  () : memref<8x16xf32>
    gpu.memcpy %memref_src, %src : memref<8x16xf32>, memref<8x16xf32>
    %memref_dst = gpu.alloc  () : memref<8x16xf32>
    %srcc = memref.memory_space_cast %memref_src : memref<8x16xf32> to memref<8x16xf32, 1>
    %dstt = memref.memory_space_cast %memref_dst : memref<8x16xf32> to memref<8x16xf32, 1>
    %srccc = memref.reinterpret_cast %srcc to offset: [0], sizes: [128],
           strides: [1] : memref<8x16xf32, 1> to memref<128xf32, 1>
    %dstte = memref.reinterpret_cast %dstt to offset: [0], sizes: [128],
           strides: [1] : memref<8x16xf32, 1> to memref<128xf32, 1>
    gpu.launch_func @kernel::@load_store_2d blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1) args(%srccc : memref<128xf32, 1>, %dstte : memref<128xf32, 1>)
    gpu.wait
    %out = memref.alloc () : memref<8x16xf32>
    gpu.memcpy %out, %memref_dst : memref<8x16xf32>, memref<8x16xf32>
    gpu.dealloc %memref_src : memref<8x16xf32>
    gpu.dealloc %memref_dst : memref<8x16xf32>
    return %out : memref<8x16xf32>
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
