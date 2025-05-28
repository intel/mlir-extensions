// RUN: %python_executable %imex_runner --requires=sycl-runtime,spirv-backend -i %s --pass-pipeline-file=%p/xevm-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_sycl_runtime --filecheck

module @gemm attributes {gpu.container_module} {

  gpu.module @kernel {
    gpu.func @store_constant(%ptr: !llvm.ptr<1>) kernel {
      %const_val = arith.constant 42.0 : f32
      %thread_x = gpu.lane_id
      %thread_x_i64 = arith.index_cast %thread_x : index to i64
      %ptr_next_1 = llvm.getelementptr %ptr[%thread_x_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
      llvm.store %const_val, %ptr_next_1 : f32, !llvm.ptr<1>
      gpu.return
    }
  }
  func.func @test(%src : memref<8x16xf32>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %memref_0 = gpu.alloc host_shared () : memref<8x16xf32>
    memref.copy %src, %memref_0 : memref<8x16xf32> to memref<8x16xf32>
    %0 = memref.extract_aligned_pointer_as_index %memref_0 : memref<8x16xf32> -> index
    %1 = arith.index_cast %0 : index to i64
    %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %src_casted = llvm.addrspacecast %2 : !llvm.ptr to !llvm.ptr<1>
    gpu.launch_func @kernel::@store_constant blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1) args(%src_casted : !llvm.ptr<1>)
    return %memref_0 : memref<8x16xf32>
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
    // CHECK-NEXT: [42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42]
    // CHECK-COUNT-96: 11.11
    // CHECK-NEXT: [11.11{{.*}}]

    memref.dealloc %A : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
}
