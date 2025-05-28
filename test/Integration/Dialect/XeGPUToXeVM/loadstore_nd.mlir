// RUN: %python_executable %imex_runner --requires=sycl-runtime,spirv-backend -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_sycl_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @load_store_2d(%src: memref<8x16xi32, 1>, %dst: memref<8x16xi32, 1>) kernel {
        %srcce = memref.memory_space_cast %src : memref<8x16xi32, 1> to memref<8x16xi32>
        %dstte = memref.memory_space_cast %dst : memref<8x16xi32, 1> to memref<8x16xi32>

        %src_tdesc = xegpu.create_nd_tdesc %srcce[0, 0] : memref<8x16xi32> -> !xegpu.tensor_desc<8x16xi32, #xegpu.block_tdesc_attr<memory_space = global>>
        %dst_tdesc = xegpu.create_nd_tdesc %dstte[0, 0] : memref<8x16xi32> -> !xegpu.tensor_desc<8x16xi32, #xegpu.block_tdesc_attr<memory_space = global>>
        %loaded = xegpu.load_nd %src_tdesc <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<8x16xi32, #xegpu.block_tdesc_attr<memory_space = global>> -> vector<8xi32>

        %sg_id = gpu.subgroup_id : index
        %sg_id_i32 = arith.index_cast %sg_id : index to i32
        %sg_lane_id = gpu.lane_id
        %sg_lane_id_i32 = arith.index_cast %sg_lane_id : index to i32
        %n_sg = gpu.num_subgroups : index
        %n_sg_i32 = arith.index_cast %n_sg : index to i32
        %sg_size = gpu.subgroup_size : index
        %sg_size_i32 = arith.index_cast %sg_size : index to i32

        %loaded_modified = vector.insert %sg_id_i32, %loaded[0] : i32 into vector<8xi32>
        %loaded_modified1 = vector.insert %sg_lane_id_i32, %loaded_modified[1] : i32 into vector<8xi32>
        %loaded_modified2 = vector.insert %n_sg_i32, %loaded_modified1[2] : i32 into vector<8xi32>
        %loaded_modified3 = vector.insert %sg_size_i32, %loaded_modified2[3] : i32 into vector<8xi32>

        xegpu.store_nd %loaded_modified3, %dst_tdesc <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>: vector<8xi32>, !xegpu.tensor_desc<8x16xi32, #xegpu.block_tdesc_attr<memory_space = global>>
        gpu.return
    }
  }

  func.func @test(%src : memref<8x16xi32>) -> memref<8x16xi32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %memref_src = gpu.alloc host_shared () : memref<8x16xi32>
    memref.copy %src, %memref_src : memref<8x16xi32> to memref<8x16xi32>
    %memref_dst = gpu.alloc host_shared () : memref<8x16xi32>
    %srcc = memref.memory_space_cast %memref_src : memref<8x16xi32> to memref<8x16xi32, 1>
    %dstt = memref.memory_space_cast %memref_dst : memref<8x16xi32> to memref<8x16xi32, 1>

    gpu.launch_func @kernel::@load_store_2d blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1) args(%srcc : memref<8x16xi32, 1>, %dstt : memref<8x16xi32, 1>)
    return %memref_dst : memref<8x16xi32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.alloc() : memref<8x16xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c11_i32 = arith.constant 11 : i32
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %c11_i32, %A[%i, %j] : memref<8x16xi32>
      }
    }
    %B = call @test(%A) : (memref<8x16xi32>) -> memref<8x16xi32>
    %B_cast = memref.cast %B : memref<8x16xi32> to memref<*xi32>
    %A_cast = memref.cast %A : memref<8x16xi32> to memref<*xi32>
    call @printMemrefI32(%A_cast) : (memref<*xi32>) -> ()
    call @printMemrefI32(%B_cast) : (memref<*xi32>) -> ()

    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-NEXT: [11{{.*}}]
    // CHECK-COUNT-96: 11
    // CHECK-NEXT: [11{{.*}}]

    // CHECK-NEXT: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK: 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    // CHECK: 0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15
    // CHECK: 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1
    // CHECK: 16,  16,  16,  16,  16,  16,  16,  16,  16,  16,  16,  16,  16,  16,  16,  16
    // CHECK-COUNT-48: 11
    // CHECK-NEXT: [11{{.*}}]

    memref.dealloc %A : memref<8x16xi32>
    return
  }
  func.func private @printMemrefI32(%ptr : memref<*xi32>) attributes {llvm.emit_c_interface}
}
