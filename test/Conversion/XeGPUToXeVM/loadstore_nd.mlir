// RUN: imex-opt -convert-xegpu-to-xevm -allow-unregistered-dialect %s | FileCheck %s

gpu.module @load_store_check {
    gpu.func @load_store(%src: memref<8x16xf32, 1>, %dst: memref<8x16xf32, 1>) kernel {
        %srcce = memref.memory_space_cast %src : memref<8x16xf32, 1> to memref<8x16xf32>
        %dstte = memref.memory_space_cast %dst : memref<8x16xf32, 1> to memref<8x16xf32>

        // CHECK: %[[LD_PTR_AS_I64:.*]] = arith.index_castui {{.*}} : index to i64
        // CHECK: %[[LD_CREATE_DESC_I64:.*]] = vector.bitcast {{.*}} : vector<8xi32> to vector<4xi64>
        // CHECK: %[[LD_DESC_0:.*]] = vector.insert %[[LD_PTR_AS_I64]], %[[LD_CREATE_DESC_I64]] [0] : i64 into vector<4xi64>
        // CHECK: %[[LD_DESC_1:.*]] = vector.bitcast %[[LD_DESC_0]] : vector<4xi64> to vector<8xi32>
        // CHECK: %[[LD_DESC_2:.*]] = vector.insert {{.*}}, %[[LD_DESC_1]] [2] : i32 into vector<8xi32>
        // CHECK: %[[LD_DESC_3:.*]] = vector.insert {{.*}}, %[[LD_DESC_2]] [3] : i32 into vector<8xi32>
        // CHECK: %[[LD_DESC_4:.*]] = vector.insert {{.*}}, %[[LD_DESC_3]] [4] : i32 into vector<8xi32>
        // CHECK: %[[LD_DESC:.*]] = vector.insert {{.*}}, %[[LD_DESC_4]] [5] : i32 into vector<8xi32>
        %src_tdesc = xegpu.create_nd_tdesc %srcce[0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>


        //CHECK: %[[LD_DESC_I64:.*]] = vector.bitcast %[[LD_DESC]] : vector<8xi32> to vector<4xi64>
        //CHECK: %[[LD_INTPTR:.*]] = vector.extract %[[LD_DESC_I64]][0] : i64 from vector<4xi64>
        //CHECK: %[[LD_BASE_W:.*]] = vector.extract %[[LD_DESC]][2] : i32 from vector<8xi32>
        //CHECK: %[[LD_BASE_H:.*]] = vector.extract %[[LD_DESC]][3] : i32 from vector<8xi32>
        //CHECK: %[[LD_TILE_W:.*]] = vector.extract %[[LD_DESC]][4] : i32 from vector<8xi32>
        //CHECK: %[[LD_TILE_H:.*]] = vector.extract %[[LD_DESC]][5] : i32 from vector<8xi32>
        //CHECK: %[[LD_LLVMPTR:.*]] = llvm.inttoptr %[[LD_INTPTR]] : i64 to !llvm.ptr<1>
        //CHECK: %[[LD_SIZEOF_F32:.*]] = arith.constant 4 : i32
        //CHECK: %[[LD_BASE_ROW_IN_BYTES:.*]] = arith.muli %[[LD_BASE_W]], %[[LD_SIZEOF_F32]] : i32
        //CHECK: %[[LD_LOADED_I32:.*]] = xevm.blockload2d %[[LD_LLVMPTR]], %[[LD_BASE_ROW_IN_BYTES]],
        //CHECK-SAME: %[[LD_BASE_H]], %[[LD_BASE_ROW_IN_BYTES]], %[[LD_TILE_W]], %[[LD_TILE_H]]
        //CHECK-SAME: <{cache_control = #xevm.load_cache_control<L1c_L2uc_L3uc>, elem_size_in_bits = 32 : i32,
        //CHECK-SAME:   pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false,
        //CHECK-SAME:   v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
        %loaded = xegpu.load_nd %src_tdesc <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<8x16xf32> -> vector<8xf32>
        //CHECK: %[[LD_LOADED_F32:.*]] = vector.bitcast %[[LD_LOADED_I32]] : vector<8xi32> to vector<8xf32>

        %tid_x = gpu.thread_id x
        %tid_x_i32 = arith.index_cast %tid_x : index to i32
        %tid_x_f32 = arith.sitofp %tid_x_i32 : i32 to f32
        //CHECK: %[[LOADED_F32_MODIFIED:.*]] = vector.insert %{{.*}}, %[[LD_LOADED_F32]] [0] : f32 into vector<8xf32>
        %loaded_modified = vector.insert %tid_x_f32, %loaded[0] : f32 into vector<8xf32>

        // CHECK: %[[PTR_AS_I64:.*]] = arith.index_castui {{.*}} : index to i64
        // CHECK: %[[CREATE_DESC_I64:.*]] = vector.bitcast {{.*}} : vector<8xi32> to vector<4xi64>
        // CHECK: %[[DESC_0:.*]] = vector.insert %[[PTR_AS_I64]], %[[CREATE_DESC_I64]] [0] : i64 into vector<4xi64>
        // CHECK: %[[DESC_1:.*]] = vector.bitcast %[[DESC_0]] : vector<4xi64> to vector<8xi32>
        // CHECK: %[[DESC_2:.*]] = vector.insert {{.*}}, %[[DESC_1]] [2] : i32 into vector<8xi32>
        // CHECK: %[[DESC_3:.*]] = vector.insert {{.*}}, %[[DESC_2]] [3] : i32 into vector<8xi32>
        // CHECK: %[[DESC_4:.*]] = vector.insert {{.*}}, %[[DESC_3]] [4] : i32 into vector<8xi32>
        // CHECK: %[[DESC:.*]] = vector.insert {{.*}}, %[[DESC_4]] [5] : i32 into vector<8xi32>
        %dst_tdesc = xegpu.create_nd_tdesc %dstte[0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>

        //CHECK: %[[DESC_I64:.*]] = vector.bitcast %[[DESC]] : vector<8xi32> to vector<4xi64>
        //CHECK: %[[INTPTR:.*]] = vector.extract %[[DESC_I64]][0] : i64 from vector<4xi64>
        //CHECK: %[[BASE_W:.*]] = vector.extract %[[DESC]][2] : i32 from vector<8xi32>
        //CHECK: %[[BASE_H:.*]] = vector.extract %[[DESC]][3] : i32 from vector<8xi32>
        //CHECK: %[[TILE_W:.*]] = vector.extract %[[DESC]][4] : i32 from vector<8xi32>
        //CHECK: %[[TILE_H:.*]] = vector.extract %[[DESC]][5] : i32 from vector<8xi32>
        //CHECK: %[[LLVMPTR:.*]] = llvm.inttoptr %[[INTPTR]] : i64 to !llvm.ptr<1>
        //CHECK: %[[SIZEOF_F32:.*]] = arith.constant 4 : i32
        //CHECK: %[[BASE_ROW_IN_BYTES:.*]] = arith.muli %[[BASE_W]], %[[SIZEOF_F32]] : i32
        //CHECK: %[[FLAT_VALUE_I32:.*]] = vector.bitcast %[[LOADED_F32_MODIFIED]] : vector<8xf32> to vector<8xi32>
        //CHECK: xevm.blockstore2d %[[LLVMPTR]], %[[BASE_ROW_IN_BYTES]], %[[BASE_H]], %[[BASE_ROW_IN_BYTES]],
        //CHECK-SAME: %[[TILE_W]], %[[TILE_H]], %[[FLAT_VALUE_I32]]
        //CHECK-SAME: <{cache_control = #xevm.store_cache_control<L1wb_L2uc_L3uc>, elem_size_in_bits = 32 : i32,
        //CHECK-SAME:   tile_height = 8 : i32, tile_width = 16 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
        xegpu.store_nd %loaded_modified, %dst_tdesc <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>: vector<8xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>
        gpu.return
    }

    gpu.func @create_nd_tdesc_integer_source(%src: i64, %src_h : index, %src_w : index)  kernel {
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c8 = arith.constant 8 : index
        %c0 = arith.constant 0 : index
        // CHECK: %[[PAYLOAD:.*]] = arith.constant dense<0> : vector<8xi32>
        // CHECK: %[[T0:.*]] = arith.index_cast %{{.*}} : index to i64
        // CHECK: %[[T1:.*]] = arith.trunci %[[T0]] : i64 to i32
        // CHECK: %[[T2:.*]] = arith.index_cast %{{.*}} : index to i64
        // CHECK: %[[T3:.*]] = arith.trunci %[[T2]] : i64 to i32
        // CHECK: %[[T4:.*]] = arith.index_cast %{{.*}} : index to i64
        // CHECK: %[[T5:.*]] = arith.trunci %[[T4]] : i64 to i32
        // CHECK: %[[T6:.*]] = arith.index_cast %{{.*}} : index to i64
        // CHECK: %[[T7:.*]] = arith.trunci %[[T6]] : i64 to i32
        // CHECK: %[[T8:.*]] = vector.bitcast %[[PAYLOAD]] : vector<8xi32> to vector<4xi64>
        // CHECK: %[[T9:.*]] = vector.insert %{{.*}}, %[[T8]] [0] : i64 into vector<4xi64>
        // CHECK: %[[T10:.*]] = vector.bitcast %[[T9]] : vector<4xi64> to vector<8xi32>
        // CHECK: %[[T11:.*]] = vector.insert %[[T5]], %[[T10]] [2] : i32 into vector<8xi32>
        // CHECK: %[[T12:.*]] = vector.insert %[[T7]], %[[T11]] [3] : i32 into vector<8xi32>
        // CHECK: %[[T13:.*]] = vector.insert %[[T1]], %[[T12]] [4] : i32 into vector<8xi32>
        // CHECK: %[[T14:.*]] = vector.insert %[[T3]], %[[T13]] [5] : i32 into vector<8xi32>
        %src_tdesc = xegpu.create_nd_tdesc %src [%c4, %c8], shape: [%src_h, %src_w], strides: [%src_w, %c1] : i64
            -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>
        "some_op"(%src_tdesc) : (!xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>) -> ()
        gpu.return
    }
}
