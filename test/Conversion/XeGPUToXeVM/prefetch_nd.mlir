// RUN: imex-opt -convert-xegpu-to-xevm -split-input-file %s | FileCheck %s

gpu.module @fence_check {
    gpu.func @fence(%src: memref<8x16xf32, 1>, %dst: memref<8x16xf32, 1>) kernel {
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
        %src_tdesc = xegpu.create_nd_tdesc %srcce[0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>

        //CHECK: %[[LD_DESC_I64:.*]] = vector.bitcast %[[LD_DESC]] : vector<8xi32> to vector<4xi64>
        //CHECK: %[[PREF_INTPTR:.*]] = vector.extract %[[LD_DESC_I64]][0] : i64 from vector<4xi64>
        //CHECK: %[[PREF_BASE_W:.*]] = vector.extract %[[LD_DESC]][2] : i32 from vector<8xi32>
        //CHECK: %[[PREF_BASE_H:.*]] = vector.extract %[[LD_DESC]][3] : i32 from vector<8xi32>
        //CHECK: %[[PREF_TILE_W:.*]] = vector.extract %[[LD_DESC]][4] : i32 from vector<8xi32>
        //CHECK: %[[PREF_TILE_H:.*]] = vector.extract %[[LD_DESC]][5] : i32 from vector<8xi32>
        //CHECK: %[[PREF_LLVMPTR:.*]] = llvm.inttoptr %[[PREF_INTPTR]] : i64 to !llvm.ptr<1>
        //CHECK: %[[PREF_SIZEOF_F32:.*]] = arith.constant 4 : i32
        //CHECK: %[[PREF_BASE_ROW_IN_BYTES:.*]] = arith.muli %[[PREF_BASE_W]], %[[PREF_SIZEOF_F32]] : i32
        //CHECK: xevm.blockprefetch2d %[[PREF_LLVMPTR]], %[[PREF_BASE_ROW_IN_BYTES]], %[[PREF_BASE_H]], %[[PREF_BASE_ROW_IN_BYTES]], %[[PREF_TILE_W]], %[[PREF_TILE_H]] {elem_size_in_bits = 32, tile_width = 16, tile_height = 8, v_blocks = 1, l1_cache_control = C, l3_cache_control = UC} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
        xegpu.prefetch_nd %src_tdesc<{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>

        gpu.return
    }
}

// /home/jovyan/graph-compiler/build/bin/gc-opt /home/jovyan/graph-compiler/test/mlir/test/gc/Conversion/GPU/XeGPUToXeVM/prefetch_nd.mlir --convert-xegpu-to-xevm
