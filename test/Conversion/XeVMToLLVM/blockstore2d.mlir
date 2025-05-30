// RUN: imex-opt -convert-xevm-to-llvm -split-input-file %s | FileCheck %s


gpu.module @store_funcs {
llvm.func @xevm.blockstore2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<8xi8>) {
  // CHECK:     llvm.func @xevm.blockstore2d(%arg0: !llvm.ptr<1>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: vector<8xi8>) {
  // CHECK-DAG:   [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG:   [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG:   [[UNDEF:%.*]] = llvm.mlir.undef : vector<2xi32>
  // CHECK-NEXT:  [[COORD0:%.*]] = llvm.insertelement %arg4, [[UNDEF]][[[ZERO]] : i32] : vector<2xi32>
  // CHECK-NEXT:  [[COORD1:%.*]] = llvm.insertelement %arg5, [[COORD0]][[[ONE]] : i32] : vector<2xi32>
  // CHECK:       [[C8:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT:  [[STOREVALPTR:%.*]] = llvm.alloca [[C8]] x i8 : (i32) -> !llvm.ptr
  // CHECK-NEXT:  llvm.store %arg6, [[STOREVALPTR]] : vector<8xi8>, !llvm.ptr
  // CHECK-NEXT:  llvm.call spir_funccc @_Z41intel_sub_group_2d_block_write_8b_8r16x1cPU3AS1viiiDv2_iPh(%arg0, %arg1, %arg2, %arg3, [[COORD1]], [[STOREVALPTR]])
  // CHECK: xevm.DecorationCacheControl = {{\[\[}}6443 : i32, 0 : i32, 1 : i32, 0 : i32{{\]}}, {{\[}}6443 : i32, 1 : i32, 1 : i32, 0 : i32{{\]\]}}
  // CHECK:       : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  xevm.blockstore2d %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=16, tile_height=8, v_blocks=1, l1_cache_control=UC, l3_cache_control=UC} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi8>)
  llvm.return
}
}

// -----

gpu.module @store_funcs {
llvm.func @xevm.blockstore2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<8xi16>) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_write_8b_8r32x1cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  xevm.blockstore2d %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)
  llvm.return
}
}

// -----

gpu.module @store_funcs {
llvm.func @xevm.blockstore2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<8xi16>) {
  // CHECK:      llvm.call spir_funccc @_Z42intel_sub_group_2d_block_write_16b_8r16x1cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  xevm.blockstore2d %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=16, tile_width=16, tile_height=8, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)
  llvm.return
}
}

// -----

gpu.module @store_funcs {
llvm.func @xevm.blockstore2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<8xi32>) {
  // CHECK:      llvm.call spir_funccc @_Z42intel_sub_group_2d_block_write_32b_8r16x1cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  xevm.blockstore2d %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=32, tile_width=16, tile_height=8, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
  llvm.return
}
}
