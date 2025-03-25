// RUN: imex-opt -convert-xevm-to-llvm -split-input-file %s | FileCheck %s

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:     llvm.func @xevm.blockload2d(%arg0: !llvm.ptr<1>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  // CHECK-DAG:  [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG:  [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG:  [[UNDEF:%.*]] = llvm.mlir.undef : vector<2xi32>
  // CHECK-NEXT: [[COORD0:%.*]] = llvm.insertelement %arg4, [[UNDEF]][[[ZERO]] : i32] : vector<2xi32>
  // CHECK-NEXT: [[COORD1:%.*]] = llvm.insertelement %arg5, [[COORD0]][[[ONE]] : i32] : vector<2xi32>
  // CHECK:  [[EIGHT:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT:  [[DEST:%.*]] = llvm.alloca [[EIGHT]] x i16 : (i32) -> !llvm.ptr
  // CHECK-NEXT: llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x1cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, [[COORD1]], [[DEST]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<8xi16>

  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_8b_16r32x1cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi16>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=16, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_8b_32r32x1cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<32xi16>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=32, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi16>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r16x1cPU3AS1viiiDv2_iPh(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<8xi8>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi8>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x1cPU3AS1viiiDv2_iPh(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi8>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi8>
  llvm.return
}
}

// -----

// COM: This case come from the 06 tutorial of FP8 flash attention.
gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r16x4cPU3AS1viiiDv2_iPh(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<32xi8>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=8, v_blocks=4, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi8>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<8xi16>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_16r16x1cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi16>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=16, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x1cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<32xi16>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=32, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi16>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r32x1cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi16>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_32b_8r8x1cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<4xi32>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<4xi32>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_32b_8r16x1cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<8xi32>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=16, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_32b_16r8x1cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=16, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_32b_16r16x1cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi32>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=16, tile_height=16, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi32>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_32b_32r8x1cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi32>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=32, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi32>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_32b_8r2x1cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<1xi32>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=2, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<1xi32>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x2cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi16>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_8b_16r32x2cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<32xi16>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=16, v_blocks=2, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi16>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_8b_32r32x2cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<64xi16>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=32, v_blocks=2, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<64xi16>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x2cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi16>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_16r16x2cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<32xi16>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=16, v_blocks=2, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi16>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x2cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<64xi16>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=32, v_blocks=2, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<64xi16>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_32b_8r8x2cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_32b_16r8x2cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=16, v_blocks=2, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi32>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_32b_32r8x2cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=32, v_blocks=2, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi32>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z51intel_sub_group_2d_block_read_transform_8b_32r16x1cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<8xi32>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=32, v_blocks=1, transpose=false, vnni_transform=true, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z51intel_sub_group_2d_block_read_transform_8b_32r16x2cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi32>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=32, v_blocks=2, transpose=false, vnni_transform=true, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi32>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z51intel_sub_group_2d_block_read_transform_8b_32r16x4cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<32xi32>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=32, v_blocks=4, transpose=false, vnni_transform=true, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi32>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_16r16x1cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<8xi32>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=16, v_blocks=1, transpose=false, vnni_transform=true, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_32r16x1cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi32>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=32, v_blocks=1, transpose=false, vnni_transform=true, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi32>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_16r16x2cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi32>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=16, v_blocks=2, transpose=false, vnni_transform=true, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi32>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_32r16x2cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<32xi32>
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=32, v_blocks=2, transpose=false, vnni_transform=true, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi32>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z51intel_sub_group_2d_block_read_transpose_32b_16r8x1cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr {{.*}}) -> ()
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=16, v_blocks=1, transpose=true, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:        llvm.func @xevm.blockload2d(
  // CHECK:          llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x2cPU3AS1viiiDv2_iPt(
  // CHECK: xevm.DecorationCacheControl = {{\[\[}}6442 : i32, 0 : i32, 1 : i32, 0 : i32{{\]}}, {{\[}}6442 : i32, 1 : i32, 1 : i32, 0 : i32{{\]\]}}
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, l1_cache_control=UC, l3_cache_control=UC} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:        llvm.func @xevm.blockload2d(
  // CHECK:          llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x2cPU3AS1viiiDv2_iPt(
  // CHECK: xevm.DecorationCacheControl = {{\[\[}}6442 : i32, 0 : i32, 1 : i32, 0 : i32{{\]}}, {{\[}}6442 : i32, 1 : i32, 2 : i32, 0 : i32{{\]\]}}
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, l1_cache_control=UC, l3_cache_control=C} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:        llvm.func @xevm.blockload2d(
  // CHECK:          llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x2cPU3AS1viiiDv2_iPt(
  // CHECK: xevm.DecorationCacheControl = {{\[\[}}6442 : i32, 0 : i32, 2 : i32, 0 : i32{{\]}}, {{\[}}6442 : i32, 1 : i32, 1 : i32, 0 : i32{{\]\]}}
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, l1_cache_control=C, l3_cache_control=UC} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:        llvm.func @xevm.blockload2d(
  // CHECK:          llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x2cPU3AS1viiiDv2_iPt(
  // CHECK: xevm.DecorationCacheControl = {{\[\[}}6442 : i32, 0 : i32, 2 : i32, 0 : i32{{\]}}, {{\[}}6442 : i32, 1 : i32, 2 : i32, 0 : i32{{\]\]}}
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, l1_cache_control=C, l3_cache_control=C} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:        llvm.func @xevm.blockload2d(
  // CHECK:          llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x2cPU3AS1viiiDv2_iPt(
  // CHECK: xevm.DecorationCacheControl = {{\[\[}}6442 : i32, 0 : i32, 3 : i32, 0 : i32{{\]}}, {{\[}}6442 : i32, 1 : i32, 1 : i32, 0 : i32{{\]\]}}
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, l1_cache_control=S, l3_cache_control=UC} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:        llvm.func @xevm.blockload2d(
  // CHECK:          llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x2cPU3AS1viiiDv2_iPt(
  // CHECK: xevm.DecorationCacheControl = {{\[\[}}6442 : i32, 0 : i32, 3 : i32, 0 : i32{{\]}}, {{\[}}6442 : i32, 1 : i32, 2 : i32, 0 : i32{{\]\]}}
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, l1_cache_control=S, l3_cache_control=C} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:        llvm.func @xevm.blockload2d(
  // CHECK:          llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x2cPU3AS1viiiDv2_iPt(
  // CHECK: xevm.DecorationCacheControl = {{\[\[}}6442 : i32, 0 : i32, 4 : i32, 0 : i32{{\]}}, {{\[}}6442 : i32, 1 : i32, 2 : i32, 0 : i32{{\]\]}}
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, l1_cache_control=IAR, l3_cache_control=C} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}
}

// -----

gpu.module @load_funcs {
llvm.func @xevm.blockload2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:        llvm.func @xevm.blockload2d(
  // CHECK:          llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x2cPU3AS1viiiDv2_iPt(
  // CHECK-NOT:        xevm.DecorationCacheControl
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}
}
