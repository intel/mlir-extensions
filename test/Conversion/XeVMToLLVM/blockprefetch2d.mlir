// RUN: imex-opt -convert-xevm-to-llvm -split-input-file %s | FileCheck %s

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:     llvm.func @xevm.blockprefetch2d(%arg0: !llvm.ptr<1>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  // CHECK-DAG:  [[UNDEF:%.*]] = llvm.mlir.undef : vector<2xi32>
  // CHECK-DAG:  [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG:  [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: [[COORD0:%.*]] = llvm.insertelement %arg4, [[UNDEF]][[[ZERO]] : i32] : vector<2xi32>
  // CHECK-NEXT: [[COORD1:%.*]] = llvm.insertelement %arg5, [[COORD0]][[[ONE]] : i32] : vector<2xi32>
  // CHECK-NEXT: llvm.call spir_funccc @_Z44intel_sub_group_2d_block_prefetch_8b_8r32x1cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, [[COORD1]])
  // CHECK: xevm.DecorationCacheControl = {{\[\[}}6442 : i32, 0 : i32, 1 : i32, 0 : i32{{\]}}, {{\[}}6442 : i32, 1 : i32, 1 : i32, 0 : i32{{\]\]}}
  // CHECK: (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, l1_cache_control=UC, l3_cache_control=UC} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----


gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_8b_16r32x1cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=16, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_8b_32r32x1cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=32, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x1cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=8, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z46intel_sub_group_2d_block_prefetch_16b_16r16x1cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=16, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z46intel_sub_group_2d_block_prefetch_16b_32r16x1cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=32, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z44intel_sub_group_2d_block_prefetch_32b_8r8x1cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_32b_8r16x1cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=16, tile_height=8, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_32b_16r8x1cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=16, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z46intel_sub_group_2d_block_prefetch_32b_16r16x1cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=16, tile_height=16, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_32b_32r8x1cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=32, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z46intel_sub_group_2d_block_prefetch_32b_32r16x1cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=16, tile_height=32, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z44intel_sub_group_2d_block_prefetch_8b_8r32x2cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_8b_16r32x2cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=16, v_blocks=2, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_8b_32r32x2cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=32, v_blocks=2, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x2cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=8, v_blocks=2, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z46intel_sub_group_2d_block_prefetch_16b_16r16x2cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=16, v_blocks=2, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z46intel_sub_group_2d_block_prefetch_16b_32r16x2cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=32, v_blocks=2, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z44intel_sub_group_2d_block_prefetch_32b_8r8x2cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=2, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_32b_16r8x2cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=16, v_blocks=2, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_32b_32r8x2cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=32, v_blocks=2, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_8b_32r16x1cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=32, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_8b_32r16x2cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=32, v_blocks=2, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

gpu.module @prefetch_funcs {
llvm.func @xevm.blockprefetch2d(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_8b_32r16x4cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (!llvm.ptr<1> {{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=32, v_blocks=4, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}
