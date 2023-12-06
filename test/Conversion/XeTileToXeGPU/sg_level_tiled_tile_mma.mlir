// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu --remove-dead-values %s -verify-diagnostics -o -| FileCheck %s
func.func @sglevel_tiled_gemm(%a: memref<1024x1024xf16>, %b: memref<1024x1024xf16>) {
    //CHECK: arith.constant 0 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index

	%tile_0_dim_0 = arith.constant 128 : index
	%tile_0_dim_1 = arith.constant 64 : index
    %dl_0_dim_0 = arith.constant 2 : index
    %dl_0_dim_1 = arith.constant 1 : index
	%tile_0_block_size_dim_0 = arith.divsi %tile_0_dim_0, %dl_0_dim_0 : index
	%tile_0_block_size_dim_1 = arith.divsi %tile_0_dim_1, %dl_0_dim_1 : index
	%x0 = arith.remsi %c0, %dl_0_dim_0 : index
	%y0 = arith.remsi %c0, %dl_0_dim_1 : index
    %tmp_offset_0_dim_0 = arith.muli %x0, %tile_0_block_size_dim_0 : index
    %tmp_offset_0_dim_1 = arith.muli %y0, %tile_0_block_size_dim_1 : index
    %offset_0_dim_0 = arith.addi %c64, %tmp_offset_0_dim_0 : index
    %offset_0_dim_1 = arith.addi %c0, %tmp_offset_0_dim_1: index
    //CHECK:      arith.constant 0 : index
    //CHECK-NEXT: arith.constant 0 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 0 : index
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 0 : index
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 0 : index
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 8 : index
    //CHECK-NEXT: arith.constant 0 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 8 : index
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 8 : index
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 8 : index
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.constant 0 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 24 : index
    //CHECK-NEXT: arith.constant 0 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 24 : index
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 24 : index
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 24 : index
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.constant 0 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 40 : index
    //CHECK-NEXT: arith.constant 0 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 40 : index
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 40 : index
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 40 : index
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.constant 0 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 56 : index
    //CHECK-NEXT: arith.constant 0 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 56 : index
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 56 : index
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 56 : index
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
	%1 = xetile.init_tile %a[%offset_0_dim_0, %offset_0_dim_1] : memref<1024x1024xf16> -> !xetile.tile<8x4x8x16xf16>

    //CHECK:      xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    %2 = xetile.load_tile %1 : !xetile.tile<8x4x8x16xf16> -> vector<8x4x8x16xf16>

    %tile_1_dim_0 = arith.constant 64 : index
    %tile_1_dim_1 = arith.constant 128 : index
    %dl_1_dim_0 = arith.constant 1 : index
    %dl_1_dim_1 = arith.constant 2 : index
    %tile_1_block_size_dim_0 = arith.divsi %tile_1_dim_0, %dl_1_dim_0 : index
    %tile_1_block_size_dim_1 = arith.divsi %tile_1_dim_1, %dl_1_dim_1 : index
    %x1 = arith.remsi %c0, %dl_1_dim_0 : index
    %y1 = arith.remsi %c0, %dl_1_dim_1 : index
    %tmp_offset_1_dim_0 = arith.muli %x1, %tile_1_block_size_dim_0 : index
    %tmp_offset_1_dim_1 = arith.muli %y1, %tile_1_block_size_dim_1 : index
    %offset_1_dim_0 = arith.addi %c0, %tmp_offset_1_dim_0 : index
    %offset_1_dim_1 = arith.addi %c0, %tmp_offset_1_dim_1: index

    //CHECK: arith.constant 0 : index
    //CHECK-NEXT: arith.constant 0 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 0 : index
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 0 : index
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 0 : index
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.constant 0 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.constant 0 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.constant 0 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: arith.addi {{.*}} : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
	%3 = xetile.init_tile %b[%offset_1_dim_0, %offset_1_dim_1] : memref<1024x1024xf16> -> !xetile.tile<4x4x16x16xf16>

    //CHECK:      xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
  	%4 = xetile.load_tile %3 : !xetile.tile<4x4x16x16xf16> -> vector<4x4x16x16xf16>


    //CHECK: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<8x16xf32>
	%subC = arith.constant dense<0.0> : vector<8x4x8x16xf32>


    //CHECK:      xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %6 = xetile.tile_mma %2, %4, %subC: vector<8x4x8x16xf16>, vector<4x4x16x16xf16>, vector<8x4x8x16xf32> -> vector<8x4x8x16xf32>


	return
}
