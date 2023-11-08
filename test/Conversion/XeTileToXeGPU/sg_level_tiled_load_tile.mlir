// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu --remove-dead-values %s -verify-diagnostics -o -| FileCheck %s
func.func @sglevel_tiled_load_tile(%a: memref<1024x1024xf16>, %b: memref<1024x1024xf16>, %c: memref<1024x1024xf32>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    //CHECK: arith.constant 0 : index
    //CHECK-NEXT: arith.constant 64 : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 8 : index
    //CHECK-NEXT: arith.constant 64 : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
	%1 = xetile.init_tile %a[%c0, %c64] : memref<1024x1024xf16> -> !xetile.tile<2x1x8x16xf16>
    //CHECK: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
    %2 = xetile.load_tile %1 : !xetile.tile<2x1x8x16xf16> -> vector<2x1x8x16xf16>
	return
}
