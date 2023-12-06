// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu --remove-dead-values %s -verify-diagnostics -o -| FileCheck %s
// CHECK: sglevel
func.func @sglevel_tiled_gemm(%a: memref<1024x1024xf16>, %b: memref<1024x1024xf16>) {
  //CHECK: arith.constant 0 : index
  %c0 = arith.constant 0 : index
  //CHECK: arith.constant 64 : index
  %c64 = arith.constant 64 : index
  %c1024 = arith.constant 1024 : index
  //CHECK: arith.constant 0 : index
  //CHECK: arith.constant 64 : index
  //CHECK: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  //CHECK: arith.constant 8 : index
  //CHECK: arith.constant 64 : index
  //CHECK: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
	%1 = xetile.init_tile %a[%c0, %c64] : memref<1024x1024xf16> -> !xetile.tile<2x1x8x16xf16>
  //CHECK: arith.constant dense<0.000000e+00> : vector<8x16xf16>
  //CHECK: arith.constant dense<0.000000e+00> : vector<8x16xf16>
  %2 = arith.constant dense<0.0> : vector<2x1x8x16xf16>

  //CHECK: !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, vector<8x16xf16>, vector<8x16xf16>
  %nexta, %res = scf.for %k= %c0 to %c1024 step %c64 iter_args(%subA = %1, %subB = %2) -> (!xetile.tile<2x1x8x16xf16>, vector<2x1x8x16xf16>) {
    //CHECK: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
    //CHECK: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
    %3 = xetile.load_tile %subA : !xetile.tile<2x1x8x16xf16> -> vector<2x1x8x16xf16>
    //CHECK: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    %5 = xetile.update_tile_offset %subA, [%c0, %c64]: !xetile.tile<2x1x8x16xf16>, index, index -> !xetile.tile<2x1x8x16xf16>
    //CHECK: !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, vector<8x16xf16>, vector<8x16xf16>
    scf.yield %5, %3: !xetile.tile<2x1x8x16xf16>, vector<2x1x8x16xf16>
  }
  //CHECK: arith.constant 0 : index
  //CHECK-NEXT: arith.constant 64 : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  //CHECK-NEXT: arith.constant 8 : index
  //CHECK-NEXT: arith.constant 64 : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
	%5 = xetile.init_tile %b[%c0, %c64] : memref<1024x1024xf16> -> !xetile.tile<2x1x8x16xf16>
  //CHECK: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
  //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
  xetile.store_tile %res, %5: vector<2x1x8x16xf16>, !xetile.tile<2x1x8x16xf16>

	return
}
