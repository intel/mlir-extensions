// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu --remove-dead-values %s -verify-diagnostics -o -| FileCheck %s
func.func @sglevel_tiled_gemm(%a: memref<1024x1024xf16>, %b: memref<1024x1024xf16>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index

    //CHECK:      arith.constant 0 : index
    //CHECK-NEXT: arith.constant 64 : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 0 : index
    //CHECK-NEXT: arith.constant 80 : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 8 : index
    //CHECK-NEXT: arith.constant 64 : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 8 : index
    //CHECK-NEXT: arith.constant 80 : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.constant 64 : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.constant 80 : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 24 : index
    //CHECK-NEXT: arith.constant 64 : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: arith.constant 24 : index
    //CHECK-NEXT: arith.constant 80 : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
	%1 = xetile.init_tile %a[%c0, %c64] : memref<1024x1024xf16> -> !xetile.tile<4x2x8x16xf16>

    //CHECK:      xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
    %2 = xetile.load_tile %1 : !xetile.tile<4x2x8x16xf16> -> vector<4x2x8x16xf16>

    //CHECK:      arith.constant 0 : index
    //CHECK-NEXT: arith.constant 64 : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.constant 64 : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.constant 64 : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.constant 64 : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 0 : index
    //CHECK-NEXT: arith.constant 80 : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 16 : index
    //CHECK-NEXT: arith.constant 80 : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 32 : index
    //CHECK-NEXT: arith.constant 80 : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: arith.constant 48 : index
    //CHECK-NEXT: arith.constant 80 : index
    //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
	%3 = xetile.init_tile %b[%c64, %c0] : memref<1024x1024xf16> -> !xetile.tile<2x4x16x16xf16>

    //CHECK:      xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
  	%4 = xetile.load_tile %3 : !xetile.tile<2x4x16x16xf16> -> vector<2x4x16x16xf16>

    //CHECK:      xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %6 = xetile.tile_mma %2, %4: vector<4x2x8x16xf16>, vector<2x4x16x16xf16> -> vector<4x4x8x16xf32>
	return
}
