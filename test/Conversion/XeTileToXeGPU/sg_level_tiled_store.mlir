// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu --remove-dead-values %s -verify-diagnostics -o -| FileCheck %s
func.func @sglevel_tiled_store(%a: memref<1024x1024xf32>) {
	// CHECK: arith.constant 0 : index
	// CHECK-NEXT: arith.constant 32 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 0 : index
	// CHECK-NEXT: arith.constant 48 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 0 : index
	// CHECK-NEXT: arith.constant 64 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 0 : index
	// CHECK-NEXT: arith.constant 80 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 8 : index
	// CHECK-NEXT: arith.constant 32 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 8 : index
	// CHECK-NEXT: arith.constant 48 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 8 : index
	// CHECK-NEXT: arith.constant 64 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 8 : index
	// CHECK-NEXT: arith.constant 80 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 16 : index
	// CHECK-NEXT: arith.constant 32 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 16 : index
	// CHECK-NEXT: arith.constant 48 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 16 : index
	// CHECK-NEXT: arith.constant 64 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 16 : index
	// CHECK-NEXT: arith.constant 80 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 24 : index
	// CHECK-NEXT: arith.constant 32 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 24 : index
	// CHECK-NEXT: arith.constant 48 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 24 : index
	// CHECK-NEXT: arith.constant 64 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 24 : index
	// CHECK-NEXT: arith.constant 80 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 32 : index
	// CHECK-NEXT: arith.constant 32 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 32 : index
	// CHECK-NEXT: arith.constant 48 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 32 : index
	// CHECK-NEXT: arith.constant 64 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 32 : index
	// CHECK-NEXT: arith.constant 80 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 40 : index
	// CHECK-NEXT: arith.constant 32 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 40 : index
	// CHECK-NEXT: arith.constant 48 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 40 : index
	// CHECK-NEXT: arith.constant 64 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 40 : index
	// CHECK-NEXT: arith.constant 80 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 48 : index
	// CHECK-NEXT: arith.constant 32 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 48 : index
	// CHECK-NEXT: arith.constant 48 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 48 : index
	// CHECK-NEXT: arith.constant 64 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 48 : index
	// CHECK-NEXT: arith.constant 80 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 56 : index
	// CHECK-NEXT: arith.constant 32 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 56 : index
	// CHECK-NEXT: arith.constant 48 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 56 : index
	// CHECK-NEXT: arith.constant 64 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	// CHECK-NEXT: arith.constant 56 : index
	// CHECK-NEXT: arith.constant 80 : index
	// CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc, boundary_check = true} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
	%1 = xetile.init_tile %a[0, 32] : memref<1024x1024xf32> -> !xetile.tile<8x4x8x16xf32>

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
	%result = arith.constant dense<0.0>: vector<8x4x8x16xf32>


    //CHECK:      xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    //CHECK-NEXT: xegpu.store_nd {{.*}} {mode = vc, {{.*}}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
	xetile.store_tile %result, %1: vector<8x4x8x16xf32>, !xetile.tile<8x4x8x16xf32>
	return
}
