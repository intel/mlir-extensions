// RUN: imex-opt --xetile-tiling --convert-xetile-to-xegpu --remove-dead-values %s | FileCheck %s

// CHECK-LABEL: func @test_gemm({{.*}}) {
func.func @test_gemm(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // %c8 = arith.constant 8 : index
  // %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %c1024 = arith.constant 1024 : index
  %block_id_x = gpu.block_id x
  %block_id_y = gpu.block_id y
  %m = arith.muli %block_id_x, %c64 : index
  %n = arith.muli %block_id_y, %c64 : index
  // intialize C tile and load it
  //CHECK: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 0 : index
  //CHECK-NEXT: arith.constant 16 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 0 : index
  //CHECK-NEXT: arith.constant 32 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 0 : index
  //CHECK-NEXT: arith.constant 48 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 8 : index
  //CHECK-NEXT: arith.constant 0 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 8 : index
  //CHECK-NEXT: arith.constant 16 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 8 : index
  //CHECK-NEXT: arith.constant 32 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 8 : index
  //CHECK-NEXT: arith.constant 48 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 16 : index
  //CHECK-NEXT: arith.constant 0 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 16 : index
  //CHECK-NEXT: arith.constant 16 : index
  //CHECK-NEXT: arith.addi %2, %c16_14 : index
  //CHECK-NEXT: arith.addi %3, %c16_15 : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 16 : index
  //CHECK-NEXT: arith.constant 32 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 16 : index
  //CHECK-NEXT: arith.constant 48 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 24 : index
  //CHECK-NEXT: arith.constant 0 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 24 : index
  //CHECK-NEXT: arith.constant 16 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 24 : index
  //CHECK-NEXT: arith.constant 32 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 24 : index
  //CHECK-NEXT: arith.constant 48 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 32 : index
  //CHECK-NEXT: arith.constant 0 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 32 : index
  //CHECK-NEXT: arith.constant 16 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 32 : index
  //CHECK-NEXT: arith.constant 32 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 32 : index
  //CHECK-NEXT: arith.constant 48 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 40 : index
  //CHECK-NEXT: arith.constant 0 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 40 : index
  //CHECK-NEXT: arith.constant 16 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 40 : index
  //CHECK-NEXT: arith.constant 32 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 40 : index
  //CHECK-NEXT: arith.constant 48 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 48 : index
  //CHECK-NEXT: arith.constant 0 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 48 : index
  //CHECK-NEXT: arith.constant 16 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 48 : index
  //CHECK-NEXT: arith.constant 32 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 48 : index
  //CHECK-NEXT: arith.constant 48 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 56 : index
  //CHECK-NEXT: arith.constant 0 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 56 : index
  //CHECK-NEXT: arith.constant 16 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 56 : index
  //CHECK-NEXT: arith.constant 32 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK-NEXT: arith.constant 56 : index
  //CHECK-NEXT: arith.constant 48 : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: arith.addi {{.*}} : index
  //CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  %c_init_tile = xetile.init_tile %C[%m, %n] : memref<1024x1024xf32> -> !xetile.tile<64x64xf32>
  //CHECK: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, {{.*}}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<64x64xf32> -> vector<64x64xf32>
  // initalize A and B tiles
  // CHECK:  arith.constant 0 : index
  // CHECK-NEXT:  arith.constant 0 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 0 : index
  // CHECK-NEXT:  arith.constant 16 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 0 : index
  // CHECK-NEXT:  arith.constant 32 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 0 : index
  // CHECK-NEXT:  arith.constant 48 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 8 : index
  // CHECK-NEXT:  arith.constant 0 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 8 : index
  // CHECK-NEXT:  arith.constant 16 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 8 : index
  // CHECK-NEXT:  arith.constant 32 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 8 : index
  // CHECK-NEXT:  arith.constant 48 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 16 : index
  // CHECK-NEXT:  arith.constant 0 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 16 : index
  // CHECK-NEXT:  arith.constant 16 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 16 : index
  // CHECK-NEXT:  arith.constant 32 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 16 : index
  // CHECK-NEXT:  arith.constant 48 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 24 : index
  // CHECK-NEXT:  arith.constant 0 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 24 : index
  // CHECK-NEXT:  arith.constant 16 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 24 : index
  // CHECK-NEXT:  arith.constant 32 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 24 : index
  // CHECK-NEXT:  arith.constant 48 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 32 : index
  // CHECK-NEXT:  arith.constant 0 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 32 : index
  // CHECK-NEXT:  arith.constant 16 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 32 : index
  // CHECK-NEXT:  arith.constant 32 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 32 : index
  // CHECK-NEXT:  arith.constant 48 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 40 : index
  // CHECK-NEXT:  arith.constant 0 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 40 : index
  // CHECK-NEXT:  arith.constant 16 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 40 : index
  // CHECK-NEXT:  arith.constant 32 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 40 : index
  // CHECK-NEXT:  arith.constant 48 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 48 : index
  // CHECK-NEXT:  arith.constant 0 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 48 : index
  // CHECK-NEXT:  arith.constant 16 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 48 : index
  // CHECK-NEXT:  arith.constant 32 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 48 : index
  // CHECK-NEXT:  arith.constant 48 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 56 : index
  // CHECK-NEXT:  arith.constant 0 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 56 : index
  // CHECK-NEXT:  arith.constant 16 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 56 : index
  // CHECK-NEXT:  arith.constant 32 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK-NEXT:  arith.constant 56 : index
  // CHECK-NEXT:  arith.constant 48 : index
  // CHECK-NEXT:  arith.addi {{.*}} : index
  // CHECK-NEXT:  xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
  %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<1024x1024xf16> -> !xetile.tile<64x64xf16>
  // CHECK: arith.constant 0 : index
  // CHECK-NEXT: arith.constant 0 : index
  // CHECK-NEXT: arith.addi {{.*}} : index
  // CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
  // CHECK-NEXT: arith.constant 0 : index
  // CHECK-NEXT: arith.constant 16 : index
  // CHECK-NEXT: arith.addi {{.*}} : index
  // CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
  // CHECK-NEXT: arith.constant 0 : index
  // CHECK-NEXT: arith.constant 32 : index
  // CHECK-NEXT: arith.addi {{.*}} : index
  // CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
  // CHECK-NEXT: arith.constant 0 : index
  // CHECK-NEXT: arith.constant 48 : index
  // CHECK-NEXT: arith.addi {{.*}} : index
  // CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
  // CHECK-NEXT: arith.constant 16 : index
  // CHECK-NEXT: arith.constant 0 : index
  // CHECK-NEXT: arith.addi {{.*}} : index
  // CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
  // CHECK-NEXT: arith.constant 16 : index
  // CHECK-NEXT: arith.constant 16 : index
  // CHECK-NEXT: arith.addi {{.*}} : index
  // CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
  // CHECK-NEXT: arith.constant 16 : index
  // CHECK-NEXT: arith.constant 32 : index
  // CHECK-NEXT: arith.addi {{.*}} : index
  // CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
  // CHECK-NEXT: arith.constant 16 : index
  // CHECK-NEXT: arith.constant 48 : index
  // CHECK-NEXT: arith.addi {{.*}} : index
  // CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
  // CHECK-NEXT: arith.constant 32 : index
  // CHECK-NEXT: arith.constant 0 : index
  // CHECK-NEXT: arith.addi {{.*}} : index
  // CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
  // CHECK-NEXT: arith.constant 32 : index
  // CHECK-NEXT: arith.constant 16 : index
  // CHECK-NEXT: arith.addi {{.*}} : index
  // CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
  // CHECK-NEXT: arith.constant 32 : index
  // CHECK-NEXT: arith.constant 32 : index
  // CHECK-NEXT: arith.addi {{.*}} : index
  // CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
  // CHECK-NEXT: arith.constant 32 : index
  // CHECK-NEXT: arith.constant 48 : index
  // CHECK-NEXT: arith.addi {{.*}} : index
  // CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
  // CHECK-NEXT: arith.constant 48 : index
  // CHECK-NEXT: arith.constant 0 : index
  // CHECK-NEXT: arith.addi {{.*}} : index
  // CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
  // CHECK-NEXT: arith.constant 48 : index
  // CHECK-NEXT: arith.constant 16 : index
  // CHECK-NEXT: arith.addi {{.*}} : index
  // CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
  // CHECK-NEXT: arith.constant 48 : index
  // CHECK-NEXT: arith.constant 32 : index
  // CHECK-NEXT: arith.addi {{.*}} : index
  // CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
  // CHECK-NEXT: arith.constant 48 : index
  // CHECK-NEXT: arith.constant 48 : index
  // CHECK-NEXT: arith.addi {{.*}} : index
  // CHECK-NEXT: xegpu.create_nd_tdesc {{.*}} {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
  %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<1024x1024xf16> -> !xetile.tile<64x64xf16>
  // compute the value of C tile by iterating over tiles in k-dimension and doing dpas
  // CHECK: scf.for
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>,
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>,
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>,
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>,
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>,
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>,
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>,
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>,
  // CHECK-SAME: !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>,
  // CHECK-SAME: !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>,
  // CHECK-SAME: !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>,
  // CHECK-SAME: !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>,
  // CHECK-SAME: vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>,
  // CHECK-SAME: vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>,
  // CHECK-SAME: vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>,
  // CHECK-SAME: vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>,
  // CHECK-SAME: vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>
  %out:3 = scf.for %k = %c0 to %c1024 step %c64
    iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
    -> (!xetile.tile<64x64xf16>, !xetile.tile<64x64xf16>, vector<64x64xf32>) {

    // load A and B tiles
    //CHECK: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 1, {{.*}}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
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
    %a_value = xetile.load_tile %a_tile : !xetile.tile<64x64xf16> -> vector<64x64xf16>
    // CHECK: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    // CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    // CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    // CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    // CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    // CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    // CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    // CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    // CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    // CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    // CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    // CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    // CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    // CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    // CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    // CHECK-NEXT: xegpu.load_nd {{.*}} {mode = vc, vnni_axis = 0, {{.*}}} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    %b_value = xetile.load_tile %b_tile : !xetile.tile<64x64xf16> -> vector<64x64xf16>
    // perform dpas and accumulate
    // CHECK: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK-NEXT: xegpu.dpas {{.*}} {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value : vector<64x64xf16>, vector<64x64xf16>, vector<64x64xf32> -> vector<64x64xf32>
    // update the offsets for A and B tiles
    // CHECK: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c64]
      : !xetile.tile<64x64xf16>, index, index -> !xetile.tile<64x64xf16>

    //CHECK: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK-NEXT: xegpu.update_nd_offset {{.*}} {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
    %b_next_tile = xetile.update_tile_offset %b_tile, [%c64, %c0]
      : !xetile.tile<64x64xf16>, index, index -> !xetile.tile<64x64xf16>
    // partial C tile result
    // CHECK: scf.yield
    // CHECK-SAME: !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>,
    // CHECK-SAME: !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>,
    // CHECK-SAME: !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>,
    // CHECK-SAME: !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>,
    // CHECK-SAME: !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>,
    // CHECK-SAME: !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>,
    // CHECK-SAME: !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>,
    // CHECK-SAME: !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>,
    // CHECK-SAME: !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>,
    // CHECK-SAME: !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>,
    // CHECK-SAME: !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>,
    // CHECK-SAME: !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>,
    // CHECK-SAME: vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>,
    // CHECK-SAME: vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>,
    // CHECK-SAME: vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>,
    // CHECK-SAME: vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>,
    // CHECK-SAME: vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>
    scf.yield %a_next_tile, %b_next_tile, %c_new_value
      : !xetile.tile<64x64xf16>, !xetile.tile<64x64xf16>, vector<64x64xf32>
  }
  // store the final accumulated C tile result back to memory
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
  xetile.store_tile %out#2, %c_init_tile: vector<64x64xf32>, !xetile.tile<64x64xf32>
  return
}
