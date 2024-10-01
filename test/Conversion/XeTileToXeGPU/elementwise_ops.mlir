// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu %s -verify-diagnostics -o -| FileCheck %s
  gpu.module @test_kernel {
    gpu.func @arith_binary_ops() {
        %0 = arith.constant dense<0.9>: vector<4x4x16x16xf16>
        %1 = arith.constant dense<2.3>: vector<64x4x1x16xf16>
        %2 = xetile.tile_unpack %0 {inner_blocks = array<i64: 16, 16>}: vector<4x4x16x16xf16> -> vector<64x64xf16>
        %3 = xetile.tile_pack %2 {inner_blocks = array<i64: 1, 16>}: vector<64x64xf16> -> vector<64x4x1x16xf16>
        // CHECK-COUNT-256: arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
        // CHECK-COUNT-256: arith.sub
        // CHECK-COUNT-256: arith.mulf
        // CHECK-COUNT-256: arith.maximumf
        // CHECK-COUNT-256: arith.minimumf
        // CHECK-COUNT-256: arith.divf
        // CHECK-COUNT-256: arith.remf
        %result = arith.addf %3, %1 : vector<64x4x1x16xf16>
        %subf_result = arith.subf %result, %1 : vector<64x4x1x16xf16>
        %mulf_result = arith.mulf %subf_result, %1 : vector<64x4x1x16xf16>
        %maxf_result = arith.maximumf %mulf_result, %1 : vector<64x4x1x16xf16>
        %minf_result = arith.minimumf %maxf_result, %mulf_result : vector<64x4x1x16xf16>
        %divf_result = arith.divf %minf_result, %1 : vector<64x4x1x16xf16>
        %remf_result = arith.remf %minf_result, %divf_result : vector<64x4x1x16xf16>
        gpu.return
    }

    gpu.func @arith_xori_ops() {
        %0 = arith.constant dense<1>: vector<4x4x16x16xi16>
        %1 = arith.constant dense<2>: vector<64x4x1x16xi16>
        %2 = xetile.tile_unpack %0 {inner_blocks = array<i64: 16, 16>}: vector<4x4x16x16xi16> -> vector<64x64xi16>
        %3 = xetile.tile_pack %2 {inner_blocks = array<i64: 1, 16>}: vector<64x64xi16> -> vector<64x4x1x16xi16>
        // CHECK-COUNT-256: arith.xori {{.*}}, {{.*}} : vector<1x16xi16>
        %xori_result = arith.xori %3, %1 : vector<64x4x1x16xi16>
        gpu.return
    }

    gpu.func @arith_unary_ops() {
        %0 = arith.constant dense<0.9>: vector<4x4x16x16xf16>
        %1 = arith.constant dense<2.3>: vector<64x4x1x16xf16>
        %2 = xetile.tile_unpack %0 {inner_blocks = array<i64: 16, 16>}: vector<4x4x16x16xf16> -> vector<64x64xf16>
        %3 = xetile.tile_pack %2 {inner_blocks = array<i64: 1, 16>}: vector<64x64xf16> -> vector<64x4x1x16xf16>
        // CHECK-COUNT-256: arith.addf
        // CHECK-COUNT-256: arith.negf {{.*}} : vector<1x16xf16>
        %result = arith.addf %3, %1 : vector<64x4x1x16xf16>
        %negf_result = arith.negf %result : vector<64x4x1x16xf16>
        gpu.return
    }


    gpu.func @math_binary_ops() {
        %0 = arith.constant dense<0.9>: vector<4x4x16x16xf16>
        %1 = arith.constant dense<2.3>: vector<64x4x1x16xf16>
        %2 = xetile.tile_unpack %0 {inner_blocks = array<i64: 16, 16>}: vector<4x4x16x16xf16> -> vector<64x64xf16>
        %3 = xetile.tile_pack %2 {inner_blocks = array<i64: 1, 16>}: vector<64x64xf16> -> vector<64x4x1x16xf16>
        // CHECK-COUNT-256: math.powf {{.*}}, {{.*}} : vector<1x16xf16>
        %result = math.powf %3, %1 : vector<64x4x1x16xf16>
        gpu.return
    }

    gpu.func @math_unary_ops() {
        %0 = arith.constant dense<0.9>: vector<4x4x16x16xf16>
        %1 = arith.constant dense<2.3>: vector<64x4x1x16xf16>
        %2 = xetile.tile_unpack %0 {inner_blocks = array<i64: 16, 16>}: vector<4x4x16x16xf16> -> vector<64x64xf16>
        %3 = xetile.tile_pack %2 {inner_blocks = array<i64: 1, 16>}: vector<64x64xf16> -> vector<64x4x1x16xf16>
        // CHECK-COUNT-256: math.exp {{.*}} : vector<1x16xf16>
        // CHECK-COUNT-256: math.sin
        // CHECK-COUNT-256: math.cos
        // CHECK-COUNT-256: math.tanh
        // CHECK-COUNT-256: math.sqrt
        // CHECK-COUNT-256: math.log
        // CHECK-COUNT-256: math.rsqrt
        // CHECK-COUNT-256: math.erf
        %result = arith.addf %3, %1 : vector<64x4x1x16xf16>
        %exp_result = math.exp %result : vector<64x4x1x16xf16>
        %sin_result = math.sin %exp_result : vector<64x4x1x16xf16>
        %cos_result = math.cos %sin_result : vector<64x4x1x16xf16>
        %tan_result = math.tanh %cos_result : vector<64x4x1x16xf16>
        %sqrt_result = math.sqrt %tan_result : vector<64x4x1x16xf16>
        %log_result = math.log %sqrt_result : vector<64x4x1x16xf16>
        %rsqrt_result = math.rsqrt %log_result : vector<64x4x1x16xf16>
        %erf_result = math.erf %rsqrt_result : vector<64x4x1x16xf16>
        gpu.return
    }

    gpu.func @sglevel_type_cast(%arg0: memref<1024x1024xf16>) {
      %0 = xetile.init_tile %arg0[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
      %1 = xetile.load_tile %0 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x1x32x32xf16>
      %2 = xetile.tile_unpack %1 {inner_blocks = array<i64: 32, 32>}  : vector<1x1x32x32xf16> -> vector<32x32xf16>
      %3 = xetile.tile_pack %2 {inner_blocks = array<i64: 1, 16>}: vector<32x32xf16> -> vector<32x2x1x16xf16>

      //CHECK-COUNT-64: arith.extf {{.*}} : vector<1x16xf16> to vector<1x16xf32>
      %4 = arith.extf %3 : vector<32x2x1x16xf16> to vector<32x2x1x16xf32>

      //CHECK-COUNT-64: math.exp {{.*}} : vector<1x16xf32>
      %5 = math.exp %4 : vector<32x2x1x16xf32>

      //CHECK-COUNT-64: arith.truncf {{.*}} : vector<1x16xf32> to vector<1x16xf16>
      %6 = arith.truncf %5 : vector<32x2x1x16xf32> to vector<32x2x1x16xf16>

      %7 = xetile.tile_unpack %6 {inner_blocks = array<i64: 1, 16>}: vector<32x2x1x16xf16> -> vector<32x32xf16>
      %8 = xetile.init_tile %arg0[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
      %9 = xetile.tile_pack %7 {inner_blocks = array<i64: 8, 32>}: vector<32x32xf16> -> vector<4x1x8x32xf16>
      xetile.store_tile %9,  %8 : vector<4x1x8x32xf16>, !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
      gpu.return
    }

    gpu.func @sglevel_extsi_test(%arg0: memref<32x32xi16>, %arg1: memref<32x32xi32>) {
      %0 = xetile.init_tile %arg0[0, 0] : memref<32x32xi16> -> !xetile.tile<32x32xi16, #xetile.tile_attr<inner_blocks = [32, 32]>>
      %1 = xetile.load_tile %0 { padding = 0 : i32 }  : !xetile.tile<32x32xi16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x1x32x32xi16>
      %2 = xetile.tile_unpack %1 {inner_blocks = array<i64: 32, 32>}  : vector<1x1x32x32xi16> -> vector<32x32xi16>
      %3 = xetile.tile_pack %2 {inner_blocks = array<i64: 1, 16>}: vector<32x32xi16> -> vector<32x2x1x16xi16>

      //CHECK-COUNT-64: arith.extsi {{.*}} : vector<1x16xi16> to vector<1x16xi32>
      %4 = arith.extsi %3 : vector<32x2x1x16xi16> to vector<32x2x1x16xi32>

      %5 = xetile.tile_unpack %4 {inner_blocks = array<i64: 1, 16>}: vector<32x2x1x16xi32> -> vector<32x32xi32>
      %6 = xetile.init_tile %arg1[0, 0] : memref<32x32xi32> -> !xetile.tile<32x32xi32, #xetile.tile_attr<inner_blocks = [8, 16]>>
      %7 = xetile.tile_pack %5 {inner_blocks = array<i64: 8, 16>}  : vector<32x32xi32> -> vector<4x2x8x16xi32>
      xetile.store_tile %7,  %6 : vector<4x2x8x16xi32>, !xetile.tile<32x32xi32, #xetile.tile_attr<inner_blocks = [8, 16]>>
      gpu.return
    }

    gpu.func @sglevel_extui_test(%arg0: memref<32x32xi16>, %arg1: memref<32x32xi32>) {
      %0 = xetile.init_tile %arg0[0, 0] : memref<32x32xi16> -> !xetile.tile<32x32xi16, #xetile.tile_attr<inner_blocks = [32, 32]>>
      %1 = xetile.load_tile %0 { padding = 0 : i32 }  : !xetile.tile<32x32xi16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x1x32x32xi16>
      %2 = xetile.tile_unpack %1 {inner_blocks = array<i64: 32, 32>}  : vector<1x1x32x32xi16> -> vector<32x32xi16>
      %3 = xetile.tile_pack %2 {inner_blocks = array<i64: 1, 16>}: vector<32x32xi16> -> vector<32x2x1x16xi16>

      //CHECK-COUNT-64: arith.extui {{.*}} : vector<1x16xi16> to vector<1x16xi32>
      %4 = arith.extui %3 : vector<32x2x1x16xi16> to vector<32x2x1x16xi32>

      %5 = xetile.tile_unpack %4 {inner_blocks = array<i64: 1, 16>}: vector<32x2x1x16xi32> -> vector<32x32xi32>
      %6 = xetile.init_tile %arg1[0, 0] : memref<32x32xi32> -> !xetile.tile<32x32xi32, #xetile.tile_attr<inner_blocks = [8, 16]>>
      %7 = xetile.tile_pack %5 {inner_blocks = array<i64: 8, 16>}  : vector<32x32xi32> -> vector<4x2x8x16xi32>
      xetile.store_tile %7,  %6 : vector<4x2x8x16xi32>, !xetile.tile<32x32xi32, #xetile.tile_attr<inner_blocks = [8, 16]>>
      gpu.return
    }

    gpu.func @sglevel_fptosi_test(%arg0: memref<32x32xf16>, %arg1: memref<32x32xi32>) {
      %0 = xetile.init_tile %arg0[0, 0] : memref<32x32xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
      %1 = xetile.load_tile %0 { padding = 0.0 : f32 }  : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x1x32x32xf16>
      %2 = xetile.tile_unpack %1 {inner_blocks = array<i64: 32, 32>}  : vector<1x1x32x32xf16> -> vector<32x32xf16>
      %3 = xetile.tile_pack %2 {inner_blocks = array<i64: 1, 16>}: vector<32x32xf16> -> vector<32x2x1x16xf16>

      //CHECK-COUNT-64: arith.fptosi {{.*}} : vector<1x16xf16> to vector<1x16xi32>
      %4 = arith.fptosi %3 : vector<32x2x1x16xf16> to vector<32x2x1x16xi32>

      %5 = xetile.tile_unpack %4 {inner_blocks = array<i64: 1, 16>}: vector<32x2x1x16xi32> -> vector<32x32xi32>
      %6 = xetile.init_tile %arg1[0, 0] : memref<32x32xi32> -> !xetile.tile<32x32xi32, #xetile.tile_attr<inner_blocks = [8, 16]>>
      %7 = xetile.tile_pack %5 {inner_blocks = array<i64: 8, 16>}  : vector<32x32xi32> -> vector<4x2x8x16xi32>
      xetile.store_tile %7,  %6 : vector<4x2x8x16xi32>, !xetile.tile<32x32xi32, #xetile.tile_attr<inner_blocks = [8, 16]>>
      gpu.return
    }

    gpu.func @sglevel_fptoui_test(%arg0: memref<32x32xf16>, %arg1: memref<32x32xi32>) {
      %0 = xetile.init_tile %arg0[0, 0] : memref<32x32xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
      %1 = xetile.load_tile %0 { padding = 0.0 : f32 }  : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x1x32x32xf16>
      %2 = xetile.tile_unpack %1 {inner_blocks = array<i64: 32, 32>}  : vector<1x1x32x32xf16> -> vector<32x32xf16>
      %3 = xetile.tile_pack %2 {inner_blocks = array<i64: 1, 16>}: vector<32x32xf16> -> vector<32x2x1x16xf16>

      //CHECK-COUNT-64: arith.fptoui {{.*}} : vector<1x16xf16> to vector<1x16xi32>
      %4 = arith.fptoui %3 : vector<32x2x1x16xf16> to vector<32x2x1x16xi32>

      %5 = xetile.tile_unpack %4 {inner_blocks = array<i64: 1, 16>}: vector<32x2x1x16xi32> -> vector<32x32xi32>
      %6 = xetile.init_tile %arg1[0, 0] : memref<32x32xi32> -> !xetile.tile<32x32xi32, #xetile.tile_attr<inner_blocks = [8, 16]>>
      %7 = xetile.tile_pack %5 {inner_blocks = array<i64: 8, 16>}  : vector<32x32xi32> -> vector<4x2x8x16xi32>
      xetile.store_tile %7,  %6 : vector<4x2x8x16xi32>, !xetile.tile<32x32xi32, #xetile.tile_attr<inner_blocks = [8, 16]>>
      gpu.return
    }

    gpu.func @sglevel_sitofp_test(%arg0: memref<32x32xi32>, %arg1: memref<32x32xf32>) {
      %0 = xetile.init_tile %arg0[0, 0] : memref<32x32xi32> -> !xetile.tile<32x32xi32, #xetile.tile_attr<inner_blocks = [32, 16]>>
      %1 = xetile.load_tile %0 { padding = 0 : i32 }  : !xetile.tile<32x32xi32, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<1x2x32x16xi32>
      %2 = xetile.tile_unpack %1 {inner_blocks = array<i64: 32, 16>}  : vector<1x2x32x16xi32> -> vector<32x32xi32>
      %3 = xetile.tile_pack %2 {inner_blocks = array<i64: 1, 16>}: vector<32x32xi32> -> vector<32x2x1x16xi32>
      //CHECK-COUNT-64: arith.sitofp {{.*}} : vector<1x16xi32> to vector<1x16xf32>
      %4 = arith.sitofp %3 : vector<32x2x1x16xi32> to vector<32x2x1x16xf32>
      %5 = xetile.tile_unpack %4 {inner_blocks = array<i64: 1, 16>}: vector<32x2x1x16xf32> -> vector<32x32xf32>
      %6 = xetile.init_tile %arg1[0, 0] : memref<32x32xf32> -> !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
      %7 = xetile.tile_pack %5 {inner_blocks = array<i64: 8, 16>}  : vector<32x32xf32> -> vector<4x2x8x16xf32>
      xetile.store_tile %7,  %6 : vector<4x2x8x16xf32>, !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
      gpu.return
    }

    gpu.func @sglevel_uitofp_test(%arg0: memref<32x32xi32>, %arg1: memref<32x32xf32>) {
      %0 = xetile.init_tile %arg0[0, 0] : memref<32x32xi32> -> !xetile.tile<32x32xi32, #xetile.tile_attr<inner_blocks = [32, 16]>>
      %1 = xetile.load_tile %0 { padding = 0 : i32 }  : !xetile.tile<32x32xi32, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<1x2x32x16xi32>
      %2 = xetile.tile_unpack %1 {inner_blocks = array<i64: 32, 16>}  : vector<1x2x32x16xi32> -> vector<32x32xi32>
      %3 = xetile.tile_pack %2 {inner_blocks = array<i64: 1, 16>}: vector<32x32xi32> -> vector<32x2x1x16xi32>
      //CHECK-COUNT-64: arith.uitofp {{.*}} : vector<1x16xi32> to vector<1x16xf32>
      %4 = arith.uitofp %3 : vector<32x2x1x16xi32> to vector<32x2x1x16xf32>
      %5 = xetile.tile_unpack %4 {inner_blocks = array<i64: 1, 16>}: vector<32x2x1x16xf32> -> vector<32x32xf32>
      %6 = xetile.init_tile %arg1[0, 0] : memref<32x32xf32> -> !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
      %7 = xetile.tile_pack %5 {inner_blocks = array<i64: 8, 16>}  : vector<32x32xf32> -> vector<4x2x8x16xf32>
      xetile.store_tile %7,  %6 : vector<4x2x8x16xf32>, !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
      gpu.return
    }

    gpu.func @sglevel_truncf_test(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf16>) {
      %0 = xetile.init_tile %arg0[0, 0] : memref<32x32xf32> -> !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [32, 16]>>
      %1 = xetile.load_tile %0 { padding = 0.0 : f32 }  : !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<1x2x32x16xf32>
      %2 = xetile.tile_unpack %1 {inner_blocks = array<i64: 32, 16>}  : vector<1x2x32x16xf32> -> vector<32x32xf32>
      %3 = xetile.tile_pack %2 {inner_blocks = array<i64: 1, 16>}: vector<32x32xf32> -> vector<32x2x1x16xf32>
      //CHECK-COUNT-64: arith.truncf {{.*}} : vector<1x16xf32> to vector<1x16xf16>
      %4 = arith.truncf %3 : vector<32x2x1x16xf32> to vector<32x2x1x16xf16>
      %5 = xetile.tile_unpack %4 {inner_blocks = array<i64: 1, 16>}: vector<32x2x1x16xf16> -> vector<32x32xf16>
      %6 = xetile.init_tile %arg1[0, 0] : memref<32x32xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [8, 16]>>
      %7 = xetile.tile_pack %5 {inner_blocks = array<i64: 8, 16>}  : vector<32x32xf16> -> vector<4x2x8x16xf16>
      xetile.store_tile %7,  %6 : vector<4x2x8x16xf16>, !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [8, 16]>>
      gpu.return
    }

    gpu.func @sglevel_trunci_test(%arg0: memref<32x32xi32>, %arg1: memref<32x32xi16>) {
      %0 = xetile.init_tile %arg0[0, 0] : memref<32x32xi32> -> !xetile.tile<32x32xi32, #xetile.tile_attr<inner_blocks = [32, 16]>>
      %1 = xetile.load_tile %0 { padding = 0 : i32 }  : !xetile.tile<32x32xi32, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<1x2x32x16xi32>
      %2 = xetile.tile_unpack %1 {inner_blocks = array<i64: 32, 16>}  : vector<1x2x32x16xi32> -> vector<32x32xi32>
      %3 = xetile.tile_pack %2 {inner_blocks = array<i64: 1, 16>}: vector<32x32xi32> -> vector<32x2x1x16xi32>
      //CHECK-COUNT-64: arith.trunci {{.*}} : vector<1x16xi32> to vector<1x16xi16>
      %4 = arith.trunci %3 : vector<32x2x1x16xi32> to vector<32x2x1x16xi16>
      %5 = xetile.tile_unpack %4 {inner_blocks = array<i64: 1, 16>}: vector<32x2x1x16xi16> -> vector<32x32xi16>
      %6 = xetile.init_tile %arg1[0, 0] : memref<32x32xi16> -> !xetile.tile<32x32xi16, #xetile.tile_attr<inner_blocks = [8, 16]>>
      %7 = xetile.tile_pack %5 {inner_blocks = array<i64: 8, 16>}  : vector<32x32xi16> -> vector<4x2x8x16xi16>
      xetile.store_tile %7,  %6 : vector<4x2x8x16xi16>, !xetile.tile<32x32xi16, #xetile.tile_attr<inner_blocks = [8, 16]>>
      gpu.return
    }
  }
