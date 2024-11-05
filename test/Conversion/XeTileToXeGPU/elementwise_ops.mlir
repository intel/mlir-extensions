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

    gpu.func @arith_binary_ops_int() {
        %0 = arith.constant dense<1>: vector<4x4x16x16xi16>
        %1 = arith.constant dense<2>: vector<64x4x1x16xi16>
        %2 = xetile.tile_unpack %0 {inner_blocks = array<i64: 16, 16>}: vector<4x4x16x16xi16> -> vector<64x64xi16>
        %3 = xetile.tile_pack %2 {inner_blocks = array<i64: 1, 16>}: vector<64x64xi16> -> vector<64x4x1x16xi16>
        // CHECK-COUNT-256: arith.addi {{.*}}, {{.*}} : vector<1x16xi16>
        // CHECK-COUNT-256: arith.subi
        // CHECK-COUNT-256: arith.muli
        // CHECK-COUNT-256: arith.maxsi
        // CHECK-COUNT-256: arith.maxui
        // CHECK-COUNT-256: arith.minsi
        // CHECK-COUNT-256: arith.minui
        // CHECK-COUNT-256: arith.divsi
        // CHECK-COUNT-256: arith.divui
        // CHECK-COUNT-256: arith.remsi
        // CHECK-COUNT-256: arith.remui
        // CHECK-COUNT-256: arith.andi
        %result = arith.addi %3, %1 : vector<64x4x1x16xi16>
        %subi_result = arith.subi %3, %1 : vector<64x4x1x16xi16>
        %muli_result = arith.muli %subi_result, %1 : vector<64x4x1x16xi16>
        %maxsi_result = arith.maxsi %muli_result, %1 : vector<64x4x1x16xi16>
        %maxui_result = arith.maxui %muli_result, %1 : vector<64x4x1x16xi16>
        %minsi_result = arith.minsi %maxsi_result, %muli_result : vector<64x4x1x16xi16>
        %minui_result = arith.minui %maxui_result, %muli_result : vector<64x4x1x16xi16>
        %divsi_result = arith.divsi %minui_result, %1 : vector<64x4x1x16xi16>
        %divui_result = arith.divui %minui_result, %1 : vector<64x4x1x16xi16>
        %remsi_result = arith.remsi %minsi_result, %divsi_result : vector<64x4x1x16xi16>
        %remui_result = arith.remui %minui_result, %divui_result : vector<64x4x1x16xi16>
        %and_result = arith.andi %remsi_result, %remui_result : vector<64x4x1x16xi16>
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


    gpu.func @sglevel_and_test(%arg0: memref<1x4096xi8>, %arg1: memref<1x4096xi8>, %arg2: memref<1x4096xi8>) {
      %c0 = arith.constant 0 : index
      %c4096 = arith.constant 4096 : index
      %c32 = arith.constant 32 : index
      %c1024_i32 = arith.constant 1024 : i32
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %block_dim_y = gpu.block_dim  y
      %0 = arith.muli %thread_id_x, %block_dim_y : index
      %1 = arith.addi %0, %thread_id_y : index
      %block_id_x = gpu.block_id  x
      %2 = arith.index_cast %block_id_x : index to i32
      %3 = arith.muli %2, %c1024_i32 : i32
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.remsi %1, %c32 : index
      %6 = arith.muli %5, %c32 : index
      %7 = arith.remsi %6, %c4096 : index
      %8 = arith.addi %7, %4 : index
      %9 = xetile.init_tile %arg0[%c0, %8] : memref<1x4096xi8> -> !xetile.tile<1x32xi8, #xetile.tile_attr<inner_blocks = [1, 32]>>
      %10 = xetile.load_tile %9 {padding = 0 : i32} : !xetile.tile<1x32xi8, #xetile.tile_attr<inner_blocks = [1, 32]>> -> vector<1x1x1x32xi8>
      %11 = xetile.tile_unpack %10 {inner_blocks = array<i64: 1, 32>} : vector<1x1x1x32xi8> -> vector<1x32xi8>
      %12 = xetile.init_tile %arg1[%c0, %8] : memref<1x4096xi8> -> !xetile.tile<1x32xi8, #xetile.tile_attr<inner_blocks = [1, 32]>>
      %13 = xetile.load_tile %12 {padding = 0 : i32} : !xetile.tile<1x32xi8, #xetile.tile_attr<inner_blocks = [1, 32]>> -> vector<1x1x1x32xi8>
      %14 = xetile.tile_unpack %13 {inner_blocks = array<i64: 1, 32>} : vector<1x1x1x32xi8> -> vector<1x32xi8>
      %15 = xetile.tile_pack %11 {inner_blocks = array<i64: 1, 32>} : vector<1x32xi8> -> vector<1x1x1x32xi8>
      %16 = xetile.tile_pack %14 {inner_blocks = array<i64: 1, 32>} : vector<1x32xi8> -> vector<1x1x1x32xi8>
      //CHECK: %{{.*}} = arith.andi %{{.*}}, %{{.*}} : vector<1x32xi8>
      %17 = arith.andi %15, %16 : vector<1x1x1x32xi8>
      %18 = xetile.tile_unpack %17 {inner_blocks = array<i64: 1, 32>} : vector<1x1x1x32xi8> -> vector<1x32xi8>
      %19 = xetile.init_tile %arg2[%c0, %8] : memref<1x4096xi8> -> !xetile.tile<1x32xi8, #xetile.tile_attr<inner_blocks = [1, 32]>>
      %20 = xetile.tile_pack %18 {inner_blocks = array<i64: 1, 32>} : vector<1x32xi8> -> vector<1x1x1x32xi8>
      xetile.store_tile %20,  %19 : vector<1x1x1x32xi8>, !xetile.tile<1x32xi8, #xetile.tile_attr<inner_blocks = [1, 32]>>
      gpu.return
    }
  }
