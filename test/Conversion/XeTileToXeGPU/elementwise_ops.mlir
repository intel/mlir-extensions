// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu %s -verify-diagnostics -o -| FileCheck %s
  gpu.module @test_kernel {
    gpu.func @arith_binary_ops() {
        %0 = arith.constant dense<0.9>: vector<4x4x16x16xf16>
        %1 = arith.constant dense<2.3>: vector<64x4x1x16xf16>
        %2 = xetile.tile_unpack %0 {inner_blocks = [16,16]}: vector<4x4x16x16xf16> -> vector<64x64xf16>
        %3 = xetile.tile_pack %2 { inner_blocks = [1, 16] } : vector<64x64xf16> -> vector<64x4x1x16xf16>
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
        %2 = xetile.tile_unpack %0 {inner_blocks = [16,16]}: vector<4x4x16x16xi16> -> vector<64x64xi16>
        %3 = xetile.tile_pack %2 { inner_blocks = [1, 16] } : vector<64x64xi16> -> vector<64x4x1x16xi16>
        // CHECK-COUNT-256: arith.xori {{.*}}, {{.*}} : vector<1x16xi16>
        %xori_result = arith.xori %3, %1 : vector<64x4x1x16xi16>
        gpu.return
    }

    gpu.func @arith_unary_ops() {
        %0 = arith.constant dense<0.9>: vector<4x4x16x16xf16>
        %1 = arith.constant dense<2.3>: vector<64x4x1x16xf16>
        %2 = xetile.tile_unpack %0 {inner_blocks = [16,16]}: vector<4x4x16x16xf16> -> vector<64x64xf16>
        %3 = xetile.tile_pack %2 { inner_blocks = [1, 16] } : vector<64x64xf16> -> vector<64x4x1x16xf16>
        // CHECK-COUNT-256: arith.addf
        // CHECK-COUNT-256: arith.negf {{.*}} : vector<1x16xf16>
        %result = arith.addf %3, %1 : vector<64x4x1x16xf16>
        %negf_result = arith.negf %result : vector<64x4x1x16xf16>
        gpu.return
    }


    gpu.func @math_binary_ops() {
        %0 = arith.constant dense<0.9>: vector<4x4x16x16xf16>
        %1 = arith.constant dense<2.3>: vector<64x4x1x16xf16>
        %2 = xetile.tile_unpack %0 {inner_blocks = [16,16]}: vector<4x4x16x16xf16> -> vector<64x64xf16>
        %3 = xetile.tile_pack %2 { inner_blocks = [1, 16] } : vector<64x64xf16> -> vector<64x4x1x16xf16>
        // CHECK-COUNT-256: math.powf {{.*}}, {{.*}} : vector<1x16xf16>
        %result = math.powf %3, %1 : vector<64x4x1x16xf16>
        gpu.return
    }

    gpu.func @math_unary_ops() {
        %0 = arith.constant dense<0.9>: vector<4x4x16x16xf16>
        %1 = arith.constant dense<2.3>: vector<64x4x1x16xf16>
        %2 = xetile.tile_unpack %0 {inner_blocks = [16,16]}: vector<4x4x16x16xf16> -> vector<64x64xf16>
        %3 = xetile.tile_pack %2 { inner_blocks = [1, 16] } : vector<64x64xf16> -> vector<64x4x1x16xf16>
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
  }
