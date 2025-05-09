// RUN: imex-opt --split-input-file --xetile-wg-to-sg %s -verify-diagnostics | FileCheck %s

gpu.module @test_arith_extf {
    gpu.func @test_kernel(%arg0: memref<128x32xf16>) {
        %c0 = arith.constant 0 : index
        %tile = xetile.init_tile %arg0[%c0, %c0] : memref<128x32xf16> -> !xetile.tile<128x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
        %load_tile = xetile.load_tile %tile : !xetile.tile<128x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>> -> vector<128x32xf16>
        //CHECK: arith.extf {{%.*}} : vector<32x32xf16> to vector<32x32xf32>
        //CHECK: arith.fptosi {{%.*}} : vector<32x32xf16> to vector<32x32xi16>
        //CHECK: arith.fptoui {{%.*}} : vector<32x32xf16> to vector<32x32xi16>
        //CHECK: arith.bitcast {{%.*}} : vector<32x32xf16> to vector<32x32xi16>
        %extf = arith.extf %load_tile {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xf16> to vector<128x32xf32>
        %fptosi = arith.fptosi %load_tile {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xf16> to vector<128x32xi16>
        %fptoui = arith.fptoui %load_tile {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xf16> to vector<128x32xi16>
        %bitcast = arith.bitcast %load_tile {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xf16> to vector<128x32xi16>
        gpu.return
    }

     gpu.func @test_truncf(%arg0: memref<128x32xf32>) {
        %c0 = arith.constant 0 : index
        %tile = xetile.init_tile %arg0[%c0, %c0] : memref<128x32xf32> -> !xetile.tile<128x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
        %load_tile = xetile.load_tile %tile : !xetile.tile<128x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>> -> vector<128x32xf32>
        //CHECK: arith.truncf {{%.*}} : vector<32x32xf32> to vector<32x32xf16>
        %trucf = arith.truncf %load_tile {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xf32> to vector<128x32xf16>
        gpu.return
    }

    gpu.func @test_extsi(%arg0: memref<128x32xi16>) {
        %c0 = arith.constant 0 : index
        %tile = xetile.init_tile %arg0[%c0, %c0] : memref<128x32xi16> -> !xetile.tile<128x32xi16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>>>
        %load_tile = xetile.load_tile %tile : !xetile.tile<128x32xi16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>>> -> vector<128x32xi16>
        //CHECK: arith.extsi {{%.*}} : vector<32x32xi16> to vector<32x32xi32>
        //CHECK: arith.extui {{%.*}} : vector<32x32xi16> to vector<32x32xi32>
        //CHECK: arith.uitofp {{%.*}} : vector<32x32xi16> to vector<32x32xf16>
        //CHECK: arith.sitofp {{%.*}} : vector<32x32xi16> to vector<32x32xf16>
        //CHECK: arith.bitcast {{%.*}} : vector<32x32xi16> to vector<32x32xf16>
        //CHECK: arith.index_castui {{%.*}} : vector<32x32xi16> to vector<32x32xindex>
        //CHECK: arith.index_cast {{%.*}} : vector<32x32xi16> to vector<32x32xindex>
        %extsi = arith.extsi %load_tile {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xi16> to vector<128x32xi32>
        %extui = arith.extui %load_tile {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xi16> to vector<128x32xi32>
        %uitofp = arith.uitofp %load_tile {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xi16> to vector<128x32xf16>
        %sitofp = arith.sitofp %load_tile {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xi16> to vector<128x32xf16>
        %bitcast = arith.bitcast %load_tile {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xi16> to vector<128x32xf16>
        %uitoindex = arith.index_castui %load_tile {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xi16> to vector<128x32xindex>
        %itoindex = arith.index_cast %load_tile {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xi16> to vector<128x32xindex>
        gpu.return
    }

    gpu.func @test_trunci(%arg0: memref<128x32xi32>) {
        %c0 = arith.constant 0 : index
        %tile = xetile.init_tile %arg0[%c0, %c0] : memref<128x32xi32> -> !xetile.tile<128x32xi32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>>>
        %load_tile = xetile.load_tile %tile : !xetile.tile<128x32xi32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>>> -> vector<128x32xi32>
        //CHECK: arith.trunci {{%.*}} : vector<32x32xi32> to vector<32x32xi16>
        %truci = arith.trunci %load_tile {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xi32> to vector<128x32xi16>
        gpu.return
    }

    gpu.func @test_reduction_and_shape_cast(%arg0 : vector<256x128xf32>) {
        //CHECK: %[[CST_0:.*]] = arith.constant dense<-1.000000e+00> : vector<32x32xf32>
        //CHECK: %[[CST_1:.*]] = arith.constant dense<0.000000e+00> : vector<1x32xf32>
        //CHECK: %[[SQRT:.*]] = math.sqrt %[[CST_0]] : vector<32x32xf32>
        //CHECK: %[[SHAPECAST_0:.*]] = vector.shape_cast %[[CST_1]] : vector<1x32xf32> to vector<32xf32>
        //CHECK: %[[REDUCTION_0:.*]] = vector.multi_reduction <add>, %[[SQRT]], %[[SHAPECAST_0]] [0] : vector<32x32xf32> to vector<32xf32>
        //CHECK: %[[SHAPECAST_1:.*]] = vector.shape_cast %[[REDUCTION_0]] : vector<32xf32> to vector<1x32xf32>
        %cst = arith.constant {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>} dense<-1.0> : vector<256x128xf32>
        %cst_0 = arith.constant {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [1, 32]>} dense<0.000000e+00> : vector<8x128xf32>
        %sqrt = math.sqrt %cst {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>} : vector<256x128xf32>
        %reshape = vector.shape_cast %sqrt {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>} : vector<256x128xf32> to vector<8x32x128xf32>
        %reduction = vector.multi_reduction <add>, %reshape, %cst_0 {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [1, 32]>} [1] : vector<8x32x128xf32> to vector<8x128xf32>
        //CHECK: xetile.store_tile  {{%.*}}, {{%.*}} : vector<1x32xf32>, !xetile.tile<1x32xf32, #xetile.tile_attr<memory_space = 3 : i32>>
        //CHECK: gpu.barrier
        //CHECK: %[[LOADTILE_SLM:.*]] = xetile.load_tile {{%.*}} : !xetile.tile<8x4xf32, #xetile.tile_attr<memory_space = 3 : i32>> -> vector<8x4xf32>
        //CHECK: %[[CST_2:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
        //CHECK: %[[REDUCTION_1:.*]] = vector.multi_reduction <add>, %[[LOADTILE_SLM]], %[[CST_2]] [0] : vector<8x4xf32> to vector<4xf32>
        //CHECK: %[[SHAPECAST_2:.*]] = vector.shape_cast %[[REDUCTION_1]] : vector<4xf32> to vector<1x4xf32>
        %conv_layout = xetile.convert_layout %reduction {wg_map_result = #xetile.wg_map<sg_layout = [1, 32], sg_data = [8, 4]>} : vector<8x128xf32>
        %cst_1 = arith.constant {map = #xetile.wg_map<sg_layout = [1, 32], sg_data = [1, 4]>} dense<0.000000e+00> : vector<128xf32>
        %reduce = vector.multi_reduction <add>, %conv_layout, %cst_1 {map = #xetile.wg_map<sg_layout = [1, 32], sg_data = [1, 4]>} [0] : vector<8x128xf32> to vector<128xf32>
        %shape_cast = vector.shape_cast %reduce {map = #xetile.wg_map<sg_layout = [1, 32], sg_data = [1, 4]>} : vector<128xf32> to vector<1x128xf32>
        gpu.return
    }



    gpu.func @test_cmpi_vector(%arg0 : memref<128x32xi64>, %arg1 : memref<128x32xi64>) {
        %c0 = arith.constant 0 : index
        %tile = xetile.init_tile %arg0[%c0, %c0] : memref<128x32xi64> -> !xetile.tile<128x32xi64, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
        %load_tile_0 = xetile.load_tile %tile : !xetile.tile<128x32xi64, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>> -> vector<128x32xi64>
        %tile_1 = xetile.init_tile %arg1[%c0, %c0] : memref<128x32xi64> -> !xetile.tile<128x32xi64, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
        %load_tile_1 = xetile.load_tile %tile : !xetile.tile<128x32xi64, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>> -> vector<128x32xi64>
        //CHECK: arith.cmpi ult, {{%.*}}, {{%.*}} : vector<32x32xi64>
        %cmpi = arith.cmpi ult, %load_tile_0, %load_tile_1 {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xi64>
        gpu.return
    }

    gpu.func @test_cmpf_vector(%arg0 : memref<128x32xf32>, %arg1 : memref<128x32xf32>) {
        //CHECK: arith.cmpf ult, {{%.*}}, {{%.*}} : vector<32x32xf32>
        //CHECK: arith.select {{%.*}}, {{%.*}}, {{%.*}} : vector<32x32xi1>, vector<32x32xf32>
        %c0 = arith.constant 0 : index
        %cst = arith.constant {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} dense<true> : vector<128x32xi1>
        %tile = xetile.init_tile %arg0[%c0, %c0] : memref<128x32xf32> -> !xetile.tile<128x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
        %load_tile_0 = xetile.load_tile %tile : !xetile.tile<128x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>> -> vector<128x32xf32>
        %tile_1 = xetile.init_tile %arg1[%c0, %c0] : memref<128x32xf32> -> !xetile.tile<128x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
        %load_tile_1 = xetile.load_tile %tile : !xetile.tile<128x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>> -> vector<128x32xf32>
        %cmpf = arith.cmpf ult, %load_tile_0, %load_tile_1 {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xf32>
        %select = arith.select %cmpf, %load_tile_0, %load_tile_1 {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xi1>, vector<128x32xf32>
        gpu.return
    }

     gpu.func @test_math_fpowi(%arg0 : memref<128x32xf32>, %arg1 : memref<128x32xi32>) {
        %c0 = arith.constant 0 : index
        %cst = arith.constant {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} dense<true> : vector<128x32xi1>
        %tile = xetile.init_tile %arg0[%c0, %c0] : memref<128x32xf32> -> !xetile.tile<128x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
        %load_tile_0 = xetile.load_tile %tile : !xetile.tile<128x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>> -> vector<128x32xf32>
        %tile_1 = xetile.init_tile %arg1[%c0, %c0] : memref<128x32xi32> -> !xetile.tile<128x32xi32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
        %load_tile_1 = xetile.load_tile %tile_1 : !xetile.tile<128x32xi32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>> -> vector<128x32xi32>
        //CHECK: math.fpowi {{%.*}}, {{%.*}} : vector<32x32xf32>, vector<32x32xi32>
        %fpowi = math.fpowi %load_tile_0, %load_tile_1 {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xf32>, vector<128x32xi32>
        gpu.return
    }

    gpu.func @test_create_mask() {
        %c128 = arith.constant 128 : index
        %c100 = arith.constant 100 : index
        //CHECK: vector.create_mask {{%.*}}, {{%.*}} : vector<32x32xi1>
        %create_mask = vector.create_mask %c128, %c100 {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xi1>
        gpu.return
    }

    gpu.func @test_if(%cond: i1) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %cst = arith.constant {map = #xetile.wg_map<sg_layout = [4, 4], sg_data = [8, 8]>} dense<0.0> : vector<32x32xf32>
        // CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<8x8xf32>
        %result = scf.for %i = %c0 to %c2 step %c1 iter_args(%arg0 = %cst) -> (vector<32x32xf32>) {
            // CHECK: scf.if {{%.*}} -> (vector<8x8xf32>) {
            %if_result = scf.if %cond -> (vector<32x32xf32>) {
            scf.yield %cst : vector<32x32xf32>
            // CHECK: scf.yield {{%.*}}  : vector<8x8xf32>
            } else {
            scf.yield %arg0 : vector<32x32xf32>
            // CHECK: scf.yield {{%.*}} : vector<8x8xf32>
            }
            scf.yield %cst : vector<32x32xf32>
        }
        gpu.return
    }

    gpu.func @test_if_basic(%cond: i1) {
        %cst = arith.constant {map = #xetile.wg_map<sg_layout = [4, 4], sg_data = [4, 4]>} dense<1.0> : vector<16x16xf32>
        // CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<4x4xf32>
        %result = scf.if %cond -> (vector<16x16xf32>) {
            // CHECK: scf.if {{%.*}} -> (vector<4x4xf32>) {
            // CHECK: scf.yield {{%.*}}  : vector<4x4xf32>
            scf.yield %cst : vector<16x16xf32>
        } else {
            // CHECK: scf.yield {{%.*}}  : vector<4x4xf32>
            scf.yield %cst : vector<16x16xf32>
        }
        gpu.return
    }

    gpu.func @test_init_tile_using_addr(%src: i64, %dim0_size : index, %dim1_size : index,
      %dim0_stride : index, %dim1_stride : index ) {
      //CHECK: xetile.init_tile {{%.*}}[{{%.*}}, {{%.*}}], [{{%.*}}, {{%.*}}], [{{%.*}}, {{%.*}}] : i64 -> !xetile.tile<4x64xf16>
      %1 = xetile.init_tile %src[8, 16], [%dim0_size, %dim1_size], [%dim0_stride, %dim1_stride]
        : i64 ->  !xetile.tile<128x64xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [4, 64]>>>
      gpu.return
    }

    gpu.func @test_while_vector_op(%cond1: i1, %cond2: i1) {
        %cst = arith.constant {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} dense<1.0> : vector<128x256xf32>
        %zero = arith.constant {map = #xetile.wg_map<sg_layout = [2, 4], sg_data = [64, 64]>} dense<0.0> : vector<128x256xf32>

        // CHECK: [[RES1:%.+]] = scf.while ({{.*}}) : (vector<32x32xf32>, i1) -> vector<32x32xf32>
        %result1 = scf.while (%arg0 = %cst, %arg1 = %cond1) : (vector<128x256xf32>, i1) -> (vector<128x256xf32>) {
          %cond = arith.andi %arg1, %cond1 : i1
          scf.condition(%cond) %arg0 : vector<128x256xf32>
        } do {
        ^bb0(%arg0: vector<128x256xf32>):
          // CHECK：scf.yield {{.*}} : vector<32x32xf32>, i1
          scf.yield %cst, %cond1 : vector<128x256xf32>, i1
        }

        // CHECK: [[RES2:%.+]] = scf.while ({{.*}}) : (vector<64x64xf32>, i1) -> vector<64x64xf32>
        %result2 = scf.while (%arg2 = %zero, %arg3 = %cond2) : (vector<128x256xf32>, i1) -> (vector<128x256xf32>) {
          %cond = arith.andi %arg3, %cond2 : i1
          scf.condition(%cond) %arg2 : vector<128x256xf32>
        } do {
        ^bb0(%arg2: vector<128x256xf32>):
          // CHECK：scf.yield {{.*}} : vector<64x64xf32>, i1
          scf.yield %zero, %cond2 : vector<128x256xf32>, i1
        }

        gpu.return
    }
    gpu.func @test_if_vector_op(%cond1: i1, %cond2: i1) {
        %cst = arith.constant {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} dense<1.0> : vector<128x256xf32>
        %zero = arith.constant {map = #xetile.wg_map<sg_layout = [2, 4], sg_data = [64, 64]>} dense<0.0> : vector<128x256xf32>
        // CHECK: %[[RES1:.*]] = scf.if {{.*}} -> (vector<32x32xf32>)
        %result = scf.if %cond1 -> (vector<128x256xf32>) {
            // CHECK：scf.yield {{.*}} : vector<32x32xf32>
            scf.yield %cst : vector<128x256xf32>
        } else {
            // CHECK：scf.yield {{.*}} : vector<32x32xf32>
            scf.yield %cst : vector<128x256xf32>
        }

        // CHECK: %[[RES2:.*]] = scf.if {{.*}} -> (vector<64x64xf32>)
        %result1 = scf.if %cond2 -> (vector<128x256xf32>) {
            // CHECK：scf.yield {{.*}} : vector<64x64xf32>
            scf.yield %zero : vector<128x256xf32>
        } else {
            // CHECK：scf.yield {{.*}} : vector<64x64xf32>
            scf.yield %zero : vector<128x256xf32>
        }
        gpu.return
    }

}
