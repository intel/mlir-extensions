// RUN: imex-opt --split-input-file -imex-xegpu-materialize-matrix-op -cse -canonicalize %s -verify-diagnostics -o -| FileCheck %s

gpu.module @test {

  //CHECK: load_matrix([[m:%.+]]: memref<2048xi8, 3>)
  gpu.func @load_matrix(%m: memref<2048xi8, 3>) -> vector<8x16xf16> {
    //CHECK: [[offset:%.+]] = arith.constant 192 : index
    //CHECK: [[c0:%.+]] = arith.constant 0 : index
    //CHECK: [[view:%.+]] = memref.view [[m]][[[c0]]][] : memref<2048xi8, 3> to memref<512xf32, 3>
    //CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[view]][[[offset]]] : memref<512xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm>>
    //CHECK: [[load:%.+]] = xegpu.load_nd [[tdesc]]  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm>> -> vector<64xf32>
    //CHECK: [[cast:%.+]] = vector.bitcast [[load]] : vector<64xf32> to vector<128xf16>
    //CHECK: [[result:%.+]] = vector.shape_cast [[cast]] : vector<128xf16> to vector<8x16xf16>
    %mem_desc = xegpu.create_mem_desc %m : memref<2048xi8, 3> -> !xegpu.mem_desc<16x64xf16, #xegpu.mem_layout<block = [16, 16]>>
    %load = xegpu.load_matrix %mem_desc[8, 16] : !xegpu.mem_desc<16x64xf16, #xegpu.mem_layout<block = [16, 16]>> -> vector<8x16xf16>
    gpu.return %load: vector<8x16xf16>
  }


  //CHECK: store_matrix([[A:%.+]]: memref<64x64xf16>)
  gpu.func @store_matrix(%A: memref<64x64xf16>) {

    //CHECK: [[offset:%.+]] = arith.constant 192 : index
    //CHECK: [[c0:%.+]] = arith.constant 0 : index
    //CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[A]][0, 0] : memref<64x64xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK: [[load:%.+]] = xegpu.load_nd [[tdesc]] : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
    //CHECK: [[alloca:%.+]] = memref.alloca() {alignment = 1024 : i64} : memref<2048xi8, 3>
    //CHECK: [[view:%.+]] = memref.view [[alloca]][[[c0]]][] : memref<2048xi8, 3> to memref<512xf32, 3>
    //CHECK: [[slmtdesc:%.+]] = xegpu.create_nd_tdesc [[view]][[[offset]]] : memref<512xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm>>
    //CHECK: [[flatten:%.+]] = vector.shape_cast [[load]] : vector<8x16xf16> to vector<128xf16>
    //CHECK: [[data:%.+]] = vector.bitcast [[flatten]] : vector<128xf16> to vector<64xf32>
    //CHECK: xegpu.store_nd [[data]], [[slmtdesc]]  : vector<64xf32>, !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm>>
    %tdesc = xegpu.create_nd_tdesc %A[0, 0] : memref<64x64xf16> -> !xegpu.tensor_desc<8x16xf16>
    %data = xegpu.load_nd %tdesc: !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
    %m = memref.alloca() {alignment = 1024} : memref<2048xi8, 3>
    %mem_desc_2 = xegpu.create_mem_desc %m : memref<2048xi8, 3> -> !xegpu.mem_desc<16x64xf16, #xegpu.mem_layout<block = [16, 16]>>
    xegpu.store_matrix %data, %mem_desc_2[8, 16] : vector<8x16xf16>, !xegpu.mem_desc<16x64xf16, #xegpu.mem_layout<block = [16, 16]>>
    gpu.return
  }

  //CHECK: store_matrix_strided([[A:%.+]]: memref<64x64xf16>)
  gpu.func @store_matrix_strided(%A: memref<64x64xf16>) {
    //CHECK: [[offsets:%.+]] = arith.constant dense<[132, 140, 148, 156, 164, 172, 180, 188, 196, 204, 212, 220, 228, 236, 244, 252]> : vector<16xindex>
    //CHECK: [[mask:%.+]] = arith.constant dense<true> : vector<16xi1>
    //CHECK: [[c0:%.+]] = arith.constant 0 : index
    //CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[A]][0, 0] : memref<64x64xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK: [[load:%.+]] = xegpu.load_nd [[tdesc]] <{packed}> : !xegpu.tensor_desc<8x16xf16> -> vector<4x16x2xf16>
    //CHECK: [[alloca:%.+]] = memref.alloca() {alignment = 1024 : i64} : memref<2048xi8, 3>
    //CHECK: [[view:%.+]] = memref.view [[alloca]][[[c0]]][] : memref<2048xi8, 3> to memref<512xf32, 3>
    //CHECK: [[shapecast:%.+]] = vector.shape_cast [[load]] : vector<4x16x2xf16> to vector<4x32xf16>
    //CHECK: [[bcast:%.+]] = vector.bitcast [[shapecast]] : vector<4x32xf16> to vector<4x16xf32>
    //CHECK: [[data:%.+]] = vector.transpose [[bcast]], [1, 0] : vector<4x16xf32> to vector<16x4xf32>
    //CHECK: [[tdesc2:%.+]] = xegpu.create_tdesc [[view]], [[offsets]] : memref<512xf32, 3>, vector<16xindex> -> !xegpu.tensor_desc<16x4xf32, #xegpu.scatter_tdesc_attr<memory_space = slm, chunk_size = 4 : i64>>
    //CHECK: xegpu.store [[data]], [[tdesc2]], [[mask]] : vector<16x4xf32>, !xegpu.tensor_desc<16x4xf32, #xegpu.scatter_tdesc_attr<memory_space = slm, chunk_size = 4 : i64>>, vector<16xi1>
    %tdesc = xegpu.create_nd_tdesc %A[0, 0] : memref<64x64xf16> -> !xegpu.tensor_desc<8x16xf16>
    %data = xegpu.load_nd %tdesc: !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
    %m = memref.alloca() {alignment = 1024} : memref<2048xi8, 3>
    %mem_desc_2 = xegpu.create_mem_desc %m : memref<2048xi8, 3> -> !xegpu.mem_desc<16x64xf16, #xegpu.mem_layout<stride = [1, 16], block = [16, 16]>>
    xegpu.store_matrix %data, %mem_desc_2[8, 16] : vector<8x16xf16>, !xegpu.mem_desc<16x64xf16, #xegpu.mem_layout<stride = [1, 16], block = [16, 16]>>
    gpu.return
  }

 //CHECK: store_matrix_strided_v2([[A:%.+]]: memref<64x64xf16>)
  gpu.func @store_matrix_strided_v2(%A: memref<64x64xf16>) {
    //CHECK: [[offsets:%.+]] = arith.constant dense<[132, 140, 148, 156, 164, 172, 180, 188, 196, 204, 212, 220, 228, 236, 244, 252]> : vector<16xindex>
    //CHECK: [[mask:%.+]] = arith.constant dense<true> : vector<16xi1>
    //CHECK: [[c0:%.+]] = arith.constant 0 : index
    //CHECK: [[tdesc:%.+]] = xegpu.create_nd_tdesc [[A]][0, 0] : memref<64x64xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK: [[load:%.+]] = xegpu.load_nd [[tdesc]] : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
    //CHECK: [[alloca:%.+]] = memref.alloca() {alignment = 1024 : i64} : memref<2048xi8, 3>
    //CHECK: [[view:%.+]] = memref.view [[alloca]][[[c0]]][] : memref<2048xi8, 3> to memref<512xf32, 3>
    //CHECK: [[shapecast1:%.+]] = vector.shape_cast [[load]] {packed} : vector<8x16xf16> to vector<128xf16>
    //CHECK: [[shuffle:%.+]] = vector.shuffle [[shapecast1]], [[shapecast1]]
    //CHECK: [[shapecast:%.+]] = vector.shape_cast [[shuffle]] : vector<128xf16> to vector<4x32xf16>
    //CHECK: [[bcast:%.+]] = vector.bitcast [[shapecast]] : vector<4x32xf16> to vector<4x16xf32>
    //CHECK: [[data:%.+]] = vector.transpose [[bcast]], [1, 0] : vector<4x16xf32> to vector<16x4xf32>
    //CHECK: [[tdesc2:%.+]] = xegpu.create_tdesc [[view]], [[offsets]] : memref<512xf32, 3>, vector<16xindex> -> !xegpu.tensor_desc<16x4xf32, #xegpu.scatter_tdesc_attr<memory_space = slm, chunk_size = 4 : i64>>
    //CHECK: xegpu.store [[data]], [[tdesc2]], [[mask]] : vector<16x4xf32>, !xegpu.tensor_desc<16x4xf32, #xegpu.scatter_tdesc_attr<memory_space = slm, chunk_size = 4 : i64>>, vector<16xi1>
    %tdesc = xegpu.create_nd_tdesc %A[0, 0] : memref<64x64xf16> -> !xegpu.tensor_desc<8x16xf16>
    %data = xegpu.load_nd %tdesc: !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
    %m = memref.alloca() {alignment = 1024} : memref<2048xi8, 3>
    %mem_desc_2 = xegpu.create_mem_desc %m : memref<2048xi8, 3> -> !xegpu.mem_desc<16x64xf16, #xegpu.mem_layout<stride = [1, 16], block = [16, 16]>>
    xegpu.store_matrix %data, %mem_desc_2[8, 16] : vector<8x16xf16>, !xegpu.mem_desc<16x64xf16, #xegpu.mem_layout<stride = [1, 16], block = [16, 16]>>
    xegpu.store_nd %data, %tdesc: vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
    gpu.return
  }

}
