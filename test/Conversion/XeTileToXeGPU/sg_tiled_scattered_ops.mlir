// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu --cse --canonicalize %s -verify-diagnostics -o -| FileCheck %s

gpu.module @test {
  //CHECK-LABEL: @test_init_tile_for_scattered
  //CHECK-SAME: %[[arg0:.*]]: memref<1024xf16>
  gpu.func @test_init_tile_for_scattered(%arg0: memref<1024xf16>) {

    //CHECK: %[[cst:.*]] = arith.constant dense<1> : vector<16xindex>
    //CHECK: %[[cst_0:.*]] = arith.constant dense<true> : vector<16xi1>
    //CHECK: %[[cst_1:.*]] = arith.constant dense<{{.*}}0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15{{.*}}> : vector<1x16xindex>
    //CHECK: %[[cst_2:.*]] = arith.constant dense<{{.*}}16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31{{.*}}]> : vector<1x16xindex>
    //CHECK: %[[cst_3:.*]] = arith.constant dense<{{.*}}32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47{{.*}}]> : vector<1x16xindex>
    //CHECK: %[[cst_4:.*]] = arith.constant dense<{{.*}}48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63{{.*}}]> : vector<1x16xindex>
    //CHECK: %[[cst_5:.*]] = arith.constant dense<{{.*}}64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79{{.*}}]> : vector<1x16xindex>
    //CHECK: %[[cst_6:.*]] = arith.constant dense<{{.*}}80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95{{.*}}]> : vector<1x16xindex>
    //CHECK: %[[cst_7:.*]] = arith.constant dense<{{.*}}96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111{{.*}}> : vector<1x16xindex>
    //CHECK: %[[cst_8:.*]] = arith.constant dense<{{.*}}112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127{{.*}}> : vector<1x16xindex>
    //CHECK: %[[r0:.*]] = vector.shape_cast %[[cst_1]] : vector<1x16xindex> to vector<16xindex>
    //CHECK: %[[r1:.*]] = xegpu.create_tdesc %[[arg0]], %[[r0]] : memref<1024xf16>, vector<16xindex> -> !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>
    //CHECK: %[[r2:.*]] = vector.shape_cast %[[cst_2]] : vector<1x16xindex> to vector<16xindex>
    //CHECK: %[[r3:.*]] = xegpu.create_tdesc %[[arg0]], %[[r2]] : memref<1024xf16>, vector<16xindex> -> !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>
    //CHECK: %[[r4:.*]] = vector.shape_cast %[[cst_3]] : vector<1x16xindex> to vector<16xindex>
    //CHECK: %[[r5:.*]] = xegpu.create_tdesc %[[arg0]], %[[r4]] : memref<1024xf16>, vector<16xindex> -> !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>
    //CHECK: %[[r6:.*]] = vector.shape_cast %[[cst_4]] : vector<1x16xindex> to vector<16xindex>
    //CHECK: %[[r7:.*]] = xegpu.create_tdesc %[[arg0]], %[[r6]] : memref<1024xf16>, vector<16xindex> -> !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>
    //CHECK: %[[r8:.*]] = vector.shape_cast %[[cst_5]] : vector<1x16xindex> to vector<16xindex>
    //CHECK: %[[r9:.*]] = xegpu.create_tdesc %[[arg0]], %[[r8]] : memref<1024xf16>, vector<16xindex> -> !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>
    //CHECK: %[[r10:.*]] = vector.shape_cast %[[cst_6]] : vector<1x16xindex> to vector<16xindex>
    //CHECK: %[[r11:.*]] = xegpu.create_tdesc %[[arg0]], %[[r10]] : memref<1024xf16>, vector<16xindex> -> !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>
    //CHECK: %[[r12:.*]] = vector.shape_cast %[[cst_7]] : vector<1x16xindex> to vector<16xindex>
    //CHECK: %[[r13:.*]] = xegpu.create_tdesc %[[arg0]], %[[r12]] : memref<1024xf16>, vector<16xindex> -> !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>
    //CHECK: %[[r14:.*]] = vector.shape_cast %[[cst_8]] : vector<1x16xindex> to vector<16xindex>
    //CHECK: %[[r15:.*]] = xegpu.create_tdesc %[[arg0]], %[[r14]] : memref<1024xf16>, vector<16xindex> -> !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>
    //CHECK: %[[r16:.*]] = xegpu.load %[[r1]], %[[cst_0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1> -> vector<16xf16>
    //CHECK: %[[r17:.*]] = xegpu.load %[[r3]], %[[cst_0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1> -> vector<16xf16>
    //CHECK: %[[r18:.*]] = xegpu.load %[[r5]], %[[cst_0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1> -> vector<16xf16>
    //CHECK: %[[r19:.*]] = xegpu.load %[[r7]], %[[cst_0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1> -> vector<16xf16>
    //CHECK: %[[r20:.*]] = xegpu.load %[[r9]], %[[cst_0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1> -> vector<16xf16>
    //CHECK: %[[r21:.*]] = xegpu.load %[[r11]], %[[cst_0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1> -> vector<16xf16>
    //CHECK: %[[r22:.*]] = xegpu.load %[[r13]], %[[cst_0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1> -> vector<16xf16>
    //CHECK: %[[r23:.*]] = xegpu.load %[[r15]], %[[cst_0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1> -> vector<16xf16>
    //CHECK: %[[r24:.*]] = xegpu.update_offset %[[r1]], %[[cst]] : !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xindex>
    //CHECK: %[[r25:.*]] = xegpu.update_offset %[[r3]], %[[cst]] : !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xindex>
    //CHECK: %[[r26:.*]] = xegpu.update_offset %[[r5]], %[[cst]] : !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xindex>
    //CHECK: %[[r27:.*]] = xegpu.update_offset %[[r7]], %[[cst]] : !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xindex>
    //CHECK: %[[r28:.*]] = xegpu.update_offset %[[r9]], %[[cst]] : !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xindex>
    //CHECK: %[[r29:.*]] = xegpu.update_offset %[[r11]], %[[cst]] : !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xindex>
    //CHECK: %[[r30:.*]] = xegpu.update_offset %[[r13]], %[[cst]] : !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xindex>
    //CHECK: %[[r31:.*]] = xegpu.update_offset %[[r15]], %[[cst]] : !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xindex>
    //CHECK: xegpu.store %[[r16]], %[[r1]], %[[cst_0]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<16xf16>, !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1>
    //CHECK: xegpu.store %[[r17]], %[[r3]], %[[cst_0]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<16xf16>, !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1>
    //CHECK: xegpu.store %[[r18]], %[[r5]], %[[cst_0]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<16xf16>, !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1>
    //CHECK: xegpu.store %[[r19]], %[[r7]], %[[cst_0]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<16xf16>, !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1>
    //CHECK: xegpu.store %[[r20]], %[[r9]], %[[cst_0]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<16xf16>, !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1>
    //CHECK: xegpu.store %[[r21]], %[[r11]], %[[cst_0]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<16xf16>, !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1>
    //CHECK: xegpu.store %[[r22]], %[[r13]], %[[cst_0]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<16xf16>, !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1>
    //CHECK: xegpu.store %[[r23]], %[[r15]], %[[cst_0]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<16xf16>, !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<16xi1>


    %cst = arith.constant dense<true> : vector<4x2x1x16xi1>
    %cst_0 = arith.constant dense<"0x00000000000000000100000000000000020000000000000003000000000000000400000000000000050000000000000006000000000000000700000000000000080000000000000009000000000000000A000000000000000B000000000000000C000000000000000D000000000000000E000000000000000F0000000000000010000000000000001100000000000000120000000000000013000000000000001400000000000000150000000000000016000000000000001700000000000000180000000000000019000000000000001A000000000000001B000000000000001C000000000000001D000000000000001E000000000000001F0000000000000020000000000000002100000000000000220000000000000023000000000000002400000000000000250000000000000026000000000000002700000000000000280000000000000029000000000000002A000000000000002B000000000000002C000000000000002D000000000000002E000000000000002F0000000000000030000000000000003100000000000000320000000000000033000000000000003400000000000000350000000000000036000000000000003700000000000000380000000000000039000000000000003A000000000000003B000000000000003C000000000000003D000000000000003E000000000000003F0000000000000040000000000000004100000000000000420000000000000043000000000000004400000000000000450000000000000046000000000000004700000000000000480000000000000049000000000000004A000000000000004B000000000000004C000000000000004D000000000000004E000000000000004F0000000000000050000000000000005100000000000000520000000000000053000000000000005400000000000000550000000000000056000000000000005700000000000000580000000000000059000000000000005A000000000000005B000000000000005C000000000000005D000000000000005E000000000000005F0000000000000060000000000000006100000000000000620000000000000063000000000000006400000000000000650000000000000066000000000000006700000000000000680000000000000069000000000000006A000000000000006B000000000000006C000000000000006D000000000000006E000000000000006F0000000000000070000000000000007100000000000000720000000000000073000000000000007400000000000000750000000000000076000000000000007700000000000000780000000000000079000000000000007A000000000000007B000000000000007C000000000000007D000000000000007E000000000000007F00000000000000"> : vector<4x2x1x16xindex>
    %offsets = arith.constant dense<1> : vector<4x2x1x16xindex>
    %0 = xetile.init_tile %arg0, %cst_0 : memref<1024xf16>, vector<4x2x1x16xindex> -> !xetile.tile<4x32xf16, #xetile.tile_attr<inner_blocks = [1, 16], scattered = true>>
    %1 = xetile.load %0, %cst : !xetile.tile<4x32xf16, #xetile.tile_attr<inner_blocks = [1, 16], scattered = true>>, vector<4x2x1x16xi1> -> vector<4x2x1x16xf16>
    %2 = xetile.update_tile_offset %0, %offsets : !xetile.tile<4x32xf16, #xetile.tile_attr<inner_blocks = [1, 16], scattered = true>>, vector<4x2x1x16xindex>
    xetile.store %1, %0, %cst : vector<4x2x1x16xf16>, !xetile.tile<4x32xf16, #xetile.tile_attr<inner_blocks = [1, 16], scattered = true>>, vector<4x2x1x16xi1>
    gpu.return
  }
}
