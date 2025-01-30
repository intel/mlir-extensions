
#scattered = #xetile.tile_attr<scattered = true>
gpu.module @test {
  // gpu.func @test_init_tile_for_scattered(%a: memref<1024xf16>) {
  //   %indices = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
  //                                    [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
  //                                    [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
  //                                    [96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]]>: vector<4x32xindex>
  //   %mask = arith.constant dense<1> : vector<4x32xi1>
  //   %1 = xetile.init_tile %a, %indices : memref<1024xf16>, vector<4x32xindex> -> !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>
  //   %2 = xetile.load %1, %mask : !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>, vector<4x32xi1> -> vector<4x32xf16>
  //   %3 = xetile.update_tile_offset %1, %indices : !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>, vector<4x32xindex>
  //   xetile.store %2, %1, %mask : vector<4x32xf16>, !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>, vector<4x32xi1>
  //   gpu.return
  // }

  // gpu.func @sg_store_scatter(%a: memref<1024xf16>, %indices: vector<4x32xindex>) {
  //   // expected-error@+1
  //   %mask = arith.constant dense<1> : vector<4x32xi1>
  //   %data = arith.constant dense<42.0> : vector<4x32xf16>
  //   %tile = xetile.init_tile %a, %indices : memref<1024xf16>, vector<4x32xindex> -> !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>
  //   xetile.store %data, %tile, %mask : vector<4x32xf16>, !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>, vector<4x32xi1>
  //   gpu.return
  // }

  // gpu.func @add_kernel(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>) {
  //     %cst = arith.constant dense<true> : vector<1x32xi1>
  //     %c1024 = arith.constant 1024 : index
  //     %cast = memref.cast %arg0 : memref<*xf32> to memref<?xf32>
  //     %cast_0 = memref.cast %arg1 : memref<*xf32> to memref<?xf32>
  //     %cast_1 = memref.cast %arg2 : memref<*xf32> to memref<?xf32>
  //     %block_id_x = gpu.block_id  x
  //     %0 = arith.muli %block_id_x, %c1024 : index
  //     %1 = vector.splat %0 : vector<1x32xindex>
  //     %2 = xetile.init_tile %cast, %1 : memref<?xf32>, vector<1x32xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>
  //     %3 = xetile.load %2, %cst : !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xi1> -> vector<1x32xf32>
  //     %4 = xetile.init_tile %cast_0, %1 : memref<?xf32>, vector<1x32xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>
  //     %5 = xetile.load %4, %cst : !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xi1> -> vector<1x32xf32>
  //     %6 = arith.addf %3, %5 : vector<1x32xf32>
  //     %7 = xetile.init_tile %cast_1, %1 : memref<?xf32>, vector<1x32xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>
  //     xetile.store %6, %7, %cst : vector<1x32xf32>, !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>
  //     gpu.return
  // }

  // gpu.func @sg_store_scatter(%a: memref<1024xf16>, %indices: vector<4x32xindex>) {
  //   %mask = arith.constant dense<1> : vector<4x32xi1>
  //   %data = arith.constant dense<42.0> : vector<4x32xf16>
  //   %tile = xetile.init_tile %a, %indices : memref<1024xf16>, vector<4x32xindex> -> !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>
  //   xetile.store %data, %tile, %mask : vector<4x32xf16>, !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>, vector<4x32xi1>
  //   %offsets = arith.constant dense<1> : vector<4x32xindex>
  //   %next = xetile.update_tile_offset %tile, %offsets : !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>, vector<4x32xindex>
  //   xetile.store %data, %next, %mask : vector<4x32xf16>, !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>, vector<4x32xi1>
  //   gpu.return
  // }


  gpu.func @copy(%a: memref<4096xf16>, %b: memref<4096xf16>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %mask = arith.constant dense<1> : vector<4x32xi1>
    // view memref as 4x32xf16, each iter will copy a 4x32xf16 tile
    %indices = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                                     [1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055],
                                     [2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079],
                                     [3072, 3073, 3074, 3075, 3076, 3077, 3078, 3079, 3080, 3081, 3082, 3083, 3084, 3085, 3086, 3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3098, 3099, 3100, 3101, 3102, 3103]]> : vector<4x32xindex>
    %offsets = arith.constant dense<1> : vector<4x32xindex>

    %1 = xetile.init_tile %a, %indices : memref<4096xf16>, vector<4x32xindex> -> !xetile.tile<4x32xf16, #scattered>
    %2 = xetile.init_tile %b, %indices : memref<4096xf16>, vector<4x32xindex> -> !xetile.tile<4x32xf16, #scattered>

    scf.for %i = %c0 to %c1024 step %c32 {
      %out:2 = scf.for %k = %c0 to %c1024 step %c32 iter_args(%a_tile = %1, %b_tile = %2) -> (!xetile.tile<4x32xf16, #scattered>, !xetile.tile<4x32xf16, #scattered>) {
        %a_next = xetile.update_tile_offset %a_tile, %offsets {flag3}: !xetile.tile<4x32xf16, #scattered>, vector<4x32xindex>
        scf.yield %a_next, %b_tile : !xetile.tile<4x32xf16, #scattered>, !xetile.tile<4x32xf16, #scattered>
      }
      %data = xetile.load %out#0, %mask {flag5}: !xetile.tile<4x32xf16, #scattered>, vector<4x32xi1> -> vector<4x32xf16>
      xetile.store %data, %out#1, %mask {flag6}: vector<4x32xf16>, !xetile.tile<4x32xf16, #scattered>, vector<4x32xi1>
      scf.yield
    }
    gpu.return
  }

}
