// RUN: imex-opt --split-input-file --xetile-blockop-fallback=device=pvc %s -verify-diagnostics -o -| FileCheck %s

gpu.module @test_module {
  // CHECK-LABEL: @test_pitch_not_multiple_of_tile_width
  gpu.func @test_pitch_not_multiple_of_tile_width(%arg0: memref<512x250xf32>) {
    // CHECK: %[[VAR0:.*]] = xetile.init_tile %arg0[0, 0] : memref<512x250xf32> -> !xetile.tile<32x16xf32
    %0 = xetile.init_tile %arg0 [0, 0] : memref<512x250xf32> -> !xetile.tile<32x16xf32, #xetile.tile_attr<order = [1, 0]>>
    // CHECK: %[[VAR1:.*]] = xetile.load_tile %[[VAR0]]
    %1 = xetile.load_tile %0 : !xetile.tile<32x16xf32, #xetile.tile_attr<order = [1, 0]>> -> vector<32x16xf32>
    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: @test_pitch_one_elems_and_offset_attr
  gpu.func @test_pitch_one_elems_and_offset_attr(%arg0: memref<512x1xf32>) {
    // CHECK:  %[[CST:.*]] = arith.constant dense<true> : vector<32x1xi1>
    // CHECK:  %[[CAST:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [512], strides: [1] : memref<512x1xf32> to memref<512xf32>
    // CHECK:  %[[VAR0:.*]] = vector.step : vector<32xindex>
    // CHECK:  %[[VAR1:.*]] = vector.shape_cast %[[VAR0]] : vector<32xindex> to vector<32x1xindex>
    // CHECK:  %[[VAR2:.*]] = xetile.init_tile %[[CAST]], %[[VAR1]] : memref<512xf32>, vector<32x1xindex> -> !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    %0 = xetile.init_tile %arg0 [0, 0] : memref<512x1xf32> -> !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>
    // CHECK:  %[[VAR3:.*]] = xetile.load %[[VAR2]], %[[CST]] : !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<32x1xi1> -> vector<32x1xf32>
    %1 = xetile.load_tile %0 : !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>> -> vector<32x1xf32>
    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: @test_pitch_one_elems_and_offset_vars
  gpu.func @test_pitch_one_elems_and_offset_vars(%arg0: memref<512x1xf32>) {
    // CHECK:  %[[CST:.*]] = arith.constant dense<true> : vector<32x1xi1>
    // CHECK:  %[[CAST:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [512], strides: [1] : memref<512x1xf32> to memref<512xf32>
    // CHECK:  %[[VAR0:.*]] = vector.step : vector<32xindex>
    // CHECK:  %[[VAR1:.*]] = vector.shape_cast %[[VAR0]] : vector<32xindex> to vector<32x1xindex>
    // CHECK:  %[[VAR2:.*]] = xetile.init_tile %[[CAST]], %[[VAR1]] : memref<512xf32>, vector<32x1xindex> -> !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    %cst0 = arith.constant 0 : index
    %0 = xetile.init_tile %arg0 [%cst0, %cst0] : memref<512x1xf32> -> !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>
    // CHECK:  %[[VAR3:.*]] = xetile.load %[[VAR2]], %[[CST]] : !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<32x1xi1> -> vector<32x1xf32>
    %1 = xetile.load_tile %0 : !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>> -> vector<32x1xf32>
    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: @test_pitch_one_elems_and_mixed_offsets
  gpu.func @test_pitch_one_elems_and_mixed_offsets(%arg0: memref<512x1xf32>) {
    // CHECK:  %[[CST:.*]] = arith.constant dense<true> : vector<32x1xi1>
    // CHECK:  %[[CAST:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [512], strides: [1] : memref<512x1xf32> to memref<512xf32>
    // CHECK:  %[[VAR0:.*]] = vector.step : vector<32xindex>
    // CHECK:  %[[VAR1:.*]] = vector.shape_cast %[[VAR0]] : vector<32xindex> to vector<32x1xindex>
    // CHECK:  %[[VAR2:.*]] = xetile.init_tile %[[CAST]], %[[VAR1]] : memref<512xf32>, vector<32x1xindex> -> !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    %cst0 = arith.constant 0 : index
    %0 = xetile.init_tile %arg0 [%cst0, 0] : memref<512x1xf32> -> !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>
    // CHECK:  %[[VAR3:.*]] = xetile.load %[[VAR2]], %[[CST]] : !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<32x1xi1> -> vector<32x1xf32>
    %1 = xetile.load_tile %0 : !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>> -> vector<32x1xf32>
    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: @test_pitch_two_elems
  gpu.func @test_pitch_two_elems(%arg0: memref<512x2xf32>) {
    // CHECK:  %[[CST:.*]] = arith.constant dense<true> : vector<32x1xi1>
    // CHECK:  %[[CST0:.*]] = arith.constant dense<2> : vector<32x1xindex>
    // CHECK:  %[[CST1:.*]] = arith.constant dense<16> : vector<32xindex>
    // CHECK:  %[[CST2:.*]] = arith.constant dense<1> : vector<32x1xindex>
    // CHECK:  %[[CAST:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1024], strides: [1] : memref<512x2xf32> to memref<1024xf32>
    // CHECK:  %[[VAR0:.*]] = vector.step : vector<32xindex>
    // CHECK:  %[[VAR1:.*]] = arith.addi %[[VAR0]], %[[CST1]] : vector<32xindex>
    // CHECK:  %[[VAR2:.*]] = vector.shape_cast %[[VAR1]] : vector<32xindex> to vector<32x1xindex>
    // CHECK:  %[[VAR3:.*]] = arith.muli %[[VAR2]], %[[CST0]] : vector<32x1xindex>
    // CHECK:  %[[VAR4:.*]] = arith.addi %[[VAR3]], %[[CST2]] : vector<32x1xindex>
    // CHECK:  %[[VAR5:.*]] = xetile.init_tile %[[CAST]], %[[VAR4]] : memref<1024xf32>, vector<32x1xindex> -> !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    %0 = xetile.init_tile %arg0 [16, 1] : memref<512x2xf32> -> !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>
    // CHECK:  %[[VAR6:.*]] = xetile.load %[[VAR5]], %[[CST]] : !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<32x1xi1> -> vector<32x1xf32>
    %1 = xetile.load_tile %0 : !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>> -> vector<32x1xf32>
    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: @test_store_tile
  gpu.func @test_store_tile(%arg0: memref<512x1xf32>, %arg1: memref<512x1xf32>) {
    // CHECK:  %[[CST:.*]] = arith.constant dense<true> : vector<32x1xi1>
    // CHECK:  %[[CAST:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [512], strides: [1] : memref<512x1xf32> to memref<512xf32>
    // CHECK:  %[[VAR0:.*]] = vector.step : vector<32xindex>
    // CHECK:  %[[VAR1:.*]] = vector.shape_cast %[[VAR0]] : vector<32xindex> to vector<32x1xindex>
    // CHECK:  %[[VAR2:.*]] = xetile.init_tile %[[CAST]], %[[VAR1]] : memref<512xf32>, vector<32x1xindex> -> !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    %0 = xetile.init_tile %arg0 [0, 0] : memref<512x1xf32> -> !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>
    // CHECK:  %[[CAST0:.*]] = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512], strides: [1] : memref<512x1xf32> to memref<512xf32>
    // CHECK:  %[[VAR3:.*]] = vector.step : vector<32xindex>
    // CHECK:  %[[VAR4:.*]] = vector.shape_cast %[[VAR3:.*]] : vector<32xindex> to vector<32x1xindex>
    // CHECK:  %[[VAR5:.*]] = xetile.init_tile %[[CAST0]], %[[VAR4]] : memref<512xf32>, vector<32x1xindex> -> !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    %1 = xetile.init_tile %arg1 [0, 0] : memref<512x1xf32> -> !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>
    // CHECK:  %[[VAR6:.*]] = xetile.load %[[VAR2]], %[[CST]] : !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<32x1xi1> -> vector<32x1xf32>
    %2 = xetile.load_tile %0 : !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>> -> vector<32x1xf32>
    // CHECK:  xetile.store %[[VAR6]], %[[VAR5]], %[[CST]] : vector<32x1xf32>, !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<32x1xi1>
    xetile.store_tile %2, %1 : vector<32x1xf32>, !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>
    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: @test_update_tile_offset
  gpu.func @test_update_tile_offset(%arg0: memref<512x1xf32>, %arg1: memref<512x1xf32>) {
    // CHECK:  %[[CST:.*]] = arith.constant dense<32> : vector<32x1xindex>
    // CHECK:  %[[CST0:.*]] = arith.constant dense<true> : vector<32x1xi1>
    // CHECK:  %[[CAST:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [512], strides: [1] : memref<512x1xf32> to memref<512xf32>
    // CHECK:  %[[VAR0:.*]] = vector.step : vector<32xindex>
    // CHECK:  %[[VAR1:.*]] = vector.shape_cast %[[VAR0]] : vector<32xindex> to vector<32x1xindex>
    // CHECK:  %[[VAR2:.*]] = xetile.init_tile %[[CAST]], %[[VAR1]] : memref<512xf32>, vector<32x1xindex> -> !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    %cst0 = arith.constant 0 : index
    %cst32 = arith.constant 32 : index
    %0 = xetile.init_tile %arg0 [0, 0] : memref<512x1xf32> -> !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>
    // CHECK:  %[[VAR3:.*]] = xetile.load %[[VAR2]], %[[CST0]] : !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<32x1xi1> -> vector<32x1xf32>
    %1 = xetile.load_tile %0 : !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>> -> vector<32x1xf32>
    // CHECK:  %[[VAR4:.*]] = xetile.update_tile_offset %[[VAR2]], %[[CST]] : !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<32x1xindex>
    %2 = xetile.update_tile_offset %0, [%cst32, %cst0] : !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>
    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: @test_multiple_update_tile_offset
  gpu.func @test_multiple_update_tile_offset(%arg0: memref<512x1xf32>, %arg1: memref<512x1xf32>) {
    // CHECK:  %[[CST:.*]] = arith.constant dense<16> : vector<32x1xindex>
    // CHECK:  %[[CST0:.*]] = arith.constant dense<32> : vector<32x1xindex>
    // CHECK:  %[[CST1:.*]] = arith.constant dense<true> : vector<32x1xi1>
    // CHECK:  %[[CAST:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [512], strides: [1] : memref<512x1xf32> to memref<512xf32>
    // CHECK:  %[[VAR0:.*]] = vector.step : vector<32xindex>
    // CHECK:  %[[VAR1:.*]] = vector.shape_cast %[[VAR0]] : vector<32xindex> to vector<32x1xindex>
    // CHECK:  %[[VAR2:.*]] = xetile.init_tile %[[CAST]], %[[VAR1]] : memref<512xf32>, vector<32x1xindex> -> !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    %cst0 = arith.constant 0 : index
    %cst32 = arith.constant 32 : index
    %cst16 = arith.constant 16 : index
    %0 = xetile.init_tile %arg0 [0, 0] : memref<512x1xf32> -> !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>
    // CHECK:  %[[VAR6:.*]] = xetile.load %[[VAR2]], %[[CST1]] : !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<32x1xi1> -> vector<32x1xf32>
    %1 = xetile.load_tile %0 : !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>> -> vector<32x1xf32>
    // CHECK:  %[[VAR7:.*]] = xetile.update_tile_offset %[[VAR2]], %[[CST0]] : !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<32x1xindex>
    %2 = xetile.update_tile_offset %0, [%cst32, %cst0] : !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>
    // CHECK:  %[[VAR8:.*]] = xetile.update_tile_offset %[[VAR7]], %[[CST]] : !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<32x1xindex>
    %3 = xetile.update_tile_offset %2, [%cst16, %cst0] : !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>
    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: @test_scf_for
  gpu.func @test_scf_for(%arg0: memref<512x1xf32>, %arg1: memref<512x1xf32>) {
    // CHECK:  %[[CST:.*]] = arith.constant dense<32> : vector<32x1xindex>
    // CHECK:  %[[CST0:.*]] = arith.constant dense<true> : vector<32x1xi1>
    // CHECK:  %[[C0:.*]] = arith.constant 0 : index
    // CHECK:  %[[C32:.*]] = arith.constant 32 : index
    // CHECK:  %[[C480:.*]] = arith.constant 480 : index
    // CHECK:  %[[CAST:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [512], strides: [1] : memref<512x1xf32> to memref<512xf32>
    // CHECK:  %[[VAR0:.*]] = vector.step : vector<32xindex>
    // CHECK:  %[[VAR3:.*]] = vector.shape_cast %[[VAR0]] : vector<32xindex> to vector<32x1xindex>
    // CHECK:  %[[VAR5:.*]] = xetile.init_tile %[[CAST]], %[[VAR3]] : memref<512xf32>, vector<32x1xindex> -> !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    %cst0 = arith.constant 0 : index
    %cst32 = arith.constant 32 : index
    %cst512 = arith.constant 512 : index
    %cst480 = arith.constant 480 : index
    %0 = xetile.init_tile %arg0 [0, 0] : memref<512x1xf32> -> !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>
    // CHECK:  %[[CAST1:.*]] = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512], strides: [1] : memref<512x1xf32> to memref<512xf32>
    // CHECK:  %[[VAR6:.*]] = vector.step : vector<32xindex>
    // CHECK:  %[[VAR9:.*]] = vector.shape_cast %[[VAR6]] : vector<32xindex> to vector<32x1xindex>
    // CHECK:  %[[VAR11:.*]] = xetile.init_tile %[[CAST1]], %[[VAR9]] : memref<512xf32>, vector<32x1xindex> -> !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    %1 = xetile.init_tile %arg1 [0, 0] : memref<512x1xf32> -> !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>
    // CHECK:  %[[VAR12:.*]]:2 = scf.for %arg2 = %[[C0]] to %[[C480]] step %[[C32]] iter_args(%arg3 = %[[VAR5]], %arg4 = %[[VAR11]]) -> (!xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>) {
    %out:2 = scf.for %k = %cst0 to %cst480 step %cst32
      iter_args(%a_tile = %0, %b_tile = %1)
      -> (!xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>, !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>) {
        // CHECK:  %[[VAR14:.*]] = xetile.load %arg3, %[[CST0]] : !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<32x1xi1> -> vector<32x1xf32>
      %a_value = xetile.load_tile %a_tile : !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>> -> vector<32x1xf32>
      // CHECK:  xetile.store %[[VAR14]], %arg4, %[[CST0]] : vector<32x1xf32>, !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<32x1xi1>
      xetile.store_tile %a_value, %b_tile : vector<32x1xf32>, !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>
      // CHECK:  %[[VAR15:.*]] = xetile.update_tile_offset %arg3, %[[CST]] : !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<32x1xindex>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%cst32, %cst0] : !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>
      // CHECK:  %[[VAR16:.*]] = xetile.update_tile_offset %arg4, %[[CST]] : !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<32x1xindex>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%cst32, %cst0] : !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>
      // CHECK:  scf.yield %[[VAR15]], %[[VAR16]] : !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
      scf.yield %a_next_tile, %b_next_tile : !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>, !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>
    }
    // CHECK:  %[[VAR13:.*]] = xetile.load %[[VAR12]]#0, %[[CST0]] : !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<32x1xi1> -> vector<32x1xf32>
    %2 = xetile.load_tile %out#0 : !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>> -> vector<32x1xf32>
    // CHECK:  xetile.store %[[VAR13]], %[[VAR12]]#1, %[[CST0]] : vector<32x1xf32>, !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<32x1xi1>
    xetile.store_tile %2, %out#1 : vector<32x1xf32>, !xetile.tile<32x1xf32, #xetile.tile_attr<order = [1, 0]>>
    gpu.return
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.module @test_module {
    // CHECK-LABEL: func @test_nested_scf_for
    gpu.func @test_nested_scf_for(%arg0: memref<16384x1xf32>, %arg1: memref<16384x1xf32>) {
      // CHECK: %[[CST:.*]] = arith.constant dense<32> : vector<32x1xindex>
      // CHECK: %[[CST0:.*]] = arith.constant dense<true> : vector<32x1xi1>
      // CHECK: %[[CST1:.*]] = arith.constant dense<128> : vector<32x1xindex>
      // CHECK: %[[CST2:.*]] = arith.constant dense<512> : vector<32x1xindex>
      // CHECK: %[[C0:.*]] = arith.constant 0 : index
      // CHECK: %[[C512:.*]] = arith.constant 512 : index
      // CHECK: %[[C128:.*]] = arith.constant 128 : index
      // CHECK: %[[C32:.*]] = arith.constant 32 : index
      // CHECK: %[[C16384:.*]] = arith.constant 16384 : index
      // CHECK: %[[CAST:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16384], strides: [1] : memref<16384x1xf32> to memref<16384xf32>
      // CHECK: %[[VAR0:.*]] = vector.step : vector<32xindex>
      // CHECK: %[[VAR3:.*]] = vector.shape_cast %[[VAR0]] : vector<32xindex> to vector<32x1xindex>
      // CHECK: %[[VAR5:.*]] = xetile.init_tile %[[CAST]], %[[VAR3]] : memref<16384xf32>, vector<32x1xindex> -> !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
      %c0 = arith.constant 0 : index
      %c512 = arith.constant 512 : index
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %c16384 = arith.constant 16384 : index
      %1 = xetile.init_tile %arg0[%c0, %c0] : memref<16384x1xf32> -> !xetile.tile<32x1xf32>
      // CHECK: %[[CAST3:.*]] = memref.reinterpret_cast %arg1 to offset: [0], sizes: [16384], strides: [1] : memref<16384x1xf32> to memref<16384xf32>
      // CHECK: %[[VAR6:.*]] = vector.step : vector<32xindex>
      // CHECK: %[[VAR9:.*]] = vector.shape_cast %[[VAR6]] : vector<32xindex> to vector<32x1xindex>
      // CHECK: %[[VAR11:.*]] = xetile.init_tile %[[CAST3]], %[[VAR9]] : memref<16384xf32>, vector<32x1xindex> -> !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
      %2 = xetile.init_tile %arg1[%c0, %c0] : memref<16384x1xf32> -> !xetile.tile<32x1xf32>
      // CHECK: %[[VAR12:.*]]:2 = scf.for %arg2 = %[[C0]] to %[[C16384]] step %[[C512]] iter_args(%arg3 = %[[VAR5]], %arg4 = %[[VAR11]]) -> (!xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>) {
      %3:2 = scf.for %arg3 = %c0 to %c16384 step %c512 iter_args(%arg4 = %1, %arg5 = %2) -> (!xetile.tile<32x1xf32>, !xetile.tile<32x1xf32>) {
        %4 = xetile.update_tile_offset %arg4, [%c512, %c0] : !xetile.tile<32x1xf32>
        %5 = xetile.update_tile_offset %arg5, [%c512, %c0] : !xetile.tile<32x1xf32>
        // CHECK: %[[VAR15:.*]]:2 = scf.for %arg5 = %[[C0]] to %[[C512]] step %[[C128]] iter_args(%arg6 = %arg3, %arg7 = %arg4) -> (!xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>) {
        %6:2 = scf.for %arg6 = %c0 to %c512 step %c128 iter_args(%arg7 = %arg4, %arg8 = %arg5) -> (!xetile.tile<32x1xf32>, !xetile.tile<32x1xf32>) {
          %7 = xetile.update_tile_offset %arg7, [%c128, %c0] : !xetile.tile<32x1xf32>
          %8 = xetile.update_tile_offset %arg8, [%c128, %c0] : !xetile.tile<32x1xf32>
          // CHECK: %[[VAR18:.*]]:2 = scf.for %arg8 = %[[C0]] to %[[C128]] step %[[C32]] iter_args(%arg9 = %arg6, %arg10 = %arg7) -> (!xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, !xetile.tile<32x1xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>) {
          %9:2 = scf.for %arg9 = %c0 to %c128 step %c32 iter_args(%arg10 = %arg7, %arg11 = %arg8) -> (!xetile.tile<32x1xf32>, !xetile.tile<32x1xf32>) {
            %10 = xetile.load_tile %arg10 : !xetile.tile<32x1xf32> -> vector<32x1xf32>
            xetile.store_tile %10, %arg11 : vector<32x1xf32>, !xetile.tile<32x1xf32>
            %11 = xetile.update_tile_offset %arg10, [%c32, %c0] : !xetile.tile<32x1xf32>
            %12 = xetile.update_tile_offset %arg11, [%c32, %c0] : !xetile.tile<32x1xf32>
            scf.yield %11, %12 : !xetile.tile<32x1xf32>, !xetile.tile<32x1xf32>
          }
          scf.yield %7, %8 : !xetile.tile<32x1xf32>, !xetile.tile<32x1xf32>
        }
        scf.yield %4, %5 : !xetile.tile<32x1xf32>, !xetile.tile<32x1xf32>
      }
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @postop_reduce_m_entry(%arg0: memref<16384x12288xbf16>, %arg1: memref<2048x12288xbf16>, %arg2: memref<32x2048xf32>) attributes {gemm_tiles_b = 1 : i64, gemm_tiles_x = dense<[8, 2, 4, 8]> : vector<4xi64>, gemm_tiles_y = dense<[1, 2, 8, 4]> : vector<4xi64>, physical_nd_range = dense<[8, 32]> : vector<2xi64>, region_partition = 0 : i64, region_size = 32 : i64, syn.fusion_successful, syn.tensor_signature = (tensor<16384x12288xbf16>, tensor<2048x12288xbf16>) -> tensor<32x2048xf32>, synFusionGenOps = 6 : i64, synFusionRequiredBeamSize = 1 : i64, synFusionTotalCost = 1003595802.6 : f64} {
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @postop_reduce_m::@postop_reduce_m blocks in (%c8, %c32, %c1) threads in (%c8, %c4, %c1)  args(%arg0 : memref<16384x12288xbf16>, %arg1 : memref<2048x12288xbf16>, %arg2 : memref<32x2048xf32>)
    return
  }
  gpu.module @postop_reduce_m attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Bfloat16ConversionINTEL, BFloat16TypeKHR, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, VectorAnyINTEL, VectorComputeINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_bfloat16, SPV_KHR_expect_assume, SPV_INTEL_bfloat16_conversion, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    // CHECK-LABEL: func @postop_reduce_m
    gpu.func @postop_reduce_m(%arg0: memref<16384x12288xbf16>, %arg1: memref<2048x12288xbf16>, %arg2: memref<32x2048xf32>) kernel attributes {VectorComputeFunctionINTEL, known_block_size = array<i32: 8, 4, 1>, known_grid_size = array<i32: 8, 32, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // CHECK: %[[CST:.*]] = arith.constant dense<true> : vector<8x4xi1>
      // CHECK: %[[CST0:.*]] = arith.constant dense<128> : vector<8x4xindex>
      // CHECK: %[[CST1:.*]] = arith.constant dense<true> : vector<1x32xi1>
      // CHECK: %[[CST2:.*]] = arith.constant dense<128> : vector<1x32xindex>
      %c12288 = arith.constant 12288 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c256 = arith.constant 256 : index
      %c2048 = arith.constant 2048 : index
      %c128 = arith.constant 128 : index
      %c4 = arith.constant 4 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %cst = arith.constant dense<0.000000e+00> : vector<32x32xf32>
      %cst_0 = arith.constant dense<0.000000e+00> : vector<1x32xf32>
      %cst_1 = arith.constant dense<0.000000e+00> : vector<4xf32>
      %cst_2 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.divsi %block_id_y, %c8 : index
      %1 = arith.remsi %block_id_y, %c8 : index
      %2 = arith.muli %block_id_x, %c4 : index
      %3 = arith.addi %2, %0 : index
      %4 = arith.muli %1, %c128 : index
      %5 = gpu.subgroup_id : index
      %6 = index.floordivs %5, %c32
      %7 = index.remu %5, %c32
      %8 = index.remu %6, %c1
      %9 = index.add %3, %8
      %10 = index.remu %7, %c32
      %11 = index.mul %10, %c4
      %12 = index.add %4, %11
      %13 = xetile.init_tile %arg2[%9, %12] : memref<32x2048xf32> -> !xetile.tile<1x4xf32>
      %14 = arith.muli %block_id_x, %c2048 : index
      %15 = arith.muli %0, %c256 : index
      %16 = arith.addi %14, %15 : index
      %17 = index.floordivs %5, %c4
      %18 = index.remu %5, %c4
      %19 = index.remu %17, %c8
      %20 = index.mul %19, %c32
      %21 = index.add %16, %20
      %22 = index.remu %18, %c1
      %23 = index.mul %22, %c32
      %24 = xetile.init_tile %arg0[%21, %23] : memref<16384x12288xbf16> -> !xetile.tile<32x32xbf16>
      %25 = index.floordivs %5, %c8
      %26 = index.remu %5, %c8
      %27 = index.remu %26, %c4
      %28 = index.mul %27, %c32
      %29 = index.add %4, %28
      %30 = index.remu %25, %c1
      %31 = index.mul %30, %c32
      %32 = xetile.init_tile %arg1[%29, %31] : memref<2048x12288xbf16> -> !xetile.tile<32x32xbf16>
      %33:2 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %13, %arg5 = %32) -> (!xetile.tile<1x4xf32>, !xetile.tile<32x32xbf16>) {
        %34 = xetile.update_tile_offset %arg5, [%c1024, %c0] : !xetile.tile<32x32xbf16>
        %35 = xetile.update_tile_offset %arg4, [%c0, %c1024] : !xetile.tile<1x4xf32>
        %36:2 = scf.for %arg6 = %c0 to %c2 step %c1 iter_args(%arg7 = %cst_2, %arg8 = %24) -> (vector<1x4xf32>, !xetile.tile<32x32xbf16>) {
          %37 = xetile.update_tile_offset %arg8, [%c1024, %c0] : !xetile.tile<32x32xbf16>
          %38:3 = scf.for %arg9 = %c0 to %c12288 step %c32 iter_args(%arg10 = %cst, %arg11 = %arg8, %arg12 = %arg5) -> (vector<32x32xf32>, !xetile.tile<32x32xbf16>, !xetile.tile<32x32xbf16>) {
            %56 = xetile.update_tile_offset %arg12, [%c0, %c32] : !xetile.tile<32x32xbf16>
            %57 = xetile.update_tile_offset %arg11, [%c0, %c32] : !xetile.tile<32x32xbf16>
            %58 = xetile.load_tile %arg11 : !xetile.tile<32x32xbf16> -> vector<32x32xbf16>
            %59 = math.exp %58 : vector<32x32xbf16>
            %60 = xetile.load_tile %arg12 : !xetile.tile<32x32xbf16> -> vector<32x32xbf16>
            %61 = xetile.transpose %60, [1, 0] : vector<32x32xbf16> -> vector<32x32xbf16>
            xegpu.compile_hint
            %62 = xetile.tile_mma %59, %61, %arg10 : vector<32x32xbf16>, vector<32x32xbf16>, vector<32x32xf32> -> vector<32x32xf32>
            xegpu.compile_hint
            scf.yield %62, %57, %56 : vector<32x32xf32>, !xetile.tile<32x32xbf16>, !xetile.tile<32x32xbf16>
          }
          %39 = math.exp %38#0 : vector<32x32xf32>
          %40 = vector.shape_cast %cst_0 : vector<1x32xf32> to vector<32xf32>
          %41 = xetile.reduction <add>, %39 [0] : vector<32x32xf32> -> vector<1x32xf32>
          %42 = vector.shape_cast %41 : vector<1x32xf32> to vector<32xf32>
          %43 = arith.addf %42, %40 : vector<32xf32>
          %44 = vector.shape_cast %43 : vector<32xf32> to vector<1x32xf32>
          %alloc = memref.alloc() : memref<4096xi8, 3>
          %view = memref.view %alloc[%c0][] : memref<4096xi8, 3> to memref<8x128xf32, 3>
          %45 = index.mul %18, %c32
          // CHECK: %[[CAST:.*]] = memref.reinterpret_cast %[[VIEW:.*]] to offset: [0], sizes: [1024], strides: [1] : memref<8x128xf32, 3> to memref<1024xf32, 3>
          // CHECK: %[[VAR48:.*]] = vector.step : vector<32xindex>
          // CHECK: %[[VAR49:.*]] = vector.broadcast %[[VAR48]] : vector<32xindex> to vector<1x32xindex>
          // CHECK: %[[VAR50:.*]] = vector.splat %[[VAR45:.*]] : vector<1x32xindex>
          // CHECK: %[[VAR52:.*]] = arith.addi %[[VAR49]], %[[VAR50]] : vector<1x32xindex>
          // CHECK: %[[VAR54:.*]] = vector.splat %[[VAR17:.*]] : vector<1x32xindex>
          // CHECK: %[[VAR55:.*]] = arith.muli %[[VAR54]], %[[CST2]] : vector<1x32xindex>
          // CHECK: %[[VAR56:.*]] = arith.addi %[[VAR55]], %[[VAR52]] : vector<1x32xindex>
          // CHECK: %[[VAR57:.*]] = xetile.init_tile %[[CAST]], %[[VAR56]] : memref<1024xf32, 3>, vector<1x32xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<memory_space = 3 : i32, scattered = true>>
          %46 = xetile.init_tile %view[%17, %45] : memref<8x128xf32, 3> -> !xetile.tile<1x32xf32, #xetile.tile_attr<memory_space = 3 : i32>>
          // CHECK: xetile.store %[[VAR44:.*]], %[[VAR57]], %[[CST1]] : vector<1x32xf32>, !xetile.tile<1x32xf32, #xetile.tile_attr<memory_space = 3 : i32, scattered = true>>, vector<1x32xi1>
          xetile.store_tile %44,  %46 : vector<1x32xf32>, !xetile.tile<1x32xf32, #xetile.tile_attr<memory_space = 3 : i32>>
          gpu.barrier
          // CHECK: %[[VAR58:.*]] = index.mul
          // CHECK: %[[VAR59:.*]] = index.mul
          %47 = index.mul %6, %c8
          %48 = index.mul %7, %c4
          // CHECK: %[[CAST7:.*]] = memref.reinterpret_cast %[[VIEW]] to offset: [0], sizes: [1024], strides: [1] : memref<8x128xf32, 3> to memref<1024xf32, 3>
          // CHECK: %[[VAR62:.*]] = vector.step : vector<4xindex>
          // CHECK: %[[VAR63:.*]] = vector.broadcast %[[VAR62]] : vector<4xindex> to vector<8x4xindex>
          // CHECK: %[[VAR61:.*]] = vector.splat %[[VAR59]] : vector<8x4xindex>
          // CHECK: %[[VAR64:.*]] = arith.addi %[[VAR63]], %[[VAR61]] : vector<8x4xindex>
          // CHECK: %[[VAR65:.*]] = vector.step : vector<8xindex>
          // CHECK: %[[VAR60:.*]] = vector.splat %[[VAR58]] : vector<8xindex>
          // CHECK: %[[VAR66:.*]] = arith.addi %[[VAR65]], %[[VAR60]] : vector<8xindex>
          // CHECK: %[[VAR67:.*]] = vector.shape_cast %[[VAR66]] : vector<8xindex> to vector<8x1xindex>
          // CHECK: %[[VAR68:.*]] = vector.broadcast %[[VAR67]] : vector<8x1xindex> to vector<8x4xindex>
          // CHECK: %[[VAR69:.*]] = arith.muli %[[VAR68]], %[[CST0]] : vector<8x4xindex>
          // CHECK: %[[VAR70:.*]] = arith.addi %[[VAR69]], %[[VAR64]] : vector<8x4xindex>
          // CHECK: %[[VAR71:.*]] = xetile.init_tile %[[CAST7]], %[[VAR70]] : memref<1024xf32, 3>, vector<8x4xindex> -> !xetile.tile<8x4xf32, #xetile.tile_attr<memory_space = 3 : i32, scattered = true>>
          %49 = xetile.init_tile %view[%47, %48] : memref<8x128xf32, 3> -> !xetile.tile<8x4xf32, #xetile.tile_attr<memory_space = 3 : i32>>
          // CHECK: %[[VAR72:.*]] = xetile.load %[[VAR71]], %[[CST]] : !xetile.tile<8x4xf32, #xetile.tile_attr<memory_space = 3 : i32, scattered = true>>, vector<8x4xi1> -> vector<8x4xf32>
          %50 = xetile.load_tile %49 : !xetile.tile<8x4xf32, #xetile.tile_attr<memory_space = 3 : i32>> -> vector<8x4xf32>
          %51 = xetile.reduction <add>, %50 [0] : vector<8x4xf32> -> vector<1x4xf32>
          %52 = vector.shape_cast %51 : vector<1x4xf32> to vector<4xf32>
          %53 = arith.addf %52, %cst_1 : vector<4xf32>
          %54 = vector.shape_cast %53 : vector<4xf32> to vector<1x4xf32>
          %55 = arith.addf %54, %arg7 : vector<1x4xf32>
          scf.yield %55, %37 : vector<1x4xf32>, !xetile.tile<32x32xbf16>
        }
        xetile.store_tile %36#0,  %arg4 : vector<1x4xf32>, !xetile.tile<1x4xf32>
        scf.yield %35, %34 : !xetile.tile<1x4xf32>, !xetile.tile<32x32xbf16>
      }
      gpu.return
    }
  }
}

// -----
// code is unchanged intentionally for optimal case
gpu.module @kernel {
  gpu.func @test_optimal(%arg0: memref<64x64xf16>, %arg1: memref<64x64xf16>, %arg2: memref<64x64xf16>) kernel {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %0 = gpu.subgroup_id : index
    %1 = index.divu %0, %c2
    %2 = index.remu %0, %c2
    %3 = index.remu %1, %c16
    %4 = index.mul %3, %c4
    %5 = index.add %block_id_x, %4
    %6 = index.remu %2, %c2
    %7 = index.mul %6, %c32
    %8 = index.add %block_id_y, %7
    %9 = xetile.init_tile %arg0[%5, %8] : memref<64x64xf16> -> !xetile.tile<4x32xf16>
    %10 = xetile.load_tile %9 : !xetile.tile<4x32xf16> -> vector<4x32xf16>
    %alloc = memref.alloc() : memref<8192xi8, 3>
    %view = memref.view %alloc[%c0][] : memref<8192xi8, 3> to memref<64x64xf16, 3>
    %11 = index.mul %1, %c4
    %12 = index.mul %2, %c32
    //CHECK: xetile.init_tile {{.*}} : memref<64x64xf16, 3> -> !xetile.tile<4x32xf16, #xetile.tile_attr<memory_space = 3 : i32>>
    //CHECK: xetile.store_tile {{.*}} : vector<4x32xf16>, !xetile.tile<4x32xf16, #xetile.tile_attr<memory_space = 3 : i32>>
    %13 = xetile.init_tile %view[%11, %12] : memref<64x64xf16, 3> -> !xetile.tile<4x32xf16, #xetile.tile_attr<memory_space = 3 : i32>>
    xetile.store_tile %10,  %13 : vector<4x32xf16>, !xetile.tile<4x32xf16, #xetile.tile_attr<memory_space = 3 : i32>>
    gpu.barrier
    %14 = index.divu %0, %c4
    %15 = index.remu %0, %c4
    %16 = index.mul %14, %c8
    %17 = index.mul %15, %c16
    //CHECK: xetile.init_tile {{.*}} : memref<64x64xf16, 3> -> !xetile.tile<8x16xf16, #xetile.tile_attr<memory_space = 3 : i32>>
    //CHECK: xetile.load_tile {{.*}} : !xetile.tile<8x16xf16, #xetile.tile_attr<memory_space = 3 : i32>> -> vector<8x16xf16>
    %18 = xetile.init_tile %view[%16, %17] : memref<64x64xf16, 3> -> !xetile.tile<8x16xf16, #xetile.tile_attr<memory_space = 3 : i32>>
    %19 = xetile.load_tile %18 : !xetile.tile<8x16xf16, #xetile.tile_attr<memory_space = 3 : i32>> -> vector<8x16xf16>
    %20 = index.remu %14, %c8
    %21 = index.mul %20, %c8
    %22 = index.add %block_id_x, %21
    %23 = index.remu %15, %c4
    %24 = index.mul %23, %c16
    %25 = index.add %block_id_y, %24
    %26 = xetile.init_tile %arg1[%22, %25] : memref<64x64xf16> -> !xetile.tile<8x16xf16>
    %27 = xetile.load_tile %26 : !xetile.tile<8x16xf16> -> vector<8x16xf16>
    %28 = arith.addf %27, %19 : vector<8x16xf16>
    %29 = xetile.init_tile %arg2[%22, %25] : memref<64x64xf16> -> !xetile.tile<8x16xf16>
    xetile.store_tile %28,  %29 : vector<8x16xf16>, !xetile.tile<8x16xf16>
    gpu.return
  }
}
