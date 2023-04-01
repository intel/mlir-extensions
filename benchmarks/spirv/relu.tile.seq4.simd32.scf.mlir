module attributes {torch.debug_module_name = "ReLU"} {
  memref.global "private" constant @__constant_512x640x20x15xf32 : memref<512x640x20x15xf32> = dense<1.300000e+00>
  func.func @forward(%arg0: memref<512x640x20x15xf32>) -> memref<512x640x20x15xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1, 2, 3]] : memref<512x640x20x15xf32> into memref<98304000xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<98304000xf32>
    %c0 = arith.constant 0 : index
    %c24000 = arith.constant 24000 : index
    %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
          %c4 = arith.constant 4 : index
    scf.parallel (%arg1) = (%c0) to (%c24000) step (%c1) {
      scf.parallel (%arg3, %arg2) = (%c0, %c0) to (%c32, %c32) step (%c1,%c1) {
          scf.parallel (%arg4) = (%c0) to (%c4) step (%c1) {
            %c4096 = arith.constant 4096 : index
            %0 = arith.muli %arg1, %c4096 : index
            %c128 = arith.constant 128 : index
            %1 = arith.muli %arg2, %c128 : index
            %2 = arith.addi %0, %1 : index
            %3 = arith.addi %2, %arg3 : index
            %4 = arith.muli %arg4, %c32 : index
            %5 = arith.addi %3, %4 : index
            %6 = memref.load %collapse_shape[%5] : memref<98304000xf32>
            %7 = arith.cmpf ugt, %6, %cst : f32
            %8 = arith.select %7, %6, %cst : f32
            memref.store %8, %alloc[%5] : memref<98304000xf32>
            scf.yield
          }
        scf.yield
      }
      scf.yield
    }
    %expand_shape = memref.expand_shape %alloc [[0, 1, 2, 3]] : memref<98304000xf32> into memref<512x640x20x15xf32>
    return %expand_shape : memref<512x640x20x15xf32>
  }
  func.func @main() {
    %0 = memref.get_global @__constant_512x640x20x15xf32 : memref<512x640x20x15xf32>
    %1 = call @forward(%0) : (memref<512x640x20x15xf32>) -> memref<512x640x20x15xf32>
    return
  }
}
