// RUN: imex-opt --pass-pipeline="builtin.module(func.func(sharding-propagation),coalesce-shard-ops,canonicalize,func.func(mesh-spmdization),canonicalize,convert-mesh-to-mpi,canonicalize,convert-ndarray-to-linalg,linalg-generalize-named-ops,linalg-fuse-elementwise-ops,empty-tensor-to-alloc-tensor,canonicalize,one-shot-bufferize,canonicalize,imex-remove-temporaries)" %s -o - | FileCheck %s

builtin.module {
    memref.global constant @static_mpi_rank : memref<index> = dense<10>
    mesh.mesh @mesh4x4(shape = 4x4)
    func.func @test_shard_propagate_insert_slice_2d(%arg0: tensor<1200x1200xi64>) {
        %s = mesh.sharding @mesh4x4 split_axes = [[0], [1]] : !mesh.sharding
        %0 = mesh.shard %arg0 to %s : tensor<1200x1200xi64>
        %1 = ndarray.subview %0[0, 0][1000, 1000][1, 1] : tensor<1200x1200xi64> to tensor<1000x1000xi64>
        %2 = ndarray.subview %0[0, 4][1000, 1000][1, 1] : tensor<1200x1200xi64> to tensor<1000x1000xi64>
        %3 = ndarray.subview %0[4, 0][1000, 1000][1, 1] : tensor<1200x1200xi64> to tensor<1000x1000xi64>
        %o1 = tensor.empty() : tensor<1000x1000xi64>
        %4 = linalg.add ins(%1, %2 : tensor<1000x1000xi64>, tensor<1000x1000xi64>) outs(%o1 : tensor<1000x1000xi64>) -> tensor<1000x1000xi64>
        %o2 = tensor.empty() : tensor<1000x1000xi64>
        %5 = linalg.add ins(%3, %4 : tensor<1000x1000xi64>, tensor<1000x1000xi64>) outs(%o2 : tensor<1000x1000xi64>) -> tensor<1000x1000xi64>
        ndarray.insert_slice %5 into %0[2, 2][1000, 1000][1, 1] : tensor<1000x1000xi64> into tensor<1200x1200xi64>
        return
    }
}
// CHECK: mesh.mesh @mesh4x4(shape = 4x4)
// CHECK-LABEL: func.func @test_shard_propagate_insert_slice_2d(
// CHECK-SAME: [[varg0:%.*]]: tensor<300x300xi64>) {
// CHECK-NEXT: [[vc91_i32:%.*]] = arith.constant 91 : i32
// CHECK-NEXT: [[vc9_i32:%.*]] = arith.constant 9 : i32
// CHECK-NEXT: [[vc11_i32:%.*]] = arith.constant 11 : i32
// CHECK-NEXT: [[vc6_i32:%.*]] = arith.constant 6 : i32
// CHECK-NEXT: [[vc14_i32:%.*]] = arith.constant 14 : i32
// CHECK-NEXT: [[v0:%.*]] = bufferization.to_memref [[varg0]] : tensor<300x300xi64> to memref<300x300xi64, strided<[?, ?], offset: ?>>
// CHECK-NEXT: [[valloc:%.*]] = memref.alloc() {alignment = 64 : i64} : memref<304x304xi64>
// CHECK-NEXT: [[vsubview:%.*]] = memref.subview [[valloc]][2, 2] [300, 300] [1, 1] : memref<304x304xi64> to memref<300x300xi64, strided<[304, 1], offset: 610>>
// CHECK-NEXT: memref.copy [[v0]], [[vsubview]] : memref<300x300xi64, strided<[?, ?], offset: ?>> to memref<300x300xi64, strided<[304, 1], offset: 610>>
// CHECK-NEXT: [[vsubview_0:%.*]] = memref.subview [[valloc]][2, 0] [300, 2] [1, 1] : memref<304x304xi64> to memref<300x2xi64, strided<[304, 1], offset: 608>>
// CHECK-NEXT: [[vsubview_1:%.*]] = memref.subview [[valloc]][2, 300] [300, 2] [1, 1] : memref<304x304xi64> to memref<300x2xi64, strided<[304, 1], offset: 908>>
// CHECK-NEXT: memref.copy [[vsubview_1]], [[vsubview_0]] : memref<300x2xi64, strided<[304, 1], offset: 908>> to memref<300x2xi64, strided<[304, 1], offset: 608>>
// CHECK-NEXT: mpi.send([[vsubview_0]], [[vc91_i32]], [[vc9_i32]]) : memref<300x2xi64, strided<[304, 1], offset: 608>>, i32, i32
// CHECK-NEXT: mpi.recv([[vsubview_0]], [[vc91_i32]], [[vc11_i32]]) : memref<300x2xi64, strided<[304, 1], offset: 608>>, i32, i32
// CHECK-NEXT: [[valloc_2:%.*]] = memref.alloc() : memref<300x2xi64>
// CHECK-NEXT: [[vsubview_3:%.*]] = memref.subview [[valloc]][2, 2] [300, 2] [1, 1] : memref<304x304xi64> to memref<300x2xi64, strided<[304, 1], offset: 610>>
// CHECK-NEXT: memref.copy [[vsubview_3]], [[valloc_2]] : memref<300x2xi64, strided<[304, 1], offset: 610>> to memref<300x2xi64>
// CHECK-NEXT: mpi.send([[valloc_2]], [[vc91_i32]], [[vc11_i32]]) : memref<300x2xi64>, i32, i32
// CHECK-NEXT: mpi.recv([[valloc_2]], [[vc91_i32]], [[vc9_i32]]) : memref<300x2xi64>, i32, i32
// CHECK-NEXT: [[vsubview_4:%.*]] = memref.subview [[valloc]][2, 302] [300, 2] [1, 1] : memref<304x304xi64> to memref<300x2xi64, strided<[304, 1], offset: 910>>
// CHECK-NEXT: memref.copy [[valloc_2]], [[vsubview_4]] : memref<300x2xi64> to memref<300x2xi64, strided<[304, 1], offset: 910>>
// CHECK-NEXT: memref.dealloc [[valloc_2]] : memref<300x2xi64>
// CHECK-NEXT: [[vsubview_5:%.*]] = memref.subview [[valloc]][0, 0] [2, 304] [1, 1] : memref<304x304xi64> to memref<2x304xi64, strided<[304, 1]>>
// CHECK-NEXT: [[vsubview_6:%.*]] = memref.subview [[valloc]][300, 0] [2, 304] [1, 1] : memref<304x304xi64> to memref<2x304xi64, strided<[304, 1], offset: 91200>>
// CHECK-NEXT: memref.copy [[vsubview_6]], [[vsubview_5]] : memref<2x304xi64, strided<[304, 1], offset: 91200>> to memref<2x304xi64, strided<[304, 1]>>
// CHECK-NEXT: mpi.send([[vsubview_5]], [[vc91_i32]], [[vc6_i32]]) : memref<2x304xi64, strided<[304, 1]>>, i32, i32
// CHECK-NEXT: mpi.recv([[vsubview_5]], [[vc91_i32]], [[vc14_i32]]) : memref<2x304xi64, strided<[304, 1]>>, i32, i32
// CHECK-NEXT: [[valloc_7:%.*]] = memref.alloc() : memref<2x304xi64>
// CHECK-NEXT: [[vsubview_8:%.*]] = memref.subview [[valloc]][2, 0] [2, 304] [1, 1] : memref<304x304xi64> to memref<2x304xi64, strided<[304, 1], offset: 608>>
// CHECK-NEXT: memref.copy [[vsubview_8]], [[valloc_7]] : memref<2x304xi64, strided<[304, 1], offset: 608>> to memref<2x304xi64>
// CHECK-NEXT: mpi.send([[valloc_7]], [[vc91_i32]], [[vc14_i32]]) : memref<2x304xi64>, i32, i32
// CHECK-NEXT: mpi.recv([[valloc_7]], [[vc91_i32]], [[vc6_i32]]) : memref<2x304xi64>, i32, i32
// CHECK-NEXT: [[vsubview_9:%.*]] = memref.subview [[valloc]][302, 0] [2, 304] [1, 1] : memref<304x304xi64> to memref<2x304xi64, strided<[304, 1], offset: 91808>>
// CHECK-NEXT: memref.copy [[valloc_7]], [[vsubview_9]] : memref<2x304xi64> to memref<2x304xi64, strided<[304, 1], offset: 91808>>
// CHECK-NEXT: memref.dealloc [[valloc_7]] : memref<2x304xi64>
// CHECK-NEXT: [[vsubview_10:%.*]] = memref.subview [[valloc]][0, 0] [300, 300] [1, 1] : memref<304x304xi64> to memref<300x300xi64, strided<[304, 1]>>
// CHECK-NEXT: [[vsubview_11:%.*]] = memref.subview [[valloc]][0, 4] [300, 300] [1, 1] : memref<304x304xi64> to memref<300x300xi64, strided<[304, 1], offset: 4>>
// CHECK-NEXT: [[vsubview_12:%.*]] = memref.subview [[valloc]][4, 0] [300, 300] [1, 1] : memref<304x304xi64> to memref<300x300xi64, strided<[304, 1], offset: 1216>>
// CHECK-NEXT: [[valloc_13:%.*]] = memref.alloc() {alignment = 64 : i64} : memref<300x300xi64>
// CHECK-NEXT: linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins([[vsubview_12]], [[vsubview_10]], [[vsubview_11]] : memref<300x300xi64, strided<[304, 1], offset: 1216>>, memref<300x300xi64, strided<[304, 1]>>, memref<300x300xi64, strided<[304, 1], offset: 4>>) outs([[valloc_13]] : memref<300x300xi64>) {
// CHECK-NEXT: ^bb0([[vin:%.*]]: i64, [[vin_15:%.*]]: i64, [[vin_16:%.*]]: i64, [[vout:%.*]]: i64):
  // CHECK-NEXT: [[v1:%.*]] = arith.addi [[vin_15]], [[vin_16]] : i64
  // CHECK-NEXT: [[v2:%.*]] = arith.addi [[vin]], [[v1]] : i64
  // CHECK-NEXT: linalg.yield [[v2]] : i64
// CHECK-NEXT: }
// CHECK-NEXT: [[vsubview_14:%.*]] = memref.subview [[valloc]][2, 2] [300, 300] [1, 1] : memref<304x304xi64> to memref<300x300xi64, strided<[304, 1], offset: 610>>
// CHECK-NEXT: memref.copy [[valloc_13]], [[vsubview_14]] : memref<300x300xi64> to memref<300x300xi64, strided<[304, 1], offset: 610>>
// CHECK-NEXT: return
