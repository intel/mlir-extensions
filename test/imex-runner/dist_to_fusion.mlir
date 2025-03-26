// RUN: imex-opt --pass-pipeline="builtin.module(func.func(sharding-propagation),coalesce-shard-ops,canonicalize,func.func(mesh-spmdization),canonicalize,convert-mesh-to-mpi,canonicalize,convert-ndarray-to-linalg,linalg-generalize-named-ops,linalg-fuse-elementwise-ops,empty-tensor-to-alloc-tensor,canonicalize,one-shot-bufferize,canonicalize,imex-remove-temporaries)" %s -o - | FileCheck %s

builtin.module attributes {dlti.map = #dlti.map<"MPI:Implementation" = "MPICH", "MPI:comm_world_rank" = 0 : i32>}{
    memref.global constant @static_mpi_rank : memref<index> = dense<10>
    mesh.mesh @mesh4x4(shape = 4x4)
    func.func @test_shard_propagate_insert_slice_2d(%arg0: tensor<1200x1200xi64>) -> tensor<1200x1200xi64> {
        %s = mesh.sharding @mesh4x4 split_axes = [[0], [1]] : !mesh.sharding
        %0 = mesh.shard %arg0 to %s : tensor<1200x1200xi64>
        %1 = ndarray.subview %0[0, 0][1000, 1000][1, 1] : tensor<1200x1200xi64> to tensor<1000x1000xi64>
        %2 = ndarray.subview %0[0, 4][1000, 1000][1, 1] : tensor<1200x1200xi64> to tensor<1000x1000xi64>
        %3 = ndarray.subview %0[4, 0][1000, 1000][1, 1] : tensor<1200x1200xi64> to tensor<1000x1000xi64>
        %o1 = tensor.empty() : tensor<1000x1000xi64>
        %4 = linalg.add ins(%1, %2 : tensor<1000x1000xi64>, tensor<1000x1000xi64>) outs(%o1 : tensor<1000x1000xi64>) -> tensor<1000x1000xi64>
        %o2 = tensor.empty() : tensor<1000x1000xi64>
        %5 = linalg.add ins(%3, %4 : tensor<1000x1000xi64>, tensor<1000x1000xi64>) outs(%o2 : tensor<1000x1000xi64>) -> tensor<1000x1000xi64>
        %6 = ndarray.insert_slice %5 into %0[2, 2][1000, 1000][1, 1] : tensor<1000x1000xi64> into tensor<1200x1200xi64>
        return %6 : tensor<1200x1200xi64>
    }
}
// CHECK: mesh.mesh @mesh4x4(shape = 4x4)
// CHECK-LABEL: func.func @test_shard_propagate_insert_slice_2d(
// CHECK-SAME: [[varg0:%.*]]: tensor<300x300xi64>) -> tensor<304x304xi64> {
// CHECK: mpi.send
// CHECK: mpi.recv
// CHECK: mpi.send
// CHECK: mpi.recv
// CHECK: mpi.send
// CHECK: mpi.recv
// CHECK: mpi.send
// CHECK: mpi.recv
// CHECK: mpi.send
// CHECK: mpi.recv
// CHECK: mpi.send
// CHECK: mpi.recv
// CHECK: memref.subview
// CHECK: memref.subview
// CHECK: linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]}
// CHECK: ^bb0([[vin:%.*]]: i64, [[vin_40:%.*]]: i64, [[vin_41:%.*]]: i64, [[vout:%.*]]: i64):
  // CHECK: [[v5:%.*]] = arith.addi [[vin_40]], [[vin_41]] : i64
  // CHECK: [[v6:%.*]] = arith.addi [[vin]], [[v5]] : i64
  // CHECK: linalg.yield [[v6]] : i64
// CHECK: }
// CHECK: mpi.send
// CHECK: mpi.recv
// CHECK: mpi.send
// CHECK: mpi.recv
// CHECK: mpi.send
// CHECK: mpi.recv
// CHECK: mpi.send
// CHECK: mpi.recv
