// RUN: imex-opt --split-input-file --insert-gpu-copy %s -verify-diagnostics -o -| FileCheck %s

func.func @test_copy_gpu_region() -> (memref<?x?xi32>, memref<?x?xi32>) {
    %c5 = arith.constant 5 : index
    %alloc = memref.alloc(%c5, %c5) : memref<?x?xi32>
    %0 = region.env_region #region.gpu_env<device = "test"> -> memref<?x?xi32> {
        %memref = gpu.alloc  (%c5, %c5) : memref<?x?xi32>
        memref.copy %alloc, %memref : memref<?x?xi32> to memref<?x?xi32>
        region.env_region_yield %memref : memref<?x?xi32>
    }
    return %alloc, %0 : memref<?x?xi32>, memref<?x?xi32>
}
// CHECK-LABEL: func.func @test_copy_gpu_region
// CHECK: [[ALLOC:%.*]] = memref.alloc
// CHECK: region.env_region #region.gpu_env<device = "test">
// CHECK-NEXT: [[ALLOC2:%.*]] = gpu.alloc
// CHECK-NEXT: gpu.memcpy [[ALLOC2]], [[ALLOC]] : memref<?x?xi32>, memref<?x?xi32>
// CHECK-NEXT: region.env_region_yield [[ALLOC2]] : memref<?x?xi32>

func.func @test_copy_region() -> (memref<?x?xi32>, memref<?x?xi32>) {
    %c5 = arith.constant 5 : index
    %alloc = memref.alloc(%c5, %c5) : memref<?x?xi32>
    %0 = region.env_region "string_attr" -> memref<?x?xi32> {
        %memref = gpu.alloc  (%c5, %c5) : memref<?x?xi32>
        memref.copy %alloc, %memref : memref<?x?xi32> to memref<?x?xi32>
        region.env_region_yield %memref : memref<?x?xi32>
    }
    return %alloc, %0 : memref<?x?xi32>, memref<?x?xi32>
}
// CHECK-LABEL: func.func @test_copy_region
// CHECK: [[ALLOC:%.*]] = memref.alloc
// CHECK: region.env_region "string_attr"
// CHECK-NEXT: [[ALLOC2:%.*]] = gpu.alloc
// CHECK-NEXT: gpu.memcpy [[ALLOC2]], [[ALLOC]] : memref<?x?xi32>, memref<?x?xi32>
// CHECK-NEXT: region.env_region_yield [[ALLOC2]] : memref<?x?xi32>

func.func @test_copy() -> (memref<?x?xi32>, memref<?x?xi32>) {
    %c5 = arith.constant 5 : index
    %alloc = memref.alloc(%c5, %c5) : memref<?x?xi32>
    %memref = memref.alloc(%c5, %c5) : memref<?x?xi32>
    memref.copy %alloc, %memref : memref<?x?xi32> to memref<?x?xi32>
    return %alloc, %memref : memref<?x?xi32>, memref<?x?xi32>
}
// CHECK-LABEL: func.func @test_copy
// CHECK: [[ALLOC:%.*]] = memref.alloc
// CHECK: [[MEMREF:%.*]] = memref.alloc
// CHECK: memref.copy [[ALLOC]], [[MEMREF]] : memref<?x?xi32> to memref<?x?xi32>
