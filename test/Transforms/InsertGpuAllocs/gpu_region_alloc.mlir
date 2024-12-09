// RUN: imex-opt --insert-gpu-allocs='in-regions=1 host-shared=0' %s | FileCheck %s

func.func @test_region_alloc() {
  %0 = memref.alloc() {alignment = 128 : i64} : memref<2x5xf32>
  %1 = region.env_region #region.gpu_env<device = "XeGPU"> -> memref<2x5xf32> {
    %2 = memref.alloc() {alignment = 128 : i64} : memref<2x5xf32>
    region.env_region_yield %2 : memref<2x5xf32>
  }
  memref.dealloc %0 : memref<2x5xf32>
  region.env_region #region.gpu_env<device = "XeGPU"> {
    memref.dealloc %1: memref<2x5xf32>
    region.env_region_yield
  }
  return
}
// CHECK-LABEL: func.func @test_region_alloc
// CHECK-NEXT: memref.alloc() {alignment = 128 : i64} : memref<2x5xf32>
// CHECK-NEXT: region.env_region #region.gpu_env<device = "XeGPU"> -> memref<2x5xf32> {
// CHECK-NEXT: gpu.alloc () : memref<2x5xf32>
// CHECK-NEXT: region.env_region_yield %memref : memref<2x5xf32>
// CHECK-NEXT: }
// CHECK: memref.dealloc %alloc : memref<2x5xf32>
// CHECK-NEXT: region.env_region #region.gpu_env<device = "XeGPU"> {
// CHECK-NEXT: gpu.dealloc  %0 : memref<2x5xf32>
// CHECK-NEXT: }
// CHECK: return
