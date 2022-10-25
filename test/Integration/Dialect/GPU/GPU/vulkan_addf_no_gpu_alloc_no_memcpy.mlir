// RUN: mlir-vulkan-runner %s --mlir-print-ir-before-all --mlir-print-ir-after-all --shared-libs=%vulkan_runtime_wrappers,%mlir_runner_utils --entry-point-result=void | FileCheck %s

// CHECK: [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0]
module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  memref.global "private" constant @__constant_2x5xf32_0 : memref<2x5xf32> = dense<[[1.000000e+01, 9.000000e+00, 8.000000e+00, 7.000000e+00, 6.000000e+00], [5.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00]]>
  memref.global "private" constant @__constant_2x5xf32 : memref<2x5xf32> = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00], [6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01]]>
  func.func @addt(%arg0: memref<2x5xf32>, %arg1: memref<2x5xf32>) -> memref<2x5xf32> {
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %memref0 = memref.alloc() : memref<2x5xf32>
    memref.copy %arg0, %memref0 : memref<2x5xf32> to memref<2x5xf32>
    %memref1 = memref.alloc() : memref<2x5xf32>
    memref.copy %arg1, %memref1 : memref<2x5xf32> to memref<2x5xf32>
    %memref2 = memref.alloc() : memref<2x5xf32>
    gpu.launch_func  @addt_kernel::@addt_kernel blocks in (%c2, %c5, %c1) threads in (%c1, %c1, %c1) args(%memref0 : memref<2x5xf32>, %memref1 : memref<2x5xf32>, %memref2 : memref<2x5xf32>)
    return %memref2 : memref<2x5xf32>
    //%alloc = memref.alloc() : memref<2x5xf32>
    //memref.copy %memref2, %alloc : memref<2x5xf32> to memref<2x5xf32>
    //return %alloc : memref<2x5xf32>
  }
  gpu.module @addt_kernel {
    gpu.func @addt_kernel(%arg0: memref<2x5xf32>, %arg1: memref<2x5xf32>, %arg2: memref<2x5xf32>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[1, 1, 1]>: vector<3xi32>>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = memref.load %arg0[%0, %1] : memref<2x5xf32>
      %3 = memref.load %arg1[%0, %1] : memref<2x5xf32>
      %4 = arith.addf %2, %3 : f32
      memref.store %4, %arg2[%0, %1] : memref<2x5xf32>
      gpu.return
    }
  }
  func.func @main() {
    %0 = memref.get_global @__constant_2x5xf32 : memref<2x5xf32>
    %1 = memref.get_global @__constant_2x5xf32_0 : memref<2x5xf32>
    %2 = call @addt(%0, %1) : (memref<2x5xf32>, memref<2x5xf32>) -> memref<2x5xf32>
    %3 = memref.cast %2 : memref<2x5xf32> to memref<*xf32>
    call @printMemrefF32(%3) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>)
}
