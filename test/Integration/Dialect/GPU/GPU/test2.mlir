// RUN: %{python_executable} %{imex_tools_dir}/imex-runner.py -i %s -a -b --pass-pipeline-file=%p/gpu-to-spirv.pp -n | FileCheck %s

module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, #spv.resource_limits<>>} {
  memref.global "private" constant @__constant_2x5xf32_0 : memref<2x5xf32, #spv.storage_class<CrossWorkgroup>> = dense<[[1.000000e+01, 9.000000e+00, 8.000000e+00, 7.000000e+00, 6.000000e+00], [5.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00]]>
  memref.global "private" constant @__constant_2x5xf32 : memref<2x5xf32, #spv.storage_class<CrossWorkgroup>> = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00], [6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01]]>
  func.func @addt(%arg0: memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>, %arg1: memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>) -> memref<2x5xf32, #spv.storage_class<CrossWorkgroup>> {
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () {gpu.alloc_shared} : memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>
    memref.copy %arg1, %memref : memref<2x5xf32, #spv.storage_class<CrossWorkgroup>> to memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>
    %memref_0 = gpu.alloc  () {gpu.alloc_shared} : memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>
    memref.copy %arg0, %memref_0 : memref<2x5xf32, #spv.storage_class<CrossWorkgroup>> to memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>
    %memref_1 = gpu.alloc  () {gpu.alloc_shared} : memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>
    gpu.launch_func  @addt_kernel::@addt_kernel blocks in (%c2, %c5, %c1) threads in (%c1, %c1, %c1) args(%memref_0 : memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>, %memref : memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>, %memref_1 : memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>)
    gpu.dealloc  %memref_1 : memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>
    gpu.dealloc  %memref_0 : memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>
    gpu.dealloc  %memref : memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>
    return %memref_1 : memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>
  }
  gpu.module @addt_kernel {
    gpu.func @addt_kernel(%arg0: memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>, %arg1: memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>, %arg2: memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>) kernel attributes {spv.entry_point_abi = #spv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = memref.load %arg0[%0, %1] : memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>
      %3 = memref.load %arg1[%0, %1] : memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>
      %4 = arith.addf %2, %3 : f32
      memref.store %4, %arg2[%0, %1] : memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>
      gpu.return
    }
  }
  func.func @main() {
    %0 = memref.get_global @__constant_2x5xf32 : memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>
    %1 = memref.get_global @__constant_2x5xf32_0 : memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>
    %2 = call @addt(%0, %1) : (memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>, memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>) -> memref<2x5xf32, #spv.storage_class<CrossWorkgroup>>
    %3 = memref.cast %2 : memref<2x5xf32, #spv.storage_class<CrossWorkgroup>> to memref<*xf32, #spv.storage_class<CrossWorkgroup>>
    call @printMemrefF32(%3) : (memref<*xf32, #spv.storage_class<CrossWorkgroup>>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32, #spv.storage_class<CrossWorkgroup>>)
}
