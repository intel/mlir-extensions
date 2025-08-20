// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck


module @max_pool_1d attributes {gpu.container_module} {
  memref.global "private" constant @__constant_3xi64 : memref<3xi64> = dense<[1, 2, 3]>
  func.func @main() attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_3xi64 : memref<3xi64>
    scf.for %arg0 = %c0 to %c100 step %c1 {
      %1 = func.call @test(%0) : (memref<3xi64>) -> memref<1xi64>
      %cast = memref.cast %1 : memref<1xi64> to memref<*xi64>
      func.call @printMemrefI64(%cast) : (memref<*xi64>) -> ()
      // CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
      // CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [1] strides = {{.*}} data =
      // CHECK:   3
    }
    return
  }
  func.func private @printMemrefI64(memref<*xi64>) attributes {llvm.emit_c_interface}
  func.func @test(%arg0: memref<3xi64>) -> memref<1xi64> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %memref = gpu.alloc  host_shared () : memref<3xi64>
    memref.copy %arg0, %memref : memref<3xi64> to memref<3xi64>
    %memref_0 = gpu.alloc  host_shared () : memref<1xi64>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%c0_i64 : i64, %memref_0 : memref<1xi64>, %c0 : index)
    %memref_1 = gpu.alloc  host_shared () : memref<1xi64>
    memref.copy %memref_0, %memref_1 : memref<1xi64> to memref<1xi64>
    gpu.launch_func  @test_kernel_0::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<3xi64>, %memref_1 : memref<1xi64>, %c0 : index, %c3 : index, %c1 : index)
    gpu.dealloc  %memref_0 : memref<1xi64>
    gpu.dealloc  %memref : memref<3xi64>
    return %memref_1 : memref<1xi64>
  }
  spirv.module @__spv__test_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.func @test_kernel(%arg0: i64, %arg1: !spirv.ptr<!spirv.array<1 x i64>, CrossWorkgroup>, %arg2: i64) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, workgroup_attributions = 0 : i64} {
      %cst0_i64 = spirv.Constant 0 : i64
      %cst1_i64 = spirv.Constant 1 : i64
      %0 = spirv.IMul %cst1_i64, %arg2 : i64
      %1 = spirv.IAdd %cst0_i64, %0 : i64
      %2 = spirv.AccessChain %arg1[%1] : !spirv.ptr<!spirv.array<1 x i64>, CrossWorkgroup>, i64 -> !spirv.ptr<i64, CrossWorkgroup>
      spirv.Store "CrossWorkgroup" %2, %arg0 : i64
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: i64, %arg1: memref<1xi64>, %arg2: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      memref.store %arg0, %arg1[%arg2] : memref<1xi64>
      gpu.return
    }
  }
  spirv.module @__spv__test_kernel_0 Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<3 x i64>, CrossWorkgroup>, %arg1: !spirv.ptr<!spirv.array<1 x i64>, CrossWorkgroup>, %arg2: i64, %arg3: i64, %arg4: i64) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, workgroup_attributions = 0 : i64} {
      spirv.mlir.loop {
        spirv.Branch ^bb1(%arg2 : i64)
      ^bb1(%0: i64):  // 2 preds: ^bb0, ^bb2
        %1 = spirv.SLessThan %0, %arg3 : i64
        spirv.BranchConditional %1, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        %cst0_i64 = spirv.Constant 0 : i64
        %cst1_i64 = spirv.Constant 1 : i64
        %2 = spirv.IMul %cst1_i64, %0 : i64
        %3 = spirv.IAdd %cst0_i64, %2 : i64
        %4 = spirv.AccessChain %arg0[%3] : !spirv.ptr<!spirv.array<3 x i64>, CrossWorkgroup>, i64 -> !spirv.ptr<i64, CrossWorkgroup>
        %5 = spirv.Load "CrossWorkgroup" %4 : i64
        %cst0_i64_0 = spirv.Constant 0 : i64
        %cst1_i64_1 = spirv.Constant 1 : i64
        %6 = spirv.IMul %cst1_i64_1, %arg2 : i64
        %7 = spirv.IAdd %cst0_i64_0, %6 : i64
        %8 = spirv.AccessChain %arg1[%7] : !spirv.ptr<!spirv.array<1 x i64>, CrossWorkgroup>, i64 -> !spirv.ptr<i64, CrossWorkgroup>
        %9 = spirv.Load "CrossWorkgroup" %8 : i64
        %10 = spirv.UGreaterThan %9, %5 : i64
        %11 = spirv.Select %10, %9, %5 : i1, i64
        %cst0_i64_2 = spirv.Constant 0 : i64
        %cst1_i64_3 = spirv.Constant 1 : i64
        %12 = spirv.IMul %cst1_i64_3, %arg2 : i64
        %13 = spirv.IAdd %cst0_i64_2, %12 : i64
        %14 = spirv.AccessChain %arg1[%13] : !spirv.ptr<!spirv.array<1 x i64>, CrossWorkgroup>, i64 -> !spirv.ptr<i64, CrossWorkgroup>
        spirv.Store "CrossWorkgroup" %14, %11 : i64
        %15 = spirv.IAdd %0, %arg4 : i64
        spirv.Branch ^bb1(%15 : i64)
      ^bb3:  // pred: ^bb1
        spirv.mlir.merge
      }
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel
  }
  gpu.module @test_kernel_0 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<3xi64>, %arg1: memref<1xi64>, %arg2: index, %arg3: index, %arg4: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      scf.for %arg5 = %arg2 to %arg3 step %arg4 {
        %0 = memref.load %arg0[%arg5] : memref<3xi64>
        %1 = memref.load %arg1[%arg2] : memref<1xi64>
        %2 = arith.cmpi ugt, %1, %0 : i64
        %3 = arith.select %2, %1, %0 : i64
        memref.store %3, %arg1[%arg2] : memref<1xi64>
      }
      gpu.return
    }
  }
}
