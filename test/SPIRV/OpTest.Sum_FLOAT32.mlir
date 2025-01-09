// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck


module @sum attributes {gpu.container_module} {
  memref.global "private" constant @__constant_10x20xf32 : memref<10x20xf32> = dense<[
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
  ]>

  memref.global "private" constant @__constant_1xf32 : memref<1xf32> = dense<[2100.0]>

  func.func @test(%arg0: memref<10x20xf32>) -> memref<f32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c20 = arith.constant 20 : index
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %memref = gpu.alloc  host_shared () : memref<10x20xf32>
    memref.copy %arg0, %memref : memref<10x20xf32> to memref<10x20xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc[] : memref<f32>
    %memref_0 = gpu.alloc  host_shared () : memref<f32>
    memref.copy %alloc, %memref_0 : memref<f32> to memref<f32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<10x20xf32>, %memref_0 : memref<f32>, %c0 : index, %c20 : index, %c1 : index, %c10 : index)
    gpu.dealloc  %memref : memref<10x20xf32>
    return %memref_0 : memref<f32>
  }
  spirv.module @__spv__test_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, %arg1: !spirv.ptr<!spirv.array<1 x f32>, CrossWorkgroup>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, workgroup_attributions = 0 : i64} {
      spirv.mlir.loop {
        spirv.Branch ^bb1(%arg2 : i64)
      ^bb1(%0: i64):  // 2 preds: ^bb0, ^bb2
        %1 = spirv.SLessThan %0, %arg5 : i64
        spirv.BranchConditional %1, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        spirv.mlir.loop {
          spirv.Branch ^bb1(%arg2 : i64)
        ^bb1(%3: i64):  // 2 preds: ^bb0, ^bb2
          %4 = spirv.SLessThan %3, %arg3 : i64
          spirv.BranchConditional %4, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %cst0_i64 = spirv.Constant 0 : i64
          %cst20_i64 = spirv.Constant 20 : i64
          %5 = spirv.IMul %cst20_i64, %0 : i64
          %6 = spirv.IAdd %cst0_i64, %5 : i64
          %cst1_i64 = spirv.Constant 1 : i64
          %7 = spirv.IMul %cst1_i64, %3 : i64
          %8 = spirv.IAdd %6, %7 : i64
          %9 = spirv.AccessChain %arg0[%8] : !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
          %10 = spirv.Load "CrossWorkgroup" %9 : f32
          %cst0_i64_0 = spirv.Constant 0 : i64
          %11 = spirv.AccessChain %arg1[%cst0_i64_0] : !spirv.ptr<!spirv.array<1 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
          %12 = spirv.Load "CrossWorkgroup" %11 : f32
          %13 = spirv.FAdd %12, %10 : f32
          %cst0_i64_1 = spirv.Constant 0 : i64
          %14 = spirv.AccessChain %arg1[%cst0_i64_1] : !spirv.ptr<!spirv.array<1 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
          spirv.Store "CrossWorkgroup" %14, %13 : f32
          %15 = spirv.IAdd %3, %arg4 : i64
          spirv.Branch ^bb1(%15 : i64)
        ^bb3:  // pred: ^bb1
          spirv.mlir.merge
        }
        %2 = spirv.IAdd %0, %arg4 : i64
        spirv.Branch ^bb1(%2 : i64)
      ^bb3:  // pred: ^bb1
        spirv.mlir.merge
      }
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<10x20xf32>, %arg1: memref<f32>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      scf.for %arg6 = %arg2 to %arg5 step %arg4 {
        scf.for %arg7 = %arg2 to %arg3 step %arg4 {
          %0 = memref.load %arg0[%arg6, %arg7] : memref<10x20xf32>
          %1 = memref.load %arg1[] : memref<f32>
          %2 = arith.addf %1, %0 : f32
          memref.store %2, %arg1[] : memref<f32>
        }
      }
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index

    %ref_result = memref.get_global @__constant_1xf32 : memref<1xf32>
    %0 = memref.get_global @__constant_10x20xf32 : memref<10x20xf32>
    %unranked_ref_result = memref.cast %ref_result : memref<1xf32> to memref<*xf32>

    scf.for %arg0 = %c0 to %c100 step %c1 {
      %1 = func.call @test(%0) : (memref<10x20xf32>) -> memref<f32>
      %cast = memref.cast %1 : memref<f32> to memref<*xf32>
      func.call @printAllcloseF32(%cast, %unranked_ref_result) : (memref<*xf32>, memref<*xf32>) -> ()
      func.call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
      // CHECK:   [ALLCLOSE: TRUE]
    }
    return
  }
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
