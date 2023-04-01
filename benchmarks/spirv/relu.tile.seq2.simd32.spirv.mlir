// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime
module attributes {gpu.container_module, torch.debug_module_name = "ReLU"} {
  memref.global "private" constant @__constant_512x640x20x15xf32 : memref<512x640x20x15xf32> = dense<1.300000e+00>
  func.func @forward(%arg0: memref<512x640x20x15xf32>) -> memref<512x640x20x15xf32> attributes {llvm.emit_c_interface} {
    %c32 = arith.constant 32 : index
    %c48000 = arith.constant 48000 : index
    %c64 = arith.constant 64 : index
    %c2048 = arith.constant 2048 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %memref = gpu.alloc  host_shared () : memref<512x640x20x15xf32>
    memref.copy %arg0, %memref : memref<512x640x20x15xf32> to memref<512x640x20x15xf32>
    %collapse_shape = memref.collapse_shape %memref [[0, 1, 2, 3]] : memref<512x640x20x15xf32> into memref<98304000xf32>
    %memref_0 = gpu.alloc  host_shared () : memref<98304000xf32>
    gpu.launch_func  @forward_kernel::@forward_kernel blocks in (%c48000, %c1, %c1) threads in (%c32, %c32, %c1) args(%c2048 : index, %c64 : index, %c32 : index, %collapse_shape : memref<98304000xf32>, %cst : f32, %memref_0 : memref<98304000xf32>, %c0 : index, %c2 : index, %c1 : index)
    %expand_shape = memref.expand_shape %memref_0 [[0, 1, 2, 3]] : memref<98304000xf32> into memref<512x640x20x15xf32>
    gpu.dealloc  %memref : memref<512x640x20x15xf32>
    return %expand_shape : memref<512x640x20x15xf32>
  }
  spirv.module @__spv__forward_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @forward_kernel(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: !spirv.ptr<!spirv.array<98304000 x f32>, CrossWorkgroup>, %arg4: f32, %arg5: !spirv.ptr<!spirv.array<98304000 x f32>, CrossWorkgroup>, %arg6: i64, %arg7: i64, %arg8: i64) "None" attributes {gpu.known_block_size = array<i32: 32, 32, 1>, gpu.known_grid_size = array<i32: 48000, 1, 1>, workgroup_attributions = 0 : i64} {
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %__builtin_var_LocalInvocationId___addr = spirv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spirv.ptr<vector<3xi64>, Input>
      %2 = spirv.Load "Input" %__builtin_var_LocalInvocationId___addr : vector<3xi64>
      %3 = spirv.CompositeExtract %2[0 : i32] : vector<3xi64>
      %__builtin_var_LocalInvocationId___addr_0 = spirv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spirv.ptr<vector<3xi64>, Input>
      %4 = spirv.Load "Input" %__builtin_var_LocalInvocationId___addr_0 : vector<3xi64>
      %5 = spirv.CompositeExtract %4[1 : i32] : vector<3xi64>
      spirv.mlir.loop {
        spirv.Branch ^bb1(%arg6 : i64)
      ^bb1(%6: i64):  // 2 preds: ^bb0, ^bb2
        %7 = spirv.SLessThan %6, %arg7 : i64
        spirv.BranchConditional %7, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        %8 = spirv.IMul %1, %arg0 : i64
        %9 = spirv.IMul %5, %arg1 : i64
        %10 = spirv.IAdd %8, %9 : i64
        %11 = spirv.IAdd %10, %3 : i64
        %12 = spirv.IMul %6, %arg2 : i64
        %13 = spirv.IAdd %11, %12 : i64
        %cst0_i64 = spirv.Constant 0 : i64
        %cst1_i64 = spirv.Constant 1 : i64
        %14 = spirv.IMul %cst1_i64, %13 : i64
        %15 = spirv.IAdd %cst0_i64, %14 : i64
        %16 = spirv.AccessChain %arg3[%15] : !spirv.ptr<!spirv.array<98304000 x f32>, CrossWorkgroup>, i64
        %17 = spirv.Load "CrossWorkgroup" %16 : f32
        %18 = spirv.FUnordGreaterThan %17, %arg4 : f32
        %19 = spirv.Select %18, %17, %arg4 : i1, f32
        %cst0_i64_1 = spirv.Constant 0 : i64
        %cst1_i64_2 = spirv.Constant 1 : i64
        %20 = spirv.IMul %cst1_i64_2, %13 : i64
        %21 = spirv.IAdd %cst0_i64_1, %20 : i64
        %22 = spirv.AccessChain %arg5[%21] : !spirv.ptr<!spirv.array<98304000 x f32>, CrossWorkgroup>, i64
        spirv.Store "CrossWorkgroup" %22, %19 : f32
        %23 = spirv.IAdd %6, %arg8 : i64
        spirv.Branch ^bb1(%23 : i64)
      ^bb3:  // pred: ^bb1
        spirv.mlir.merge
      }
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @forward_kernel, @__builtin_var_WorkgroupId__, @__builtin_var_LocalInvocationId__
  }
  gpu.module @forward_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @forward_kernel(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<98304000xf32>, %arg4: f32, %arg5: memref<98304000xf32>, %arg6: index, %arg7: index, %arg8: index) kernel attributes {gpu.known_block_size = array<i32: 32, 32, 1>, gpu.known_grid_size = array<i32: 48000, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.thread_id  y
      scf.for %arg9 = %arg6 to %arg7 step %arg8 {
        %3 = arith.muli %0, %arg0 : index
        %4 = arith.muli %2, %arg1 : index
        %5 = arith.addi %3, %4 : index
        %6 = arith.addi %5, %1 : index
        %7 = arith.muli %arg9, %arg2 : index
        %8 = arith.addi %6, %7 : index
        %9 = memref.load %arg3[%8] : memref<98304000xf32>
        %10 = arith.cmpf ugt, %9, %arg4 : f32
        %11 = arith.select %10, %9, %arg4 : f32
        memref.store %11, %arg5[%8] : memref<98304000xf32>
      }
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_512x640x20x15xf32 : memref<512x640x20x15xf32>
    %1 = call @forward(%0) : (memref<512x640x20x15xf32>) -> memref<512x640x20x15xf32>
    return
  }
}
