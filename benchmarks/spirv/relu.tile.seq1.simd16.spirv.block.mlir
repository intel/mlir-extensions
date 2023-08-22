// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime
module attributes {gpu.container_module, torch.debug_module_name = "ReLU"} {
  memref.global "private" constant @__constant_512x640x20x15xf32 : memref<512x640x20x15xf32> = dense<1.300000e+00>
  func.func @forward(%arg0: memref<512x640x20x15xf32>) -> memref<512x640x20x15xf32> attributes {llvm.emit_c_interface} {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c96000 = arith.constant 96000 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %memref = gpu.alloc  host_shared () : memref<512x640x20x15xf32>
    memref.copy %arg0, %memref : memref<512x640x20x15xf32> to memref<512x640x20x15xf32>
    %collapse_shape = memref.collapse_shape %memref [[0, 1, 2, 3]] : memref<512x640x20x15xf32> into memref<98304000xf32>
    %memref_0 = gpu.alloc  host_shared () : memref<98304000xf32>
    gpu.launch_func  @forward_kernel::@forward_kernel blocks in (%c96000, %c1, %c1) threads in (%c16, %c64, %c1) args(%c1024 : index, %c64 : index, %collapse_shape : memref<98304000xf32>, %cst : f32, %memref_0 : memref<98304000xf32>)
    %expand_shape = memref.expand_shape %memref_0 [[0, 1, 2, 3]] : memref<98304000xf32> into memref<512x640x20x15xf32>
    gpu.dealloc  %memref : memref<512x640x20x15xf32>
    return %expand_shape : memref<512x640x20x15xf32>
  }
    spirv.module @__spv__forward_kernel Physical64 OpenCL requires  #spirv.vce<v1.1, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, Groups, SubgroupDispatch, SubgroupBufferBlockIOINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_subgroups, SPV_KHR_no_integer_wrap_decoration]> {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.GlobalVariable @__builtin_var_SubgroupId__ built_in("SubgroupId") : !spirv.ptr<i32, Input>
    spirv.func @forward_kernel(%arg0: i64, %arg1: i64, %arg2: !spirv.ptr<!spirv.array<98304000 x f32>, CrossWorkgroup>, %arg3: f32, %arg4: !spirv.ptr<!spirv.array<98304000 x f32>, CrossWorkgroup>) "None" attributes {gpu.known_block_size = array<i32: 32, 16, 1>, gpu.known_grid_size = array<i32: 96000, 1, 1>, workgroup_attributions = 0 : i64} {
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %__builtin_var_SubgroupId___addr = spirv.mlir.addressof @__builtin_var_SubgroupId__ : !spirv.ptr<i32, Input>
      %2 = spirv.Load "Input" %__builtin_var_SubgroupId___addr : i32
      %3 = spirv.UConvert %2 : i32 to i64
      %4 = spirv.IMul %1, %arg0 : i64
      %5 = spirv.IMul %3, %arg1 : i64
      %6 = spirv.IAdd %4, %5 : i64
      %cst0_i64 = spirv.Constant 0 : i64
      %cst1_i64 = spirv.Constant 1 : i64
      %7 = spirv.IMul %cst1_i64, %6 : i64
      %8 = spirv.IAdd %cst0_i64, %7 : i64
      %9 = spirv.AccessChain %arg2[%8] : !spirv.ptr<!spirv.array<98304000 x f32>, CrossWorkgroup>, i64
      %91 = spirv.Bitcast %9 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<i32, CrossWorkgroup>
      %92 = spirv.INTEL.SubgroupBlockRead "CrossWorkgroup" %91 : i32
      %10 = spirv.Bitcast %92 : i32 to f32
      %11 = spirv.FUnordGreaterThan %10, %arg3 : f32
      %12 = spirv.Select %11, %10, %arg3 : i1, f32
      %cst0_i64_0 = spirv.Constant 0 : i64
      %cst1_i64_1 = spirv.Constant 1 : i64
      %13 = spirv.IMul %cst1_i64_1, %6 : i64
      %14 = spirv.IAdd %cst0_i64_0, %13 : i64
      %15 = spirv.AccessChain %arg4[%14] : !spirv.ptr<!spirv.array<98304000 x f32>, CrossWorkgroup>, i64
      %151 = spirv.Bitcast %15 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<i32, CrossWorkgroup>
      %122 = spirv.Bitcast %12 : f32 to i32
      spirv.INTEL.SubgroupBlockWrite "CrossWorkgroup" %151, %122 : i32
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @forward_kernel, @__builtin_var_WorkgroupId__, @__builtin_var_SubgroupId__
  spirv.ExecutionMode @forward_kernel "SubgroupSize", 16
  spirv.ExecutionMode @forward_kernel "ContractionOff"
  }
  gpu.module @forward_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.1, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @forward_kernel(%arg0: index, %arg1: index, %arg2: memref<98304000xf32>, %arg3: f32, %arg4: memref<98304000xf32>) kernel attributes {gpu.known_block_size = array<i32: 32, 32, 1>, gpu.known_grid_size = array<i32: 96000, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  y
      %2 = arith.muli %0, %arg0 : index
      %3 = arith.muli %1, %arg1 : index
      %4 = arith.addi %2, %3 : index
      %5 = memref.load %arg2[%4] : memref<98304000xf32>
      %6 = arith.cmpf ugt, %5, %arg3 : f32
      %7 = arith.select %6, %5, %arg3 : f32
      memref.store %7, %arg4[%4] : memref<98304000xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_512x640x20x15xf32 : memref<512x640x20x15xf32>
    %1 = call @forward(%0) : (memref<512x640x20x15xf32>) -> memref<512x640x20x15xf32>
    return
  }
}
