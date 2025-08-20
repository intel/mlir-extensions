// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck


module @complex_conv_2d attributes {gpu.container_module} {
  memref.global "private" constant @__constant_3x3x3x3x32xf32 : memref<3x3x3x3x32xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_1x224x224x3x3xf32 : memref<1x224x224x3x3xf32> = dense<1.000000e+00>

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index

    %0 = memref.get_global @__constant_1x224x224x3x3xf32 : memref<1x224x224x3x3xf32>
    %1 = memref.get_global @__constant_3x3x3x3x32xf32 : memref<3x3x3x3x32xf32>
    %2 = func.call @test(%0, %1) : (memref<1x224x224x3x3xf32>, memref<3x3x3x3x32xf32>) -> memref<1x112x112x3x32xf32>
    %cast = memref.cast %2 : memref<1x112x112x3x32xf32> to memref<*xf32>
    func.call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    // CHECK: [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
    return
  }
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func @test(%arg0: memref<1x224x224x3x3xf32>, %arg1: memref<3x3x3x3x32xf32>) -> memref<1x112x112x3x32xf32> attributes {llvm.emit_c_interface} {
    %c112 = arith.constant 112 : index
    %c1 = arith.constant 1 : index
    %c229 = arith.constant 229 : index
    %c224 = arith.constant 224 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c32 = arith.constant 32 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %memref = gpu.alloc  host_shared () : memref<3x3x3x3x32xf32>
    memref.copy %arg1, %memref : memref<3x3x3x3x32xf32> to memref<3x3x3x3x32xf32>
    %memref_0 = gpu.alloc  host_shared () : memref<1x224x224x3x3xf32>
    memref.copy %arg0, %memref_0 : memref<1x224x224x3x3xf32> to memref<1x224x224x3x3xf32>
    %memref_1 = gpu.alloc  host_shared () : memref<1x224x224x3x3xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c224, %c224) threads in (%c1, %c1, %c1) args(%memref_0 : memref<1x224x224x3x3xf32>, %c0 : index, %memref_1 : memref<1x224x224x3x3xf32>, %c3 : index, %c1 : index)
    %memref_2 = gpu.alloc  host_shared () : memref<1x229x229x3x3xf32>
    gpu.launch_func  @test_kernel_0::@test_kernel blocks in (%c1, %c229, %c229) threads in (%c1, %c1, %c1) args(%cst : f32, %memref_2 : memref<1x229x229x3x3xf32>, %c0 : index, %c3 : index, %c1 : index)
    %memref_3 = gpu.alloc  host_shared () : memref<1x229x229x3x3xf32>
    memref.copy %memref_2, %memref_3 : memref<1x229x229x3x3xf32> to memref<1x229x229x3x3xf32>
    %subview = memref.subview %memref_3[0, 2, 2, 0, 0] [1, 224, 224, 3, 3] [1, 1, 1, 1, 1] : memref<1x229x229x3x3xf32> to memref<1x224x224x3x3xf32, strided<[471969, 2061, 9, 3, 1], offset: 4140>>
    memref.copy %memref_1, %subview : memref<1x224x224x3x3xf32> to memref<1x224x224x3x3xf32, strided<[471969, 2061, 9, 3, 1], offset: 4140>>
    %memref_4 = gpu.alloc  host_shared () : memref<1x112x112x3x32xf32>
    gpu.launch_func  @test_kernel_1::@test_kernel blocks in (%c1, %c112, %c112) threads in (%c1, %c1, %c1) args(%cst : f32, %memref_4 : memref<1x112x112x3x32xf32>, %c0 : index, %c32 : index, %c1 : index, %c3 : index)
    %memref_5 = gpu.alloc  host_shared () : memref<1x112x112x3x32xf32>
    memref.copy %memref_4, %memref_5 : memref<1x112x112x3x32xf32> to memref<1x112x112x3x32xf32>
    gpu.launch_func  @test_kernel_2::@test_kernel blocks in (%c1, %c112, %c112) threads in (%c1, %c1, %c1) args(%memref_3 : memref<1x229x229x3x3xf32>, %c0 : index, %memref : memref<3x3x3x3x32xf32>, %memref_5 : memref<1x112x112x3x32xf32>, %c3 : index, %c1 : index, %c32 : index)
    gpu.dealloc  %memref_1 : memref<1x224x224x3x3xf32>
    gpu.dealloc  %memref_2 : memref<1x229x229x3x3xf32>
    gpu.dealloc  %memref_4 : memref<1x112x112x3x32xf32>
    gpu.dealloc  %memref_0 : memref<1x224x224x3x3xf32>
    gpu.dealloc  %memref : memref<3x3x3x3x32xf32>
    return %memref_5 : memref<1x112x112x3x32xf32>
  }
  spirv.module @__spv__test_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<451584 x f32>, CrossWorkgroup>, %arg1: i64, %arg2: !spirv.ptr<!spirv.array<451584 x f32>, CrossWorkgroup>, %arg3: i64, %arg4: i64) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 224, 224>, workgroup_attributions = 0 : i64} {
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[1 : i32] : vector<3xi64>
      %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
      %3 = spirv.CompositeExtract %2[2 : i32] : vector<3xi64>
      spirv.mlir.loop {
        spirv.Branch ^bb1(%arg1 : i64)
      ^bb1(%4: i64):  // 2 preds: ^bb0, ^bb2
        %5 = spirv.SLessThan %4, %arg3 : i64
        spirv.BranchConditional %5, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        spirv.mlir.loop {
          spirv.Branch ^bb1(%arg1 : i64)
        ^bb1(%7: i64):  // 2 preds: ^bb0, ^bb2
          %8 = spirv.SLessThan %7, %arg3 : i64
          spirv.BranchConditional %8, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %cst0_i64 = spirv.Constant 0 : i64
          %cst451584_i64 = spirv.Constant 451584 : i64
          %9 = spirv.IMul %cst451584_i64, %arg1 : i64
          %10 = spirv.IAdd %cst0_i64, %9 : i64
          %cst2016_i64 = spirv.Constant 2016 : i64
          %11 = spirv.IMul %cst2016_i64, %1 : i64
          %12 = spirv.IAdd %10, %11 : i64
          %cst9_i64 = spirv.Constant 9 : i64
          %13 = spirv.IMul %cst9_i64, %3 : i64
          %14 = spirv.IAdd %12, %13 : i64
          %cst3_i64 = spirv.Constant 3 : i64
          %15 = spirv.IMul %cst3_i64, %4 : i64
          %16 = spirv.IAdd %14, %15 : i64
          %cst1_i64 = spirv.Constant 1 : i64
          %17 = spirv.IMul %cst1_i64, %7 : i64
          %18 = spirv.IAdd %16, %17 : i64
          %19 = spirv.AccessChain %arg0[%18] : !spirv.ptr<!spirv.array<451584 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
          %20 = spirv.Load "CrossWorkgroup" %19 : f32
          %cst0_i64_1 = spirv.Constant 0 : i64
          %cst451584_i64_2 = spirv.Constant 451584 : i64
          %21 = spirv.IMul %cst451584_i64_2, %arg1 : i64
          %22 = spirv.IAdd %cst0_i64_1, %21 : i64
          %cst2016_i64_3 = spirv.Constant 2016 : i64
          %23 = spirv.IMul %cst2016_i64_3, %1 : i64
          %24 = spirv.IAdd %22, %23 : i64
          %cst9_i64_4 = spirv.Constant 9 : i64
          %25 = spirv.IMul %cst9_i64_4, %3 : i64
          %26 = spirv.IAdd %24, %25 : i64
          %cst3_i64_5 = spirv.Constant 3 : i64
          %27 = spirv.IMul %cst3_i64_5, %4 : i64
          %28 = spirv.IAdd %26, %27 : i64
          %cst1_i64_6 = spirv.Constant 1 : i64
          %29 = spirv.IMul %cst1_i64_6, %7 : i64
          %30 = spirv.IAdd %28, %29 : i64
          %31 = spirv.AccessChain %arg2[%30] : !spirv.ptr<!spirv.array<451584 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
          spirv.Store "CrossWorkgroup" %31, %20 : f32
          %32 = spirv.IAdd %7, %arg4 : i64
          spirv.Branch ^bb1(%32 : i64)
        ^bb3:  // pred: ^bb1
          spirv.mlir.merge
        }
        %6 = spirv.IAdd %4, %arg4 : i64
        spirv.Branch ^bb1(%6 : i64)
      ^bb3:  // pred: ^bb1
        spirv.mlir.merge
      }
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<1x224x224x3x3xf32>, %arg1: index, %arg2: memref<1x224x224x3x3xf32>, %arg3: index, %arg4: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 224, 224>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  y
      %1 = gpu.block_id  z
      scf.for %arg5 = %arg1 to %arg3 step %arg4 {
        scf.for %arg6 = %arg1 to %arg3 step %arg4 {
          %2 = memref.load %arg0[%arg1, %0, %1, %arg5, %arg6] : memref<1x224x224x3x3xf32>
          memref.store %2, %arg2[%arg1, %0, %1, %arg5, %arg6] : memref<1x224x224x3x3xf32>
        }
      }
      gpu.return
    }
  }
  spirv.module @__spv__test_kernel_0 Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: f32, %arg1: !spirv.ptr<!spirv.array<471969 x f32>, CrossWorkgroup>, %arg2: i64, %arg3: i64, %arg4: i64) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 229, 229>, workgroup_attributions = 0 : i64} {
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[1 : i32] : vector<3xi64>
      %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
      %3 = spirv.CompositeExtract %2[2 : i32] : vector<3xi64>
      spirv.mlir.loop {
        spirv.Branch ^bb1(%arg2 : i64)
      ^bb1(%4: i64):  // 2 preds: ^bb0, ^bb2
        %5 = spirv.SLessThan %4, %arg3 : i64
        spirv.BranchConditional %5, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        spirv.mlir.loop {
          spirv.Branch ^bb1(%arg2 : i64)
        ^bb1(%7: i64):  // 2 preds: ^bb0, ^bb2
          %8 = spirv.SLessThan %7, %arg3 : i64
          spirv.BranchConditional %8, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %cst0_i64 = spirv.Constant 0 : i64
          %cst471969_i64 = spirv.Constant 471969 : i64
          %9 = spirv.IMul %cst471969_i64, %arg2 : i64
          %10 = spirv.IAdd %cst0_i64, %9 : i64
          %cst2061_i64 = spirv.Constant 2061 : i64
          %11 = spirv.IMul %cst2061_i64, %1 : i64
          %12 = spirv.IAdd %10, %11 : i64
          %cst9_i64 = spirv.Constant 9 : i64
          %13 = spirv.IMul %cst9_i64, %3 : i64
          %14 = spirv.IAdd %12, %13 : i64
          %cst3_i64 = spirv.Constant 3 : i64
          %15 = spirv.IMul %cst3_i64, %4 : i64
          %16 = spirv.IAdd %14, %15 : i64
          %cst1_i64 = spirv.Constant 1 : i64
          %17 = spirv.IMul %cst1_i64, %7 : i64
          %18 = spirv.IAdd %16, %17 : i64
          %19 = spirv.AccessChain %arg1[%18] : !spirv.ptr<!spirv.array<471969 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
          spirv.Store "CrossWorkgroup" %19, %arg0 : f32
          %20 = spirv.IAdd %7, %arg4 : i64
          spirv.Branch ^bb1(%20 : i64)
        ^bb3:  // pred: ^bb1
          spirv.mlir.merge
        }
        %6 = spirv.IAdd %4, %arg4 : i64
        spirv.Branch ^bb1(%6 : i64)
      ^bb3:  // pred: ^bb1
        spirv.mlir.merge
      }
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel_0 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: f32, %arg1: memref<1x229x229x3x3xf32>, %arg2: index, %arg3: index, %arg4: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 229, 229>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  y
      %1 = gpu.block_id  z
      scf.for %arg5 = %arg2 to %arg3 step %arg4 {
        scf.for %arg6 = %arg2 to %arg3 step %arg4 {
          memref.store %arg0, %arg1[%arg2, %0, %1, %arg5, %arg6] : memref<1x229x229x3x3xf32>
        }
      }
      gpu.return
    }
  }
  spirv.module @__spv__test_kernel_1 Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: f32, %arg1: !spirv.ptr<!spirv.array<1204224 x f32>, CrossWorkgroup>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 112, 112>, workgroup_attributions = 0 : i64} {
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[1 : i32] : vector<3xi64>
      %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
      %3 = spirv.CompositeExtract %2[2 : i32] : vector<3xi64>
      spirv.mlir.loop {
        spirv.Branch ^bb1(%arg2 : i64)
      ^bb1(%4: i64):  // 2 preds: ^bb0, ^bb2
        %5 = spirv.SLessThan %4, %arg5 : i64
        spirv.BranchConditional %5, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        spirv.mlir.loop {
          spirv.Branch ^bb1(%arg2 : i64)
        ^bb1(%7: i64):  // 2 preds: ^bb0, ^bb2
          %8 = spirv.SLessThan %7, %arg3 : i64
          spirv.BranchConditional %8, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %cst0_i64 = spirv.Constant 0 : i64
          %cst1204224_i64 = spirv.Constant 1204224 : i64
          %9 = spirv.IMul %cst1204224_i64, %arg2 : i64
          %10 = spirv.IAdd %cst0_i64, %9 : i64
          %cst10752_i64 = spirv.Constant 10752 : i64
          %11 = spirv.IMul %cst10752_i64, %1 : i64
          %12 = spirv.IAdd %10, %11 : i64
          %cst96_i64 = spirv.Constant 96 : i64
          %13 = spirv.IMul %cst96_i64, %3 : i64
          %14 = spirv.IAdd %12, %13 : i64
          %cst32_i64 = spirv.Constant 32 : i64
          %15 = spirv.IMul %cst32_i64, %4 : i64
          %16 = spirv.IAdd %14, %15 : i64
          %cst1_i64 = spirv.Constant 1 : i64
          %17 = spirv.IMul %cst1_i64, %7 : i64
          %18 = spirv.IAdd %16, %17 : i64
          %19 = spirv.AccessChain %arg1[%18] : !spirv.ptr<!spirv.array<1204224 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
          spirv.Store "CrossWorkgroup" %19, %arg0 : f32
          %20 = spirv.IAdd %7, %arg4 : i64
          spirv.Branch ^bb1(%20 : i64)
        ^bb3:  // pred: ^bb1
          spirv.mlir.merge
        }
        %6 = spirv.IAdd %4, %arg4 : i64
        spirv.Branch ^bb1(%6 : i64)
      ^bb3:  // pred: ^bb1
        spirv.mlir.merge
      }
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel_1 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: f32, %arg1: memref<1x112x112x3x32xf32>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 112, 112>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  y
      %1 = gpu.block_id  z
      scf.for %arg6 = %arg2 to %arg5 step %arg4 {
        scf.for %arg7 = %arg2 to %arg3 step %arg4 {
          memref.store %arg0, %arg1[%arg2, %0, %1, %arg6, %arg7] : memref<1x112x112x3x32xf32>
        }
      }
      gpu.return
    }
  }
  spirv.module @__spv__test_kernel_2 Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<471969 x f32>, CrossWorkgroup>, %arg1: i64, %arg2: !spirv.ptr<!spirv.array<2592 x f32>, CrossWorkgroup>, %arg3: !spirv.ptr<!spirv.array<1204224 x f32>, CrossWorkgroup>, %arg4: i64, %arg5: i64, %arg6: i64) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 112, 112>, workgroup_attributions = 0 : i64} {
      %cst3_i64 = spirv.Constant 3 : i64
      %cst2_i64 = spirv.Constant 2 : i64
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[1 : i32] : vector<3xi64>
      %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
      %3 = spirv.CompositeExtract %2[2 : i32] : vector<3xi64>
      spirv.mlir.loop {
        spirv.Branch ^bb1(%arg1 : i64)
      ^bb1(%4: i64):  // 2 preds: ^bb0, ^bb2
        %5 = spirv.SLessThan %4, %arg4 : i64
        spirv.BranchConditional %5, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        spirv.mlir.loop {
          spirv.Branch ^bb1(%arg1 : i64)
        ^bb1(%7: i64):  // 2 preds: ^bb0, ^bb2
          %8 = spirv.SLessThan %7, %arg6 : i64
          spirv.BranchConditional %8, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          spirv.mlir.loop {
            spirv.Branch ^bb1(%arg1 : i64)
          ^bb1(%10: i64):  // 2 preds: ^bb0, ^bb2
            %11 = spirv.SLessThan %10, %arg4 : i64
            spirv.BranchConditional %11, ^bb2, ^bb3
          ^bb2:  // pred: ^bb1
            spirv.mlir.loop {
              spirv.Branch ^bb1(%arg1 : i64)
            ^bb1(%13: i64):  // 2 preds: ^bb0, ^bb2
              %14 = spirv.SLessThan %13, %arg4 : i64
              spirv.BranchConditional %14, ^bb2, ^bb3
            ^bb2:  // pred: ^bb1
              spirv.mlir.loop {
                spirv.Branch ^bb1(%arg1 : i64)
              ^bb1(%16: i64):  // 2 preds: ^bb0, ^bb2
                %17 = spirv.SLessThan %16, %arg4 : i64
                spirv.BranchConditional %17, ^bb2, ^bb3
              ^bb2:  // pred: ^bb1
                %18 = spirv.IMul %1, %cst2_i64 : i64
                %19 = spirv.IMul %10, %cst3_i64 : i64
                %20 = spirv.IAdd %18, %19 : i64
                %21 = spirv.IMul %3, %cst2_i64 : i64
                %22 = spirv.IMul %13, %cst3_i64 : i64
                %23 = spirv.IAdd %21, %22 : i64
                %cst0_i64 = spirv.Constant 0 : i64
                %cst471969_i64 = spirv.Constant 471969 : i64
                %24 = spirv.IMul %cst471969_i64, %arg1 : i64
                %25 = spirv.IAdd %cst0_i64, %24 : i64
                %cst2061_i64 = spirv.Constant 2061 : i64
                %26 = spirv.IMul %cst2061_i64, %20 : i64
                %27 = spirv.IAdd %25, %26 : i64
                %cst9_i64 = spirv.Constant 9 : i64
                %28 = spirv.IMul %cst9_i64, %23 : i64
                %29 = spirv.IAdd %27, %28 : i64
                %cst3_i64_1 = spirv.Constant 3 : i64
                %30 = spirv.IMul %cst3_i64_1, %4 : i64
                %31 = spirv.IAdd %29, %30 : i64
                %cst1_i64 = spirv.Constant 1 : i64
                %32 = spirv.IMul %cst1_i64, %16 : i64
                %33 = spirv.IAdd %31, %32 : i64
                %34 = spirv.AccessChain %arg0[%33] : !spirv.ptr<!spirv.array<471969 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
                %35 = spirv.Load "CrossWorkgroup" %34 : f32
                %cst0_i64_2 = spirv.Constant 0 : i64
                %cst864_i64 = spirv.Constant 864 : i64
                %36 = spirv.IMul %cst864_i64, %10 : i64
                %37 = spirv.IAdd %cst0_i64_2, %36 : i64
                %cst288_i64 = spirv.Constant 288 : i64
                %38 = spirv.IMul %cst288_i64, %13 : i64
                %39 = spirv.IAdd %37, %38 : i64
                %cst96_i64 = spirv.Constant 96 : i64
                %40 = spirv.IMul %cst96_i64, %4 : i64
                %41 = spirv.IAdd %39, %40 : i64
                %cst32_i64 = spirv.Constant 32 : i64
                %42 = spirv.IMul %cst32_i64, %16 : i64
                %43 = spirv.IAdd %41, %42 : i64
                %cst1_i64_3 = spirv.Constant 1 : i64
                %44 = spirv.IMul %cst1_i64_3, %7 : i64
                %45 = spirv.IAdd %43, %44 : i64
                %46 = spirv.AccessChain %arg2[%45] : !spirv.ptr<!spirv.array<2592 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
                %47 = spirv.Load "CrossWorkgroup" %46 : f32
                %cst0_i64_4 = spirv.Constant 0 : i64
                %cst1204224_i64 = spirv.Constant 1204224 : i64
                %48 = spirv.IMul %cst1204224_i64, %arg1 : i64
                %49 = spirv.IAdd %cst0_i64_4, %48 : i64
                %cst10752_i64 = spirv.Constant 10752 : i64
                %50 = spirv.IMul %cst10752_i64, %1 : i64
                %51 = spirv.IAdd %49, %50 : i64
                %cst96_i64_5 = spirv.Constant 96 : i64
                %52 = spirv.IMul %cst96_i64_5, %3 : i64
                %53 = spirv.IAdd %51, %52 : i64
                %cst32_i64_6 = spirv.Constant 32 : i64
                %54 = spirv.IMul %cst32_i64_6, %4 : i64
                %55 = spirv.IAdd %53, %54 : i64
                %cst1_i64_7 = spirv.Constant 1 : i64
                %56 = spirv.IMul %cst1_i64_7, %7 : i64
                %57 = spirv.IAdd %55, %56 : i64
                %58 = spirv.AccessChain %arg3[%57] : !spirv.ptr<!spirv.array<1204224 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
                %59 = spirv.Load "CrossWorkgroup" %58 : f32
                %60 = spirv.FMul %35, %47 : f32
                %61 = spirv.FAdd %59, %60 : f32
                %cst0_i64_8 = spirv.Constant 0 : i64
                %cst1204224_i64_9 = spirv.Constant 1204224 : i64
                %62 = spirv.IMul %cst1204224_i64_9, %arg1 : i64
                %63 = spirv.IAdd %cst0_i64_8, %62 : i64
                %cst10752_i64_10 = spirv.Constant 10752 : i64
                %64 = spirv.IMul %cst10752_i64_10, %1 : i64
                %65 = spirv.IAdd %63, %64 : i64
                %cst96_i64_11 = spirv.Constant 96 : i64
                %66 = spirv.IMul %cst96_i64_11, %3 : i64
                %67 = spirv.IAdd %65, %66 : i64
                %cst32_i64_12 = spirv.Constant 32 : i64
                %68 = spirv.IMul %cst32_i64_12, %4 : i64
                %69 = spirv.IAdd %67, %68 : i64
                %cst1_i64_13 = spirv.Constant 1 : i64
                %70 = spirv.IMul %cst1_i64_13, %7 : i64
                %71 = spirv.IAdd %69, %70 : i64
                %72 = spirv.AccessChain %arg3[%71] : !spirv.ptr<!spirv.array<1204224 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
                spirv.Store "CrossWorkgroup" %72, %61 : f32
                %73 = spirv.IAdd %16, %arg5 : i64
                spirv.Branch ^bb1(%73 : i64)
              ^bb3:  // pred: ^bb1
                spirv.mlir.merge
              }
              %15 = spirv.IAdd %13, %arg5 : i64
              spirv.Branch ^bb1(%15 : i64)
            ^bb3:  // pred: ^bb1
              spirv.mlir.merge
            }
            %12 = spirv.IAdd %10, %arg5 : i64
            spirv.Branch ^bb1(%12 : i64)
          ^bb3:  // pred: ^bb1
            spirv.mlir.merge
          }
          %9 = spirv.IAdd %7, %arg5 : i64
          spirv.Branch ^bb1(%9 : i64)
        ^bb3:  // pred: ^bb1
          spirv.mlir.merge
        }
        %6 = spirv.IAdd %4, %arg5 : i64
        spirv.Branch ^bb1(%6 : i64)
      ^bb3:  // pred: ^bb1
        spirv.mlir.merge
      }
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel_2 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<1x229x229x3x3xf32>, %arg1: index, %arg2: memref<3x3x3x3x32xf32>, %arg3: memref<1x112x112x3x32xf32>, %arg4: index, %arg5: index, %arg6: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 112, 112>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %0 = gpu.block_id  y
      %1 = gpu.block_id  z
      scf.for %arg7 = %arg1 to %arg4 step %arg5 {
        scf.for %arg8 = %arg1 to %arg6 step %arg5 {
          scf.for %arg9 = %arg1 to %arg4 step %arg5 {
            scf.for %arg10 = %arg1 to %arg4 step %arg5 {
              scf.for %arg11 = %arg1 to %arg4 step %arg5 {
                %2 = arith.muli %0, %c2 : index
                %3 = arith.muli %arg9, %c3 : index
                %4 = arith.addi %2, %3 : index
                %5 = arith.muli %1, %c2 : index
                %6 = arith.muli %arg10, %c3 : index
                %7 = arith.addi %5, %6 : index
                %8 = memref.load %arg0[%arg1, %4, %7, %arg7, %arg11] : memref<1x229x229x3x3xf32>
                %9 = memref.load %arg2[%arg9, %arg10, %arg7, %arg11, %arg8] : memref<3x3x3x3x32xf32>
                %10 = memref.load %arg3[%arg1, %0, %1, %arg7, %arg8] : memref<1x112x112x3x32xf32>
                %11 = arith.mulf %8, %9 : f32
                %12 = arith.addf %10, %11 : f32
                memref.store %12, %arg3[%arg1, %0, %1, %arg7, %arg8] : memref<1x112x112x3x32xf32>
              }
            }
          }
        }
      }
      gpu.return
    }
  }
}
