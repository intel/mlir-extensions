// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck


module @softmax attributes {gpu.container_module} {
  memref.global "private" constant @__constant_10x20xf32 : memref<10x20xf32> = dense<[
    [0.9, 0.1, 0.4, 0.9, 0.1, 0.4, 0.9, 0.1, 0.7, 0.4, 0.9, 0.1, 0.4, 0.9, 0.1, 0.4, 0.9, 0.1, 0.7, 0.4],
    [1.9, 0.7, 1.2, 1.9, 0.7, 1.2, 1.9, 0.6, 1.5, 1.6, 1.9, 0.7, 1.2, 1.9, 0.7, 1.2, 1.9, 0.7, 1.5, 1.6],
    [0.9, 0.1, 0.4, 0.9, 0.1, 0.4, 0.9, 0.1, 0.7, 0.4, 1.9, 0.7, 1.2, 1.9, 0.6, 1.2, 1.9, 0.6, 1.5, 1.6],
    [1.9, 0.6, 1.2, 1.9, 0.7, 1.2, 1.9, 0.7, 1.5, 1.6, 1.9, 0.6, 1.2, 1.9, 0.6, 1.2, 1.9, 0.7, 1.5, 1.6],
    [0.9, 0.1, 0.4, 0.9, 0.1, 0.4, 0.9, 0.1, 0.7, 0.4, 1.9, 0.6, 1.2, 1.9, 0.7, 1.2, 1.9, 0.6, 1.5, 1.6],
    [1.9, 0.7, 1.2, 1.9, 0.7, 1.2, 1.9, 0.7, 1.5, 1.6, 1.9, 0.7, 1.2, 1.9, 0.7, 1.2, 1.9, 0.7, 1.5, 1.6],
    [0.9, 0.1, 0.4, 0.9, 0.1, 0.4, 0.9, 0.1, 0.7, 0.4, 1.9, 0.6, 1.2, 1.9, 0.6, 1.2, 1.9, 0.6, 1.5, 1.6],
    [1.9, 0.6, 1.2, 1.9, 0.7, 1.2, 1.9, 0.7, 1.5, 1.6, 1.9, 0.7, 1.2, 1.9, 0.7, 1.2, 1.9, 0.7, 1.5, 1.6],
    [0.9, 0.1, 0.4, 0.9, 0.1, 0.4, 0.9, 0.1, 0.7, 0.4, 1.9, 0.6, 1.2, 1.9, 0.6, 1.2, 1.9, 0.6, 1.5, 1.6],
    [1.9, 0.7, 1.2, 1.9, 0.7, 1.2, 1.9, 0.6, 1.5, 1.6, 1.9, 0.7, 1.2, 1.9, 0.7, 1.2, 1.9, 0.7, 1.5, 1.6]
  ]>
  memref.global "private" constant @__constant_10x20xf32_ref_result : memref<10x20xf32> = dense<[
    [0.0715685,   0.0321578,   0.0434085,   0.0715685,   0.0321578,   0.0434085,   0.0715685,   0.0321578,   0.0585954,   0.0434085,   0.0715685,   0.0321578,   0.0434085,   0.0715685,   0.0321578,   0.0434085,   0.0715685,   0.0321578,   0.0585954,   0.0434085],
    [0.0794463,   0.0239288,   0.0394519,   0.0794463,   0.0239288,   0.0394519,   0.0794463,   0.0216516,   0.0532544,   0.0588553,   0.0794463,   0.0239288,   0.0394519,   0.0794463,   0.0239288,   0.0394519,   0.0794463,   0.0239288,   0.0532544,   0.0588553],
    [0.0417064,   0.0187399,   0.0252962,   0.0417064,   0.0187399,   0.0252962,   0.0417064,   0.0187399,   0.0341463,   0.0252962,   0.11337,   0.0341463,   0.0562978,   0.11337,   0.0308969,   0.0562978,   0.11337,   0.0308969,   0.0759941,   0.0839865],
    [0.0798098,   0.0217507,   0.0396324,   0.0798098,   0.0240382,   0.0396324,   0.0798098,   0.0240382,   0.0534981,   0.0591245,   0.0798098,   0.0217507,   0.0396324,   0.0798098,   0.0217507,   0.0396324,   0.0798098,   0.0240382,   0.0534981,   0.0591245],
    [0.0417064,   0.0187399,   0.0252962,   0.0417064,   0.0187399,   0.0252962,   0.0417064,   0.0187399,   0.0341463,   0.0252962,   0.11337,   0.0308969,   0.0562978,   0.11337,   0.0341463,   0.0562978,   0.11337,   0.0308969,   0.0759941,   0.0839865],
    [0.0792658,   0.0238744,   0.0393622,   0.0792658,   0.0238744,   0.0393622,   0.0792658,   0.0238744,   0.0531335,   0.0587215,   0.0792658,   0.0238744,   0.0393622,   0.0792658,   0.0238744,   0.0393622,   0.0792658,   0.0238744,   0.0531335,   0.0587215],
    [0.0418424,   0.018801,   0.0253787,   0.0418424,   0.018801,   0.0253787,   0.0418424,   0.018801,   0.0342577,   0.0253787,   0.113739,   0.0309976,   0.0564813,   0.113739,   0.0309976,   0.0564813,   0.113739,   0.0309976,   0.0762418,   0.0842603],
    [0.0794463,   0.0216516,   0.0394519,   0.0794463,   0.0239288,   0.0394519,   0.0794463,   0.0239288,   0.0532544,   0.0588553,   0.0794463,   0.0239288,   0.0394519,   0.0794463,   0.0239288,   0.0394519,   0.0794463,   0.0239288,   0.0532544,   0.0588553],
    [0.0418424,   0.018801,   0.0253787,   0.0418424,   0.018801,   0.0253787,   0.0418424,   0.018801,   0.0342577,   0.0253787,   0.113739,   0.0309976,   0.0564813,   0.113739,   0.0309976,   0.0564813,   0.113739,   0.0309976,   0.0762418,   0.0842603],
    [0.0794463,   0.0239288,   0.0394519,   0.0794463,   0.0239288,   0.0394519,   0.0794463,   0.0216516,   0.0532544,   0.0588553,   0.0794463,   0.0239288,   0.0394519,   0.0794463,   0.0239288,   0.0394519,   0.0794463,   0.0239288,   0.0532544,   0.0588553]
  ]>


  func.func @main() attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_10x20xf32 : memref<10x20xf32>
    %ref_result = memref.get_global @__constant_10x20xf32_ref_result : memref<10x20xf32>
    %unranked_ref_result =  memref.cast %ref_result : memref<10x20xf32> to memref<*xf32>

    scf.for %arg0 = %c0 to %c100 step %c1 {
      %1 = func.call @test(%0) : (memref<10x20xf32>) -> memref<10x20xf32>
      %cast = memref.cast %1 : memref<10x20xf32> to memref<*xf32>
      func.call @printAllcloseF32(%cast, %unranked_ref_result) : (memref<*xf32>, memref<*xf32>) -> ()
      func.call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
      // CHECK:   [ALLCLOSE: TRUE]
    }
    return
  }
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func @test(%arg0: memref<10x20xf32>) -> memref<10x20xf32> attributes {llvm.emit_c_interface} {
    %c20 = arith.constant 20 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %memref = gpu.alloc  host_shared () : memref<10x20xf32>
    memref.copy %arg0, %memref : memref<10x20xf32> to memref<10x20xf32>
    %memref_1 = gpu.alloc  host_shared () : memref<10x1xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c10, %c1, %c1) threads in (%c1, %c1, %c1) args(%cst : f32, %memref_1 : memref<10x1xf32>, %c0 : index)
    %memref_2 = gpu.alloc  host_shared () : memref<10x1xf32>
    memref.copy %memref_1, %memref_2 : memref<10x1xf32> to memref<10x1xf32>
    gpu.launch_func  @test_kernel_0::@test_kernel blocks in (%c10, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<10x20xf32>, %memref_2 : memref<10x1xf32>, %c0 : index, %c20 : index, %c1 : index)
    %memref_3 = gpu.alloc  () : memref<10x20xf32>
    gpu.launch_func  @test_kernel_1::@test_kernel blocks in (%c10, %c20, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<10x20xf32>, %memref_2 : memref<10x1xf32>, %c0 : index, %memref_3 : memref<10x20xf32>)
    %memref_4 = gpu.alloc  () : memref<10x20xf32>
    gpu.launch_func  @test_kernel_2::@test_kernel blocks in (%c10, %c20, %c1) threads in (%c1, %c1, %c1) args(%memref_3 : memref<10x20xf32>, %memref_4 : memref<10x20xf32>)
    %memref_5 = gpu.alloc  host_shared () : memref<10x1xf32>
    gpu.launch_func  @test_kernel_3::@test_kernel blocks in (%c10, %c1, %c1) threads in (%c1, %c1, %c1) args(%cst_0 : f32, %memref_5 : memref<10x1xf32>, %c0 : index)
    %memref_6 = gpu.alloc  host_shared () : memref<10x1xf32>
    memref.copy %memref_5, %memref_6 : memref<10x1xf32> to memref<10x1xf32>
    gpu.launch_func  @test_kernel_4::@test_kernel blocks in (%c10, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref_4 : memref<10x20xf32>, %memref_6 : memref<10x1xf32>, %c0 : index, %c20 : index, %c1 : index)
    %memref_7 = gpu.alloc  host_shared () : memref<10x20xf32>
    gpu.launch_func  @test_kernel_5::@test_kernel blocks in (%c10, %c20, %c1) threads in (%c1, %c1, %c1) args(%memref_4 : memref<10x20xf32>, %memref_6 : memref<10x1xf32>, %c0 : index, %memref_7 : memref<10x20xf32>)
    gpu.dealloc  %memref_1 : memref<10x1xf32>
    gpu.dealloc  %memref_3 : memref<10x20xf32>
    gpu.dealloc  %memref_4 : memref<10x20xf32>
    gpu.dealloc  %memref_5 : memref<10x1xf32>
    gpu.dealloc  %memref : memref<10x20xf32>
    return %memref_7 : memref<10x20xf32>
  }
  spirv.module @__spv__test_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: f32, %arg1: !spirv.ptr<!spirv.array<10 x f32>, CrossWorkgroup>, %arg2: i64) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 10, 1, 1>, workgroup_attributions = 0 : i64} {
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %cst0_i64 = spirv.Constant 0 : i64
      %cst1_i64 = spirv.Constant 1 : i64
      %2 = spirv.IMul %cst1_i64, %1 : i64
      %3 = spirv.IAdd %cst0_i64, %2 : i64
      %cst1_i64_0 = spirv.Constant 1 : i64
      %4 = spirv.IMul %cst1_i64_0, %arg2 : i64
      %5 = spirv.IAdd %3, %4 : i64
      %6 = spirv.AccessChain %arg1[%5] : !spirv.ptr<!spirv.array<10 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      spirv.Store "CrossWorkgroup" %6, %arg0 : f32
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: f32, %arg1: memref<10x1xf32>, %arg2: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 10, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      memref.store %arg0, %arg1[%0, %arg2] : memref<10x1xf32>
      gpu.return
    }
  }
  spirv.module @__spv__test_kernel_0 Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, %arg1: !spirv.ptr<!spirv.array<10 x f32>, CrossWorkgroup>, %arg2: i64, %arg3: i64, %arg4: i64) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 10, 1, 1>, workgroup_attributions = 0 : i64} {
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      spirv.mlir.loop {
        spirv.Branch ^bb1(%arg2 : i64)
      ^bb1(%2: i64):  // 2 preds: ^bb0, ^bb2
        %3 = spirv.SLessThan %2, %arg3 : i64
        spirv.BranchConditional %3, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        %cst0_i64 = spirv.Constant 0 : i64
        %cst20_i64 = spirv.Constant 20 : i64
        %4 = spirv.IMul %cst20_i64, %1 : i64
        %5 = spirv.IAdd %cst0_i64, %4 : i64
        %cst1_i64 = spirv.Constant 1 : i64
        %6 = spirv.IMul %cst1_i64, %2 : i64
        %7 = spirv.IAdd %5, %6 : i64
        %8 = spirv.AccessChain %arg0[%7] : !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
        %9 = spirv.Load "CrossWorkgroup" %8 : f32
        %cst0_i64_0 = spirv.Constant 0 : i64
        %cst1_i64_1 = spirv.Constant 1 : i64
        %10 = spirv.IMul %cst1_i64_1, %1 : i64
        %11 = spirv.IAdd %cst0_i64_0, %10 : i64
        %cst1_i64_2 = spirv.Constant 1 : i64
        %12 = spirv.IMul %cst1_i64_2, %arg2 : i64
        %13 = spirv.IAdd %11, %12 : i64
        %14 = spirv.AccessChain %arg1[%13] : !spirv.ptr<!spirv.array<10 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
        %15 = spirv.Load "CrossWorkgroup" %14 : f32
        %16 = spirv.FOrdGreaterThan %15, %9 : f32
        %17 = spirv.Select %16, %15, %9 : i1, f32
        %cst0_i64_3 = spirv.Constant 0 : i64
        %cst1_i64_4 = spirv.Constant 1 : i64
        %18 = spirv.IMul %cst1_i64_4, %1 : i64
        %19 = spirv.IAdd %cst0_i64_3, %18 : i64
        %cst1_i64_5 = spirv.Constant 1 : i64
        %20 = spirv.IMul %cst1_i64_5, %arg2 : i64
        %21 = spirv.IAdd %19, %20 : i64
        %22 = spirv.AccessChain %arg1[%21] : !spirv.ptr<!spirv.array<10 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
        spirv.Store "CrossWorkgroup" %22, %17 : f32
        %23 = spirv.IAdd %2, %arg4 : i64
        spirv.Branch ^bb1(%23 : i64)
      ^bb3:  // pred: ^bb1
        spirv.mlir.merge
      }
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel_0 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<10x20xf32>, %arg1: memref<10x1xf32>, %arg2: index, %arg3: index, %arg4: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 10, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      scf.for %arg5 = %arg2 to %arg3 step %arg4 {
        %1 = memref.load %arg0[%0, %arg5] : memref<10x20xf32>
        %2 = memref.load %arg1[%0, %arg2] : memref<10x1xf32>
        %3 = arith.cmpf ogt, %2, %1 : f32
        %4 = arith.select %3, %2, %1 : f32
        memref.store %4, %arg1[%0, %arg2] : memref<10x1xf32>
      }
      gpu.return
    }
  }
  spirv.module @__spv__test_kernel_1 Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, %arg1: !spirv.ptr<!spirv.array<10 x f32>, CrossWorkgroup>, %arg2: i64, %arg3: !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 10, 20, 1>, workgroup_attributions = 0 : i64} {
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
      %3 = spirv.CompositeExtract %2[1 : i32] : vector<3xi64>
      %cst0_i64 = spirv.Constant 0 : i64
      %cst20_i64 = spirv.Constant 20 : i64
      %4 = spirv.IMul %cst20_i64, %1 : i64
      %5 = spirv.IAdd %cst0_i64, %4 : i64
      %cst1_i64 = spirv.Constant 1 : i64
      %6 = spirv.IMul %cst1_i64, %3 : i64
      %7 = spirv.IAdd %5, %6 : i64
      %8 = spirv.AccessChain %arg0[%7] : !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      %9 = spirv.Load "CrossWorkgroup" %8 : f32
      %cst0_i64_1 = spirv.Constant 0 : i64
      %cst1_i64_2 = spirv.Constant 1 : i64
      %10 = spirv.IMul %cst1_i64_2, %1 : i64
      %11 = spirv.IAdd %cst0_i64_1, %10 : i64
      %cst1_i64_3 = spirv.Constant 1 : i64
      %12 = spirv.IMul %cst1_i64_3, %arg2 : i64
      %13 = spirv.IAdd %11, %12 : i64
      %14 = spirv.AccessChain %arg1[%13] : !spirv.ptr<!spirv.array<10 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      %15 = spirv.Load "CrossWorkgroup" %14 : f32
      %16 = spirv.FSub %9, %15 : f32
      %cst0_i64_4 = spirv.Constant 0 : i64
      %cst20_i64_5 = spirv.Constant 20 : i64
      %17 = spirv.IMul %cst20_i64_5, %1 : i64
      %18 = spirv.IAdd %cst0_i64_4, %17 : i64
      %cst1_i64_6 = spirv.Constant 1 : i64
      %19 = spirv.IMul %cst1_i64_6, %3 : i64
      %20 = spirv.IAdd %18, %19 : i64
      %21 = spirv.AccessChain %arg3[%20] : !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      spirv.Store "CrossWorkgroup" %21, %16 : f32
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel_1 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<10x20xf32>, %arg1: memref<10x1xf32>, %arg2: index, %arg3: memref<10x20xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 10, 20, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = memref.load %arg0[%0, %1] : memref<10x20xf32>
      %3 = memref.load %arg1[%0, %arg2] : memref<10x1xf32>
      %4 = arith.subf %2, %3 : f32
      memref.store %4, %arg3[%0, %1] : memref<10x20xf32>
      gpu.return
    }
  }
  spirv.module @__spv__test_kernel_2 Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, %arg1: !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 10, 20, 1>, workgroup_attributions = 0 : i64} {
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
      %3 = spirv.CompositeExtract %2[1 : i32] : vector<3xi64>
      %cst0_i64 = spirv.Constant 0 : i64
      %cst20_i64 = spirv.Constant 20 : i64
      %4 = spirv.IMul %cst20_i64, %1 : i64
      %5 = spirv.IAdd %cst0_i64, %4 : i64
      %cst1_i64 = spirv.Constant 1 : i64
      %6 = spirv.IMul %cst1_i64, %3 : i64
      %7 = spirv.IAdd %5, %6 : i64
      %8 = spirv.AccessChain %arg0[%7] : !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      %9 = spirv.Load "CrossWorkgroup" %8 : f32
      %10 = spirv.CL.exp %9 : f32
      %cst0_i64_1 = spirv.Constant 0 : i64
      %cst20_i64_2 = spirv.Constant 20 : i64
      %11 = spirv.IMul %cst20_i64_2, %1 : i64
      %12 = spirv.IAdd %cst0_i64_1, %11 : i64
      %cst1_i64_3 = spirv.Constant 1 : i64
      %13 = spirv.IMul %cst1_i64_3, %3 : i64
      %14 = spirv.IAdd %12, %13 : i64
      %15 = spirv.AccessChain %arg1[%14] : !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      spirv.Store "CrossWorkgroup" %15, %10 : f32
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel_2 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<10x20xf32>, %arg1: memref<10x20xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 10, 20, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = memref.load %arg0[%0, %1] : memref<10x20xf32>
      %3 = math.exp %2 : f32
      memref.store %3, %arg1[%0, %1] : memref<10x20xf32>
      gpu.return
    }
  }
  spirv.module @__spv__test_kernel_3 Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: f32, %arg1: !spirv.ptr<!spirv.array<10 x f32>, CrossWorkgroup>, %arg2: i64) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 10, 1, 1>, workgroup_attributions = 0 : i64} {
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %cst0_i64 = spirv.Constant 0 : i64
      %cst1_i64 = spirv.Constant 1 : i64
      %2 = spirv.IMul %cst1_i64, %1 : i64
      %3 = spirv.IAdd %cst0_i64, %2 : i64
      %cst1_i64_0 = spirv.Constant 1 : i64
      %4 = spirv.IMul %cst1_i64_0, %arg2 : i64
      %5 = spirv.IAdd %3, %4 : i64
      %6 = spirv.AccessChain %arg1[%5] : !spirv.ptr<!spirv.array<10 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      spirv.Store "CrossWorkgroup" %6, %arg0 : f32
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel_3 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: f32, %arg1: memref<10x1xf32>, %arg2: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 10, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      memref.store %arg0, %arg1[%0, %arg2] : memref<10x1xf32>
      gpu.return
    }
  }
  spirv.module @__spv__test_kernel_4 Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, %arg1: !spirv.ptr<!spirv.array<10 x f32>, CrossWorkgroup>, %arg2: i64, %arg3: i64, %arg4: i64) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 10, 1, 1>, workgroup_attributions = 0 : i64} {
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      spirv.mlir.loop {
        spirv.Branch ^bb1(%arg2 : i64)
      ^bb1(%2: i64):  // 2 preds: ^bb0, ^bb2
        %3 = spirv.SLessThan %2, %arg3 : i64
        spirv.BranchConditional %3, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        %cst0_i64 = spirv.Constant 0 : i64
        %cst20_i64 = spirv.Constant 20 : i64
        %4 = spirv.IMul %cst20_i64, %1 : i64
        %5 = spirv.IAdd %cst0_i64, %4 : i64
        %cst1_i64 = spirv.Constant 1 : i64
        %6 = spirv.IMul %cst1_i64, %2 : i64
        %7 = spirv.IAdd %5, %6 : i64
        %8 = spirv.AccessChain %arg0[%7] : !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
        %9 = spirv.Load "CrossWorkgroup" %8 : f32
        %cst0_i64_0 = spirv.Constant 0 : i64
        %cst1_i64_1 = spirv.Constant 1 : i64
        %10 = spirv.IMul %cst1_i64_1, %1 : i64
        %11 = spirv.IAdd %cst0_i64_0, %10 : i64
        %cst1_i64_2 = spirv.Constant 1 : i64
        %12 = spirv.IMul %cst1_i64_2, %arg2 : i64
        %13 = spirv.IAdd %11, %12 : i64
        %14 = spirv.AccessChain %arg1[%13] : !spirv.ptr<!spirv.array<10 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
        %15 = spirv.Load "CrossWorkgroup" %14 : f32
        %16 = spirv.FAdd %15, %9 : f32
        %cst0_i64_3 = spirv.Constant 0 : i64
        %cst1_i64_4 = spirv.Constant 1 : i64
        %17 = spirv.IMul %cst1_i64_4, %1 : i64
        %18 = spirv.IAdd %cst0_i64_3, %17 : i64
        %cst1_i64_5 = spirv.Constant 1 : i64
        %19 = spirv.IMul %cst1_i64_5, %arg2 : i64
        %20 = spirv.IAdd %18, %19 : i64
        %21 = spirv.AccessChain %arg1[%20] : !spirv.ptr<!spirv.array<10 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
        spirv.Store "CrossWorkgroup" %21, %16 : f32
        %22 = spirv.IAdd %2, %arg4 : i64
        spirv.Branch ^bb1(%22 : i64)
      ^bb3:  // pred: ^bb1
        spirv.mlir.merge
      }
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel_4 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<10x20xf32>, %arg1: memref<10x1xf32>, %arg2: index, %arg3: index, %arg4: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 10, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      scf.for %arg5 = %arg2 to %arg3 step %arg4 {
        %1 = memref.load %arg0[%0, %arg5] : memref<10x20xf32>
        %2 = memref.load %arg1[%0, %arg2] : memref<10x1xf32>
        %3 = arith.addf %2, %1 : f32
        memref.store %3, %arg1[%0, %arg2] : memref<10x1xf32>
      }
      gpu.return
    }
  }
  spirv.module @__spv__test_kernel_5 Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, %arg1: !spirv.ptr<!spirv.array<10 x f32>, CrossWorkgroup>, %arg2: i64, %arg3: !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 10, 20, 1>, workgroup_attributions = 0 : i64} {
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
      %3 = spirv.CompositeExtract %2[1 : i32] : vector<3xi64>
      %cst0_i64 = spirv.Constant 0 : i64
      %cst20_i64 = spirv.Constant 20 : i64
      %4 = spirv.IMul %cst20_i64, %1 : i64
      %5 = spirv.IAdd %cst0_i64, %4 : i64
      %cst1_i64 = spirv.Constant 1 : i64
      %6 = spirv.IMul %cst1_i64, %3 : i64
      %7 = spirv.IAdd %5, %6 : i64
      %8 = spirv.AccessChain %arg0[%7] : !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      %9 = spirv.Load "CrossWorkgroup" %8 : f32
      %cst0_i64_1 = spirv.Constant 0 : i64
      %cst1_i64_2 = spirv.Constant 1 : i64
      %10 = spirv.IMul %cst1_i64_2, %1 : i64
      %11 = spirv.IAdd %cst0_i64_1, %10 : i64
      %cst1_i64_3 = spirv.Constant 1 : i64
      %12 = spirv.IMul %cst1_i64_3, %arg2 : i64
      %13 = spirv.IAdd %11, %12 : i64
      %14 = spirv.AccessChain %arg1[%13] : !spirv.ptr<!spirv.array<10 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      %15 = spirv.Load "CrossWorkgroup" %14 : f32
      %16 = spirv.FDiv %9, %15 : f32
      %cst0_i64_4 = spirv.Constant 0 : i64
      %cst20_i64_5 = spirv.Constant 20 : i64
      %17 = spirv.IMul %cst20_i64_5, %1 : i64
      %18 = spirv.IAdd %cst0_i64_4, %17 : i64
      %cst1_i64_6 = spirv.Constant 1 : i64
      %19 = spirv.IMul %cst1_i64_6, %3 : i64
      %20 = spirv.IAdd %18, %19 : i64
      %21 = spirv.AccessChain %arg3[%20] : !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      spirv.Store "CrossWorkgroup" %21, %16 : f32
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel_5 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<10x20xf32>, %arg1: memref<10x1xf32>, %arg2: index, %arg3: memref<10x20xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 10, 20, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = memref.load %arg0[%0, %1] : memref<10x20xf32>
      %3 = memref.load %arg1[%0, %arg2] : memref<10x1xf32>
      %4 = arith.divf %2, %3 : f32
      memref.store %4, %arg3[%0, %1] : memref<10x20xf32>
      gpu.return
    }
  }
}
