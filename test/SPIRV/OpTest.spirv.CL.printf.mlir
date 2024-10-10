// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck


module @print_simple attributes {gpu.container_module} {

 func.func @test() -> () attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c100_i32 = arith.constant 100 : i32
    %cst_f32 = arith.constant 3.144000e+02 : f32

    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%c100_i32 : i32, %cst_f32 : f32)
    return
  }
  spirv.module @__spv__test_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {

    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.SpecConstant @printfMsg0_sc0 = 72 : i8
    spirv.SpecConstant @printfMsg0_sc1 = 101 : i8
    spirv.SpecConstant @printfMsg0_sc2 = 108 : i8
    spirv.SpecConstant @printfMsg0_sc3 = 111 : i8
    spirv.SpecConstant @printfMsg0_sc4 = 58 : i8
    spirv.SpecConstant @printfMsg0_sc5 = 32 : i8
    spirv.SpecConstant @printfMsg0_sc6 = 37 : i8
    spirv.SpecConstant @printfMsg0_sc7 = 100 : i8
    spirv.SpecConstant @printfMsg0_sc8 = 102 : i8
    spirv.SpecConstant @printfMsg0_sc9 = 10 : i8
    spirv.SpecConstant @printfMsg0_sc10 = 0 : i8

    // Print Fmt String -  "Hello\n"
    spirv.SpecConstantComposite @printfMsg0_scc (@printfMsg0_sc0, @printfMsg0_sc1, @printfMsg0_sc2, @printfMsg0_sc2, @printfMsg0_sc3, @printfMsg0_sc9, @printfMsg0_sc10) : !spirv.array<7 x i8>
    spirv.GlobalVariable @printfMsg0 initializer(@printfMsg0_scc)  {Constant} : !spirv.ptr<!spirv.array<7 x i8>, UniformConstant>

    // Print Fmt String - "Hello: %d %f\n"
    spirv.SpecConstantComposite @printfMsg1_scc (@printfMsg0_sc0, @printfMsg0_sc1, @printfMsg0_sc2, @printfMsg0_sc2, @printfMsg0_sc3, @printfMsg0_sc4, @printfMsg0_sc5, @printfMsg0_sc6, @printfMsg0_sc7, @printfMsg0_sc5, @printfMsg0_sc6, @printfMsg0_sc8, @printfMsg0_sc9, @printfMsg0_sc10) : !spirv.array<14 x i8>
    spirv.GlobalVariable @printfMsg1 initializer(@printfMsg1_scc) {Constant} : !spirv.ptr<!spirv.array<14 x i8>, UniformConstant>


    spirv.func @test_kernel(%arg0: i32, %arg1: f32) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, workgroup_attributions = 0 : i64, VectorComputeFunctionINTEL} {
      %printfMsg0_addr = spirv.mlir.addressof @printfMsg0 : !spirv.ptr<!spirv.array<7 x i8>, UniformConstant>
      %2 = spirv.Bitcast %printfMsg0_addr : !spirv.ptr<!spirv.array<7 x i8>, UniformConstant> to !spirv.ptr<i8, UniformConstant>
      %3 = spirv.CL.printf %2 : !spirv.ptr<i8, UniformConstant> -> i32

      %printfMsg1_addr = spirv.mlir.addressof @printfMsg1 : !spirv.ptr<!spirv.array<14 x i8>, UniformConstant>
      %0 = spirv.Bitcast %printfMsg1_addr : !spirv.ptr<!spirv.array<14 x i8>, UniformConstant> to !spirv.ptr<i8, UniformConstant>
      %1 = spirv.CL.printf %0 %arg0, %arg1 : !spirv.ptr<i8, UniformConstant>, i32, f32 -> i32

     spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__, @printfMsg0
  }

  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: i32, %arg1: f32) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {

      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    func.call @test() : ()-> ()
    // CHECK: Hello
    // CHECK: Hello: 100 314.399994

    return
  }
}
