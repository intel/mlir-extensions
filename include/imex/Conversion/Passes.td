//===-- IMEXPasses.td - Conversion pass definition file ----*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the base classes of IMEX conversino passes.
///
//===----------------------------------------------------------------------===//

#ifndef _IMEX_CONVERSION_PASSES_TD_INCLUDED_
#define _IMEX_CONVERSION_PASSES_TD_INCLUDED_

include "mlir/Pass/PassBase.td"


include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/GPU/IR/GPUBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// PTensorToLinalg
//===----------------------------------------------------------------------===//

def ConvertPTensorToLinalg : Pass<"convert-ptensor-to-linalg", "::mlir::ModuleOp"> {
  let summary = "Convert from the PTensor dialect to the Linalg and Dist dialects";
  let description = [{
    Convert PTensor dialect operations into the LLVM IR dialect operations.

    #### Input invariant

    -   all tensors are PTensorType;

    #### Output IR

    - PTensorTypes are converted to mlir::Tensor
    - PTensor operations are converted to Linalg operations, accompaigned by
      operations of the Dist dialect if the input tensors are distributed
  }];
  let constructor = "imex::createConvertPTensorToLinalgPass()";
  let dependentDialects = ["::imex::dist::DistDialect",
                           "::mlir::linalg::LinalgDialect",
                           "::mlir::AffineDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::tensor::TensorDialect",
                           "::mlir::arith::ArithmeticDialect",
                           "::mlir::shape::ShapeDialect"];
  let options = [];
}
//===----------------------------------------------------------------------===//
// GPUToSPIRV
//===----------------------------------------------------------------------===//

def ConvertGPUXToSPIRV : Pass<"convert-gpux-to-spirv", "::mlir::ModuleOp"> {
  let summary = "Convert GPU dialect to SPIR-V dialect";
  let description = [{
    This pass extends upstream GPU dialect to SPIR-V dialect pass by adding more
    conversion patterns like SCF, math and control flow.
    This pass converts gpu.func ops inside gpu.module op.

    For more detailed documentation, refer upstream MLIR Pass -convert-gpu-to-spirv
    https://mlir.llvm.org/docs/Passes/#-convert-gpu-to-spirv-convert-gpu-dialect-to-spir-v-dialect


    Below example shows how GPU kernel module is converted to SPIRV module along with other ops conversion within gpu.func. from dialect like
memref, arith and math.

    Input

    ```mlir
      module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spv.resource_limits<>>} {
        func.func @load_store(%arg0: memref<12x4xf32>, %arg1: memref<12x4xf32>, %arg2: memref<12x4xf32>) {
          %c0 = arith.constant 0 : index
          .
          .
          .
          gpu.launch_func  @kernels::@load_store_kernel blocks in (%0, %c1_2, %c1_2) threads in (%1, %c1_2, %c1_2) args(%arg0 : memref<12x4xf32>, %arg1 : memref<12x4xf32>, %arg2 : memref<12x4xf32>, %c0 : index, %c0_0 : index, %c1 : index, %c1_1 : index)
          return
        }
          gpu.module @kernels {
          gpu.func @load_store_kernel(%arg0: memref<12x4xf32>, %arg1: memref<12x4xf32>, %arg2: memref<12x4xf32>, %arg3: index, %arg4: index, %arg5: index, %arg6: index) kernel attributes {spv.entry_point_abi = #spv.entry_point_abi<local_size = dense<[16, 1, 1]> : vector<3xi32>>} {
            cf.br ^bb1
          ^bb1:  // pred: ^bb0
            %0 = gpu.block_id  x
          .
          .
          .
            %6 = gpu.grid_dim  x
            .
            .
            .
            %12 = arith.addi %arg3, %0 : index
            %13 = arith.addi %arg4, %3 : index
            %14 = memref.load %arg0[%12, %13] : memref<12x4xf32>
            %15 = memref.load %arg1[%12, %13] : memref<12x4xf32>
            %16 = arith.addf %14, %15 : f32
            memref.store %16, %arg2[%12, %13] : memref<12x4xf32>
            %17 = math.rsqrt %14 : f32
            gpu.return
          }
        }
     ```

    Output

    ```mlir
      spv.module @__spv__kernels Logical GLSL450 {
        spv.GlobalVariable @__builtin_var_NumWorkgroups__ built_in("NumWorkgroups") : !spv.ptr<vector<3xi32>, Input>
        spv.GlobalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
        spv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
        spv.func @load_store_kernel(%arg0: !spv.ptr<!spv.struct<(!spv.array<48 x f32, stride=4> [0])>, StorageBuffer> {spv.interface_var_abi = #spv.interface_var_abi<(0, 0)>}, %arg1: !spv.ptr<!spv.struct<(!spv.array<48 x f32, stride=4> [0])>, StorageBuffer> {spv.interface_var_abi = #spv.interface_var_abi<(0, 1)>}, %arg2: !spv.ptr<!spv.struct<(!spv.array<48 x f32, stride=4> [0])>, StorageBuffer> {spv.interface_var_abi = #spv.interface_var_abi<(0, 2)>}, %arg3: i32 {spv.interface_var_abi = #spv.interface_var_abi<(0, 3), StorageBuffer>}, %arg4: i32 {spv.interface_var_abi = #spv.interface_var_abi<(0, 4), StorageBuffer>}, %arg5: i32 {spv.interface_var_abi = #spv.interface_var_abi<(0, 5), StorageBuffer>}, %arg6: i32 {spv.interface_var_abi = #spv.interface_var_abi<(0, 6), StorageBuffer>}) "None" attributes {spv.entry_point_abi = #spv.entry_point_abi<local_size = dense<[16, 1, 1]> : vector<3xi32>>, workgroup_attributions = 0 : i64} {
          spv.Branch ^bb1
        ^bb1:  // pred: ^bb0
          %__builtin_var_WorkgroupId___addr = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
          %0 = spv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi32>
          %1 = spv.CompositeExtract %0[0 : i32] : vector<3xi32>
        .
        .
        .
        %cst0_i32 = spv.Constant 0 : i32
          %cst0_i32_7 = spv.Constant 0 : i32
          %cst4_i32 = spv.Constant 4 : i32
          %20 = spv.IMul %cst4_i32, %18 : i32
          %21 = spv.IAdd %cst0_i32_7, %20 : i32
          %cst1_i32_8 = spv.Constant 1 : i32
          %22 = spv.IMul %cst1_i32_8, %19 : i32
          %23 = spv.IAdd %21, %22 : i32
          %24 = spv.AccessChain %arg0[%cst0_i32, %23] : !spv.ptr<!spv.struct<(!spv.array<48 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
          %25 = spv.Load "StorageBuffer" %24 : f32
          %cst0_i32_9 = spv.Constant 0 : i32
          %cst0_i32_10 = spv.Constant 0 : i32
          %cst4_i32_11 = spv.Constant 4 : i32
          %26 = spv.IMul %cst4_i32_11, %18 : i32
          %27 = spv.IAdd %cst0_i32_10, %26 : i32
          %cst1_i32_12 = spv.Constant 1 : i32
          %28 = spv.IMul %cst1_i32_12, %19 : i32
          %29 = spv.IAdd %27, %28 : i32
          %30 = spv.AccessChain %arg1[%cst0_i32_9, %29] : !spv.ptr<!spv.struct<(!spv.array<48 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
          %31 = spv.Load "StorageBuffer" %30 : f32
          %32 = spv.FAdd %25, %31 : f32
          %cst0_i32_13 = spv.Constant 0 : i32
          %cst0_i32_14 = spv.Constant 0 : i32
          %cst4_i32_15 = spv.Constant 4 : i32
          %33 = spv.IMul %cst4_i32_15, %18 : i32
          %34 = spv.IAdd %cst0_i32_14, %33 : i32
          %cst1_i32_16 = spv.Constant 1 : i32
          %35 = spv.IMul %cst1_i32_16, %19 : i32
          %36 = spv.IAdd %34, %35 : i32
          %37 = spv.AccessChain %arg2[%cst0_i32_13, %36] : !spv.ptr<!spv.struct<(!spv.array<48 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
          spv.Store "StorageBuffer" %37, %32 : f32
          %38 = spv.GL.InverseSqrt %25 : f32
          spv.Return
        }
      }
    ```

    Below example shows conversion of SCF for loop along with memref conversions into SPIRV within gpu kernel module.
    Input

    ```mlir
    gpu.module @kernels {
      gpu.func @loop_kernel(%arg0: memref<10xf32>, %arg1: memref<10xf32>) kernel attributes {spv.entry_point_abi = #spv.entry_point_abi<local_size = dense<[32, 4, 1]> : vector<3xi32>>} {
        cf.br ^bb1
      ^bb1:  // pred: ^bb0
        %c4 = arith.constant 4 : index
        %c42 = arith.constant 42 : index
        %c2 = arith.constant 2 : index
        scf.for %arg2 = %c4 to %c42 step %c2 {
          %0 = memref.load %arg0[%arg2] : memref<10xf32>
          memref.store %0, %arg1[%arg2] : memref<10xf32>
        }
        gpu.return
    }
    ```

    Output

    ```mlir
    spv.module @__spv__kernels Logical GLSL450 {
      spv.func @loop_kernel(%arg0: !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer> {spv.interface_var_abi = #spv.interface_var_abi<(0, 0)>}, %arg1: !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer> {spv.interface_var_abi = #spv.interface_var_abi<(0, 1)>}) "None" attributes {spv.entry_point_abi = #spv.entry_point_abi<local_size = dense<[32, 4, 1]> : vector<3xi32>>, workgroup_attributions = 0 : i64} {
        spv.Branch ^bb1
      ^bb1:  // pred: ^bb0
        .
        .
        .
        spv.mlir.loop {
          spv.Branch ^bb1(%cst4_i32 : i32)
        ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
          %1 = spv.SLessThan %0, %cst42_i32 : i32
          spv.BranchConditional %1, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
        .
        .
        .
          %4 = spv.AccessChain %arg0[%cst0_i32, %3] : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
          %5 = spv.Load "StorageBuffer" %4 : f32
          .
          .
          .
          %8 = spv.AccessChain %arg1[%cst0_i32_1, %7] : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
          spv.Store "StorageBuffer" %8, %5 : f32
          %9 = spv.IAdd %0, %cst2_i32 : i32
          spv.Branch ^bb1(%9 : i32)
        ^bb3:  // pred: ^bb1
          spv.mlir.merge
        }
        spv.Return
      }
    }
    ```
  }];
  let constructor = "imex::createConvertGPUXToSPIRVPass()";
  let dependentDialects = ["::mlir::spirv::SPIRVDialect"];
}
#endif // _IMEX_CONVERSION_PASSES_TD_INCLUDED_
