//===-- IMEXPasses.td - Conversion pass definition file ----*- tablegen -*-===//
//
// Copyright 2023 Intel Corporation
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
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// NDArrayToLinalg
//===----------------------------------------------------------------------===//

def ConvertNDArrayToLinalg : Pass<"convert-ndarray-to-linalg"> {
  let summary = "Convert from the NDArray dialect to the Linalg and Dist dialects";
  let description = [{
    Convert NDArray dialect operations into the LLVM IR dialect operations.

    #### Input invariant

    -   all tensors are NDArrayType

    #### Output IR

    - a NDArrayType will be lowered to a unrealized_conversion_cast to tuple of
      * tensor: RankedTensor
      * device: device where the tensor lives (AnyType, default=None)
      * team: a group of processes among which the tensor is distributed
        (AnyType, default=None)
      * rthandle: a handle for communication with the dist dialect/runtime
        (AnyType, default=None)
    - On function boundaries (func::callOp, func::returnOp, func::functionOp)
      NDArrayTypes get expanded into 4 separate arguments (rtensor, device, team, handle).
    - NDArray operations are converted to Linalg operations, accompaigned by
      * operations of the Dist dialect if the input tensors are distributed
      * FIXME iGPU::deviceRegionOps if the input tensors live on a device
  }];
  let constructor = "imex::createConvertNDArrayToLinalgPass()";
  let dependentDialects = ["::mlir::linalg::LinalgDialect",
                           "::mlir::affine::AffineDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::tensor::TensorDialect",
                           "::mlir::tosa::TosaDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::shape::ShapeDialect",
                           "::mlir::bufferization::BufferizationDialect",
                           "::imex::region::RegionDialect"];
  let options = [];
}

//===----------------------------------------------------------------------===//
// DistToStandard
//===----------------------------------------------------------------------===//

def ConvertDistToStandard: Pass<"convert-dist-to-standard"> {
  let summary = "Convert from the Dist dialect to runtime calls.";
  let description = [{
    Convert Dist dialect operations into standard dialect operations
    by inserting calls into a distributed runtime.

    Necessary prototypes of runtime functions will be added.
  }];
  let constructor = "::imex::createConvertDistToStandardPass()";
  let dependentDialects = ["::imex::ndarray::NDArrayDialect",
                           "::imex::distruntime::DistRuntimeDialect",
                           "::mlir::linalg::LinalgDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::tensor::TensorDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::bufferization::BufferizationDialect"];
  let options = [];
}

//===----------------------------------------------------------------------===//
// DropRegions
//===----------------------------------------------------------------------===//

def DropRegions: Pass<"drop-regions"> {
  let summary = "Remove all Region dialect operations.";
  let description = [{
    Removes all Region dialect operations.
  }];
  let constructor = "::imex::createDropRegionsPass()";
  let dependentDialects = [];
  let options = [];
}


//===----------------------------------------------------------------------===//
// GPUToSPIRV
//===----------------------------------------------------------------------===//

def ConvertGPUXToSPIRV : Pass<"imex-convert-gpu-to-spirv", "::mlir::ModuleOp"> {
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

//===----------------------------------------------------------------------===//
// GPUToGPUX
//===----------------------------------------------------------------------===//

def ConvertGPUToGPUX: Pass<"convert-gpu-to-gpux"> {
  let summary = "Convert from the GPU dialect to the GPUX dialect.";
  let description = [{
    This pass converts GPU dialect operations into the GPUX dialect operations.
    GPUX dialect is a custom dialect. GPUX dialect ops extend the upstream GPU
    Dialect ops and adds a stream argument to them for specifying which stream
    to run the gpu operations on.

    Below is an example of how this Pass will convert the IR from GPU to GPUX.

    Input IR

        func.func @main() attributes {llvm.emit_c_interface} {
        %c8 = arith.constant 8 : index
        %c1 = arith.constant 1 : index
        %cst = arith.constant 2.200000e+00 : f32
        %cst_0 = arith.constant 1.100000e+00 : f32
        %cst_1 = arith.constant 0.000000e+00 : f32
        %0 = gpu.alloc  () : memref<8xf32>
        %1 = gpu.alloc  () : memref<8xf32>
        %2 = gpu.alloc  () : memref<8xf32>
        gpu.launch_func  @main_kernel::@main_kernel blocks in (%c8, %c1, %c1) threads in (%c1, %c1, %c1) args(%0 : memref<8xf32>, %1 : memref<8xf32>, %2 : memref<8xf32>)
        gpu.dealloc(%0) : memref<8xf32>
        gpu.dealloc(%1) : memref<8xf32>
        gpu.dealloc(%2) : memref<8xf32>
        return
      }

    Output IR

        func.func @main() attributes {llvm.emit_c_interface} {
        %c8 = arith.constant 8 : index
        %c1 = arith.constant 1 : index
        %cst = arith.constant 2.200000e+00 : f32
        %cst_0 = arith.constant 1.100000e+00 : f32
        %cst_1 = arith.constant 0.000000e+00 : f32
        %stream = gpux.create_stream () : () -> !gpux.StreamType
        %0 = gpux.alloc (%stream) : (!gpux.StreamType) -> memref<8xf32>
        %1 = gpux.alloc (%stream) : (!gpux.StreamType) -> memref<8xf32>
        %2 = gpux.alloc  (%stream) : (!gpux.StreamType) -> memref<8xf32>
        gpux.launch_func  @main_kernel::@main_kernel blocks in (%c8, %c1, %c1) threads in (%c1, %c1, %c1) args(%stream : !gpux.StreamType, %memref : memref<8xf32>, %memref_2 : memref<8xf32>, %memref_3 : memref<8xf32>)
        gpux.dealloc(%stream, %0) : (!gpux.StreamType, memref<8xf32>) -> ()
        gpux.dealloc(%stream, %1) : (!gpux.StreamType, memref<8xf32>) -> ()
        gpux.dealloc(%stream, %2) : (!gpux.StreamType, memref<8xf32>) -> ()
        gpux.stream.destroy(%stream) : (!gpux.StreamType) -> ()
    return
  }

  }];
  let constructor = "imex::createConvertGPUToGPUXPass()";
  let dependentDialects = ["::imex::gpux::GPUXDialect"];
}


//===----------------------------------------------------------------------===//
// GPUXToLLVM
//===----------------------------------------------------------------------===//

def ConvertGPUXToLLVM : Pass<"convert-gpux-to-llvm", "::mlir::ModuleOp"> {
  let summary = "Convert from the GPUX dialect to LLVM dialect with GPU runtime calls";
  let description = [{
    Converts gpux dialect operations into the LLVM IR dialect operations.

    #### Input invariant

    #### Output IR

  }];
  let constructor = "imex::createConvertGPUXToLLVMPass()";
  let dependentDialects = [];
  let options = [];
}


//===----------------------------------------------------------------------===//
// XeTileToXeGPU
//===----------------------------------------------------------------------===//

def ConvertXeTileToXeGPU: Pass<"convert-xetile-to-xegpu", "::mlir::gpu::GPUModuleOp"> {
  let summary = "Convert from the XeTile dialect to the XeGPU dialect.";
  let description = [{
    Convert XeTile dialect operations into the XeGPU dialect operations. It expects
    the input code is tiled using xetile-blocking.

    #### Input invariant

    func.func @sglevel_tiled_load_tile(%a: memref<1024x1024xf16>, %b: memref<1024x1024xf16>, %c: memref<1024x1024xf32>) {
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %1 = xetile.init_tile %a[%c0, %c64] : memref<1024x1024xf16> -> !xetile.tile<8x16xf16>
      %2 = xetile.load_tile %1 : !xetile.tile<8x16xf16> -> vector<8x16xf16>
	    return
    }

    #### Output IR

    func.func @sglevel_tiled_load_tile(%a: memref<1024x1024xf16>, %b: memref<1024x1024xf16>, %c: memref<1024x1024xf32>) {
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %0 = xegpu.create_nd_tdesc %arg0[%c0, %c64] {boundary_check = true} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %2 = xegpu.load_nd %0 {l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<uncached>, l3_hint = #xegpu.cache_hint<uncached>} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
	    return
    }
  }];

  let constructor = "::imex::createConvertXeTileToXeGPUPass()";
  let dependentDialects = ["::mlir::xegpu::XeGPUDialect",
                           "::imex::xetile::XeTileDialect",
                           "::mlir::gpu::GPUDialect",
                           "::mlir::vector::VectorDialect",
                           "::mlir::arith::ArithDialect",
                           ];
  let options = [
    Option<"device", "device", "std::string",
            /*default=*/"\"pvc\"",
            "gpu platform architecture where these ops are running">,
    Option<"EnableTransform", "enable-2d-transform", "bool",
            /*default=*/"false",
            "Using 2D transform or 4D Conversion.">
 ];
}


//===----------------------------------------------------------------------===//
// XeGPUToVC
//===----------------------------------------------------------------------===//

def ConvertXeGPUToVC : Pass<"convert-xegpu-to-vc", "::mlir::gpu::GPUModuleOp"> {
  let summary = "Generate vc-intrinsics functions for xegpu dialect operations";
  let description = [{
    Convert XeGPU dialect operations into the Func dialect calls to vc-intrinsics
    functions.
    }];
  let options = [
  ];

  let dependentDialects = ["::mlir::xegpu::XeGPUDialect",
                           "::mlir::vector::VectorDialect",
                            "::mlir::memref::MemRefDialect",
                            "::mlir::LLVM::LLVMDialect",
                            "::mlir::func::FuncDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::spirv::SPIRVDialect"
                            ];
  let constructor = "imex::createConvertXeGPUToVCPass()";
}


//===----------------------------------------------------------------------===//
// MathToVC
//===----------------------------------------------------------------------===//
// high-precision-interim-calculation
def ConvertMathToVC : Pass<"convert-math-to-vc", "::mlir::gpu::GPUModuleOp"> {
  let summary = "Generate vc-intrinsics functions for select math dialect operations";
  let description = [{
    Convert select math dialect operations into the Func dialect calls to vc-intrinsics
    functions.
    Some math operations are not supported by the VC compiler (IGC vector backend)
    and need to be converted to vc-intrinsic calls.
    This pass converts these math operations to vc-intrinsics.
    }];
  let options = [
    Option<"enableHighPrecisionInterimCalculation", "enable-high-precision-interim-calculation", "bool",
           /*default=*/"false",
           "Enables high precision (f32) interim  calculation for math operations."
           "For any interim instruction added as part of the conversion will be high precision(f32).">
  ];

  let dependentDialects = ["::mlir::math::MathDialect",
                           "::mlir::vector::VectorDialect",
                            "::mlir::LLVM::LLVMDialect",
                            "::mlir::func::FuncDialect",
                           "::mlir::arith::ArithDialect"
                            ];
  let constructor = "imex::createConvertMathToVCPass()";
}


//===----------------------------------------------------------------------===//
// ArithToVC
//===----------------------------------------------------------------------===//
// high-precision-interim-calculation
def ConvertArithToVC : Pass<"convert-arith-to-vc", "::mlir::gpu::GPUModuleOp"> {
  let summary = "Generate vc-intrinsics functions for select arith dialect operations";
  let description = [{
    Convert select arith dialect operations into the Func dialect calls to vc-intrinsics
    functions.
    Some arith operations SPIR-V counterpart are not supported by the VC compiler
    (IGC vector backend) or not performant enough
    and need to be converted to vc-intrinsic calls.
    This pass converts these arith operations to vc-intrinsics.
    }];
  let options = [
    Option<"enableHighPrecisionInterimCalculation", "enable-high-precision-interim-calculation", "bool",
           /*default=*/"false",
           "Enables high precision (f32) interim  calculation for arith operations."
           "For any interim instruction added as part of the conversion will be high precision(f32).">
  ];

  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::vector::VectorDialect",
                            "::mlir::LLVM::LLVMDialect",
                            "::mlir::func::FuncDialect"
                            ];
  let constructor = "imex::createConvertArithToVCPass()";
}


#endif // _IMEX_CONVERSION_PASSES_TD_INCLUDED_
