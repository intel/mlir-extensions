# RFC: Intel Vector Compute Dialect (IVC)

## Summary

We propose a dialect for Intel Vector Compute Operations (e.g., dpas). These operations are based on Intel vc-intrinsics (https://github.com/intel/vc-intrinsics/blob/master/GenXIntrinsics/include/llvm/GenXIntrinsics/Intrinsic_definitions.py). The operations provides an MLIR entry point for these instrinsics. Currently a subset of the vc-intrinsics are added as ops in here. We may potentially add more ops down the line on necessasity basis. 


## Motivation

The native execution pattern in Intel GPUs (e.g., PVC) is SIMD (Single Instruction Multiple Data) as opposed to SIMT (Single Instruction, Multiple Threads), the familiar GPU execution pattern used in NVIDIA GPUs. However, as SIMT has been the prominent execution pattern, Intel Graphics Compiler (IGC) provides two back-end (essentially two different compilers, IGC and Vector compiler) to support both SIMD and SIMT style programming.

- Vector back-end (SIMD)
    - Used by DPC++ ESIMD instructions
- Scalar back-end (SIMT)
    - Used by OpenCL

Although IGC supports SIMT style programming via scalar backend, the computation units (and ISA) in Intel GPUs are vector units. Hence, IGC auto-vectorizes the SIMT code to target Intel GPUs, this extra auto-vectorization step may reduce the performance. One of the reasons auto vectorizations often performs sub-optimally are, utilization of the full length of the vector registers.
Therefore, it is important that we target Vector back-end for the code generation purpose in the MLIR path.  

Currently, IGC provides a list of intrinsics via vc-intrinsics library that exposes a list of vector ops that can be used for vector code generation.
However, ***INEL IS MOVING AWAY FROM vc-intrisics BY THE END OF Q1’2023. THESE INTRINSICS OPS WILL BE ADDED AS PART OF SPIR-V INTEL VECTOR EXTENSION***. Until the move to SPIR-V is complete however, we are still dependent on these intrinsics. 

Hence, we need a dialect that can be a bridge between this transition. We design this dialect, **spirvX** (spirv extension) in such a way that it serves the current need while still maintain some core aspect of the dialect (ops and transformations) that can be carried over once the transition is complete to spirv.

Most of the vc-intrinsics functions will transition to SPIR-V as standalone SPIR-V ops (e.g., llvm.genx.nbarrier to OpNamedControlBarrierSubgroupArriveINTEL and OpNamedControlBarrierSubgroupWaitINTEL).


However, ***not all vc-intrinsics ops will transition to SPIR-V as standalone ops***, some ops from vc-intrinsics will be added to the SPIR-V as **Decoration** to existing SPIR-V ops. A decoration is essentially a qualifier to an entity (e.g., op, variable) in SPIR-V. Decorations are represented as **attributes** in spirv dialect. An example of such a case is the cache control/prefetching instructions in vc-intrinsics, *llvm.genx.lsc.load.*. and *llvm.genx.lsc.store.*.*. These instructions would essentially translate to spirv load and store instructions with necessary decorations (attributes in MLIR).

vc-intrinsics:

```
`// Using llvm.genx.lsc.load. with L1 and L3(L2) both cached`

%v2 =  spv.FunctionCall @llvm_genx_lsc_load_stateless_v32i32_i1_i64(%true, %uchar_0, %uchar_2, %uchar_2, %ushort_1, %uint_0, %uchar_3, %uchar_7, %uchar_2, %uchar_0, %arg_2, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i64, i32) -> vector<32 x i32>
```

SPIR-V extension in spirv dialect:

```
// Using potential SPIR-V extensions, L1 and L3(L2) both cached
%cst_0 = spv.Constant 0 : i32
%ptr_to_load_from = spirv.Variable : !spirv.ptr<vector<32xi32>, Function>
%offset_ptr = spirv.AccessChain %ptr_to_load_from[%cst_0] : !spirv.ptr<vector<32xi32>, CrossWorkgroup>
%loaded_val = spirv.Load "CrossWorkgroup", ["CacheControlLoadINTEL", 1, 1]  %offset_ptr ["Volatile"] : vector<32xi32>

```

As a result, we cannot define the ops just yet untill the specification is complete. Therefore, in this dialect, we are proposing like for like replacement for all vc-intrinsic functions. In other words, each vc-intrinsics has a corresponding op here. However, we are only focusing on the vc-intrinsics that are found in DL applications, hence not all vc-intrinsics have a corresponding op at this point.


## Ops

### ALU Ops

#### spirvx.dpas (spirvX::DpasOp)

Intel DPAS instruction.

Syntax:

```
operation ::= 
```



<!-- 
Syntax:

```

operation ::= 
```

Syntax:

```
operation ::= 
``` -->


<!--

The primary differences we expect to see between the vc-intrinsics and their potential transition to SPIR-V is that 


   

Therefore, this dialect is expected to be 


Here are a few reasons why eSIMD might be better than SIMT for Intel GPUs:


Although, IGC provides both scalar and vector back-end, generating explicit SIMD (eSIMD) code targeting the vector back-end is more performant. 






Historically GPUs have followed SIMT (Single Instruction Multi Threading) programming style (e.g., NVIDIA gpus). However, the Intel GPUs inherent programming model is SIMD (Single Instruction Multiple Data), although it supports SIMT model as well. However, to get the best performance one should utilize the SIMD style programming, which means generating SIMD instructions.


Intel exposes a range of vector compute instructions through a range of LLVM intrinsics (vc-intrinsics library). The intrinsics are an easy way to instruct the IGC compiler to generate respective vector instructions. The library contains intrinsics that:

1. Represents Intel vISA instructions and
2. Some CM (C for Metal) specific instructions


are a combination of instructions that repre  
-->
