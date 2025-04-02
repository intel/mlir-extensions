//===-- XeVMOps.td - XeVM dialect definition ---------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef XEVMIR_OPS
#define XEVMIR_OPS

include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def XeVM_Dialect : Dialect {
  let name = "xevm";
  let cppNamespace = "::imex::xevm";
  let dependentDialects = ["LLVM::LLVMDialect"];

  let extraClassDeclaration = [{
    /// Get the name for the attribute used to specify cache control
    /// decorations.
    static constexpr ::llvm::StringRef getCacheControlsAttrName() {
      return ::llvm::StringLiteral("xevm.DecorationCacheControl");
    }
  }];

  let useDefaultAttributePrinterParser = 1;
}

class XeVM_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<XeVM_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

class XeVM_Op<string mnemonic, list<Trait> traits = []> :
  Op<XeVM_Dialect, mnemonic, traits>;

def XeVM_ElemType : AnyTypeOf<[AnyI8, AnyI16, AnyI32, F32, F16, BF16]>;

def XeVM_LoadCacheControl : I32EnumAttr<"LoadCacheControl", "XeVM load ops cache control",
  [
    I32EnumAttrCase<"DEFAULT",     0,  "Default">,
    I32EnumAttrCase<"UC",          1,  "UC">,     // uncached
    I32EnumAttrCase<"C",           2,  "C">,      // cached
    I32EnumAttrCase<"S",           3,  "S">,      // streaming
    I32EnumAttrCase<"IAR",         4,  "IAR">,    // invalidate-after-read
  ]> {
  let cppNamespace = "::imex::xevm";
  let genSpecializedAttr = 0;
}

def XeVM_LoadCacheControlAttr:
  EnumAttr<XeVM_Dialect, XeVM_LoadCacheControl, "load_cache_control"> {
    let summary = [{ }];
    let assemblyFormat = "$value";
}

def XeVM_StoreCacheControl : I32EnumAttr<"StoreCacheControl", "XeVM store ops cache control",
  [
    I32EnumAttrCase<"DEFAULT",     0,  "Default">,
    I32EnumAttrCase<"UC",          1,  "UC">,    // uncached
    I32EnumAttrCase<"WT",          2,  "WT">,    // write-through
    I32EnumAttrCase<"S",           3,  "S">,     // streaming
    I32EnumAttrCase<"WB",          4,  "WB">,    // write back
  ]> {
  let cppNamespace = "::imex::xevm";
  let genSpecializedAttr = 0;
}

def XeVM_StoreCacheControlAttr:
  EnumAttr<XeVM_Dialect, XeVM_StoreCacheControl, "store_cache_control"> {
    let summary = [{ }];
    let assemblyFormat = "$value";
}

def XeVM_BlockLoad2dOp : XeVM_Op<"blockload2d">,
  Results<(outs FixedVectorOfRankAndType<[1,2,3], [XeVM_ElemType]>:$res)>,
  Arguments<(ins
    Arg<LLVM_AnyPointer, "", [MemRead]>:$ptr,
    I32:$base_width,
    I32:$base_height,
    I32:$base_pitch,
    I32:$x,
    I32:$y,
    I32Attr:$elem_size_in_bits,
    I32Attr:$tile_width,
    I32Attr:$tile_height,
    I32Attr:$v_blocks,
    I1Attr:$transpose,
    I1Attr:$vnni_transform,
    DefaultValuedAttr<XeVM_LoadCacheControlAttr, "::imex::xevm::LoadCacheControl::DEFAULT">:$l1_cache_control,
    DefaultValuedAttr<XeVM_LoadCacheControlAttr, "::imex::xevm::LoadCacheControl::DEFAULT">:$l3_cache_control
  )> {

  let summary = "2D block load";

  let description = [{
    The `xevm.blockload2d` operation loads a two dimensional matrix tile
    from a larger matrix residing in memory. The parameters are:
      $ptr - the base address of the matrix containing the tile to load
      $base_width, $base_height, $base_pitch - the shape of matrix
      $x, $y, $tile_width, $tile_height - the starting offsets and shape of the tile to load
      $elem_size_in_bits - the size in bits of the matrix element
        - 32 for f32, bf32
        - 16 for f16, int16, bf16
        - 8 for int8, int4, int2
      $v_blocks - number of tiles to load
      $transpose - transpose the tile in registers (useful for 32 bit element type)
      $vnni_transform - transpose and pack the submatrix in registers (useful for < 32 bit element types)
      $cache_control - an enumerator that sets the L1 and L3 cache behaviour

    Notes:
      - pitch is the physical stride between the first columns of the current row and the subsequent row,
        this may include (possibly implicit) padding, alignment, or other factors.
      - the $transpose and $vnni_transform parameters are mutual exclusive
      - transposing the tile loaded is typically used for the B matrix operand
        (D = C + A * B), where A has row-major layout in registers and B should have column-major layout.
      - if the tile loaded contains out of bound elements of the matrix, they are filled with 0.
      - coordinate is provided in elements, while width and pitch are provided in bytes.

    Example:
    ```mlir
      %base_width_a = arith.constant 32 : i32
      %base_height_a = arith.constant 8 : i32
      %base_pitch_a = arith.constant 32 : i32
      %x = arith.constant 0 : i32
      %y = arith.constant 0 : i32
      %loaded_a = xevm.blockload2d %src, %base_width_a, %base_height_a, %base_pitch_a, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
    ```
  }];

  let assemblyFormat = [{
    operands ` ` `{` `elem_size_in_bits` `=` $elem_size_in_bits `,` `tile_width` `=` $tile_width `,`
    `tile_height` `=` $tile_height `,` `v_blocks` `=` $v_blocks `,` `transpose` `=` $transpose `,`
    `vnni_transform` `=` $vnni_transform `,` `l1_cache_control` `=` $l1_cache_control `,`
    `l3_cache_control` `=` $l3_cache_control `}` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def XeVM_BlockStore2dOp : XeVM_Op<"blockstore2d">,
  Arguments<(ins
    Arg<LLVM_AnyPointer, "", [MemWrite]>:$ptr,
    I32:$base_width,
    I32:$base_height,
    I32:$base_pitch,
    I32:$x,
    I32:$y,
    I32Attr:$elem_size_in_bits,
    I32Attr:$tile_width,
    I32Attr:$tile_height,
    I32Attr:$v_blocks,
    FixedVectorOfRankAndType<[1, 2, 3], [XeVM_ElemType]>:$stored_val,
    DefaultValuedAttr<XeVM_StoreCacheControlAttr, "::imex::xevm::StoreCacheControl::DEFAULT">:$l1_cache_control,
    DefaultValuedAttr<XeVM_StoreCacheControlAttr, "::imex::xevm::StoreCacheControl::DEFAULT">:$l3_cache_control
  )> {

  let summary = "2D block store";

  let description = [{
    The `xevm.blockstore2d` operation stores a two dimensional tile into a
    larger matrix residing in memory. The parameters are:
      $ptr - the base address of the matrix where to store the tile
      $base_width, $base_height, $base_pitch - the shape of the matrix
      $x, $y, $tile_width, $tile_height - the starting offsets and shape of the tile to store
      $elem_size_in_bits - the size in bits of the matrix element
        - 32 for f32, bf32
        - 16 for f16, int16, bf16
        - 8 for int8, int4, int2
      $v_blocks - number of tiles to store
      $cache_control - an enumerator that sets the L1 and L3 cache behaviour
      $stored_val - the tile to store

    Notes:
      - pitch is the physical stride between the first columns of the current row and the subsequent row,
        this may include (possibly implicit) padding, alignment, or other factors.
      - coordinate is provided in elements, while width and pitch are provided in bytes.

    Example:
    ```mlir
      %base_width_c = arith.constant 64 : i32
      %base_height_c = arith.constant 8 : i32
      %base_pitch_c = arith.constant 64 : i32
      %x = arith.constant 0 : i32
      %y = arith.constant 0 : i32
      xevm.blockstore2d %dst, %base_width_c, %base_height_c, %base_pitch_c, %x, %y, %src {elem_size_in_bits=32, tile_width=16, tile_height=8, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
    ```
  }];

  let assemblyFormat = [{
    operands ` ` `{` `elem_size_in_bits` `=` $elem_size_in_bits `,` `tile_width` `=` $tile_width `,`
      `tile_height` `=` $tile_height `,` `v_blocks` `=` $v_blocks `,` `l1_cache_control` `=` $l1_cache_control `,`
      `l3_cache_control` `=` $l3_cache_control `}`
      attr-dict `:` `(` type(operands) `)`
  }];

  let hasVerifier = 1;
}

def XeVM_MemoryScope : I32EnumAttr<"MemoryScope", "Memory scope for memory operations",
    [
      I32EnumAttrCase<"WORKGROUP", 0, "workgroup">,
      I32EnumAttrCase<"CLUSTER", 1, "cluster">,
      I32EnumAttrCase<"GPU", 2, "gpu">,
      I32EnumAttrCase<"SYSTEM", 3, "system">
    ]>{
  let cppNamespace = "imex::xevm";
}

def XeVM_AddrSpace : I32EnumAttr<"OclAddrSpace", "Address spaces in OpenCL",
    [
      I32EnumAttrCase<"kPrivate",  0, "private">,  // OpenCL Workitem address space, SPIRV function
      I32EnumAttrCase<"kGlobal",   1, "global">,   // OpenCL Global memory, SPIRV crossworkgroup
      I32EnumAttrCase<"kConstant", 2, "constant">, // OpenCL Constant memory, SPIRV uniform constant
      I32EnumAttrCase<"kShared",   3, "shared">,   // OpenCL Local memory, SPIRV workgroup
      I32EnumAttrCase<"kGeneric",  4, "generic">   // OpenCL Generic memory, SPIRV generic
    ]>{
  let cppNamespace = "imex::xevm";
}

def XeVM_MemfenceOp : XeVM_Op<"memfence">,
  Arguments<(ins
    XeVM_MemoryScope:$scope,
    DefaultValuedAttr<XeVM_AddrSpace, "imex::xevm::OclAddrSpace::kGeneric"> :$addrspace
  )> {
  let summary = "Work-item's memory fence.";
  let description = [{
    This operation ensures that all prior memory accesses of this
    work-item to `addrspace` are visible to all other work-items in `scope`.
    Parameters description:
    $scope - specify the memory scope at which all other work-items should observe
                memory operations prior to the fence.
    $addrspace - specify the address space of work-item's memory accesses
                to be affected by the fence.
  }];
  let assemblyFormat = [{`addrspace` `=` $addrspace `,` `scope` `=` $scope attr-dict}];
}

def XeVM_PrefetchOp : XeVM_Op<"prefetch">,
  Arguments<(ins
    Arg<LLVM_AnyPointer, "", [MemRead]>:$ptr,
    XeVM_AddrSpace:$addrspace,
    DefaultValuedAttr<XeVM_LoadCacheControlAttr, "::imex::xevm::LoadCacheControl::DEFAULT">:$l1_cache_control,
    DefaultValuedAttr<XeVM_LoadCacheControlAttr, "::imex::xevm::LoadCacheControl::DEFAULT">:$l3_cache_control
  )> {
  let summary = "Prefetch data into a cache subsystem.";
  let description = [{
    Work-item issues a prefetch from global memory to L1/L3 cache:
    $ptr - memory pointer.
    $addrspace - address space of a pointer, must be generic or global.
    $cache_control - specify caching options (e.g., L1c, L3uc).
  }];
  let assemblyFormat = [{
    operands ` ` `{` `addrspace` `=` $addrspace `,`  `l1_cc` `=` $l1_cache_control `,`  `l3_cc` `=` $l3_cache_control `}`
      attr-dict `:` `(` type(operands) `)`
  }];

  // let hasVerifier = 1;
}

def XeVM_BlockPrefetch2dOp : XeVM_Op<"blockprefetch2d">,
  Arguments<(ins
    Arg<LLVM_AnyPointer, "", [MemRead]>:$ptr,
    I32:$base_width,
    I32:$base_height,
    I32:$base_pitch,
    I32:$x,
    I32:$y,
    I32Attr:$elem_size_in_bits,
    I32Attr:$tile_width,
    I32Attr:$tile_height,
    I32Attr:$v_blocks,
    DefaultValuedAttr<XeVM_LoadCacheControlAttr, "::imex::xevm::LoadCacheControl::DEFAULT">:$l1_cache_control,
    DefaultValuedAttr<XeVM_LoadCacheControlAttr, "::imex::xevm::LoadCacheControl::DEFAULT">:$l3_cache_control
  )> {

  let summary = "2D block prefetch";

  let description = [{
    The `xevm.blockprefetch2d` operation prefetches a two dimensional tile
    from a larger matrix residing in memory. The parameters are:
      $ptr - the base address of the matrix containing the tile to prefetch
      $base_width, $base_height, $base_pitch - the shape of the matrix
    $x, $y, $tile_width, $tile_height - the starting offsets and shape of tile to prefetch
    $elem_size_in_bits - the size in bits of the matrix element
      - 32 for f32, bf32
      - 16 for f16, int16, bf16
      - 8 for int8, int4, int2
    $v_blocks - number of tiles to prefetch
    $cache_control - an enumerator that sets the L1 and L3 cache behaviour

    Notes:
      - pitch is the physical stride between the first columns of the current row and the subsequent row,
        this may include (possibly implicit) padding, alignment, or other factors.
      - coordinate is provided in elements, while width and pitch are provided in bytes.

    Example:
    ```mlir
      xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, l1_cache_control=UC, l3_cache_control=UC} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
    ```
  }];

  let assemblyFormat = [{
    operands ` ` `{` `elem_size_in_bits` `=` $elem_size_in_bits `,` `tile_width` `=` $tile_width `,`
      `tile_height` `=` $tile_height `,` `v_blocks` `=` $v_blocks `,`  `l1_cache_control` `=` $l1_cache_control `,`
      `l3_cache_control` `=` $l3_cache_control `}`
      attr-dict `:` `(` type(operands) `)`
  }];

  let hasVerifier = 1;
}

def XeVM_MatrixElemType : AnyTypeOf<[AnyI8, AnyI16, AnyI32, F32, F16, BF16]>;

/// Enum attribute of the different precision types.
def XeVM_PrecisionTypeAttr : I32EnumAttr<"PrecisionType",
  "XeVM precision type",
  [
    I32EnumAttrCase<"UNUSED", 0,  "unused">,
    I32EnumAttrCase<"U8",     1,  "u8">,
    I32EnumAttrCase<"U4",     2,  "u4">,
    I32EnumAttrCase<"U2",     3,  "u2">,
    I32EnumAttrCase<"S8",     4,  "i8">,
    I32EnumAttrCase<"S4",     5,  "i4">,
    I32EnumAttrCase<"S2",     6,  "i2">,
    I32EnumAttrCase<"BF8",    7,  "bf8">,
    I32EnumAttrCase<"TF32",   8,  "tf32">,
    I32EnumAttrCase<"BF16",   9,  "bf16">,
    I32EnumAttrCase<"FP16",   10, "f16">
  ]> {
  let cppNamespace = "::imex::xevm";
}

def XeVM_DpasOp : XeVM_Op<"dpas">,
  Results<(outs FixedVectorOfRankAndType<[1], [XeVM_MatrixElemType]>:$d)>,
  Arguments<(ins
    FixedVectorOfRankAndType<[1], [XeVM_MatrixElemType]>:$c,
    FixedVectorOfRankAndType<[1], [XeVM_MatrixElemType]>:$a,
    FixedVectorOfRankAndType<[1], [XeVM_MatrixElemType]>:$b,
    XeVM_PrecisionTypeAttr:$pa,
    XeVM_PrecisionTypeAttr:$pb,
    I32Attr:$rc
  )> {

  let summary = "Matrix multiply-add";

  let description = [{
    The `xevm.dpas` operation is a matrix multiplication plus accumulation:

      D = C + A x B

      where the A, B, C input matrices and the result D have shapes:
        D : MxN
        C : MxN
        A : MxK
        B : KxN

        Shape restrictions:
        M : must be 1, 2, 4, or 8
        N : fixed execution size, must be 16
        K : systolic_depth * OPS_PER_CHAN
            OPS_PER_CHAN
              1 : for TF32
              2 : for 16-bit precision(BF, HF)
              4 : for 8-bit precision (FP8, UB, B)
              8 : for less-then 8 bit precision (U4/S4, U2/S2).

            If systolic_depth is 8, K would be 8, 16, 32, or 64 (based on OPS_PER_CHAN).
    $a, $b, $c, $d - matrix A, B, C, D, respectively
    $pa, $pb - precision of matrix A and B resepectively
    $rc - repeat count

    Further restrictions as well as more details can be found here:
    https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html

    Example:
    ```mlir
      %c_result = xevm.dpas %c, %a, %b {pa = f16, pb = f16, rc = 8} : (vector<8xf32>, vector<8xi16>, vector<8xi32>) -> vector<8xf32>
    ```
  }];

  let assemblyFormat = [{
    operands ` ` `{` `pa` `=` $pa `,` `pb` `=` $pb `,` `rc` `=` $rc `}` attr-dict `:` functional-type(operands, results)
  }];

  // let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// XeVM target attribute.
//===----------------------------------------------------------------------===//

def XeVM_TargetAttr : XeVM_Attr<"XeVMTarget", "target"> {
  let description = [{
    GPU target attribute for controlling compilation of Intel GPU targets. All
    parameters decay into default values if not present.

    Examples:

    1. Target with default values.
    ```
      gpu.module @mymodule [#xevm.target] attributes {...} {
        ...
      }
    ```
  }];
  let parameters = (ins
    DefaultValuedParameter<"int", "2", "Optimization level to apply.">:$O,
    StringRefParameter<"Target triple.", "\"spirv64-unknown-unknown\"">:$triple,
    StringRefParameter<"Target chip.", "\"pvc\"">:$chip,
    OptionalParameter<"::mlir::DictionaryAttr", "Target specific flags.">:$flags,
    OptionalParameter<"::mlir::ArrayAttr", "Files to link to the LLVM module.">:$linkFiles
  );
  let assemblyFormat = [{
    (`<` struct($O, $triple, $chip, $flags, $linkFiles)^ `>`)?
  }];
  let builders = [
    AttrBuilder<(ins CArg<"int", "2">:$optLevel,
                     CArg<"::llvm::StringRef", "\"spirv64-unknown-unknown\"">:$triple,
                     CArg<"::llvm::StringRef", "\"pvc\"">:$chip,
                     CArg<"::mlir::DictionaryAttr", "nullptr">:$targetFlags,
                     CArg<"::mlir::ArrayAttr", "nullptr">:$linkFiles), [{
      return Base::get($_ctxt, optLevel, triple, chip, targetFlags, linkFiles);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
}

#endif // XEVMIR_OPS
