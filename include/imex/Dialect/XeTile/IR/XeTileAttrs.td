//===------------ XeTileAttr.td - XeTile dialect  -------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines custom attributes used by XeTile dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _XETILE_ATTR_DEF_TD_INCLUDED_
#define _XETILE_ATTR_DEF_TD_INCLUDED_

include "mlir/IR/AttrTypeBase.td"
include "imex/Dialect/XeTile/IR/XeTileDialect.td"
include "mlir/IR/EnumAttr.td"

class XeTile_Attr<string name, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<XeTile_Dialect, name, traits> {
    let mnemonic = attrMnemonic;
}

def XeTile_SubGroupMapAttr : XeTile_Attr<"SubGroupMap", "sg_map"> {
  let parameters = (ins
    "mlir::DenseI32ArrayAttr":$wi_layout,
    "mlir::DenseI32ArrayAttr":$wi_data
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = true;
  let builders = [
    AttrBuilder<(ins  "llvm::ArrayRef<int32_t>":$wi_layout,
                      "llvm::ArrayRef<int32_t>":$wi_data),
    [{
        return $_get($_ctxt, mlir::DenseI32ArrayAttr::get($_ctxt, wi_layout),
                            mlir::DenseI32ArrayAttr::get($_ctxt, wi_data));
    }]>
  ];
}

def XeTile_WorkGroupMapAttr : XeTile_Attr<"WorkGroupMap", "wg_map"> {
  let parameters = (ins
    "mlir::DenseI32ArrayAttr":$sg_layout,
    "mlir::DenseI32ArrayAttr":$sg_data
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = true;
  let builders = [
    AttrBuilder<(ins "llvm::ArrayRef<int32_t>":$sg_layout,
                      "llvm::ArrayRef<int32_t>":$sg_data),
    [{
        return $_get($_ctxt, mlir::DenseI32ArrayAttr::get($_ctxt, sg_layout),
                            mlir::DenseI32ArrayAttr::get($_ctxt, sg_data));
    }]>
  ];
}

def XeTile_TileAttr : XeTile_Attr<"XeTile", "tile_attr"> {
  let parameters = (ins
    OptionalParameter<"xetile::SubGroupMapAttr">:$sg_map,
    OptionalParameter<"xetile::WorkGroupMapAttr">:$wg_map,
    DefaultValuedParameter<"mlir::DenseI32ArrayAttr", "mlir::DenseI32ArrayAttr::get($_ctxt, {1, 0})">:$order,
    OptionalParameter<"mlir::Attribute">:$memory_space,
    OptionalParameter<"mlir::BoolAttr">:$scattered
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = true;
  let builders = [
    AttrBuilder<(ins CArg<"llvm::ArrayRef<int32_t>", "{1, 0}">:$order,
                     CArg<"int", "0">:$memory_space, CArg<"bool", "false">:$scattered),
    [{
        mlir::Type intType = mlir::IntegerType::get($_ctxt, 32);
        mlir::BoolAttr scatteredAttr = mlir::BoolAttr::get($_ctxt, scattered);
        return $_get($_ctxt, xetile::SubGroupMapAttr(), xetile::WorkGroupMapAttr(),
                      mlir::DenseI32ArrayAttr::get($_ctxt, order),
                      mlir::IntegerAttr::get(intType, memory_space), scatteredAttr);
    }]>,
    AttrBuilder<(ins CArg<"xetile::SubGroupMapAttr", "{}">:$sg_map,
                      CArg<"xetile::WorkGroupMapAttr", "{}">:$wg_map,
                      CArg<"llvm::ArrayRef<int32_t>", "{1, 0}">:$order,
                      CArg<"int", "0">:$memory_space, CArg<"bool", "false">:$scattered),
    [{
        mlir::Type intType = mlir::IntegerType::get($_ctxt, 32);
        mlir::BoolAttr scatteredAttr = mlir::BoolAttr::get($_ctxt, scattered);
        return $_get($_ctxt, sg_map, wg_map, mlir::DenseI32ArrayAttr::get($_ctxt, order),
                      mlir::IntegerAttr::get(intType, memory_space), scatteredAttr);
    }]>
  ];
}

// RMW kind attribute
def ATOMIC_RMW_KIND_ADDF    : I64EnumAttrCase<"addf", 0>;
def ATOMIC_RMW_KIND_ADDI    : I64EnumAttrCase<"addi", 1>;
def ATOMIC_RMW_KIND_ASSIGN  : I64EnumAttrCase<"assign", 2>;
def ATOMIC_RMW_KIND_MAXF    : I64EnumAttrCase<"maxf", 3>;
def ATOMIC_RMW_KIND_MAXS    : I64EnumAttrCase<"maxs", 4>;
def ATOMIC_RMW_KIND_MAXU    : I64EnumAttrCase<"maxu", 5>;
def ATOMIC_RMW_KIND_MINF    : I64EnumAttrCase<"minf", 6>;
def ATOMIC_RMW_KIND_MINS    : I64EnumAttrCase<"mins", 7>;
def ATOMIC_RMW_KIND_MINU    : I64EnumAttrCase<"minu", 8>;
def ATOMIC_RMW_KIND_MULF    : I64EnumAttrCase<"mulf", 9>;
def ATOMIC_RMW_KIND_MULI    : I64EnumAttrCase<"muli", 10>;
def ATOMIC_RMW_KIND_ORI     : I64EnumAttrCase<"ori", 11>;
def ATOMIC_RMW_KIND_ANDI    : I64EnumAttrCase<"andi", 12>;

def XeTile_AtomicRMWKindAttr : I64EnumAttr<
    "AtomicRMWKind", "",
    [ATOMIC_RMW_KIND_ADDF, ATOMIC_RMW_KIND_ADDI, ATOMIC_RMW_KIND_ASSIGN,
     ATOMIC_RMW_KIND_MAXF, ATOMIC_RMW_KIND_MAXS, ATOMIC_RMW_KIND_MAXU,
     ATOMIC_RMW_KIND_MINF, ATOMIC_RMW_KIND_MINS, ATOMIC_RMW_KIND_MINU,
     ATOMIC_RMW_KIND_MULF, ATOMIC_RMW_KIND_MULI, ATOMIC_RMW_KIND_ORI,
     ATOMIC_RMW_KIND_ANDI]> {
  let cppNamespace = "::imex::xetile";
}

//TODO: !!!This is target specific information, cache attributes have to be passed transparently
// as custom arguments and handled properly on XeGPU side
//===----------------------------------------------------------------------===//
// XeTile Cache Enums.
//===----------------------------------------------------------------------===//
def XeTile_CachePolicyCached:        I32EnumAttrCase<"CACHED", 0, "cached">;                    // valid for read and write
def XeTile_CachePolicyUncached:      I32EnumAttrCase<"UNCACHED", 1, "uncached">;                // valid for read and write
def XeTile_CachePolicyStreaming:     I32EnumAttrCase<"STREAMING", 2, "streaming">;              // valid for read only
def XeTile_CachePolicyInvalid:       I32EnumAttrCase<"READ_INVALIDATE", 3, "read_invalidate">;  // valid for read only
def XeTile_CachePolicyWriteBack:     I32EnumAttrCase<"WRITE_BACK", 4, "write_back">;            // valid for write only
def XeTile_CachePolicyWriteThrough:  I32EnumAttrCase<"WRITE_THROUGH", 5, "write_through">;      // valid for write only

def XeTile_CachePolicyEnums : I32EnumAttr<"CachePolicy", "Cache policy",
  [XeTile_CachePolicyCached, XeTile_CachePolicyUncached,
   XeTile_CachePolicyStreaming, XeTile_CachePolicyInvalid,
   XeTile_CachePolicyWriteBack, XeTile_CachePolicyWriteThrough]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::imex::xetile";
}

def XeTile_CacheHintAttr
  : EnumAttr<XeTile_Dialect, XeTile_CachePolicyEnums, "cache_hint"> {
    let summary = [{Describe the cache settings for prefetch/load/store operators}];
    let assemblyFormat = "`<` $value `>`";
}


#endif // _XETILE_ATTR_DEF_TD_INCLUDED_
