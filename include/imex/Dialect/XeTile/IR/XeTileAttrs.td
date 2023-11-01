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

class XeTile_Attr<string name, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<XeTile_Dialect, name, traits> {
    let mnemonic = attrMnemonic;
}

def XeTile_SubGroupMapAttr : XeTile_Attr<"SubGroupMap", "sg_map"> {
  let parameters = (ins
    OptionalParameter<"mlir::DenseI32ArrayAttr">:$mma_block_size,
    "mlir::DenseI32ArrayAttr":$wi_layout,
    "mlir::DenseI32ArrayAttr":$wi_data
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = true;
  let builders = [
    AttrBuilder<(ins "llvm::ArrayRef<int32_t>":$mma_block_size,
                      "llvm::ArrayRef<int32_t>":$wi_layout,
                      "llvm::ArrayRef<int32_t>":$wi_data),
    [{
        return $_get($_ctxt, mlir::DenseI32ArrayAttr::get($_ctxt, mma_block_size),
                            mlir::DenseI32ArrayAttr::get($_ctxt, wi_layout),
                            mlir::DenseI32ArrayAttr::get($_ctxt, wi_data));
    }]>,
    AttrBuilder<(ins "llvm::ArrayRef<int32_t>":$wi_layout,
                      "llvm::ArrayRef<int32_t>":$wi_data),
    [{
        return $_get($_ctxt, mlir::DenseI32ArrayAttr(), mlir::DenseI32ArrayAttr::get($_ctxt, wi_layout),
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

def XeTile_XeMapAttr : XeTile_Attr<"XeMap", "xe_map"> {
  let parameters = (ins
    XeTile_WorkGroupMapAttr:$wg,
    XeTile_SubGroupMapAttr:$sg
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let builders = [
    AttrBuilder<(ins "llvm::ArrayRef<int32_t>":$mma_block_size,
                      "llvm::ArrayRef<int32_t>":$wi_layout,
                      "llvm::ArrayRef<int32_t>":$wi_data,
                      "llvm::ArrayRef<int32_t>":$sg_layout,
                      "llvm::ArrayRef<int32_t>":$sg_data),
    [{
        return $_get($_ctxt, WorkGroupMapAttr::get($_ctxt, sg_layout, sg_data),
          SubGroupMapAttr::get($_ctxt, mma_block_size, wi_layout, wi_data)) ;
    }]>,
    AttrBuilder<(ins  "llvm::ArrayRef<int32_t>":$wi_layout,
                      "llvm::ArrayRef<int32_t>":$wi_data,
                      "llvm::ArrayRef<int32_t>":$sg_layout,
                      "llvm::ArrayRef<int32_t>":$sg_data),
    [{
        return $_get($_ctxt, WorkGroupMapAttr::get($_ctxt, sg_layout, sg_data),
          SubGroupMapAttr::get($_ctxt, wi_layout, wi_data)) ;
    }]>
  ];
}

#endif // _XETILE_ATTR_DEF_TD_INCLUDED_
