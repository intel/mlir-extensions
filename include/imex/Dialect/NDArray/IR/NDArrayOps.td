//===- NDArrayOps.h - NDArray dialect  --------------------*- tablegen -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines basic types and operations of the NDArray dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _NDARRAY_OPS_TD_INCLUDED_
#define _NDARRAY_OPS_TD_INCLUDED_

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
// include "mlir/Interfaces/ShapedOpInterfaces.td"
// include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CastInterfaces.td"


def NDArray_Dialect : Dialect {
    let name = "ndarray";
    let cppNamespace = "::imex::ndarray";
    let summary = "A high-level dialect for parallel tensor operations";
    let description = [{
            The ndarray dialect describes high-lvel operations on arrays.
            It extends the tenssor and linalg dialects with operations which
            have array specific semantics, like mutating operations.

            The dialects differs from tensor dialects in MLIR because it
            it is meant to allow operations with in-place semantics and
            creating subviews which are guaranteed to be views.

            Generally the ndarray dialect is intended to provide high-level
            abstractions to allow compute-follows-data semantics. For this,
            the dialect operates on ranked tensors and attaches information
            about the location (device, team) of the tensor-data when
            arrays are created. These annotations are done through the
            mesh dialect.

            The functional scope of the dialect (together with tensor and
            linalg dialects) is the
            [array-API](https://data-apis.org/array-api/latest/index.html).

        }];

    //  We use the default parser/printer which handles registered attrs
    let useDefaultAttributePrinterParser = true;
}

def NDArray_EnvsAttr : AttrDef<NDArray_Dialect, "Envs"> {
  let mnemonic = "envs";
  let parameters = (ins ArrayRefParameter<"::mlir::Attribute">:$envs);
  let assemblyFormat = "`<` $envs `>`";
}

// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class NDArray_Op<string mnemonic, list<Trait> traits = []> :
    Op<NDArray_Dialect, mnemonic, traits>;


def CopyOp : NDArray_Op<"copy", [CopyOpInterface, SameOperandsAndResultShape, SameOperandsAndResultElementType]> {

  let description = [{
    Copies the data from the source to the new result array.

    Source and result are expected to have the same element type and shape.
    Otherwise, the result is undefined.
  }];

  let arguments = (ins Arg<AnyRankedTensor, "the array to copy from", [MemRead]>:$source);
  let results = (outs AnyRankedTensor:$target);

  let assemblyFormat = [{
    $source attr-dict `:` qualified(type($source)) `->` qualified(type($target))
  }];
}


def DeleteOp : NDArray_Op<"delete", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Explicitly delete an nd-array, freeing its memory";
  let description = [{
      Allow explicitly deleting the memory of an nd-array. It is assumed
      that the memory had been allocated by one of nd-array's creation functions.
      It must be the last use of the input array.
  }];

  let arguments = (ins AnyRankedTensor:$input);

  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input))
  }];
}


// Base class for ops with static/dynamic offset, sizes and strides
// attributes/arguments.
class NDArray_OpWithOffsetSizesAndStrides<string mnemonic,
                                          list<Trait> traits = []>
    : NDArray_Op<mnemonic, traits> {
  code extraBaseClassDeclaration = [{
    /// Returns the dynamic sizes for this subview operation if specified.
    ::mlir::Operation::operand_range getDynamicSizes() { return getSizes(); }

    /// Return the list of Range (i.e. offset, size, stride). Each
    /// Range entry contains either the dynamic value or a ConstantIndexOp
    /// constructed with `b` at location `loc`.
    ::mlir::SmallVector<::mlir::Range, 8> getOrCreateRanges(
        ::mlir::OpBuilder &b, ::mlir::Location loc) {
      return ::mlir::getOrCreateRanges(*this, b, loc);
    }
  }];
}


def SubviewOp : NDArray_OpWithOffsetSizesAndStrides<"subview", [
    Pure, AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
    OffsetSizeAndStrideOpInterface,
    ViewLikeOpInterface
  ]> {
  let summary = "array subview operation";
  let description = [{
    The "subview" operation converts a array type to another array type
    which represents a reduced-size view of the original array as specified by
    the operation's offsets, sizes and strides arguments.

    This operation is expected to eventually lower to memref.subview.
  }];

  let arguments = (ins
    AnyRankedTensor:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` qualified(type($source)) `to` qualified(type($result))
  }];

  let builders = [
    // Build a SubViewOp with mixed static and dynamic entries. Result type is inferred.
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$offsets,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$sizes,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>,

    // Build a SubViewOp with mixed static and dynamic entries and custom
    // result type. If the type passed is nullptr, it is inferred.
    OpBuilder<(ins
      "::mlir::RankedTensorType":$resultType,
      "::mlir::Value":$source,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$offsets,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$sizes,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>,

    // Build a SubViewOp with static entries. Result type is inferred.
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::ArrayRef<int64_t>":$offsets,
      "::mlir::ArrayRef<int64_t>":$sizes,
      "::mlir::ArrayRef<int64_t>":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>,

    // Build a SubViewOp with static entries and custom
    // result type. If the type passed is nullptr, it is inferred.
    OpBuilder<(ins
      "::mlir::RankedTensorType":$resultType,
      "::mlir::Value":$source,
      "::mlir::ArrayRef<int64_t>":$offsets,
      "::mlir::ArrayRef<int64_t>":$sizes,
      "::mlir::ArrayRef<int64_t>":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>,

    // Build a SubViewOp with dynamic entries.
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::ValueRange":$offsets,
      "::mlir::ValueRange":$sizes,
      "::mlir::ValueRange":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>,

    // Build a SubViewOp with dynamic entries and custom
    // result type. If the type passed is nullptr, it is inferred.
    OpBuilder<(ins
      "::mlir::RankedTensorType":$resultType,
      "::mlir::Value":$source,
      "::mlir::ValueRange":$offsets,
      "::mlir::ValueRange":$sizes,
      "::mlir::ValueRange":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    /// Returns the type of the base tensor operand.
    ::mlir::ShapedType getSourceType();

    /// The result of an subview is always a tensor.
    ::mlir::RankedTensorType getType() {
      return mlir::cast<::mlir::RankedTensorType>(getResult().getType());
    }

    /// Compute the rank-reduction mask that can be applied to map the source
    /// tensor type to the result tensor type by dropping unit dims.
    std::optional<llvm::SmallDenseSet<unsigned>>
    computeRankReductionMask() {
      return ::mlir::computeRankReductionMask(getSourceType().getShape(),
                                              getType().getShape());
    };

    /// A subview result type can be inferred, when it is not
    /// rank-reduced, from the source type and the static representation of
    /// offsets, sizes and strides. Special sentinels encode the dynamic case.
    static ::mlir::RankedTensorType inferResultType(
      ::mlir::RankedTensorType sourceType,
      ::mlir::ArrayRef<int64_t> staticOffsets,
      ::mlir::ArrayRef<int64_t> staticSizes,
      ::mlir::ArrayRef<int64_t> staticStrides);
    static ::mlir::RankedTensorType inferResultType(
      ::mlir::RankedTensorType sourceType,
      ::mlir::ArrayRef<::mlir::OpFoldResult> staticOffsets,
      ::mlir::ArrayRef<::mlir::OpFoldResult> staticSizes,
      ::mlir::ArrayRef<::mlir::OpFoldResult> staticStrides);

    /// A rank-reducing result type can be inferred from the desired result
    /// shape. Only the layout map is inferred.
    ///
    /// Note: The result shape cannot be inferred with just the result rank and
    /// and the desired sizes. In case there are more "ones" among the sizes
    /// than the difference in source/result rank, it is not clear which dims of
    /// size one should be dropped.
    static ::mlir::RankedTensorType inferRankReducedResultType(::mlir::ArrayRef<int64_t> resultShape,
                                                  ::mlir::RankedTensorType sourceType,
                                                  ::mlir::ArrayRef<int64_t> staticOffsets,
                                                  ::mlir::ArrayRef<int64_t> staticSizes,
                                                  ::mlir::ArrayRef<int64_t> staticStrides);
    static ::mlir::RankedTensorType inferRankReducedResultType(::mlir::ArrayRef<int64_t> resultShape,
                                                  ::mlir::RankedTensorType sourceType,
                                                  ::mlir::ArrayRef<::mlir::OpFoldResult> staticOffsets,
                                                  ::mlir::ArrayRef<::mlir::OpFoldResult> staticSizes,
                                                  ::mlir::ArrayRef<::mlir::OpFoldResult> staticStrides);

    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getSourceType().getRank();
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

    /// Return the dimensions of the source that are dropped in the
    /// result when the result is rank-reduced.
    ::llvm::SmallBitVector getDroppedDims();

    ::mlir::Value getViewSource() { return getSource(); }
  }];

  let hasCanonicalizer = 1;
}


def ExtractSliceOp : NDArray_OpWithOffsetSizesAndStrides<"extract_slice", [
    Pure,
    AttrSizedOperandSegments,
    OffsetSizeAndStrideOpInterface
  ]> {
  let summary = "extract slice operation";
  let description = [{
    The "extract_slice" operation extract a view from another array as
    specified by the operation's offsets, sizes and strides arguments.
    The returned array is not guaranteed to be a view of the source array.

    This operation is expectecd to eventually lower to tensor.extract_slice.
  }];

  let arguments = (ins
    AnyRankedTensor:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` qualified(type($source)) `to` qualified(type($result))
  }];

  let builders = [
    // Build a ExtractSliceOp with mixed static and dynamic entries. Result type is inferred.
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$offsets,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$sizes,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>,

    // Build a ExtractSliceOp with mixed static and dynamic entries and custom
    // result type.
    OpBuilder<(ins
      "::mlir::RankedTensorType":$resultType,
      "::mlir::Value":$source,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$offsets,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$sizes,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>,

    // Build a ExtractSliceOp with static entries. Result type is inferred.
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::ArrayRef<int64_t>":$offsets,
      "::mlir::ArrayRef<int64_t>":$sizes,
      "::mlir::ArrayRef<int64_t>":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>,

    // Build a ExtractSliceOp with static entries and custom
    // result type.
    OpBuilder<(ins
      "::mlir::RankedTensorType":$resultType,
      "::mlir::Value":$source,
      "::mlir::ArrayRef<int64_t>":$offsets,
      "::mlir::ArrayRef<int64_t>":$sizes,
      "::mlir::ArrayRef<int64_t>":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>,

    // Build a ExtractSliceOp with dynamic entries.
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::ValueRange":$offsets,
      "::mlir::ValueRange":$sizes,
      "::mlir::ValueRange":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>,

    // Build a ExtractSliceOp with dynamic entries and custom
    // result type.
    OpBuilder<(ins
      "::mlir::RankedTensorType":$resultType,
      "::mlir::Value":$source,
      "::mlir::ValueRange":$offsets,
      "::mlir::ValueRange":$sizes,
      "::mlir::ValueRange":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    /// Returns the source type
    ::mlir::RankedTensorType getSourceType() {
      auto sourceType = mlir::cast<::mlir::RankedTensorType>(getSource().getType());
      return sourceType;
    }

    /// The result is always a tensor.
    ::mlir::RankedTensorType getType() {
      return mlir::cast<::mlir::RankedTensorType>(getResult().getType());
    }

    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getSourceType().getRank();
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

    /// Return the dimensions of the source that are dropped in the
    /// result when the result is rank-reduced.
    ::llvm::SmallBitVector getDroppedDims();

    ::mlir::Value getViewSource() { return getSource(); }
  }];

  let hasCanonicalizer = 1;
}


def InsertSliceOp : NDArray_OpWithOffsetSizesAndStrides<"insert_slice", [
    AttrSizedOperandSegments,
    OffsetSizeAndStrideOpInterface,
  ]> {
  let summary = "Copy values from a array into a slice of another.";
  let description = [{
    Copy values from an array into a slice of another by updating the
    target array in-place.

    This operation is expected to eventually lower to memref.subview and memref.copy.
  }];

  let arguments = (ins
    AnyRankedTensor:$destination,
    AnyRankedTensor:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );

  let assemblyFormat = [{
    $source `into` $destination ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` qualified(type($source)) `into` qualified(type($destination))
  }];

  let builders = [
    // Build an InsertSliceOp with mixed static and dynamic entries.
    OpBuilder<(ins
      "::mlir::Value":$destination,
      "::mlir::Value":$source,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$offsets,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$sizes,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>,

    // Build an InsertSliceOp with dynamic entries.
    OpBuilder<(ins
      "::mlir::Value":$destination,
      "::mlir::Value":$source,
      "::mlir::ValueRange":$offsets,
      "::mlir::ValueRange":$sizes,
      "::mlir::ValueRange":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>,

    // Build an InsertSliceOp with static entries.
    OpBuilder<(ins
      "::mlir::Value":$destination,
      "::mlir::Value":$source,
      "::mlir::ArrayRef<int64_t>":$offsets,
      "::mlir::ArrayRef<int64_t>":$sizes,
      "::mlir::ArrayRef<int64_t>":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>,
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    /// Returns the source type
    mlir::RankedTensorType getSourceType() {
      auto sourceType = mlir::cast<mlir::RankedTensorType>(getSource().getType());
      return sourceType;
    }

    /// Returns the destination type
    mlir::RankedTensorType getDestinationType() {
      auto dstType = mlir::cast<mlir::RankedTensorType>(getDestination().getType());
      return dstType;
    }

    /// Returns the destination rank
    unsigned getDestinationRank();

    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getDestinationRank();
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 2; }

  }];

  let hasCanonicalizer = 1;
}


def ImmutableInsertSliceOp : NDArray_OpWithOffsetSizesAndStrides<"immutable_insert_slice", [
    Pure, AttrSizedOperandSegments,
    OffsetSizeAndStrideOpInterface,
    TypesMatchWith<"expected result type to match destination type",
                   "destination", "result", "$_self">
  ]> {
  let summary = "Copy values from a tensor into a slice of another.";
  let description = [{
    Copy values from an array into a slice of a copy of another.
    Return the modified copy.

    Returns a new value as an output.
    This operation is expected to lower to tensor.insert_slice.
  }];

  let arguments = (ins
    AnyRankedTensor:$destination,
    AnyRankedTensor:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source `into` $destination custom<DynamicIndexList>($offsets, $static_offsets) custom<DynamicIndexList>($sizes, $static_sizes) custom<DynamicIndexList>($strides, $static_strides) attr-dict `:` qualified(type($source)) `into` qualified(type($destination))
  }];

  let builders = [
    // Build an ImmutableInsertSliceOp with mixed static and dynamic entries.
    OpBuilder<(ins
      "::mlir::Value":$destination,
      "::mlir::Value":$source,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$offsets,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$sizes,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>,

    // Build an ImmutableInsertSliceOp with dynamic entries.
    OpBuilder<(ins
      "::mlir::Value":$destination,
      "::mlir::Value":$source,
      "::mlir::ValueRange":$offsets,
      "::mlir::ValueRange":$sizes,
      "::mlir::ValueRange":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>,

    // Build an ImmutableInsertSliceOp with static entries.
    OpBuilder<(ins
      "::mlir::Value":$destination,
      "::mlir::Value":$source,
      "::mlir::ArrayRef<int64_t>":$offsets,
      "::mlir::ArrayRef<int64_t>":$sizes,
      "::mlir::ArrayRef<int64_t>":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>,
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    /// Returns the source type
    mlir::RankedTensorType getSourceType() {
      auto sourceType =
          mlir::dyn_cast<mlir::RankedTensorType>(getSource().getType());
      return sourceType;
    }

    /// Returns the destination type
    mlir::RankedTensorType getDestinationType() {
      auto dstType = mlir::cast<mlir::RankedTensorType>(getDestination().getType());
      return dstType;
    }

    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getSourceType().getRank();
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 2; }
  }];

  let hasCanonicalizer = 1;
}


def LinSpaceOp : NDArray_Op<"linspace", [Pure]> {
  let summary = "Returns evenly spaced numbers over a specified interval.";
  let description = [{
      Number of of generated values is either num or num+1 depending on whether endpoint is True or False, respectively.
      See Array API.
  }];

  let arguments = (ins AnyType:$start, AnyType:$stop, AnyType:$num, UnitAttr:$endpoint);
  let results = (outs AnyRankedTensor);

  let assemblyFormat = [{
    $start $stop $num (`true` $endpoint^):(`false`)? attr-dict `:` `(` type(operands) `)` `->` qualified(type(results))
  }];

  let builders = [
      // auto-deduce return type
      OpBuilder<(ins "::mlir::Value":$start, "::mlir::Value":$stop, "::mlir::Value":$num,
                      CArg<"bool", "true">:$endpoint, CArg<"::imex::ndarray::DType", "imex::ndarray::F64">:$dtype,
                      CArg<"::mlir::ArrayRef<::mlir::Attribute>", "{}">:$environments), [{
          auto dt = toMLIR($_builder, dtype);
          build($_builder, $_state,
                ::mlir::RankedTensorType::get(getShapeFromValues(num), dt),
                start, stop, num, endpoint);
      }]>,
  ];

  let hasCanonicalizer = 1;
}


def CreateOp : NDArray_Op<"create", [Pure, AttrSizedOperandSegments]> {
  let summary = "Returns a new tensor having a specified shape and type and optionally filled with a value.";

  let arguments = (ins Variadic<Index>:$shape, Optional<AnyType>:$value);
  // result is a ndarray
  let results = (outs AnyRankedTensor);

  let assemblyFormat = [{
    $shape oilist(`value` $value) attr-dict `:` `(` type(operands) `)` `->` qualified(type(results))
  }];

  let builders = [
      // auto-deduce return type
      OpBuilder<(ins "::mlir::ValueRange":$shape, "::imex::ndarray::DType":$dtype, "::mlir::Value":$value,
                      CArg<"::mlir::ArrayRef<::mlir::Attribute>", "{}">:$environments), [{
          auto dt = toMLIR($_builder, dtype);
          build($_builder, $_state,
                ::mlir::RankedTensorType::get(getShapeFromValues(shape), dt),
                shape, value);
      }]>,
  ];

  let hasCanonicalizer = 1;
}


def ReshapeOp : NDArray_Op<"reshape", []> {
  let summary = "Reshapes an array without changing its data.";
  let description = [{
      Reshapes an array without changing its data. Memory is re-used as requested.
      See Array API.
  }];

  let arguments = (ins AnyType:$source, Variadic<Index>:$shape, OptionalAttr<I1Attr>:$copy);
  let results = (outs AnyType);

  let assemblyFormat = [{
    $source $shape attr-dict `:` qualified(type($source)) `->` qualified(type(results))
  }];
}


def CastElemTypeOp: NDArray_Op<"cast_elemtype", [Pure]> {
    let summary = "Cast array from one element type to another";

    let arguments = (ins AnyRankedTensor:$input, OptionalAttr<I1Attr>:$copy);
    let results = (outs AnyRankedTensor:$output);

    let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `to` qualified(type($output))";

    let hasCanonicalizer = 1;
}

#endif // _NDARRAY_OPS_TD_INCLUDED_
