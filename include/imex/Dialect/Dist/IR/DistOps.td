//===- DistOps.td - Dist dialect  --------------------------*- tablegen -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines basic operations of the Dist dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _Dist_OPS_TD_INCLUDED_
#define _Dist_OPS_TD_INCLUDED_

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Dialect/Mesh/IR/MeshBase.td"

// Provide a definition of the 'Dist' dialect in the ODS framework so that we
// can define our operations.
def Dist_Dialect : Dialect {
  // The namespace of our dialect
  let name = "dist";

  // A short one-line summary of our dialect.
  let summary = "A high-level dialect for distributing NDArray operations";

    // A longer description of our dialect.
  let description = [{
    This dialect provides basic features to allow automatic
    partitioning and distribution of NDArrays. The dialect assumes SPMD execution
    model. More specifically, each execution unit (or process) executes the same
    program but locally owns only a part of the globally distributed data. There is
    no central entity which partitions data or assigns work to workers.

    The Dist dialect is related to the NDArray and DistRuntime dialect. It is
    expected that the Dist dialect will eventually get lowered to NDArray and
    DistRuntime.
  }];

  let dependentDialects = [
    "::imex::ndarray::NDArrayDialect"
  ];

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "::imex::dist";
  // let useDefaultTypePrinterParser = true;
  let useDefaultAttributePrinterParser = true;
  let useDefaultTypePrinterParser = 1;
}


class Dist_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Dist_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

def DistSliceAttr : Dist_Attr<"DistSlice", "dist_slice"> {
  let summary = "List of offsets, sizes and strides for a number of slices.";
  let parameters = (ins
    ArrayRefParameter<"DenseI64ArrayAttr">:$slcOffsets,
    ArrayRefParameter<"DenseI64ArrayAttr">:$slcSizes,
    ArrayRefParameter<"DenseI64ArrayAttr">:$slcStrides
    // OptionalArrayRefParameter<"::mlir::DenseI64ArrayAttr">:$baseSizes
  );
  let assemblyFormat = [{
    `<` `[` $slcOffsets `]`  `[` $slcSizes `]` `[` $slcStrides `]` `>`
  }];
}

def DistEnvAttr
    : Dist_Attr<"DistEnv", "dist_env"> {
  let summary = "Environment for indicating that a NDArray is distributed";
  let description = [{
    The environment attribute `DistEnvAttr` can be attached to NDArrays.
    It carries the information required to describe the partitioning of the global NDArray.

    As an example, let's assume we want to equally distribute `ndarray.ndarray<33xi64>` across a team of 4. The third member of the team would attach label the type as a distributed array by attaching the following `DistEnvAttr`:

    `ndarray.ndarray<44xi64, #dist.dist_env<team = 37416 loffs = 22 lparts = 0,11,0>>`

    This defines the following:

    * the array has global size `44`
    * the array is distributed across team `37416`
    * the local data starts at global index `22`
    * the local part is of size `11`
    * the size of the right and left halos is `0`

    Notice that `lparts` encodes the shapes of 3 parts that are held locally:

    1. left halo
    2. locally owned data
    3. right halo

    All parts are of type `ndarray.NDArray`. Halos parts are copies of data owned by
    remote team members. Parts always represent pieces of the global array resulting
    from block-partitioning, i.e. they represent a contiguous block of the global
    index space. Furthermore, the concatenation of left halo, local data and right
    halo also represents a contiguous subset of the global index space.

    At this point, arrays are split only in the first dimension. A more general
    scheme can be added once required. However, when more than one dimension is cut
    it requires more than two halo parts and 'left' and 'right' are no longer
    sufficient to describe their position relative to the locally owned data.

    Notice that any part can be empty - even the locally owned part. For example: a
    subview of a global array might not intersect with the locally owned part.

    Parts and offsets are omitted (only) for 0d arrays:
    `ndarray.ndarray<f64, #dist.dist_env<team = 1>`

    The local offset represents offsets in all dimensions, so in principle allows
    partitions across multiple dimensions. For each dimension, the offset is
    provided to the first part (in most cases that's the left halo).

    The offsets and sizes in `DistEnvAttr` can be static as in the example.
    Alternatively, they can be partially or fully dynamic - even if the global size
    is static. The above example with fully dynamic local offsets and sizes would become:

    `ndarray.ndarray<44xi64, #dist.dist_env<team = 37416 loffs = ? lparts = ?,?,?>>`

    There is no placeholder for unknown teams.

    The distribution metadata generalizes to arrays of arbitrary dimensions. Here is
    an example of a distributed 2d array type:

    `ndarray.ndarray<44x55xi64, #dist.dist_env<team = 1 loffs = 22,0 lparts = 0x0,11x55,0x0>>`

    To indicate that the array is distributed across devices/GPUs, an additional
    environment gets attached to `ndarray.ndarray`. The additional environment
    defines on which device/GPU the local data should be stored. See NDArray spec
    for details about GPU support.

    As an example, consider distributing an array across two GPUs in the same computer. The types could look like this

    * team member 0:
    `ndarray.ndarray<22xi64, #dist.dist_env<team = 1 loffs = 22 lparts = 0,22,0>, #region.gpu_env<device = "sycl:gpu:0">>`
    * team member 1:
    `ndarray.ndarray<22xi64, #dist.dist_env<team = 1 loffs = 22 lparts = 0,22,0>, #region.gpu_env<device = "sycl:gpu:1">>`
  }];

  let parameters = (ins "::mlir::Attribute":$team,
                        ArrayRefParameter<"int64_t">:$lOffsets,
                        "::mlir::SmallVector<::mlir::SmallVector<int64_t>>":$parts_shapes);

  let assemblyFormat = "`<` custom<DistEnv>($team, $lOffsets, $parts_shapes) `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::Attribute":$team,
                                        "::llvm::ArrayRef<int64_t>":$lOffsets,
                                        "::mlir::SmallVector<::mlir::SmallVector<int64_t>>":$partsShapes)>,
    AttrBuilderWithInferredContext<(ins "::mlir::Attribute":$team, "int64_t":$rank)>
  ];

  let extraClassDeclaration = [{
    DistEnvAttr cloneWithDynOffsAndDims() const;
  }];
}


// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Dist_Op<string mnemonic, list<Trait> traits = []> :
    Op<Dist_Dialect, mnemonic, traits>;

def InitDistArrayOp : Dist_Op<"init_dist_array", [AttrSizedOperandSegments, Pure]> {
  let summary = "Instantiate a distributed array, binding  to distributed meta information.";
  let description = [{
    Accepted dynamic distributed meta information:
        - the local offset
        - local parts

    The team and resulting global shape is encoded in the result type.
  }];
  let arguments = (ins Variadic<Index>:$l_offset, Variadic<AnyType>:$parts);
  let results = (outs AnyType);

  let assemblyFormat = [{
    oilist(`l_offset` $l_offset | `parts` $parts) attr-dict `:` qualified(type(operands)) `to` qualified(type(results))
  }];

  let builders = [
    // auto-deduce return type
    OpBuilder<(ins "::mlir::Attribute":$team, "::mlir::ArrayRef<int64_t>":$g_shape, "::mlir::ValueRange":$l_offset,
                    "::mlir::ValueRange":$parts, "::mlir::ArrayRef<::mlir::Attribute>":$environments, "::mlir::ArrayRef<int64_t>":$s_Offs)>
];
}

def LocalOffsetsOfOp : Dist_Op<"local_offsets_of", [Pure]> {
  let summary = "Get local offsets of a distributed array.";
  let description = [{
    Returns `rank`-many values, one for each dimension of `$array`.
  }];
  let arguments = (ins AnyType:$array);
  let results = (outs Variadic<Index>:$l_offsets);
  let builders = [
    // autodeduce return type from from operands
    OpBuilder<(ins "::mlir::Value":$array), [{
      auto rank = mlir::cast<::imex::ndarray::NDArrayType>(array.getType()).getRank();
      auto IndexType = $_builder.getIndexType();
      ::imex::TypVec rt(rank, IndexType);
        build($_builder, $_state, ::mlir::TypeRange(rt), array);
      }]>,
  ];
}

def PartsOfOp : Dist_Op<"parts_of", [Pure]> {
  let summary = "Get local parts of a distributed array.";
  let description = [{
    Returns either one (0d array) or 3 parts
    (all other cases: left halo, locally owned data, right halo) as
    `ndarray.ndarray`. Returned arrays have the same rank as the input array.
  }];
  let arguments = (ins AnyType:$array);
  let results = (outs Variadic<AnyType>:$parts);
  let builders = [OpBuilder<(ins "::mlir::Value":$array)>];
  let hasVerifier = 1;
}

def DefaultPartitionOp : Dist_Op<"default_partition", [SameVariadicResultSize, Pure]> {
  let summary = "Compute the default shape and offsets of the local partition.";
  let description = [{
    All input and output shapes/offsets are vectors with same length.

    Arrays are cut along the first dimension and partitions are equally distributed
    among all members of the team. Member "i" of the team gets assigned to part "i".
    Odd elements in the cut dimension are equally distributed among the last team
    members. This guarantees that the sizes of local parts differ by at most one
    element in the cut dimension.

    For example, an array of size 8 will yield the local part sizes (2, 2, 2, 2) if
    the team has 4 members. For a team of 3 it will render (2, 3, 3).

    Other partition strategies could be added later.
  }];
  let arguments = (ins Index:$num_procs, Index:$p_rank, Variadic<Index>:$g_shape);
  let results = (outs Variadic<Index>:$l_offsets, Variadic<Index>:$l_shape);
  let builders = [
    // auto-deduce return type
    OpBuilder<(ins "::mlir::Value":$num_procs, "::mlir::Value":$prank, "::mlir::ValueRange":$gshape), [{
      auto IndexType = $_builder.getIndexType();
      ::imex::TypVec rt(gshape.size()*2, IndexType);
      build($_builder, $_state, ::mlir::TypeRange(rt), num_procs, prank, gshape);
    }]>,
  ];
}

def LocalTargetOfSliceOp : Dist_Op<"local_target_of_slice",
    [SameVariadicOperandSize, SameVariadicResultSize, Pure]> {
  let summary = "Compute local intersection of a distributed array with a slice.";
  let description = [{
    This operation computes the intersection of the local part of the array and the
    provided slice. The slice is provided as a triplet of offsets, sizes and strides
    (similar to a subview). While the slice refers to the global index space of the
    distributed array, the operation returns local offsets and sizes, relative to
    the local part (e.g. these are not global indices).

    All input and output shapes/offsets/strides are `$array.rank()`-long vectors.
  }];

  let arguments = (ins
      AnyType:$array,
      Variadic<Index>:$offsets,
      Variadic<Index>:$sizes,
      Variadic<Index>:$strides
  );
  let results = (outs Variadic<Index>:$t_offsets, Variadic<Index>:$t_sizes);

  let assemblyFormat = [{
    $array `[` $offsets `]``[` $sizes `]``[` $strides `]` attr-dict `:` qualified(type($array)) `to` qualified(type(results))
  }];

  let builders = [
      // auto-deduce return type
    OpBuilder<(ins "::mlir::Value":$array, "::mlir::ValueRange":$offsets, "::mlir::ValueRange":$sizes, "::mlir::ValueRange":$strides), [{
      auto IndexType = $_builder.getIndexType();
      ::imex::TypVec rt(offsets.size()*2, IndexType);
      build($_builder, $_state, ::mlir::TypeRange(rt), array, offsets, sizes, strides);
    }]>,
  ];
}

def LocalBoundingBoxOp : Dist_Op<"local_bounding_box", [AttrSizedOperandSegments, SameVariadicResultSize, Pure]> {
  let summary = "Compute (or extend) bounding box for data locally required by given view and target.";
  let description = [{
    The locally required view is the intersection of the given view and target.

    If an existing bounding box is provided, update the bounding box. The update strategy is determined by the `inner` attribute:

    * if `inner` is unset (default) return the convex hull of given bounding box and
      locally required view.
    * else return the intersection of given bounding box and locally required view.

    If no bounding box is provided (through `b_b_offsets` and `b_b_sizes`) return the offset and shape of the locally required view.

    The bounding box is returned as global offsets and shape.

    All input and output shapes/offsets/strides are vectors with same length.
  }];

  let arguments = (ins I1Attr:$inner,
                       Variadic<Index>:$offsets, Variadic<Index>:$sizes, Variadic<Index>:$strides,
                       Variadic<Index>:$target_offsets, Variadic<Index>:$target_sizes,
                       Variadic<Index>:$b_b_offsets, Variadic<Index>:$b_b_sizes);
  let results = (outs Variadic<Index>:$result_offsets, Variadic<Index>:$result_sizes);

  let assemblyFormat = [{
    $inner `[` $offsets `]``[` $sizes `]``[` $strides `]` `[` $target_offsets `]``[` $target_sizes `]` oilist(`bboffs` $b_b_offsets | `bb_sizes` $b_b_sizes) attr-dict `:` qualified(type(results))
  }];

  let builders = [
    // auto-deduce return type: same as input
    OpBuilder<(ins "bool":$inner, "::mlir::ValueRange":$offs, "::mlir::ValueRange":$sizes, "::mlir::ValueRange":$strides,
                    "::mlir::ValueRange":$toffs, "::mlir::ValueRange":$tsizes,
                    "::mlir::ValueRange":$bboffs, "::mlir::ValueRange":$bbsizes), [{
      size_t rank = offs.size();
      assert(sizes.size() == rank);
      ::imex::TypVec rt(2 * rank, $_builder.getIndexType());
      build($_builder, $_state, ::mlir::TypeRange(rt), inner, offs, sizes, strides, toffs, tsizes, bboffs, bbsizes);
    }]>,
  ];
}

def LocalCoreOp : Dist_Op<"local_core", [AttrSizedOperandSegments, SameVariadicResultSize, Pure]> {
  let summary = "Compute or update overlap of given core, locally owned data and locally required data.";
  let description = [{
    The locally required view is the intersection of the given slice and target.

    If no local core is provided, return the intersection of locally owned data and
    locally required data. Otherwise return the intersection of given core, locally
    owned data and locally required data.

    The intersection is returned as global offsets and shape.
  }];

  let arguments = (ins AnyType:$array,
                       Variadic<Index>:$targetOffsets, Variadic<Index>:$targetSizes,
                       Variadic<Index>:$sliceOffsets, Variadic<Index>:$sliceSizes, Variadic<Index>:$sliceStrides,
                       Variadic<Index>:$coreOffsets, Variadic<Index>:$coreSizes);
  let results = (outs Variadic<Index>:$resultOffsets, Variadic<Index>:$resultSizes);

  let assemblyFormat = [{
    $array oilist(`toffs` $targetOffsets | `tsizes` $targetSizes | `soffs` $sliceOffsets | `ssizes` $sliceSizes | `sstrides` $sliceStrides | `coffs` $coreOffsets | `csizes` $coreSizes) attr-dict `:` qualified(type($array)) `to` qualified(type(results))
  }];

  let builders = [
    // auto-deduce return type: same as input
    OpBuilder<(ins "::mlir::Value":$array,
                    "::mlir::ValueRange":$toffs, "::mlir::ValueRange":$tsizes,
                    "::mlir::ValueRange":$soffs, "::mlir::ValueRange":$ssizes ,"::mlir::ValueRange":$sstrides,
                    "::mlir::ValueRange":$coffs, "::mlir::ValueRange":$csizes), [{
      size_t rank = mlir::cast<::imex::ndarray::NDArrayType>(array.getType()).getRank();
      ::imex::TypVec rt(rank + rank, $_builder.getIndexType());
      build($_builder, $_state, ::mlir::TypeRange(rt), array, toffs, tsizes, soffs, ssizes, sstrides, coffs, csizes);
    }]>,
  ];
}

def RePartitionOp : Dist_Op<"repartition", [SameVariadicOperandSize, Pure]> {
  let summary = "Repartition an array so that each process holds the requested data locally.";
  let description = [{
    Creates a new NDArray by repartitioning the input array. It is assumed to be a
    collective call. All participating processes request which part of the global
    array they need. The halo parts of the returned array get filled with data that
    is owned by remote team members. The local data is not modified, the returned
    local part is a subview of the local part of the input.

    Target offset and target shape are optional arguments. If missing the operations
    returns a default-partitioned array.
  }];

  let arguments = (ins AnyType:$array,
                       Variadic<Index>:$target_offsets, Variadic<Index>:$target_sizes);
  let results = (outs AnyType);

  let assemblyFormat = [{
    $array oilist(`loffs` $target_offsets | `lsizes` $target_sizes) attr-dict `:` qualified(type(operands)) `to` qualified(type(results))
  }];

  let builders = [
    // auto-deduce return type: same as input
    OpBuilder<(ins "::mlir::Value":$array), [{
      build($_builder, $_state, array.getType(), array, {}, {});
    }]>,
  ];
}


// ============================================================================
// (Extended) operations from NDArray
// ============================================================================

def SubviewOp : Dist_Op<"subview", [AttrSizedOperandSegments, Pure]> {
  let summary = "Distributed extract slice operation";
  let description = [{
    The distributed subview operation is a shallow wrapper around NDArray.subview.
    It extends the ndarray.SubviewOp with optional target offsets and target sizes.
  }];

  let arguments = (ins AnyType:$source,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$strides,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$static_sizes,
                       DenseI64ArrayAttr:$static_strides,
                       Variadic<Index>:$target_offsets,
                       Variadic<Index>:$target_sizes
  );
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    oilist(`toffs` $target_offsets | `tsizes` $target_sizes)
    attr-dict `:` qualified(type($source)) `to` qualified(type($result))
  }];
}

def EWBinOp : Dist_Op<"ewbin", [Pure, SameVariadicOperandSize]> {
  let summary = "Distributed elementwise binary operation";
  let description = [{
    The distributed EWBinOp is a shallow wrapper around NDArray.ewbinop.
    It extends the ndarray.SubviewOp with optional core offsets, core sizes and target offsets.
  }];

  // ewbin takes 2 NDArrayType operands: lhs and rhs
  let arguments = (ins AnyAttr:$op, AnyType:$lhs, AnyType:$rhs,
                       Variadic<Index>:$coreOffsets, Variadic<Index>:$coreSizes,
                       Variadic<Index>:$targetOffsets);
  // result is a ndarray
  let results = (outs AnyType);
  let hasVerifier = 1;
}

def EWUnyOp : Dist_Op<"ewuny", [Pure, SameVariadicOperandSize]> {
  let summary = "Distributed elementwise unary operation";
  let description = [{
    The distributed EWUnyOp is a shallow wrapper around NDArray.ewunyop.
    It extends the ndarray.EWUnyOp with optional core offsets, core sizes and target offsets.
  }];

  // ewuny takes 1 operand (NDArrayType) and one attribute (unary operation)
  let arguments = (ins AnyAttr:$op, AnyType:$src,
                       Variadic<Index>:$coreOffsets, Variadic<Index>:$coreSizes,
                       Variadic<Index>:$targetOffsets);
  // result is a ndarray
  let results = (outs AnyType);
}

////////////////////////////////////////////////////
// shard

// common base classes for types in NDArray dialect
class Dist_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<Dist_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def Dist_ShardingConstraint : Dist_Type<"ShardingConstraint", "shardingconstraint"> {
  let summary = "sharding constraint";
}

def ShardOp : Dist_Op<"shard", [Pure,
    AllTypesMatch<["result", "src"]>
  ]> {
  let summary = "Annotate on how a tensor is sharded across a mesh.";
  let description = [{
    The mesh.shard operation is designed to specify and guide the sharding
    behavior of a tensor value across a mesh topology. This operation has one
    operand and two attributes:

    1. `input`: This operand represents the tensor value that needs to be
    annotated for sharding.

    2. `shard`: This attribute is type of `MeshSharding`, which is the core data
    structure to represent distribution of a tensor on a mesh.

    3. `annotate_for_users`: A unit attribute addressing the scenario when a
    tensor's sharding annotation differs based on its context of use (either as
    a result or an operand). If specified, the sharding pertains to specific
    users of the tensor value, indicating how it should be considered when used
    as an operand in subsequent operations. If not, the sharding applies to the
    operation that defines the tensor value.

    Example:
    ```
    func.func @only_result_annotated(%arg0 : tensor<4x8xf32>) -> () {
      %0 = mesh.shard %arg0 to <@mesh0, [[0]]> : tensor<4x8xf32>
      ...
    }

    func.func @only_operand_annotated(%arg0 : tensor<4x8xf32>) -> () {
      %0 = mesh.shard %arg0 to <@mesh0, [[0]]> annotate_for_users : tensor<4x8xf32>
      ...
    }

    // The first mesh.shard op applies to %arg0, the second mesh.shard op
    // applies for the operand of op0, the third mesh.shard op applies for the
    // operand of op2
    func.func @both_result_and_multi_operands_annotated(
        %arg0 : tensor<4x8xf32>) -> () {
      %0 = mesh.shard %arg0 to <@mesh0, [[0]]> : tensor<4x8xf32>
      %1 = mesh.shard %0 to <@mesh0, [[1]]> annotate_for_users : tensor<4x8xf32>
      %2 = mesh.shard %0 to <@mesh0, [[2]]> annotate_for_users : tensor<4x8xf32>
      "op0"(%1) : ...
      "op1"(%2) : ...
      ...
    }
    ```

    The following usages are undefined:
    ```
    func.func @annotate_on_same_result_with_different_sharding(
        %arg0 : tensor<4x8xf32>) -> () {
      %0 = mesh.shard %arg0 to <@mesh0, [[0]]> : tensor<4x8xf32>
      %1 = mesh.shard %0 to <@mesh0, [[1]]> : tensor<4x8xf32>
      ...
    }

    func.func @annotate_on_same_result_same_value_with_different_sharding(
        %arg0 : tensor<4x8xf32>) -> () {
      %0 = mesh.shard %arg0 to <@mesh0, [[0]]> : tensor<4x8xf32>
      %1 = mesh.shard %arg0 to <@mesh0, [[1]]> : tensor<4x8xf32>
      ...
    }

    func.func @annotate_on_same_operand_with_different_sharding(
        %arg0 : tensor<4x8xf32>) -> () {
      %0 = mesh.shard %arg0 to <@mesh0, [[0]]> annotate_for_users : tensor<4x8xf32>
      %1 = mesh.shard %0 to <@mesh0, [[1]]> annotate_for_users : tensor<4x8xf32>
      ...
    }

    func.func @result_annotated_after_operand(
        %arg0 : tensor<4x8xf32>) -> () {
      %0 = mesh.shard %arg0 to <@mesh0, [[0]]> annotate_for_users : tensor<4x8xf32>
      %1 = mesh.shard %0 to <@mesh0, [[1]]> : tensor<4x8xf32>
      ...
    }
    ```
  }];
  let arguments = (ins
    AnyRankedTensor:$src,
    AnyAttr:$shard,
    UnitAttr:$annotate_for_users,
    Optional<Dist_ShardingConstraint>:$constraint
  );
  let results = (outs
    AnyRankedTensor:$result
  );
  let assemblyFormat = [{
    $src `to` ($constraint^)? attr-dict `:` type($result)
  }];
}

def TargetOfSliceOp : Dist_Op<"target_of_slice",
    [Pure]> {
  let summary = "Compute intersections of a distributed array with a slice.";
  let description = [{
    This operation computes the intersection of the shards of the array and the
    provided slice. The slice is provided as a triplet of offsets, sizes and strides
    (similar to a subview).

    All input shapes/offsets/strides are `$array.rank()`-long vectors.
    The operation produces $array.rank()*split_axes.size() many values.
  }];

  let arguments = (ins
    AnyType:$array,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides,
    FlatSymbolRefAttr:$mesh,
    Mesh_MeshAxesArrayAttr:$split_axes
  );
  let results = (outs Variadic<Index>:$result);

  let assemblyFormat = [{
    $array attr-dict `:` qualified(type($array)) `to` qualified(type(results))
  }];
}

def BoundingBoxOp : Dist_Op<"bounding_box", [SameVariadicOperandSize, Pure]> {
  let summary = "Compute bounding box.";

  let arguments = (ins
    DistSliceAttr:$static_slices,
    Variadic<Dist_ShardingConstraint>:$subviewConstraints);
  let results = (outs Dist_ShardingConstraint:$constraint);

  let assemblyFormat = [{
    `[` $subviewConstraints `]` attr-dict `:` qualified(type(results))
  }];
}

def ExtendHaloForSliceOp : Dist_Op<"extend_halo_for_slice", [Pure]> {
  let summary = "Extend halo for slice.";
  let description = [{
    This operation computes the halo required to compute the slice of the array
    And extends provided halo if necessary.
    The halo is returned as a flattened array.
  }];

  let arguments = (ins
    DenseI64ArrayAttr:$static_shape,
    FlatSymbolRefAttr:$mesh,
    Mesh_MeshAxesArrayAttr:$split_axes,
    Variadic<I64>:$halo_sizes,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides,
    DenseI64ArrayAttr:$sharded_dims_offsets
  );
  let results = (outs Variadic<I64>:$result);

  let assemblyFormat = [{
    `[` $halo_sizes `]` attr-dict `:` qualified(type(results))
  }];
}


#endif // _Dist_OPS_TD_INCLUDED_
