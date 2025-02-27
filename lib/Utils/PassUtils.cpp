//===- PassUtils.h - Pass Utility Functions --------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utility functions for writing passes.
//
//===----------------------------------------------------------------------===//

#include <imex/Utils/PassUtils.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Pass/Pass.h>

namespace imex {

/// @return new index ::mlir::Value with given Value
::mlir::Value createIndex(const ::mlir::Location &loc,
                          ::mlir::OpBuilder &builder, int64_t val) {
  auto attr = builder.getIndexAttr(val);
  return builder.create<::mlir::arith::ConstantOp>(loc, attr);
}

::mlir::Value createIndexCast(const ::mlir::Location &loc,
                              ::mlir::OpBuilder &builder, ::mlir::Value val,
                              ::mlir::Type intTyp) {
  if (!intTyp)
    intTyp = builder.getIndexType();
  return val.getType() == intTyp
             ? val
             : builder.create<::mlir::arith::IndexCastOp>(loc, intTyp, val)
                   .getResult();
}

// cast int type to signed
::mlir::Value toSignedInt(const ::mlir::Location &loc,
                          ::mlir::OpBuilder &builder, ::mlir::Value val) {
  auto intTyp = val.getType();
  assert(intTyp.isIntOrIndex() && !intTyp.isIndex());
  if (intTyp.isUnsignedInteger()) {
    return builder
        .create<::mlir::UnrealizedConversionCastOp>(
            loc, builder.getIntegerType(intTyp.getIntOrFloatBitWidth()), val)
        .getResult(0);
  }
  return val;
}

// cast integer type to desired signedness type intTyp
::mlir::Value convertSignedness(const ::mlir::Location &loc,
                                ::mlir::OpBuilder &builder, ::mlir::Value val,
                                ::mlir::Type intTyp) {
  auto vTyp = val.getType();
  assert(vTyp.isIntOrIndex() && !vTyp.isIndex());
  if (vTyp == intTyp) {
    return val;
  }
  if (intTyp.getIntOrFloatBitWidth() == vTyp.getIntOrFloatBitWidth()) {
    return builder.create<::mlir::arith::BitcastOp>(loc, intTyp, val);
  }
  return builder.create<::mlir::UnrealizedConversionCastOp>(loc, intTyp, val)
      .getResult(0);
}

// cast between different scalar types
::mlir::Value createCast(const ::mlir::Location &loc,
                         ::mlir::OpBuilder &builder, ::mlir::Value val,
                         ::mlir::Type dTyp) {
  auto vTyp = val.getType();
  assert(vTyp.isIntOrIndexOrFloat() && dTyp.isIntOrIndexOrFloat());
  bool bitExtend = false;
  if (!vTyp.isIndex() && !dTyp.isIndex()) {
    bitExtend = vTyp.getIntOrFloatBitWidth() < dTyp.getIntOrFloatBitWidth();
  }
  if (vTyp == dTyp) {
    // nothing to do
    return val;
  } else if (vTyp.isIntOrIndex() && dTyp.isIntOrIndex()) {
    // int/index to int/index
    if ((vTyp.isIndex() || dTyp.isIndex())) {
      // all index cases
      return createIndexCast(loc, builder, val, dTyp);
    }
    // intermediate type if converting to uint, dtyp otherwise
    auto iTyp = (dTyp.isUnsignedInteger())
                    ? builder.getIntegerType(dTyp.getIntOrFloatBitWidth())
                    : dTyp;
    if (bitExtend) {
      // int to int
      val = toSignedInt(loc, builder, val);
      // extend bits
      if (vTyp.isUnsignedInteger()) {
        val = builder.createOrFold<::mlir::arith::ExtUIOp>(loc, iTyp, val);
      } else {
        val = builder.createOrFold<::mlir::arith::ExtSIOp>(loc, iTyp, val);
      }
    } else {
      // truncate bits
      val = convertSignedness(
          loc, builder, val,
          builder.getIntegerType(val.getType().getIntOrFloatBitWidth()));
      val = builder.create<::mlir::arith::TruncIOp>(loc, iTyp, val);
    }
    // cast return type to uint if needed
    return convertSignedness(loc, builder, val, dTyp);
  } else if (vTyp.isIntOrIndex()) {
    // int/index to float
    if (vTyp.isUnsignedInteger()) {
      // from uint
      return builder.createOrFold<::mlir::arith::UIToFPOp>(
          loc, dTyp, toSignedInt(loc, builder, val));
    } else {
      if (vTyp.isIndex()) {
        // from index
        val = createIndexCast(loc, builder, val, builder.getIntegerType(64));
      }
      // from int
      return builder.createOrFold<::mlir::arith::SIToFPOp>(loc, dTyp, val);
    }
  } else if (dTyp.isIntOrIndex()) {
    // float to int/index
    if (dTyp == builder.getIndexType()) {
      // to index
      val = builder.createOrFold<::mlir::arith::FPToSIOp>(
          loc, builder.getIntegerType(64), val);
      return createIndexCast(loc, builder, val, dTyp);
    } else {
      if (dTyp.isUnsignedInteger()) {
        // to uint
        val = builder.create<::mlir::arith::FPToUIOp>(
            loc, builder.getIntegerType(dTyp.getIntOrFloatBitWidth()), val);
      } else {
        // to int
        val = builder.createOrFold<::mlir::arith::FPToSIOp>(loc, dTyp, val);
      }
      // cast return type to uint if needed
      return convertSignedness(loc, builder, val, dTyp);
    }
  } else if (bitExtend) {
    // float to float, extend bits
    return builder.createOrFold<::mlir::arith::ExtFOp>(loc, dTyp, val);
  }
  // float to float, truncate bits
  assert(!(vTyp.isIntOrIndex() || vTyp.isIntOrIndex()) && !bitExtend);
  return builder.createOrFold<::mlir::arith::TruncFOp>(loc, dTyp, val);
}

/// @return array of static sizes form ValueRange: actual size if constant,
/// kDynamic if not
::mlir::SmallVector<int64_t>
getShapeFromValues(const ::mlir::ValueRange &sizes) {
  auto rank = sizes.size();
  ::mlir::SmallVector<int64_t> szVec(rank, ::mlir::ShapedType::kDynamic);
  for (size_t i = 0; i < rank; ++i) {
    if (auto cval = ::mlir::getConstantIntValue(sizes[i])) {
      szVec[i] = cval.value();
    }
  }
  return szVec;
}

/// @return number of elements for given shape if all sizes are constant,
/// kDynamic otherwise
int64_t getSizeFromValues(const ::mlir::ValueRange &sizes) {
  int64_t sz = 0;
  for (auto s : sizes) {
    if (auto cval = ::mlir::getConstantIntValue(s)) {
      sz *= cval.value();
    } else {
      return ::mlir::ShapedType::kDynamic;
    }
  }
  return sz;
}

/// get dyn-sized mlir::RankedTensorType for given size values and elType
::mlir::RankedTensorType getTensorType(::mlir::MLIRContext *ctxt,
                                       const ::mlir::ValueRange &sizes,
                                       ::mlir::Type elType) {
  auto shape = getShapeFromValues(sizes);
  return ::mlir::RankedTensorType::get(shape, elType); //, layout);
}

/// get dyn-sized mlir::RankedTensorType for given shape and elType
::mlir::RankedTensorType getTensorType(::mlir::MLIRContext *ctxt,
                                       ::mlir::ArrayRef<int64_t> shape,
                                       ::mlir::Type elType) {
  return ::mlir::RankedTensorType::get(shape, elType); //, layout);
}

/// get dyn-sized mlir::RankedTensorType for given rank and elType
::mlir::RankedTensorType getTensorType(::mlir::MLIRContext *ctxt, int64_t rank,
                                       ::mlir::Type elType) {
  return ::mlir::RankedTensorType::get(
      std::vector<int64_t>(rank, ::mlir::ShapedType::kDynamic),
      elType); //, layout);
}

// convert a range of values into their equivalent constant int or
// ::mlir::ShapedType::kDynamic
::mlir::SmallVector<int64_t> mkConstant(::mlir::ValueRange vals) {
  ::mlir::SmallVector<int64_t> gshp;
  for (auto v : vals) {
    auto cval = ::mlir::getConstantIntValue(v);
    gshp.emplace_back(cval.value_or(::mlir::ShapedType::kDynamic));
  }
  return gshp;
}

/// combine dynamic and static sizes (as used by SubviewOps) into a
/// single ValueRange (vector of values)
::imex::ValVec getMixedAsValues(const ::mlir::Location &loc,
                                ::mlir::OpBuilder &builder,
                                const ::mlir::ValueRange &dyns,
                                ::llvm::ArrayRef<int64_t> statics, bool asI64) {
  ::imex::ValVec out;
  auto dyn = dyns.begin();
  for (auto s : statics) {
    out.emplace_back(::mlir::ShapedType::isDynamic(s)
                         ? *(dyn++)
                         : (asI64 ? createInt(loc, builder, s)
                                  : createIndex(loc, builder, s)));
  }
  return out;
}

/// similar to mlir::decomposeMixedValues but converting const values tot
/// statics
void dispatchIndexValues(::mlir::OpBuilder &builder, ::mlir::Location loc,
                         const ::mlir::ValueRange &sizes,
                         ::mlir::SmallVectorImpl<::mlir::Value> &dynamicVec,
                         ::mlir::SmallVectorImpl<int64_t> &staticVec) {
  for (auto v : sizes) {
    if (auto cval = ::mlir::getConstantIntValue(v); cval && cval.value() == 1) {
      staticVec.emplace_back(cval.value());
    } else {
      dynamicVec.emplace_back(createIndexCast(loc, builder, v));
      staticVec.emplace_back(::mlir::ShapedType::kDynamic);
    }
  }
}

/// create an empty RankedTensor with given shape and elType
::mlir::Value createEmptyTensor(::mlir::OpBuilder &builder,
                                ::mlir::Location loc, ::mlir::Type elType,
                                const ::mlir::ValueRange &shp) {
  ::mlir::SmallVector<int64_t> staticSizes;
  ::mlir::SmallVector<mlir::Value> dynamicSizes;
  dispatchIndexValues(builder, loc, shp, dynamicSizes, staticSizes);
  return builder
      .create<::mlir::tensor::EmptyOp>(loc, staticSizes, elType, dynamicSizes)
      .getResult();
}

/// create an empty RankedTensor for given result tensor type and operand shapes
::mlir::Value createEmptyTensor(::mlir::OpBuilder &builder,
                                ::mlir::Location loc,
                                ::mlir::TensorType resType,
                                const ::mlir::ValueRange &operands) {

  ::imex::ValVec dynamicSizes(resType.getRank());
  ::mlir::SmallVector<int64_t> staticSizes(resType.getRank(), 0);

  // for each operand extract dispatch static and dynamic sizes
  // and define new shape as follows:
  // - dynamic always overwrites static sizes of other operands
  // - static sizes > 1 overwrite other static sizes
  // - static sizes == 1 remain only if all operands agree
  for (auto arg : operands) {
    auto shapedTy = mlir::cast<::mlir::ShapedType>(arg.getType());
    for (int i = 0; i < shapedTy.getRank(); i++) {
      if (shapedTy.isDynamicDim(i)) {
        auto dimOp = builder.create<::mlir::tensor::DimOp>(loc, arg, i);
        dynamicSizes[i] = createIndexCast(loc, builder, dimOp);
        staticSizes[i] = ::mlir::ShapedType::kDynamic;
      } else {
        auto v = shapedTy.getDimSize(i);
        if (staticSizes[i] != ::mlir::ShapedType::kDynamic &&
            (v > 1 || staticSizes[i] == 0)) {
          staticSizes[i] = v;
        }
      }
    }
  }

  // now build the range of dynamic sizes
  ::imex::ValVec filteredDims;
  for (auto value : dynamicSizes) {
    if (value) {
      filteredDims.push_back(value);
    }
  }

  // create empty tensor with dims as determined above
  return builder
      .create<::mlir::tensor::EmptyOp>(loc, staticSizes,
                                       resType.getElementType(), filteredDims)
      .getResult();
}

/// get dyn-sized mlir::RankedTensorType for given rank and elType
/// if strided==true make it a strided layout
::mlir::MemRefType getMemRefType(::mlir::MLIRContext *ctxt, int64_t rank,
                                 ::mlir::Type elType, bool strided) {
  static auto kDynamic = ::mlir::ShapedType::kDynamic;
  auto layout = ::mlir::StridedLayoutAttr::get(
      ctxt, kDynamic, ::mlir::SmallVector<int64_t>(rank, kDynamic));
  return ::mlir::MemRefType::get(std::vector<int64_t>(rank, kDynamic), elType,
                                 strided ? layout
                                         : ::mlir::StridedLayoutAttr{});
}

/// get mixed statically and dynamically sized mlir::MemRefType for given sizes
/// and elType if strided==true make it a strided layout
::mlir::MemRefType getMemRefType(::mlir::MLIRContext *ctxt,
                                 ::mlir::ArrayRef<int64_t> sizes,
                                 ::mlir::Type elType, bool strided) {
  auto rank = sizes.size();
  static auto kDynamic = ::mlir::ShapedType::kDynamic;
  auto layout = ::mlir::StridedLayoutAttr::get(
      ctxt, kDynamic, ::mlir::SmallVector<int64_t>(rank, kDynamic));

  return ::mlir::MemRefType::get(
      sizes, elType, strided ? layout : ::mlir::StridedLayoutAttr{});
}

/// get mixed statically and dynamically sized mlir::MemRefType for given sizes
/// and elType if strided==true make it a strided layout
::mlir::MemRefType getMemRefType(::mlir::MLIRContext *ctxt,
                                 const ::mlir::ValueRange &sizes,
                                 ::mlir::Type elType, bool strided) {
  return getMemRefType(ctxt, getShapeFromValues(sizes), elType, strided);
}

/// Create a 1d MemRef alloc with given size and elType
::mlir::Value createAllocMR(::mlir::OpBuilder &builder, ::mlir::Location loc,
                            ::mlir::Type elType, int64_t sz) {
  return builder.create<::mlir::memref::AllocOp>(
      loc, ::mlir::MemRefType::get({sz}, elType), builder.getI64IntegerAttr(8));
}

/// Create a 1d MemRef from given elements and elType
::mlir::Value createMemRefFromElements(::mlir::OpBuilder &builder,
                                       ::mlir::Location loc,
                                       ::mlir::Type elType,
                                       ::mlir::ValueRange elts) {
  int64_t N = elts.size();
  auto mr = createAllocMR(builder, loc, elType, N);
  for (auto i = 0; i < N; ++i) {
    auto idx = createIndex(loc, builder, i);
    (void)builder.createOrFold<::mlir::memref::StoreOp>(loc, elts[i], mr, idx);
  }
  return mr;
}

/// Create a cast op from ranked to unranked memref
::mlir::Value createUnrankedMemRefCast(::mlir::OpBuilder &builder,
                                       ::mlir::Location loc, ::mlir::Value mr) {
  auto elType = mlir::cast<::mlir::MemRefType>(mr.getType()).getElementType();
  auto umrType = ::mlir::UnrankedMemRefType::get(elType, {});
  auto umr = builder.create<::mlir::memref::CastOp>(loc, umrType, mr);
  return umr;
}

/// Create a 1d UnrankedMemRef from given elements and elType
::mlir::Value createURMemRefFromElements(::mlir::OpBuilder &builder,
                                         ::mlir::Location loc,
                                         ::mlir::Type elType,
                                         ::mlir::ValueRange elts) {
  auto mr = createMemRefFromElements(builder, loc, elType, elts);
  return createUnrankedMemRefCast(builder, loc, mr);
}

/// @return members of given 1d memref as individual values
::imex::ValVec createValuesFromMemRef(::mlir::OpBuilder &builder,
                                      ::mlir::Location loc, ::mlir::Value mr) {
  auto mrTyp = mlir::dyn_cast<::mlir::MemRefType>(mr.getType());
  assert(mrTyp && mrTyp.getShape().size() == 1);
  auto rank = mrTyp.getShape()[0];
  ::imex::ValVec vals(rank);
  for (auto i = 0; i < rank; ++i) {
    auto _i = createIndex(loc, builder, i);
    vals[i] = builder.createOrFold<::mlir::memref::LoadOp>(loc, mr, _i);
  }
  return vals;
}

/// @return if op can be moved directly after markerOp
/// @param [out] toBeMoved The op gets added here together with all the
/// dependences which also need to be moved
/// @param [in] dep set to true if examining a dependent op (these are allowed
/// to dominate markerOp!) This is a fairly simple recursive search. A move
/// with deps might still be possible even if returning false. toBeMoved is
/// filled bottom up, depth-first
bool canMoveAfter(::mlir::DominanceInfo &dom, ::mlir::Operation *op,
                  ::mlir::Operation *markerOp,
                  ::mlir::SmallVector<::mlir::Operation *> &toBeMoved,
                  bool dep) {
  if (markerOp == op) {
    assert(dep);
    return true;
  }

  // we cannot move the top-level op unless markerOp dominates it
  // if so, we recurse into the operands since they might need to move as well
  if (dom.dominates(markerOp, op)) {
    for (auto o : op->getOperands()) {
      auto dOp = o.getDefiningOp();
      if (dOp && !canMoveAfter(dom, dOp, markerOp, toBeMoved, true)) {
        // operand cannot move and does not dominate markerOp
        return false;
      }
    }
  } else if (dep) {
    // operands/dependencies are ok if they dominate markerOp
    return dom.dominates(op, markerOp);
  } else {
    // op does not dominate markerOp nor vice versa (op is top-level!)
    return false;
  }

  // we are good to move
  toBeMoved.emplace_back(op);
  return true;
}
} // namespace imex

// FIXME
#include <imex/Utils/ArithUtils.h>

namespace imex {

::mlir::Value createExtractPtrFromMemRef(::mlir::OpBuilder &builder,
                                         ::mlir::Location loc,
                                         ::mlir::Value mr) {
  auto meta = builder.create<::mlir::memref::ExtractStridedMetadataOp>(loc, mr);
  return createExtractPtrFromMemRef(builder, loc, mr, meta);
}

::mlir::Value createExtractPtrFromMemRefFromValues(::mlir::OpBuilder &builder,
                                                   ::mlir::Location loc,
                                                   ::mlir::ValueRange elts) {
  auto mr =
      createMemRefFromElements(builder, loc, builder.getIndexType(), elts);
  return createExtractPtrFromMemRef(builder, loc, mr);
}

void printValsAsMemRef(::mlir::Location loc, ::mlir::OpBuilder &builder,
                       ::mlir::ValueRange vals) {
  auto et = vals[0].getType();
  auto memrefType = ::mlir::UnrankedMemRefType::get(et, {});
  auto mr = createMemRefFromElements(builder, loc, et, vals);
  auto cmr = builder.create<mlir::memref::CastOp>(loc, memrefType, mr);
  if (et == builder.getIndexType()) {
    builder.create<::mlir::func::CallOp>(loc, "printMemrefInd",
                                         ::mlir::TypeRange(), cmr.getResult());
  } else if (et == builder.getI64Type()) {
    builder.create<::mlir::func::CallOp>(loc, "printMemrefI64",
                                         ::mlir::TypeRange(), cmr.getResult());
  } else {
    assert(false);
  }
}

// create ops to converting a tensor to a memref.
// First creates a toMemrefOp with the same shape as tensor.
// If this memref has a different shape than mrTyp, also creates a memref.cast
::mlir::Value createToMemRef(::mlir::Location loc, ::mlir::OpBuilder &builder,
                             ::mlir::Value input, ::mlir::Type toTyp,
                             bool clone) {
  auto iTyp = mlir::cast<::mlir::RankedTensorType>(input.getType());
  auto mrTyp = mlir::cast<::mlir::MemRefType>(toTyp);
  auto shapedMrTyp =
      mlir::cast<::mlir::ShapedType>(mrTyp).clone(iTyp.getShape());
  ::mlir::Value shapedMr = builder.create<::mlir::bufferization::ToMemrefOp>(
      loc, shapedMrTyp, input);
  if (clone) {
    shapedMr = builder.create<::mlir::bufferization::CloneOp>(loc, shapedMr);
  }
  return shapedMrTyp == toTyp
             ? shapedMr
             : builder.create<::mlir::memref::CastOp>(loc, toTyp, shapedMr)
                   .getResult();
}

} // namespace imex

mlir::LogicalResult parseShape(mlir::AsmParser &parser,
                               llvm::SmallVector<int64_t> &shape,
                               mlir::Type &type) {
  llvm::SmallVector<int64_t> dimensions;
  if (parser.parseDimensionList(dimensions))
    return mlir::failure();

  mlir::Type t;
  if (parser.parseType(t))
    return mlir::failure();

  shape = std::move(dimensions);
  type = std::move(t);
  return mlir::success();
}

void printShape(mlir::AsmPrinter &printer, llvm::ArrayRef<int64_t> shape,
                mlir::Type type) {
  for (int64_t dim : shape) {
    if (mlir::ShapedType::isDynamic(dim))
      printer << '?';
    else
      printer << dim;
    printer << 'x';
  }
  printer << type;
}
