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

#ifndef _IMEX_PASSUTILS_H_
#define _IMEX_PASSUTILS_H_

#include <llvm/Support/raw_os_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dominance.h>

namespace imex {

using ValVec = ::mlir::SmallVector<::mlir::Value>;
using TypVec = ::mlir::SmallVector<::mlir::Type>;

/// @return get ::mlir::FloatAttr with given Value and bitwidth W
template <typename T>
::mlir::FloatAttr getFloatAttr(::mlir::OpBuilder &builder, T val, int W = 64) {
  if (W == 64)
    return builder.getF64FloatAttr(val);
  if (W == 32)
    return builder.getF32FloatAttr(val);
  if (W == 16)
    return builder.getF16FloatAttr(val);
  assert(false && "only 32- and 64-bit floats supported");
}

/// @return new float ::mlir::Value with given Value and bitwidth W
template <typename T>
::mlir::Value createFloat(const ::mlir::Location &loc,
                          ::mlir::OpBuilder &builder, T val, int W = 64) {
  auto attr = getFloatAttr(builder, val, W);
  return builder.create<::mlir::arith::ConstantOp>(loc, attr);
}

/// @return get ::mlir::IntegerAttr with given Value and bitwidth W
inline ::mlir::IntegerAttr getIntAttr(::mlir::OpBuilder &builder, int64_t val,
                                      int W = 64) {
  return builder.getIntegerAttr(builder.getIntegerType(W), val);
}

/// @return new integer ::mlir::Value with given Value and bitwidth W
inline ::mlir::Value createInt(const ::mlir::Location &loc,
                               ::mlir::OpBuilder &builder, int64_t val,
                               int W = 64) {
  auto attr = getIntAttr(builder, val, W);
  return builder.create<::mlir::arith::ConstantOp>(loc, attr);
}

/// @return new index ::mlir::Value with given Value
extern ::mlir::Value createIndex(const ::mlir::Location &loc,
                                 ::mlir::OpBuilder &builder, int64_t val);

extern ::mlir::Value createIndexCast(const ::mlir::Location &loc,
                                     ::mlir::OpBuilder &builder,
                                     ::mlir::Value val,
                                     ::mlir::Type intTyp = ::mlir::Type());

// cast between different scalar types
extern ::mlir::Value createCast(const ::mlir::Location &loc,
                                ::mlir::OpBuilder &builder, ::mlir::Value val,
                                ::mlir::Type dTyp);

/// @return array of static sizes form ValueRange: actual size if constant,
/// kDynamic if not
extern ::mlir::SmallVector<int64_t>
getShapeFromValues(const ::mlir::ValueRange &sizes);

/// @return number of elements for given shape if all sizes are constant,
/// kDynamic otherwise
extern int64_t getSizeFromValues(const ::mlir::ValueRange &sizes);

/// get dyn-sized mlir::RankedTensorType for given size values and elType
extern ::mlir::RankedTensorType getTensorType(::mlir::MLIRContext *ctxt,
                                              const ::mlir::ValueRange &sizes,
                                              ::mlir::Type elType);

/// get dyn-sized mlir::RankedTensorType for given shape and elType
extern ::mlir::RankedTensorType getTensorType(::mlir::MLIRContext *ctxt,
                                              ::mlir::ArrayRef<int64_t> shape,
                                              ::mlir::Type elType);

/// get dyn-sized mlir::RankedTensorType for given rank and elType
extern ::mlir::RankedTensorType
getTensorType(::mlir::MLIRContext *ctxt, int64_t rank, ::mlir::Type elType);

// convert a range of values into their equivalent constant int or
// ::mlir::ShapedType::kDynamic
extern ::mlir::SmallVector<int64_t> mkConstant(::mlir::ValueRange vals);

/// combine dynamic and static sizes (as used by SubviewOps) into a
/// single ValueRange (vecotr of values)
extern ::imex::ValVec getMixedAsValues(const ::mlir::Location &loc,
                                       ::mlir::OpBuilder &builder,
                                       const ::mlir::ValueRange &dyns,
                                       ::llvm::ArrayRef<int64_t> statics,
                                       bool asI64 = false);

/// similar to mlir::decomposeMixedValues but converting const values tot
/// statics
extern void
dispatchIndexValues(::mlir::OpBuilder &builder, ::mlir::Location loc,
                    const ::mlir::ValueRange &sizes,
                    ::mlir::SmallVectorImpl<::mlir::Value> &dynamicVec,
                    ::mlir::SmallVectorImpl<int64_t> &staticVec);

/// create an empty RankedTensor with given shape and elType
extern ::mlir::Value createEmptyTensor(::mlir::OpBuilder &builder,
                                       ::mlir::Location loc,
                                       ::mlir::Type elType,
                                       const ::mlir::ValueRange &shp);

/// create an empty RankedTensor for given result tensor type and operand shapes
extern ::mlir::Value createEmptyTensor(::mlir::OpBuilder &builder,
                                       ::mlir::Location loc,
                                       ::mlir::TensorType resType,
                                       const ::mlir::ValueRange &operands);

/// get dyn-sized mlir::RankedTensorType for given rank and elType
/// if strided==true make it a strided layout
extern ::mlir::MemRefType getMemRefType(::mlir::MLIRContext *ctxt, int64_t rank,
                                        ::mlir::Type elType,
                                        bool strided = true);

/// get mixed statically and dynamically sized mlir::MemRefType for given sizes
/// and elType if strided==true make it a strided layout
extern ::mlir::MemRefType getMemRefType(::mlir::MLIRContext *ctxt,
                                        ::mlir::ArrayRef<int64_t> sizes,
                                        ::mlir::Type elType,
                                        bool strided = true);

/// get mixed statically and dynamically sized mlir::MemRefType for given sizes
/// and elType if strided==true make it a strided layout
extern ::mlir::MemRefType getMemRefType(::mlir::MLIRContext *ctxt,
                                        const ::mlir::ValueRange &sizes,
                                        ::mlir::Type elType,
                                        bool strided = true);

inline ::mlir::MemRefType getMemRefType(::mlir::RankedTensorType ttype) {
  return getMemRefType(ttype.getContext(), ttype.getShape(),
                       ttype.getElementType());
}

/// Create a 1d MemRef alloc with given size and elType
extern ::mlir::Value createAllocMR(::mlir::OpBuilder &builder,
                                   ::mlir::Location loc, ::mlir::Type elType,
                                   int64_t sz);

/// Create a 1d MemRef from given elements and elType
extern ::mlir::Value createMemRefFromElements(::mlir::OpBuilder &builder,
                                              ::mlir::Location loc,
                                              ::mlir::Type elType,
                                              ::mlir::ValueRange elts);

/// Create a cast op from ranked to unranked memref
extern ::mlir::Value createUnrankedMemRefCast(::mlir::OpBuilder &builder,
                                              ::mlir::Location loc,
                                              ::mlir::Value mr);

/// Create a 1d UnrankedMemRef from given elements and elType
extern ::mlir::Value createURMemRefFromElements(::mlir::OpBuilder &builder,
                                                ::mlir::Location loc,
                                                ::mlir::Type elType,
                                                ::mlir::ValueRange elts);

/// @return members of given 1d memref as individual values
extern ::imex::ValVec createValuesFromMemRef(::mlir::OpBuilder &builder,
                                             ::mlir::Location loc,
                                             ::mlir::Value mr);

/// @return if op can be moved directly after markerOp
/// @param [out] toBeMoved The op gets added here together with all the
/// dependences which also need to be moved
/// @param [in] dep set to true if examining a dependent op (these are allowed
/// to dominate markerOp!) This is a fairly simple recursive search. A move
/// with deps might still be possible even if returning false. toBeMoved is
/// filled bottom up, depth-first
extern bool canMoveAfter(::mlir::DominanceInfo &dom, ::mlir::Operation *op,
                         ::mlir::Operation *markerOp,
                         ::mlir::SmallVector<::mlir::Operation *> &toBeMoved,
                         bool dep = false);
} // namespace imex

// FIXME
#include <imex/Utils/ArithUtils.h>

namespace imex {
template <typename T>
extern ::mlir::Value createExtractPtrFromMemRef(::mlir::OpBuilder &builder,
                                                ::mlir::Location loc,
                                                ::mlir::Value mr, T meta) {
  auto off = easyIdx(loc, builder, meta.getOffset());
  auto aptr = easyIdx(
      loc, builder,
      builder.create<::mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, mr));
  return (aptr + (off * easyIdx(loc, builder, sizeof(uint64_t)))).get();
}

extern ::mlir::Value createExtractPtrFromMemRef(::mlir::OpBuilder &builder,
                                                ::mlir::Location loc,
                                                ::mlir::Value mr);

extern ::mlir::Value createExtractPtrFromMemRefFromValues(
    ::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange elts);

// when possible, move up operations of a certain type so that they are
// close together.
template <typename OP, typename SELECT, typename GETINPUTS,
          typename SINGULARIZE>
void groupOps(::mlir::DominanceInfo &domA, ::mlir::Operation *root,
              SELECT select, GETINPUTS getInputs,
              SINGULARIZE singularize = nullptr) {

  llvm::SmallVector<OP> dominators, dominators2;

  // Find all operations of type OP within root
  root->walk([&](OP op) {
    if (select(op)) {
      dominators.emplace_back(op);
      return;
    }
  });

  // we treat the first found op as the dominating operation
  // We try to move up all found ops to right after the dominator
  // Ops which cannot be be moved will serve as new dominators and we
  // recursively try to move remaining ops to them
  while (dominators.size() > 1) {
    auto dominator = dominators.front();
    auto iPnt = dominator;
    while (dominators.size() > 1) {
      auto op = dominators.pop_back_val();
      if (domA.properlyDominates(dominator, op, false)) {
        bool can_move = true;
        auto oprnds = getInputs(op);
        for (auto d : oprnds) {
          auto defOp = d.getDefiningOp();
          if (defOp && !domA.properlyDominates(defOp, dominator)) {
            can_move = false;
            break;
          }
        }
        if (can_move) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waddress"
          if constexpr (singularize != nullptr) {
            if (singularize(dominator, op)) {
              op->replaceAllUsesWith(dominator);
              op->erase();
              continue;
            }
          }
          op->moveAfter(iPnt);
          iPnt = op;
          if constexpr (singularize == nullptr) {
            continue;
          }
#pragma GCC diagnostic pop
        }
      }
      // not dominated or not movable
      dominators2.emplace_back(op);
    }
    dominators.clear();
    dominators.swap(dominators2);
  }
}

extern void printValsAsMemRef(::mlir::Location loc, ::mlir::OpBuilder &builder,
                              ::mlir::ValueRange vals);

// create ops to converting a tensor to a memref.
// First creates a toMemrefOp with the same shape as tensor.
// If this memref has a different shape than mrTyp, also creates a memref.cast
extern ::mlir::Value createToMemRef(::mlir::Location loc,
                                    ::mlir::OpBuilder &builder,
                                    ::mlir::Value input, ::mlir::Type toTyp,
                                    bool clone = false);

// broadcast 2 shapes into one according to the array-API
template <typename V1, typename V2>
::mlir::SmallVector<int64_t> broadcast(const V1 &shape1, const V2 &shape2) {
  int64_t N1 = shape1.size();
  int64_t N2 = shape2.size();
  int64_t N = std::max(N1, N2);
  ::mlir::SmallVector<int64_t> shape(N);

  for (int64_t i = N - 1; i >= 0; --i) {
    auto n1 = N1 - N + i;
    auto d1 = n1 >= 0 ? shape1[n1] : 1;
    auto n2 = N2 - N + i;
    auto d2 = n2 >= 0 ? shape2[n2] : 1;
    if (d1 == 0 || d2 == 0) {
      shape[i] = 0;
    } else if (d1 < 0) {
      shape[i] = d1;
    } else if (d2 < 0) {
      shape[i] = d2;
    } else if (d1 == 1) {
      shape[i] = d2;
    } else if (d2 == 1) {
      shape[i] = d1;
    } else if (d1 == d2) {
      shape[i] = d1;
    } else {
      assert(false && "Trying to broadcast incomaptible shapes");
    }
  }
  return shape;
}

inline std::string mlirTypeToString(::mlir::Type type) {
  std::ostringstream oss;
  llvm::raw_os_ostream os(oss);
  type.print(os);
  os.flush();
  return oss.str();
}

inline std::string mkTypedFunc(const ::std::string &base, ::mlir::Type elType) {
  return base + "_" + mlirTypeToString(elType);
}
} // namespace imex

extern mlir::LogicalResult parseShape(mlir::AsmParser &parser,
                                      llvm::SmallVector<int64_t> &shape,
                                      mlir::Type &type);

extern void printShape(mlir::AsmPrinter &printer, llvm::ArrayRef<int64_t> shape,
                       mlir::Type type);

#endif // _IMEX_PASSUTILS_H_
