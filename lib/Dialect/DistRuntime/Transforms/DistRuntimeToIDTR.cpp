//===- DistRuntimeToIDTR.cpp - DistRuntimeToIDTR Transform ------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements lowering distruntime ops to calls to IDTR.
//===----------------------------------------------------------------------===//

#include <imex/Dialect/Dist/Utils/Utils.h>
#include <imex/Dialect/DistRuntime/IR/DistRuntimeOps.h>
#include <imex/Dialect/DistRuntime/Transforms/Passes.h>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Dialect/NDArray/Utils/Utils.h>
#include <imex/Utils/PassUtils.h>
#include <imex/Utils/PassWrapper.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>

namespace imex {
#define GEN_PASS_DEF_LOWERDISTRUNTIMETOIDTR
#include <imex/Dialect/DistRuntime/Transforms/Passes.h.inc>
} // namespace imex

namespace imex {
namespace distruntime {

namespace {

// RuntimePrototypesOp -> func.func ops
// adding required function prototypes to the module level
struct RuntimePrototypes {

  // create function prototype fo given function name, arg-types and
  // return-types
  // If NoneTypes are present, it will generate mutiple functions, one for
  // each integer/float type, where all the NoneTypes get replaced by the
  // respective UnrankedMemref<elType>
  static void requireFunc(::mlir::Location &loc, ::mlir::OpBuilder &builder,
                          ::mlir::ModuleOp module, const char *fname,
                          ::mlir::TypeRange args, ::mlir::TypeRange results) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    // Insert before module terminator.
    builder.setInsertionPoint(module.getBody(),
                              std::prev(module.getBody()->end()));
    auto dataMRType = ::mlir::NoneType::get(builder.getContext());
    ::mlir::SmallVector<int> dmrs;
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i] == dataMRType)
        dmrs.emplace_back(i);
    }

    auto decl = [&](auto _fname, auto _args) {
      auto funcType = builder.getFunctionType(_args, results);
      auto func = builder.create<::mlir::func::FuncOp>(loc, _fname, funcType);
      func.setPrivate();
    };

    if (dmrs.empty()) {
      decl(fname, args);
    } else {
      ::imex::TypVec pargs(args);
      for (auto t :
           {::imex::ndarray::F64, ::imex::ndarray::F32, ::imex::ndarray::I64,
            ::imex::ndarray::I32, ::imex::ndarray::I16, ::imex::ndarray::I8,
            ::imex::ndarray::I1}) {
        auto elType = ::imex::ndarray::toMLIR(builder, t);
        auto mrtyp = ::mlir::UnrankedMemRefType::get(elType, {});
        for (auto i : dmrs) {
          pargs[i] = mrtyp;
        }
        auto tfname = mkTypedFunc(fname, elType);
        decl(tfname, pargs);
      }
    }
  }

  static void add_prototypes(::mlir::OpBuilder builder, ::mlir::Operation *op) {
    auto loc = op->getLoc();
    ::mlir::ModuleOp module = ::mlir::cast<mlir::ModuleOp>(op);
    auto indexType = builder.getIndexType();
    auto i64Type = builder.getI64Type();
    auto opType =
        builder.getIntegerType(sizeof(::imex::ndarray::ReduceOpId) * 8);
    auto i64MRType = ::mlir::UnrankedMemRefType::get(i64Type, {});
    // requireFunc will generate functions for multiple typed memref-types
    auto dataMRType = ::mlir::NoneType::get(builder.getContext());
    requireFunc(loc, builder, module, "printMemrefI64", {i64MRType}, {});
    auto idxMRType =
        ::mlir::UnrankedMemRefType::get(builder.getIndexType(), {});
    requireFunc(loc, builder, module, "printMemrefInd", {idxMRType}, {});
    requireFunc(loc, builder, module, "_idtr_nprocs", {i64Type}, {indexType});
    requireFunc(loc, builder, module, "_idtr_prank", {i64Type}, {indexType});
    requireFunc(loc, builder, module, "_idtr_reduce_all", {dataMRType, opType},
                {});
    requireFunc(loc, builder, module, "_idtr_copy_reshape",
                // team, gshape, loffs, lPart, ngshape, nloffs, nPart
                {i64Type, idxMRType, idxMRType, dataMRType, idxMRType,
                 idxMRType, dataMRType},
                {i64Type});
    requireFunc(loc, builder, module, "_idtr_update_halo",
                // team,    gshape,    loffs,     lPart,      bbOffset, bbShape,
                // lHalo,   rHalo, key
                {i64Type, idxMRType, idxMRType, dataMRType, idxMRType,
                 idxMRType, dataMRType, dataMRType, i64Type},
                {i64Type});
    requireFunc(loc, builder, module, "_idtr_wait",
                // handle
                {i64Type}, {});
    requireFunc(loc, builder, module, "_idtr_copy_permute",
                // team, gshape, loffs, lPart, nloffs, nPart, axes
                {i64Type, idxMRType, idxMRType, dataMRType, idxMRType,
                 dataMRType, idxMRType},
                {i64Type});
  }
};

/// Convert ::imex::distruntime::TeamSizeOp into call to _idtr_nprocs
struct TeamSizeOpPattern
    : public ::mlir::OpRewritePattern<::imex::distruntime::TeamSizeOp> {
  using ::mlir::OpRewritePattern<
      ::imex::distruntime::TeamSizeOp>::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::distruntime::TeamSizeOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto team = mlir::cast<::mlir::IntegerAttr>(op.getTeam()).getInt();
    auto loc = op.getLoc();

    rewriter.replaceOpWithNewOp<::mlir::func::CallOp>(
        op, "_idtr_nprocs", rewriter.getIndexType(),
        createInt(loc, rewriter, team));

    return ::mlir::success();
  }
};

/// Convert ::imex::distruntime::TeamMemberOp into call to _idtr_prank
struct TeamMemberOpPattern
    : public ::mlir::OpRewritePattern<::imex::distruntime::TeamMemberOp> {
  using ::mlir::OpRewritePattern<
      ::imex::distruntime::TeamMemberOp>::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::distruntime::TeamMemberOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto team = mlir::cast<::mlir::IntegerAttr>(op.getTeam()).getInt();
    auto loc = op.getLoc();

    rewriter.replaceOpWithNewOp<::mlir::func::CallOp>(
        op, "_idtr_prank", rewriter.getIndexType(),
        createInt(loc, rewriter, team));

    return ::mlir::success();
  }
};

struct CopyReshapeOpPattern
    : public ::mlir::OpRewritePattern<::imex::distruntime::CopyReshapeOp> {
  using ::mlir::OpRewritePattern<
      ::imex::distruntime::CopyReshapeOp>::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::distruntime::CopyReshapeOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto lArray = op.getLArray();
    auto arType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(lArray.getType());
    auto resType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getNlArray().getType());
    if (!arType || !resType) {
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto elType = resType.getElementType();
    auto team = op.getTeam();
    auto gShape = op.getGShape();
    auto lOffs = op.getLOffsets();
    auto ngShape = op.getNgShape();
    auto nlOffs = op.getNlOffsets();
    auto nlShape = op.getNlShape();

    // create output array with target size
    auto nlArray = rewriter.create<::imex::ndarray::CreateOp>(
        loc, nlShape, ::imex::ndarray::fromMLIR(elType), nullptr,
        resType.getEnvironments());

    auto idxType = rewriter.getIndexType();
    auto teamC = rewriter.create<::mlir::arith::ConstantOp>(
        loc, mlir::cast<::mlir::IntegerAttr>(team));
    auto gShapeMR = createURMemRefFromElements(rewriter, loc, idxType, gShape);
    auto lOffsMR = createURMemRefFromElements(rewriter, loc, idxType, lOffs);
    auto lArrayMR = ::imex::ndarray::mkURMemRef(loc, rewriter, lArray);
    auto ngShapeMR =
        createURMemRefFromElements(rewriter, loc, idxType, ngShape);
    auto nlOffsMR = createURMemRefFromElements(rewriter, loc, idxType, nlOffs);
    auto nlArrayMR = ::imex::ndarray::mkURMemRef(loc, rewriter, nlArray);

    auto fun =
        rewriter.getStringAttr(mkTypedFunc("_idtr_copy_reshape", elType));
    auto handle = rewriter.create<::mlir::func::CallOp>(
        loc, fun, rewriter.getI64Type(),
        ::mlir::ValueRange{teamC, gShapeMR, lOffsMR, lArrayMR, ngShapeMR,
                           nlOffsMR, nlArrayMR});
    rewriter.replaceOp(op, {handle.getResult(0), nlArray});
    return ::mlir::success();
  }
};

/// @brief  lower GetHaloOp
/// Determine sizes of halos, alloc halos and call idtr.
/// Before accessing/reading from returned halos, the caller must
/// call the appropriate wait call in idtr.
/// @return handle, left halo, right halo
struct GetHaloOpPattern
    : public ::mlir::OpRewritePattern<::imex::distruntime::GetHaloOp> {
  using ::mlir::OpRewritePattern<
      ::imex::distruntime::GetHaloOp>::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::distruntime::GetHaloOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto lData = op.getLocal();
    auto arTyp = mlir::dyn_cast<::imex::ndarray::NDArrayType>(lData.getType());
    if (!arTyp)
      return ::mlir::failure();

    auto elType = arTyp.getElementType();

    auto mkHalo = [&](const ::imex::ValVec &szs) {
      ::mlir::Value iVal =
#ifdef DEBUG_HALO
          createCast(loc, rewriter,
                     elType.isIntOrIndex()
                         ? createInt(loc, rewriter, 4711,
                                     elType.getIntOrFloatBitWidth())
                         : createFloat(loc, rewriter, 4711,
                                       elType.getIntOrFloatBitWidth()),
                     elType);
#else
          nullptr;
#endif
      auto outPTnsr = rewriter.create<::imex::ndarray::CreateOp>(
          loc, szs, ::imex::ndarray::fromMLIR(elType), iVal,
          ::imex::dist::getNonDistEnvs(arTyp));
      auto outUMR = ::imex::ndarray::mkURMemRef(loc, rewriter, outPTnsr);
      return std::make_pair(outPTnsr, outUMR);
    };

    ::imex::ValVec lSizes =
        ::imex::ndarray::createShapeOf(loc, rewriter, lData);
    auto lOffsets = op.getLOffsets();
    auto gShape = op.getGShape();
    ::imex::ValVec bbOffs = op.getBbOffsets();
    ::imex::ValVec bbSizes = op.getBbSizes();

    // Prepare args for calling update_halo
    auto idxType = rewriter.getIndexType();
    auto gShapeMR = createURMemRefFromElements(rewriter, loc, idxType, gShape);
    auto lOffsMR = createURMemRefFromElements(rewriter, loc, idxType, lOffsets);
    // we pass the entire local data to update_halo, not just the subview
    auto lPart = ::imex::ndarray::mkURMemRef(loc, rewriter, lData);
    auto bbOffsMR = createURMemRefFromElements(rewriter, loc, idxType, bbOffs);
    auto bbSizesMR =
        createURMemRefFromElements(rewriter, loc, idxType, bbSizes);

    // determine overlap of new local part, we split dim 0 only
    auto zero = easyIdx(loc, rewriter, 0);
    auto one = easyIdx(loc, rewriter, 1);
    auto bbOff = easyIdx(loc, rewriter, bbOffs[0]);
    auto bbSize = easyIdx(loc, rewriter, bbSizes[0]);
    auto oldOff = easyIdx(loc, rewriter, lOffsets[0]);
    auto oldSize = easyIdx(loc, rewriter, lSizes[0]);
    auto tEnd = bbOff + bbSize;
    auto oldEnd = oldOff + oldSize;
    auto ownOff = oldOff.max(bbOff);
    auto ownSize = (oldEnd.min(tEnd) - ownOff).max(zero);

    // compute left and right halo sizes, we split dim 0 only
    ::imex::ValVec lHSizes(bbSizes), rHSizes(bbSizes);
    auto sgShape = getShapeFromValues(gShape);
    if (::imex::ndarray::isUnitShape(sgShape)) {
      lHSizes[0] =
          oldSize.eq(zero).land(oldOff.sgt(zero)).select(one, zero).get();
      rHSizes[0] =
          oldSize.eq(zero).land(oldOff.sle(zero)).select(one, zero).get();
    } else {
      lHSizes[0] = (ownOff.min(tEnd) - bbOff).get();
      rHSizes[0] = (tEnd - (ownOff + ownSize)).max(zero).get();
    }

    auto lOut = mkHalo(lHSizes);
    auto rOut = mkHalo(rHSizes);
    auto key = createInt(loc, rewriter, op.getKey());

    // call our runtime function to redistribute data across processes
    auto fun = rewriter.getStringAttr(mkTypedFunc("_idtr_update_halo", elType));
    auto handle = rewriter.create<::mlir::func::CallOp>(
        loc, fun, rewriter.getI64Type(),
        ::mlir::ValueRange{createInt(loc, rewriter, 0), gShapeMR, lOffsMR,
                           lPart, bbOffsMR, bbSizesMR, lOut.second, rOut.second,
                           key});

    rewriter.replaceOp(op, {handle.getResult(0), lOut.first, rOut.first});
    return ::mlir::success();
  }
};

/// Convert ::imex::distruntime::AllReduceOp into runtime call to
/// "_idtr_reduce_all". Pass local RankedTensor as argument. Replaces op with
/// new distributed array.
struct AllReduceOpPattern
    : public ::mlir::OpRewritePattern<::imex::distruntime::AllReduceOp> {
  using ::mlir::OpRewritePattern<
      ::imex::distruntime::AllReduceOp>::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::distruntime::AllReduceOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    // get guid and rank and call runtime function
    auto loc = op.getLoc();
    auto mRef = op.getData();
    auto mRefType = mlir::dyn_cast<::mlir::MemRefType>(mRef.getType());
    if (!mRefType)
      return ::mlir::failure();

    auto opV = rewriter.create<::mlir::arith::ConstantOp>(
        loc, ::mlir::cast<::mlir::TypedAttr>(op.getOp()));
    auto elType = mRefType.getElementType();

    auto fsa = rewriter.getStringAttr(mkTypedFunc("_idtr_reduce_all", elType));
    auto dataUMR = createUnrankedMemRefCast(rewriter, loc, mRef);

    rewriter.replaceOpWithNewOp<::mlir::func::CallOp>(
        op, fsa, ::mlir::TypeRange(), ::mlir::ValueRange({dataUMR, opV}));
    return ::mlir::success();
  }
};

/// Convert ::imex::distruntime::WaitOp into call to _idtr_wait
struct WaitOpPattern
    : public ::mlir::OpRewritePattern<::imex::distruntime::WaitOp> {
  using ::mlir::OpRewritePattern<::imex::distruntime::WaitOp>::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::distruntime::WaitOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto fsa = rewriter.getStringAttr("_idtr_wait");
    rewriter.replaceOpWithNewOp<::mlir::func::CallOp>(
        op, fsa, ::mlir::TypeRange(), ::mlir::ValueRange{op.getHandle()});

    return ::mlir::success();
  }
};

struct CopyPermuteOpPattern
    : public ::mlir::OpRewritePattern<::imex::distruntime::CopyPermuteOp> {
  using ::mlir::OpRewritePattern<
      ::imex::distruntime::CopyPermuteOp>::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::distruntime::CopyPermuteOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto lArray = op.getLArray();
    auto arType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(lArray.getType());
    auto resType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getNlArray().getType());
    if (!arType || !resType) {
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto elType = resType.getElementType();

    auto team = op.getTeam();
    auto gShape = op.getGShape();
    auto lOffs = op.getLOffsets();
    auto nlOffs = op.getNlOffsets();
    auto nlShape = op.getNlShape();
    auto axes = op.getAxes();

    ::mlir::SmallVector<::mlir::Value> axesValues;
    for (auto axis : axes) {
      axesValues.emplace_back(
          rewriter.create<::mlir::arith::ConstantIndexOp>(loc, axis));
    }

    // create output array with target size
    auto nlArray = rewriter.create<::imex::ndarray::CreateOp>(
        loc, nlShape, ::imex::ndarray::fromMLIR(elType), nullptr,
        resType.getEnvironments());

    auto idxType = rewriter.getIndexType();
    auto teamC = rewriter.create<::mlir::arith::ConstantOp>(
        loc, mlir::cast<::mlir::IntegerAttr>(team));
    auto gShapeMR = createURMemRefFromElements(rewriter, loc, idxType, gShape);
    auto lOffsMR = createURMemRefFromElements(rewriter, loc, idxType, lOffs);
    auto lArrayMR = ::imex::ndarray::mkURMemRef(loc, rewriter, lArray);
    auto nlOffsMR = createURMemRefFromElements(rewriter, loc, idxType, nlOffs);
    auto nlArrayMR = ::imex::ndarray::mkURMemRef(loc, rewriter, nlArray);
    auto axesMR =
        createURMemRefFromElements(rewriter, loc, idxType, axesValues);

    auto fun =
        rewriter.getStringAttr(mkTypedFunc("_idtr_copy_permute", elType));
    auto handle = rewriter.create<::mlir::func::CallOp>(
        loc, fun, rewriter.getI64Type(),
        ::mlir::ValueRange{teamC, gShapeMR, lOffsMR, lArrayMR, nlOffsMR,
                           nlArrayMR, axesMR});
    rewriter.replaceOp(op, {handle.getResult(0), nlArray});
    return ::mlir::success();
  }
};

struct DistRuntimeToIDTRPass
    : public impl::LowerDistRuntimeToIDTRBase<DistRuntimeToIDTRPass> {

  DistRuntimeToIDTRPass() = default;

  void runOnOperation() override {

    ::mlir::OpBuilder builder(&getContext());
    RuntimePrototypes::add_prototypes(builder, this->getOperation());

    ::mlir::FrozenRewritePatternSet patterns;
    insertPatterns<CopyReshapeOpPattern, TeamSizeOpPattern, TeamMemberOpPattern,
                   GetHaloOpPattern, AllReduceOpPattern, WaitOpPattern,
                   CopyPermuteOpPattern>(getContext(), patterns);
    (void)::mlir::applyPatternsGreedily(this->getOperation(), patterns);
  }; // runOnOperation()

}; // DistRuntimeToIDTRPass

} // namespace
} // namespace distruntime

/// Create DistRuntimeToIDTRPass
std::unique_ptr<::mlir::Pass> createDistRuntimeToIDTRPass() {
  return std::make_unique<::imex::distruntime::DistRuntimeToIDTRPass>();
}
} // namespace imex

extern "C" {
int _idtr_nprocs(void *) __attribute__((weak));
int _idtr_prank(void *) __attribute__((weak));
}

namespace imex {
namespace distruntime {

static auto DNDA_NPROCS = getenv("DNDA_NPROCS");
static auto DNDA_PRANK = getenv("DNDA_PRANK");

::mlir::OpFoldResult TeamSizeOp::fold(FoldAdaptor adaptor) {
  // call runtime at compile time if available and team is constant
  if (DNDA_NPROCS) {
    auto np = std::stoi(DNDA_NPROCS);
    ::mlir::Builder builder(getContext());
    return builder.getIndexAttr(np);
  }
  if (_idtr_nprocs != NULL) {
    ::mlir::Builder builder(getContext());
    auto team = mlir::cast<::mlir::IntegerAttr>(adaptor.getTeam()).getInt();
    auto np = _idtr_nprocs(reinterpret_cast<void *>(team));
    return builder.getIndexAttr(np);
  }
  return nullptr;
}

::mlir::OpFoldResult TeamMemberOp::fold(FoldAdaptor adaptor) {
  // call runtime at compile time if available and team is constant
  if (DNDA_PRANK) {
    auto np = std::stoi(DNDA_PRANK);
    ::mlir::Builder builder(getContext());
    return builder.getIndexAttr(np);
  }
  if (_idtr_prank != NULL) {
    ::mlir::Builder builder(getContext());
    auto team = mlir::cast<::mlir::IntegerAttr>(adaptor.getTeam()).getInt();
    auto np = _idtr_prank(reinterpret_cast<void *>(team));
    return builder.getIndexAttr(np);
  }
  return nullptr;
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
/// Ported from mlir::tensor dialect
mlir::Operation *imex::distruntime::DistRuntimeDialect::materializeConstant(
    mlir::OpBuilder &builder, mlir::Attribute value, mlir::Type type,
    mlir::Location loc) {
  if (auto op = mlir::arith::ConstantOp::materialize(builder, value, type, loc))
    return op;
  return nullptr;
}

} // namespace distruntime
} // namespace imex
