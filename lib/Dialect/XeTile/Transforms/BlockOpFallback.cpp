//====-- BlockOpFallback.cpp - XeTile Block Op Fallback Pass  ----*- C++-*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This pass detects XeTile InitTile ops that does not meet HW restrictions
/// and rewrite into InitTile ops with scattered description.
/// This triggers change to tile type. Shape is same but scattered attribute
/// is added. As a result, ops requiring block description gets invalid.
/// Patterns that legalizes those ops are added and GreedyPatternRewriteDriver
/// is used to apply the patterns.
///
//===----------------------------------------------------------------------===//

#include "imex/Dialect/XeTile/IR/XeTileOps.h"
#include "imex/Dialect/XeTile/Transforms/Passes.h"
#include "imex/Utils/XeArch.h"
#include "imex/Utils/XeCommon.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace imex {
#define GEN_PASS_DEF_XETILEBLOCKOPFALLBACK
#include "imex/Dialect/XeTile/Transforms/Passes.h.inc"
} // namespace imex

//
// Limitations and future plan:
// Currently limited to 2D static shaped source memref of row major order.
// Dynamic shape with known pitch value will be supported in the future.
// Scattered offset is calculated by generated code sequence but will be
// using constant immediate values for static shapes in the future.
// Current code sequence is not optimal for supporting blocking pass and
// SIMT lowering. This will be improved in the future.
// Correct mask generation requires getting absolute offsets from source
// memref. This is not supported in the current implementation and will be
// added in the future. For now, mask generation assume all accesses are
// in bounds
//

namespace blockopfallback {

static imex::xetile::TileType addScatterAttr(imex::xetile::TileType tileTy) {
  auto tileAttr =
      mlir::dyn_cast_or_null<imex::xetile::XeTileAttr>(tileTy.getEncoding());
  int memorySpace = tileTy.getMemorySpaceAsInt();
  if (!tileAttr) {
    std::vector<int32_t> orderVec = {1, 0};
    llvm::ArrayRef<int32_t> order(orderVec);
    auto encoding = imex::xetile::XeTileAttr::get(tileTy.getContext(), order,
                                                  memorySpace, true);
    return imex::xetile::TileType::get(tileTy.getShape(),
                                       tileTy.getElementType(), encoding);
  }
  auto sgMap = tileAttr.getSgMap();
  auto wgMap = tileAttr.getWgMap();
  auto order = tileAttr.getOrder().asArrayRef();
  auto scatterTileAttr = imex::xetile::XeTileAttr::get(
      tileTy.getContext(), sgMap, wgMap, order, memorySpace, true);
  return imex::xetile::TileType::get(tileTy.getShape(), tileTy.getElementType(),
                                     scatterTileAttr);
}

struct InitTileOpPattern final
    : public mlir::OpRewritePattern<imex::xetile::InitTileOp> {
public:
  llvm::DenseSet<mlir::Value> &convertToScatteredType;

  InitTileOpPattern(mlir::MLIRContext *context,
                    std::shared_ptr<imex::XeuArchInterface> uArch,
                    llvm::DenseSet<mlir::Value> &map)
      : OpRewritePattern<imex::xetile::InitTileOp>(context),
        convertToScatteredType(map) {
    uArchInterface = uArch;
  }

  mlir::LogicalResult
  matchAndRewrite(imex::xetile::InitTileOp initTileOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto tileTy = initTileOp.getType();
    // Skip if tile is scattered
    if (tileTy.isScattered()) {
      return mlir::failure();
    }
    // Skip 1D tile
    if (tileTy.getRank() < 2) {
      return mlir::failure();
    }
    // Skip if tile is column major
    if (tileTy.getOrder().asArrayRef() != llvm::ArrayRef<int32_t>{1, 0}) {
      return mlir::failure();
    }
    // Currenty only supports memref source
    if (!initTileOp.isSourceMemRef()) {
      return mlir::failure();
    }
    // Cannot handle non static shape
    if (!initTileOp.sourceMemRefHasStaticShape()) {
      return mlir::failure();
    }

    auto srcShape = initTileOp.getSourceMemrefStaticShape();
    // Cannot handle non 2D source memref
    if (srcShape.size() != 2) {
      return mlir::failure();
    }

    // Cannot handle sub byte element type
    if (tileTy.getElementTypeBitWidth() < 8) {
      return mlir::failure();
    }

    // Check if memspace is SLM
    auto memorySpace = initTileOp.getSourceMemorySpaceAsInt();
    bool isSLM = memorySpace == 3;
    // Check pitch >= 64bytes and pitch is multiple of 16bytes
    bool isValidPitch = true;
    auto pitchNumElems = srcShape[srcShape.size() - 1];
    auto elemBitwidth =
        initTileOp.getSourceMemrefElemType().getIntOrFloatBitWidth();
    auto pitchNumBytes = pitchNumElems * elemBitwidth / 8;
    auto config = uArchInterface->get2DPrefetchConfig(initTileOp.getOperation(),
                                                      elemBitwidth);
    auto conf = config.value();
    isValidPitch = (pitchNumBytes >= conf.minPitch) &&
                   (pitchNumBytes % conf.pitchMultiple == 0);

    bool convertToScatter =
        convertToScatteredType.contains(initTileOp.getResult());
    // If memspace is not SLM, pitch is valid, and no tile type conversion
    // to scatter is needed, then no need to rewrite
    if (!isSLM && isValidPitch && !convertToScatter) {
      return mlir::failure();
    }
    bool mayNeedMask = (pitchNumElems % tileTy.getShape().back() != 0);
    if (mayNeedMask) {
      return mlir::failure();
    }

    // memspace is SLM, but can be hanled with optimal SLM accesses,
    // no need to rewrite too, but the shape of all uses (init_tile) of
    // the SLM must support optimal SLM accesses.
    std::function<bool(mlir::Value)> isSupportedOptimalSLMAccessForAllUsers =
        [&](mlir::Value value) {
          bool res = true;
          for (mlir::Operation *u : value.getUsers()) {
            if (auto init = mlir::dyn_cast<imex::xetile::InitTileOp>(u))
              res &= imex::isSupportedOptimalSLMAccess(init.getType());

            if (auto trans = mlir::dyn_cast<mlir::memref::TransposeOp>(u))
              res &= isSupportedOptimalSLMAccessForAllUsers(trans);
          }
          return res;
        };
    auto src = initTileOp.getSource();
    if (isSLM && isSupportedOptimalSLMAccessForAllUsers(src)) {
      return mlir::failure();
    }

    // Get flat shape size
    int64_t flatSize = 1;
    for (auto dim : srcShape) {
      flatSize *= dim;
    }

    // reinterpret_cast to flat memref of flatSize
    mlir::MemRefLayoutAttrInterface layout = {};
    // Is source offset always 0? No API to check.
    auto flatMemref = mlir::memref::ReinterpretCastOp::create(
        rewriter, initTileOp.getLoc(),
        mlir::MemRefType::get({flatSize}, initTileOp.getSourceMemrefElemType(),
                              layout, initTileOp.getSourceMemorySpace()),
        initTileOp.getSource(), 0, llvm::ArrayRef<int64_t>{flatSize},
        llvm::ArrayRef<int64_t>{1});

    // Create indices for scatter
    auto offsets = initTileOp.getMixedOffsets();
    auto loc = initTileOp.getLoc();
    auto offsetX = imex::getValueOrConstantOp(offsets[0], loc, rewriter,
                                              rewriter.getIndexType());
    auto offsetY = imex::getValueOrConstantOp(offsets[1], loc, rewriter,
                                              rewriter.getIndexType());
    auto indexVecTy =
        mlir::VectorType::get(tileTy.getShape(), rewriter.getIndexType());
    bool isSingleCol = tileTy.getShape().back() == 1;
    bool isSingleRow = tileTy.getShape().front() == 1;
    auto rowIndexVecTy = mlir::VectorType::get({tileTy.getShape().front()},
                                               rewriter.getIndexType());
    auto colIndexVecTy = mlir::VectorType::get({tileTy.getShape().back()},
                                               rewriter.getIndexType());

    // Create
    // [0, ...., TileShape[1]-1] broadcasted to TileShape
    // if isSingleCol, splat offsetY to TileShape
    mlir::Value stepOffsetTile;
    if (isSingleCol) {
      stepOffsetTile = rewriter.createOrFold<mlir::vector::BroadcastOp>(
          loc, indexVecTy, offsetY);
    } else {
      auto stepVec =
          rewriter.createOrFold<mlir::vector::StepOp>(loc, colIndexVecTy);
      auto stepTile = rewriter.createOrFold<mlir::vector::BroadcastOp>(
          loc, indexVecTy, stepVec);
      auto offsetYTile = rewriter.createOrFold<mlir::vector::BroadcastOp>(
          loc, indexVecTy, offsetY);
      // Add offsetY to step
      stepOffsetTile = rewriter.createOrFold<mlir::arith::AddIOp>(
          loc, indexVecTy, stepTile, offsetYTile);
    }

    // create [0, 1, 2, ...., TileShape[0]-1]^T broadcasted to TileShape
    // if isSingleRow, splat offsetX to TileShape
    mlir::Value rowOffsetTile;
    if (isSingleRow) {
      rowOffsetTile = rewriter.createOrFold<mlir::vector::BroadcastOp>(
          loc, indexVecTy, offsetX);
    } else {
      auto rowVecT =
          rewriter.createOrFold<mlir::vector::StepOp>(loc, rowIndexVecTy);
      auto offsetXVec = rewriter.createOrFold<mlir::vector::BroadcastOp>(
          loc,
          mlir::VectorType::get({tileTy.getShape().front()},
                                rewriter.getIndexType()),
          offsetX);
      // Add offsetX to rowVecT
      auto rowOffsetVecT = rewriter.createOrFold<mlir::arith::AddIOp>(
          loc, rowIndexVecTy, rowVecT, offsetXVec);
      // reshape to row x 1
      auto rowOffsetVec = rewriter.createOrFold<mlir::vector::ShapeCastOp>(
          loc,
          mlir::VectorType::get({tileTy.getShape().front(), 1},
                                rewriter.getIndexType()),
          rowOffsetVecT);
      // broadcast to TileShape
      rowOffsetTile = rewriter.createOrFold<mlir::vector::BroadcastOp>(
          loc, indexVecTy, rowOffsetVec);
    }

    // create [pitchNumElems] splatted to TileShape
    auto stride = rewriter.createOrFold<mlir::arith::ConstantOp>(
        loc, rewriter.getIndexType(), rewriter.getIndexAttr(pitchNumElems));
    auto strideTile = rewriter.createOrFold<mlir::vector::BroadcastOp>(
        loc, indexVecTy, stride);
    // Create a temp with just rowTile * strideTile
    auto rowStrideTile = rewriter.createOrFold<mlir::arith::MulIOp>(
        loc, indexVecTy, rowOffsetTile, strideTile);
    // Create scatter indices complete row*stride + step
    auto indices = mlir::dyn_cast_or_null<mlir::TypedValue<mlir::VectorType>>(
        rewriter.createOrFold<mlir::arith::AddIOp>(
            loc, indexVecTy, rowStrideTile, stepOffsetTile));
    if (!indices) {
      return rewriter.notifyMatchFailure(initTileOp,
                                         "Could not generate scatter indices.");
    }
    // Add scatter attribute to tile type
    auto scatterTileTy = addScatterAttr(tileTy);
    // Replace InitTileOp
    rewriter.replaceOpWithNewOp<imex::xetile::InitTileOp>(
        initTileOp, scatterTileTy, flatMemref, indices);

    return mlir::success();
  }

private:
  std::shared_ptr<imex::XeuArchInterface> uArchInterface = nullptr;
};

struct LoadTileOpPattern final
    : public mlir::OpRewritePattern<imex::xetile::LoadTileOp> {
  LoadTileOpPattern(mlir::MLIRContext *context)
      : OpRewritePattern<imex::xetile::LoadTileOp>(context) {}
  mlir::LogicalResult
  matchAndRewrite(imex::xetile::LoadTileOp loadTileOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto tile = loadTileOp.getTile();
    auto tileTy = tile.getType();
    if (!tileTy.isScattered()) {
      return mlir::failure();
    }
    auto one = rewriter.createOrFold<mlir::arith::ConstantOp>(
        loadTileOp.getLoc(), rewriter.getI1Type(),
        rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
    auto mask = rewriter.createOrFold<mlir::vector::BroadcastOp>(
        loadTileOp.getLoc(),
        mlir::VectorType::get(tileTy.getShape(), rewriter.getI1Type()), one);
    rewriter.replaceOpWithNewOp<imex::xetile::LoadGatherOp>(
        loadTileOp, loadTileOp.getType(0), tile, mask,
        loadTileOp.getPaddingAttr(), loadTileOp.getL1HintAttr(),
        loadTileOp.getL2HintAttr(), loadTileOp.getL3HintAttr());
    return mlir::success();
  }
};

struct StoreTileOpPattern final
    : public mlir::OpRewritePattern<imex::xetile::StoreTileOp> {
  StoreTileOpPattern(mlir::MLIRContext *context)
      : OpRewritePattern<imex::xetile::StoreTileOp>(context) {}
  mlir::LogicalResult
  matchAndRewrite(imex::xetile::StoreTileOp storeTileOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto tile = storeTileOp.getTile();
    auto tileTy = tile.getType();
    if (!tileTy.isScattered()) {
      return mlir::failure();
    }
    auto one = rewriter.createOrFold<mlir::arith::ConstantOp>(
        storeTileOp.getLoc(), rewriter.getI1Type(),
        rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
    auto mask = rewriter.createOrFold<mlir::vector::BroadcastOp>(
        storeTileOp.getLoc(),
        mlir::VectorType::get(tileTy.getShape(), rewriter.getI1Type()), one);
    rewriter.replaceOpWithNewOp<imex::xetile::StoreScatterOp>(
        storeTileOp, storeTileOp.getValue(), tile, mask,
        storeTileOp.getL1HintAttr(), storeTileOp.getL2HintAttr(),
        storeTileOp.getL3HintAttr());
    return mlir::success();
  }
};

static imex::xetile::InitTileOp
getInitTileOp(mlir::TypedValue<imex::xetile::TileType> tile) {
  // Three sources of tile value
  auto dop = tile.getDefiningOp();
  // 1. BlockArgument of scf.for
  if (!dop) {
    if (!mlir::isa<mlir::BlockArgument>(tile)) {
      return nullptr;
    }
    auto blockArg = llvm::cast<mlir::BlockArgument>(tile);
    auto blockArgNum = blockArg.getArgNumber();
    auto parentOp = blockArg.getOwner()->getParentOp();
    if (!mlir::isa<mlir::scf::ForOp>(parentOp)) {
      return nullptr;
    }
    auto scfForOp = mlir::dyn_cast<mlir::scf::ForOp>(parentOp);
    auto numInductionVars = scfForOp.getNumInductionVars();
    auto init = scfForOp.getInits()[blockArgNum - numInductionVars];
    if (!mlir::isa<mlir::TypedValue<imex::xetile::TileType>>(init)) {
      return nullptr;
    }
    return getInitTileOp(
        mlir::dyn_cast<mlir::TypedValue<imex::xetile::TileType>>(init));
  }
  // 2. InitTileOp
  if (mlir::isa<imex::xetile::InitTileOp>(dop)) {
    return mlir::dyn_cast<imex::xetile::InitTileOp>(dop);
  }
  // 3. UpdateTileOffsetOp
  else if (mlir::isa<imex::xetile::UpdateTileOffsetOp>(dop)) {
    auto updateTileOffsetOp =
        mlir::dyn_cast<imex::xetile::UpdateTileOffsetOp>(dop);
    return getInitTileOp(updateTileOffsetOp.getTile());
  }
  return nullptr;
}

struct UpdateTileOffsetOpPattern final
    : public mlir::OpRewritePattern<imex::xetile::UpdateTileOffsetOp> {
  UpdateTileOffsetOpPattern(mlir::MLIRContext *context)
      : OpRewritePattern<imex::xetile::UpdateTileOffsetOp>(context) {}
  mlir::LogicalResult
  matchAndRewrite(imex::xetile::UpdateTileOffsetOp updateTileOffsetOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto tile = updateTileOffsetOp.getTile();
    auto tileTy = tile.getType();
    if (!tileTy.isScattered()) {
      return mlir::failure();
    }
    // Return if indices are already set
    if (updateTileOffsetOp.getIndices()) {
      return mlir::failure();
    }
    auto initTileOp = getInitTileOp(tile);
    if (!initTileOp) {
      return rewriter.notifyMatchFailure(updateTileOffsetOp,
                                         "Could not find InitTileOp.");
    }
    auto srcMemref = initTileOp.getSource();
    auto castOp = srcMemref.getDefiningOp();
    if (!castOp || !mlir::isa<mlir::memref::ReinterpretCastOp>(castOp)) {
      return rewriter.notifyMatchFailure(updateTileOffsetOp,
                                         "Source is not flat memref.");
    }
    auto reinterOp = mlir::dyn_cast<mlir::memref::ReinterpretCastOp>(castOp);
    auto baseMemref = reinterOp.getSource();
    if (!mlir::isa<mlir::MemRefType>(baseMemref.getType())) {
      return rewriter.notifyMatchFailure(updateTileOffsetOp,
                                         "Source is not a ranked memref.");
    }
    auto baseMemrefTy = mlir::dyn_cast<mlir::MemRefType>(baseMemref.getType());
    auto baseShape = baseMemrefTy.getShape();
    auto pitchNumElems = baseShape[baseShape.size() - 1];
    auto loc = updateTileOffsetOp.getLoc();
    // Create update indices by doing vector.splat with (offX*stride + offY)
    auto pitch =
        mlir::arith::ConstantOp::create(rewriter, loc, rewriter.getIndexType(),
                                        rewriter.getIndexAttr(pitchNumElems));
    auto offX = updateTileOffsetOp.getOffsetX();
    auto stride = rewriter.createOrFold<mlir::arith::MulIOp>(
        loc, rewriter.getIndexType(), offX, pitch);
    auto offY = updateTileOffsetOp.getOffsetY();
    auto index = rewriter.createOrFold<mlir::arith::AddIOp>(
        loc, rewriter.getIndexType(), stride, offY);
    auto indices = rewriter.createOrFold<mlir::vector::BroadcastOp>(
        loc, mlir::VectorType::get(tileTy.getShape(), rewriter.getIndexType()),
        index);
    rewriter.replaceOpWithNewOp<imex::xetile::UpdateTileOffsetOp>(
        updateTileOffsetOp, tile, nullptr, nullptr, indices);
    return mlir::success();
  }
};

struct SCFForOpPattern final : public mlir::OpRewritePattern<mlir::scf::ForOp> {
  SCFForOpPattern(mlir::MLIRContext *context)
      : OpRewritePattern<mlir::scf::ForOp>(context) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp scfForOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto initArgs = scfForOp.getInitArgs();
    auto regionIterArgs = scfForOp.getRegionIterArgs();
    auto results = scfForOp.getResults();
    bool isUpdated = false;
    for (auto [init, arg, res] :
         llvm::zip_equal(initArgs, regionIterArgs, results)) {
      auto initTy = init.getType();
      if (mlir::isa<imex::xetile::TileType>(initTy)) {
        if (!mlir::isa<imex::xetile::TileType>(arg.getType()) ||
            !mlir::isa<imex::xetile::TileType>(res.getType())) {
          return rewriter.notifyMatchFailure(scfForOp, "TileType mismatch.");
        }
        auto initTileTy = mlir::dyn_cast<imex::xetile::TileType>(initTy);
        if (initTileTy.isScattered()) {
          auto argTileTy =
              mlir::dyn_cast<imex::xetile::TileType>(arg.getType());
          if (argTileTy.isScattered()) {
            continue;
          }
          auto scatterTileTy = addScatterAttr(argTileTy);
          arg.setType(scatterTileTy);
          res.setType(scatterTileTy);
          isUpdated = true;
        }
      }
    }
    if (!isUpdated) {
      return mlir::failure();
    }
    return mlir::success();
  }
};

// This function traverses backwards through loop-carried dependencies in SCF
//  `for` loops to find the original (pre-loop) value.
mlir::Value getDefiningInitOp(mlir::Value val) {
  while (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(
            blockArg.getOwner()->getParentOp())) {
      // Get the index of blockArg in the region
      unsigned argIndex = blockArg.getArgNumber();

      // Ensure the block argument belongs to iter_args, not the induction
      // variable
      unsigned numIterArgs = forOp.getInitArgs().size();
      unsigned firstIterArgIdx =
          forOp.getRegion().getArguments().size() - numIterArgs;

      if (argIndex >= firstIterArgIdx) {
        val =
            forOp.getInitArgs()[argIndex - firstIterArgIdx]; // Corrected index
      } else {
        break; // If it's not an iter_arg, stop traversal
      }
    } else {
      break;
    }
  }
  return val;
}

// Helper function to find an InitTileOp that leads to a given mlir::Value
mlir::Operation *findInitializeOp(mlir::Value val) {
  llvm::SmallVector<mlir::Value, 4> worklist{val};
  llvm::DenseSet<mlir::Value> visited;

  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!current || !visited.insert(current).second)
      continue; // Avoid cycles

    // Handle scf.for iter_args
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(current)) {
      current = getDefiningInitOp(current);
    }

    // Check if the defining operation is an InitTileOp
    if (mlir::Operation *defOp = current.getDefiningOp()) {
      if (llvm::isa<imex::xetile::InitTileOp>(defOp))
        return defOp;
      for (mlir::Value operand : defOp->getOperands()) {
        worklist.push_back(operand);
      }
    }
  }
  return nullptr;
}

static void
analyzeAtomicRMWOp(mlir::Operation *op,
                   llvm::DenseSet<mlir::Value> &convertToScatteredType) {

  op->walk([&](imex::xetile::AtomicRMWOp atomicrmwOp) -> mlir::WalkResult {
    auto tileTy = atomicrmwOp.getTile().getType();
    if (tileTy.isScattered()) {
      return mlir::WalkResult::advance();
    }
    mlir::Value tile = atomicrmwOp->getOperand(1);

    mlir::Operation *initializeOp = findInitializeOp(tile);
    if (!initializeOp)
      return mlir::failure();

    // At this point, we have a candidate def-use chain for optimization.
    convertToScatteredType.insert(initializeOp->getResult(0));
    return mlir::WalkResult::advance();
  });
}

class XeTileBlockOpFallbackPass final
    : public imex::impl::XeTileBlockOpFallbackBase<XeTileBlockOpFallbackPass> {
public:
  llvm::DenseSet<mlir::Value> convertToScatteredType;
  XeTileBlockOpFallbackPass() {
    uArchInterface = std::make_shared<imex::XePVCuArch>();
  }

  XeTileBlockOpFallbackPass(const std::string &deviceName) {
    if (deviceName == "pvc") {
      uArchInterface = std::make_shared<imex::XePVCuArch>();
    }
  }

  mlir::LogicalResult
  initializeOptions(mlir::StringRef options,
                    mlir::function_ref<mlir::LogicalResult(const llvm::Twine &)>
                        errorHandler) override {
    if (failed(Pass::initializeOptions(options, errorHandler)))
      return mlir::failure();
    if (device == "pvc")
      uArchInterface = std::make_shared<imex::XePVCuArch>();
    else
      return errorHandler(llvm::Twine("Invalid device: ") + device);
    return mlir::success();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    mlir::Operation *op = getOperation();

    if (!uArchInterface) {
      op->emitOpError("Can not get GPU Arch Definition for given Arch param");
      return signalPassFailure();
    }
    analyzeAtomicRMWOp(op, convertToScatteredType);
    mlir::RewritePatternSet patterns(context);
    mlir::GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);
    config.setUseTopDownTraversal(true);
    config.setStrictness(mlir::GreedyRewriteStrictness::ExistingAndNewOps);
    patterns.add<InitTileOpPattern>(context, uArchInterface,
                                    convertToScatteredType);
    patterns.add<LoadTileOpPattern, StoreTileOpPattern,
                 UpdateTileOffsetOpPattern, SCFForOpPattern>(context);
    if (failed(applyPatternsGreedily(op, std::move(patterns), config))) {
      return signalPassFailure();
    }
  }

private:
  std::shared_ptr<imex::XeuArchInterface> uArchInterface = nullptr;
};

} // namespace blockopfallback

namespace imex {
std::unique_ptr<mlir::Pass>
createXeTileBlockOpFallbackPass(const std::string &deviceName) {
  return std::make_unique<blockopfallback::XeTileBlockOpFallbackPass>(
      deviceName);
}
} // namespace imex
