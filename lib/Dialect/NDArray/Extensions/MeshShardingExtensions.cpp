//===- MeshShardingExtensions.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "imex/Dialect/NDArray/IR/NDArrayOps.h"
#include "imex/Dialect/NDArray/Transforms/Utils.h"
#include "mlir/Dialect/Shard/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Shard/Interfaces/ShardingInterfaceImpl.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/Support/Debug.h"
#include <sstream>
#include <string>
#include <vector>

#define DEBUG_TYPE "ndarray-sharding-impl"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::shard;
using imex::easyI64;
using imex::easyIdx;

namespace imex {
namespace ndarray {

// Converts a comma-separated string of integers into a std::vector<int>.
static std::vector<int> convertStringToVector(const std::string &str) {
  std::vector<int> result;
  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, ',')) {
    result.push_back(std::stoi(item));
  }
  return result;
}

// Retrieves the multi-index for the current process in a mesh.
// If the environment variable "DEBUG_MESH_INDEX" is set, it uses the value
// from the environment variable. Otherwise, it creates a ProcessMultiIndexOp
// to get the index.
static SmallVector<Value> getMyMultiIndex(OpBuilder &b, ::GridOp mesh,
                                          bool asI64 = false) {
  if (auto envStr = getenv("DEBUG_MESH_INDEX")) {
    auto myIdx = convertStringToVector(envStr);
    if (myIdx.size() < mesh.getShape().size()) {
      mesh->emitError() << "DEBUG_MESH_INDEX has wrong size";
      return {};
    }
    SmallVector<Value> idxs;
    for (auto i : myIdx) {
      if (asI64)
        idxs.push_back(easyI64(mesh.getLoc(), b, i).get());
      else
        idxs.push_back(easyIdx(mesh.getLoc(), b, i).get());
      if (idxs.size() == mesh.getShape().size())
        break;
    }
    return idxs;
  }
  SmallVector<Value> res =
      ProcessMultiIndexOp::create(b, mesh.getLoc(), mesh).getResult();
  if (asI64) {
    for (auto &v : res) {
      v = createCast(mesh->getLoc(), b, v, b.getI64Type());
    }
  }
  return res;
}

template <typename T>
T getBaseShardDimSize(T shard, T numShards, T extend, T one, T zero) {
  return extend / numShards +
         shard.sge(numShards - (extend % numShards)).select(one, zero);
}

template <typename T,
          typename std::enable_if<std::is_integral<T>::value>::type * = nullptr>
T getBaseShardDimSize(T shard, T numShards, T extend) {
  return extend / numShards + (shard >= numShards - (extend % numShards)
                                   ? static_cast<T>(1)
                                   : static_cast<T>(0));
}

template <typename T>
static T getBaseShardDimOff(T shard, T numShards, T extend, T zero) {
  return (shard * (extend / numShards)) +
         (shard - (numShards - (extend % numShards))).max(zero);
}

template <typename T,
          typename std::enable_if<std::is_integral<T>::value>::type * = nullptr>
static T getBaseShardDimOff(T shard, T numShards, T extend) {
  return (shard * (extend / numShards)) +
         std::max(static_cast<T>(0),
                  shard - (numShards - (extend % numShards)));
}

static Sharding ShardingFromOption(const ShardingOption &option,
                                   MLIRContext *ctxt) {
  SmallVector<GridAxesAttr> res;
  for (const auto &v : option.shardingArray) {
    res.emplace_back(GridAxesAttr::get(ctxt, v));
  }
  return Sharding::get(option.grid, res);
}

//===----------------------------------------------------------------------===//
// helpers for ops with offsets/sizes/strides
//===----------------------------------------------------------------------===//

// Computes result sharding by extending a copy of the input sharding with
// shard sizes. The shard sizes reflect the sizes resulting from the
// non-copying subview operation. ShardSizes are represented as relative
// offsets to the previous shard.
// Requires sharding of input tensor.
static FailureOr<Sharding>
getShardingWithShardedDimsOffs(Value ary, OffsetSizeAndStrideOpInterface op) {
  SymbolTableCollection symbolTable;
  auto aryType = cast<RankedTensorType>(ary.getType());
  // currently no support for dynamic input shapes
  if (!aryType.hasStaticShape())
    return op->emitOpError("Dynamic shapes are not supported.");
  auto aryShape = aryType.getShape();
  auto rank = cast<RankedTensorType>(ary.getType()).getRank();

  auto aryShardOp = ary.getDefiningOp<shard::ShardOp>();
  // currently no support for non-sharded source
  if (!aryShardOp)
    return op->emitOpError("Exptected a ShardOp on input, got ")
           << ary.getDefiningOp();

  auto offs = op.getStaticOffsets();
  auto sizes = op.getStaticSizes();
  auto strides = op.getStaticStrides();
  // currently no support for dynamic subviews
  if (ShapedType::isDynamicShape(offs) || ShapedType::isDynamicShape(sizes) ||
      ShapedType::isDynamicShape(strides))
    return op->emitOpError("Dynamic offsets/sizes/strides are not supported");

  auto arySharding =
      aryShardOp.getSharding().getDefiningOp<shard::ShardingOp>();
  // currently no support for sharding dims sizes on input
  if (!arySharding.getStaticShardedDimsOffsets().empty())
    return op->emitOpError(
        "Sharded dims sizes on input are not supported yet.");

  auto mesh = getGrid(arySharding, symbolTable);
  if (!mesh)
    return op->emitOpError("Invalid mesh.");
  auto meshShape = mesh.getShape();
  // currently no support for dynamic mesh shape
  if (ShapedType::isDynamicShape(meshShape))
    return op->emitOpError("Dynamic mesh shape is not supported.");

  auto splitAxes = arySharding.getSplitAxes();
  assert((int64_t)splitAxes.size() <= rank);

  // flattened shard offsets for each dimension (see
  // sharding.sharded_dims_offsets) after subview
  SmallVector<int64_t> splitOffs;
  // iterate split tensor dimensions
  for (auto dim = 0u; dim < splitAxes.size(); ++dim) {
    auto axes = arySharding.getSplitAxes().getAxes()[dim].asArrayRef();
    if (axes.empty())
      continue;
    splitOffs.emplace_back(0);
    int64_t splitSz = 1; // number of shards in this dimension
    for (auto i : axes)
      splitSz *= meshShape[i];
    int64_t pos = offs[dim]; // current position in split tensor dim
    int64_t mx = sizes[dim]; // max #elements we assign to current shard
    for (int64_t shard = 0; shard < splitSz; ++shard) {
      // extract size of overlap of subview with current input shard
      auto shardStart = getBaseShardDimOff(shard, splitSz, aryShape[dim]);
      auto shardSz = getBaseShardDimSize(shard, splitSz, aryShape[dim]);
      auto shardEnd = shardStart + shardSz;
      auto num = shardEnd - pos;
      int64_t sz = 0;
      if (num > 0) { // if starts before end of shard
        sz = (num + (strides[dim] - 1)) / strides[dim];
        sz = std::min(mx, sz);
      }
      splitOffs.emplace_back((shard ? splitOffs.back() : 0) + sz);
      // update pos and max for next result shard
      pos += sz * strides[dim];
      mx -= sz;
    }
  }

  return Sharding::get(arySharding.getGridAttr(),
                       arySharding.getSplitAxes().getAxes(), {}, // static halo
                       splitOffs, {}, {});
}

static std::pair<Value, Value>
getOffsetAndSize(const EasyI64 &myID, const EasyI64 &zero, const EasyI64 &one,
                 Value targetOffs, int64_t currPos, OpBuilder &builder,
                 Location loc) {
  auto pos = easyI64(loc, builder, currPos);
  auto eMyPos = pos + myID;
  auto myPos = createIndexCast(loc, builder, eMyPos.get());
  auto myPos1 = createIndexCast(loc, builder, (eMyPos + one).get());
  auto myOff = easyI64(
      loc, builder,
      tensor::ExtractOp::create(builder, loc, targetOffs, myPos).getResult());
  auto nextOff = easyI64(
      loc, builder,
      tensor::ExtractOp::create(builder, loc, targetOffs, myPos1).getResult());
  return {myOff.get(), (nextOff - myOff).get()};
}

// ***************************************************************************
static std::array<Value, 2> getShardSliceOffAndSz(
    ValueRange myIdx, int64_t dim, ArrayRef<int64_t> meshShape,
    ArrayRef<GridAxesAttr> splitAxes, Value targetOffs,
    ArrayRef<int64_t> srcShape, const SmallVector<OpFoldResult> &slcOffs,
    const SmallVector<OpFoldResult> &slcSizes,
    const SmallVector<OpFoldResult> &slcStrides,
    const SmallVector<OpFoldResult> &haloSizes, const EasyI64 &zero,
    const EasyI64 &one, OpBuilder &builder, Location loc) {
  assert(splitAxes[dim].size() == 1);
  int64_t currPos = 0, haloDim = 0;
  for (auto i = 0; i < dim; ++i) {
    if (!splitAxes[i].empty()) {
      currPos += meshShape[splitAxes[i][0]] + 1;
      ++haloDim;
    }
  }

  auto extend = easyI64(loc, builder, srcShape[dim]);
  auto meshAxis = splitAxes[dim][0];
  auto numShards = easyI64(loc, builder, meshShape[meshAxis]);
  auto myID = easyI64(loc, builder, myIdx[meshAxis]);
  auto myOff_ = getBaseShardDimOff(myID, numShards, extend, zero);

  Value resOff, resSize;
  if (targetOffs) {
    std::tie(resOff, resSize) =
        getOffsetAndSize(myID, zero, one, targetOffs, currPos, builder, loc);
  } else {
    auto slcSz = easyI64(loc, builder, slcSizes[dim]);
    resSize = getBaseShardDimSize(myID, numShards, slcSz, one, zero).get();
    resOff = getBaseShardDimOff(myID, numShards, slcSz, zero).get();
  }

  // The global offset of the local shard is slice offset plus the computed
  // offset in the target tensor. The latter is in number of elements after
  // slicing, which means we need to multiply it by stride
  auto targetOff =
      easyI64(loc, builder, slcOffs[dim]) +
      easyI64(loc, builder, resOff) * easyI64(loc, builder, slcStrides[dim]);
  auto myShardOff = myOff_ - easyI64(loc, builder, haloSizes[haloDim * 2]);
  // Convert global to local indices. If size is <= 0 off is always set to 0.
  auto localOff = easyI64(loc, builder, resSize)
                      .sgt(zero)
                      .select(targetOff - myShardOff, zero);

  return {createIndexCast(loc, builder, localOff.get()),
          createIndexCast(loc, builder, resSize)};
}

// ***************************************************************************
template <typename OP>
FailureOr<std::tuple<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>,
                     SmallVector<OpFoldResult>>>
getLocalOffSzAndStrFromSlice(OP op, ArrayRef<int64_t> srcShape,
                             const Sharding &haloSharding,
                             const Sharding &offsSharding,
                             const Sharding &splitSharding,
                             SymbolTableCollection &symbolTableCollection,
                             OpBuilder &builder) {

  if (!haloSharding.getStaticShardedDimsOffsets().empty() ||
      (!haloSharding.getStaticHaloSizes().empty() &&
       ShapedType::isDynamicShape(haloSharding.getStaticHaloSizes())) ||
      (!offsSharding.getStaticShardedDimsOffsets().empty() &&
       ShapedType::isDynamicShape(
           offsSharding.getStaticShardedDimsOffsets()))) {
    return op->emitOpError("Dynamic sharding dims offsets or halo sizes are "
                           "not supported yet.");
  }

  auto slcOffs =
      mlir::getMixedValues(op.getStaticOffsets(), op.getOffsets(), builder);
  auto slcSizes =
      mlir::getMixedValues(op.getStaticSizes(), op.getSizes(), builder);
  auto slcStrides =
      mlir::getMixedValues(op.getStaticStrides(), op.getStrides(), builder);

  auto loc = op->getLoc();
  auto rank = slcOffs.size();
  auto splitAxes = splitSharding.getSplitAxes();
  auto mesh = getGrid(op, offsSharding.getGridAttr(), symbolTableCollection);
  auto myIdx = getMyMultiIndex(builder, mesh);

  auto haloSizes =
      haloSharding.getStaticHaloSizes().empty()
          ? SmallVector<OpFoldResult>(rank * 2, builder.getI64IntegerAttr(0))
          : mlir::getMixedValues(haloSharding.getStaticHaloSizes(),
                                 haloSharding.getDynamicHaloSizes(), builder);

  Value targetOffs;
  if (!offsSharding.getStaticShardedDimsOffsets().empty()) {
    auto shardedDimsOffsets = imex::getMixedAsValues(
        loc, builder, offsSharding.getDynamicShardedDimsOffsets(),
        offsSharding.getStaticShardedDimsOffsets());
    targetOffs =
        tensor::FromElementsOp::create(builder, loc, shardedDimsOffsets);
  }

  auto zero = easyI64(loc, builder, 0);
  auto one = easyI64(loc, builder, 1);
  SmallVector<OpFoldResult> lShardOffs, lShardSizes;
  for (auto dim = 0ul; dim < (uint64_t)rank; ++dim) {
    assert(!ShapedType::isDynamic(srcShape[dim]));
    if (dim >= splitAxes.size() || splitAxes[dim].empty()) {
      lShardOffs.emplace_back(slcOffs[dim]);
      lShardSizes.emplace_back(slcSizes[dim]);
    } else {
      auto offAndSz = getShardSliceOffAndSz(
          myIdx, dim, mesh.getShape(), splitAxes, targetOffs, srcShape, slcOffs,
          slcSizes, slcStrides, haloSizes, zero, one, builder, loc);
      lShardOffs.emplace_back(offAndSz[0]);
      lShardSizes.emplace_back(offAndSz[1]);
    }
  }
  return std::make_tuple(lShardOffs, lShardSizes, slcStrides);
}

namespace {

//===----------------------------------------------------------------------===//
// BaseShardingInterface
//===----------------------------------------------------------------------===//

// Sharding of tensor.empty
template <typename T, typename OpType>
struct BaseShardingInterface
    : public ShardingInterface::ExternalModel<T, OpType> {

  std::pair<int, int64_t> getNumTensorsAndRank(::mlir::Operation *op) const {
    int numTensors = 0;
    int64_t rank = -1;
    for (auto o : op->getOperands()) {
      if (auto type = dyn_cast<RankedTensorType>(o.getType())) {
        assert(rank < 0 || type.getRank() == 0 || type.getRank() == rank);
        rank = std::max(rank, type.getRank());
        numTensors++;
      }
    }
    for (auto o : op->getResults()) {
      if (auto type = dyn_cast<RankedTensorType>(o.getType())) {
        assert(rank < 0 || type.getRank() == 0 || type.getRank() == rank);
        rank = std::max(rank, type.getRank());
        numTensors++;
      }
    }
    return {numTensors, rank};
  }

  SmallVector<mlir::utils::IteratorType>
  getLoopIteratorTypes(::mlir::Operation *op) const {
    LLVM_DEBUG(DBGS() << "getLoopIteratorTypes\n");
    auto [numTensors, rank] = getNumTensorsAndRank(op);
    if (numTensors == 0)
      return {};
    SmallVector<utils::IteratorType> types(rank, utils::IteratorType::parallel);
    return types;
  }

  SmallVector<AffineMap> getIndexingMaps(::mlir::Operation *op) const {
    LLVM_DEBUG(DBGS() << "getIndexingMaps\n");
    MLIRContext *ctx = op->getContext();
    SmallVector<AffineMap> maps;
    for (auto o : op->getOperands()) {
      auto type = dyn_cast<RankedTensorType>(o.getType());
      auto rank = type ? type.getRank() : 0;
      maps.emplace_back(AffineMap::getMultiDimIdentityMap(rank, ctx));
    }
    for (auto r : op->getResults()) {
      auto type = dyn_cast<RankedTensorType>(r.getType());
      auto rank = type ? type.getRank() : 0;
      maps.emplace_back(AffineMap::getMultiDimIdentityMap(rank, ctx));
    }
    return maps;
  }
};

//===----------------------------------------------------------------------===//
// SubviewShardingInterface
//===----------------------------------------------------------------------===//

struct SubviewShardingInterface
    : public BaseShardingInterface<SubviewShardingInterface,
                                   imex::ndarray::SubviewOp> {
  LogicalResult
  addShardingAnnotations(::mlir::Operation *op, OpBuilder &b,
                         const ShardingOption &shardingOption) const {
    auto svop = cast<SubviewOp>(op);
    auto srcShardOp = svop.getSource().getDefiningOp<shard::ShardOp>();
    Sharding srcSharding;
    if (srcShardOp) {
      srcSharding = srcShardOp.getSharding();
    } else {
      LLVM_DEBUG(DBGS() << "no sharding on input, using default\n");
      srcSharding =
          ShardingFromOption(shardingOption, srcShardOp->getContext());
    }
    maybeInsertSourceShardingAnnotation(srcSharding, op->getOpOperand(0), b);

    auto sharding = getShardingWithShardedDimsOffs(svop.getSource(), svop);
    if (failed(sharding))
      return failure();
    maybeInsertTargetShardingAnnotation(sharding.value(), op->getResult(0), b);

    return success();
  }

  LogicalResult spmdize(::mlir::Operation *op, ArrayRef<Value> spmdizedOperands,
                        ArrayRef<Sharding> operandShardings,
                        ArrayRef<Sharding> resultShardings,
                        IRMapping &spmdizationMap,
                        SymbolTableCollection &symbolTableCollection,
                        OpBuilder &builder) const {
    if (resultShardings.size() != 1) {
      return failure();
    }
    auto typedOp = cast<imex::ndarray::SubviewOp>(op);
    auto shp = cast<RankedTensorType>(typedOp.getSource().getType()).getShape();
    auto offSzStr = getLocalOffSzAndStrFromSlice(
        typedOp, shp, operandShardings[0], resultShardings[0],
        operandShardings[0], symbolTableCollection, builder);
    if (failed(offSzStr)) {
      return failure();
    }
    auto &[lShardOffs, lShardSizes, lShardStrides] = offSzStr.value();
    auto newSubview = imex::ndarray::SubviewOp::create(builder,
        op->getLoc(), spmdizedOperands[0], lShardOffs, lShardSizes,
        lShardStrides);
    spmdizationMap.map(op->getResult(0), newSubview.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// InsertSliceShardingInterface
//===----------------------------------------------------------------------===//

struct InsertSliceShardingInterface
    : public BaseShardingInterface<InsertSliceShardingInterface,
                                   imex::ndarray::InsertSliceOp> {
  LogicalResult
  addShardingAnnotations(::mlir::Operation *op, OpBuilder &b,
                         const ShardingOption &shardingOption) const {
    LLVM_DEBUG(DBGS() << "addShardingAnnotations\n");
    auto svop = cast<InsertSliceOp>(op);
    Sharding srcSharding(shardingOption.grid);
    auto srcRank = svop.getSource().getType().getRank();

    if (srcRank > 0) {
      auto sharding =
          getShardingWithShardedDimsOffs(svop.getDestination(), svop);
      if (failed(sharding))
        return failure();
      srcSharding = sharding.value();
    }
    maybeInsertSourceShardingAnnotation(srcSharding, op->getOpOperand(1), b);

    auto dstSharding = ShardingFromOption(shardingOption, op->getContext());
    maybeInsertSourceShardingAnnotation(dstSharding, op->getOpOperand(0), b);
    maybeInsertTargetShardingAnnotation(dstSharding, op->getResult(0), b);

    return success();
  }

  LogicalResult spmdize(::mlir::Operation *op, ArrayRef<Value> spmdizedOperands,
                        ArrayRef<Sharding> operandShardings,
                        ArrayRef<Sharding> resultShardings,
                        IRMapping &spmdizationMap,
                        SymbolTableCollection &symbolTableCollection,
                        OpBuilder &builder) const {
    if (resultShardings.size() != 1 || operandShardings.size() < 2 ||
        resultShardings[0] != operandShardings[0]) {
      return op->emitOpError("incorrect sharding annotations");
    }

    auto typedOp = cast<imex::ndarray::InsertSliceOp>(op);
    auto dstSharding = operandShardings[0];
    auto srcSharding = operandShardings[1];
    SmallVector<OpFoldResult> lShardOffs, lShardSizes, lShardStrides;

    if (isFullReplication(dstSharding)) {
      lShardOffs = mlir::getMixedValues(typedOp.getStaticOffsets(),
                                        typedOp.getOffsets(), builder);
      lShardSizes = mlir::getMixedValues(typedOp.getStaticSizes(),
                                         typedOp.getSizes(), builder);
      lShardStrides = mlir::getMixedValues(typedOp.getStaticStrides(),
                                           typedOp.getStrides(), builder);
    } else {
      if (typedOp.getSource().getType().getRank() == 0) {
        auto sharding =
            getShardingWithShardedDimsOffs(typedOp.getDestination(), typedOp);
        if (failed(sharding)) {
          return failure();
        }
        srcSharding = sharding.value();
      }
      auto shp =
          cast<RankedTensorType>(typedOp.getDestination().getType()).getShape();
      auto offSzStr = getLocalOffSzAndStrFromSlice(
          typedOp, shp, dstSharding, srcSharding, dstSharding,
          symbolTableCollection, builder);
      if (failed(offSzStr)) {
        return failure();
      }
      std::tie(lShardOffs, lShardSizes, lShardStrides) = offSzStr.value();
    }

    auto loc = op->getLoc();
    auto zero = easyI64(loc, builder, 0);
    auto hasSize = zero.eq(zero);
    for (auto &v : lShardSizes) {
      hasSize = hasSize.land(easyI64(loc, builder, v).sgt(zero));
    }

    scf::IfOp ifOp = scf::IfOp::create(builder,
        loc, hasSize.get(),
        [&](OpBuilder &b, Location loc) {
          auto res = imex::ndarray::InsertSliceOp::create(b,
              loc, spmdizedOperands[0], spmdizedOperands[1], lShardOffs,
              lShardSizes, lShardStrides);
          scf::YieldOp::create(b,loc, res.getResult());
        },
        [&](OpBuilder &b, Location loc) {
          scf::YieldOp::create(b,loc, spmdizedOperands[0]);
        });

    auto res = UpdateHaloOp::create(builder,
        loc, spmdizedOperands[0].getType(), ifOp.getResult(0),
        dstSharding.getGridAttr(),
        GridAxesArrayAttr::get(op->getContext(), dstSharding.getSplitAxes()),
        dstSharding.getDynamicHaloSizes(),
        DenseI64ArrayAttr::get(op->getContext(),
                               dstSharding.getStaticHaloSizes()));

    spmdizationMap.map(op->getResult(0), res->getResult(0));
    spmdizationMap.map(op, res.getOperation());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// LinspaceShardingInterface
//===----------------------------------------------------------------------===//

struct LinspaceShardingInterface
    : public BaseShardingInterface<LinspaceShardingInterface, LinSpaceOp> {
  LogicalResult
  addShardingAnnotations(::mlir::Operation *op, OpBuilder &b,
                         const ShardingOption &shardingOption) const {
    auto sharding = ShardingFromOption(shardingOption, op->getContext());
    maybeInsertTargetShardingAnnotation(sharding, op->getResult(0), b);

    return success();
  }

  LogicalResult spmdize(::mlir::Operation *op, ArrayRef<Value> spmdizedOperands,
                        ArrayRef<Sharding> operandShardings,
                        ArrayRef<Sharding> resultShardings,
                        IRMapping &spmdizationMap,
                        SymbolTableCollection &symbolTableCollection,
                        OpBuilder &builder) const {
    if (resultShardings.size() != 1) {
      return failure();
    }

    auto lsop = cast<LinSpaceOp>(op);
    auto loc = lsop.getLoc();
    auto retArType = cast<RankedTensorType>(lsop.getType());
    auto sharding = resultShardings[0];
    Value start = spmdizedOperands[0];
    Value stop = spmdizedOperands[1];
    Value count = spmdizedOperands[2];

    if (!(start.getType().isIntOrIndexOrFloat() &&
          stop.getType().isIntOrIndexOrFloat() &&
          count.getType().isIntOrIndex() && retArType &&
          sharding.getSplitAxes().size() == 1)) {
      return ::mlir::failure();
    } // FIXME type promotion

    auto splitAxes = sharding.getSplitAxes()[0];
    SymbolTableCollection symbolTable;
    auto mesh = getGrid(op, sharding.getGridAttr(), symbolTable);

    // get number of procs to distribute linspace
    auto nProcs = collectiveProcessGroupSize(splitAxes.asArrayRef(), mesh);
    if (ShapedType::isDynamic(nProcs)) {
      return failure();
    }

    auto zero = easyI64(loc, builder, 0);
    auto one = easyI64(loc, builder, 1);

    // pRank is the canonicalized index of the local process along the split
    // axes (only). Notice: this is not the same as shard::ProcessLinearIndexOp
    //         because the latter includes replication axes.
    auto myMIdx = getMyMultiIndex(builder, mesh, true);
    auto tileSz = one;
    auto pRank = zero;
    for (int64_t i = (int64_t)splitAxes.size() - 1; i >= 0; --i) {
      auto idx = easyI64(loc, builder, myMIdx[i]);
      pRank = pRank + (idx * tileSz);
      if (i > 0) {
        tileSz = tileSz * easyI64(loc, builder, mesh.getShape()[splitAxes[i]]);
      }
    }

    // cast types and get step
    auto elTyp = retArType.getElementType();
    auto bw = elTyp.isIndex() ? 64 : elTyp.getIntOrFloatBitWidth();
    ::mlir::Type cType =
        bw > 32 ? builder.getF64Type()
                : (bw > 16 ? builder.getF32Type() : builder.getF16Type());

    // Get local shape and offset
    // Check if offsets spec is present in the sharding.
    // If not, use the default offset computation and
    // add halo if present in the sharding.
    Value off, lSz; // placeholder for offset, local size
    auto i64Type = builder.getI64Type();
    count = createCast(loc, builder, count, i64Type);
    if (sharding.getStaticShardedDimsOffsets().empty()) {
      auto nShards = easyI64(loc, builder, nProcs);
      auto extend = easyI64(loc, builder, count);
      auto eOff = getBaseShardDimOff(pRank, nShards, extend, zero);
      auto eSz = getBaseShardDimSize(pRank, nShards, extend, one, zero);
      if (!sharding.getStaticHaloSizes().empty()) {
        auto haloSizes =
            getMixedAsValues(loc, builder, sharding.getDynamicHaloSizes(),
                             sharding.getStaticHaloSizes(), true);
        auto h0 = easyI64(loc, builder, haloSizes[0]);
        auto h1 = easyI64(loc, builder, haloSizes[1]);
        eOff = eOff - h0;
        eSz = eSz + h0 + h1;
      }
      off = eOff.get();
      lSz = eSz.get();
    } else {
      auto shardedDimsOffsets = getMixedAsValues(
          loc, builder, sharding.getDynamicShardedDimsOffsets(),
          sharding.getStaticShardedDimsOffsets(), true);
      auto targetOffs =
          tensor::FromElementsOp::create(builder, loc, shardedDimsOffsets);
      auto myOffAndSize =
          getOffsetAndSize(pRank, zero, one, targetOffs, 0, builder, loc);
      off = myOffAndSize.first;
      lSz = myOffAndSize.second;
    }

    // use local shape and offset to compute local linspace
    off = createCast(loc, builder, off, cType);
    lSz = createCast(loc, builder, lSz, cType);
    start = createCast(loc, builder, start, cType);
    stop = createCast(loc, builder, stop, cType);
    auto step = createStepLinSpace(builder, loc, start, stop, count,
                                   lsop.getEndpoint(), cType);

    start = builder.createOrFold<::mlir::arith::AddFOp>(
        loc, builder.createOrFold<::mlir::arith::MulFOp>(loc, step, off),
        start);
    stop = builder.createOrFold<::mlir::arith::AddFOp>(
        loc, builder.createOrFold<::mlir::arith::MulFOp>(loc, step, lSz),
        start);

    // finally create local linspace
    auto retType = RankedTensorType::get({ShapedType::kDynamic}, elTyp,
                                         retArType.getEncoding());
    lSz = createCast(loc, builder, lSz, builder.getIndexType());
    auto res = ::imex::ndarray::LinSpaceOp::create(builder, loc, retType, start,
                                                           stop, lSz, false);
    // update mapping
    spmdizationMap.map(op->getResult(0), res->getResult(0));
    spmdizationMap.map(op, res.getOperation());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ReshapeShardingInterface
//===----------------------------------------------------------------------===//

struct ReshapeShardingInterface
    : public BaseShardingInterface<ReshapeShardingInterface, ReshapeOp> {

  SmallVector<mlir::utils::IteratorType>
  getLoopIteratorTypes(::mlir::Operation *op) const {
    LLVM_DEBUG(DBGS() << "ReshapeShardingInterface::getLoopIteratorTypes\n");
    assert(op->getNumResults() == 1);
    auto rank = dyn_cast<ShapedType>(op->getResultTypes()[0]).getRank();
    return SmallVector<utils::IteratorType>(rank,
                                            utils::IteratorType::parallel);
  }

  // Currently only replicated input and output sharding is supported.
  // This covers cases where 0d tensors get reshaped to higher ranks.

  // Add replication sharding for input and result.
  LogicalResult
  addShardingAnnotations(::mlir::Operation *op, OpBuilder &b,
                         const ShardingOption &shardingOption) const {
    // if (shardingOption.shardingArray.size() > 0)
    //   return op->emitOpError("Only full replication is implemented.");

    // auto sharding = ShardingFromOption(shardingOption, op->getContext());
    Sharding sharding = Sharding::get(shardingOption.grid, {});
    maybeInsertSourceShardingAnnotation(sharding, op->getOpOperand(0), b);
    maybeInsertTargetShardingAnnotation(sharding, op->getResult(0), b);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Interface registration
//===----------------------------------------------------------------------===//

template <typename T1, typename... T> void registerTrivial(MLIRContext *ctx) {
  T1::template attachInterface<ElementwiseShardingInterface<T1>>(*ctx);
  if constexpr (sizeof...(T) > 0)
    registerTrivial<T...>(ctx);
}

void registerShardingInterfaceExternalModels(mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, imex::ndarray::NDArrayDialect *dialect) {
        SubviewOp::attachInterface<SubviewShardingInterface>(*ctx);
        InsertSliceOp::attachInterface<InsertSliceShardingInterface>(*ctx);
        LinSpaceOp::attachInterface<LinspaceShardingInterface>(*ctx);
        ReshapeOp::attachInterface<ReshapeShardingInterface>(*ctx);
        registerTrivial<CopyOp, DeleteOp, CastElemTypeOp>(ctx);
      });
}

} // namespace ndarray
} // namespace imex
