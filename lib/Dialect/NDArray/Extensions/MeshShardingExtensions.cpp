//===- ShardingInterfaceImpl.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Mesh/Interfaces/ShardingInterfaceImpl.h"
#include "imex/Dialect/NDArray/IR/NDArrayOps.h"
#include "imex/Dialect/NDArray/IR/NDArrayOps.h"
#include "imex/Dialect/Dist/IR/DistOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/Support/Debug.h"
#include <vector>
#include <string>
#include <sstream>

#define DEBUG_TYPE "ndarray-sharding-impl"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")
// #define DBGS() (std::cerr << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::mesh;
using imex::easyIdx;
using imex::easyI64;

namespace imex {
namespace ndarray {
namespace {

static std::vector<int> convertStringToVector(const std::string &str) {
  std::vector<int> result;
  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, ',')) {
    result.push_back(std::stoi(item));
  }
  return result;
}

static SmallVector<Value> getMyMultiIndex(OpBuilder &b,
                                          ::mlir::mesh::MeshOp mesh) {
  if (auto envStr = getenv("DEBUG_MESH_INDEX")) {
    auto myIdx = convertStringToVector(envStr);
    if (myIdx.size() != mesh.getShape().size()) {
      mesh->emitError() << "DEBUG_MESH_INDEX has wrong size";
      return {};
    }
    SmallVector<Value> idxs;
    for (auto i : myIdx) {
      idxs.push_back(easyIdx(mesh.getLoc(), b, i).get());
    }
    return idxs;
  }
  return b.create<ProcessMultiIndexOp>(mesh.getLoc(), mesh).getResult();
}

// Sharding of tensor.empty
template<typename T, typename OpType>
struct OffsetSizeAndStrideShardingInterface
    :  public ShardingInterface::ExternalModel<T, OpType> {
  
  SmallVector<mlir::utils::IteratorType> getLoopIteratorTypes(::mlir::Operation *op) const {
    LLVM_DEBUG(DBGS() << "getLoopIteratorTypes\n");
    Value val = op->getOperand(0);
    auto type = dyn_cast<RankedTensorType>(val.getType());
    if (!type)
      return {};
    SmallVector<utils::IteratorType> types(type.getRank(),
                                          utils::IteratorType::parallel);
    return types;
  }
  
  SmallVector<AffineMap> getIndexingMaps(::mlir::Operation *op) const {
    LLVM_DEBUG(DBGS() << "getIndexingMaps\n");
    MLIRContext *ctx = op->getContext();
    Value val = op->getOperand(0);
    auto type = dyn_cast<RankedTensorType>(val.getType());
    if (!type)
      return {};
    int64_t rank = type.getRank();
    int64_t num = op->getNumOperands() + op->getNumResults();
    SmallVector<AffineMap> maps(num,
                                AffineMap::getMultiDimIdentityMap(rank, ctx));
    return maps;
  }

protected:
  // Computes result sharding by extending a copy of the input sharding with
  // shard sizes. The shard sizes reflect the sizes resulting from the
  // non-copying subview operation. Requires sharding of input tensor.
  FailureOr<MeshSharding>
  getShardedDimsOffsetsSharding(Value ary,
                                OffsetSizeAndStrideOpInterface op) const {
    SymbolTableCollection symbolTable;
    auto aryType = cast<RankedTensorType>(ary.getType());
    // currently no support for dynamic input shapes
    if (!aryType.hasStaticShape())
      return failure();
    auto aryShape = aryType.getShape();
    auto rank = cast<RankedTensorType>(ary.getType()).getRank();

    auto aryShardOp = ary.getDefiningOp<mesh::ShardOp>();
    // currently no support for non-sharded source
    if (!aryShardOp)
      return failure();

    auto offs = op.getStaticOffsets();
    auto sizes = op.getStaticSizes();
    auto strides = op.getStaticStrides();
    // currently no support for dynamic subviews
    if (ShapedType::isDynamicShape(offs) || ShapedType::isDynamicShape(sizes) ||
        ShapedType::isDynamicShape(strides))
      return failure();

    auto arySharding =
        aryShardOp.getSharding().getDefiningOp<mesh::ShardingOp>();
    // currently no support for sharding dims sizes on input
    if (!arySharding.getStaticShardedDimsOffsets().empty())
      return failure();

    auto mesh = getMesh(arySharding, symbolTable);
    if (!mesh)
      return failure();
    auto meshShape = mesh.getShape();
    // currently no support for dynamic mesh shape
    if (llvm::any_of(meshShape,
                     [](int64_t dim) { return dim == ShapedType::kDynamic; }))
      return failure();

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
      int64_t shardSz = shardDimension(aryShape[dim], splitSz);
      int64_t pos = offs[dim]; // current position in split tensor dim
      int64_t mx =
          sizes[dim]; // max number of elements we assign to current shard
      for (auto shard = 0; shard < splitSz; ++shard) {
        // extract size of overlap of subview with current input shard
        auto shardStart = shard * shardSz;
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
    return MeshSharding::get(
        arySharding.getMeshAttr(), arySharding.getSplitAxes().getAxes(),
        arySharding.getPartialAxes().value_or(llvm::ArrayRef<MeshAxis>{}),
        arySharding.getPartialType().value_or(ReductionKind::Sum),
        {}, // static halo
        splitOffs, {}, {});
  }

  std::array<Value, 2> getShardSliceOffAndSz(
      ValueRange myIdx, int64_t dim, ArrayRef<int64_t> meshShape,
      ArrayRef<MeshAxesAttr> splitAxes, Value targetOffs,
      ArrayRef<int64_t> srcShape, const SmallVector<OpFoldResult> &slcOffs,
      const SmallVector<OpFoldResult> &slcStrides,
      const SmallVector<OpFoldResult> &haloSizes, const EasyIdx &zero,
      const EasyIdx &one, OpBuilder &builder, Location loc) const {
    assert(splitAxes[dim].size() == 1);
    int64_t currPos = 0, shardedDim = 0;
    for (auto i = 0; i < dim; ++i) {
      if (!splitAxes[i].empty()) {
        currPos += meshShape[splitAxes[i][0]] + 1;
        ++shardedDim;
      }
    }

    auto extend = easyIdx(loc, builder, srcShape[dim]);
    auto meshAxis = splitAxes[dim][0];
    auto numShards = easyIdx(loc, builder, meshShape[meshAxis]);
    auto myID = easyIdx(loc, builder, myIdx[meshAxis]);
    auto pos = easyIdx(loc, builder, currPos);
    auto myPos = pos + myID;

    // compute offset of our target shard by summing sizes of "previous" shards
    // (for current dim) the result is in number of elements after slicing, e.g.
    // it does not include stride
    auto nextOff = easyIdx(
        loc, builder,
        builder.create<tensor::ExtractOp>(loc, targetOffs, (myPos + one).get())
            .getResult());
    auto myOffAndSize = builder.create<::mlir::scf::IfOp>(
        loc, myID.eq(zero).get(),
        [&](::mlir::OpBuilder &builder, ::mlir::Location loc) {
          builder.create<::mlir::scf::YieldOp>(
              loc, ValueRange{zero.get(), nextOff.get()});
        },
        [&](::mlir::OpBuilder &builder, ::mlir::Location loc) {
          auto myOff = easyIdx(
              loc, builder,
              builder.create<tensor::ExtractOp>(loc, targetOffs, (myPos).get())
                  .getResult());
          builder.create<::mlir::scf::YieldOp>(
              loc, ValueRange{myOff.get(), (nextOff - myOff).get()});
        });

    // the global offset of the local shard is slice offset plus the computed
    // offset in the target tensor the latter is in number of elements after
    // slicing, which means we need to multiply it by stride
    auto targetOff = easyIdx(loc, builder, slcOffs[dim]) +
                     easyIdx(loc, builder, myOffAndSize.getResult(0)) *
                         easyIdx(loc, builder, slcStrides[dim]);
    auto shardOff =
        imex::dist::getBaseShardDimOff(myID, numShards, extend, zero) -
        easyIdx(loc, builder, haloSizes[shardedDim * 2]);
    return {(targetOff - shardOff).get(), myOffAndSize.getResult(1)};
  }

  template <typename OP>
  FailureOr<std::tuple<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>,
                       SmallVector<OpFoldResult>>>
  getLocalOffSzAndStrFromSlice(OP op, ArrayRef<int64_t> srcShape,
                               const MeshSharding &operandSharding,
                               const MeshSharding &resultSharding,
                               SymbolTableCollection &symbolTableCollection,
                               OpBuilder &builder) const {
    if (!operandSharding.getStaticShardedDimsOffsets().empty() ||
        operandSharding.getStaticHaloSizes().empty() ||
        mlir::ShapedType::isDynamicShape(
            operandSharding.getStaticHaloSizes()) ||
        resultSharding.getStaticShardedDimsOffsets().empty() ||
        mlir::ShapedType::isDynamicShape(
            resultSharding.getStaticShardedDimsOffsets())) {
      return failure();
    }

    auto loc = op->getLoc();
    auto splitAxes = operandSharding.getSplitAxes();
    auto mesh =
        getMesh(op, operandSharding.getMeshAttr(), symbolTableCollection);
    auto myIdx = getMyMultiIndex(builder, mesh);
    auto haloSizes =
        mlir::getMixedValues(operandSharding.getStaticHaloSizes(),
                             operandSharding.getDynamicHaloSizes(), builder);
    auto shardedDimsOffsets = imex::getMixedAsValues(
        loc, builder, resultSharding.getDynamicShardedDimsOffsets(),
        resultSharding.getStaticShardedDimsOffsets());
    auto slcOffs =
        mlir::getMixedValues(op.getStaticOffsets(), op.getOffsets(), builder);
    auto slcSizes =
        mlir::getMixedValues(op.getStaticSizes(), op.getSizes(), builder);
    auto slcStrides =
        mlir::getMixedValues(op.getStaticStrides(), op.getStrides(), builder);
    auto rank = slcOffs.size();
    auto zero = easyIdx(loc, builder, 0);
    auto one = easyIdx(loc, builder, 1);
    ::SmallVector<OpFoldResult> lShardOffs, lShardSizes;
    auto targetOffs =
        builder.create<tensor::FromElementsOp>(loc, shardedDimsOffsets);

    for (auto dim = 0ul; dim < (uint64_t)rank; ++dim) {
      assert(!ShapedType::isDynamic(srcShape[dim]));
      if (dim >= splitAxes.size() || splitAxes[dim].empty()) {
        lShardOffs.emplace_back(slcOffs[dim]);
        lShardSizes.emplace_back(slcSizes[dim]);
      } else {
        auto offAndSz = getShardSliceOffAndSz(
            myIdx, dim, mesh.getShape(), splitAxes, targetOffs, srcShape,
            slcOffs, slcStrides, haloSizes, zero, one, builder, loc);
        lShardOffs.emplace_back(offAndSz[0]);
        lShardSizes.emplace_back(offAndSz[1]);
      }
    }
    return std::make_tuple(lShardOffs, lShardSizes, slcStrides);
  }
};

struct SubviewShardingInterface : public OffsetSizeAndStrideShardingInterface<SubviewShardingInterface, imex::ndarray::SubviewOp> {
  LogicalResult
  addShardingAnnotations(::mlir::Operation *op, OpBuilder &b,
                         const ShardingOption &shardingOption) const {
    auto svop = cast<SubviewOp>(op);
    auto srcShardOp = svop.getSource().getDefiningOp<mesh::ShardOp>();
    if (!srcShardOp) {
      LLVM_DEBUG(DBGS() << "no sharding on input, bailing out\n");
      return failure();
    }
    maybeInsertSourceShardingAnnotation(srcShardOp.getSharding(), op->getOpOperand(0), b);

    auto sharding = getShardedDimsOffsetsSharding(svop.getSource(), svop);
    if(failed(sharding)) return failure();
    maybeInsertTargetShardingAnnotation(sharding.value(), op->getResult(0), b);

    return success();
  }

  LogicalResult spmdize(::mlir::Operation *op, ArrayRef<Value> spmdizedOperands, ArrayRef<MeshSharding> operandShardings, ArrayRef<MeshSharding> resultShardings, IRMapping&spmdizationMap, SymbolTableCollection &symbolTableCollection, OpBuilder &builder) const {
    if (resultShardings.size() != 1) {
      return failure();
    }
    auto typedOp = cast<imex::ndarray::SubviewOp>(op);
    auto shp = cast<RankedTensorType>(typedOp.getSource().getType()).getShape();
    auto offSzStr = getLocalOffSzAndStrFromSlice(
        typedOp, shp, operandShardings[0], resultShardings[0],
        symbolTableCollection, builder);
    if (failed(offSzStr)) {
      return failure();
    }
    auto &[lShardOffs, lShardSizes, lShardStrides] = offSzStr.value();
    // auto opResultType = cast<RankedTensorType>(op->getResult(0).getType());
    // auto resultType =
    // cast<RankedTensorType>(opResultType.cloneWith(resultShape,
    // opResultType.getElementType()));
    auto newSubview = builder.create<imex::ndarray::SubviewOp>(
        op->getLoc(), spmdizedOperands[0], lShardOffs, lShardSizes,
        lShardStrides);
    spmdizationMap.map(op->getResult(0), newSubview.getResult());
    return success();
  }
};

struct InsertSliceShardingInterface : public OffsetSizeAndStrideShardingInterface<InsertSliceShardingInterface, imex::ndarray::InsertSliceOp> {
  LogicalResult addShardingAnnotations(::mlir::Operation *op, OpBuilder &b, const ShardingOption &shardingOption) const {
    LLVM_DEBUG(DBGS() << "addShardingAnnotations\n");
    auto svop = cast<InsertSliceOp>(op);

    auto srcSharding =
        getShardedDimsOffsetsSharding(svop.getDestination(), svop);
    if (failed(srcSharding))
      return failure();
    maybeInsertSourceShardingAnnotation(srcSharding.value(),
                                        op->getOpOperand(1), b);

    SmallVector<MeshAxesAttr> res;
    for (const auto &v : shardingOption.shardingArray) {
      res.emplace_back(MeshAxesAttr::get(op->getContext(), v));
    }
    auto dstSharding = mlir::mesh::MeshSharding::get(shardingOption.mesh, res);
    maybeInsertSourceShardingAnnotation(dstSharding, op->getOpOperand(0), b);

    return success();
  }

  LogicalResult spmdize(::mlir::Operation *op, ArrayRef<Value> spmdizedOperands, ArrayRef<MeshSharding> operandShardings, ArrayRef<MeshSharding> resultShardings, IRMapping&spmdizationMap, SymbolTableCollection &symbolTableCollection, OpBuilder &builder) const {
    if (resultShardings.size() != 0) {
      return failure();
    }

    auto typedOp = cast<imex::ndarray::InsertSliceOp>(op);
    auto shp =
        cast<RankedTensorType>(typedOp.getDestination().getType()).getShape();
    auto offSzStr = getLocalOffSzAndStrFromSlice(
        typedOp, shp, operandShardings[0], operandShardings[1],
        symbolTableCollection, builder);
    if (failed(offSzStr)) {
      return failure();
    }
    auto &[lShardOffs, lShardSizes, lShardStrides] = offSzStr.value();
    // auto opResultType = cast<RankedTensorType>(op->getResult(0).getType());
    // auto resultType =
    // cast<RankedTensorType>(opResultType.cloneWith(resultShape,
    // opResultType.getElementType()));

    Operation *newOp = builder.create<imex::ndarray::InsertSliceOp>(
        op->getLoc(), spmdizedOperands[0], spmdizedOperands[1], lShardOffs,
        lShardSizes, lShardStrides);
    spmdizationMap.map(op, newOp);

    builder.create<mlir::mesh::UpdateHaloOp>(
        op->getLoc(), spmdizedOperands[0].getType(), spmdizedOperands[0],
        operandShardings[0].getMeshAttr(),
        mlir::mesh::MeshAxesArrayAttr::get(op->getContext(),
                                           operandShardings[0].getSplitAxes()),
        operandShardings[0].getDynamicHaloSizes(),
        DenseI64ArrayAttr::get(op->getContext(),
                               operandShardings[0].getStaticHaloSizes()));
    return success();
  }
};
} // namespace

void registerShardingInterfaceExternalModels(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, imex::ndarray::NDArrayDialect *dialect) {
    SubviewOp::template attachInterface<SubviewShardingInterface>(*ctx);
    InsertSliceOp::template attachInterface<InsertSliceShardingInterface>(*ctx);
  });
}

} // namespace ndarray
} // namespace imex
