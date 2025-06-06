//===- BufferizableOpInterfaceImpl.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "imex/Dialect/NDArray/Extensions/BufferizableOpInterfaceImpl.h"
#include "imex/Dialect/NDArray/IR/NDArrayOps.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"

using namespace mlir;
using namespace bufferization;

namespace imex {
namespace ndarray {
namespace {

/// Bufferization of tensor.extract_slice. Replace with memref.subview.
struct SubviewOpInterface
    : public BufferizableOpInterface::ExternalModel<SubviewOpInterface,
                                                    SubviewOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {{op->getOpResult(0), BufferRelation::Unknown}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          const mlir::bufferization::BufferizationState& state) const {
    auto subviewOp = cast<SubviewOp>(op);
    SmallVector<OpFoldResult> mixedOffsets = subviewOp.getMixedOffsets();
    SmallVector<OpFoldResult> mixedSizes = subviewOp.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = subviewOp.getMixedStrides();
    Location loc = subviewOp.getLoc();

    // Get source buffer.
    FailureOr<Value> srcMemref =
        getBuffer(rewriter, subviewOp.getSource(), options, state);
    if (failed(srcMemref))
      return failure();

    // Take a subview of the source buffer.
    auto resultMemrefType =
        bufferization::getBufferType(subviewOp.getResult(), options, state);
    if (failed(resultMemrefType))
      return failure();
    Value subView = rewriter.create<memref::SubViewOp>(
        loc, llvm::cast<MemRefType>(*resultMemrefType), *srcMemref,
        mixedOffsets, mixedSizes, mixedStrides);

    replaceOpWithBufferizedValues(rewriter, op, subView);
    return success();
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                const mlir::bufferization::BufferizationState& state,
                SmallVector<Value> &invocationStack) const {
    auto subviewOp = cast<SubviewOp>(op);
    assert(value == subviewOp.getResult() && "invalid value");
    auto srcMemrefType = bufferization::getBufferType(subviewOp.getSource(),
                                                      options, state, invocationStack);
    if (failed(srcMemrefType))
      return failure();
    SmallVector<OpFoldResult> mixedOffsets = subviewOp.getMixedOffsets();
    SmallVector<OpFoldResult> mixedSizes = subviewOp.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = subviewOp.getMixedStrides();
    return cast<BaseMemRefType>(memref::SubViewOp::inferRankReducedResultType(
        subviewOp.getType().getShape(), llvm::cast<MemRefType>(*srcMemrefType),
        mixedOffsets, mixedSizes, mixedStrides));
  }
};

/// Bufferization of tensor.insert_slice. Replace with a memory copy. Under
/// certain circumstances, this op can also be a no-op.
///
/// Note: DstBufferizableOpInterfaceExternalModel provides many default method
/// implementations for DestinationStyle ops.
struct InsertSliceOpInterface
    : public DstBufferizableOpInterfaceExternalModel<
          InsertSliceOpInterface, imex::ndarray::InsertSliceOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return opOperand ==
           cast<imex::ndarray::InsertSliceOp>(op).getSourceMutable();
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    return true;
  }

  bool isNotConflicting(Operation *op, OpOperand *uRead, OpOperand *uWrite,
                        const AnalysisState &state) const {
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          const mlir::bufferization::BufferizationState& state) const {
    // insert_slice ops arise from tiling and bufferizing them out-of-place is
    // generally a deal breaker. When used with loops, this ends up cloning the
    // whole tensor on every single iteration and is a symptom of a
    // catastrophically bad scheduling decision.
    // TODO: be very loud about it or even consider failing the pass.
    auto insertSliceOp = cast<imex::ndarray::InsertSliceOp>(op);
    SmallVector<OpFoldResult> mixedOffsets = insertSliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> mixedSizes = insertSliceOp.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = insertSliceOp.getMixedStrides();
    Location loc = insertSliceOp.getLoc();

    // Get destination buffer.
    auto dstMemref =
        getBuffer(rewriter, insertSliceOp.getDestination(), options, state);
    if (failed(dstMemref))
      return failure();
    auto srcMemref = getBuffer(rewriter, insertSliceOp.getSource(), options, state);
    if (failed(srcMemref))
      return failure();
    auto srcRank = mlir::cast<mlir::ShapedType>(srcMemref->getType()).getRank();

    if (srcRank == 0) {
      // If the source tensor is basically a scalar, we need to copy the scalar
      // value using a linalg.generic into the view of the destination buffer.
      auto subView = rewriter.create<memref::SubViewOp>(
          loc, *dstMemref, mixedOffsets, mixedSizes, mixedStrides);
      // emit a loop that broadcasts a scalar to dst shape
      // construct broadcasting affine map; srcRank==0 case is simple
      auto dstRank =
          mlir::cast<mlir::ShapedType>(dstMemref->getType()).getRank();
      auto srcMap =
          ::mlir::AffineMap::get(dstRank, srcRank, {}, rewriter.getContext());
      auto dstMap = rewriter.getMultiDimIdentityMap(dstRank);
      ::mlir::SmallVector<mlir::utils::IteratorType> iterators(
          dstRank, ::mlir::utils::IteratorType::parallel);
      (void)rewriter.create<::mlir::linalg::GenericOp>(
          loc, srcMemref.value(), subView.getResult(),
          ::mlir::ArrayRef({srcMap, dstMap}), iterators,
          [](::mlir::OpBuilder &b, ::mlir::Location loc,
             ::mlir::ValueRange args) {
            b.create<::mlir::linalg::YieldOp>(loc, args.front());
          });
    } else {
      // Take a subview of the destination buffer.
      auto dstMemrefType = cast<MemRefType>(dstMemref->getType());
      auto subviewMemRefType =
          cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
              insertSliceOp.getSourceType().getShape(), dstMemrefType,
              mixedOffsets, mixedSizes, mixedStrides));
      Value subView = rewriter.create<memref::SubViewOp>(
          loc, subviewMemRefType, *dstMemref, mixedOffsets, mixedSizes,
          mixedStrides);

      // Copy tensor. If this tensor.insert_slice has a matching
      // tensor.extract_slice, the copy operation will eventually fold away.
      if (failed(options.createMemCpy(rewriter, loc, *srcMemref, subView)))
        return failure();
    }

    replaceOpWithBufferizedValues(rewriter, op, *dstMemref);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Interface registration
//===----------------------------------------------------------------------===//

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, imex::ndarray::NDArrayDialect *dialect) {
        InsertSliceOp::attachInterface<InsertSliceOpInterface>(*ctx);
        SubviewOp::attachInterface<SubviewOpInterface>(*ctx);
      });
}

} // namespace ndarray
} // namespace imex
