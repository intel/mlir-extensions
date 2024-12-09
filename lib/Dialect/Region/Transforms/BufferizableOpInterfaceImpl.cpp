//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "imex/Dialect/Region/Transforms/BufferizableOpInterfaceImpl.h"
#include "imex/Dialect/Region/IR/RegionOps.h"

#include <mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h>

namespace imex {
namespace region {
namespace {

/// Return true if none of the values is TensorType.
bool noTensorsIn(::mlir::ValueRange values) {
  auto isTensor = [](::mlir::Value v) {
    return ::llvm::isa<::mlir::TensorType>(v.getType());
  };
  if (::llvm::any_of(values, isTensor))
    return false;
  return true;
}

/// Convert values to buffers. If a value is a tensor, get a buffer for it.
::mlir::LogicalResult
convertToBuffers(::mlir::ValueRange values,
                 ::mlir::SmallVector<::mlir::Value> &buffers,
                 ::mlir::RewriterBase &rewriter,
                 const ::mlir::bufferization::BufferizationOptions &options) {
  buffers.reserve(values.size());
  for (auto val : values) {
    if (::mlir::isa<::mlir::TensorType>(val.getType())) {
      ::mlir::FailureOr<::mlir::Value> maybeBuffer =
          ::mlir::bufferization::getBuffer(rewriter, val, options);
      if (failed(maybeBuffer)) {
        return ::mlir::failure();
      }
      buffers.push_back(*maybeBuffer);
    } else {
      buffers.push_back(val);
    }
  }
  return ::mlir::success();
}

/// Bufferization of region.env_region op. Replaced with a new
/// op that takes and returns memrefs.
struct EnvironmentRegionOpInterface
    : public ::mlir::bufferization::BufferizableOpInterface::ExternalModel<
          EnvironmentRegionOpInterface, region::EnvironmentRegionOp> {
  bool bufferizesToMemoryRead(
      ::mlir::Operation *op, ::mlir::OpOperand &opOperand,
      const ::mlir::bufferization::AnalysisState &state) const {
    assert(::mlir::isa<::mlir::RankedTensorType>(opOperand.get().getType()) &&
           "only tensor types expected");
    // Assume all operands are read.
    return true;
  }

  bool bufferizesToMemoryWrite(
      ::mlir::Operation *op, ::mlir::OpOperand &opOperand,
      const ::mlir::bufferization::AnalysisState &state) const {
    assert(::mlir::isa<::mlir::RankedTensorType>(opOperand.get().getType()) &&
           "only tensor types expected");
    // Assume all operands are written to.
    return true;
  }

  ::mlir::bufferization::AliasingValueList
  getAliasingValues(::mlir::Operation *op, ::mlir::OpOperand &opOperand,
                    const ::mlir::bufferization::AnalysisState &state) const {
    // Assume no aliasing.
    return {};
  }

  ::mlir::LogicalResult
  bufferize(::mlir::Operation *op, ::mlir::RewriterBase &rewriter,
            const ::mlir::bufferization::BufferizationOptions &options) const {
    auto envOp = ::mlir::cast<region::EnvironmentRegionOp>(op);
    if (noTensorsIn(envOp.getArgs()) && noTensorsIn(envOp.getResults())) {
      // Nothing to do.
      return ::mlir::success();
    }
    // Convert op arguments to memrefs.
    ::mlir::SmallVector<::mlir::Value> newArguments;
    if (failed(convertToBuffers(envOp.getArgs(), newArguments, rewriter,
                                options))) {
      return ::mlir::failure();
    }
    // Infer result memref types by converting yield op operands to memrefs
    ::mlir::SmallVector<::mlir::Value> newResults;
    if (failed(convertToBuffers(envOp.getBody()->getTerminator()->getOperands(),
                                newResults, rewriter, options))) {
      return ::mlir::failure();
    }
    ::mlir::TypeRange resTypes(newResults);
    // Create new op via generic constructor, op will have an empty region.
    rewriter.setInsertionPoint(op);
    ::mlir::OperationState state(op->getLoc(), op->getName(), newArguments,
                                 resTypes, op->getAttrs());
    state.addRegion();
    ::mlir::Operation *newOp = ::mlir::Operation::create(state);
    // Move block from old op into the new op.
    newOp->getRegion(0).getBlocks().splice(newOp->getRegion(0).begin(),
                                           op->getRegion(0).getBlocks());
    rewriter.insert(newOp);
    ::mlir::bufferization::replaceOpWithBufferizedValues(rewriter, op,
                                                         newOp->getResults());

    return ::mlir::success();
  }
};

/// Bufferization of region.env_region_yield. Replaced with a new yield that
/// operates on memrefs.
struct EnvironmentRegionYieldOpInterface
    : public ::mlir::bufferization::BufferizableOpInterface::ExternalModel<
          EnvironmentRegionYieldOpInterface, region::EnvironmentRegionYieldOp> {
  bool bufferizesToMemoryRead(
      ::mlir::Operation *op, ::mlir::OpOperand &opOperand,
      const ::mlir::bufferization::AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(
      ::mlir::Operation *op, ::mlir::OpOperand &opOperand,
      const ::mlir::bufferization::AnalysisState &state) const {
    return false;
  }

  ::mlir::bufferization::AliasingValueList
  getAliasingValues(::mlir::Operation *op, ::mlir::OpOperand &opOperand,
                    const ::mlir::bufferization::AnalysisState &state) const {
    return {{op->getParentOp()->getResult(opOperand.getOperandNumber()),
             ::mlir::bufferization::BufferRelation::Equivalent}};
  }

  bool mustBufferizeInPlace(
      ::mlir::Operation *op, ::mlir::OpOperand &opOperand,
      const ::mlir::bufferization::AnalysisState &state) const {
    // Yield operands always bufferize inplace.
    return true;
  }

  ::mlir::LogicalResult
  bufferize(::mlir::Operation *op, ::mlir::RewriterBase &rewriter,
            const ::mlir::bufferization::BufferizationOptions &options) const {
    auto yieldOp = ::mlir::cast<region::EnvironmentRegionYieldOp>(op);

    // Create a new terminator with bufferized operands.
    ::mlir::SmallVector<::mlir::Value> newOperands;
    if (failed(convertToBuffers(yieldOp.getOperands(), newOperands, rewriter,
                                options))) {
      return ::mlir::failure();
    }
    ::mlir::bufferization::replaceOpWithNewBufferizedOp<
        region::EnvironmentRegionYieldOp>(rewriter, op, newOperands);
    return ::mlir::success();
  }
};

} // namespace
} // namespace region
} // namespace imex

void imex::region::registerBufferizableOpInterfaceExternalModels(
    ::mlir::DialectRegistry &registry) {
  registry.addExtension(+[](::mlir::MLIRContext *ctx,
                            region::RegionDialect *dialect) {
    EnvironmentRegionOp::attachInterface<EnvironmentRegionOpInterface>(*ctx);
    EnvironmentRegionYieldOp::attachInterface<
        EnvironmentRegionYieldOpInterface>(*ctx);
  });
}
