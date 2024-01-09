//===- XeCommon.cpp -  --------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements XeTypeConverter, ValueAttributeMap and some other
/// routines used by Xe related dialects.
///
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include "imex/Dialect/XeGPU/IR/XeGPU.h"
#include "imex/Dialect/XeTile/IR/XeTileOps.h"
#include "imex/Utils/DebugUtils.h"
#include "imex/Utils/XeCommon.h"

namespace imex {

void ValueAttributeMap::add(mlir::BlockArgument arg, imex::OperandType type) {
  if (argumentMap.count(arg) == 0)
    argumentMap[arg] = int(type);
  else
    argumentMap[arg] |= int(type);
}

void ValueAttributeMap::add(mlir::Operation *op, imex::OperandType type) {
  if (operationMap.count(op) == 0)
    operationMap[op] = int(type);
  else
    operationMap[op] |= int(type);
}

imex::OperandType ValueAttributeMap::get(mlir::Operation *op) {
  if (operationMap.count(op) == 0)
    return OperandType::None;
  return OperandType(operationMap[op]);
}

imex::OperandType ValueAttributeMap::get(mlir::BlockArgument arg) {
  if (argumentMap.count(arg) == 0)
    return OperandType::None;
  return OperandType(argumentMap[arg]);
}

static bool isConvertibleOp(mlir::Operation *op) {
  if (llvm::isa<imex::xetile::LoadTileOp>(op) ||
      llvm::isa<imex::xetile::UpdateTileOffsetOp>(op)) {
    return true;
  }
  return false;
}

static int getOperandIndex(mlir::Operation *op, mlir::Value operand) {
  for (auto [i, value] : llvm::enumerate(op->getOperands())) {
    if (operand == value)
      return i;
  }
  return -1;
};

static mlir::Value getOperandForArg(mlir::scf::ForOp &forOp,
                                    mlir::Value &value) {
  auto arg = llvm::dyn_cast_or_null<mlir::BlockArgument>(value);
  if (arg && arg.getArgNumber() >= forOp.getNumInductionVars()) {
    auto &iterOperand = *forOp.getTiedLoopInit(arg);
    auto numCtrlOperands = forOp.getNumControlOperands();
    auto operandIdx = iterOperand.getOperandNumber();
    return forOp.getInitArgs()[operandIdx - numCtrlOperands];
  }
  return mlir::Value();
};

static mlir::BlockArgument getArgForOperand(mlir::scf::ForOp &forOp,
                                            mlir::Value operand) {
  auto idx = getOperandIndex(forOp, operand);
  auto numControls = forOp.getNumControlOperands();
  assert(idx >= (int)numControls);
  return forOp.getRegionIterArg(idx - numControls);
};

static mlir::Operation *getDefineOrParentOp(mlir::Value value) {
  if (llvm::isa<mlir::OpResult>(value))
    return value.getDefiningOp();
  if (auto arg = llvm::dyn_cast_or_null<mlir::BlockArgument>(value))
    return arg.getOwner()->getParentOp();
  return NULL;
};

enum class ChainType { DefChain, UseChain };

// It traverse operators and arguments in the chain,
// ops will carry on operators in the chain
// arg will carry on arguments in the chain
// skippedOps carry the operators to be skipped during the traverse
template <ChainType chain = ChainType::DefChain>
void traverseDefUseChain(mlir::Value value,
                         llvm::SmallVector<mlir::Operation *> &ops,
                         llvm::SmallVector<mlir::BlockArgument> &args,
                         llvm::SmallVector<mlir::Operation *> skippedOps = {}) {

  llvm::SmallVector<mlir::Value> queue;

  auto isSkipped = [&](mlir::Operation *op) {
    return std::find(skippedOps.begin(), skippedOps.end(), op) !=
           skippedOps.end();
  };

  auto visitDef = [&](mlir::Value value) {
    if (auto arg = llvm::dyn_cast_or_null<mlir::BlockArgument>(value))
      args.push_back(arg);

    auto *op = getDefineOrParentOp(value);
    if (op == nullptr || isSkipped(op))
      return;

    // we don't track scf.for since it is composited op
    if (!llvm::isa<mlir::scf::ForOp>(op))
      ops.push_back(op);

    if (isConvertibleOp(op)) {
      queue.append(op->operand_begin(), op->operand_end());
    } else if (auto forOp = llvm::dyn_cast_or_null<mlir::scf::ForOp>(op)) {
      auto opr = getOperandForArg(forOp, value);
      if (bool(opr))
        queue.push_back(opr);
    } else if (llvm::isa<mlir::BlockArgument>(value) &&
               !llvm::isa<mlir::func::FuncOp>(op) &&
               !llvm::isa<mlir::gpu::GPUFuncOp>(op)) {
      op->emitError("\nUnexpected operator of an BlockArgument.\n");
      llvm_unreachable("Unexpected case for when handling a BlockArgument.\n");
    }

    return;
  };

  auto visitUsers = [&](mlir::Value value) {
    if (!bool(value))
      return;
    for (mlir::Operation *user : value.getUsers()) {
      if (isSkipped(user))
        continue;
      ops.push_back(user);
      // YieldOp indicats results of a SCF ForOp, IfOp is currently not handled.
      if (llvm::isa<mlir::scf::YieldOp>(user)) {
        auto *parentOp = user->getParentOp();
        auto idx = getOperandIndex(user, value);
        if (llvm::isa<mlir::scf::ForOp>(parentOp) && idx >= 0) {
          auto opResult = parentOp->getResult(idx);
          queue.push_back(opResult);
        } else {
          llvm_unreachable(
              "Meet an unexpected/unprocessed op in preOrderVisist.\n");
        }
      } else if (auto forOp = llvm::dyn_cast_or_null<mlir::scf::ForOp>(user)) {
        auto arg = getArgForOperand(forOp, value);
        args.push_back(arg);
        queue.push_back(arg);
      } else if (isConvertibleOp(user)) {
        queue.append(user->result_begin(), user->result_end());
      }
    }
  };

  if (bool(value))
    queue.push_back(value);

  while (queue.size()) {
    auto value = queue.pop_back_val();
    if (!bool(value))
      continue;

    if (chain == ChainType::DefChain) {
      visitDef(value);
      continue;
    }

    if (chain == ChainType::UseChain) {
      visitUsers(value);
      continue;
    }
  }
}

static bool isInterestingTarget(mlir::Operation *op) {
  auto constantOp = llvm::dyn_cast_or_null<mlir::arith::ConstantOp>(op);
  return llvm::isa<imex::xetile::XeTileDialect>(op->getDialect()) ||
         (constantOp &&
          llvm::isa<mlir::VectorType>(constantOp.getResult().getType()));
}

static bool isInterestingTarget(mlir::BlockArgument arg) {
  auto ty = arg.getType();
  return llvm::isa<imex::xetile::TileType>(ty) ||
         llvm::isa<mlir::VectorType>(ty);
}

/*
 * markDefChainValues iterates over the values in the def-chain of the given
 * value, and mark/record them with the OperandType type in ValueAttributeMap.
 */
void markDefChainValues(mlir::Value value, imex::OperandType type,
                        imex::ValueAttributeMap &map) {

  auto mark = [&]<typename T>(llvm::SmallVector<T> &array) {
    for (auto v : array) {
      if (isInterestingTarget(v))
        map.add(v, type);
    }
  };

  llvm::SmallVector<mlir::Operation *> postOrderOps;
  llvm::SmallVector<mlir::BlockArgument> postOrderArgs;
  traverseDefUseChain<ChainType::DefChain>(value, postOrderOps, postOrderArgs);

  // mark the interested ops with the type in map.
  // here we only interested in ops from xetile dialect.
  // and the arith.consantOp for initializing C of mma.
  mark(postOrderOps);

  // mark the interested arguments we only interested in
  // args with TileType and VectorType
  mark(postOrderArgs);

  llvm::SmallVector<mlir::Value> TopDownVisits;
  for (auto op : postOrderOps) {
    if (isConvertibleOp(op))
      TopDownVisits.append(op->result_begin(), op->result_end());
  }

  for (auto arg : postOrderArgs) {
    if (isInterestingTarget(arg))
      TopDownVisits.push_back(arg);
  }

  llvm::SmallVector<mlir::Operation *> preOrderOps;
  llvm::SmallVector<mlir::BlockArgument> preOrderArgs;

  for (auto v : TopDownVisits) {
    // If the value just has one user, it should have been visited in
    // postOrderVisit. This is how it is added to the vector.
    if (llvm::hasSingleElement(v.getUsers()))
      continue;
    // get Operators and arguments directly and indirectly using value v
    // but skip those already visited in postOrderVisit
    traverseDefUseChain<ChainType::UseChain>(v, preOrderOps, preOrderArgs,
                                             postOrderOps);
  }

  mark(preOrderOps);
  mark(preOrderArgs);
}

/*
 * markUseChainValues iterates over the values in use-chain of the given value,
 * and mark/record them with the OperandType type in ValueAttributeMap.
 */
void markUseChainValues(mlir::Value value, imex::OperandType type,
                        imex::ValueAttributeMap &map) {

  auto mark = [&]<typename T>(llvm::SmallVector<T> &array) {
    for (auto v : array) {
      if (isInterestingTarget(v))
        map.add(v, type);
    }
  };

  llvm::SmallVector<mlir::Operation *> preOrderOps;
  llvm::SmallVector<mlir::BlockArgument> preOrderArgs;
  traverseDefUseChain<ChainType::UseChain>(value, preOrderOps, preOrderArgs);

  mark(preOrderOps);
  mark(preOrderArgs);

  llvm::SmallVector<mlir::Value> BottomUpVisits;

  // We don't do postOrderVisit on arguments since they only have one defining
  // op which should be visited in preOrderVisit.
  for (auto op : preOrderOps) {
    if (isConvertibleOp(op))
      BottomUpVisits.append(op->operand_begin(), op->operand_end());
  }

  llvm::SmallVector<mlir::Operation *> postOrderOps;
  llvm::SmallVector<mlir::BlockArgument> postOrderArgs;

  for (auto v : BottomUpVisits) {
    traverseDefUseChain<ChainType::DefChain>(v, postOrderOps, postOrderArgs,
                                             preOrderOps);
  }

  mark(postOrderOps);
  mark(postOrderArgs);
}

mlir::ValueRange buildUnrealizedCast(mlir::OpBuilder &builder,
                                     mlir::TypeRange resultTypes,
                                     mlir::ValueRange inputs) {
  mlir::Location loc = builder.getUnknownLoc();
  if (!inputs.empty())
    loc = inputs.front().getLoc();
  auto castOp = builder.create<mlir::UnrealizedConversionCastOp>(
      loc, resultTypes, inputs);
  return castOp->getResults();
}

} // namespace imex
