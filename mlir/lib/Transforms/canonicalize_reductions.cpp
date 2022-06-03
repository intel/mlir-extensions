// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir-extensions/Transforms/canonicalize_reductions.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/IR/BlockAndValueMapping.h>

namespace {
bool checkMemrefType(mlir::Value value) {
  if (auto type = value.getType().dyn_cast<mlir::MemRefType>()) {
    //        auto shape = type.getShape();
    //        return shape.empty() || (1 == shape.size() && 1 == shape[0]);
    return true;
  }
  return false;
}

bool isOutsideBlock(mlir::ValueRange values, mlir::Block &block) {
  auto blockArgs = block.getArguments();
  for (auto val : values) {
    if (llvm::is_contained(blockArgs, val))
      return false;

    auto op = val.getDefiningOp();
    if (op && block.findAncestorOpInBlock(*op))
      return false;
  }
  return true;
}

bool checkForPotentialAliases(mlir::Value value, mlir::Operation *parent) {
  assert(parent->getRegions().size() == 1);
  assert(llvm::hasNItems(parent->getRegions().front(), 1));
  if (auto effects = mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(
          value.getDefiningOp())) {
    if (!effects.onlyHasEffect<mlir::MemoryEffects::Allocate>())
      return false;
  } else {
    return false;
  }

  mlir::memref::LoadOp load;
  mlir::memref::StoreOp store;
  auto &parentBlock = parent->getRegions().front().front();
  for (auto user : value.getUsers()) {
    if (mlir::isa<mlir::ViewLikeOpInterface>(user))
      return false; // TODO: very conservative

    if (!parent->isProperAncestor(user))
      continue;

    if (auto effects =
            mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(user)) {
      if (user->getBlock() != &parentBlock)
        return false;

      if (effects.hasEffect<mlir::MemoryEffects::Read>()) {
        if (load || !mlir::isa<mlir::memref::LoadOp>(user))
          return false;

        load = mlir::cast<mlir::memref::LoadOp>(user);
      }
      if (effects.hasEffect<mlir::MemoryEffects::Write>()) {
        if (store || !mlir::isa<mlir::memref::StoreOp>(user))
          return false;

        store = mlir::cast<mlir::memref::StoreOp>(user);
      }
    }
  }
  if (!load || !store || !load->isBeforeInBlock(store) ||
      load.indices() != store.indices() ||
      !isOutsideBlock(load.indices(), parentBlock)) {
    return false;
  }
  return true;
}

bool checkSupportedOps(mlir::Value value, mlir::Operation *parent) {
  for (auto user : value.getUsers()) {
    if (user->getParentOp() == parent &&
        !mlir::isa<mlir::memref::LoadOp, mlir::memref::StoreOp>(user))
      return false;
  }
  return true;
}

bool checkMemref(mlir::Value value, mlir::Operation *parent) {
  return checkMemrefType(value) && checkForPotentialAliases(value, parent) &&
         checkSupportedOps(value, parent);
}

mlir::Value createScalarLoad(mlir::PatternRewriter &builder, mlir::Location loc,
                             mlir::Value memref, mlir::ValueRange indices) {
  return builder.create<mlir::memref::LoadOp>(loc, memref, indices);
}

void createScalarStore(mlir::PatternRewriter &builder, mlir::Location loc,
                       mlir::Value val, mlir::Value memref,
                       mlir::ValueRange indices) {
  builder.create<mlir::memref::StoreOp>(loc, val, memref, indices);
}
} // namespace

mlir::LogicalResult plier::CanonicalizeReduction::matchAndRewrite(
    mlir::scf::ForOp op, mlir::PatternRewriter &rewriter) const {
  llvm::SmallVector<std::pair<mlir::Value, mlir::ValueRange>> to_process;
  for (auto &current : op.getLoopBody().front()) {
    if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(current)) {
      auto memref = load.memref();
      if (checkMemref(memref, op))
        to_process.push_back({memref, load.indices()});
    }
  }

  if (!to_process.empty()) {
    auto loc = op.getLoc();
    auto initArgs = llvm::to_vector<8>(op.getInitArgs());
    for (auto it : to_process) {
      initArgs.emplace_back(
          createScalarLoad(rewriter, loc, it.first, it.second));
    }
    auto prevArgsOffset = op.getInitArgs().size();
    auto body = [&](mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value iter, mlir::ValueRange iterVals) {
      auto &oldBody = op.getLoopBody().front();
      mlir::BlockAndValueMapping mapping;
      mapping.map(oldBody.getArguments().front(), iter);
      mapping.map(oldBody.getArguments().drop_front(), iterVals);
      auto yield_args = llvm::to_vector<8>(iterVals);
      for (auto &body_op : oldBody.without_terminator()) {
        auto invalid_index = static_cast<unsigned>(-1);
        auto get_iter_index = [&](auto op) -> unsigned {
          auto arg = op.memref();
          for (auto it : llvm::enumerate(llvm::make_first_range(to_process))) {
            if (arg == it.value()) {
              return static_cast<unsigned>(it.index() + prevArgsOffset);
            }
          }
          return invalid_index;
        };
        if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(body_op)) {
          auto index = get_iter_index(load);
          if (index != invalid_index) {
            mapping.map(body_op.getResults().front(), yield_args[index]);
          } else {
            builder.clone(body_op, mapping);
          }
        } else if (auto store =
                       mlir::dyn_cast<mlir::memref::StoreOp>(body_op)) {
          auto index = get_iter_index(store);
          if (index != invalid_index) {
            yield_args[index] = mapping.lookup(store.value());
          } else {
            builder.clone(body_op, mapping);
          }
        } else {
          builder.clone(body_op, mapping);
        }
      }
      auto yield = mlir::cast<mlir::scf::YieldOp>(oldBody.getTerminator());
      llvm::copy(yield.getResults(), yield_args.begin());
      builder.create<mlir::scf::YieldOp>(loc, yield_args);
    };
    auto results = rewriter
                       .create<mlir::scf::ForOp>(loc, op.getLowerBound(),
                                                 op.getUpperBound(),
                                                 op.getStep(), initArgs, body)
                       .getResults();
    for (auto it : llvm::enumerate(to_process)) {
      auto index = prevArgsOffset + it.index();
      auto result = results[static_cast<unsigned>(index)];
      createScalarStore(rewriter, loc, result, it.value().first,
                        it.value().second);
    }
    rewriter.replaceOp(op, results.take_front(prevArgsOffset));
    return mlir::success();
  }

  return mlir::failure();
}
