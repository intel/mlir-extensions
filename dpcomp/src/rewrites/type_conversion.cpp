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

#include "plier/rewrites/type_conversion.hpp"

#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/SCF/SCF.h>

#include "plier/dialect.hpp"

namespace
{
mlir::LogicalResult setBlockSig(
    mlir::Block& block, mlir::OpBuilder& builder,
    const mlir::TypeConverter::SignatureConversion& conversion)
{
    llvm::SmallVector<unsigned> argsToErase;
    for (auto it : llvm::enumerate(block.getArguments()))
    {
        auto arg = it.value();
        auto i = static_cast<unsigned>(it.index());
        auto input = conversion.getInputMapping(i);
        bool hasInput = static_cast<bool>(input);
        auto getNewType = [&]()
        {
            assert(hasInput);
            return conversion.getConvertedTypes()[input->inputNo];
        };
        if (!hasInput || arg.getType() != getNewType())
        {
            auto type = (hasInput ? getNewType() : arg.getType());
            if (!arg.getUses().empty())
            {
                builder.setInsertionPointToStart(&block);
                auto loc = builder.getUnknownLoc();
                if (hasInput)
                {
                    auto res = builder.create<plier::CastOp>(loc, arg.getType(), arg);
                    arg.replaceAllUsesExcept(res, res);
                }
                else
                {
                    auto res = builder.create<plier::UndefOp>(loc, arg.getType());
                    arg.replaceAllUsesWith(res);
                }
            }
            if (!hasInput)
            {
                argsToErase.emplace_back(i);
            }
            for (auto& use : block.getUses())
            {
                auto op = use.getOwner();
                builder.setInsertionPoint(op);
                if (auto br = mlir::dyn_cast<mlir::BranchOp>(op))
                {
                    assert(&block == br.dest());
                    auto src = br.destOperands()[i];
                    auto new_op = builder.create<plier::CastOp>(op->getLoc(), type, src);
                    br.destOperandsMutable().slice(i, 1).assign(new_op);
                }
                else if (auto cond_br = mlir::dyn_cast<mlir::CondBranchOp>(op))
                {
                    if (&block == cond_br.trueDest())
                    {
                        auto src = cond_br.trueDestOperands()[i];
                        auto new_op = builder.create<plier::CastOp>(op->getLoc(), type, src);
                        cond_br.trueDestOperandsMutable().slice(i, 1).assign(new_op);
                    }
                    if (&block == cond_br.falseDest())
                    {
                        auto src = cond_br.falseDestOperands()[i];
                        auto new_op = builder.create<plier::CastOp>(op->getLoc(), type, src);
                        cond_br.falseDestOperandsMutable().slice(i, 1).assign(new_op);
                    }
                }
                else
                {
                    llvm_unreachable("setBlockSig: unknown operation type");
                }
            }
            arg.setType(type);
        }
    }
    block.eraseArguments(argsToErase);
    return mlir::success();
}

mlir::LogicalResult convertRegionTypes(
    mlir::Region *region, mlir::TypeConverter &converter, bool apply)
{
    assert(nullptr != region);
    if (region->empty())
    {
        return mlir::failure();
    }

    mlir::OpBuilder builder(region->getContext());

    // Convert the arguments of each block within the region.
    auto sig = converter.convertBlockSignature(&region->front());
    assert(static_cast<bool>(sig));
    if (apply)
    {
        auto res = setBlockSig(region->front(), builder, *sig);
        if (mlir::failed(res))
        {
            return mlir::failure();
        }
    }
    for (auto &block : llvm::make_early_inc_range(llvm::drop_begin(*region, 1)))
    {
        sig = converter.convertBlockSignature(&block);
        if (!sig)
        {
            return mlir::failure();
        }
        if (apply)
        {
            if (mlir::failed(setBlockSig(block, builder, *sig)))
            {
                return mlir::failure();
            }
        }
    }
    return mlir::success();
}
}

plier::FuncOpSignatureConversion::FuncOpSignatureConversion(mlir::TypeConverter& conv,
    mlir::MLIRContext* ctx)
    : OpRewritePattern(ctx), converter(conv) {}

mlir::LogicalResult plier::FuncOpSignatureConversion::matchAndRewrite(
    mlir::FuncOp funcOp, mlir::PatternRewriter& rewriter) const
{
    auto oldFuncType = funcOp.getType();

    // Convert the original function types.
    mlir::TypeConverter::SignatureConversion result(oldFuncType.getNumInputs());
    llvm::SmallVector<mlir::Type, 1> newResults;
    if (mlir::failed(converter.convertSignatureArgs(oldFuncType.getInputs(), result)) ||
        mlir::failed(converter.convertTypes(oldFuncType.getResults(), newResults)) ||
        mlir::failed(convertRegionTypes(&funcOp.getBody(), converter, false)))
    {
        return mlir::failure();
    }

    auto newFuncType = mlir::FunctionType::get(
        funcOp.getContext(), result.getConvertedTypes(), newResults);
    if (newFuncType == oldFuncType)
    {
        return mlir::failure();
    }

    bool ret_type_changed = (newResults != oldFuncType.getResults());
    // Update the function signature in-place.
    rewriter.startRootUpdate(funcOp);
    funcOp.setType(newFuncType);
    auto res = convertRegionTypes(&funcOp.getBody(), converter, true);
    if (mlir::failed(res))
    {
        funcOp.setType(oldFuncType);
        rewriter.cancelRootUpdate(funcOp);
        return mlir::failure();
    }

    if (ret_type_changed)
    {
        auto ret_types = funcOp.getType().getResults();
        funcOp.walk([&](mlir::ReturnOp ret)
        {
            if (ret->getParentOp() == funcOp)
            {
                mlir::OpBuilder::InsertionGuard g(rewriter);
                rewriter.setInsertionPoint(ret);
                for (auto it : llvm::enumerate(llvm::zip(ret.getOperandTypes(), ret_types)))
                {
                    auto prev_type = std::get<0>(it.value());
                    auto new_type = std::get<1>(it.value());
                    if (prev_type != new_type)
                    {
                        auto index = static_cast<unsigned>(it.index());
                        auto cast = rewriter.create<plier::CastOp>(ret.getLoc(), new_type, ret.getOperand(index));
                        rewriter.updateRootInPlace(ret, [&]()
                        {
                            ret.setOperand(index, cast);
                        });
                    }
                }
            }
        });
        auto mod = funcOp->getParentOfType<mlir::ModuleOp>();
        auto uses = funcOp.getSymbolUses(mod);
        if (uses)
        {
            for (auto use : *uses)
            {
                if (auto call = mlir::dyn_cast<mlir::CallOp>(use.getUser()))
                {
                    rewriter.updateRootInPlace(call, [&]()
                    {
                        for (auto it : llvm::zip(call.getResults(), ret_types))
                        {
                            auto res = std::get<0>(it);
                            auto type = std::get<1>(it);
                            res.setType(type);
                        }
                    });
                }
            }
        }
    }
    rewriter.finalizeRootUpdate(funcOp);
    return mlir::success();
}

plier::FixupIfTypes::FixupIfTypes(mlir::TypeConverter& typeConverter, mlir::MLIRContext* context):
    OpRewritePattern(context), converter(typeConverter) {}

mlir::LogicalResult plier::FixupIfTypes::matchAndRewrite(mlir::scf::IfOp op, mlir::PatternRewriter& rewriter) const
{
    if (op->getNumResults() == 0)
    {
        return mlir::failure();
    }

    llvm::SmallVector<mlir::Type> newTypes;
    newTypes.reserve(op->getNumResults());
    auto trueYield = mlir::cast<mlir::scf::YieldOp>(op.thenBlock()->getTerminator());
    auto falseYield = mlir::cast<mlir::scf::YieldOp>(op.elseBlock()->getTerminator());
    bool needUpdate = false;
    for (auto it : llvm::zip(op->getResultTypes(), trueYield->getOperandTypes(), falseYield->getOperandTypes()))
    {
        auto retType = std::get<0>(it);
        auto trueType = std::get<1>(it);
        auto falseType = std::get<2>(it);
        if (trueType != falseType)
        {
            return mlir::failure();
        }

        if (retType == trueType)
        {
            newTypes.emplace_back(trueType);
        }
        else if (converter.convertType(retType) == trueType)
        {
            newTypes.emplace_back(trueType);
            needUpdate = true;
        }
        else
        {
            return mlir::failure();
        }
    }

    if (needUpdate)
    {
        rewriter.updateRootInPlace(op, [&]()
        {
            for (auto it : llvm::enumerate(op->getResults()))
            {
                it.value().setType(newTypes[it.index()]);
            }
        });
        return mlir::success();
    }
    return mlir::failure();
}
