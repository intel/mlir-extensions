#include "plier/rewrites/type_conversion.hpp"

#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include "plier/dialect.hpp"

namespace
{
mlir::LogicalResult setBlockSig(
    mlir::Block& block, mlir::OpBuilder& builder,
    const mlir::TypeConverter::SignatureConversion& conversion)
{
    if (conversion.getConvertedTypes().size() != block.getNumArguments())
    {
        return mlir::failure();
    }
    unsigned i = 0;
    for (auto it : llvm::zip(block.getArguments(), conversion.getConvertedTypes()))
    {
        auto arg = std::get<0>(it);
        auto type = std::get<1>(it);
        if (arg.getType() != type)
        {
            builder.setInsertionPointToStart(&block);
            auto res = builder.create<plier::CastOp>(builder.getUnknownLoc(), arg.getType(), arg);
            arg.replaceUsesWithIf(res, [&](mlir::OpOperand& op)
            {
                return op.getOwner() != res;
            });

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
        ++i;
    }
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
        assert(mlir::succeeded(res));
        (void)res;
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
    auto type = funcOp.getType();

    // Convert the original function types.
    mlir::TypeConverter::SignatureConversion result(type.getNumInputs());
    llvm::SmallVector<mlir::Type, 1> newResults;
    if (mlir::failed(converter.convertSignatureArgs(type.getInputs(), result)) ||
        mlir::failed(converter.convertTypes(type.getResults(), newResults)) ||
        mlir::failed(convertRegionTypes(&funcOp.getBody(), converter, false)))
    {
        return mlir::failure();
    }

    bool ret_type_changed = false;
    // Update the function signature in-place.
    rewriter.updateRootInPlace(funcOp, [&] {
        ret_type_changed = (newResults != funcOp.getType().getResults());
        funcOp.setType(mlir::FunctionType::get(
            funcOp.getContext(), result.getConvertedTypes(), newResults));
        auto res = convertRegionTypes(&funcOp.getBody(), converter, true);
        assert(mlir::succeeded(res));
    });

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
    return mlir::success();
}
