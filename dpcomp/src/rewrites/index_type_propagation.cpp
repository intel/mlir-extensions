#include "plier/rewrites/index_type_propagation.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

namespace
{
bool is_index_compatible(mlir::Type lhs_type, mlir::Type rhs_type)
{
    if (!lhs_type.isa<mlir::IntegerType>() || lhs_type != rhs_type)
    {
        return false;
    }

    if (lhs_type.cast<mlir::IntegerType>().getWidth() < 64)
    {
        return false;
    }
    return true;
}

template<typename Op>
struct ArithIndexCastSimplify : public mlir::OpRewritePattern<Op>
{
    using mlir::OpRewritePattern<Op>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        Op op, mlir::PatternRewriter &rewriter) const override
    {
        auto lhs_type = op.lhs().getType();
        auto rhs_type = op.rhs().getType();
        if (!is_index_compatible(lhs_type, rhs_type))
        {
            return mlir::failure();
        }

        auto get_cast = [](mlir::Value val)->mlir::Value
        {
            if (auto op = mlir::dyn_cast_or_null<mlir::IndexCastOp>(val.getDefiningOp()))
            {
                return op.getOperand();
            }
            return {};
        };

        auto get_const = [](mlir::Value val)->mlir::IntegerAttr
        {
            if (auto op = mlir::dyn_cast_or_null<mlir::ConstantOp>(val.getDefiningOp()))
            {
                return op.getValue().cast<mlir::IntegerAttr>();
            }
            return {};
        };

        auto lhs = get_cast(op.lhs());
        auto rhs = get_cast(op.rhs());
        auto lhs_const = get_const(op.lhs());
        auto rhs_const = get_const(op.rhs());
        if (lhs && rhs)
        {
            auto new_op = rewriter.create<Op>(op.getLoc(), lhs, rhs);
            auto result = rewriter.create<mlir::IndexCastOp>(op.getLoc(), new_op.getResult(), lhs_type);
            rewriter.replaceOp(op, result.getResult());
            return mlir::success();
        }
        if (lhs && rhs_const)
        {
            auto new_const = rewriter.create<mlir::ConstantIndexOp>(op.getLoc(), rhs_const.getInt());
            auto new_op = rewriter.create<Op>(op.getLoc(), lhs, new_const);
            auto result = rewriter.create<mlir::IndexCastOp>(op.getLoc(), new_op.getResult(), lhs_type);
            rewriter.replaceOp(op, result.getResult());
            return mlir::success();
        }
        if (lhs_const && rhs)
        {
            auto new_const = rewriter.create<mlir::ConstantIndexOp>(op.getLoc(), lhs_const.getInt());
            auto new_op = rewriter.create<Op>(op.getLoc(), new_const, rhs);
            auto result = rewriter.create<mlir::IndexCastOp>(op.getLoc(), new_op.getResult(), lhs_type);
            rewriter.replaceOp(op, result.getResult());
            return mlir::success();
        }

        return mlir::failure();
    }
};

struct CmpIndexCastSimplify : public mlir::OpRewritePattern<mlir::CmpIOp>
{
    using mlir::OpRewritePattern<mlir::CmpIOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::CmpIOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto lhs_type = op.lhs().getType();
        auto rhs_type = op.rhs().getType();
        if (!is_index_compatible(lhs_type, rhs_type))
        {
            return mlir::failure();
        }

        auto get_cast = [](mlir::Value val)->mlir::Value
        {
            if (auto op = mlir::dyn_cast_or_null<mlir::IndexCastOp>(val.getDefiningOp()))
            {
                return op.getOperand();
            }
            return {};
        };

        auto get_const = [](mlir::Value val)->mlir::IntegerAttr
        {
            if (auto op = mlir::dyn_cast_or_null<mlir::ConstantOp>(val.getDefiningOp()))
            {
                return op.getValue().cast<mlir::IntegerAttr>();
            }
            return {};
        };

        auto lhs = get_cast(op.lhs());
        auto rhs = get_cast(op.rhs());
        auto lhs_const = get_const(op.lhs());
        auto rhs_const = get_const(op.rhs());
        if (lhs && rhs)
        {
            auto new_cmp = rewriter.create<mlir::CmpIOp>(op.getLoc(), op.predicate(), lhs, rhs);
            rewriter.replaceOp(op, new_cmp.getResult());
            return mlir::success();
        }
        if (lhs && rhs_const)
        {
            auto new_const = rewriter.create<mlir::ConstantIndexOp>(op.getLoc(), rhs_const.getInt());
            auto new_cmp = rewriter.create<mlir::CmpIOp>(op.getLoc(), op.predicate(), lhs, new_const);
            rewriter.replaceOp(op, new_cmp.getResult());
            return mlir::success();
        }
        if (lhs_const && rhs)
        {
            auto new_const = rewriter.create<mlir::ConstantIndexOp>(op.getLoc(), lhs_const.getInt());
            auto new_cmp = rewriter.create<mlir::CmpIOp>(op.getLoc(), op.predicate(), new_const, rhs);
            rewriter.replaceOp(op, new_cmp.getResult());
            return mlir::success();
        }

        return mlir::failure();
    }
};
}

void plier::populate_index_propagate_patterns(mlir::MLIRContext& context, mlir::OwningRewritePatternList& patterns)
{
    patterns.insert<
        CmpIndexCastSimplify,
        ArithIndexCastSimplify<mlir::SubIOp>,
        ArithIndexCastSimplify<mlir::AddIOp>,
        ArithIndexCastSimplify<mlir::MulIOp>,
        ArithIndexCastSimplify<mlir::SignedDivIOp>,
        ArithIndexCastSimplify<mlir::UnsignedDivIOp>,
        ArithIndexCastSimplify<mlir::SignedRemIOp>,
        ArithIndexCastSimplify<mlir::UnsignedRemIOp>
        >(&context);
}
