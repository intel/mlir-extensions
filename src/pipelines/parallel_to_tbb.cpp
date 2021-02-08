#include "pipelines/parallel_to_tbb.hpp"

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "plier/dialect.hpp"

#include "plier/compiler/pipeline_registry.hpp"
#include "pipelines/base_pipeline.hpp"
#include "pipelines/lower_to_llvm.hpp"

namespace
{
mlir::MemRefType getReduceType(mlir::Type type, int64_t count)
{
    if (type.isIntOrFloat())
    {
        return mlir::MemRefType::get(count, type);
    }
    return {};
}

mlir::Value getZeroVal(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type type)
{
    if (type.isa<mlir::IntegerType>())
    {
        return builder.create<mlir::ConstantIntOp>(loc, 0, type.cast<mlir::IntegerType>());
    }
    if (type.isa<mlir::FloatType>())
    {
        return builder.create<mlir::ConstantFloatOp>(loc, llvm::APFloat(0.0), type.cast<mlir::FloatType>());
    }
    llvm_unreachable("Unhandled type");
}

struct ParallelToTbb : public mlir::OpRewritePattern<mlir::scf::ParallelOp>
{
    using mlir::OpRewritePattern<mlir::scf::ParallelOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::scf::ParallelOp op, mlir::PatternRewriter &rewriter) const override
    {
        if (mlir::isa<plier::ParallelOp>(op->getParentOp()))
        {
            return mlir::failure();
        }
        if (op.getNumLoops() != 1)
        {
            return mlir::failure();
        }

        int64_t max_concurrency = 0;
        auto mod = op->getParentOfType<mlir::ModuleOp>();
        if (auto mc = mod->getAttrOfType<mlir::IntegerAttr>(plier::attributes::getMaxConcurrencyName()))
        {
            max_concurrency = mc.getInt();
        }

        if (max_concurrency <= 1)
        {
            return mlir::failure();
        }
        for (auto type : op.getResultTypes())
        {
            if (!getReduceType(type, max_concurrency))
            {
                return mlir::failure();
            }
        }

        auto loc = op.getLoc();
        mlir::BlockAndValueMapping mapping;
        llvm::SmallVector<mlir::Value, 8> reduce_vars(op.getNumResults());
        for (auto it : llvm::enumerate(op.getResultTypes()))
        {
            auto type = it.value();
            auto reduce_type = getReduceType(type, max_concurrency);
            assert(reduce_type);
            auto reduce = rewriter.create<mlir::AllocaOp>(loc, reduce_type);
            auto index = static_cast<unsigned>(it.index());
            reduce_vars[index] = reduce;
            auto zero = getZeroVal(rewriter, loc, type);
            mapping.map(op.initVals()[index], zero);
            for (unsigned i = 0; i < max_concurrency; ++i)
            {
                mlir::Value index = rewriter.create<mlir::ConstantIndexOp>(loc, i);
                rewriter.create<mlir::StoreOp>(loc, zero, reduce, index);
            }
        }

        auto& old_body = op.getLoopBody().front();
        auto orig_lower_bound = op.lowerBound().front();
        auto orig_upper_bound = op.upperBound().front();
        auto orig_step = op.step().front();
        auto body_builder = [&](mlir::OpBuilder &builder, ::mlir::Location loc, mlir::Value lower_bound, mlir::Value upper_bound, mlir::Value thread_index)
        {
            mapping.map(orig_lower_bound, lower_bound);
            mapping.map(orig_upper_bound, upper_bound);
            for (auto it : llvm::enumerate(op.initVals()))
            {
                auto reduce_var = reduce_vars[it.index()];
                auto val = builder.create<mlir::LoadOp>(loc, reduce_var, thread_index);
                mapping.map(it.value(), val);
            }
            auto new_op = builder.clone(*op, mapping);
            assert(new_op->getNumResults() == reduce_vars.size());
            for (auto it : llvm::enumerate(new_op->getResults()))
            {
                auto reduce_var = reduce_vars[it.index()];
                builder.create<mlir::StoreOp>(loc, it.value(), reduce_var, thread_index);
            }
        };

        rewriter.create<plier::ParallelOp>(loc, orig_lower_bound, orig_upper_bound, orig_step, body_builder);

        auto reduce_lower_bound = rewriter.create<mlir::ConstantIndexOp>(loc, 0);
        auto reduce_upper_bound = rewriter.create<mlir::ConstantIndexOp>(loc, max_concurrency);
        auto reduce_step = rewriter.create<mlir::ConstantIndexOp>(loc, 1);

        auto reduce_body_builder = [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value index, mlir::ValueRange args)
        {
            assert(args.size() == reduce_vars.size());
            mapping.clear();
            auto reduce_ops = llvm::make_filter_range(old_body.without_terminator(), [](auto& op)
            {
                return mlir::isa<mlir::scf::ReduceOp>(op);
            });
            llvm::SmallVector<mlir::Value, 8> yield_args;
            yield_args.reserve(args.size());
            for (auto it : llvm::enumerate(reduce_ops))
            {
                auto& reduce_var = reduce_vars[it.index()];
                auto arg = args[static_cast<unsigned>(it.index())];
                auto reduce_op = mlir::cast<mlir::scf::ReduceOp>(it.value());
                auto& reduce_op_body = reduce_op.reductionOperator().front();
                assert(reduce_op_body.getNumArguments() == 2);
                auto prev_val = builder.create<mlir::LoadOp>(loc, reduce_var, index);
                mapping.map(reduce_op_body.getArgument(0), arg);
                mapping.map(reduce_op_body.getArgument(1), prev_val);
                for (auto& old_reduce_op : reduce_op_body.without_terminator())
                {
                    builder.clone(old_reduce_op, mapping);
                }
                auto result = mlir::cast<mlir::scf::ReduceReturnOp>(reduce_op_body.getTerminator()).result();
                result = mapping.lookupOrNull(result);
                assert(result);
                yield_args.emplace_back(result);
            }
            builder.create<mlir::scf::YieldOp>(loc, yield_args);
        };

        auto reduce_loop = rewriter.create<mlir::scf::ForOp>(loc, reduce_lower_bound, reduce_upper_bound, reduce_step, op.initVals(), reduce_body_builder);
        rewriter.replaceOp(op, reduce_loop.getResults());

        return mlir::success();
    }
};

struct ParallelToTbbPass :
    public mlir::PassWrapper<ParallelToTbbPass, mlir::OperationPass<mlir::FuncOp>>
{
    virtual void getDependentDialects(
        mlir::DialectRegistry &registry) const override
    {
        registry.insert<plier::PlierDialect>();
        registry.insert<mlir::StandardOpsDialect>();
        registry.insert<mlir::scf::SCFDialect>();
    }

    void runOnOperation() override;
};

void ParallelToTbbPass::runOnOperation()
{
    mlir::OwningRewritePatternList patterns;

    patterns.insert<
        ParallelToTbb
        >(&getContext());

    mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

void populate_parallel_to_tbb_pipeline(mlir::OpPassManager& pm)
{
    pm.addNestedPass<mlir::FuncOp>(std::make_unique<ParallelToTbbPass>());
}
}

void register_parallel_to_tbb_pipeline(plier::PipelineRegistry& registry)
{
    registry.register_pipeline([](auto sink)
    {
        auto stage = get_lower_lowering_stage();
        auto llvm_pipeline = lower_to_llvm_pipeline_name();
        sink(parallel_to_tbb_pipeline_name(), {stage.begin}, {llvm_pipeline}, {}, &populate_parallel_to_tbb_pipeline);
    });
}

llvm::StringRef parallel_to_tbb_pipeline_name()
{
    return "parallel_to_tbb";
}
