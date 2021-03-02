#include "pipelines/plier_to_linalg.hpp"

#include <mlir/IR/Dialect.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tensor/Transforms/Passes.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/SCF/Passes.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/LoopUtils.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "plier/dialect.hpp"

#include "pipelines/plier_to_std.hpp"

#include "plier/transforms/pipeline_utils.hpp"
#include "plier/rewrites/call_lowering.hpp"
#include "plier/rewrites/canonicalize_reductions.hpp"
#include "plier/rewrites/cast_lowering.hpp"
#include "plier/rewrites/cse.hpp"
#include "plier/rewrites/promote_to_parallel.hpp"
#include "plier/rewrites/type_conversion.hpp"
#include "plier/rewrites/force_inline.hpp"
#include "plier/rewrites/index_type_propagation.hpp"
#include "plier/rewrites/loop_rewrites.hpp"
#include "plier/transforms/loop_utils.hpp"

#include "base_pipeline.hpp"
#include "plier/compiler/pipeline_registry.hpp"
#include "py_linalg_resolver.hpp"

#include <cctype>

namespace
{
enum class ArrayLayout
{
    C,
    F
};

bool parse_layout(llvm::StringRef& name, ArrayLayout& layout)
{
    if (name.consume_back("C"))
    {
        layout = ArrayLayout::C;
        return true;
    }
    if (name.consume_back("F"))
    {
        layout = ArrayLayout::F;
        return true;
    }
    return false;
}

template<typename T>
bool consume_int_back(llvm::StringRef& name, T& result)
{
    unsigned len = 0;
    auto tmp_name = name;
    while (!tmp_name.empty() && std::isdigit(tmp_name.back()))
    {
        ++len;
        tmp_name = tmp_name.drop_back();
    }
    tmp_name = name.substr(name.size() - len);
    if (!tmp_name.consumeInteger<T>(10, result))
    {
        name = name.substr(0, name.size() - len);
        return true;
    }
    return false;
}

struct ArrayDesc
{
    unsigned dims = 0;
    ArrayLayout layout = {};
    llvm::StringRef name;
};

llvm::Optional<ArrayDesc> parse_array_desc(llvm::StringRef& name)
{
    unsigned num_dims = 0;
    ArrayLayout layout = {};
    if (name.consume_front("array(") &&
        name.consume_back(")") &&
        parse_layout(name, layout) &&
        name.consume_back(", ") &&
        name.consume_back("d") &&
        consume_int_back(name, num_dims) &&
        name.consume_back(", ") &&
        !name.empty())
    {
        return ArrayDesc{num_dims, layout, name};
    }
    return {};
}

mlir::Type map_array_type(mlir::MLIRContext& ctx, mlir::TypeConverter& conveter,
                          llvm::StringRef& name)
{
    if (auto desc = parse_array_desc(name))
    {
        if (desc->layout == ArrayLayout::C)
        {
            if (auto type = conveter.convertType(plier::PyType::get(&ctx, desc->name)))
            {
                llvm::SmallVector<int64_t> shape(desc->dims, -1);
                return mlir::RankedTensorType::get(shape, type);
            }
        }
    }
    return nullptr;
}


mlir::Type map_plier_type(mlir::TypeConverter& converter, mlir::Type type)
{
    if (type.isa<plier::PyType>())
    {
        auto name = type.cast<plier::PyType>().getName();
        return map_array_type(*type.getContext(), converter, name);
    }
    return nullptr;
}

bool check_numpy_args(llvm::ArrayRef<mlir::Value> args, unsigned expected_count)
{
    if (args.size() != expected_count)
    {
        return false;
    }
    for (auto arg : args)
    {
        auto type = arg.getType();
        if (!type.isa<mlir::MemRefType>() && !type.isa<mlir::TensorType>())
        {
            return false;
        }
    }
    return true;
}

void rerun_std_pipeline(mlir::Operation* op)
{
    assert(nullptr != op);
    auto marker = mlir::StringAttr::get(op->getContext(), plier_to_std_pipeline_name());
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    assert(nullptr != mod);
    plier::add_pipeline_jump_marker(mod, marker);
}

bool is_int(mlir::Type type)
{
    assert(type);
    return type.isa<mlir::IntegerType>();
}

mlir::LogicalResult lower_prange(plier::PyCallOp op, llvm::ArrayRef<mlir::Value> operands, llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs, mlir::PatternRewriter& rewriter)
{
    if (!kwargs.empty())
    {
        return mlir::failure();
    }
    if ((operands.size() < 1 || operands.size() > 3) ||
        !llvm::all_of(operands, [](mlir::Value val) { return is_int(val.getType());}))
    {
        return mlir::failure();
    }
    mlir::Value val = op.getResult();
    if (!val.getUsers().empty())
    {
        auto user = mlir::dyn_cast<plier::GetiterOp>(*val.getUsers().begin());
        auto get_bounds = [&](mlir::OpBuilder& builder, mlir::Location loc)
        {
            auto lower_bound = (operands.size() >= 2 ? operands[0] : builder.create<mlir::ConstantIndexOp>(loc, 0));
            auto upper_bound = (operands.size() >= 2 ? operands[1] : operands[0]);
            auto step = (operands.size() == 3 ? operands[2] : builder.create<mlir::ConstantIndexOp>(loc, 1));
            return std::make_tuple(lower_bound, upper_bound, step);
        };
        auto get_index = [](mlir::OpBuilder& builder, mlir::Location loc, mlir::Type dst_type, mlir::Value index)
        {
            return builder.create<plier::CastOp>(loc, dst_type, index);
        };
        auto set_attr = [](mlir::scf::ForOp op)
        {
            op->setAttr(plier::attributes::getParallelName(), mlir::UnitAttr::get(op->getContext()));
        };
        if (!user || mlir::failed(lower_while_to_for(user, rewriter, get_bounds, get_index, set_attr)))
        {
            return mlir::failure();
        }
    }

    rerun_std_pipeline(op);
    if (val.getUsers().empty())
    {
        rewriter.eraseOp(op);
    }
    return mlir::success();
}

struct CallLowerer
{
    using args_t = llvm::ArrayRef<mlir::Value>;
    using kwargs_t = llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>>;
    mlir::LogicalResult operator()(
        plier::PyCallOp op, llvm::StringRef name, args_t args,
        kwargs_t kwargs,
        mlir::PatternRewriter& rewriter)
    {
        using func_t = mlir::LogicalResult(*)(plier::PyCallOp, args_t, kwargs_t, mlir::PatternRewriter&);
        std::pair<llvm::StringRef, func_t> handlers[] = {
            {"numba.prange", lower_prange},
        };
        for (auto& handler : handlers)
        {
            if (handler.first == name)
            {
                return handler.second(op, args, kwargs, rewriter);
            }
        }

        if (mlir::succeeded(applyRewrite(op, rewriter, linalg_resolver.rewrite_func(name, op.getLoc(), rewriter, args, kwargs))))
        {
            return mlir::success();
        }

        if (name == "len" && check_numpy_args(args, 1) && kwargs.empty())
        {
            auto loc = op.getLoc();
            mlir::Value dim = rewriter.create<mlir::DimOp>(loc, args[0], 0);
            mlir::Value res = rewriter.create<plier::CastOp>(loc, op.getType(), dim);
            rerun_std_pipeline(op);
            rewriter.replaceOp(op, res);
            return mlir::success();
        }
        return mlir::failure();
    }

    mlir::LogicalResult operator()(
        plier::GetattrOp op, llvm::StringRef name, mlir::Value arg,
        mlir::PatternRewriter& rewriter)
    {
        if (!arg.getType().isa<mlir::ShapedType>())
        {
            return mlir::failure();
        }
        auto full_name = (llvm::Twine("array.") + name).str();
        return applyRewrite(op, rewriter, linalg_resolver.rewrite_attr(full_name, op.getLoc(), rewriter, arg));
    }

    mlir::LogicalResult operator()(
        plier::BinOp op, llvm::StringRef name, mlir::Value lhs, mlir::Value rhs,
        mlir::PatternRewriter& rewriter)
    {
        if (!lhs.getType().isa<mlir::ShapedType>() &&
            !rhs.getType().isa<mlir::ShapedType>())
        {
            return mlir::failure();
        }
        const std::pair<llvm::StringRef, llvm::StringRef> names[] = {
            {"+", "operator.add"},
            {"-", "operator.sub"},
            {"*", "operator.mul"},
        };
        for (auto it : names)
        {
            if (it.first == name)
            {
                return applyRewrite(op, rewriter, linalg_resolver.rewrite_func(it.second, op.getLoc(), rewriter, {lhs, rhs}, {}));
            }
        }
        return mlir::failure();
    }

private:
    PyLinalgResolver linalg_resolver;

    mlir::LogicalResult applyRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter, llvm::Optional<PyLinalgResolver::Values> result)
    {
        if (result)
        {
            assert(result->size() == op->getNumResults());
            rerun_std_pipeline(op);
            if (result->empty())
            {
                rewriter.eraseOp(op);
            }
            else
            {
                rewriter.replaceOp(op, *result);
            }
            return mlir::success();
        }
        return mlir::failure();
    }
};

mlir::Value index_cast(mlir::Value value, mlir::Location loc, mlir::OpBuilder& builder)
{
    if (!value.getType().isa<mlir::IndexType>())
    {
        auto index_type = mlir::IndexType::get(value.getContext());
        auto res = builder.create<plier::CastOp>(loc, index_type, value);
        rerun_std_pipeline(res);
        return res;
    }
    return value;
}

bool isValidGetitemIndex(mlir::Type type)
{
    return type.isa<mlir::IntegerType, mlir::IndexType, mlir::TupleType>();
}

template<typename T>
struct GetitemOpLowering : public mlir::OpRewritePattern<T>
{
    using mlir::OpRewritePattern<T>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        T op, mlir::PatternRewriter &rewriter) const override
    {
        assert(op.getNumOperands() == 2);
        auto val = op.getOperand(0);
        auto index = op.getOperand(1);
        auto type = val.getType();
        bool is_memref = type.template isa<mlir::MemRefType>();
        bool is_tensor = type.template isa<mlir::TensorType>();
        if (!is_memref && !is_tensor)
        {
            return mlir::failure();
        }
        if (!isValidGetitemIndex(index.getType()))
        {
            return mlir::failure();
        }
        auto loc = op.getLoc();

        llvm::SmallVector<mlir::Value> indices;
        if (auto tuple_type = index.getType().template dyn_cast<mlir::TupleType>())
        {
            indices.resize(tuple_type.size());
            for (auto it : llvm::enumerate(tuple_type))
            {
                auto getitem_ind = rewriter.create<mlir::ConstantIndexOp>(loc, it.index());
                auto ind = rewriter.create<plier::GetItemOp>(loc, index, getitem_ind);
                indices[it.index()] = index_cast(ind, loc, rewriter);
            }
        }
        else
        {
            indices.push_back(index_cast(index, loc, rewriter));
        }

        mlir::Value res;
        if (is_memref)
        {
            res = rewriter.create<mlir::LoadOp>(loc, val, indices);
        }
        else if (is_tensor)
        {
            res = rewriter.create<mlir::tensor::ExtractOp>(loc, val, indices);
        }
        else
        {
            llvm_unreachable("Invalid getitem");
        }
        rerun_std_pipeline(op);
        rewriter.replaceOp(op, res);
        return mlir::success();
    }
};

bool can_replace_ssa(mlir::Operation* op)
{
    assert(nullptr != op);
    if (op->getParentRegion()->getBlocks().size() != 1)
    {
        return false;
    }
    auto parent = op->getParentOp();
    if (mlir::isa<mlir::FuncOp>(parent))
    {
        return true;
    }
    return false;
//    return can_replace_ssa(parent);
}

bool replace_ssa_in_block(mlir::Value value, mlir::Value new_value, mlir::PatternRewriter &rewriter)
{
    auto new_op = new_value.getDefiningOp();
    assert(nullptr != new_op);
    auto block = new_op->getBlock();
    bool changed = false;
    for (auto user : llvm::make_early_inc_range(value.getUsers()))
    {
        if (auto op = block->findAncestorOpInBlock(*user))
        {
            if (op != new_op && new_op->isBeforeInBlock(op))
            {
                rewriter.updateRootInPlace(user, [&]()
                {
                    for (auto it2 : llvm::enumerate(user->getOperands()))
                    {
                        if (it2.value() == value)
                        {
                            user->setOperand(static_cast<unsigned>(it2.index()), new_value);
                            break;
                        }
                    }
                });
                changed = true;
            }
        }
    }
    return changed;
}

bool replace_ssa_value(mlir::Value value, mlir::Value new_value, mlir::PatternRewriter &rewriter)
{
    bool changed = replace_ssa_in_block(value, new_value, rewriter);
    auto parent = new_value.getDefiningOp()->getParentOp();
    if (auto func = mlir::dyn_cast<mlir::FuncOp>(parent))
    {
        // TODO update return
        return changed;
    }
    llvm_unreachable("Unhandled parent op");
}

template<typename T>
struct SetitemOpLoweringSSA : public mlir::OpRewritePattern<T>
{
    using mlir::OpRewritePattern<T>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        T op, mlir::PatternRewriter &rewriter) const override
    {
        if (!can_replace_ssa(op))
        {
            return mlir::failure();
        }
        auto target = op.getOperand(0);
        auto index = op.getOperand(1);
        auto value = op.getOperand(2);
        auto target_type = target.getType().template dyn_cast<mlir::RankedTensorType>();
        if (!target_type)
        {
            return mlir::failure();
        }
        auto elem_type = target_type.getElementType();
        auto loc = op.getLoc();
        if (value.getType() != elem_type)
        {
            // TODO
            value = rewriter.create<plier::CastOp>(loc, elem_type, value);
            rerun_std_pipeline(op);
//            return mlir::failure();
        }

        auto new_tensor = rewriter.create<mlir::tensor::FromElementsOp>(loc, value);
        auto new_index = index_cast(index, loc, rewriter);
        mlir::Value one = rewriter.create<mlir::ConstantIndexOp>(loc, 1);
        auto new_value = rewriter.create<mlir::SubTensorInsertOp>(loc, new_tensor, target, new_index, one, one);
        replace_ssa_value(target, new_value, rewriter);
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct PlierToLinalgPass :
    public mlir::PassWrapper<PlierToLinalgPass, mlir::OperationPass<mlir::ModuleOp>>
{
    virtual void getDependentDialects(
        mlir::DialectRegistry &registry) const override
    {
        registry.insert<plier::PlierDialect>();
        registry.insert<mlir::StandardOpsDialect>();
        registry.insert<mlir::linalg::LinalgDialect>();
    }

    void runOnOperation() override;
};

template<typename T>
struct SetitemOpLowering : public mlir::OpRewritePattern<T>
{
    using mlir::OpRewritePattern<T>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        T op, mlir::PatternRewriter &rewriter) const override
    {
        auto get_target_type = [&]()
        {
            return op.getOperand(0).getType();
        };

        auto index = op.index();
        if (!isValidGetitemIndex(index.getType()))
        {
            return mlir::failure();
        }

        if (auto target_type = get_target_type().template dyn_cast<mlir::RankedTensorType>())
        {
            auto target = op.getOperand(0);
            mlir::OpBuilder::InsertionGuard g(rewriter);
            if (auto parent_op = target.getDefiningOp())
            {
                rewriter.setInsertionPointAfter(parent_op);
            }
            else
            {
                rewriter.setInsertionPointToStart(target.getParentBlock());
            }
            auto memref_type = mlir::MemRefType::get(target_type.getShape(), target_type.getElementType());
            auto memref = rewriter.create<mlir::TensorToMemrefOp>(target.getLoc(), memref_type, target);
            for (auto& use : llvm::make_early_inc_range(target.getUses()))
            {
                auto use_op = use.getOwner();
                assert(nullptr != use_op);
                if (use_op != memref)
                {
                    if (mlir::isa<plier::SetItemOp>(use_op))
                    {
                        use_op->setOperand(use.getOperandNumber(), memref);
                    }
                    else
                    {
                        mlir::OpBuilder::InsertionGuard g(rewriter);
                        rewriter.setInsertionPoint(use_op);
                        auto new_val = rewriter.create<mlir::TensorLoadOp>(use_op->getLoc(), memref);
                        rewriter.updateRootInPlace(use_op, [&]()
                        {
                            use_op->setOperand(use.getOperandNumber(), new_val);
                        });
                    }
                }
            }
        }
        else if (get_target_type().template isa<mlir::MemRefType>())
        {
            // nothing
        }
        else
        {
            return mlir::failure();
        }
        auto target = op.getOperand(0);
        auto value = op.getOperand(2);
        auto loc = op.getLoc();
        auto elem_type = target.getType().template cast<mlir::MemRefType>().getElementType();
        if (value.getType() != elem_type)
        {
            // TODO
            value = rewriter.create<plier::CastOp>(loc, elem_type, value);
            rerun_std_pipeline(op);
        }

        llvm::SmallVector<mlir::Value> indices;
        if (auto tuple_type = index.getType().template dyn_cast<mlir::TupleType>())
        {
            indices.resize(tuple_type.size());
            for (auto it : llvm::enumerate(tuple_type))
            {
                auto getitem_ind = rewriter.create<mlir::ConstantIndexOp>(loc, it.index());
                auto ind = rewriter.create<plier::GetItemOp>(loc, index, getitem_ind);
                indices[it.index()] = index_cast(ind, loc, rewriter);
            }
            rerun_std_pipeline(op);
        }
        else
        {
            indices.push_back(index_cast(index, loc, rewriter));
        }
        rewriter.create<mlir::StoreOp>(loc, value, target, indices);
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct ArrayShape : public mlir::OpRewritePattern<plier::GetattrOp>
{
    ArrayShape(mlir::TypeConverter& type_converter,
               mlir::MLIRContext* context):
        OpRewritePattern(context),
        converter(type_converter) {}

    mlir::LogicalResult matchAndRewrite(
        plier::GetattrOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto type = op.value().getType().dyn_cast<mlir::ShapedType>();
        if (!type || op.name() != "shape" || !type.hasRank())
        {
            return mlir::failure();
        }

        auto rank = static_cast<size_t>(type.getRank());
        auto elem_type = converter.convertType(op.getType()).dyn_cast_or_null<mlir::TupleType>();
        if (!elem_type || elem_type.size() != rank)
        {
            return mlir::failure();
        }

        llvm::SmallVector<mlir::Value> dims(rank);
        for (size_t i = 0; i < rank; ++i)
        {
            auto dim = rewriter.create<mlir::DimOp>(op.getLoc(), op.value(), i);
            dims[i] = rewriter.create<plier::CastOp>(op.getLoc(), elem_type.getType(i), dim);
        }
        auto res = rewriter.create<plier::BuildTupleOp>(op.getLoc(), op.getType(), dims);
        rerun_std_pipeline(op);
        rewriter.replaceOp(op, res.getResult());
        return mlir::success();
    }

private:
    mlir::TypeConverter& converter;
};

template<typename T>
bool has_compatibale_shape(T&& a1, T&& a2)
{
    if (!a1.hasRank() || !a2.hasRank() || a1.getRank() != a2.getRank())
    {
        return false;
    }
    for (auto it : llvm::zip(a1.getShape(), a2.getShape()))
    {
        auto s1 = std::get<0>(it);
        auto s2 = std::get<1>(it);
        if (s1 >= 0 && s2 >= 0 && s1 != s2)
        {
            return false;
        }
    }
    return true;
}

struct RankedTypesCasts : public mlir::OpRewritePattern<plier::CastOp>
{
    RankedTypesCasts(mlir::TypeConverter& /*type_converter*/,
                     mlir::MLIRContext* context):
        OpRewritePattern(context){}

    mlir::LogicalResult matchAndRewrite(
        plier::CastOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto src_type = op.value().getType();
        auto dst_type = op.getType();
        if (src_type.isa<mlir::TensorType>() && dst_type.isa<mlir::TensorType>())
        {
            auto src = src_type.cast<mlir::TensorType>();
            auto dst = dst_type.cast<mlir::TensorType>();
            if (!has_compatibale_shape(src,dst))
            {
                return mlir::failure();
            }
            rewriter.replaceOpWithNewOp<mlir::tensor::CastOp>(op, dst, op.value());
            return mlir::success();
        }
        return mlir::failure();
    }
};

struct GetattrRewriter : public mlir::OpRewritePattern<plier::GetattrOp>
{
    using resolver_t = std::function<mlir::LogicalResult(plier::GetattrOp, llvm::StringRef, mlir::Value,
                                                         mlir::PatternRewriter&)>;

    GetattrRewriter(mlir::TypeConverter &/*typeConverter*/,
                    mlir::MLIRContext *context,
                    resolver_t resolver):
        OpRewritePattern(context),
        resolver(resolver)
    {}

    mlir::LogicalResult matchAndRewrite(
        plier::GetattrOp op, mlir::PatternRewriter &rewriter) const override
    {
        return resolver(op, op.name(), op.value(), rewriter);
    }

private:
    resolver_t resolver;
};

struct BinopRewriter : public mlir::OpRewritePattern<plier::BinOp>
{
    using resolver_t = std::function<mlir::LogicalResult(plier::BinOp, llvm::StringRef, mlir::Value, mlir::Value,
                                                         mlir::PatternRewriter&)>;

    BinopRewriter(mlir::TypeConverter &/*typeConverter*/,
                  mlir::MLIRContext *context,
                  resolver_t resolver):
        OpRewritePattern(context),
        resolver(resolver)
    {}

    mlir::LogicalResult matchAndRewrite(
        plier::BinOp op, mlir::PatternRewriter &rewriter) const override
    {
        return resolver(op, op.op(), op.lhs(), op.rhs(), rewriter);
    }

private:
    resolver_t resolver;
};

void PlierToLinalgPass::runOnOperation()
{
    auto context = &getContext();

    mlir::TypeConverter type_converter;
    // Convert unknown types to itself
    type_converter.addConversion([](mlir::Type type) { return type; });
    populate_std_type_converter(getContext(), type_converter);
    type_converter.addConversion([&](plier::PyType type)->llvm::Optional<mlir::Type>
    {
        auto ret =  map_plier_type(type_converter, type);
        if (!ret)
        {
            return llvm::None;
        }
        return ret;
    });

    mlir::OwningRewritePatternList patterns;
    patterns.insert<
        plier::FuncOpSignatureConversion,
        plier::CastOpLowering,
        RankedTypesCasts,
        ArrayShape
        >(type_converter, context);

    CallLowerer callLowerer;

    patterns.insert<
        plier::CallOpLowering,
        GetattrRewriter,
        BinopRewriter
        >(type_converter, context, std::ref(callLowerer));

    patterns.insert<
        GetitemOpLowering<plier::GetItemOp>,
        GetitemOpLowering<plier::StaticGetItemOp>,
        SetitemOpLowering<plier::SetItemOp>,
        plier::ForceInline
        >(&getContext());

    // range/prange lowering need dead branch pruning to properly
    // handle negative steps
    for (auto *op : context->getRegisteredOperations())
    {
        op->getCanonicalizationPatterns(patterns, context);
    }

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

struct LowerLinalgPass :
    public mlir::PassWrapper<LowerLinalgPass, mlir::OperationPass<mlir::ModuleOp>>
{
    virtual void getDependentDialects(
        mlir::DialectRegistry &registry) const override
    {
        registry.insert<mlir::StandardOpsDialect>();
        registry.insert<mlir::tensor::TensorDialect>();
        registry.insert<mlir::linalg::LinalgDialect>();
        registry.insert<mlir::scf::SCFDialect>();
        registry.insert<mlir::AffineDialect>();
    }

    void runOnOperation() override;
};

void LowerLinalgPass::runOnOperation()
{
    mlir::OwningRewritePatternList patterns;

    patterns.insert<
        mlir::linalg::LinalgLoweringPattern<mlir::linalg::GenericOp>,
        mlir::linalg::LinalgLoweringPattern<mlir::linalg::CopyOp>
        >(&getContext(), mlir::linalg::LinalgLoweringType::ParallelLoops);


    (void)mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

struct PostFusionOptPass :
    public mlir::PassWrapper<PostFusionOptPass, mlir::OperationPass<mlir::ModuleOp>>
{
    virtual void getDependentDialects(
        mlir::DialectRegistry &registry) const override
    {
        registry.insert<mlir::StandardOpsDialect>();
        registry.insert<mlir::linalg::LinalgDialect>();
        registry.insert<mlir::scf::SCFDialect>();
        registry.insert<mlir::AffineDialect>();
    }

    void runOnOperation() override;
};

void PostFusionOptPass::runOnOperation()
{
    mlir::OwningRewritePatternList patterns;

    auto& context = getContext();
    for (auto *op : context.getRegisteredOperations())
    {
        op->getCanonicalizationPatterns(patterns, &context);
    }

    patterns.insert<
        //        LoopInvariantCodeMotion, TODO
        plier::CSERewrite<mlir::FuncOp>
        >(&context);

    plier::populate_index_propagate_patterns(context, patterns);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

struct LoopInvariantCodeMotion : public mlir::OpRewritePattern<mlir::scf::ForOp>
{
    using mlir::OpRewritePattern<mlir::scf::ForOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::scf::ForOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto parentOp = op->getParentOp();
        rewriter.startRootUpdate(parentOp);
        auto res = mlir::moveLoopInvariantCode(op);
        if (mlir::succeeded(res))
        {
            rewriter.finalizeRootUpdate(parentOp);
        }
        else
        {
            rewriter.cancelRootUpdate(parentOp);
        }
        return res;
    }
};

struct PostLinalgOptPass :
    public mlir::PassWrapper<PostLinalgOptPass, mlir::OperationPass<mlir::ModuleOp>>
{
    virtual void getDependentDialects(
        mlir::DialectRegistry &registry) const override
    {
        registry.insert<mlir::StandardOpsDialect>();
        registry.insert<mlir::linalg::LinalgDialect>();
        registry.insert<mlir::scf::SCFDialect>();
        registry.insert<mlir::AffineDialect>();
    }

    void runOnOperation() override;
};

void PostLinalgOptPass::runOnOperation()
{
    mlir::OwningRewritePatternList patterns;

    auto& context = getContext();
    for (auto *op : context.getRegisteredOperations())
    {
        op->getCanonicalizationPatterns(patterns, &context);
    }

    patterns.insert<
        plier::CanonicalizeReduction,
//        LoopInvariantCodeMotion, TODO
        plier::PromoteToParallel,
        plier::CmpLoopBoundsSimplify,
        plier::CSERewrite<mlir::FuncOp>
        >(&context);

    plier::populate_index_propagate_patterns(context, patterns);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

void populate_plier_to_linalg_gen_pipeline(mlir::OpPassManager& pm)
{
    pm.addPass(std::make_unique<PlierToLinalgPass>());
    pm.addPass(mlir::createSymbolDCEPass());
}

void populate_plier_to_linalg_opt_pipeline(mlir::OpPassManager& pm)
{
    pm.addPass(mlir::createLinalgFusionOfTensorOpsPass());
    pm.addPass(std::make_unique<PostFusionOptPass>());

    pm.addPass(mlir::createTensorConstantBufferizePass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createSCFBufferizePass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createLinalgBufferizePass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createStdBufferizePass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createTensorBufferizePass());
    pm.addPass(mlir::createFuncBufferizePass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createFinalizingBufferizePass());

    pm.addNestedPass<mlir::FuncOp>(mlir::createBufferHoistingPass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createBufferLoopHoistingPass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createPromoteBuffersToStackPass());

    pm.addNestedPass<mlir::FuncOp>(mlir::createBufferDeallocationPass());

    pm.addPass(std::make_unique<LowerLinalgPass>());
    pm.addPass(std::make_unique<PostLinalgOptPass>());
    pm.addPass(mlir::createSymbolDCEPass());
}
}

void register_plier_to_linalg_pipeline(plier::PipelineRegistry& registry)
{
    registry.register_pipeline([](auto sink)
    {
        auto stage = get_high_lowering_stage();
        sink(plier_to_linalg_gen_pipeline_name(), {plier_to_std_pipeline_name()}, {plier_to_linalg_opt_pipeline_name()}, {plier_to_std_pipeline_name()}, &populate_plier_to_linalg_gen_pipeline);
        sink(plier_to_linalg_opt_pipeline_name(), {plier_to_linalg_gen_pipeline_name()}, {stage.end}, {}, &populate_plier_to_linalg_opt_pipeline);
    });
}

llvm::StringRef plier_to_linalg_gen_pipeline_name()
{
    return "plier_to_linalg_gen";
}

llvm::StringRef plier_to_linalg_opt_pipeline_name()
{
    return "plier_to_linalg_opt";
}
