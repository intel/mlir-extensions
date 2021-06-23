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
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/LoopUtils.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "plier/dialect.hpp"

#include "pipelines/plier_to_std.hpp"

#include "plier/transforms/pipeline_utils.hpp"
#include "plier/rewrites/arg_lowering.hpp"
#include "plier/rewrites/call_lowering.hpp"
#include "plier/rewrites/canonicalize_reductions.hpp"
#include "plier/rewrites/cast_lowering.hpp"
#include "plier/rewrites/common_opts.hpp"
#include "plier/rewrites/cse.hpp"
#include "plier/rewrites/promote_to_parallel.hpp"
#include "plier/rewrites/type_conversion.hpp"
#include "plier/rewrites/loop_rewrites.hpp"
#include "plier/rewrites/memory_rewrites.hpp"
#include "plier/transforms/const_utils.hpp"
#include "plier/transforms/loop_utils.hpp"

#include "base_pipeline.hpp"
#include "plier/compiler/pipeline_registry.hpp"
#include "py_linalg_resolver.hpp"
#include "py_func_resolver.hpp"
#include "mangle.hpp"

#include <cctype>

namespace
{
void applyOptimizations(mlir::FuncOp op, const mlir::FrozenRewritePatternSet& patterns, mlir::AnalysisManager am, llvm::function_ref<mlir::LogicalResult(mlir::FuncOp)> additionalOpts = nullptr)
{
    bool repeat = false;
    do
    {
        repeat = false;
        (void)mlir::applyPatternsAndFoldGreedily(op, patterns);
        if (mlir::succeeded(plier::applyCSE(op.getRegion(), false)))
        {
            repeat = true;
        }
        if (mlir::succeeded(plier::optimizeMemoryOps(am)))
        {
            repeat = true;
        }
        if (additionalOpts && mlir::succeeded(additionalOpts(op)))
        {
            repeat = true;
        }
        if (repeat)
        {
            am.invalidate({});
        }
    }
    while(repeat);
}

enum class ArrayLayout
{
    C,
    F,
    A
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
    if (name.consume_back("A"))
    {
        layout = ArrayLayout::A;
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
        if (desc->layout == ArrayLayout::C ||
            desc->layout == ArrayLayout::F ||
            desc->layout == ArrayLayout::A)
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
            mlir::Value dim = rewriter.create<mlir::memref::DimOp>(loc, args[0], 0);
            mlir::Value res = rewriter.create<plier::CastOp>(loc, op.getType(), dim);
            rerun_std_pipeline(op);
            rewriter.replaceOp(op, res);
            return mlir::success();
        }

        mlir::ValueRange r(args);
        auto mangled_name = mangle(name, r.getTypes());
        if (!mangled_name.empty())
        {
            auto mod = op->getParentOfType<mlir::ModuleOp>();
            assert(mod);
            auto func = mod.lookupSymbol<mlir::FuncOp>(mangled_name);
            if (!func)
            {
                func = py_resolver.get_func(name, r.getTypes());
                if (func)
                {
                    func.setPrivate();
                    func.setName(mangled_name);
                }
            }
            if (func)
            {
                assert(func.getType().getNumResults() == op->getNumResults());
                auto new_func_call = rewriter.create<mlir::CallOp>(op.getLoc(), func, args);
                rerun_std_pipeline(op);
                rewriter.replaceOp(op, new_func_call.getResults());
                return mlir::success();
            }
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
        for (auto it : plier::getOperators())
        {
            if (it.op == name)
            {
                return applyRewrite(op, rewriter, linalg_resolver.rewrite_func(llvm::Twine("operator.") + it.name, op.getLoc(), rewriter, {lhs, rhs}, {}));
            }
        }
        return mlir::failure();
    }

private:
    PyLinalgResolver linalg_resolver;
    PyFuncResolver py_resolver;

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
    if (type.isa<plier::SliceType>())
    {
        return true;
    }
    if (auto tupleType = type.dyn_cast<mlir::TupleType>())
    {
        return llvm::all_of(tupleType.getTypes(), &isValidGetitemIndex);
    }
    return type.isa<mlir::IntegerType, mlir::IndexType>();
}

struct GetitemOpLowering : public mlir::OpRewritePattern<plier::GetItemOp>
{
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        plier::GetItemOp op, mlir::PatternRewriter &rewriter) const override
    {
        assert(op.getNumOperands() == 2);
        auto value = op.value();
        auto index = op.index();
        auto type = value.getType();
        bool is_memref = type.isa<mlir::MemRefType>();
        bool is_tensor = type.isa<mlir::TensorType>();
        if (!is_memref && !is_tensor)
        {
            return mlir::failure();
        }
        if (!isValidGetitemIndex(index.getType()))
        {
            return mlir::failure();
        }
        auto loc = op.getLoc();
        auto indexType = rewriter.getIndexType();
        auto getPos = [&](mlir::Value val, unsigned dim)->std::tuple<mlir::OpFoldResult, mlir::OpFoldResult, mlir::OpFoldResult, bool>
        {
            if (auto sliceType = val.getType().dyn_cast<plier::SliceType>())
            {
                auto createInd = [&](int64_t i)
                {
                    return rewriter.create<mlir::ConstantIndexOp>(loc, i);
                };
                auto getItemOrConst = [&](unsigned i)->mlir::Value
                {
                    assert(i < 3);
                    if (sliceType.getTypes()[i].isa<plier::NoneType>())
                    {
                        if (i == 0)
                        {
                            return createInd(0);
                        }
                        else if (i == 1)
                        {
                            return rewriter.createOrFold<mlir::memref::DimOp>(loc, value, dim);
                        }
                        else // i == 2
                        {
                            return createInd(1);
                        }
                    }
                    return rewriter.create<plier::GetItemOp>(loc, indexType, val, createInd(i));
                };
                auto offset = getItemOrConst(0);
                auto end    = getItemOrConst(1);
                auto stride = getItemOrConst(2);
                auto size = rewriter.create<mlir::SubIOp>(loc, end, offset).getResult();
                return {offset, size, stride, true};
            }
            else
            {
                auto offset = index_cast(val, loc, rewriter);
                return {offset, rewriter.getIndexAttr(1), rewriter.getIndexAttr(1), false};
            }
        };
        llvm::SmallVector<mlir::OpFoldResult> offsets(1);
        llvm::SmallVector<mlir::OpFoldResult> sizes(1);
        llvm::SmallVector<mlir::OpFoldResult> strides(1);
        llvm::SmallVector<unsigned> dimsIndices;
        if (auto tupleType = index.getType().dyn_cast<mlir::TupleType>())
        {
            offsets.resize(tupleType.size());
            sizes.resize(tupleType.size());
            strides.resize(tupleType.size());
            for (auto it : llvm::enumerate(tupleType))
            {
                auto i = it.index();
                auto getitem_ind = rewriter.create<mlir::ConstantIndexOp>(loc, it.index());
                auto ind = rewriter.create<plier::GetItemOp>(loc, it.value(), index, getitem_ind);
                bool isSlice = false;
                std::tie(offsets[i], sizes[i], strides[i], isSlice) = getPos(ind.getResult(), static_cast<unsigned>(i));
                if (isSlice)
                {
                    dimsIndices.emplace_back(i);
                }
            }
        }
        else
        {
            bool isSlice = false;
            std::tie(offsets[0], sizes[0], strides[0], isSlice) = getPos(index, 0);
            if (isSlice)
            {
                dimsIndices.emplace_back(0);
            }
        }

        mlir::Value res;
        if (!dimsIndices.empty())
        {
            auto numDims = static_cast<unsigned>(dimsIndices.size());
            auto needReshape = (numDims != type.cast<mlir::ShapedType>().getRank());
            auto elemType = type.cast<mlir::ShapedType>().getElementType();
            if (is_memref)
            {
                if (needReshape)
                {
                    return mlir::failure(); // TODO: not implemented
                }
                res = rewriter.create<mlir::memref::SubViewOp>(loc, value, offsets, sizes, strides);
            }
            else if (is_tensor)
            {
                res = rewriter.create<mlir::SubTensorOp>(loc, value, offsets, sizes, strides);
                if (needReshape)
                {
                    auto resultType = mlir::RankedTensorType::get(llvm::SmallVector<int64_t>(numDims, -1), elemType);
                    llvm::SmallVector<mlir::Value> elements(numDims);
                    for (auto it : llvm::enumerate(dimsIndices))
                    {
                        auto dim = rewriter.create<mlir::memref::DimOp>(loc, value, it.value());
                        elements[it.index()] = dim;
                    }
                    auto shape = rewriter.create<mlir::tensor::FromElementsOp>(loc, elements);
                    res = rewriter.create<mlir::tensor::ReshapeOp>(loc, resultType, res, shape);
                }
            }
            else
            {
                llvm_unreachable("Invalid getitem");
            }
        }
        else
        {
            auto toValues = [](auto vals)
            {
                llvm::SmallVector<mlir::Value> ret(vals.size());
                for (auto it : llvm::enumerate(vals))
                {
                    ret[it.index()] = it.value().template get<mlir::Value>();
                }
                return ret;
            };
            if (is_memref)
            {
                res = rewriter.create<mlir::memref::LoadOp>(loc, value, toValues(offsets));
            }
            else if (is_tensor)
            {
                res = rewriter.create<mlir::tensor::ExtractOp>(loc, value, toValues(offsets));
            }
            else
            {
                llvm_unreachable("Invalid getitem");
            }
        }
        rerun_std_pipeline(op);
        rewriter.replaceOpWithNewOp<plier::CastOp>(op, op.getType(), res);
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

struct SetitemOpLowering : public mlir::OpRewritePattern<plier::SetItemOp>
{
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        plier::SetItemOp op, mlir::PatternRewriter &rewriter) const override
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

        if (auto target_type = get_target_type().dyn_cast<mlir::RankedTensorType>())
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
            auto memref = rewriter.create<mlir::memref::BufferCastOp>(target.getLoc(), memref_type, target);
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
                        auto new_val = rewriter.create<mlir::memref::TensorLoadOp>(use_op->getLoc(), memref);
                        rewriter.updateRootInPlace(use_op, [&]()
                        {
                            use_op->setOperand(use.getOperandNumber(), new_val);
                        });
                    }
                }
            }
        }
        else if (get_target_type().isa<mlir::MemRefType>())
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
        rewriter.create<mlir::memref::StoreOp>(loc, value, target, indices);
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct SliceNoneLowering : public mlir::OpRewritePattern<mlir::SubTensorOp>
{
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::SubTensorOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto source = op.source();
        auto handleVal = [&](mlir::Value val, unsigned argIndex)->mlir::Value
        {
            if (!val)
            {
                return nullptr;
            }
            auto parent = val.getDefiningOp<plier::GetItemOp>();
            if (!parent)
            {
                return nullptr;
            }
            auto buildSlice = parent.value().getDefiningOp<plier::BuildSliceOp>();
            if (!buildSlice)
            {
                return nullptr;
            }
            auto indexAttr = plier::getConstVal<mlir::IntegerAttr>(parent.index());
            if (!indexAttr)
            {
                return nullptr;
            }
            auto index = plier::getIntAttrValue(indexAttr);
            if (index < 0 || index >= 3)
            {
                return nullptr;
            }
            if (!buildSlice.getOperand(static_cast<unsigned>(index)).getType().isa<plier::NoneType>())
            {
                return nullptr;
            }
            if (index == 0)
            {
                // begin
                return rewriter.create<mlir::ConstantIndexOp>(loc, 0);
            }
            else if (index == 1)
            {
                // end
                return rewriter.create<mlir::memref::DimOp>(loc, source, argIndex);
            }
            else // index == 2
            {
                // stride
                return rewriter.create<mlir::ConstantIndexOp>(loc, 1);
            }
        };

        bool changed = false;
        auto tryReplace = [&](mlir::OpFoldResult src, unsigned argIndex)->mlir::OpFoldResult
        {
            if (auto val = handleVal(src.dyn_cast<mlir::Value>(), argIndex))
            {
                changed = true;
                return val;
            }
            return src;
        };

        auto srcOffsets = op.getMixedOffsets();
        auto srcSizes   = op.getMixedSizes();
        auto srcStrides = op.getMixedStrides();

        auto numDims = srcOffsets.size();
        llvm::SmallVector<mlir::OpFoldResult> offsets(numDims);
        llvm::SmallVector<mlir::OpFoldResult> sizes(numDims);
        llvm::SmallVector<mlir::OpFoldResult> strides(numDims);

        for (unsigned i = 0; i < numDims; ++i)
        {
            offsets[i] = tryReplace(srcOffsets[i], i);
            sizes[i]   = tryReplace(srcSizes[i], i);
            strides[i] = tryReplace(srcStrides[i], i);
        }

        if (changed)
        {
            rewriter.replaceOpWithNewOp<mlir::SubTensorOp>(op, op.getType(), source, offsets, sizes, strides);
            return mlir::success();
        }
        return mlir::failure();
    }
};

struct CheckForBuildTuple : public mlir::OpRewritePattern<plier::BuildTupleOp>
{
    using mlir::OpRewritePattern<plier::BuildTupleOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        plier::BuildTupleOp op, mlir::PatternRewriter &/*rewriter*/) const override
    {
        auto tupleType = op.getType().dyn_cast<mlir::TupleType>();
        if (!tupleType)
        {
            rerun_std_pipeline(op);
            return mlir::failure();
        }
        for (auto it : llvm::zip(op.getOperandTypes(), tupleType.getTypes()))
        {
            auto srcType = std::get<0>(it);
            auto dstType = std::get<1>(it);
            if (srcType != dstType && srcType.isa<mlir::ShapedType>())
            {
                rerun_std_pipeline(op);
                break;
            }
        }
        return mlir::failure();
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
            auto dim = rewriter.create<mlir::memref::DimOp>(op.getLoc(), op.value(), i);
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
    if (a1.getRank() != a2.getRank())
    {
        return false;
    }
    for (auto it : llvm::zip(a1.getShape(), a2.getShape()))
    {
        auto s1 = std::get<0>(it);
        auto s2 = std::get<1>(it);
        if (s1 != mlir::ShapedType::kDynamicSize &&
            s2 != mlir::ShapedType::kDynamicSize &&
            s1 != s2)
        {
            return false;
        }
    }
    return true;
}

struct RankedTypesCasts : public mlir::OpRewritePattern<plier::CastOp>
{
    RankedTypesCasts(mlir::TypeConverter& typeConverter,
                     mlir::MLIRContext* context):
        OpRewritePattern(context), converter(typeConverter) {}

    mlir::LogicalResult matchAndRewrite(
        plier::CastOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto src_type = op.value().getType();
        auto dst_type = converter.convertType(op.getType());
        if (!dst_type)
        {
            return mlir::failure();
        }
        if (src_type.isa<mlir::RankedTensorType>() &&
            dst_type.isa<mlir::RankedTensorType>())
        {
            auto src = src_type.cast<mlir::RankedTensorType>();
            auto dst = dst_type.cast<mlir::RankedTensorType>();
            if (!has_compatibale_shape(src,dst))
            {
                return mlir::failure();
            }
            rewriter.replaceOpWithNewOp<mlir::tensor::CastOp>(op, dst, op.value());
            return mlir::success();
        }
        return mlir::failure();
    }
private:
    mlir::TypeConverter& converter;
};

struct UnrankedToElementCasts : public mlir::OpRewritePattern<plier::CastOp>
{
    UnrankedToElementCasts(mlir::TypeConverter& /*type_converter*/,
                           mlir::MLIRContext* context):
        OpRewritePattern(context){}

    mlir::LogicalResult matchAndRewrite(
        plier::CastOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto srcType = op.value().getType();
        auto dstType = op.getType();
        auto isCompatible = [](mlir::Type tensor, mlir::Type element)
        {
            if (auto tensorType = tensor.dyn_cast<mlir::RankedTensorType>())
            {
                return tensorType.getRank() == 0 && tensorType.getElementType() == element;
            }
            return false;
        };
        if (isCompatible(srcType, dstType))
        {
            rewriter.replaceOpWithNewOp<mlir::tensor::ExtractOp>(op, op.value());
            return mlir::success();
        }
        if (isCompatible(dstType, srcType))
        {
            auto singleElemTensor = rewriter.create<mlir::tensor::FromElementsOp>(op.getLoc(), op.value());
            rewriter.replaceOpWithNewOp<mlir::linalg::TensorCollapseShapeOp>(
                op, dstType, singleElemTensor, llvm::ArrayRef<mlir::linalg::ReassociationExprs>{});
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

struct SimplifyExpandDims : public mlir::OpRewritePattern<mlir::linalg::GenericOp>
{
    using mlir::OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::linalg::GenericOp op, mlir::PatternRewriter &rewriter) const override
    {
        if (!op.hasTensorSemantics())
        {
            return mlir::failure();
        }
        if (op.getNumInputs() != 1 || op.getNumOutputs() != 1)
        {
            return mlir::failure();
        }

        auto context = op.getContext();
        auto parallel_attr = mlir::StringAttr::get(context, "parallel");
        if (llvm::any_of(op.iterator_types(), [&](auto attr) { return  attr != parallel_attr; }))
        {
            return mlir::failure();
        }

        auto maps = op.indexing_maps();
        assert(maps.size() == 2);
        auto out_map = maps[1].cast<mlir::AffineMapAttr>().getValue();
        if (!out_map.isIdentity())
        {
            return mlir::failure();
        }
        auto in_map = maps[0].cast<mlir::AffineMapAttr>().getValue();
        auto num_dims = op.getNumLoops();
        if (in_map.getNumResults() != num_dims)
        {
            return mlir::failure();
        }

        bool changed = false;
        auto out_shape = op.getOutputOperand(0)->get().getType().cast<mlir::RankedTensorType>().getShape();
        llvm::SmallVector<mlir::AffineExpr> exprs(num_dims);
        for (unsigned i = 0; i < num_dims; ++i)
        {
            auto prev_expr = in_map.getResult(i);
            bool can_convert = [&]()
            {
                if (out_shape[i] == 1)
                {
                    auto const_expr = prev_expr.dyn_cast<mlir::AffineConstantExpr>();
                    if (const_expr && const_expr.getValue() == 0)
                    {
                        return true;
                    }
                }
                return false;
            }();
            if (can_convert)
            {
                changed = true;
                exprs[i] = mlir::getAffineDimExpr(i, context);
            }
            else
            {
                exprs[i] = prev_expr;
            }
        }

        if (changed)
        {
            const mlir::Attribute new_maps[] = {
                mlir::AffineMapAttr::get(mlir::AffineMap::get(num_dims, 0, exprs, context)),
                maps[1]
            };
            auto new_maps_attr = mlir::ArrayAttr::get(context, new_maps);
            rewriter.updateRootInPlace(op, [&]()
            {
                op.indexing_mapsAttr(new_maps_attr);
            });
        }

        return mlir::success(changed);
    }
};

struct LowerEnforceShape : public mlir::OpRewritePattern<plier::EnforceShapeOp>
{
    using mlir::OpRewritePattern<plier::EnforceShapeOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        plier::EnforceShapeOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto type = op.getType();
        auto src = op.value();
        rewriter.replaceOpWithNewOp<mlir::tensor::CastOp>(op, type, src);
        return mlir::success();
    }
};

void PlierToLinalgPass::runOnOperation()
{
    auto context = &getContext();

    mlir::TypeConverter type_converter;
    // Convert unknown types to itself
    type_converter.addConversion([](mlir::Type type) { return type; });
    populate_std_type_converter(*context, type_converter);
    populate_array_type_converter(*context, type_converter);

    mlir::OwningRewritePatternList patterns(context);
    patterns.insert<
        plier::FuncOpSignatureConversion,
        plier::FixupIfTypes,
        plier::CastOpLowering,
        plier::ArgOpLowering,
        plier::FixCallOmittedArgs,
        RankedTypesCasts,
        UnrankedToElementCasts,
        ArrayShape
        >(type_converter, context);

    CallLowerer callLowerer;

    patterns.insert<
        plier::CallOpLowering,
        GetattrRewriter,
        BinopRewriter
        >(type_converter, context, std::ref(callLowerer));

    patterns.insert<
        GetitemOpLowering,
        SetitemOpLowering,
        SliceNoneLowering,
        CheckForBuildTuple
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
    mlir::OwningRewritePatternList patterns(&getContext());

    patterns.insert<
        mlir::linalg::LinalgLoweringPattern<mlir::linalg::GenericOp>,
        mlir::linalg::LinalgLoweringPattern<mlir::linalg::CopyOp>
        >(&getContext(), mlir::linalg::LinalgLoweringType::ParallelLoops);


    (void)mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

struct PostPlierToLinalgPass :
    public mlir::PassWrapper<PostPlierToLinalgPass, mlir::FunctionPass>
{
    void runOnFunction() override;
};

void PostPlierToLinalgPass::runOnFunction()
{
    auto& context = getContext();
    mlir::OwningRewritePatternList patterns(&context);

    plier::populate_common_opts_patterns(context, patterns);

    patterns.insert<
        SimplifyExpandDims
        >(&context);

    applyOptimizations(getFunction(), std::move(patterns), getAnalysisManager());
}

struct TensorFusionPass :
    public mlir::PassWrapper<TensorFusionPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void runOnOperation() override;
};

void TensorFusionPass::runOnOperation()
{
    auto& context = getContext();
    mlir::OwningRewritePatternList patterns(&context);

    plier::populate_common_opts_patterns(context, patterns);

    patterns.insert<
        SimplifyExpandDims,
        LowerEnforceShape
        >(&context);

    mlir::linalg::populateElementwiseOpsFusionPatterns(patterns);

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


struct BufferizeTuplesRewrite : public mlir::OpRewritePattern<plier::BuildTupleOp>
{
    using mlir::OpRewritePattern<plier::BuildTupleOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        plier::BuildTupleOp op, mlir::PatternRewriter &rewriter) const override
    {
        bool needConversion = false;
        for (auto arg : op.args())
        {
            if (auto tensor = arg.getType().dyn_cast<mlir::TensorType>())
            {
                needConversion = true;
            }
        }

        if (!needConversion)
        {
            return mlir::failure();
        }

        llvm::SmallVector<mlir::Value> newArgs;
        llvm::SmallVector<mlir::Type> newArgsTypes;
        newArgs.reserve(op.args().size());
        newArgsTypes.reserve(op.args().size());
        auto loc = op.getLoc();
        for (auto arg : op.args())
        {
            if (auto tensor = arg.getType().dyn_cast<mlir::TensorType>())
            {
                auto memref = mlir::MemRefType::get(tensor.getShape(), tensor.getElementType());
                auto newOp = rewriter.create<mlir::memref::BufferCastOp>(loc, memref, arg);
                newArgs.emplace_back(newOp);
                newArgsTypes.emplace_back(memref);
            }
            else
            {
                newArgs.emplace_back(arg);
                newArgsTypes.emplace_back(arg.getType());
            }
        }
        auto newType = mlir::TupleType::get(op.getContext(), newArgsTypes);
        rewriter.replaceOpWithNewOp<plier::BuildTupleOp>(op, newType, newArgs);
        return mlir::success();
    }
};

struct FixTupleTypesRewrite : public mlir::OpRewritePattern<plier::BuildTupleOp>
{
    using mlir::OpRewritePattern<plier::BuildTupleOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        plier::BuildTupleOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto srcType = mlir::TupleType::get(op.getContext(), op.args().getTypes());
        if (srcType == op.getType())
        {
            return mlir::failure();
        }

        rewriter.replaceOpWithNewOp<plier::BuildTupleOp>(op, srcType, op.args());
        return mlir::success();
    }
};

struct BufferizeReshapeRewrite : public mlir::OpRewritePattern<mlir::tensor::ReshapeOp>
{
    using mlir::OpRewritePattern<mlir::tensor::ReshapeOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::tensor::ReshapeOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto getType = [&](mlir::Type type)
        {
            auto shapedType = type.cast<mlir::ShapedType>();
            return mlir::MemRefType::get(shapedType.getShape(), shapedType.getElementType());
        };
        auto sourceType = getType(op.source().getType());
        auto shapeType = getType(op.shape().getType());
        auto resType = getType(op.getType());
        auto source = rewriter.createOrFold<mlir::memref::BufferCastOp>(loc, sourceType, op.source());
        auto shape = rewriter.createOrFold<mlir::memref::BufferCastOp>(loc, shapeType, op.shape());
        auto res = rewriter.create<mlir::memref::ReshapeOp>(loc, resType, source, shape);
        rewriter.replaceOpWithNewOp<mlir::memref::TensorLoadOp>(op, res);
        return mlir::success();
    }
};

struct FixDeallocPlacement : public mlir::OpRewritePattern<mlir::memref::DeallocOp>
{
    using mlir::OpRewritePattern<mlir::memref::DeallocOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::memref::DeallocOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto block = op->getBlock();
        auto blockIt = mlir::Block::iterator(op);
        mlir::Operation* newPos = op;
        ++blockIt;
        auto memref = op.memref();
        for (auto& it : llvm::make_range(blockIt, block->end()))
        {
            auto visitor = [&](mlir::Operation* inner)
            {
                for (auto arg : inner->getOperands())
                {
                    if (arg == memref)
                    {
                        return mlir::WalkResult::interrupt();
                    }
                }
                return mlir::WalkResult::advance();
            };
            if (it.walk(visitor).wasInterrupted())
            {
                newPos = &it;
            }
        }

        if (newPos != op)
        {
            rewriter.setInsertionPointAfter(newPos);
            rewriter.create<mlir::memref::DeallocOp>(op.getLoc(), memref);
            rewriter.eraseOp(op);
            return mlir::success();
        }
        return mlir::failure();
    }
};


struct AdditionalBufferize :
    public mlir::PassWrapper<AdditionalBufferize, mlir::FunctionPass>
{
    virtual void getDependentDialects(
        mlir::DialectRegistry &registry) const override
    {
        registry.insert<plier::PlierDialect>();
    }

    void runOnFunction() override;
};

void AdditionalBufferize::runOnFunction()
{
    auto& context = getContext();
    mlir::OwningRewritePatternList patterns(&context);

    patterns.insert<
        BufferizeTuplesRewrite,
        FixTupleTypesRewrite,
        BufferizeReshapeRewrite
        >(&context);

    auto func = getFunction();
    (void)mlir::applyPatternsAndFoldGreedily(func, std::move(patterns));
}

struct CloneArgsPass :
    public mlir::PassWrapper<CloneArgsPass, mlir::FunctionPass>
{
    virtual void getDependentDialects(
        mlir::DialectRegistry &registry) const override
    {
        registry.insert<plier::PlierDialect>();
    }

    void runOnFunction() override;
};

void CloneArgsPass::runOnFunction()
{
    auto func = getFunction();
    if (func.isPrivate() || func.isDeclaration() || func.body().empty())
    {
        return;
    }

    mlir::OpBuilder builder(&getContext());
    auto loc = builder.getUnknownLoc();
    auto block = &func.body().front();
    builder.setInsertionPointToStart(block);
    for (auto arg : block->getArguments())
    {
        if (arg.getType().isa<mlir::MemRefType>())
        {
            auto retained = builder.create<mlir::memref::CloneOp>(loc, arg);
            llvm::SmallPtrSet<mlir::Operation*, 1> except({retained});
            arg.replaceAllUsesExcept(retained, except);
        }
    }
}

struct LowerCloneOpsPass :
    public mlir::PassWrapper<LowerCloneOpsPass, mlir::FunctionPass>
{
    void runOnFunction() override;
};

struct ReplaceClones : public mlir::OpRewritePattern<mlir::memref::CloneOp>
{
    using mlir::OpRewritePattern<mlir::memref::CloneOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::memref::CloneOp op, mlir::PatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<plier::RetainOp>(op, op.getSource());
        return mlir::success();
    }
};

void LowerCloneOpsPass::runOnFunction()
{
    auto& context = getContext();
    mlir::OwningRewritePatternList patterns(&context);

    patterns.insert<
        ReplaceClones
        >(&context);


    auto func = getFunction();
    (void)mlir::applyPatternsAndFoldGreedily(func, std::move(patterns));
}

struct PostLinalgOptPass :
    public mlir::PassWrapper<PostLinalgOptPass, mlir::FunctionPass>
{
    void runOnFunction() override;
};

void PostLinalgOptPass::runOnFunction()
{
    auto& context = getContext();
    mlir::OwningRewritePatternList patterns(&context);

    plier::populate_common_opts_patterns(context, patterns);

    patterns.insert<
        plier::CanonicalizeReduction,
        FixDeallocPlacement
        >(&context);

    applyOptimizations(getFunction(), std::move(patterns), getAnalysisManager(),
                       [](mlir::FuncOp op)
    {
        return plier::naivelyFuseParallelOps(op.getRegion());
    });
}

struct PromoteParallelPass :
    public mlir::PassWrapper<PromoteParallelPass, mlir::FunctionPass>
{
    void runOnFunction() override;
};

void PromoteParallelPass::runOnFunction()
{
    auto& context = getContext();
    mlir::OwningRewritePatternList patterns(&context);

    plier::populate_common_opts_patterns(context, patterns);

    patterns.insert<
        plier::CanonicalizeReduction,
        plier::PromoteToParallel,
        plier::MergeNestedForIntoParallel
        >(&context);

    applyOptimizations(getFunction(), std::move(patterns), getAnalysisManager());
}

void populate_plier_to_linalg_gen_pipeline(mlir::OpPassManager& pm)
{
    pm.addPass(std::make_unique<PlierToLinalgPass>());
    pm.addNestedPass<mlir::FuncOp>(std::make_unique<PostPlierToLinalgPass>());
    pm.addPass(mlir::createSymbolDCEPass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createCSEPass());
}

void populate_plier_to_linalg_opt_pipeline(mlir::OpPassManager& pm)
{
    pm.addPass(std::make_unique<TensorFusionPass>());

    pm.addPass(mlir::createTensorConstantBufferizePass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createSCFBufferizePass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createLinalgBufferizePass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createStdBufferizePass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createTensorBufferizePass());
    pm.addPass(mlir::createFuncBufferizePass());
    pm.addNestedPass<mlir::FuncOp>(std::make_unique<AdditionalBufferize>());
    pm.addNestedPass<mlir::FuncOp>(mlir::createFinalizingBufferizePass());

    pm.addNestedPass<mlir::FuncOp>(mlir::createBufferHoistingPass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createBufferLoopHoistingPass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createPromoteBuffersToStackPass());

    pm.addNestedPass<mlir::FuncOp>(std::make_unique<CloneArgsPass>());
    pm.addNestedPass<mlir::FuncOp>(mlir::createBufferDeallocationPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::FuncOp>(std::make_unique<LowerCloneOpsPass>());

    pm.addPass(std::make_unique<LowerLinalgPass>());
    pm.addNestedPass<mlir::FuncOp>(std::make_unique<PostLinalgOptPass>());
    pm.addPass(mlir::createSymbolDCEPass());
    pm.addNestedPass<mlir::FuncOp>(std::make_unique<PromoteParallelPass>());
}
}

void populate_array_type_converter(mlir::MLIRContext& /*context*/, mlir::TypeConverter& converter)
{
    converter.addConversion([&](plier::PyType type)->llvm::Optional<mlir::Type>
    {
        auto ret =  map_plier_type(converter, type);
        if (!ret)
        {
            return llvm::None;
        }
        return ret;
    });
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
