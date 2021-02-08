#include "pipelines/lower_to_llvm.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/Triple.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/Host.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>

#include "plier/dialect.hpp"

#include "plier/transforms/func_utils.hpp"

#include "base_pipeline.hpp"
#include "plier/compiler/pipeline_registry.hpp"

#include "plier/utils.hpp"

namespace
{
const mlir::LowerToLLVMOptions &getLLVMOptions()
{
    static mlir::LowerToLLVMOptions options = []()
    {
        llvm::InitializeNativeTarget();
        auto triple = llvm::sys::getProcessTriple();
        std::string err_str;
        auto target = llvm::TargetRegistry::lookupTarget(triple, err_str);
        if (nullptr == target)
        {
            plier::report_error(llvm::Twine("Unable to get target: ") + err_str);
        }
        llvm::TargetOptions target_opts;
        std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(triple, llvm::sys::getHostCPUName(), "", target_opts, llvm::None));
        mlir::LowerToLLVMOptions opts;
        opts.dataLayout = machine->createDataLayout();
        opts.useBarePtrCallConv = true;
        return opts;
    }();
    return options;
}

struct LLVMTypeHelper
{
    LLVMTypeHelper(mlir::MLIRContext& ctx):
        type_converter(&ctx) {}

    mlir::Type i(unsigned bits)
    {
        return mlir::IntegerType::get(&type_converter.getContext(), bits);
    }

    mlir::Type ptr(mlir::Type type)
    {
        assert(static_cast<bool>(type));
        auto ll_type = type_converter.convertType(type);
        assert(static_cast<bool>(ll_type));
        return mlir::LLVM::LLVMPointerType::get(ll_type);
    }

    mlir::MLIRContext& get_context()
    {
        return type_converter.getContext();
    }

    mlir::LLVMTypeConverter& get_type_converter()
    {
        return type_converter;
    }

private:
    mlir::LLVMTypeConverter type_converter;
};

mlir::Type getExceptInfoType(LLVMTypeHelper& type_helper)
{
    mlir::Type elems[] = {
        type_helper.ptr(type_helper.i(8)),
        type_helper.i(32),
        type_helper.ptr(type_helper.i(8)),
    };
    return mlir::LLVM::LLVMStructType::getLiteral(&type_helper.get_context(), elems);
}

mlir::LLVM::LLVMStructType get_array_type(mlir::TypeConverter& converter, mlir::MemRefType type)
{
    assert(type);
    auto ctx = type.getContext();
    auto i8p = mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(ctx, 8));
    auto i64 = mlir::IntegerType::get(ctx, 64);
    auto data_type = converter.convertType(type.getElementType());
    assert(data_type);
    auto shape_type = mlir::LLVM::LLVMArrayType::get(i64, static_cast<unsigned>(type.getRank()));
    const mlir::Type members[] = {
        i8p, // 0, meminfo
        i8p, // 1, parent
        i64, // 2, nitems
        i64, // 3, itemsize
        mlir::LLVM::LLVMPointerType::get(data_type), // 4, data
        shape_type, // 5, shape
        shape_type, // 6, strides
    };
    return mlir::LLVM::LLVMStructType::getLiteral(ctx, members);
}

template<typename F>
void flatten_type(mlir::Type type, F&& func)
{
    if (auto struct_type = type.dyn_cast<mlir::LLVM::LLVMStructType>())
    {
        for (auto elem : struct_type.getBody())
        {
            flatten_type(elem, std::forward<F>(func));
        }
    }
    else if (auto arr_type = type.dyn_cast<mlir::LLVM::LLVMArrayType>())
    {
        auto elem = arr_type.getElementType();
        auto size = arr_type.getNumElements();
        for (unsigned i = 0 ; i < size; ++i)
        {
            flatten_type(elem, std::forward<F>(func));
        }
    }
    else
    {
        func(type);
    }
}

template<typename F>
mlir::Value unflatten(mlir::Type type, mlir::Location loc, mlir::OpBuilder& builder, F&& next_func)
{
    namespace mllvm = mlir::LLVM;
    if (auto struct_type = type.dyn_cast<mlir::LLVM::LLVMStructType>())
    {
        mlir::Value val = builder.create<mllvm::UndefOp>(loc, struct_type);
        for (auto elem : llvm::enumerate(struct_type.getBody()))
        {
            auto elem_index = builder.getI64ArrayAttr(static_cast<int64_t>(elem.index()));
            auto elem_type = elem.value();
            auto elem_val = unflatten(elem_type, loc, builder, std::forward<F>(next_func));
            val = builder.create<mlir::LLVM::InsertValueOp>(loc, val, elem_val, elem_index);
        }
        return val;
    }
    else if (auto arr_type = type.dyn_cast<mlir::LLVM::LLVMArrayType>())
    {
        auto elem_type = arr_type.getElementType();
        auto size = arr_type.getNumElements();
        mlir::Value val = builder.create<mllvm::UndefOp>(loc, arr_type);
        for (unsigned i = 0 ; i < size; ++i)
        {
            auto elem_index = builder.getI64ArrayAttr(static_cast<int64_t>(i));
            auto elem_val = unflatten(elem_type, loc, builder, std::forward<F>(next_func));
            val = builder.create<mlir::LLVM::InsertValueOp>(loc, val, elem_val, elem_index);
        }
        return val;
    }
    else
    {
        return next_func();
    }
}

std::string gen_conversion_func_name(mlir::MemRefType memref_type)
{
    assert(memref_type);
    std::string ret;
    llvm::raw_string_ostream ss(ret);
    ss << "__convert_memref_";
    memref_type.getElementType().print(ss);
    ss.flush();
    return ret;
}

struct MemRefConversionCache
{
    mlir::FuncOp get_conversion_func(
        mlir::ModuleOp module, mlir::OpBuilder& builder, mlir::MemRefType memref_type,
        mlir::LLVM::LLVMStructType src_type, mlir::LLVM::LLVMStructType dst_type)
    {
        assert(memref_type);
        assert(src_type);
        assert(dst_type);
        auto it = cache.find(memref_type);
        if (it != cache.end())
        {
            auto func = it->second;
            assert(func.getType().getNumResults() == 1);
            assert(func.getType().getResult(0) == dst_type);
            return func;
        }
        auto func_name = gen_conversion_func_name(memref_type);
        auto func_type = mlir::FunctionType::get(builder.getContext(),src_type, dst_type);
        auto loc = builder.getUnknownLoc();
        auto new_func = plier::add_function(builder, module, func_name, func_type);
        auto alwaysinline = mlir::StringAttr::get("alwaysinline", builder.getContext());
        new_func->setAttr("passthrough", mlir::ArrayAttr::get(alwaysinline, builder.getContext()));
        cache.insert({memref_type, new_func});
        mlir::OpBuilder::InsertionGuard guard(builder);
        auto block = new_func.addEntryBlock();
        builder.setInsertionPointToStart(block);
        namespace mllvm = mlir::LLVM;
        mlir::Value arg = block->getArgument(0);
        auto extract = [&](unsigned index)
        {
            auto res_type = src_type.getBody()[index];
            auto i = builder.getI64ArrayAttr(index);
            return builder.create<mllvm::ExtractValueOp>(loc, res_type, arg, i);
        };
        auto ptr = extract(4);
        auto shape = extract(5);
        auto strides = extract(6);
        auto i64 = mlir::IntegerType::get(builder.getContext(), 64);
        auto offset = builder.create<mllvm::ConstantOp>(loc, i64, builder.getI64IntegerAttr(0));
        mlir::Value res = builder.create<mllvm::UndefOp>(loc, dst_type);
        auto insert = [&](unsigned index, mlir::Value val)
        {
            auto i = builder.getI64ArrayAttr(index);
            res = builder.create<mllvm::InsertValueOp>(loc, res, val, i);
        };
        insert(0, ptr);
        insert(1, ptr);
        insert(2, offset);
        insert(3, shape);
        insert(4, strides);
        builder.create<mllvm::ReturnOp>(loc, res);
        return new_func;
    }
private:
    llvm::DenseMap<mlir::Type, mlir::FuncOp> cache;
};

mlir::Attribute get_fastmath_attrs(mlir::MLIRContext& ctx)
{
    auto add_pair = [&](auto name, auto val)
    {
        const mlir::Attribute attrs[] = {
            mlir::StringAttr::get(name, &ctx),
            mlir::StringAttr::get(val, &ctx)
        };
        return mlir::ArrayAttr::get(attrs, &ctx);
    };
    const mlir::Attribute attrs[] = {
        add_pair("denormal-fp-math", "preserve-sign,preserve-sign"),
        add_pair("denormal-fp-math-f32", "ieee,ieee"),
        add_pair("no-infs-fp-math", "true"),
        add_pair("no-nans-fp-math", "true"),
        add_pair("no-signed-zeros-fp-math", "true"),
        add_pair("unsafe-fp-math", "true"),
        add_pair(plier::attributes::getFastmathName(), "1"),
    };
    return mlir::ArrayAttr::get(attrs, &ctx);
}

void fix_func_sig(LLVMTypeHelper& type_helper, mlir::FuncOp func)
{
    if (func.isPrivate())
    {
        return;
    }
    if (func->getAttr(plier::attributes::getFastmathName()))
    {
        func->setAttr("passthrough", get_fastmath_attrs(*func.getContext()));
    }
    auto old_type = func.getType();
    assert(old_type.getNumResults() <= 1);
    auto& ctx = *old_type.getContext();
    llvm::SmallVector<mlir::Type, 8> args;

    auto ptr = [&](auto arg)
    {
        return type_helper.ptr(arg);
    };

    unsigned index = 0;
    auto add_arg = [&](mlir::Type type)
    {
        args.push_back(type);
        auto ret = func.getBody().insertArgument(index, type);
        ++index;
        return ret;
    };

    MemRefConversionCache conversion_cache;

    mlir::OpBuilder builder(&ctx);
    builder.setInsertionPointToStart(&func.getBody().front());

    auto loc = builder.getUnknownLoc();
    llvm::SmallVector<mlir::Value, 8> new_args;
    auto process_arg = [&](mlir::Type type)
    {
        if (auto memref_type = type.dyn_cast<mlir::MemRefType>())
        {
            new_args.clear();
            auto arr_type = get_array_type(type_helper.get_type_converter(), memref_type);
            flatten_type(arr_type, [&](mlir::Type new_type)
            {
                new_args.push_back(add_arg(new_type));
            });
            auto it = new_args.begin();
            mlir::Value desc = unflatten(arr_type, loc, builder, [&]()
            {
                auto ret = *it;
                ++it;
                return ret;
            });

            auto mod = mlir::cast<mlir::ModuleOp>(func->getParentOp());
            auto dst_type = type_helper.get_type_converter().convertType(memref_type);
            assert(dst_type);
            auto conv_func = conversion_cache.get_conversion_func(mod, builder, memref_type, arr_type, dst_type.cast<mlir::LLVM::LLVMStructType>());
            auto converted = builder.create<mlir::CallOp>(loc, conv_func, desc).getResult(0);
            auto casted = builder.create<plier::CastOp>(loc, memref_type, converted);
            func.getBody().getArgument(index).replaceAllUsesWith(casted);
            func.getBody().eraseArgument(index);
        }
        else
        {
            args.push_back(type);
            ++index;
        }
    };

    auto orig_ret_type = (old_type.getNumResults() != 0 ? old_type.getResult(0) : type_helper.ptr(type_helper.i(8)));
    add_arg(ptr(orig_ret_type));
    add_arg(ptr(ptr(getExceptInfoType(type_helper))));

    auto old_args = old_type.getInputs();
    for (auto arg : old_args)
    {
        process_arg(arg);
    }
    auto ret_type = mlir::IntegerType::get(&ctx, 32);
    func.setType(mlir::FunctionType::get(&ctx, args, ret_type));
}

struct ReturnOpLowering : public mlir::OpRewritePattern<mlir::ReturnOp>
{
    ReturnOpLowering(mlir::MLIRContext* ctx, mlir::TypeConverter& converter):
        OpRewritePattern(ctx), type_converter(converter) {}

    mlir::LogicalResult matchAndRewrite(mlir::ReturnOp op,
                                        mlir::PatternRewriter& rewriter) const
    {
        auto parent = op->getParentOfType<mlir::FuncOp>();
        if (nullptr == parent || parent.isPrivate())
        {
            return mlir::failure();
        }

        auto insert_ret = [&]()
        {
            auto ctx = op.getContext();
            auto ret_type = mlir::IntegerType::get(ctx, 32);
            auto ll_ret_type = mlir::IntegerType::get(ctx, 32);
            mlir::Value ret = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), ll_ret_type, mlir::IntegerAttr::get(ret_type, 0));
            rewriter.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(op, ret);
        };

        rewriter.setInsertionPoint(op);
        auto addr = op->getParentRegion()->front().getArgument(0);
        if (op.getNumOperands() == 0)
        {
            assert(addr.getType().isa<mlir::LLVM::LLVMPointerType>());
            auto null_type = addr.getType().cast<mlir::LLVM::LLVMPointerType>().getElementType();
            auto ll_val = rewriter.create<mlir::LLVM::NullOp>(op.getLoc(), null_type);
            rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), ll_val, addr);
            insert_ret();
            return mlir::success();
        }
        else if (op.getNumOperands() == 1)
        {
            auto val = op.getOperand(0);
            auto ll_ret_type = type_converter.convertType(val.getType());
            assert(static_cast<bool>(ll_ret_type));
            auto ll_val = rewriter.create<mlir::LLVM::BitcastOp>(op.getLoc(), ll_ret_type, val); // TODO: hack to make verifier happy
            rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), ll_val, addr);
            insert_ret();
            return mlir::success();
        }
        else
        {
            return mlir::failure();
        }
    }

private:
    mlir::TypeConverter& type_converter;
};

// Remove redundant bitcasts we have created on PreLowering
struct RemoveBitcasts : public mlir::OpRewritePattern<mlir::LLVM::BitcastOp>
{
    using mlir::OpRewritePattern<mlir::LLVM::BitcastOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::LLVM::BitcastOp op,
                                        mlir::PatternRewriter& rewriter) const
    {
        if (op.getType() == op.getOperand().getType())
        {
            rewriter.replaceOp(op, op.getOperand());
            return mlir::success();
        }
        return mlir::failure();
    }
};

template<typename Op>
struct ApplyFastmathFlags : public mlir::OpRewritePattern<Op>
{
    using mlir::OpRewritePattern<Op>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        Op op, mlir::PatternRewriter& rewriter) const
    {
        auto parent = mlir::cast<mlir::LLVM::LLVMFuncOp>(op->getParentOp());
        bool changed = false;

        rewriter.startRootUpdate(op);
        auto fmf = op.fastmathFlags();
        getFastmathFlags(parent, [&](auto flag)
        {
            if (!mlir::LLVM::bitEnumContains(fmf, flag))
            {
                fmf = fmf | flag;
                changed = true;
            }
        });
        if (changed)
        {
            op.fastmathFlagsAttr(mlir::LLVM::FMFAttr::get(fmf, op.getContext()));
            rewriter.finalizeRootUpdate(op);
        }
        else
        {
            rewriter.cancelRootUpdate(op);
        }

        return mlir::success(changed);
    }

private:
    template<typename F>
    static void getFastmathFlags(mlir::LLVM::LLVMFuncOp func, F&& sink)
    {
        if (func->hasAttr(plier::attributes::getFastmathName()))
        {
            sink(mlir::LLVM::FastmathFlags::fast);
        }
    }
};

class CheckForPlierTypes :
    public mlir::PassWrapper<CheckForPlierTypes, mlir::OperationPass<void>>
{
    void runOnOperation() override
    {
        markAllAnalysesPreserved();
        getOperation()->walk([&](mlir::Operation* op)
        {
            if (op->getName().getDialect() == plier::PlierDialect::getDialectNamespace())
            {
                op->emitOpError(": not all plier ops were translated\n");
                signalPassFailure();
                return;
            }

            auto check_type = [](mlir::Type type)
            {
                return type.isa<plier::PyType>();
            };

            if (llvm::any_of(op->getResultTypes(), check_type) ||
                llvm::any_of(op->getOperandTypes(), check_type))
            {
                op->emitOpError(": plier types weren't translated\n");
                signalPassFailure();
            }
        });
    }
};

class LLVMFunctionPass : public mlir::OperationPass<mlir::LLVM::LLVMFuncOp>
{
public:
  using OperationPass<mlir::LLVM::LLVMFuncOp>::OperationPass;

  /// The polymorphic API that runs the pass over the currently held function.
  virtual void runOnFunction() = 0;

  /// The polymorphic API that runs the pass over the currently held operation.
  void runOnOperation() final {
    if (!getFunction().isExternal())
      runOnFunction();
  }

  /// Return the current function being transformed.
  mlir::LLVM::LLVMFuncOp getFunction() { return this->getOperation(); }
};

void copyAttrs(mlir::Operation* src, mlir::Operation* dst)
{
    const mlir::StringRef attrs[] = {
        plier::attributes::getFastmathName(),
        plier::attributes::getParallelName(),
        plier::attributes::getMaxConcurrencyName(),
    };
    for (auto name : attrs)
    {
        if (auto attr = src->getAttr(name))
        {
            dst->setAttr(name, attr);
        }
    }
}

struct LowerParallel : public mlir::OpRewritePattern<plier::ParallelOp>
{
    LowerParallel(mlir::MLIRContext* context):
        OpRewritePattern(context),
        converter(context) {}

    mlir::LogicalResult
    matchAndRewrite(plier::ParallelOp op,
                    mlir::PatternRewriter &rewriter) const override {
        llvm::SmallVector<mlir::Value, 8> context_vars;
        llvm::SmallVector<mlir::Operation*, 8> context_constants;
        llvm::DenseSet<mlir::Value> context_vars_set;
        auto add_context_var = [&](mlir::Value value)
        {
            if (0 != context_vars_set.count(value))
            {
                return;
            }
            context_vars_set.insert(value);
            if (auto op = value.getDefiningOp())
            {
                mlir::ConstantOp a;
                if (op->hasTrait<mlir::OpTrait::ConstantLike>())
                {
                    context_constants.emplace_back(op);
                    return;
                }
            }
            context_vars.emplace_back(value);
        };

        auto is_defined_inside = [&](mlir::Value value)
        {
            auto& this_region = op.getLoopBody();
            auto op_region = value.getParentRegion();
            assert(nullptr != op_region);
            do
            {
                if (op_region == &this_region)
                {
                    return true;
                }
                op_region = op_region->getParentRegion();
            }
            while (nullptr != op_region);
            return false;
        };

        if (op->walk([&](mlir::Operation* inner)->mlir::WalkResult
        {
            if (op != inner)
            {
                for (auto arg : inner->getOperands())
                {
                    if (!is_defined_inside(arg))
                    {
                        add_context_var(arg);
                    }
                }
            }
            return mlir::WalkResult::advance();
        }).wasInterrupted())
        {
            return mlir::failure();
        }

        auto context_type = [&]()->mlir::LLVM::LLVMStructType
        {
            llvm::SmallVector<mlir::Type, 8> fields;
            fields.reserve(context_vars.size());
            for (auto var : context_vars)
            {
                auto type = converter.convertType(var.getType());
                if (!type)
                {
                    return {};
                }
                fields.emplace_back(type);
            }
            return mlir::LLVM::LLVMStructType::getLiteral(op.getContext(), fields);
        }();

        if (!context_type)
        {
            return mlir::failure();
        }
        auto context_ptr_type = mlir::LLVM::LLVMPointerType::get(context_type);

        auto loc = op.getLoc();
        auto llvm_i32_type = mlir::IntegerType::get(op.getContext(), 32);
        auto zero = rewriter.create<mlir::LLVM::ConstantOp>(loc, llvm_i32_type, rewriter.getI32IntegerAttr(0));
        auto one = rewriter.create<mlir::LLVM::ConstantOp>(loc, llvm_i32_type, rewriter.getI32IntegerAttr(1));
        auto context = rewriter.create<mlir::LLVM::AllocaOp>(loc, context_ptr_type, one, 0);
        for (auto it : llvm::enumerate(context_vars))
        {
            auto type = context_type.getBody()[it.index()];
            auto llvm_val = rewriter.create<plier::CastOp>(loc, type, it.value());
            auto i = rewriter.getI32IntegerAttr(static_cast<int32_t>(it.index()));
            mlir::Value indices[] = {
                zero,
                rewriter.create<mlir::LLVM::ConstantOp>(loc, llvm_i32_type, i)
            };
            auto pointer_type = mlir::LLVM::LLVMPointerType::get(type);
            auto ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, pointer_type, context, indices);
            rewriter.create<mlir::LLVM::StoreOp>(loc, llvm_val, ptr);
        }
        auto void_ptr_type = mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(op.getContext(), 8));
        auto context_abstract = rewriter.create<mlir::LLVM::BitcastOp>(loc, void_ptr_type, context);

        auto index_type = rewriter.getIndexType();
        auto func_type = [&]()
        {
            mlir::Type args[] = {
                index_type, // lower_bound
                index_type, // upper_bound
                index_type, // thread index
                void_ptr_type // context
            };
            return mlir::FunctionType::get(op.getContext(), args, {});
        }();

        auto mod = op->getParentOfType<mlir::ModuleOp>();
        auto outlined_func = [&]()->mlir::FuncOp
        {
            auto func = [&]()
            {
                auto parent_func = op->getParentOfType<mlir::FuncOp>();
                assert(parent_func);
                auto func_name = [&]()
                {
                    auto old_name = parent_func.getName();
                    for (int i = 0;;++i)
                    {
                        auto name = (0 == i ?
                            (llvm::Twine(old_name) + "_outlined").str() :
                            (llvm::Twine(old_name) + "_outlined_" + llvm::Twine(i)).str());
                        if (!mod.lookupSymbol<mlir::FuncOp>(name))
                        {
                            return name;
                        }
                    }
                }();

                auto func = plier::add_function(rewriter, mod, func_name, func_type);
                copyAttrs(parent_func, func);
                return func;
            }();
            mlir::BlockAndValueMapping mapping;
            auto& old_entry = op.getLoopBody().front();
            auto entry = func.addEntryBlock();
            auto loc = rewriter.getUnknownLoc();
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            mapping.map(old_entry.getArgument(0), entry->getArgument(0));
            mapping.map(old_entry.getArgument(1), entry->getArgument(1));
            mapping.map(old_entry.getArgument(2), entry->getArgument(2));
            rewriter.setInsertionPointToStart(entry);
            for (auto arg : context_constants)
            {
                rewriter.clone(*arg, mapping);
            }
            auto context_ptr = rewriter.create<mlir::LLVM::BitcastOp>(loc, context_ptr_type, entry->getArgument(3));
            auto zero = rewriter.create<mlir::LLVM::ConstantOp>(loc, llvm_i32_type, rewriter.getI32IntegerAttr(0));
            for (auto it : llvm::enumerate(context_vars))
            {
                auto index = it.index();
                auto old_val = it.value();
                mlir::Value indices[] = {
                    zero,
                    rewriter.create<mlir::LLVM::ConstantOp>(loc, llvm_i32_type, rewriter.getI32IntegerAttr(static_cast<int32_t>(index)))
                };
                auto pointer_type = mlir::LLVM::LLVMPointerType::get(context_type.getBody()[index]);
                auto ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, pointer_type, context_ptr, indices);
                auto llvm_val = rewriter.create<mlir::LLVM::LoadOp>(loc, ptr);
                auto val = rewriter.create<plier::CastOp>(loc, old_val.getType(), llvm_val);
                mapping.map(old_val, val);
            }
            op.getLoopBody().cloneInto(&func.getBody(), mapping);
            auto& orig_entry = *std::next(func.getBody().begin());
            rewriter.create<mlir::BranchOp>(loc, &orig_entry);
            for (auto& block : func.getBody())
            {
                if (auto term = mlir::dyn_cast<plier::YieldOp>(block.getTerminator()))
                {
                    rewriter.eraseOp(term);
                    rewriter.setInsertionPointToEnd(&block);
                    rewriter.create<mlir::ReturnOp>(loc);
                }
            }
            return func;
        }();

        auto parallel_for = [&]()
        {
            auto func_name = "numba_parallel_for2";
            if (auto sym = mod.lookupSymbol<mlir::FuncOp>(func_name))
            {
                return sym;
            }
            mlir::Type args[] = {
                index_type, // lower bound
                index_type, // upper bound
                index_type, // step
                func_type,
                void_ptr_type
            };
            auto func_type = mlir::FunctionType::get(op.getContext(), args, {});
            return plier::add_function(rewriter, mod, func_name, func_type);
        }();
        auto func_addr = rewriter.create<mlir::ConstantOp>(loc, func_type, rewriter.getSymbolRefAttr(outlined_func));
        mlir::Value pf_args[] = {
            op.lowerBound(),
            op.upperBound(),
            op.step(),
            func_addr,
            context_abstract
        };
        rewriter.create<mlir::CallOp>(loc, parallel_for, pf_args);
        rewriter.eraseOp(op);
        return mlir::success();
    }

private:
    mutable mlir::LLVMTypeConverter converter; // TODO
};

struct LowerParallelToCFGPass :
    public mlir::PassWrapper<LowerParallelToCFGPass, mlir::OperationPass<void>>
{
    virtual void getDependentDialects(
        mlir::DialectRegistry &registry) const override
    {
        registry.insert<mlir::StandardOpsDialect>();
        registry.insert<mlir::LLVM::LLVMDialect>();
    }

    void runOnOperation() override final
    {
        mlir::OwningRewritePatternList patterns;
        patterns.insert<LowerParallel>(&getContext());

        mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }
};

struct PreLLVMLowering : public mlir::PassWrapper<PreLLVMLowering, mlir::FunctionPass>
{
    virtual void getDependentDialects(
        mlir::DialectRegistry &registry) const override
    {
        registry.insert<mlir::StandardOpsDialect>();
        registry.insert<mlir::LLVM::LLVMDialect>();
    }

    void runOnFunction() override final
    {
        LLVMTypeHelper type_helper(getContext());

        mlir::OwningRewritePatternList patterns;
        auto func = getFunction();
        fix_func_sig(type_helper, func);

        patterns.insert<ReturnOpLowering>(&getContext(),
                                          type_helper.get_type_converter());

        mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }
};

struct PostLLVMLowering :
    public mlir::PassWrapper<PostLLVMLowering, LLVMFunctionPass>
{
    virtual void getDependentDialects(
        mlir::DialectRegistry &registry) const override
    {
        registry.insert<mlir::LLVM::LLVMDialect>();
    }

    void runOnFunction() override final
    {
        mlir::OwningRewritePatternList patterns;

        patterns.insert<
            RemoveBitcasts,
            ApplyFastmathFlags<mlir::LLVM::FAddOp>,
            ApplyFastmathFlags<mlir::LLVM::FSubOp>,
            ApplyFastmathFlags<mlir::LLVM::FMulOp>,
            ApplyFastmathFlags<mlir::LLVM::FDivOp>,
            ApplyFastmathFlags<mlir::LLVM::FRemOp>,
            ApplyFastmathFlags<mlir::LLVM::FCmpOp>,
            ApplyFastmathFlags<mlir::LLVM::CallOp>
            >(&getContext());

        (void)mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }
};

struct LowerCasts : public mlir::OpConversionPattern<plier::CastOp>
{
    using mlir::OpConversionPattern<plier::CastOp>::OpConversionPattern;

    mlir::LogicalResult
    matchAndRewrite(plier::CastOp op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
        assert(operands.size() == 1);
        auto converter = getTypeConverter();
        assert(nullptr != converter);
        auto src_type = operands[0].getType();
        auto dst_type = converter->convertType(op.getType());
        if (src_type == dst_type)
        {
            rewriter.replaceOp(op, operands[0]);
            return mlir::success();
        }
        return mlir::failure();
    }
};

// Copypasted from mlir
struct LLVMLoweringPass : public mlir::PassWrapper<LLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
  LLVMLoweringPass(const mlir::LowerToLLVMOptions& opts):
    options(opts) {}

  /// Run the dialect converter on the module.
  void runOnOperation() override {
    using namespace mlir;
    if (options.useBarePtrCallConv && options.emitCWrappers) {
      getOperation().emitError()
          << "incompatible conversion options: bare-pointer calling convention "
             "and C wrapper emission";
      signalPassFailure();
      return;
    }
    if (failed(LLVM::LLVMDialect::verifyDataLayoutString(
            options.dataLayout.getStringRepresentation(), [this](const Twine &message) {
              getOperation().emitError() << message.str();
            }))) {
      signalPassFailure();
      return;
    }

    ModuleOp m = getOperation();

    LLVMTypeConverter typeConverter(&getContext(), options);

    OwningRewritePatternList patterns;
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    patterns.insert<LowerCasts>(typeConverter, &getContext());

    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
    m->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
               StringAttr::get(options.dataLayout.getStringRepresentation(), m.getContext()));
  }

private:
  mlir::LowerToLLVMOptions options;
};

void populate_lower_to_llvm_pipeline(mlir::OpPassManager& pm)
{
    pm.addPass(std::make_unique<LowerParallelToCFGPass>());
    pm.addPass(mlir::createLowerToCFGPass());
//    pm.addPass(std::make_unique<CheckForPlierTypes>());
    pm.addNestedPass<mlir::FuncOp>(std::make_unique<PreLLVMLowering>());
    pm.addPass(std::make_unique<LLVMLoweringPass>(getLLVMOptions()));
//    pm.addPass(mlir::createLowerToLLVMPass(getLLVMOptions()));
    pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(std::make_unique<PostLLVMLowering>());
}
}


void register_lower_to_llvm_pipeline(plier::PipelineRegistry& registry)
{
    registry.register_pipeline([](auto sink)
    {
        auto stage = get_lower_lowering_stage();
        sink(lower_to_llvm_pipeline_name(), {stage.begin}, {stage.end}, {}, &populate_lower_to_llvm_pipeline);
    });
}

llvm::StringRef lower_to_llvm_pipeline_name()
{
    return "lower_to_llvm";
}
