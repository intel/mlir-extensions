#include "pipelines/plier_to_std.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/TypeSwitch.h>

#include "plier/dialect.hpp"

#include "plier/rewrites/call_lowering.hpp"
#include "plier/rewrites/cast_lowering.hpp"
#include "plier/rewrites/type_conversion.hpp"
#include "plier/transforms/const_utils.hpp"
#include "plier/transforms/func_utils.hpp"
#include "plier/transforms/loop_utils.hpp"

#include "base_pipeline.hpp"
#include "plier/compiler/pipeline_registry.hpp"
#include "py_func_resolver.hpp"
#include "mangle.hpp"

namespace
{
mlir::Type map_int_type(mlir::MLIRContext& ctx, llvm::StringRef& name)
{
    unsigned num_bits = 0;
    if (name.consume_front("int") &&
        !name.consumeInteger<unsigned>(10, num_bits))
    {
        return mlir::IntegerType::get(&ctx, num_bits);
    }
    return nullptr;
}

mlir::Type map_int_literal_type(mlir::MLIRContext& ctx, llvm::StringRef& name)
{
    unsigned dummy = 0;
    if (name.consume_front("Literal[int](") &&
        !name.consumeInteger<unsigned>(10, dummy) && name.consume_front(")"))
    {
        return mlir::IntegerType::get(&ctx, 64); // TODO
    }
    return nullptr;
}

mlir::Type map_bool_type(mlir::MLIRContext& ctx, llvm::StringRef& name)
{
    if (name.consume_front("bool"))
    {
        return mlir::IntegerType::get(&ctx, 1);
    }
    return nullptr;
}

mlir::Type map_float_type(mlir::MLIRContext& ctx, llvm::StringRef& name)
{
    unsigned num_bits = 0;
    if (name.consume_front("float") &&
        !name.consumeInteger<unsigned>(10, num_bits))
    {
        switch(num_bits)
        {
        case 64: return mlir::Float64Type::get(&ctx);
        case 32: return mlir::Float32Type::get(&ctx);
        case 16: return mlir::Float16Type::get(&ctx);
        }
    }
    return nullptr;
}

mlir::Type map_plier_type_name(mlir::MLIRContext& ctx, llvm::StringRef& name);
bool map_type_helper(mlir::MLIRContext& ctx, llvm::StringRef& name, mlir::Type& ret)
{
    auto type = map_plier_type_name(ctx, name);
    if (static_cast<bool>(type))
    {
        ret = type;
        return true;
    }
    return false;
}

mlir::Type map_pair_type(mlir::MLIRContext& ctx, llvm::StringRef& name)
{
    mlir::Type first;
    mlir::Type second;
    if (name.consume_front("pair<") &&
        map_type_helper(ctx, name, first) &&
        name.consume_front(", ") &&
        map_type_helper(ctx, name, second) &&
        name.consume_front(">"))
    {
        return mlir::TupleType::get(&ctx, {first, second});
    }
    return nullptr;
}

mlir::Type map_unituple_type(mlir::MLIRContext& ctx, llvm::StringRef& name)
{
    mlir::Type type;
    unsigned count = 0;
    if (name.consume_front("UniTuple(") &&
        map_type_helper(ctx, name, type) &&
        name.consume_front(" x ") &&
        !name.consumeInteger<unsigned>(10, count) &&
        name.consume_front(")"))
    {
        llvm::SmallVector<mlir::Type, 8> types(count, type);
        return mlir::TupleType::get(&ctx, types);
    }
    return nullptr;
}

mlir::Type map_tuple_type(mlir::MLIRContext& ctx, llvm::StringRef& name)
{
    if (!name.consume_front("Tuple("))
    {
        return nullptr;
    }
    llvm::SmallVector<mlir::Type, 8> types;
    while (true)
    {
        if (name.consume_front(")"))
        {
            break;
        }
        auto type = map_plier_type_name(ctx, name);
        if (!static_cast<bool>(type))
        {
            return nullptr;
        }
        types.push_back(type);
        (void)name.consume_front(", ");
    }
    return mlir::TupleType::get(&ctx, types);
}

mlir::Type map_func_type(mlir::MLIRContext& ctx, llvm::StringRef& name)
{
    if (name.consume_front("Function(") &&
        name.consume_front("<class 'bool'>") && // TODO unhardcode;
        name.consume_front(")"))
    {
        return mlir::FunctionType::get(&ctx, {}, {});
    }
    return nullptr;
}

mlir::Type map_plier_type_name(mlir::MLIRContext& ctx, llvm::StringRef& name)
{
    using func_t = mlir::Type(*)(mlir::MLIRContext& ctx, llvm::StringRef& name);
    const func_t handlers[] = {
        &map_int_type,
        &map_int_literal_type,
        &map_bool_type,
        &map_float_type,
        &map_pair_type,
        &map_unituple_type,
        &map_tuple_type,
        &map_func_type,
    };
    for (auto h : handlers)
    {
        auto temp_name = name;
        auto t = h(ctx, temp_name);
        if (static_cast<bool>(t))
        {
            name = temp_name;
            return t;
        }
    }
    return nullptr;
}

mlir::Type map_plier_type(mlir::Type type)
{
    assert(type);
    if (!type.isa<plier::PyType>())
    {
        return type;
    }
    auto name = type.cast<plier::PyType>().getName();
    return map_plier_type_name(*type.getContext(), name);
}

bool is_supported_type(mlir::Type type)
{
    assert(type);
    return type.isIntOrFloat();
}

bool is_int(mlir::Type type)
{
    assert(type);
    return type.isa<mlir::IntegerType>();
}

bool is_float(mlir::Type type)
{
    assert(type);
    return type.isa<mlir::FloatType>();
}

bool is_index(mlir::Type type)
{
    assert(type);
    return type.isa<mlir::IndexType>();
}

struct ConstOpLowering : public mlir::OpRewritePattern<plier::ConstOp>
{
    ConstOpLowering(mlir::TypeConverter &/*typeConverter*/,
                   mlir::MLIRContext *context):
        OpRewritePattern(context) {}

    mlir::LogicalResult matchAndRewrite(
        plier::ConstOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto value = op.val();
        if (!is_supported_type(value.getType()))
        {
            return mlir::failure();
        }
        rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, value);
        return mlir::success();
    }
};

struct ArgOpLowering : public mlir::OpRewritePattern<plier::ArgOp>
{
    ArgOpLowering(mlir::TypeConverter &typeConverter,
                  mlir::MLIRContext *context):
        OpRewritePattern(context), converter(typeConverter) {}

    mlir::LogicalResult matchAndRewrite(
        plier::ArgOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto func = op->getParentOfType<mlir::FuncOp>();
        if (!func)
        {
            return mlir::failure();
        }

        auto index= op.index();
        if (index >= func.getNumArguments())
        {
            return mlir::failure();
        }

        auto arg = func.getArgument(index);
        if(converter.convertType(op.getType()) != arg.getType())
        {
            return mlir::failure();
        }
        rewriter.replaceOp(op, arg);
        return mlir::success();
    }
private:
    mlir::TypeConverter& converter;
};



struct ReturnOpLowering : public mlir::OpRewritePattern<mlir::ReturnOp>
{
    ReturnOpLowering(mlir::TypeConverter &/*typeConverter*/,
                     mlir::MLIRContext *context):
        OpRewritePattern(context) {}

    mlir::LogicalResult matchAndRewrite(
        mlir::ReturnOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto operands = op.getOperands();
        auto func = mlir::cast<mlir::FuncOp>(op->getParentOp());
        auto res_types = func.getType().getResults();
        assert(res_types.size() == operands.size() || res_types.empty());
        bool converted = (res_types.size() != operands.size());
        llvm::SmallVector<mlir::Value, 4> new_vals;
        for (auto it : llvm::zip(operands, res_types))
        {
            auto src = std::get<0>(it);
            auto dst = std::get<1>(it);
            if (src.getType() != dst)
            {
                auto new_op = rewriter.create<plier::CastOp>(op.getLoc(), dst, src);
                new_vals.push_back(new_op);
                converted = true;
            }
            else
            {
                new_vals.push_back(src);
            }
        }
        if (converted)
        {
            rewriter.create<mlir::ReturnOp>(op.getLoc(), new_vals);
            rewriter.eraseOp(op);
            return mlir::success();
        }
        return mlir::failure();
    }
};

struct SelectOpLowering : public mlir::OpRewritePattern<mlir::SelectOp>
{
    SelectOpLowering(mlir::TypeConverter &/*typeConverter*/,
                     mlir::MLIRContext *context):
        OpRewritePattern(context) {}

    mlir::LogicalResult matchAndRewrite(
        mlir::SelectOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto operands = op.getOperands();
        assert(operands.size() == 3);
        auto true_val = operands[1];
        auto false_val = operands[2];
        if (true_val.getType() == false_val.getType() &&
            true_val.getType() != op.getType())
        {
            auto cond = operands[0];
            rewriter.replaceOpWithNewOp<mlir::SelectOp>(op, cond, true_val, false_val);
            return mlir::success();
        }
        return mlir::failure();
    }
};

struct CondBrOpLowering : public mlir::OpRewritePattern<mlir::CondBranchOp>
{
    CondBrOpLowering(mlir::TypeConverter &/*typeConverter*/,
                     mlir::MLIRContext *context):
        OpRewritePattern(context) {}

    mlir::LogicalResult matchAndRewrite(
        mlir::CondBranchOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto operands = op.getOperands();
        assert(!operands.empty());
        auto cond = operands.front();
        operands = operands.drop_front();
        bool changed = false;

        auto process_operand = [&](mlir::Block& block, auto& ret)
        {
            for (auto arg : block.getArguments())
            {
                assert(!operands.empty());
                auto val = operands.front();
                operands = operands.drop_front();
                auto src_type = val.getType();
                auto dst_type = arg.getType();
                if (src_type != dst_type)
                {
                    ret.push_back(rewriter.create<plier::CastOp>(op.getLoc(), dst_type, val));
                    changed = true;
                }
                else
                {
                    ret.push_back(val);
                }
            }
        };

        llvm::SmallVector<mlir::Value, 4> true_vals;
        llvm::SmallVector<mlir::Value, 4> false_vals;
        auto true_dest = op.getTrueDest();
        auto false_dest = op.getFalseDest();
        process_operand(*true_dest, true_vals);
        process_operand(*false_dest, false_vals);
        if (changed)
        {
            rewriter.create<mlir::CondBranchOp>(op.getLoc(), cond, true_dest, true_vals, false_dest, false_vals);
            rewriter.eraseOp(op);
            return mlir::success();
        }
        return mlir::failure();
    }
};

mlir::Type coerce(mlir::Type type0, mlir::Type type1)
{
    // TODO: proper rules
    assert(type0 != type1);
    auto get_bits_count = [](mlir::Type type)->unsigned
    {
        assert(type);
        if (type.isa<mlir::IntegerType>())
        {
            return type.cast<mlir::IntegerType>().getWidth();
        }
        if (type.isa<mlir::Float16Type>())
        {
            return 11;
        }
        if (type.isa<mlir::Float32Type>())
        {
            return 24;
        }
        if (type.isa<mlir::Float64Type>())
        {
            return 53;
        }
        llvm_unreachable("Unhandled type");
    };
    auto f0 = is_float(type0);
    auto f1 = is_float(type1);
    if (f0 && !f1)
    {
        return type0;
    }
    if (!f0 && f1)
    {
        return type1;
    }
    return get_bits_count(type0) < get_bits_count(type1) ? type1 : type0;
}

template <bool Signed>
mlir::Value int_cast(mlir::Type dst_type, mlir::Value val, mlir::PatternRewriter& rewriter)
{
    auto src_bits = val.getType().cast<mlir::IntegerType>().getWidth();
    auto dst_bits = dst_type.cast<mlir::IntegerType>().getWidth();
    assert(src_bits != dst_bits);
    if (dst_bits > src_bits)
    {
        using T = std::conditional_t<Signed, mlir::SignExtendIOp, mlir::ZeroExtendIOp>;
        return rewriter.create<T>(val.getLoc(), val, dst_type);
    }
    else
    {
        return rewriter.create<mlir::TruncateIOp>(val.getLoc(), val, dst_type);
    }
}

template <bool Signed>
mlir::Value int_float_cast(mlir::Type dst_type, mlir::Value val, mlir::PatternRewriter& rewriter)
{
    using T = std::conditional_t<Signed, mlir::SIToFPOp, mlir::UIToFPOp>;
    return rewriter.create<T>(val.getLoc(), val, dst_type);
}

template <bool Signed>
mlir::Value float_int_cast(mlir::Type dst_type, mlir::Value val, mlir::PatternRewriter& rewriter)
{
    using T = std::conditional_t<Signed, mlir::FPToSIOp, mlir::FPToUIOp>;
    return rewriter.create<T>(val.getLoc(), val, dst_type);
}

mlir::Value index_cast_impl(mlir::Type dst_type, mlir::Value val, mlir::PatternRewriter& rewriter)
{
    return rewriter.create<mlir::IndexCastOp>(val.getLoc(), val, dst_type);
}

mlir::Value do_cast(mlir::Type dst_type, mlir::Value val, mlir::PatternRewriter& rewriter)
{
    assert(dst_type);
    auto src_type = val.getType();
    if (src_type == dst_type)
    {
        return val;
    }

    struct Handler
    {
        using selector_t = bool(*)(mlir::Type);
        using cast_op_t = mlir::Value(*)(mlir::Type, mlir::Value, mlir::PatternRewriter&);
        selector_t src;
        selector_t dst;
        cast_op_t cast_op;
    };

    const Handler handlers[] = {
        {&is_int, &is_int, &int_cast<true>},
        {&is_int, &is_float, &int_float_cast<true>},
        {&is_float, &is_int, &float_int_cast<true>},
        {&is_index, &is_int, &index_cast_impl},
        {&is_int, &is_index, &index_cast_impl},
    };

    for (auto& h : handlers)
    {
        if (h.src(src_type) && h.dst(dst_type))
        {
            return h.cast_op(dst_type, val, rewriter);
        }
    }

    return nullptr;
}

template<typename T>
void replace_op(mlir::Operation* op, mlir::PatternRewriter& rewriter, mlir::Type new_type, mlir::ValueRange operands)
{
    assert(nullptr != op);
    llvm::SmallVector<mlir::Value, 8> new_operands(operands.size());
    for (auto it : llvm::enumerate(operands))
    {
        new_operands[it.index()] = do_cast(new_type, it.value(), rewriter);
    }
    rewriter.replaceOpWithNewOp<T>(op, new_type, new_operands);
}

void replace_itruediv_op(mlir::Operation* op, mlir::PatternRewriter& rewriter, mlir::Type new_type, mlir::ValueRange operands)
{
    assert(nullptr != op);
    assert(new_type.isa<mlir::FloatType>());
    auto lhs = do_cast(new_type, operands[0], rewriter);
    auto rhs = do_cast(new_type, operands[1], rewriter);
    rewriter.replaceOpWithNewOp<mlir::DivFOp>(op, lhs, rhs);
}

void replace_imod_op(mlir::Operation* op, mlir::PatternRewriter& rewriter, mlir::Type /*new_type*/, mlir::ValueRange operands)
{
    auto loc = op->getLoc();
    auto a = operands[0];
    auto b = operands[1];
    auto v1 = rewriter.create<mlir::SignedRemIOp>(loc, a, b).getResult();
    auto v2 = rewriter.create<mlir::AddIOp>(loc, v1, b).getResult();
    auto res = rewriter.create<mlir::SignedRemIOp>(loc, v2, b).getResult();
    rewriter.replaceOp(op, res);
}

void replace_fmod_op(mlir::Operation* op, mlir::PatternRewriter& rewriter, mlir::Type /*new_type*/, mlir::ValueRange operands)
{
    auto loc = op->getLoc();
    auto a = operands[0];
    auto b = operands[1];
    auto v1 = rewriter.create<mlir::RemFOp>(loc, a, b).getResult();
    auto v2 = rewriter.create<mlir::AddFOp>(loc, v1, b).getResult();
    auto res = rewriter.create<mlir::RemFOp>(loc, v2, b).getResult();
    rewriter.replaceOp(op, res);
}

template<typename T, uint64_t Pred>
void replace_cmp_op(mlir::Operation* op, mlir::PatternRewriter& rewriter, mlir::Type /*new_type*/, mlir::ValueRange operands)
{
    assert(nullptr != op);
    auto pred_attr = mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 64), Pred);
    mlir::Type new_type = mlir::IntegerType::get(op->getContext(), 1);
    rewriter.replaceOpWithNewOp<T>(op, new_type, pred_attr, operands[0], operands[1]);
}


struct BinOpLowering : public mlir::OpRewritePattern<plier::BinOp>
{
    BinOpLowering(mlir::TypeConverter &typeConverter,
                  mlir::MLIRContext *context):
        OpRewritePattern(context), converter(typeConverter) {}

    mlir::LogicalResult matchAndRewrite(
        plier::BinOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto operands = op.getOperands();
        assert(operands.size() == 2);
        auto type0 = operands[0].getType();
        auto type1 = operands[1].getType();
        if (!is_supported_type(type0) || !is_supported_type(type1))
        {
            return mlir::failure();
        }
        auto res_type = converter.convertType(op.getType());
        if (!res_type || !is_supported_type(res_type))
        {
            return mlir::failure();
        }
        mlir::Type final_type;
        std::array<mlir::Value, 2> converted_operands;
        if (type0 != type1)
        {
            final_type = coerce(type0, type1);
            converted_operands = {
                do_cast(final_type, operands[0], rewriter),
                do_cast(final_type, operands[1], rewriter)};
        }
        else
        {
            final_type = type0;
            converted_operands = {operands[0], operands[1]};
        }
        assert(static_cast<bool>(final_type));

        using func_t = void(*)(mlir::Operation*, mlir::PatternRewriter&, mlir::Type, mlir::ValueRange);
        struct OpDesc
        {
            llvm::StringRef type;
            func_t iop;
            func_t fop;
        };

        const OpDesc handlers[] = {
            {"+", &replace_op<mlir::AddIOp>, &replace_op<mlir::AddFOp>},
            {"-", &replace_op<mlir::SubIOp>, &replace_op<mlir::SubFOp>},
            {"*", &replace_op<mlir::MulIOp>, &replace_op<mlir::MulFOp>},
            {"/", &replace_itruediv_op, &replace_op<mlir::DivFOp>},
            {"%", &replace_imod_op, &replace_fmod_op},

            {">", &replace_cmp_op<mlir::CmpIOp, static_cast<uint64_t>(mlir::CmpIPredicate::sgt)>,
                  &replace_cmp_op<mlir::CmpFOp, static_cast<uint64_t>(mlir::CmpFPredicate::OGT)>},
            {">=", &replace_cmp_op<mlir::CmpIOp, static_cast<uint64_t>(mlir::CmpIPredicate::sge)>,
                   &replace_cmp_op<mlir::CmpFOp, static_cast<uint64_t>(mlir::CmpFPredicate::OGE)>},
            {"<", &replace_cmp_op<mlir::CmpIOp, static_cast<uint64_t>(mlir::CmpIPredicate::slt)>,
                  &replace_cmp_op<mlir::CmpFOp, static_cast<uint64_t>(mlir::CmpFPredicate::OLT)>},
            {"<=", &replace_cmp_op<mlir::CmpIOp, static_cast<uint64_t>(mlir::CmpIPredicate::sle)>,
                   &replace_cmp_op<mlir::CmpFOp, static_cast<uint64_t>(mlir::CmpFPredicate::OLE)>},
            {"!=", &replace_cmp_op<mlir::CmpIOp, static_cast<uint64_t>(mlir::CmpIPredicate::ne)>,
                   &replace_cmp_op<mlir::CmpFOp, static_cast<uint64_t>(mlir::CmpFPredicate::ONE)>},
            {"==", &replace_cmp_op<mlir::CmpIOp, static_cast<uint64_t>(mlir::CmpIPredicate::eq)>,
                   &replace_cmp_op<mlir::CmpFOp, static_cast<uint64_t>(mlir::CmpFPredicate::OEQ)>},
        };

        using membptr_t = func_t OpDesc::*;
        auto call_handler = [&](membptr_t mem)
        {
            for (auto& h : handlers)
            {
                if (h.type == op.op())
                {
                    (h.*mem)(op, rewriter, res_type, converted_operands);
                    return mlir::success();
                }
            }
            return mlir::failure();
        };


        if (is_int(final_type))
        {
            return call_handler(&OpDesc::iop);
        }
        else if (is_float(final_type))
        {
            return call_handler(&OpDesc::fop);
        }
        return mlir::failure();
    }
private:
    mlir::TypeConverter& converter;
};

mlir::Value negate(mlir::Value val, mlir::Location loc, mlir::PatternRewriter &rewriter)
{
    auto type = val.getType();
    if (auto itype = type.dyn_cast<mlir::IntegerType>())
    {
        // TODO: not int negation?
        auto zero = rewriter.create<mlir::ConstantOp>(loc, mlir::IntegerAttr::get(itype, 0));
        return rewriter.create<mlir::SubIOp>(loc, zero, val);
    }
    if (type.isa<mlir::FloatType>())
    {
        return rewriter.create<mlir::NegFOp>(loc, val);
    }
    llvm_unreachable("negate: unsupported type");
}

struct UnaryOpLowering : public mlir::OpRewritePattern<plier::UnaryOp>
{
    UnaryOpLowering(mlir::TypeConverter &/*typeConverter*/,
                    mlir::MLIRContext *context):
        OpRewritePattern(context) {}

    mlir::LogicalResult matchAndRewrite(
        plier::UnaryOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto arg = op.getOperand();
        auto type = arg.getType();
        if (!is_supported_type(type))
        {
            return mlir::failure();
        }
        if (op.op() == "+")
        {
            rewriter.replaceOp(op, arg);
            return mlir::success();
        }
        assert(op.op() == "-");
        auto new_val = negate(arg, op.getLoc(), rewriter);
        rewriter.replaceOp(op, new_val);
        return mlir::success();
    }
};

mlir::Block* get_next_block(mlir::Block* block)
{
    assert(nullptr != block);
    if (auto br = mlir::dyn_cast_or_null<mlir::BranchOp>(block->getTerminator()))
    {
        return br.dest();
    }
    return nullptr;
};

void erase_blocks(llvm::ArrayRef<mlir::Block*> blocks)
{
    for (auto block : blocks)
    {
        assert(nullptr != block);
        block->dropAllDefinedValueUses();
    }
    for (auto block : blocks)
    {
        block->erase();
    }
}

bool is_blocks_different(llvm::ArrayRef<mlir::Block*> blocks)
{
    for (auto it : llvm::enumerate(blocks))
    {
        auto block1 = it.value();
        assert(nullptr != block1);
        for (auto block2 : blocks.drop_front(it.index() + 1))
        {
            assert(nullptr != block2);
            if (block1 == block2)
            {
                return false;
            }
        }
    }
    return true;
}

struct ScfIfRewrite : public mlir::OpRewritePattern<mlir::CondBranchOp>
{
    ScfIfRewrite(mlir::TypeConverter &/*typeConverter*/,
                 mlir::MLIRContext *context):
        OpRewritePattern(context) {}

    mlir::LogicalResult matchAndRewrite(
        mlir::CondBranchOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto getDest = [&](bool true_dest)
        {
            return true_dest ? op.getTrueDest() : op.getFalseDest();
        };
        auto getOperands = [&](bool true_dest)
        {
            return true_dest ? op.getTrueOperands() : op.getFalseOperands();
        };
        auto loc = op.getLoc();
        for (bool reverse : {false, true})
        {
            auto true_block = getDest(!reverse);
            auto post_block = get_next_block(true_block);
            if (nullptr == post_block)
            {
                continue;
            }
            auto false_block = getDest(reverse);
            if (false_block != post_block &&
                get_next_block(false_block) != post_block)
            {
                continue;
            }

            auto start_block = op.getOperation()->getBlock();
            if (!is_blocks_different({start_block, true_block, post_block}))
            {
                continue;
            }
            mlir::Value cond = op.condition();
            if (reverse)
            {
                auto i1 = mlir::IntegerType::get(op.getContext(), 1);
                auto one = rewriter.create<mlir::ConstantOp>(loc, mlir::IntegerAttr::get(i1, 1));
                cond = rewriter.create<mlir::XOrOp>(loc, cond, one);
            }

            mlir::BlockAndValueMapping mapper;
            llvm::SmallVector<mlir::Value, 8> yield_vals;
            auto copy_block = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::Block& block)
            {
                mapper.clear();
                for (auto& op : block.without_terminator())
                {
                    builder.clone(op, mapper);
                }
                auto term = mlir::cast<mlir::BranchOp>(block.getTerminator());
                yield_vals.clear();
                yield_vals.reserve(term.getNumOperands());
                for (auto op : term.getOperands())
                {
                    yield_vals.emplace_back(mapper.lookupOrDefault(op));
                }
                builder.create<mlir::scf::YieldOp>(loc, yield_vals);
            };

            auto true_body = [&](mlir::OpBuilder& builder, mlir::Location loc)
            {
                copy_block(builder, loc, *true_block);
            };

            bool has_else = false_block != post_block;
            auto res_types = mlir::cast<mlir::BranchOp>(true_block->getTerminator()).getOperandTypes();
            mlir::scf::IfOp if_op;
            if (has_else)
            {
                auto false_body = [&](mlir::OpBuilder& builder, mlir::Location loc)
                {
                    copy_block(builder, loc, *false_block);
                };
                if_op = rewriter.create<mlir::scf::IfOp>(
                    loc,
                    res_types,
                    cond,
                    true_body,
                    false_body);
            }
            else
            {
                if (res_types.empty())
                {
                    if_op = rewriter.create<mlir::scf::IfOp>(
                        loc,
                        res_types,
                        cond,
                        true_body);
                }
                else
                {
                    auto false_body = [&](mlir::OpBuilder& builder, mlir::Location loc)
                    {
                        auto res = getOperands(reverse);
                        yield_vals.clear();
                        yield_vals.reserve(res.size());
                        for (auto op : res)
                        {
                            yield_vals.emplace_back(mapper.lookupOrDefault(op));
                        }
                        builder.create<mlir::scf::YieldOp>(loc, yield_vals);
                    };
                    if_op = rewriter.create<mlir::scf::IfOp>(
                        loc,
                        res_types,
                        cond,
                        true_body,
                        false_body);
                }
            }

            rewriter.create<mlir::BranchOp>(loc, post_block, if_op.getResults());
            rewriter.eraseOp(op);

            if (true_block->getUsers().empty())
            {
                erase_blocks(true_block);
            }
            if (false_block->getUsers().empty())
            {
                erase_blocks(false_block);
            }
            return mlir::success();
        }
        return mlir::failure();
    }
};

mlir::scf::WhileOp create_while(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange iterArgs,
    llvm::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)> beforeBuilder,
    llvm::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)> afterBuilder)
{
    mlir::OperationState state(loc, mlir::scf::WhileOp::getOperationName());
    state.addOperands(iterArgs);

    {
        mlir::OpBuilder::InsertionGuard g(builder);
        auto add_region = [&](mlir::ValueRange args)->mlir::Block*
        {
            auto reg = state.addRegion();
            auto block = builder.createBlock(reg);
            for (auto arg : args)
            {
                block->addArgument(arg.getType());
            }
            return block;
        };

        auto beforeBlock = add_region(iterArgs);
        beforeBuilder(builder, state.location, beforeBlock->getArguments());
        auto cond = mlir::cast<mlir::scf::ConditionOp>(beforeBlock->getTerminator());
        state.addTypes(cond.args().getTypes());

        auto afterblock = add_region(cond.args());
        afterBuilder(builder, state.location, afterblock->getArguments());
    }
    return mlir::cast<mlir::scf::WhileOp>(builder.createOperation(state));
}

bool is_inside_block(mlir::Operation* op, mlir::Block* block)
{
    assert(nullptr != op);
    assert(nullptr != block);
    do
    {
        if (op->getBlock() == block)
        {
            return true;
        }
    }
    while((op = op->getParentOp()));
    return false;
}

struct ScfWhileRewrite : public mlir::OpRewritePattern<mlir::BranchOp>
{
    ScfWhileRewrite(mlir::TypeConverter &/*typeConverter*/,
                 mlir::MLIRContext *context):
        OpRewritePattern(context) {}

    mlir::LogicalResult matchAndRewrite(
        mlir::BranchOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto before_block = op.dest();
        auto before_term = mlir::dyn_cast<mlir::CondBranchOp>(before_block->getTerminator());
        if (!before_term)
        {
            return mlir::failure();
        }
        auto start_block = op.getOperation()->getBlock();
        auto after_block = before_term.trueDest();
        auto post_block = before_term.falseDest();
        if (get_next_block(after_block) != before_block ||
            !is_blocks_different({start_block, before_block, after_block, post_block}))
        {
            return mlir::failure();
        }

        auto check_outside_vals = [&](mlir::Operation* op)->mlir::WalkResult
        {
            for (auto user : op->getUsers())
            {
                if (!is_inside_block(user, before_block) &&
                    !is_inside_block(user, after_block))
                {
                    return mlir::WalkResult::interrupt();
                }
            }
            return mlir::WalkResult::advance();
        };

        if (after_block->walk(check_outside_vals).wasInterrupted())
        {
            return mlir::failure();
        }

        mlir::BlockAndValueMapping mapper;
        llvm::SmallVector<mlir::Value, 8> yield_vars;
        auto before_block_args = before_block->getArguments();
        llvm::SmallVector<mlir::Value, 8> orig_vars(before_block_args.begin(), before_block_args.end());

        auto before_body = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange iterargs)
        {
            mapper.map(before_block_args, iterargs);
            yield_vars.resize(before_block_args.size());
            for (auto& op : before_block->without_terminator())
            {
                auto new_op = builder.clone(op, mapper);
                for (auto user : op.getUsers())
                {
                    if (!is_inside_block(user, before_block))
                    {
                        for (auto it : llvm::zip(op.getResults(), new_op->getResults()))
                        {
                            orig_vars.emplace_back(std::get<0>(it));
                            yield_vars.emplace_back(std::get<1>(it));
                        }
                        break;
                    }
                }
            }

            llvm::transform(before_block->getArguments(), yield_vars.begin(),
                [&](mlir::Value val) { return mapper.lookupOrDefault(val); });

            auto term = mlir::cast<mlir::CondBranchOp>(before_block->getTerminator());
            for (auto arg : term.falseDestOperands())
            {
                orig_vars.emplace_back(arg);
                yield_vars.emplace_back(mapper.lookupOrDefault(arg));
            }
            auto cond = mapper.lookupOrDefault(term.condition());
            builder.create<mlir::scf::ConditionOp>(loc, cond, yield_vars);
        };
        auto after_body = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange iterargs)
        {
            mapper.clear();
            assert(orig_vars.size() == iterargs.size());
            mapper.map(orig_vars, iterargs);
            for (auto& op : after_block->without_terminator())
            {
                builder.clone(op, mapper);
            }
            yield_vars.clear();
            auto term = mlir::cast<mlir::BranchOp>(after_block->getTerminator());
            for (auto arg : term.getOperands())
            {
                yield_vars.emplace_back(mapper.lookupOrDefault(arg));
            }
            builder.create<mlir::scf::YieldOp>(loc, yield_vars);
        };

        auto while_op = create_while(
            rewriter,
            op.getLoc(),
            op.getOperands(),
            before_body,
            after_body);

        assert(orig_vars.size() == while_op.getNumResults());
        for (auto arg : llvm::zip(orig_vars, while_op.getResults()))
        {
            std::get<0>(arg).replaceAllUsesWith(std::get<1>(arg));
        }

        rewriter.create<mlir::BranchOp>(op.getLoc(), post_block, before_term.falseDestOperands());
        rewriter.eraseOp(op);
        erase_blocks({before_block, after_block});

        return mlir::success();
    }
};

struct FixupWhileTypes : public mlir::OpRewritePattern<mlir::scf::WhileOp>
{
    FixupWhileTypes(mlir::TypeConverter &/*typeConverter*/,
                    mlir::MLIRContext *context):
        OpRewritePattern(context) {}

    mlir::LogicalResult matchAndRewrite(
        mlir::scf::WhileOp op, mlir::PatternRewriter &rewriter) const override
    {
        bool changed = false;
        mlir::OpBuilder::InsertionGuard g(rewriter);
        auto before_block = &op.before().front();
        rewriter.startRootUpdate(op);
        rewriter.setInsertionPointToStart(before_block);
        assert(before_block->getNumArguments() == op.getNumOperands());
        auto loc = rewriter.getUnknownLoc();
        for (auto it : llvm::zip(op.getOperandTypes(), before_block->getArguments()))
        {
            auto new_type = std::get<0>(it);
            auto arg = std::get<1>(it);
            auto old_type = arg.getType();
            if (old_type != new_type)
            {
                rewriter.create<plier::CastOp>(loc, old_type, arg);
                arg.setType(new_type);
                changed = true;
            }
        }

        auto term = mlir::cast<mlir::scf::ConditionOp>(before_block->getTerminator());
        auto after_types = term.args().getTypes();

        auto after_block = &op.after().front();
        rewriter.setInsertionPointToStart(after_block);
        assert(after_block->getNumArguments() == term.args().size());
        for (auto it : llvm::zip(after_types, after_block->getArguments()))
        {
            auto new_type = std::get<0>(it);
            auto arg = std::get<1>(it);
            auto old_type = arg.getType();
            if (old_type != new_type)
            {
                rewriter.create<plier::CastOp>(loc, old_type, arg);
                arg.setType(new_type);
                changed = true;
            }
        }

        rewriter.setInsertionPointAfter(op);
        assert(op.getNumResults() == term.args().size());
        for (auto it : llvm::zip(after_types, op.getResults()))
        {
            auto new_type = std::get<0>(it);
            auto arg = std::get<1>(it);
            auto old_type = arg.getType();
            if (old_type != new_type)
            {
                rewriter.create<plier::CastOp>(loc, old_type, arg);
                arg.setType(new_type);
                changed = true;
            }
        }

        if (changed)
        {
            rewriter.finalizeRootUpdate(op);
        }
        else
        {
            rewriter.cancelRootUpdate(op);
        }
        return mlir::success(changed);
    }
};

template<typename Op>
struct FoldTupleGetitem : public mlir::OpRewritePattern<Op>
{
    FoldTupleGetitem(mlir::TypeConverter &/*typeConverter*/,
                     mlir::MLIRContext *context):
        OpRewritePattern(context) {}

    mlir::LogicalResult matchAndRewrite(
        Op op, mlir::PatternRewriter &rewriter) const override
    {
        auto build_tuple = op.value().template getDefiningOp<plier::BuildTupleOp>();
        if (!build_tuple)
        {
            return mlir::failure();
        }

        if (auto val = plier::getConstVal<mlir::IntegerAttr>(op.getOperand(1)))
        {
            auto index = val.getInt();
            if (index >= 0 && index < build_tuple.getNumOperands())
            {
                auto val = build_tuple.getOperand(static_cast<unsigned>(index));
                rewriter.replaceOp(op, val);
                return mlir::success();
            }
        }
        return mlir::failure();
    }
};

mlir::LogicalResult lower_range(plier::PyCallOp op, llvm::ArrayRef<mlir::Value> operands, mlir::PatternRewriter& rewriter)
{
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
        if (!user || mlir::failed(lower_while_to_for(user, rewriter, get_bounds, get_index)))
        {
            return mlir::failure();
        }
    }

    if (val.getUsers().empty())
    {
        rewriter.eraseOp(op);
    }
    return mlir::success();
}

mlir::LogicalResult lower_len(plier::PyCallOp op, llvm::ArrayRef<mlir::Value> operands, mlir::PatternRewriter& rewriter)
{
    if (operands.size() != 1)
    {
        return mlir::failure();
    }

    auto build_tuple = operands[0].getDefiningOp<plier::BuildTupleOp>();
    if (!build_tuple)
    {
        return mlir::failure();
    }

    auto size = rewriter.create<mlir::ConstantIndexOp>(op.getLoc(), build_tuple.getNumOperands());
    auto cast = rewriter.create<plier::CastOp>(op.getLoc(), op.getType(), size);
    rewriter.replaceOp(op, cast.getResult());
    return mlir::success();
}

mlir::LogicalResult lower_bool_cast(plier::PyCallOp op, llvm::ArrayRef<mlir::Value> operands, mlir::PatternRewriter& rewriter)
{
    if (operands.size() != 1)
    {
        return mlir::failure();
    }
    auto val = operands[0];
    bool success = false;
    auto replace_op = [&](mlir::Value val)
    {
        assert(!success);
        if (val)
        {
            rewriter.replaceOp(op, val);
            success = true;
        }
    };
    auto src_type = val.getType();
    auto dst_type = mlir::IntegerType::get(op.getContext(), 1);
    mlir::TypeSwitch<mlir::Type>(src_type)
        .Case<mlir::IntegerType>([&](auto) { replace_op(do_cast(dst_type, val, rewriter)); });
    return mlir::success(success);
}

mlir::FuncOp get_lib_symbol(
    mlir::ModuleOp mod, llvm::StringRef name, mlir::FunctionType type,
    mlir::PatternRewriter& rewriter)
{
    assert(!name.empty());
    if (auto op = mod.lookupSymbol<mlir::FuncOp>(name))
    {
        assert(op.getType() == type);
        return op;
    }

    return plier::add_function(rewriter, mod, name, type);
}

mlir::LogicalResult lower_math_func(
    plier::PyCallOp op, llvm::StringRef name, llvm::ArrayRef<mlir::Value> args,
    mlir::PatternRewriter& rewriter)
{
    auto ret_type = map_plier_type(op.getType());
    auto valid_type = [&](mlir::Type type)
    {
        return ret_type == type && type.isa<mlir::Float32Type, mlir::Float64Type>();
    };
    if (ret_type && name.consume_front("math.") && args.size() == 1 &&
        valid_type(args[0].getType()))
    {
        auto is_float = ret_type.isa<mlir::Float32Type>();
        auto func_type = mlir::FunctionType::get(op.getContext(), args[0].getType(), ret_type);
        auto module = op->getParentOfType<mlir::ModuleOp>();
        mlir::FuncOp func;
        if (is_float)
        {
            func = get_lib_symbol(module, name.str() + "f", func_type, rewriter);
        }
        else // double
        {
            func = get_lib_symbol(module, name, func_type, rewriter);
        }
        auto call = rewriter.create<mlir::CallOp>(op.getLoc(), func, args);
        rewriter.replaceOp(op, call.getResults());
        return mlir::success();
    }

    return mlir::failure();
}

struct CallLowerer
{
    mlir::LogicalResult operator()(plier::PyCallOp op, llvm::StringRef name,
        llvm::ArrayRef<mlir::Value> args, mlir::PatternRewriter& rewriter)
    {
        if (mlir::succeeded(lower_math_func(op, name, args, rewriter)))
        {
            return mlir::success();
        }

        using func_t = mlir::LogicalResult(*)(plier::PyCallOp, llvm::ArrayRef<mlir::Value>, mlir::PatternRewriter&);
        std::pair<llvm::StringRef, func_t> handlers[] = {
            {"bool", lower_bool_cast},
            {"range", lower_range},
            {"len", lower_len},
        };
        for (auto& handler : handlers)
        {
            if (handler.first == name)
            {
                return handler.second(op, args, rewriter);
            }
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
                rewriter.replaceOp(op, new_func_call.getResults());
                return mlir::success();
            }
        }

        return mlir::failure();
    }

private:
    PyFuncResolver py_resolver;
};

struct PlierToStdPass :
    public mlir::PassWrapper<PlierToStdPass, mlir::OperationPass<mlir::ModuleOp>>
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

void PlierToStdPass::runOnOperation()
{
    mlir::TypeConverter type_converter;
    // Convert unknown types to itself
    type_converter.addConversion([](mlir::Type type) { return type; });

    auto context = &getContext();
    populate_std_type_converter(*context, type_converter);

    mlir::OwningRewritePatternList patterns;

    patterns.insert<
        plier::FuncOpSignatureConversion,
        ArgOpLowering,
        ReturnOpLowering,
        ConstOpLowering,
        SelectOpLowering,
        CondBrOpLowering,
        BinOpLowering,
        UnaryOpLowering,
        ScfIfRewrite,
        ScfWhileRewrite,
        FixupWhileTypes,
        FoldTupleGetitem<plier::GetItemOp>,
        FoldTupleGetitem<plier::StaticGetItemOp>
        >(type_converter, context);

    patterns.insert<
        plier::CastOpLowering
        >(type_converter, context, &do_cast);

    CallLowerer callLowerer;

    patterns.insert<
        plier::CallOpLowering
        >(type_converter, context, callLowerer);

    mlir::populateStdExpandOpsPatterns(context, patterns);

    // range/prange lowering need dead branch pruning to properly
    // handle negative steps
    for (auto *op : context->getRegisteredOperations())
    {
        op->getCanonicalizationPatterns(patterns, context);
    }

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

void populate_plier_to_std_pipeline(mlir::OpPassManager& pm)
{
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(std::make_unique<PlierToStdPass>());
}
}

void populate_std_type_converter(mlir::MLIRContext& context, mlir::TypeConverter& converter)
{
    auto none_type = plier::PyType::getNone(&context);
    converter.addConversion(
    [none_type](mlir::Type type, llvm::SmallVectorImpl<mlir::Type>& ret_types)
    ->llvm::Optional<mlir::LogicalResult>
    {
        if (type == none_type)
        {
            return mlir::success();
        }
        auto ret = map_plier_type(type);
        if (!ret)
        {
            return llvm::None;
        }
        ret_types.push_back(ret);
        return mlir::success();
    });
}

void register_plier_to_std_pipeline(plier::PipelineRegistry& registry)
{
    registry.register_pipeline([](auto sink)
    {
        auto stage = get_high_lowering_stage();
        sink(plier_to_std_pipeline_name(), {stage.begin}, {stage.end}, {}, &populate_plier_to_std_pipeline);
    });
}

llvm::StringRef plier_to_std_pipeline_name()
{
    return "plier_to_std";
}
