#include "py_linalg_resolver.hpp"

#include <pybind11/pybind11.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Parser.h>
#include <mlir/IR/BuiltinAttributes.h>

#include "plier/dialect.hpp"
#include "py_map_types.hpp"
#include "plier/utils.hpp"
#include "plier/transforms/const_utils.hpp"

namespace py = pybind11;

struct PyBuilderContext
{
    mlir::Location loc;
    mlir::OpBuilder& builder;
    PyLinalgResolver::Context& context;
};

namespace
{
bool is_compatible_type(mlir::Type type)
{
    if (auto tuple_type = type.dyn_cast<mlir::TupleType>())
    {
        return llvm::all_of(tuple_type, &is_compatible_type);
    }
    return type.isa<mlir::IntegerType, mlir::IndexType, mlir::FloatType, mlir::RankedTensorType>();
}

template<typename R>
bool is_compatible_types(R&& vals)
{
    return llvm::all_of(vals, [](auto val) { return is_compatible_type(val.getType()); });
}

template<typename T>
py::capsule wrap_mlir(T val)
{
    return py::capsule(val.getAsOpaquePointer());
}

template<typename T>
T unwrap_mlir(py::capsule obj)
{
    return T::getFromOpaquePointer(static_cast<const void*>(obj));
}

auto unwrap_ssa_val(py::handle obj)
{
    return unwrap_mlir<mlir::Value>(obj.attr("_ssa_val").cast<py::capsule>());
}

auto unwrap_type(py::handle obj)
{
    return unwrap_mlir<mlir::Type>(obj.attr("_mlir_type").cast<py::capsule>());
}

size_t container_size(py::handle obj)
{
    if (py::isinstance<py::tuple>(obj))
    {
        return obj.cast<py::tuple>().size();
    }
    if (py::isinstance<py::list>(obj))
    {
        return obj.cast<py::list>().size();
    }
    return 1;
}

template<typename F>
void container_iterate(py::handle obj, F&& func)
{
    auto impl = [&](auto cont)
    {
        for (auto it : llvm::enumerate(cont))
        {
            func(it.index(), it.value());
        }
    };
    if (py::isinstance<py::tuple>(obj))
    {
        impl(obj.cast<py::tuple>());
    }
    else if (py::isinstance<py::list>(obj))
    {
        impl(obj.cast<py::list>());
    }
    else
    {
        func(std::size_t(0), obj);
    }
}

llvm::Optional<py::object> make_py_literal(mlir::Value val)
{
    assert(val);
    if (auto int_val = plier::getConstVal<mlir::IntegerAttr>(val))
    {
        return py::int_(int_val.getInt());
    }
    if (auto float_val = plier::getConstVal<mlir::FloatAttr>(val))
    {
        return py::float_(float_val.getValueAsDouble());
    }
    return {};
}

mlir::Value do_cast(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value val, mlir::Type type)
{
    if (val.getType() != type)
    {
        return builder.create<plier::CastOp>(loc, type, val);
    }
    return val;
}

bool cmp_capsule(py::capsule a1, py::capsule a2)
{
    return static_cast<void*>(a1) == static_cast<void*>(a2);
}

void setup_py_var(py::handle var);
}

struct PyLinalgResolver::Context
{
    py::handle var;
    py::handle type;
    py::handle builder;
    py::handle inspect;
    py::handle types_mod;
    py::handle compile_func;
    py::handle lookup_func;

    py::object create_var(py::capsule context, mlir::Value value)
    {
        assert(value);
        if (auto literal = make_py_literal(value))
        {
            return *literal;
        }
        auto ret = var(context, wrap_mlir(value));
        setup_py_var(ret);
        return ret;
    }

    py::object create_type(mlir::Type t)
    {
        return type(wrap_mlir(t), py::cpp_function(&cmp_capsule));
    }

    mlir::FuncOp compile_body(py::handle body, py::list arg_types)
    {
        auto func = compile_func(body, arg_types).cast<py::capsule>();
        auto mlir_func = mlir::cast<mlir::FuncOp>(static_cast<mlir::Operation*>(func));
        mlir_func.setPrivate();
        mlir_func->setAttr(plier::attributes::getForceInlineName(), mlir::UnitAttr::get(mlir_func->getContext()));
        return mlir_func;
    }

    py::object wrap_result(py::capsule context, mlir::ValueRange values)
    {
        if (values.empty())
        {
            return py::none();
        }
        if (values.size() == 1)
        {
            return create_var(context, values.front());
        }
        py::tuple ret(values.size());
        for (auto it : llvm::enumerate(values))
        {
            ret[it.index()] = create_var(context, it.value());
        }
        return std::move(ret);
    }

    mlir::Value unwrap_val(mlir::Location loc, mlir::OpBuilder& builder, py::handle obj)
    {
        if (py::isinstance(obj, var))
        {
            return unwrap_ssa_val(obj);
        }
        if (py::isinstance<py::int_>(obj))
        {
            auto attr = builder.getI64IntegerAttr(obj.cast<int64_t>());
            return builder.create<mlir::ConstantOp>(loc, attr);
        }
        if (py::isinstance<py::float_>(obj))
        {
            auto attr = builder.getF64FloatAttr(obj.cast<double>());
            return builder.create<mlir::ConstantOp>(loc, attr);
        }
        plier::report_error("Invalid element type");
    }
};

namespace
{
py::object get_args(py::handle inspect, py::handle func, llvm::function_ref<py::object(mlir::Value)> create_var,
                    mlir::ValueRange args, llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs)
{
    auto sig_func = inspect.attr("signature");
    auto sig = sig_func(func);
    auto params = sig.attr("parameters");
    auto params_list = py::list(params);
    params_list = params_list[py::slice(1, static_cast<int64_t>(params_list.size()), 1)]; // skip builder param
    auto empty = inspect.attr("Parameter").attr("empty");

    py::list ret(py::len(params_list));
    for (auto it : llvm::enumerate(params_list))
    {
        auto index = it.index();
        auto param_name = it.value();
        auto param = params[param_name];
        if (!args.empty())
        {
            ret[index] = create_var(args.front());
            args = args.drop_front();
            continue;
        }
        if (!kwargs.empty())
        {
            auto name = param_name.cast<std::string>();
            auto val = [&]()->mlir::Value
            {
                for (auto kwarg : kwargs)
                {
                    if (kwarg.first == name)
                    {
                        return kwarg.second;
                    }
                }
                return {};
            }();
            if (val)
            {
                ret[index] = create_var(val);
                continue;
            }
        }
        auto def_val = param.attr("default");
        if (!def_val.is(empty))
        {
            ret[index] = def_val;
        }
        else
        {
            return py::none();
        }
    }
    if (!args.empty())
    {
        return py::none();
    }
    return std::move(ret);
}

PyBuilderContext& get_py_context(py::capsule& ctx)
{
    return *static_cast<PyBuilderContext*>(ctx);
}

auto get_types(mlir::ValueRange values)
{
    return values.getTypes();
}

auto get_agrs_from_tuple(py::handle args, llvm::function_ref<mlir::Value(py::handle)> unpack)
{
    llvm::SmallVector<mlir::Value> ret;
    if (args.is_none())
    {
        return ret;
    }
    if (py::isinstance<py::tuple>(args))
    {
        auto tuple = args.cast<py::tuple>();
        ret.resize(tuple.size());
        for (auto it : llvm::enumerate(tuple))
        {
            ret[it.index()] = unpack(it.value());
        }
    }
    else
    {
        ret.emplace_back(unpack(args));
    }
    return ret;
}

auto get_iterators(py::list iterators, mlir::MLIRContext& ctx)
{
    llvm::SmallVector<llvm::StringRef> ret(iterators.size());
    for (auto it : llvm::enumerate(iterators))
    {
        ret[it.index()] = mlir::StringAttr::get(&ctx, it.value().cast<std::string>()).getValue();
    }
    return ret;
}

mlir::AffineMapAttr get_affine_map_attr(py::handle obj, mlir::MLIRContext& ctx)
{
    auto str = (llvm::Twine("affine_map<") + obj.cast<std::string>() + ">").str();
    return mlir::parseAttribute(str, &ctx).cast<mlir::AffineMapAttr>();
}

auto get_affine_maps(py::list maps, mlir::MLIRContext& ctx)
{
    llvm::SmallVector<mlir::AffineMap> ret(maps.size());
    for (auto it : llvm::enumerate(maps))
    {
        ret[it.index()] = get_affine_map_attr(it.value(), ctx).getValue();
    }
    return ret;
}

auto get_generic_op_body_types(mlir::ValueRange inputs, mlir::ValueRange outputs)
{
    llvm::SmallVector<mlir::Type> ret;
    ret.reserve(inputs.size() + outputs.size());
    for (auto r : {inputs, outputs})
    {
        for (auto type : r.getTypes())
        {
            auto elem_type = [&]()
            {
                if (auto tensor = type.dyn_cast<mlir::RankedTensorType>())
                {
                    return tensor.getElementType();
                }
                return type;
            }();
            ret.emplace_back(elem_type);
        }
    }
    return ret;
}

auto generic_op_body_result_types(mlir::ValueRange outputs)
{
    llvm::SmallVector<mlir::Type> ret;
    ret.reserve(outputs.size());
    for (auto type : outputs.getTypes())
    {
        auto elem_type = type.cast<mlir::RankedTensorType>().getElementType();
        ret.emplace_back(elem_type);
    }
    return ret;
}

bool is_int(mlir::Type type)
{
    return type.isa<mlir::IntegerType, mlir::IndexType>();
}

unsigned get_int_bit_width(mlir::Type type)
{
    if (type.isa<mlir::IntegerType>())
    {
        return type.cast<mlir::IntegerType>().getWidth();
    }
    if (type.isa<mlir::IndexType>())
    {
        return 64; // TODO
    }
    llvm_unreachable("No an integer type");
}

bool is_float(mlir::Type type)
{
    return type.isa<mlir::FloatType>();
}

unsigned get_float_bit_width(mlir::Type type)
{
    return type.cast<mlir::FloatType>().getWidth();
}

mlir::Type broadcast_type(mlir::Type type1, mlir::Type type2)
{
    if (type1 == type2)
    {
        return type1;
    }
    // TODO
    if (is_int(type1) && is_int(type2))
    {
        auto width = std::max(get_int_bit_width(type1), get_int_bit_width(type2));
        return mlir::IntegerType::get(type1.getContext(), width);
    }
    if (is_float(type1) && is_float(type2))
    {
        return (get_float_bit_width(type1) > get_float_bit_width(type2) ? type1 : type2);
    }
    if (is_float(type1) && is_int(type2))
    {
        return type1;
    }
    if (is_int(type1) && is_float(type2))
    {
        return type2;
    }
    llvm_unreachable("Unable to broadcast type");
}

mlir::Value broadcast_dim(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val1, mlir::Value val2)
{
    assert(val1.getType().isa<mlir::IndexType>());
    assert(val2.getType().isa<mlir::IndexType>());
    auto one = builder.create<mlir::ConstantIndexOp>(loc, 1);
    auto cond = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, val1, one);
    return builder.create<mlir::SelectOp>(loc, cond, val2, val1);
}

mlir::Value expand_dim(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value initial, mlir::Value src, unsigned dim, mlir::ValueRange target_shape)
{
    auto context = builder.getContext();
    auto src_type = src.getType().cast<mlir::ShapedType>();
    auto num_dims = static_cast<unsigned>(src_type.getRank());
    auto shape = llvm::to_vector<8>(src_type.getShape());
    shape[dim] = -1;
    mlir::Type target_type = mlir::RankedTensorType::get(shape, src_type.getElementType());
    auto dim_val = builder.create<mlir::DimOp>(loc, initial, dim);
    auto one = builder.create<mlir::ConstantIndexOp>(loc, 1);
    mlir::Value cond = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, one, dim_val);
    llvm::SmallVector<mlir::Value> new_shape(num_dims);
    for (unsigned i = 0 ; i < num_dims; ++i)
    {
        if (i == dim)
        {
            new_shape[i] = target_shape[i];
        }
        else
        {
            new_shape[i] = builder.create<mlir::DimOp>(loc, src, i);
        }
    }
    auto true_body = [&](mlir::OpBuilder &builder, mlir::Location loc)
    {
        assert(dim < shape.size());
        shape[dim] = 1;
//        mlir::Type casted_type = mlir::RankedTensorType::get(shape, src_type.getElementType());
//        auto casted = builder.create<mlir::tensor::CastOp>(loc, casted_type, src).getResult();
        auto casted = src; // TODO
        auto init = builder.create<mlir::linalg::InitTensorOp>(loc, new_shape, src_type.getElementType()).getResult();
        llvm::SmallVector<mlir::AffineExpr> exprs(num_dims);
        for (unsigned i = 0; i < num_dims; ++i)
        {
            if (i == dim)
            {
                exprs[i] = mlir::getAffineConstantExpr(0, context);
            }
            else
            {
                exprs[i] = mlir::getAffineDimExpr(i, context);
            }
        }
        const mlir::AffineMap maps[] = {
            mlir::AffineMap::get(num_dims, 0, exprs, context),
            mlir::AffineMap::getMultiDimIdentityMap(num_dims, context),
        };
        llvm::SmallVector<mlir::StringRef> iterators(num_dims, "parallel");

        auto body = [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange values)
        {
            assert(values.size() == 2);
            builder.create<mlir::linalg::YieldOp>(loc, values[0]);
        };

        auto expanded = builder.create<mlir::linalg::GenericOp>(loc, target_type, casted, init, maps, iterators, body);
        auto res = builder.create<mlir::tensor::CastOp>(loc, target_type, expanded.getResult(0));
        builder.create<mlir::scf::YieldOp>(loc, res.getResult());
    };
    auto false_body = [&](mlir::OpBuilder &builder, mlir::Location loc)
    {
        auto res = builder.create<mlir::tensor::CastOp>(loc, target_type, src);
        builder.create<mlir::scf::YieldOp>(loc, res.getResult());
    };
    return builder.create<mlir::scf::IfOp>(loc, target_type, cond, true_body, false_body).getResult(0);
}

mlir::Value expand_dims(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val, unsigned num_dims, mlir::ValueRange target_shape)
{
    assert(num_dims <= target_shape.size());
    if (num_dims < target_shape.size())
    {
        target_shape = target_shape.drop_front(target_shape.size() - num_dims);
    }
    mlir::Value current = val;
    for (unsigned i = 0; i < num_dims; ++i)
    {
        current = expand_dim(builder, loc, val, current, i, target_shape);
    }
    current = builder.create<plier::EnforceShapeOp>(loc, current, target_shape);
    return current;
}

py::object broadcast_impl(py::capsule context, py::tuple args)
{
    if (1 == args.size())
    {
        return args[0];
    }
    auto& ctx = get_py_context(context);
    auto loc = ctx.loc;
    auto& builder = ctx.builder;
    llvm::SmallVector<mlir::Value> mlir_args(args.size());
    for (auto it : llvm::enumerate(args))
    {
        mlir_args[it.index()] = ctx.context.unwrap_val(loc, builder, it.value());
    }
    using shape_t = llvm::SmallVector<mlir::Value>;
    auto get_shape = [&](mlir::Value val)->llvm::Optional<std::pair<shape_t, mlir::Type>>
    {
        auto type = val.getType();
        if (auto shaped = type.dyn_cast<mlir::ShapedType>())
        {
            if (!shaped.hasRank())
            {
                return {};
            }
            shape_t ret(static_cast<size_t>(shaped.getRank()));
            for (auto it : llvm::enumerate(ret))
            {
                auto dim = builder.create<mlir::DimOp>(loc, val, it.index());
                ret[it.index()] = dim;
            }
            return std::make_pair(ret, shaped.getElementType());
        }
        if (type.isa<mlir::IntegerType, mlir::IndexType, mlir::FloatType>())
        {
            return std::make_pair(shape_t{}, type);
        }
        return {};
    };
    mlir::Type res_type;
    mlir::SmallVector<mlir::Value> shape_vals;
    if (auto shape_and_type = get_shape(mlir_args.front()))
    {
        res_type = shape_and_type->second;
        shape_vals = shape_and_type->first;
    }
    else
    {
        return py::none();
    }

    for (auto arg : llvm::drop_begin(mlir_args))
    {
        auto shape_and_type = get_shape(arg);
        if (!shape_and_type)
        {
            py::none();
        }
        res_type = broadcast_type(res_type, shape_and_type->second);
        auto new_shape_vals = shape_and_type->first;
        for (auto it : llvm::zip(llvm::reverse(shape_vals), llvm::reverse(new_shape_vals)))
        {
            auto& old_val = std::get<0>(it);
            auto new_val =  std::get<1>(it);
            old_val = broadcast_dim(builder, loc, old_val, new_val);
        }
        if (new_shape_vals.size() > shape_vals.size())
        {
            auto front = llvm::makeArrayRef(new_shape_vals).drop_back(shape_vals.size());
            assert(!front.empty());
            shape_vals.insert(shape_vals.begin(), front.begin(), front.end());
        }
    }

    py::tuple ret(mlir_args.size());
    if (shape_vals.empty())
    {
        for (auto it : llvm::enumerate(mlir_args))
        {
            mlir::Value val = it.value();
            if (val.getType() != res_type)
            {
                val = builder.create<plier::CastOp>(loc, res_type, val);
            }
            ret[it.index()] = ctx.context.create_var(context, val);
        }
        return std::move(ret);
    }

    llvm::SmallVector<int64_t> shape(static_cast<size_t>(shape_vals.size()), -1);
    auto tensor_type = mlir::RankedTensorType::get(shape, res_type);
    for (auto it : llvm::enumerate(mlir_args))
    {
        mlir::Value val = it.value();
        if (auto src_type = val.getType().dyn_cast<mlir::ShapedType>())
        {
            assert(src_type.hasRank());
            val = expand_dims(builder, loc, val, static_cast<unsigned>(src_type.getRank()), shape_vals);
        }
        if (val.getType() != tensor_type)
        {
            auto type = val.getType();
            if (auto src_type = type.dyn_cast<mlir::ShapedType>())
            {
                assert(src_type.hasRank());
                auto src_num_dims = static_cast<unsigned>(src_type.getRank());
                auto num_dims = static_cast<unsigned>(tensor_type.getRank());
                auto init = builder.create<mlir::linalg::InitTensorOp>(loc, shape_vals, tensor_type.getElementType()).getResult();
                mlir::AffineMap maps[] = {
                    mlir::AffineMap::getMinorIdentityMap(num_dims, src_num_dims, builder.getContext()),
//                    mlir::AffineMap::getMultiDimIdentityMap(num_dims, builder.getContext()).getMajorSubMap(src_num_dims),
                    mlir::AffineMap::getMultiDimIdentityMap(num_dims, builder.getContext()),
                };
                llvm::SmallVector<llvm::StringRef> iterators(num_dims, "parallel");
                auto body = [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange values)
                {
                    assert(values.size() == 2);
                    auto res = builder.create<plier::CastOp>(loc, tensor_type.getElementType(), values[0]);
                    builder.create<mlir::linalg::YieldOp>(loc, res.getResult());
                };
                val = builder.create<mlir::linalg::GenericOp>(loc, tensor_type, val, init, maps, iterators, body).getResult(0);
            }
            else
            {
                if (tensor_type.getElementType() != type)
                {
                    val = builder.create<plier::CastOp>(loc, tensor_type.getElementType(), val);
                }
                val = builder.create<mlir::tensor::FromElementsOp>(loc, val);
                auto num_dims = static_cast<unsigned>(tensor_type.getRank());
                auto init = builder.create<mlir::linalg::InitTensorOp>(loc, shape_vals, tensor_type.getElementType()).getResult();
                mlir::AffineMap maps[] = {
                    mlir::AffineMap::get(num_dims, 0, mlir::getAffineConstantExpr(0, builder.getContext())),
                    mlir::AffineMap::getMultiDimIdentityMap(num_dims, builder.getContext()),
                    };
                llvm::SmallVector<llvm::StringRef> iterators(num_dims, "parallel");
                auto body = [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange values)
                {
                    assert(values.size() == 2);
                    builder.create<mlir::linalg::YieldOp>(loc, values[0]);
                };
                val = builder.create<mlir::linalg::GenericOp>(loc, tensor_type, val, init, maps, iterators, body).getResult(0);
            }
        }
        ret[it.index()] = ctx.context.create_var(context, val);
    }
    return std::move(ret);
}

py::object init_tensor_impl(py::capsule context, py::handle shape, py::handle dtype, py::handle init_val)
{
    auto& ctx = get_py_context(context);
    auto loc = ctx.loc;
    auto& builder = ctx.builder;
    auto elem_type = unwrap_type(dtype);
    mlir::Value init;
    auto count = py::len(shape);
    if (count == 0)
    {
        if (init_val.is_none())
        {
            // TODO: undef
            auto zero_val = plier::getZeroVal(elem_type);
            assert(zero_val);
            init = builder.create<mlir::ConstantOp>(loc, zero_val);
        }
        else
        {
            init = do_cast(loc, builder, ctx.context.unwrap_val(loc, builder, init_val), elem_type);
        }
    }
    else
    {
        auto index_type = builder.getIndexType();
        llvm::SmallVector<mlir::Value> shape_val(count);
        llvm::SmallVector<int64_t> static_shape(count, -1);
        for (size_t i = 0; i < count; ++i)
        {
            auto elem = shape[py::int_(i)];
            if (py::isinstance<py::int_>(elem))
            {
                static_shape[i] = elem.cast<int64_t>();
            }
            shape_val[i] = do_cast(loc, builder, ctx.context.unwrap_val(loc, builder, elem), index_type);
        }

        if (init_val.is_none())
        {
            init = builder.create<mlir::linalg::InitTensorOp>(loc, shape_val, elem_type);
        }
        else
        {
            auto val = do_cast(loc, builder, ctx.context.unwrap_val(loc, builder, init_val), elem_type);
            llvm::SmallVector<int64_t> shape(count, -1);
            auto type = mlir::RankedTensorType::get(shape, elem_type);
            auto body = [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange /*indices*/)
            {
                builder.create<mlir::tensor::YieldOp>(loc, val);
            };
            init = builder.create<mlir::tensor::GenerateOp>(loc, type, shape_val, body);
        }
        if (llvm::any_of(static_shape, [](auto val){ return val >= 0;}))
        {
            auto new_type = mlir::RankedTensorType::get(static_shape, elem_type);
            init = builder.create<mlir::tensor::CastOp>(loc, new_type, init);
        }
    }
    return ctx.context.create_var(context, init);
}

py::object fill_tensor_impl(py::capsule context, py::handle tensor, py::handle value)
{
    auto& ctx = get_py_context(context);
    auto loc = ctx.loc;
    auto& builder = ctx.builder;
    auto tensor_val = ctx.context.unwrap_val(loc, builder, tensor);
    auto tensor_type = tensor_val.getType().cast<mlir::ShapedType>();
    auto init_val = ctx.context.unwrap_val(loc, builder, value);
    if (init_val.getType() != tensor_type.getElementType())
    {
        init_val = builder.create<plier::CastOp>(loc, tensor_type.getElementType(), init_val);
    }

//    auto val = builder.create<mlir::linalg::FillOp>(loc, tensor_type, tensor_val, init_val);
    auto rank = static_cast<unsigned>(tensor_type.getRank());
    mlir::AffineMap affine_maps[] = {
        mlir::AffineMap::getMultiDimIdentityMap(rank, builder.getContext()),
    };
    llvm::SmallVector<llvm::StringRef> iterators(rank, "parallel");
    auto body = [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange values)
    {
        assert(values.size() == 1);
        builder.create<mlir::linalg::YieldOp>(loc, init_val);
    };
    auto val = builder.create<mlir::linalg::GenericOp>(
        loc,
        tensor_type,
        llvm::None,
        tensor_val,
        affine_maps,
        iterators,
        body);
    return ctx.context.create_var(context, val.getResult(0));
}

py::object generic_impl(py::capsule context, py::handle inputs, py::handle outputs, py::list iterators, py::list maps, py::handle body)
{
    auto& ctx = get_py_context(context);
    auto loc = ctx.loc;
    auto& builder = ctx.builder;
    auto& mlir_context = *builder.getContext();

    auto unpack = [&](py::handle obj)->mlir::Value
    {
        return ctx.context.unwrap_val(loc, builder, obj);
    };

    auto inputs_args = get_agrs_from_tuple(inputs, unpack);
    auto output_args = get_agrs_from_tuple(outputs, unpack);
    auto ret_types = get_types(output_args);
    auto mlir_iterators = get_iterators(iterators, mlir_context);

    auto func_types = map_types_to_numba(ctx.context.types_mod, get_generic_op_body_types(inputs_args, output_args));
    auto body_func = ctx.context.compile_body(body, func_types);

    auto cast_values = [&](mlir::ValueRange vals, mlir::TypeRange types)
    {
        assert(vals.size() == types.size());
        llvm::SmallVector<mlir::Value> ret(vals.size());
        auto do_cast = [&](mlir::Value val, mlir::Type type)
        {
            if (val.getType() == type)
            {
                return val;
            }
            return builder.create<plier::CastOp>(loc, type, val).getResult();
        };
        for (auto it : llvm::enumerate(vals))
        {
            auto index = static_cast<unsigned>(it.index());
            ret[index] = do_cast(it.value(), types[index]);
        }
        return ret;
    };
    if (mlir_iterators.empty())
    {
        inputs_args.append(output_args.begin(), output_args.end());
        auto res = builder.create<mlir::CallOp>(loc, body_func, inputs_args);
        return ctx.context.wrap_result(context, cast_values(res.getResults(), ret_types));
    }
    else
    {
        auto affine_maps = get_affine_maps(maps, mlir_context);
        auto body_builder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args)
        {
            auto func_type = body_func.getType();
            auto new_args = cast_values(args, func_type.getInputs());
            auto call = builder.create<mlir::CallOp>(loc, body_func, new_args);
            auto new_results = cast_values(call.getResults(), generic_op_body_result_types(output_args));
            builder.create<mlir::linalg::YieldOp>(loc, new_results);
        };

        auto generic_op = builder.create<mlir::linalg::GenericOp>(
            loc,
            ret_types,
            inputs_args,
            output_args,
            affine_maps,
            mlir_iterators,
            body_builder);
        return ctx.context.wrap_result(context, generic_op.getResults());
    }
}

py::object from_elements_impl(py::capsule context, py::handle values, py::handle dtype)
{
    auto& ctx = get_py_context(context);
    auto& builder = ctx.builder;
    auto loc = ctx.loc;
    auto type = unwrap_type(dtype);

    llvm::SmallVector<mlir::Value> vals(container_size(values));
    container_iterate(values, [&](auto index, py::handle obj)
    {
        if (py::isinstance(obj, ctx.context.var))
        {
            vals[index] = unwrap_ssa_val(obj);
        }
        else if (py::isinstance<py::int_>(obj) ||
                 py::isinstance<py::float_>(obj))
        {
            auto attr = [&]()->mlir::Attribute
            {
                if (type.isa<mlir::IntegerType>())
                {
                    return mlir::IntegerAttr::get(type, obj.cast<int64_t>());
                }
                if (type.isa<mlir::FloatType>())
                {
                    return mlir::FloatAttr::get(type, obj.cast<double>());
                }
                plier::report_error("Invalid dtype");
            }();
            vals[index] = builder.create<mlir::ConstantOp>(loc, attr);
        }
        else
        {
            plier::report_error("Invalid element type");
        }
    });
    auto res = builder.create<mlir::tensor::FromElementsOp>(loc, vals);
    return ctx.context.create_var(context, res);
}

py::object extract_impl(py::capsule context, py::handle value, py::handle indices)
{
    auto& ctx = get_py_context(context);
    auto& builder = ctx.builder;
    auto loc = ctx.loc;

    llvm::SmallVector<mlir::Value> ind(container_size(indices));
    container_iterate(indices, [&](auto index, py::handle obj)
    {
        if (py::isinstance(obj, ctx.context.var))
        {
            ind[index] = unwrap_ssa_val(obj);
        }
        else if (py::isinstance<py::int_>(obj))
        {
            ind[index] = builder.create<mlir::ConstantIndexOp>(loc, obj.cast<int64_t>());
        }
        else
        {
            plier::report_error("Invalid element type");
        }
    });
    auto res = builder.create<mlir::tensor::ExtractOp>(loc, ctx.context.unwrap_val(loc, builder, value), ind);
    return ctx.context.create_var(context, res);
}

py::object reshape_impl(py::capsule context, py::handle tensor, py::int_ out_dims, py::list maps)
{
    auto& ctx = get_py_context(context);
    auto& builder = ctx.builder;
    auto loc = ctx.loc;

    auto tensor_val = ctx.context.unwrap_val(loc, builder, tensor);
    if (!tensor_val.getType().isa<mlir::RankedTensorType>())
    {
        plier::report_error("Invalid reshapa argument");
    }
    auto elem_type = tensor_val.getType().cast<mlir::RankedTensorType>().getElementType();
    auto new_dims = out_dims.cast<size_t>();
    llvm::SmallVector<int64_t> dims(new_dims, -1);
    auto new_type = mlir::RankedTensorType::get(dims, elem_type);

    llvm::SmallVector<mlir::Attribute> affine_maps(container_size(maps));
    container_iterate(maps, [&](auto index, py::handle obj)
    {
        affine_maps[index] = get_affine_map_attr(obj, *builder.getContext());
    });
    auto affine_maps_attr = mlir::ArrayAttr::get(builder.getContext(), affine_maps);
    auto reshape = builder.create<mlir::linalg::TensorReshapeOp>(loc, new_type, tensor_val, affine_maps_attr);
    return ctx.context.create_var(context, reshape);
}

void setup_py_builder(py::handle builder, mlir::OpBuilder& b, llvm::function_ref<py::object(mlir::Type)> create_type)
{
    py::setattr(builder, "_broadcast", py::cpp_function(&broadcast_impl));
    py::setattr(builder, "_init_tensor", py::cpp_function(&init_tensor_impl));
    py::setattr(builder, "_fill_tensor", py::cpp_function(&fill_tensor_impl));
    py::setattr(builder, "_generic", py::cpp_function(&generic_impl));
    py::setattr(builder, "_from_elements", py::cpp_function(&from_elements_impl));
    py::setattr(builder, "_extract", py::cpp_function(&extract_impl));
    py::setattr(builder, "_reshape", py::cpp_function(&reshape_impl));

    auto add_type = [&](const char* name, mlir::Type type)
    {
        py::setattr(builder, name, create_type(type));
    };

    add_type("int8", b.getIntegerType(8));
    add_type("int16", b.getIntegerType(16));
    add_type("int32", b.getIntegerType(32));
    add_type("int64", b.getIntegerType(64));
    add_type("index", b.getIndexType());

    add_type("float16", b.getF16Type());
    add_type("float32", b.getF32Type());
    add_type("float64", b.getF64Type());
}

py::object shape_impl(py::capsule context, py::capsule ssa_val)
{
    auto& ctx = get_py_context(context);
    auto value = unwrap_mlir<mlir::Value>(ssa_val);
    if (value.getType().isa<mlir::RankedTensorType>())
    {
        auto& builder = ctx.builder;
        auto loc = ctx.loc;
        auto mlir_type = value.getType().cast<mlir::RankedTensorType>();
        auto shape = mlir_type.getShape();
        llvm::SmallVector<mlir::Value> shape_vals(shape.size());
        for (auto it : llvm::enumerate(shape))
        {
            auto i = it.index();
            mlir::Value mlir_dim = builder.create<mlir::DimOp>(loc, value, i);
            shape_vals[i] = mlir_dim;
        }
        llvm::SmallVector<mlir::Type> shape_types(shape.size(), builder.getIndexType());
        auto shape_type = mlir::TupleType::get(builder.getContext(), shape_types);
        auto shape_var = builder.create<plier::BuildTupleOp>(loc, shape_type, shape_vals);
        return ctx.context.create_var(context, shape_var.getResult());
    }
    return py::list();
}

py::object dtype_impl(py::capsule context, py::capsule ssa_val)
{
    auto& ctx = get_py_context(context);
    auto value = unwrap_mlir<mlir::Value>(ssa_val);
    auto type = value.getType();
    if (auto tensor_type = type.dyn_cast<mlir::RankedTensorType>())
    {
        return ctx.context.create_type(tensor_type.getElementType());
    }
    return ctx.context.create_type(type);
}

py::object len_impl(py::capsule /*context*/, py::capsule ssa_val)
{
    auto value = unwrap_mlir<mlir::Value>(ssa_val);
    auto type = value.getType();
    if (auto tuple_type = type.dyn_cast<mlir::TupleType>())
    {
        return py::int_(tuple_type.size());
    }
    return py::int_(1);
}

py::object getitem_impl(py::capsule context, py::capsule ssa_val, py::handle index)
{
    auto& ctx = get_py_context(context);
    auto value = unwrap_mlir<mlir::Value>(ssa_val);
    auto& builder = ctx.builder;
    auto loc = ctx.loc;
    auto index_val = index.cast<int64_t>();
    auto type = value.getType();
    if (auto tuple_type = type.dyn_cast<mlir::TupleType>())
    {
        if (index_val < 0 || index_val >= static_cast<int64_t>(tuple_type.size()))
        {
            plier::report_error("Invalid getitem index");
        }
        if (auto parent_op = value.getDefiningOp<plier::BuildTupleOp>())
        {
            return ctx.context.create_var(context, parent_op.getOperand(static_cast<unsigned>(index_val)));
        }
        auto elem_type = tuple_type.getType(static_cast<size_t>(index_val));
        auto ind = builder.create<mlir::ConstantIndexOp>(loc, index_val);
        auto item = builder.create<plier::GetItemOp>(loc, elem_type, value, ind);
        return ctx.context.create_var(context, item.getResult());
    }
    else
    {
        if (0 != index_val)
        {
            plier::report_error("Invalid getitem index");
        }
        return ctx.context.create_var(context, value);
    }
}

template<typename Op>
mlir::Value binop_func(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value lhs, mlir::Value rhs)
{
    return builder.create<Op>(loc, lhs, rhs);
}

py::object binop_impl(py::capsule context, py::capsule ssa_val, py::handle rhs, py::str op)
{
    auto& ctx = get_py_context(context);
    auto& builder = ctx.builder;
    auto loc = ctx.loc;
    auto lhs = unwrap_mlir<mlir::Value>(ssa_val);

    auto type = lhs.getType();
    if (!type.isa<mlir::IntegerType, mlir::IndexType, mlir::FloatType, mlir::ShapedType>())
    {
        plier::report_error("Invalid binop arg type");
    }

    auto is_float = [&]()->bool
    {
        if (auto shaped_type = type.dyn_cast<mlir::ShapedType>())
        {
            return shaped_type.getElementType().isa<mlir::FloatType>();
        }
        return type.isa<mlir::FloatType>();
    }();

    using binop_func_t = mlir::Value(*)(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value lhs, mlir::Value rhs);
    const std::tuple<llvm::StringRef, binop_func_t, binop_func_t> funcs[] = {
        {"*", &binop_func<mlir::MulIOp>, &binop_func<mlir::MulFOp>},
    };

    auto op_name = static_cast<std::string>(op);
    for (auto f : funcs)
    {
        auto name = std::get<0>(f);
        auto func = (is_float ? std::get<2>(f) : std::get<1>(f));
        if (name == op_name)
        {
            auto rhs_var = do_cast(loc, builder, ctx.context.unwrap_val(loc, builder, rhs), type);
            auto res = func(loc, builder, lhs, rhs_var);
            return ctx.context.create_var(context, res);
        }
    }
    plier::report_error("Unhandled binop type");
}

void setup_py_var(pybind11::handle var)
{
    py::setattr(var, "_shape", py::cpp_function(&shape_impl));
    py::setattr(var, "_dtype", py::cpp_function(&dtype_impl));
    py::setattr(var, "_len", py::cpp_function(&len_impl));
    py::setattr(var, "_getitem", py::cpp_function(&getitem_impl));
    py::setattr(var, "_binop", py::cpp_function(&binop_impl));
}

PyLinalgResolver::Values unpack_results(py::handle object)
{
    PyLinalgResolver::Values ret;
    if (object.is_none())
    {
        return ret;
    }
    if (py::isinstance<py::tuple>(object))
    {
        auto tuple = object.cast<py::tuple>();
        ret.resize(tuple.size());
        for (auto it : llvm::enumerate(tuple))
        {
            ret[it.index()] = unwrap_ssa_val(it.value());
        }
        return ret;
    }
    ret.emplace_back(unwrap_ssa_val(object));
    return ret;
}
}

PyLinalgResolver::PyLinalgResolver():
    context(std::make_unique<Context>())
{
    auto builder_mod = py::module::import("numba.mlir.linalg_builder");
    context->var = builder_mod.attr("Var");
    context->type = builder_mod.attr("Type");
    context->builder = builder_mod.attr("Builder");
    context->inspect = py::module::import("inspect");
    context->types_mod = py::module::import("numba.core.types");
    context->compile_func = builder_mod.attr("compile_func");
    context->lookup_func = builder_mod.attr("lookup_func");
}

PyLinalgResolver::~PyLinalgResolver()
{

}

llvm::Optional<PyLinalgResolver::Values> PyLinalgResolver::rewrite_func(llvm::StringRef name, mlir::Location loc, mlir::OpBuilder& builder, mlir::ValueRange args, KWArgs kwargs)
{
    auto mangled_name = (llvm::Twine(name) + "()").str();
    return rewrite(mangled_name, loc, builder, args, kwargs);
}

llvm::Optional<PyLinalgResolver::Values> PyLinalgResolver::rewrite_attr(llvm::StringRef name, mlir::Location loc, mlir::OpBuilder& builder, mlir::Value arg)
{
    return rewrite(name, loc, builder, arg, {});
}

llvm::Optional<PyLinalgResolver::Values> PyLinalgResolver::rewrite(llvm::StringRef name, mlir::Location loc, mlir::OpBuilder& builder, mlir::ValueRange args, KWArgs kwargs)
{
    assert(!name.empty());
    if (!is_compatible_types(args) ||
        !is_compatible_types(llvm::make_second_range(kwargs)))
    {
        return {};
    }

    auto builder_func = context->lookup_func(py::str(name.data(), name.size()));
    if (builder_func.is_none())
    {
        return {};
    }

    PyBuilderContext py_builder_context{loc, builder, *context};
    auto py_context = py::capsule(&py_builder_context);
    auto py_args = get_args(
        context->inspect,
        builder_func,
        [&](auto val){ return context->create_var(py_context, val);},
        args,
        kwargs);
    if (py_args.is_none())
    {
        return {};
    }
    auto py_builder = context->builder(py_context);
    setup_py_builder(py_builder, builder, [&](auto type){ return context->create_type(type);});

    auto result = builder_func(py_builder, *py_args);
    if (result.is_none())
    {
        return {};
    }
    return unpack_results(result);
}
