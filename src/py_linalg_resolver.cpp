#include "py_linalg_resolver.hpp"

#include <pybind11/pybind11.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
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
    return type.isIntOrFloat() || type.isa<mlir::RankedTensorType>();
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

void setup_py_var(py::handle var);
}

struct PyLinalgResolver::Context
{
    py::handle var;
    py::handle builder;
    py::handle inspect;
    py::handle types_mod;
    py::handle compile_func;
    py::handle lookup_func;

    py::object create_var(py::capsule context, mlir::Value value)
    {
        if (auto literal = make_py_literal(value))
        {
            return *literal;
        }
        auto ret = var(context, wrap_mlir(value));
        setup_py_var(ret);
        return ret;
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
py::list get_args(py::handle inspect, py::handle func, llvm::function_ref<py::object(mlir::Value)> create_var,
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
    return ret;
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
    llvm::SmallVector<mlir::Value, 8> ret;
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
    llvm::SmallVector<llvm::StringRef, 8> ret(iterators.size());
    for (auto it : llvm::enumerate(iterators))
    {
        ret[it.index()] = mlir::StringAttr::get(it.value().cast<std::string>(), &ctx).getValue();
    }
    return ret;
}

auto get_affine_maps(py::list maps, mlir::MLIRContext& ctx)
{
    llvm::SmallVector<mlir::AffineMap, 8> ret(maps.size());
    for (auto it : llvm::enumerate(maps))
    {
        auto str = (llvm::Twine("affine_map<") + it.value().cast<std::string>() + ">").str();
        auto attr = mlir::parseAttribute(str, &ctx);
        ret[it.index()] = attr.cast<mlir::AffineMapAttr>().getValue();
    }
    return ret;
}

auto get_generic_op_body_types(mlir::ValueRange inputs, mlir::ValueRange outputs)
{
    llvm::SmallVector<mlir::Type, 8> ret;
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
    llvm::SmallVector<mlir::Type, 8> ret;
    ret.reserve(outputs.size());
    for (auto type : outputs.getTypes())
    {
        auto elem_type = type.cast<mlir::RankedTensorType>().getElementType();
        ret.emplace_back(elem_type);
    }
    return ret;
}

py::object broadcast_impl(py::capsule /*context*/, py::tuple args)
{
    if (1 == args.size())
    {
        return args[0];
    }
    else
    {
        return std::move(args);
    }
}

py::object init_tensor_impl(py::capsule context, py::handle shape, py::capsule dtype, py::handle init_val)
{
    auto& ctx = get_py_context(context);
    auto loc = ctx.loc;
    auto& builder = ctx.builder;
    auto elem_type = unwrap_mlir<mlir::Type>(dtype);
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
        llvm::SmallVector<mlir::Value, 8> shape_val(count);
        for (size_t i = 0; i < count; ++i)
        {
            shape_val[i] = do_cast(loc, builder, ctx.context.unwrap_val(loc, builder, shape[py::int_(i)]), index_type);
        }

        if (init_val.is_none())
        {
            init = builder.create<mlir::linalg::InitTensorOp>(loc, shape_val, elem_type);
        }
        else
        {
            auto val = do_cast(loc, builder, ctx.context.unwrap_val(loc, builder, init_val), elem_type);
            auto body = [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange /*indices*/)
            {
                builder.create<mlir::tensor::YieldOp>(loc, val);
            };
            llvm::SmallVector<int64_t, 8> shape(count, -1);
            auto type = mlir::RankedTensorType::get(shape, elem_type);
            init = builder.create<mlir::tensor::GenerateOp>(loc, type, shape_val, body);
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
    llvm::SmallVector<llvm::StringRef, 8> iterators(rank, "parallel");
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
        llvm::SmallVector<mlir::Value, 8> ret(vals.size());
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

py::object from_elements_impl(py::capsule context, py::handle values, py::capsule dtype)
{
    auto& ctx = get_py_context(context);
    auto& builder = ctx.builder;
    auto loc = ctx.loc;
    auto type = unwrap_mlir<mlir::Type>(dtype);

    llvm::SmallVector<mlir::Value, 8> vals(container_size(values));
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

    llvm::SmallVector<mlir::Value, 8> ind(container_size(indices));
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

void setup_py_builder(py::handle builder, mlir::OpBuilder& b)
{
    py::setattr(builder, "_broadcast", py::cpp_function(&broadcast_impl));
    py::setattr(builder, "_init_tensor", py::cpp_function(&init_tensor_impl));
    py::setattr(builder, "_fill_tensor", py::cpp_function(&fill_tensor_impl));
    py::setattr(builder, "_generic", py::cpp_function(&generic_impl));
    py::setattr(builder, "_from_elements", py::cpp_function(&from_elements_impl));
    py::setattr(builder, "_extract", py::cpp_function(&extract_impl));

    auto add_type = [&](const char* name, mlir::Type type)
    {
        py::setattr(builder, name, wrap_mlir(type));
    };

    add_type("int8", b.getIntegerType(8));
    add_type("int16", b.getIntegerType(16));
    add_type("int32", b.getIntegerType(32));
    add_type("int64", b.getIntegerType(64));

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
        llvm::SmallVector<mlir::Value, 8> shape_vals(shape.size());
        for (auto it : llvm::enumerate(shape))
        {
            auto i = it.index();
            mlir::Value mlir_dim = builder.create<mlir::DimOp>(loc, value, i);
            shape_vals[i] = mlir_dim;
        }
        llvm::SmallVector<mlir::Type, 8> shape_types(shape.size(), builder.getIndexType());
        auto shape_type = mlir::TupleType::get(builder.getContext(), shape_types);
        auto shape_var = builder.create<plier::BuildTupleOp>(loc, shape_type, shape_vals);
        return ctx.context.create_var(context, shape_var.getResult());
    }
    return py::list();
}

py::object dtype_impl(py::capsule /*context*/, py::capsule ssa_val)
{
    auto value = unwrap_mlir<mlir::Value>(ssa_val);
    auto type = value.getType();
    if (auto tensor_type = type.dyn_cast<mlir::RankedTensorType>())
    {
        return wrap_mlir(tensor_type.getElementType());
    }
    return wrap_mlir(type);
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
            plier::report_error("Invelid getitem index");
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
            plier::report_error("Invelid getitem index");
        }
        return ctx.context.create_var(context, value);
    }
}

void setup_py_var(pybind11::handle var)
{
    py::setattr(var, "_shape", py::cpp_function(&shape_impl));
    py::setattr(var, "_dtype", py::cpp_function(&dtype_impl));
    py::setattr(var, "_len", py::cpp_function(&len_impl));
    py::setattr(var, "_getitem", py::cpp_function(&getitem_impl));
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
    context->builder = builder_mod.attr("Builder");
    context->inspect = py::module::import("inspect");
    context->types_mod = py::module::import("numba.core.types");
    context->compile_func = builder_mod.attr("compile_func");
    context->lookup_func = builder_mod.attr("lookup_func");
}

PyLinalgResolver::~PyLinalgResolver()
{

}

llvm::Optional<PyLinalgResolver::Values> PyLinalgResolver::rewrite(llvm::StringRef name, mlir::Location loc, mlir::OpBuilder& builder, mlir::ValueRange args, llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs)
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
    setup_py_builder(py_builder, builder);

    auto result = builder_func(py_builder, *py_args);
    if (result.is_none())
    {
        return {};
    }
    return unpack_results(result);
}
