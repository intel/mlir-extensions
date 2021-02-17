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
bool is_compatible_types(mlir::TypeRange types)
{
    return !types.empty() && llvm::all_of(types, [](mlir::Type t)
    {
        return t.isIntOrFloat() || t.isa<mlir::RankedTensorType>();
    });
}

py::handle get_dim(int64_t val)
{
    if (val == -1)
    {
        return py::none();
    }
    return py::int_(val);
}

size_t py_func_arg_count(py::handle signature, py::handle func)
{
    return py::len(signature(func).attr("parameters"));
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

auto unwrap_shape(py::list shape)
{
    llvm::SmallVector<mlir::Value, 8> ret;
    ret.reserve(shape.size());
    for (auto elem : shape)
    {
        ret.push_back(unwrap_ssa_val(elem));
    }
    return ret;
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
}

struct PyLinalgResolver::Context
{
    py::handle var;
    py::handle val;
    py::handle builder;
    py::handle signature;
    py::handle types_mod;
    py::handle compile_func;
    py::handle lookup_func;

    py::object create_var(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value value)
    {
        if (value.getType().isa<mlir::RankedTensorType>())
        {
            auto make_dim_val = [&](auto dim, auto ssa_val)
            {
                return val(get_dim(dim), wrap_mlir(ssa_val));
            };
            auto mlir_type = value.getType().cast<mlir::RankedTensorType>();
            auto shape = mlir_type.getShape();
            auto elem_type = mlir_type.getElementType();
            py::list py_shape(shape.size());
            for (auto it2 : llvm::enumerate(shape))
            {
                mlir::Value mlir_dim = builder.create<mlir::DimOp>(loc, value, it2.index());
                py_shape[it2.index()] = make_dim_val(it2.value(), mlir_dim);
            }
            return var(wrap_mlir(value), py_shape, wrap_mlir(elem_type));
        }
        return var(wrap_mlir(value), py::list(), wrap_mlir(value.getType()));
    }

    mlir::FuncOp compile_body(py::handle body, py::list arg_types)
    {
        auto func = compile_func(body, arg_types).cast<py::capsule>();
        auto mlir_func = mlir::cast<mlir::FuncOp>(static_cast<mlir::Operation*>(func));
        mlir_func.setPrivate();
        mlir_func->setAttr(plier::attributes::getForceInlineName(), mlir::UnitAttr::get(mlir_func->getContext()));
        return mlir_func;
    }

    py::object wrap_result(mlir::Location loc, mlir::OpBuilder& builder, mlir::ValueRange values)
    {
        if (values.empty())
        {
            return py::none();
        }
        if (values.size() == 1)
        {
            return create_var(loc, builder, values.front());
        }
        py::tuple ret(values.size());
        for (auto it : llvm::enumerate(values))
        {
            ret[it.index()] = create_var(loc, builder, it.value());
        }
        return std::move(ret);
    }
};

namespace
{

PyBuilderContext& get_py_context(py::capsule& ctx)
{
    return *static_cast<PyBuilderContext*>(ctx);
}

mlir::Value get_var_value(py::handle var)
{
    return unwrap_mlir<mlir::Value>(var.attr("_ssa_val").cast<py::capsule>());
}

auto get_types(mlir::ValueRange values)
{
    return values.getTypes();
}

auto get_agrs_from_tuple(py::handle args)
{
    llvm::SmallVector<mlir::Value, 8> ret;
    if (py::isinstance<py::tuple>(args))
    {
        auto tuple = args.cast<py::tuple>();
        ret.resize(tuple.size());
        for (auto it : llvm::enumerate(tuple))
        {
            ret[it.index()] = get_var_value(it.value());
        }
    }
    else
    {
        ret.emplace_back(get_var_value(args));
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

py::object init_tensor_impl(py::capsule context, py::list shape, py::capsule dtype)
{
    auto& ctx = get_py_context(context);
    auto elem_type = unwrap_mlir<mlir::Type>(dtype);
    mlir::Value init;
    if (shape.empty())
    {
        // TODO: undef
        auto zero_val = plier::getZeroVal(elem_type);
        assert(zero_val);
        init = ctx.builder.create<mlir::ConstantOp>(ctx.loc, zero_val);
    }
    else
    {
        init = ctx.builder.create<mlir::linalg::InitTensorOp>(ctx.loc, unwrap_shape(shape), elem_type);
    }
    return ctx.context.create_var(ctx.loc, ctx.builder, init);
}

py::object generic_impl(py::capsule context, py::handle inputs, py::handle outputs, py::list iterators, py::list maps, py::handle body)
{
    auto& ctx = get_py_context(context);
    auto loc = ctx.loc;
    auto& builder = ctx.builder;
    auto& mlir_context = *builder.getContext();

    auto inputs_args = get_agrs_from_tuple(inputs);
    auto output_args = get_agrs_from_tuple(outputs);
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
        return ctx.context.wrap_result(loc, builder, cast_values(res.getResults(), ret_types));
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
        return ctx.context.wrap_result(loc, builder, generic_op.getResults());
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
    return ctx.context.create_var(ctx.loc, ctx.builder, res);
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
    auto res = builder.create<mlir::tensor::ExtractOp>(loc, get_var_value(value), ind);
    return ctx.context.create_var(ctx.loc, ctx.builder, res);
}

void setup_py_builder(py::handle builder, mlir::OpBuilder& b)
{
    py::setattr(builder, "_broadcast", py::cpp_function(&broadcast_impl));
    py::setattr(builder, "_init_tensor", py::cpp_function(&init_tensor_impl));
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
    context->val = builder_mod.attr("Val");
    context->builder = builder_mod.attr("Builder");
    context->signature = py::module::import("inspect").attr("signature");
    context->types_mod = py::module::import("numba.core.types");
    context->compile_func = builder_mod.attr("compile_func");
    context->lookup_func = builder_mod.attr("lookup_func");
}

PyLinalgResolver::~PyLinalgResolver()
{

}

llvm::Optional<PyLinalgResolver::Values> PyLinalgResolver::rewrite(llvm::StringRef name, mlir::Location loc, mlir::OpBuilder& builder, mlir::ValueRange args)
{
    assert(!name.empty());
    if (!is_compatible_types(args.getTypes()))
    {
        return {};
    }

    auto builder_func = context->lookup_func(py::str(name.data(), name.size()));
    if (builder_func.is_none() || py_func_arg_count(context->signature, builder_func) != (args.size() + 1))
    {
        return {};
    }

    PyBuilderContext py_builder_context{loc, builder, *context};
    auto py_builder = context->builder(py::capsule(&py_builder_context));
    setup_py_builder(py_builder, builder);

    assert(!args.empty());
    auto module = args.front().getParentRegion()->getParentOfType<mlir::ModuleOp>();
    assert(module);

    py::list py_args(args.size());
    for (auto it : llvm::enumerate(args))
    {
        auto index = static_cast<unsigned>(it.index());
        auto mlir_arg = it.value();
        py_args[index] = context->create_var(loc, builder, mlir_arg);
    }

    auto result = builder_func(py_builder, *py_args);
    return unpack_results(result);
}
