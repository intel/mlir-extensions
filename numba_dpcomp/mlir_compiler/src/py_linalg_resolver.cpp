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

#include "py_linalg_resolver.hpp"

#include <pybind11/pybind11.h>

#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Parser.h>
#include <mlir/Transforms/DialectConversion.h>

#include "pipelines/plier_to_linalg.hpp"
#include "pipelines/plier_to_std.hpp"
#include "plier/dialect.hpp"
#include "plier/transforms/const_utils.hpp"
#include "plier/transforms/func_utils.hpp"
#include "plier/utils.hpp"
#include "py_map_types.hpp"

namespace py = pybind11;

struct PyBuilderContext {
  mlir::Location loc;
  mlir::OpBuilder &builder;
  mlir::TypeConverter typeConverter;
  PyLinalgResolver::Context &context;
};

namespace {
std::string to_str(mlir::Type type) {
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  ss << type;
  ss.flush();
  return ret;
}

std::string to_str(mlir::TypeRange typesRange) {
  std::string str;
  llvm::raw_string_ostream ss(str);
  for (auto type : typesRange) {
    ss << type << " ";
  }
  ss.flush();
  return str;
}

std::string to_str(py::handle obj) { return py::str(obj).cast<std::string>(); }

py::object mapTypesToNumbaChecked(py::handle typesMod,
                                  mlir::TypeRange typesRange) {
  auto funcTypes = map_types_to_numba(typesMod, typesRange);
  if (funcTypes.is_none()) {
    assert(!typesRange.empty());
    auto context = typesRange.front().getContext();
    mlir::TypeConverter converter;
    populate_std_type_converter(*context, converter);
    populate_array_type_converter(*context, converter);
    llvm::SmallVector<mlir::Type> convertedTypes(typesRange.size());
    for (auto it : llvm::enumerate(typesRange)) {
      auto oldType = it.value();
      auto newType = converter.convertType(oldType);
      if (!newType) {
        newType = oldType;
      }
      convertedTypes[it.index()] = newType;
    }
    funcTypes = map_types_to_numba(typesMod, convertedTypes);
    if (funcTypes.is_none()) {
      plier::report_error(llvm::Twine("map_types_to_numba failed: ") +
                          to_str(typesRange));
    }
  }
  return funcTypes;
}

mlir::Type makeSignlessType(mlir::Type type) {
  if (auto tensor = type.dyn_cast<mlir::RankedTensorType>()) {
    auto origElemType = tensor.getElementType();
    auto signlessElemType = makeSignlessType(origElemType);
    if (origElemType != signlessElemType) {
      return mlir::RankedTensorType::get(tensor.getShape(), signlessElemType,
                                         tensor.getEncoding());
    }
  } else if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
    if (!intType.isSignless()) {
      return mlir::IntegerType::get(intType.getContext(), intType.getWidth());
    }
  }
  return type;
}

mlir::Value doSignCast(mlir::OpBuilder &builder, mlir::Location &loc,
                       mlir::Value val) {
  auto origType = val.getType();
  auto signlessType = makeSignlessType(origType);
  if (signlessType != origType) {
    val = builder.create<plier::SignCastOp>(loc, signlessType, val);
  }
  return val;
}

mlir::Value doSignCast(mlir::OpBuilder &builder, mlir::Location &loc,
                       mlir::Value val, mlir::Type dstType) {
  auto origType = val.getType();
  if (dstType != origType) {
    val = builder.create<plier::SignCastOp>(loc, dstType, val);
  }
  return val;
}

auto doSignCast(mlir::OpBuilder &builder, mlir::Location &loc,
                mlir::ValueRange vals) {
  llvm::SmallVector<mlir::Value> ret(vals.size());
  for (auto it : llvm::enumerate(vals)) {
    ret[it.index()] = doSignCast(builder, loc, it.value());
  }
  return ret;
}

auto doSignCast(mlir::OpBuilder &builder, mlir::Location &loc,
                mlir::ValueRange vals, mlir::TypeRange dstTypes) {
  assert(vals.size() == dstTypes.size());
  llvm::SmallVector<mlir::Value> ret(vals.size());
  for (auto it : llvm::enumerate(llvm::zip(vals, dstTypes))) {
    auto val = std::get<0>(it.value());
    auto type = std::get<1>(it.value());
    ret[it.index()] = doSignCast(builder, loc, val, type);
  }
  return ret;
}

auto getTypes(mlir::ValueRange values) {
  auto types = values.getTypes();
  llvm::SmallVector<mlir::Type> ret(types.begin(), types.end());
  return ret;
}

bool is_compatible_type(mlir::Type type) {
  if (auto tuple_type = type.dyn_cast<mlir::TupleType>()) {
    return llvm::all_of(tuple_type, &is_compatible_type);
  }
  return type.isa<mlir::IntegerType, mlir::IndexType, mlir::FloatType,
                  mlir::RankedTensorType, mlir::NoneType, plier::LiteralType,
                  plier::TypeVar>();
}

template <typename R> bool is_compatible_types(R &&vals) {
  return llvm::all_of(
      vals, [](auto val) { return is_compatible_type(val.getType()); });
}

template <typename T> py::capsule wrap_mlir(T val) {
  return py::capsule(val.getAsOpaquePointer());
}

template <typename T> T unwrap_mlir(py::capsule obj) {
  return T::getFromOpaquePointer(static_cast<const void *>(obj));
}

auto unwrap_ssa_val(py::handle obj) {
  return unwrap_mlir<mlir::Value>(obj.attr("_ssa_val").cast<py::capsule>());
}

auto unwrap_type(py::handle obj) {
  if (py::hasattr(obj, "_ssa_val")) {
    auto val = unwrap_ssa_val(obj);
    if (auto type = val.getType().dyn_cast<plier::TypeVar>()) {
      return type.getType();
    }
  } else if (py::hasattr(obj, "_mlir_type")) {
    return unwrap_mlir<mlir::Type>(obj.attr("_mlir_type").cast<py::capsule>());
  }
  plier::report_error(llvm::Twine("Invalid type object: ") +
                      to_str(obj.get_type()));
}

size_t container_size(py::handle obj) {
  if (py::isinstance<py::tuple>(obj)) {
    return obj.cast<py::tuple>().size();
  }
  if (py::isinstance<py::list>(obj)) {
    return obj.cast<py::list>().size();
  }
  return 1;
}

template <typename F> void container_iterate(py::handle obj, F &&func) {
  auto impl = [&](auto cont) {
    for (auto it : llvm::enumerate(cont)) {
      func(it.index(), it.value());
    }
  };
  if (py::isinstance<py::tuple>(obj)) {
    impl(obj.cast<py::tuple>());
  } else if (py::isinstance<py::list>(obj)) {
    impl(obj.cast<py::list>());
  } else {
    func(std::size_t(0), obj);
  }
}

template <typename UnwrapFunc>
auto to_values(py::handle obj, UnwrapFunc &&unwrapFunc) {
  llvm::SmallVector<mlir::Value> ret(container_size(obj));
  container_iterate(
      obj, [&](auto index, py::handle elem) { ret[index] = unwrapFunc(elem); });
  return ret;
}

llvm::Optional<py::object> getPyLiteral(mlir::Attribute attr) {
  assert(attr);
  if (auto intAttr = attr.dyn_cast<mlir::IntegerAttr>()) {
    if (auto intType = attr.getType().dyn_cast<mlir::IntegerType>()) {
      // Ignore index type
      if (intType.getWidth() == 1) {
        return py::bool_(intAttr.getInt() != 0);
      }
    }
    return py::int_(plier::getIntAttrValue(intAttr));
  }
  if (auto floatAttr = attr.dyn_cast<mlir::FloatAttr>()) {
    return py::float_(floatAttr.getValueAsDouble());
  }
  return {};
}

llvm::Optional<py::object> makePyLiteral(mlir::Value val) {
  assert(val);
  if (auto literal = val.getType().dyn_cast<plier::LiteralType>()) {
    return getPyLiteral(literal.getValue());
  }

  if (auto cast = val.getDefiningOp<plier::SignCastOp>()) {
    val = cast.value();
  }

  if (auto attr = plier::getConstVal<mlir::Attribute>(val)) {
    return getPyLiteral(attr);
  }
  return {};
}

mlir::Value doCast(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value val, mlir::Type type) {
  if (val.getType() != type) {
    return builder.create<plier::CastOp>(loc, type, val);
  }
  return val;
}

bool cmp_capsule(py::capsule a1, py::capsule a2) {
  return static_cast<void *>(a1) == static_cast<void *>(a2);
}

void setup_py_var(py::handle var);
} // namespace

struct PyLinalgResolver::Context {
  py::handle var;
  py::handle type;
  py::handle builder;
  py::handle inspect;
  py::handle types_mod;
  py::handle compile_func;
  py::handle lookup_func;

  py::object create_var(py::capsule context, mlir::Value value) {
    assert(value);
    auto type = value.getType();
    if (type.isa<mlir::NoneType>()) {
      return py::none();
    }
    if (auto typevar = type.dyn_cast<plier::TypeVar>()) {
      return create_type(typevar.getType());
    }
    if (auto literal = makePyLiteral(value)) {
      return *literal;
    }
    auto ret = var(context, wrap_mlir(value));
    setup_py_var(ret);
    return ret;
  }

  py::object create_type(mlir::Type t) {
    return type(wrap_mlir(t), py::cpp_function(&cmp_capsule));
  }

  mlir::FuncOp compile_body(py::handle body, py::list arg_types) {
    auto func = compile_func(body, arg_types).cast<py::capsule>();
    auto mlir_func =
        mlir::cast<mlir::FuncOp>(static_cast<mlir::Operation *>(func));
    mlir_func.setPrivate();
    mlir_func->setAttr(plier::attributes::getForceInlineName(),
                       mlir::UnitAttr::get(mlir_func->getContext()));
    return mlir_func;
  }

  py::object wrap_result(py::capsule context, mlir::ValueRange values) {
    if (values.empty()) {
      return py::none();
    }
    if (values.size() == 1) {
      return create_var(context, values.front());
    }
    py::tuple ret(values.size());
    for (auto it : llvm::enumerate(values)) {
      ret[it.index()] = create_var(context, it.value());
    }
    return std::move(ret);
  }

  mlir::Value unwrap_val(mlir::Location loc, mlir::OpBuilder &builder,
                         py::handle obj) {
    if (py::isinstance(obj, var)) {
      return unwrap_ssa_val(obj);
    }
    if (py::isinstance(obj, type)) {
      auto type = plier::TypeVar::get(unwrap_type(obj));
      return builder.create<plier::UndefOp>(loc, type);
    }
    if (obj.is_none()) {
      auto type = mlir::NoneType::get(builder.getContext());
      return builder.create<plier::UndefOp>(loc, type);
    }
    if (py::isinstance<py::tuple>(obj)) {
      llvm::SmallVector<mlir::Value> elems(py::len(obj));
      for (auto it : llvm::enumerate(obj)) {
        elems[it.index()] = unwrap_val(loc, builder, it.value());
      }
      mlir::ValueRange vr(elems);
      auto resType = mlir::TupleType::get(builder.getContext(), vr.getTypes());
      return builder.create<plier::BuildTupleOp>(loc, resType, elems);
    }
    if (py::isinstance<py::bool_>(obj)) {
      auto type = builder.getI1Type();
      auto attr = builder.getIntegerAttr(type, (obj.cast<bool>() ? 1 : 0));
      return builder.create<mlir::ConstantOp>(loc, attr);
    }
    if (py::isinstance<py::int_>(obj)) {
      auto attr = builder.getI64IntegerAttr(obj.cast<int64_t>());
      return builder.create<mlir::ConstantOp>(loc, attr);
    }
    if (py::isinstance<py::float_>(obj)) {
      auto attr = builder.getF64FloatAttr(obj.cast<double>());
      return builder.create<mlir::ConstantOp>(loc, attr);
    }
    plier::report_error(llvm::Twine("Invalid element type: ") +
                        to_str(obj.get_type()));
  }
};

namespace {
py::object
get_args(py::handle inspect, py::handle func,
         llvm::function_ref<py::object(mlir::Value)> create_var,
         mlir::ValueRange args,
         llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs) {
  auto sig_func = inspect.attr("signature");
  auto sig = sig_func(func);
  auto params = sig.attr("parameters");
  auto params_list = py::list(params);
  params_list = params_list[py::slice(
      1, static_cast<int64_t>(params_list.size()), 1)]; // skip builder param
  auto empty = inspect.attr("Parameter").attr("empty");

  py::list ret(py::len(params_list));
  for (auto it : llvm::enumerate(params_list)) {
    auto index = it.index();
    auto param_name = it.value();
    auto param = params[param_name];
    if (!args.empty()) {
      ret[index] = create_var(args.front());
      args = args.drop_front();
      continue;
    }
    if (!kwargs.empty()) {
      auto name = param_name.cast<std::string>();
      auto val = [&]() -> mlir::Value {
        for (auto kwarg : kwargs) {
          if (kwarg.first == name) {
            return kwarg.second;
          }
        }
        return {};
      }();
      if (val) {
        ret[index] = create_var(val);
        continue;
      }
    }
    auto def_val = param.attr("default");
    if (!def_val.is(empty)) {
      ret[index] = def_val;
    } else {
      return py::none();
    }
  }
  if (!args.empty()) {
    return py::none();
  }
  return std::move(ret);
}

PyBuilderContext &get_py_context(py::capsule &ctx) {
  return *static_cast<PyBuilderContext *>(ctx);
}

auto getAgrsFromTuple(py::handle args,
                      llvm::function_ref<mlir::Value(py::handle)> unpack) {
  llvm::SmallVector<mlir::Value> ret;
  if (args.is_none()) {
    return ret;
  }
  if (py::isinstance<py::tuple>(args)) {
    auto tuple = args.cast<py::tuple>();
    ret.resize(tuple.size());
    for (auto it : llvm::enumerate(tuple)) {
      ret[it.index()] = unpack(it.value());
    }
  } else {
    ret.emplace_back(unpack(args));
  }
  return ret;
}

auto getIterators(py::list iterators, mlir::MLIRContext &ctx) {
  llvm::SmallVector<llvm::StringRef> ret(iterators.size());
  for (auto it : llvm::enumerate(iterators)) {
    ret[it.index()] =
        mlir::StringAttr::get(&ctx, it.value().cast<std::string>()).getValue();
  }
  return ret;
}

mlir::AffineMapAttr get_affine_map_attr(py::handle obj,
                                        mlir::MLIRContext &ctx) {
  auto str = (llvm::Twine("affine_map<") + obj.cast<std::string>() + ">").str();
  return mlir::parseAttribute(str, &ctx).cast<mlir::AffineMapAttr>();
}

auto getAffineMaps(py::list maps, mlir::MLIRContext &ctx) {
  llvm::SmallVector<mlir::AffineMap> ret(maps.size());
  for (auto it : llvm::enumerate(maps)) {
    ret[it.index()] = get_affine_map_attr(it.value(), ctx).getValue();
  }
  return ret;
}

auto getGenericOpBodyTypes(mlir::ValueRange inputs, mlir::ValueRange outputs) {
  llvm::SmallVector<mlir::Type> ret;
  ret.reserve(inputs.size() + outputs.size());
  for (auto r : {inputs, outputs}) {
    for (auto type : r.getTypes()) {
      auto elemType = [&]() {
        if (auto tensor = type.dyn_cast<mlir::RankedTensorType>()) {
          return tensor.getElementType();
        }
        return type;
      }();
      ret.emplace_back(elemType);
    }
  }
  return ret;
}

auto genericOpBodyResultTypes(mlir::ValueRange outputs) {
  llvm::SmallVector<mlir::Type> ret;
  ret.reserve(outputs.size());
  for (auto type : outputs.getTypes()) {
    auto elem_type = type.cast<mlir::RankedTensorType>().getElementType();
    ret.emplace_back(elem_type);
  }
  return ret;
}

bool is_int(mlir::Type type) {
  return type.isa<mlir::IntegerType, mlir::IndexType>();
}

unsigned get_int_bit_width(mlir::Type type) {
  if (type.isa<mlir::IntegerType>()) {
    return type.cast<mlir::IntegerType>().getWidth();
  }
  if (type.isa<mlir::IndexType>()) {
    return 64; // TODO
  }
  llvm_unreachable("No an integer type");
}

bool is_float(mlir::Type type) { return type.isa<mlir::FloatType>(); }

unsigned get_float_bit_width(mlir::Type type) {
  return type.cast<mlir::FloatType>().getWidth();
}

mlir::Type broadcast_type(mlir::Type type1, mlir::Type type2) {
  if (type1 == type2) {
    return type1;
  }
  // TODO
  if (is_int(type1) && is_int(type2)) {
    bool isSigned = type1.isSignedInteger() || type2.isSignedInteger();
    auto signess =
        (isSigned ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned);
    auto width = std::max(get_int_bit_width(type1), get_int_bit_width(type2));
    return mlir::IntegerType::get(type1.getContext(), width, signess);
  }
  if (is_float(type1) && is_float(type2)) {
    return (get_float_bit_width(type1) > get_float_bit_width(type2) ? type1
                                                                    : type2);
  }
  if (is_float(type1) && is_int(type2)) {
    return type1;
  }
  if (is_int(type1) && is_float(type2)) {
    return type2;
  }
  llvm_unreachable("Unable to broadcast type");
}

mlir::Value broadcast_dim(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value val1, mlir::Value val2) {
  assert(val1.getType().isa<mlir::IndexType>());
  assert(val2.getType().isa<mlir::IndexType>());
  auto one = builder.create<mlir::ConstantIndexOp>(loc, 1);
  auto cond =
      builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, val1, one);
  return builder.create<mlir::SelectOp>(loc, cond, val2, val1);
}

mlir::Value expand_dim(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value initial, mlir::Value src, unsigned dim,
                       mlir::ValueRange targetShape) {
  auto context = builder.getContext();
  auto srcType = src.getType().cast<mlir::ShapedType>();
  auto numDims = static_cast<unsigned>(srcType.getRank());
  auto shape = llvm::to_vector<8>(srcType.getShape());
  shape[dim] = -1;
  mlir::Type targetType =
      mlir::RankedTensorType::get(shape, srcType.getElementType());
  auto dimVal = builder.create<mlir::tensor::DimOp>(loc, initial, dim);
  auto one = builder.create<mlir::ConstantIndexOp>(loc, 1);
  mlir::Value cond =
      builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, one, dimVal);
  llvm::SmallVector<mlir::Value> newShape(numDims);
  for (unsigned i = 0; i < numDims; ++i) {
    if (i == dim) {
      newShape[i] = targetShape[i];
    } else {
      newShape[i] = builder.create<mlir::tensor::DimOp>(loc, src, i);
    }
  }
  auto true_body = [&](mlir::OpBuilder &builder, mlir::Location loc) {
    assert(dim < shape.size());
    shape[dim] = 1;
    //        mlir::Type casted_type = mlir::RankedTensorType::get(shape,
    //        src_type.getElementType()); auto casted =
    //        builder.create<mlir::tensor::CastOp>(loc, casted_type,
    //        src).getResult();
    auto casted = src; // TODO
    auto init = builder
                    .create<mlir::linalg::InitTensorOp>(
                        loc, newShape, srcType.getElementType())
                    .getResult();
    llvm::SmallVector<mlir::AffineExpr> exprs(numDims);
    for (unsigned i = 0; i < numDims; ++i) {
      if (i == dim) {
        exprs[i] = mlir::getAffineConstantExpr(0, context);
      } else {
        exprs[i] = mlir::getAffineDimExpr(i, context);
      }
    }
    const mlir::AffineMap maps[] = {
        mlir::AffineMap::get(numDims, 0, exprs, context),
        mlir::AffineMap::getMultiDimIdentityMap(numDims, context),
    };
    llvm::SmallVector<mlir::StringRef> iterators(numDims, "parallel");

    auto body = [&](mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::ValueRange values) {
      assert(values.size() == 2);
      builder.create<mlir::linalg::YieldOp>(loc, values[0]);
    };

    auto expanded = builder.create<mlir::linalg::GenericOp>(
        loc, init.getType(), casted, init, maps, iterators, body);
    auto res = builder.createOrFold<mlir::tensor::CastOp>(
        loc, targetType, expanded.getResult(0));
    builder.create<mlir::scf::YieldOp>(loc, res);
  };
  auto false_body = [&](mlir::OpBuilder &builder, mlir::Location loc) {
    auto res = builder.create<mlir::tensor::CastOp>(loc, targetType, src);
    builder.create<mlir::scf::YieldOp>(loc, res.getResult());
  };
  return builder
      .create<mlir::scf::IfOp>(loc, targetType, cond, true_body, false_body)
      .getResult(0);
}

mlir::Value expand_dims(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value val, unsigned num_dims,
                        mlir::ValueRange target_shape) {
  assert(num_dims <= target_shape.size());
  if (num_dims < target_shape.size()) {
    target_shape = target_shape.drop_front(target_shape.size() - num_dims);
  }
  mlir::Value current = val;
  for (unsigned i = 0; i < num_dims; ++i) {
    current = expand_dim(builder, loc, val, current, i, target_shape);
  }
  current = builder.create<plier::EnforceShapeOp>(loc, current, target_shape);
  return current;
}

py::object broadcast_impl(py::capsule context, py::tuple args) {
  if (1 == args.size()) {
    return args[0];
  }
  auto &ctx = get_py_context(context);
  auto loc = ctx.loc;
  auto &builder = ctx.builder;
  llvm::SmallVector<mlir::Value> mlirArgs(args.size());
  for (auto it : llvm::enumerate(args)) {
    auto val = ctx.context.unwrap_val(loc, builder, it.value());
    mlirArgs[it.index()] = val;
  }
  using shape_t = llvm::SmallVector<mlir::Value>;
  auto getShape =
      [&](mlir::Value val) -> llvm::Optional<std::pair<shape_t, mlir::Type>> {
    auto type = val.getType();
    if (auto shaped = type.dyn_cast<mlir::ShapedType>()) {
      if (!shaped.hasRank()) {
        return {};
      }
      shape_t ret(static_cast<size_t>(shaped.getRank()));
      for (auto it : llvm::enumerate(ret)) {
        auto dim = builder.create<mlir::tensor::DimOp>(loc, val, it.index());
        ret[it.index()] = dim;
      }
      return std::make_pair(ret, shaped.getElementType());
    }
    if (type.isa<mlir::IntegerType, mlir::IndexType, mlir::FloatType>()) {
      return std::make_pair(shape_t{}, type);
    }
    return {};
  };
  mlir::Type resType;
  mlir::SmallVector<mlir::Value> shapeVals;
  if (auto shapeAndType = getShape(mlirArgs.front())) {
    resType = shapeAndType->second;
    shapeVals = shapeAndType->first;
  } else {
    return py::none();
  }

  for (auto arg : llvm::drop_begin(mlirArgs)) {
    auto shapeAndType = getShape(arg);
    if (!shapeAndType) {
      py::none();
    }
    resType = broadcast_type(resType, shapeAndType->second);
    auto newShapeVals = shapeAndType->first;
    for (auto it :
         llvm::zip(llvm::reverse(shapeVals), llvm::reverse(newShapeVals))) {
      auto &old_val = std::get<0>(it);
      auto new_val = std::get<1>(it);
      old_val = broadcast_dim(builder, loc, old_val, new_val);
    }
    if (newShapeVals.size() > shapeVals.size()) {
      auto front = llvm::makeArrayRef(newShapeVals).drop_back(shapeVals.size());
      assert(!front.empty());
      shapeVals.insert(shapeVals.begin(), front.begin(), front.end());
    }
  }

  py::tuple ret(mlirArgs.size());
  if (shapeVals.empty()) {
    for (auto it : llvm::enumerate(mlirArgs)) {
      mlir::Value val = it.value();
      if (val.getType() != resType) {
        val = builder.create<plier::CastOp>(loc, resType, val);
      }
      ret[it.index()] = ctx.context.create_var(context, val);
    }
    return std::move(ret);
  }

  llvm::SmallVector<int64_t> shape(static_cast<size_t>(shapeVals.size()), -1);
  auto tensorType = mlir::RankedTensorType::get(shape, resType);
  auto signlessResType = makeSignlessType(resType);
  auto signlessTensorType = mlir::RankedTensorType::get(shape, signlessResType);
  for (auto it : llvm::enumerate(mlirArgs)) {
    mlir::Value val = it.value();
    val = doSignCast(builder, loc, val);
    if (auto srcType = val.getType().dyn_cast<mlir::ShapedType>()) {
      assert(srcType.hasRank());
      val = expand_dims(builder, loc, val,
                        static_cast<unsigned>(srcType.getRank()), shapeVals);
    }
    if (val.getType() != signlessTensorType) {
      auto type = val.getType();
      if (auto src_type = type.dyn_cast<mlir::ShapedType>()) {
        assert(src_type.hasRank());
        auto src_num_dims = static_cast<unsigned>(src_type.getRank());
        auto num_dims = static_cast<unsigned>(signlessTensorType.getRank());
        auto init = builder
                        .create<mlir::linalg::InitTensorOp>(
                            loc, shapeVals, signlessTensorType.getElementType())
                        .getResult();
        mlir::AffineMap maps[] = {
            mlir::AffineMap::getMinorIdentityMap(num_dims, src_num_dims,
                                                 builder.getContext()),
            //                    mlir::AffineMap::getMultiDimIdentityMap(num_dims,
            //                    builder.getContext()).getMajorSubMap(src_num_dims),
            mlir::AffineMap::getMultiDimIdentityMap(num_dims,
                                                    builder.getContext()),
        };
        llvm::SmallVector<llvm::StringRef> iterators(num_dims, "parallel");
        auto body = [&](mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::ValueRange values) {
          assert(values.size() == 2);
          auto res = builder.create<plier::CastOp>(
              loc, signlessTensorType.getElementType(), values[0]);
          builder.create<mlir::linalg::YieldOp>(loc, res.getResult());
        };
        val = builder
                  .create<mlir::linalg::GenericOp>(loc, signlessTensorType, val,
                                                   init, maps, iterators, body)
                  .getResult(0);
      } else {
        if (signlessTensorType.getElementType() != type) {
          val = builder.create<plier::CastOp>(
              loc, signlessTensorType.getElementType(), val);
        }
        val = builder.create<mlir::tensor::FromElementsOp>(loc, val);
        auto num_dims = static_cast<unsigned>(signlessTensorType.getRank());
        auto init = builder
                        .create<mlir::linalg::InitTensorOp>(
                            loc, shapeVals, signlessTensorType.getElementType())
                        .getResult();
        mlir::AffineMap maps[] = {
            mlir::AffineMap::get(
                num_dims, 0,
                mlir::getAffineConstantExpr(0, builder.getContext())),
            mlir::AffineMap::getMultiDimIdentityMap(num_dims,
                                                    builder.getContext()),
        };
        llvm::SmallVector<llvm::StringRef> iterators(num_dims, "parallel");
        auto body = [&](mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::ValueRange values) {
          assert(values.size() == 2);
          builder.create<mlir::linalg::YieldOp>(loc, values[0]);
        };
        val = builder
                  .create<mlir::linalg::GenericOp>(loc, signlessTensorType, val,
                                                   init, maps, iterators, body)
                  .getResult(0);
      }
    }
    val = doSignCast(builder, loc, val, tensorType);
    ret[it.index()] = ctx.context.create_var(context, val);
  }
  return std::move(ret);
}

py::object init_tensor_impl(py::capsule context, py::handle shape,
                            py::handle dtype, py::handle init_val) {
  auto &ctx = get_py_context(context);
  auto loc = ctx.loc;
  auto &builder = ctx.builder;
  auto elemType = unwrap_type(dtype);
  auto signlessElemType = makeSignlessType(elemType);
  mlir::Value init;
  auto indexType = builder.getIndexType();
  auto count = py::len(shape);
  llvm::SmallVector<mlir::Value> shapeVal(count);
  llvm::SmallVector<int64_t> staticShape(count, -1);
  for (size_t i = 0; i < count; ++i) {
    auto elem = shape[py::int_(i)];
    if (py::isinstance<py::int_>(elem)) {
      staticShape[i] = elem.cast<int64_t>();
    }
    auto elemVal = ctx.context.unwrap_val(loc, builder, elem);
    elemVal = doSignCast(builder, loc, elemVal);
    shapeVal[i] = doCast(builder, loc, elemVal, indexType);
  }

  if (init_val.is_none()) {
    init = builder.create<mlir::linalg::InitTensorOp>(loc, shapeVal,
                                                      signlessElemType);
  } else {
    auto val =
        doCast(builder, loc, ctx.context.unwrap_val(loc, builder, init_val),
               signlessElemType);
    llvm::SmallVector<int64_t> shape(count, -1);
    auto type = mlir::RankedTensorType::get(shape, signlessElemType);
    auto body = [&](mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::ValueRange /*indices*/) {
      builder.create<mlir::tensor::YieldOp>(loc, val);
    };
    init = builder.create<mlir::tensor::GenerateOp>(loc, type, shapeVal, body);
  }
  if (llvm::any_of(staticShape, [](auto val) { return val >= 0; })) {
    auto newType = mlir::RankedTensorType::get(staticShape, signlessElemType);
    init = builder.create<mlir::tensor::CastOp>(loc, newType, init);
  }
  auto resTensorTypeSigness = init.getType().cast<mlir::RankedTensorType>();
  auto resTensorType =
      mlir::RankedTensorType::get(resTensorTypeSigness.getShape(), elemType,
                                  resTensorTypeSigness.getEncoding());
  init = doSignCast(builder, loc, init, resTensorType);
  return ctx.context.create_var(context, init);
}

py::object fill_tensor_impl(py::capsule context, py::handle tensor,
                            py::handle value) {
  auto &ctx = get_py_context(context);
  auto loc = ctx.loc;
  auto &builder = ctx.builder;
  auto tensor_val = ctx.context.unwrap_val(loc, builder, tensor);
  auto tensor_type = tensor_val.getType().cast<mlir::ShapedType>();
  auto init_val = ctx.context.unwrap_val(loc, builder, value);
  if (init_val.getType() != tensor_type.getElementType()) {
    init_val = builder.create<plier::CastOp>(loc, tensor_type.getElementType(),
                                             init_val);
  }

  //    auto val = builder.create<mlir::linalg::FillOp>(loc, tensor_type,
  //    tensor_val, init_val);
  auto rank = static_cast<unsigned>(tensor_type.getRank());
  mlir::AffineMap affine_maps[] = {
      mlir::AffineMap::getMultiDimIdentityMap(rank, builder.getContext()),
  };
  llvm::SmallVector<llvm::StringRef> iterators(rank, "parallel");
  auto body = [&](mlir::OpBuilder &builder, mlir::Location loc,
                  mlir::ValueRange values) {
    assert(values.size() == 1);
    builder.create<mlir::linalg::YieldOp>(loc, init_val);
  };
  auto val = builder.create<mlir::linalg::GenericOp>(
      loc, tensor_type, llvm::None, tensor_val, affine_maps, iterators, body);
  return ctx.context.create_var(context, val.getResult(0));
}

py::object generic_impl(py::capsule context, py::handle inputs,
                        py::handle outputs, py::list iterators, py::list maps,
                        py::handle body) {
  auto &ctx = get_py_context(context);
  auto loc = ctx.loc;
  auto &builder = ctx.builder;
  auto &mlirContext = *builder.getContext();

  auto unpack = [&](py::handle obj) -> mlir::Value {
    return ctx.context.unwrap_val(loc, builder, obj);
  };

  auto inputsArgs = getAgrsFromTuple(inputs, unpack);
  auto outputArgs = getAgrsFromTuple(outputs, unpack);
  auto mlirIterators = getIterators(iterators, mlirContext);

  auto bodyTypes = getGenericOpBodyTypes(inputsArgs, outputArgs);
  auto funcTypes = mapTypesToNumbaChecked(ctx.context.types_mod, bodyTypes);
  auto bodyFunc = ctx.context.compile_body(body, funcTypes);

  auto castValues = [&](mlir::ValueRange vals, mlir::TypeRange types) {
    assert(vals.size() == types.size());
    llvm::SmallVector<mlir::Value> ret(vals.size());
    auto doCast = [&](mlir::Value val, mlir::Type type) {
      if (val.getType() == type) {
        return val;
      }
      return builder.create<plier::CastOp>(loc, type, val).getResult();
    };
    for (auto it : llvm::enumerate(vals)) {
      auto index = static_cast<unsigned>(it.index());
      ret[index] = doCast(it.value(), types[index]);
    }
    return ret;
  };

  auto affine_maps = getAffineMaps(maps, mlirContext);
  auto body_builder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::ValueRange args) {
    auto funcType = bodyFunc.getType();
    auto newArgs = castValues(doSignCast(builder, loc, args, bodyTypes),
                              funcType.getInputs());
    auto call = builder.create<mlir::CallOp>(loc, bodyFunc, newArgs);
    auto newResults = doSignCast(
        builder, loc,
        castValues(call.getResults(), genericOpBodyResultTypes(outputArgs)));
    builder.create<mlir::linalg::YieldOp>(loc, newResults);
  };

  auto inputsArgsSignless = doSignCast(builder, loc, inputsArgs);
  auto outputArgsSignless = doSignCast(builder, loc, outputArgs);
  auto retTypes = getTypes(outputArgsSignless);

  auto generic_op = builder.create<mlir::linalg::GenericOp>(
      loc, retTypes, inputsArgsSignless, outputArgsSignless, affine_maps,
      mlirIterators, body_builder);
  auto results =
      doSignCast(builder, loc, generic_op.getResults(), getTypes(outputArgs));
  return ctx.context.wrap_result(context, results);
}

py::object from_elements_impl(py::capsule context, py::handle values,
                              py::handle dtype) {
  auto &ctx = get_py_context(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto type = unwrap_type(dtype);

  llvm::SmallVector<mlir::Value> vals(container_size(values));
  container_iterate(values, [&](auto index, py::handle obj) {
    if (py::isinstance(obj, ctx.context.var)) {
      vals[index] = unwrap_ssa_val(obj);
    } else if (py::isinstance<py::int_>(obj) ||
               py::isinstance<py::float_>(obj)) {
      auto attr = [&]() -> mlir::Attribute {
        if (type.isa<mlir::IntegerType>()) {
          auto signless = makeSignlessType(type);
          return mlir::IntegerAttr::get(signless, obj.cast<int64_t>());
        }
        if (type.isa<mlir::FloatType>()) {
          return mlir::FloatAttr::get(type, obj.cast<double>());
        }
        plier::report_error("Invalid dtype");
      }();
      auto res = builder.create<mlir::ConstantOp>(loc, attr);
      vals[index] = doSignCast(builder, loc, res, type);
    } else {
      plier::report_error("Invalid element type");
    }
  });

  if (vals.empty()) {
    plier::report_error("Invalid from_elemets size");
  }

  auto tensorType =
      mlir::RankedTensorType::get(static_cast<int64_t>(vals.size()), type);
  for (auto &val : vals) {
    val = doSignCast(builder, loc, val);
  }
  auto res = builder.create<mlir::tensor::FromElementsOp>(loc, vals);
  return ctx.context.create_var(context,
                                doSignCast(builder, loc, res, tensorType));
}

py::object extract_impl(py::capsule context, py::handle value,
                        py::handle indices) {
  auto &ctx = get_py_context(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;

  llvm::SmallVector<mlir::Value> ind(container_size(indices));
  container_iterate(indices, [&](auto index, py::handle obj) {
    if (py::isinstance(obj, ctx.context.var)) {
      ind[index] = doSignCast(builder, loc, unwrap_ssa_val(obj));
    } else if (py::isinstance<py::int_>(obj)) {
      ind[index] =
          builder.create<mlir::ConstantIndexOp>(loc, obj.cast<int64_t>());
    } else {
      plier::report_error("Invalid element type");
    }
  });
  auto tensor = ctx.context.unwrap_val(loc, builder, value);
  auto tensorType = tensor.getType().dyn_cast<mlir::RankedTensorType>();
  if (!tensorType) {
    plier::report_error(llvm::Twine("extract: invalid source type ") +
                        to_str(tensor.getType()));
  }
  auto origElement = tensorType.getElementType();
  auto res = builder
                 .create<mlir::tensor::ExtractOp>(
                     loc, doSignCast(builder, loc, tensor), ind)
                 .getResult();
  res = doSignCast(builder, loc, res, origElement);
  return ctx.context.create_var(context, res);
}

py::object reshape_impl(py::capsule context, py::handle src,
                        py::iterable newDims) {
  auto &ctx = get_py_context(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;

  auto unwrapVal = [&](py::handle obj) {
    return ctx.context.unwrap_val(loc, builder, obj);
  };

  auto srcVal = unwrapVal(src);
  auto srcType = srcVal.getType().dyn_cast<mlir::RankedTensorType>();
  if (!srcType) {
    plier::report_error(llvm::Twine("invalid reshape argument: ") +
                        to_str(srcVal.getType()));
  }

  auto newDimsVals = [&]() {
    auto dimType = builder.getIndexType();
    auto dims = unwrapVal(newDims);
    llvm::SmallVector<mlir::Value> ret;
    if (auto tupleType = dims.getType().dyn_cast<mlir::TupleType>()) {
      auto dimsCount = tupleType.size();
      ret.resize(dimsCount);
      for (size_t i = 0; i < dimsCount; ++i) {
        auto ind = builder.create<mlir::ConstantIndexOp>(loc, i);
        auto elemType = tupleType.getType(i);
        auto item = builder.create<plier::GetItemOp>(loc, elemType, dims, ind)
                        .getResult();
        item = doSignCast(builder, loc, item);
        ret[i] = doCast(builder, loc, item, dimType);
      }
    } else {
      dims = doSignCast(builder, loc, dims);
      ret.emplace_back(doCast(builder, loc, dims, dimType));
    }
    return ret;
  }();

  auto shapeTensor =
      builder.create<mlir::tensor::FromElementsOp>(loc, newDimsVals);

  auto elemType = srcType.getElementType();
  auto signlessElemType = makeSignlessType(elemType);

  llvm::SmallVector<int64_t> shape(newDimsVals.size(), -1);
  auto resultType =
      mlir::RankedTensorType::get(shape, elemType, srcType.getEncoding());
  auto resultTypeSignless = mlir::RankedTensorType::get(shape, signlessElemType,
                                                        srcType.getEncoding());
  srcVal = doSignCast(builder, loc, srcVal);
  auto reshaped = builder
                      .create<mlir::tensor::ReshapeOp>(loc, resultTypeSignless,
                                                       srcVal, shapeTensor)
                      .getResult();
  reshaped = doSignCast(builder, loc, reshaped, resultType);

  return ctx.context.create_var(context, reshaped);
}

py::object external_call_impl(py::capsule context, py::str func_name,
                              py::handle inputs, py::handle outputs) {
  auto &ctx = get_py_context(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto unwrapVal = [&](py::handle obj) {
    return ctx.context.unwrap_val(loc, builder, obj);
  };
  auto inputVals = to_values(inputs, unwrapVal);
  auto outputVals = to_values(outputs, unwrapVal);

  inputVals.reserve(inputVals.size() + outputVals.size());
  for (auto val : outputVals) {
    if (auto tensorType = val.getType().dyn_cast<mlir::TensorType>()) {
      auto memrefType = mlir::MemRefType::get(tensorType.getShape(),
                                              tensorType.getElementType());
      auto memref =
          builder.create<mlir::memref::BufferCastOp>(loc, memrefType, val);
      inputVals.emplace_back(memref);
    } else {
      inputVals.emplace_back(val);
    }
  }

  auto func = [&]() {
    auto argTypes = getTypes(inputVals);
    auto funcType =
        mlir::FunctionType::get(builder.getContext(), argTypes, llvm::None);
    auto name = static_cast<std::string>(func_name);
    assert(!name.empty());
    auto mod =
        builder.getBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>();
    assert(mod);
    auto f = mod.lookupSymbol<mlir::FuncOp>(name);
    if (f) {
      if (f.getType() != funcType) {
        plier::report_error(llvm::Twine("linalg_builder::external_call: "
                                        "invalid function redefinition: ") +
                            name);
      }
    } else {
      f = plier::add_function(builder, mod, name, funcType);
      f->setAttr("llvm.emit_c_interface",
                 mlir::UnitAttr::get(builder.getContext()));
    }
    return f;
  }();

  auto res = builder.create<mlir::CallOp>(loc, func, inputVals).getResults();

  llvm::SmallVector<mlir::Value> results;
  results.reserve(outputVals.size() + res.size());

  for (auto it : llvm::enumerate(
           llvm::makeArrayRef(inputVals).take_back(outputVals.size()))) {
    auto val = it.value();
    if (outputVals[it.index()].getType().isa<mlir::TensorType>()) {
      val = builder.create<mlir::memref::TensorLoadOp>(loc, val);
    }
    results.emplace_back(val);
  }

  results.append(res.begin(), res.end());

  if (results.empty()) {
    return py::none();
  }

  py::tuple ret(results.size());
  for (auto it : llvm::enumerate(results)) {
    ret[it.index()] = ctx.context.create_var(context, it.value());
  }

  return std::move(ret);
}

py::object insert_impl(py::capsule context, py::handle src, py::handle dst,
                       py::handle offsets, py::handle sizes,
                       py::handle strides) {
  auto &ctx = get_py_context(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto unwrapVal = [&](py::handle obj) {
    return ctx.context.unwrap_val(loc, builder, obj);
  };
  auto indexType = builder.getIndexType();
  auto unwrapList = [&](py::handle obj) {
    auto len = py::len(obj);
    llvm::SmallVector<mlir::Value> res(len);
    for (auto it : llvm::enumerate(obj)) {
      res[it.index()] = doCast(builder, loc, unwrapVal(it.value()), indexType);
    }
    return res;
  };
  auto srcTensor = unwrapVal(src);
  auto dstTensor = unwrapVal(dst);
  auto signlessSrc = doSignCast(builder, loc, srcTensor);
  auto signlessDst = doSignCast(builder, loc, dstTensor);
  auto offsetsVec = unwrapList(offsets);
  auto sizesVec = unwrapList(sizes);
  auto stridesVec = unwrapList(strides);
  auto res =
      builder
          .create<mlir::tensor::InsertSliceOp>(loc, signlessSrc, signlessDst,
                                               offsetsVec, sizesVec, stridesVec)
          .getResult();
  res = doSignCast(builder, loc, res, dstTensor.getType());
  return ctx.context.create_var(context, res);
}

py::object inline_func_impl(py::capsule context, py::handle func,
                            py::tuple args) {
  auto &ctx = get_py_context(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;

  auto argsValues = [&]() {
    auto unwrapVal = [&](py::handle obj) {
      return ctx.context.unwrap_val(loc, builder, obj);
    };
    llvm::SmallVector<mlir::Value> ret(args.size());
    for (auto it : llvm::enumerate(args)) {
      ret[it.index()] = unwrapVal(it.value());
    }
    return ret;
  }();
  auto funcTypes =
      mapTypesToNumbaChecked(ctx.context.types_mod, getTypes(argsValues));
  auto bodyFunc = ctx.context.compile_body(func, funcTypes);
  auto funcType = bodyFunc.getType();
  auto funcArgsTypes = funcType.getInputs();
  if (funcArgsTypes.size() != argsValues.size()) {
    plier::report_error(
        llvm::Twine("Invalid function arguments count, expected ") +
        llvm::Twine(argsValues.size()) + ", got" +
        llvm::Twine(funcArgsTypes.size()));
  }
  if (funcType.getNumResults() != 1) {
    plier::report_error(llvm::Twine("Invalid number of return values: ") +
                        llvm::Twine(funcType.getNumResults()));
  }

  auto resValue =
      builder.create<mlir::CallOp>(loc, bodyFunc, argsValues).getResult(0);
  auto resType = resValue.getType();
  if (auto convertedType = ctx.typeConverter.convertType(resType)) {
    resValue = doCast(builder, loc, resValue, convertedType);
  }
  return ctx.context.create_var(context, resValue);
}

py::object cast_impl(py::capsule context, py::handle src, py::handle dtype) {
  auto &ctx = get_py_context(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto unwrapVal = [&](py::handle obj) {
    return ctx.context.unwrap_val(loc, builder, obj);
  };
  auto val = unwrapVal(src);
  auto type = unwrap_type(dtype);
  auto ret = builder.createOrFold<plier::CastOp>(loc, type, val);
  return ctx.context.create_var(context, ret);
}

py::object undef_impl(py::capsule context, py::handle dtype) {
  auto &ctx = get_py_context(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto type = unwrap_type(dtype);
  auto ret = builder.createOrFold<plier::UndefOp>(loc, type);
  return ctx.context.create_var(context, ret);
}

void setup_py_builder(py::handle builder, mlir::OpBuilder &b,
                      llvm::function_ref<py::object(mlir::Type)> create_type) {
  py::setattr(builder, "_broadcast", py::cpp_function(&broadcast_impl));
  py::setattr(builder, "_init_tensor", py::cpp_function(&init_tensor_impl));
  py::setattr(builder, "_fill_tensor", py::cpp_function(&fill_tensor_impl));
  py::setattr(builder, "_generic", py::cpp_function(&generic_impl));
  py::setattr(builder, "_from_elements", py::cpp_function(&from_elements_impl));
  py::setattr(builder, "_extract", py::cpp_function(&extract_impl));
  py::setattr(builder, "_reshape", py::cpp_function(&reshape_impl));
  py::setattr(builder, "_external_call", py::cpp_function(&external_call_impl));
  py::setattr(builder, "_insert", py::cpp_function(&insert_impl));
  py::setattr(builder, "_inline_func", py::cpp_function(&inline_func_impl));
  py::setattr(builder, "_cast", py::cpp_function(&cast_impl));
  py::setattr(builder, "_undef", py::cpp_function(&undef_impl));

  auto addType = [&](const char *name, mlir::Type type) {
    py::setattr(builder, name, create_type(type));
  };

  addType("int8", b.getIntegerType(8, true));
  addType("uint8", b.getIntegerType(8, false));
  addType("int16", b.getIntegerType(16, true));
  addType("uint16", b.getIntegerType(16, false));
  addType("int32", b.getIntegerType(32, true));
  addType("uint32", b.getIntegerType(32, false));
  addType("int64", b.getIntegerType(64, true));
  addType("uint64", b.getIntegerType(64, false));

  addType("index", b.getIndexType());

  addType("float16", b.getF16Type());
  addType("float32", b.getF32Type());
  addType("float64", b.getF64Type());
}

py::object shape_impl(py::capsule context, py::capsule ssa_val) {
  auto &ctx = get_py_context(context);
  auto value = unwrap_mlir<mlir::Value>(ssa_val);
  if (value.getType().isa<mlir::RankedTensorType>()) {
    auto &builder = ctx.builder;
    auto loc = ctx.loc;
    auto mlir_type = value.getType().cast<mlir::RankedTensorType>();
    auto shape = mlir_type.getShape();
    llvm::SmallVector<mlir::Value> shape_vals(shape.size());
    for (auto it : llvm::enumerate(shape)) {
      auto i = it.index();
      mlir::Value mlir_dim = builder.create<mlir::tensor::DimOp>(loc, value, i);
      shape_vals[i] = mlir_dim;
    }
    llvm::SmallVector<mlir::Type> shape_types(shape.size(),
                                              builder.getIndexType());
    auto shape_type = mlir::TupleType::get(builder.getContext(), shape_types);
    auto shape_var =
        builder.create<plier::BuildTupleOp>(loc, shape_type, shape_vals);
    return ctx.context.create_var(context, shape_var.getResult());
  }
  return py::list();
}

py::object dtype_impl(py::capsule context, py::capsule ssa_val) {
  auto &ctx = get_py_context(context);
  auto value = unwrap_mlir<mlir::Value>(ssa_val);
  auto type = value.getType();
  if (auto tensorType = type.dyn_cast<mlir::RankedTensorType>()) {
    type = tensorType.getElementType();
  }
  return ctx.context.create_type(type);
}

py::object len_impl(py::capsule /*context*/, py::capsule ssa_val) {
  auto value = unwrap_mlir<mlir::Value>(ssa_val);
  auto type = value.getType();
  if (auto tuple_type = type.dyn_cast<mlir::TupleType>()) {
    return py::int_(tuple_type.size());
  }
  return py::int_(0);
}

py::object getitem_impl(py::capsule context, py::capsule ssaVal,
                        py::handle index) {
  auto &ctx = get_py_context(context);
  auto value = unwrap_mlir<mlir::Value>(ssaVal);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto type = value.getType();
  if (auto tensor = type.dyn_cast<mlir::TensorType>()) {
    auto indexVal = ctx.context.unwrap_val(loc, builder, index);
    auto elemType = tensor.getElementType();
    auto res = builder.create<plier::GetItemOp>(loc, elemType, value, indexVal);
    return ctx.context.create_var(context, res.getResult());
  }

  auto indexVal = index.cast<int64_t>();
  if (auto tuple_type = type.dyn_cast<mlir::TupleType>()) {
    auto maxIndex = static_cast<int64_t>(tuple_type.size());
    if (indexVal < 0 || indexVal >= maxIndex) {
      throw py::index_error(("Invalid getitem index: " + llvm::Twine(indexVal) +
                             ", expected [0:" + llvm::Twine(maxIndex) + ")")
                                .str());
    }
    if (auto parent_op = value.getDefiningOp<plier::BuildTupleOp>()) {
      return ctx.context.create_var(
          context, parent_op.getOperand(static_cast<unsigned>(indexVal)));
    }
    auto elemType = tuple_type.getType(static_cast<size_t>(indexVal));
    auto ind = builder.create<mlir::ConstantIndexOp>(loc, indexVal);
    auto item = builder.create<plier::GetItemOp>(loc, elemType, value, ind);
    return ctx.context.create_var(context, item.getResult());
  } else {
    throw py::index_error("Invalid getitem");
  }
}

template <typename Op>
mlir::Value binop_func(mlir::Location loc, mlir::OpBuilder &builder,
                       mlir::Value lhs, mlir::Value rhs) {
  return builder.create<Op>(loc, lhs, rhs);
}

mlir::Value binop_func_idiv(mlir::Location loc, mlir::OpBuilder &builder,
                            mlir::Value lhs, mlir::Value rhs) {
  auto lhs_var = doCast(builder, loc, lhs, builder.getF64Type());
  auto rhs_var = doCast(builder, loc, rhs, builder.getF64Type());
  return builder.create<mlir::DivFOp>(loc, lhs_var, rhs_var);
}

py::object binop_impl(py::capsule context, py::capsule ssa_val, py::handle rhs,
                      py::str op) {
  auto &ctx = get_py_context(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto lhs = unwrap_mlir<mlir::Value>(ssa_val);

  auto type = lhs.getType();
  if (!type.isa<mlir::IntegerType, mlir::IndexType, mlir::FloatType,
                mlir::ShapedType>()) {
    plier::report_error("Invalid binop arg type");
  }

  auto is_float = [&]() -> bool {
    if (auto shaped_type = type.dyn_cast<mlir::ShapedType>()) {
      return shaped_type.getElementType().isa<mlir::FloatType>();
    }
    return type.isa<mlir::FloatType>();
  }();

  using binop_func_t =
      mlir::Value (*)(mlir::Location loc, mlir::OpBuilder & builder,
                      mlir::Value lhs, mlir::Value rhs);
  const std::tuple<llvm::StringRef, binop_func_t, binop_func_t> funcs[] = {
      {"+", &binop_func<mlir::AddIOp>, &binop_func<mlir::AddFOp>},
      {"*", &binop_func<mlir::MulIOp>, &binop_func<mlir::MulFOp>},
      {"/", &binop_func_idiv, &binop_func<mlir::DivFOp>},
  };

  auto op_name = static_cast<std::string>(op);
  for (auto f : funcs) {
    auto name = std::get<0>(f);
    auto func = (is_float ? std::get<2>(f) : std::get<1>(f));
    if (name == op_name) {
      auto rhs_var =
          doCast(builder, loc, ctx.context.unwrap_val(loc, builder, rhs), type);
      auto res = func(loc, builder, lhs, rhs_var);
      return ctx.context.create_var(context, res);
    }
  }
  plier::report_error("Unhandled binop type");
}

void setup_py_var(pybind11::handle var) {
  py::setattr(var, "_shape", py::cpp_function(&shape_impl));
  py::setattr(var, "_dtype", py::cpp_function(&dtype_impl));
  py::setattr(var, "_len", py::cpp_function(&len_impl));
  py::setattr(var, "_getitem", py::cpp_function(&getitem_impl));
  py::setattr(var, "_binop", py::cpp_function(&binop_impl));
}

PyLinalgResolver::Values unpack_results(PyBuilderContext &ctx,
                                        py::handle object) {
  PyLinalgResolver::Values ret;
  if (object.is_none()) {
    return ret;
  }
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto unwrapVal = [&](py::handle obj) {
    return ctx.context.unwrap_val(loc, builder, obj);
  };
  if (py::isinstance<py::tuple>(object)) {
    auto tuple = object.cast<py::tuple>();
    llvm::SmallVector<mlir::Value> vals(tuple.size());
    for (auto it : llvm::enumerate(tuple)) {
      vals[it.index()] = unwrapVal(it.value());
    }
    ret.emplace_back(builder.create<plier::BuildTupleOp>(loc, vals));
  } else {
    ret.emplace_back(unwrapVal(object));
  }
  return ret;
}
} // namespace

PyLinalgResolver::PyLinalgResolver() : context(std::make_unique<Context>()) {
  auto builder_mod = py::module::import("numba_dpcomp.mlir.linalg_builder");
  context->var = builder_mod.attr("Var");
  context->type = builder_mod.attr("Type");
  context->builder = builder_mod.attr("Builder");
  context->inspect = py::module::import("inspect");
  context->types_mod = py::module::import("numba.core.types");
  context->compile_func = builder_mod.attr("compile_func");
  context->lookup_func = builder_mod.attr("lookup_func");
}

PyLinalgResolver::~PyLinalgResolver() {}

llvm::Optional<PyLinalgResolver::Values>
PyLinalgResolver::rewrite_func(llvm::Twine name, mlir::Location loc,
                               mlir::OpBuilder &builder, mlir::ValueRange args,
                               KWArgs kwargs) {
  return rewrite((name + "()").str(), loc, builder, args, kwargs);
}

llvm::Optional<PyLinalgResolver::Values>
PyLinalgResolver::rewrite_attr(llvm::Twine name, mlir::Location loc,
                               mlir::OpBuilder &builder, mlir::Value arg) {
  return rewrite(name.str(), loc, builder, arg, {});
}

llvm::Optional<PyLinalgResolver::Values>
PyLinalgResolver::rewrite(llvm::StringRef name, mlir::Location loc,
                          mlir::OpBuilder &builder, mlir::ValueRange args,
                          KWArgs kwargs) {
  assert(!name.empty());
  if (!is_compatible_types(args) ||
      !is_compatible_types(llvm::make_second_range(kwargs))) {
    return {};
  }

  auto builder_func = context->lookup_func(py::str(name.data(), name.size()));
  if (builder_func.is_none()) {
    return {};
  }

  PyBuilderContext py_builder_context{loc, builder, {}, *context};
  auto &mlirContext = *builder.getContext();
  populate_std_type_converter(mlirContext, py_builder_context.typeConverter);
  populate_array_type_converter(mlirContext, py_builder_context.typeConverter);
  auto py_context = py::capsule(&py_builder_context);
  auto py_args = get_args(
      context->inspect, builder_func,
      [&](auto val) { return context->create_var(py_context, val); }, args,
      kwargs);
  if (py_args.is_none()) {
    return {};
  }
  auto py_builder = context->builder(py_context);
  setup_py_builder(py_builder, builder,
                   [&](auto type) { return context->create_type(type); });

  auto result = builder_func(py_builder, *py_args);
  if (result.is_none()) {
    return {};
  }
  return unpack_results(py_builder_context, result);
}
