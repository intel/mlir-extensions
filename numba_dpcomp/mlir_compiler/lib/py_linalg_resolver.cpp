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

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Parser.h>

#include "plier/dialect.hpp"
#include "plier/transforms/const_utils.hpp"
#include "plier/transforms/func_utils.hpp"
#include "plier/utils.hpp"
#include "py_map_types.hpp"

namespace py = pybind11;

struct PyBuilderContext {
  mlir::Location loc;
  mlir::OpBuilder &builder;
  PyLinalgResolver::Context &context;
};

namespace {
std::string toStr(mlir::Value val) {
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  ss << val;
  ss.flush();
  return ret;
}

std::string toStr(mlir::Type type) {
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  ss << type;
  ss.flush();
  return ret;
}

std::string toStr(mlir::TypeRange typesRange) {
  std::string str;
  llvm::raw_string_ostream ss(str);
  for (auto type : typesRange)
    ss << type << " ";

  ss.flush();
  return str;
}

std::string toStr(py::handle obj) { return py::str(obj).cast<std::string>(); }

py::object mapTypesToNumbaChecked(py::handle typesMod,
                                  mlir::TypeRange typesRange) {
  auto funcTypes = map_types_to_numba(typesMod, typesRange);
  if (funcTypes.is_none())
    plier::reportError(llvm::Twine("map_types_to_numba failed: ") +
                       toStr(typesRange));
  return funcTypes;
}

mlir::Type makeSignlessType(mlir::Type type) {
  if (auto tensor = type.dyn_cast<mlir::RankedTensorType>()) {
    auto origElemType = tensor.getElementType();
    auto signlessElemType = makeSignlessType(origElemType);
    if (origElemType != signlessElemType)
      return mlir::RankedTensorType::get(tensor.getShape(), signlessElemType,
                                         tensor.getEncoding());
  } else if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
    if (!intType.isSignless())
      return mlir::IntegerType::get(intType.getContext(), intType.getWidth());
  }
  return type;
}

mlir::Value doSignCast(mlir::OpBuilder &builder, mlir::Location &loc,
                       mlir::Value val) {
  auto origType = val.getType();
  auto signlessType = makeSignlessType(origType);
  if (signlessType != origType) {
    val = builder.createOrFold<plier::SignCastOp>(loc, signlessType, val);
  }
  return val;
}

mlir::Value doSignCast(mlir::OpBuilder &builder, mlir::Location &loc,
                       mlir::Value val, mlir::Type dstType) {
  auto origType = val.getType();
  if (dstType != origType) {
    val = builder.createOrFold<plier::SignCastOp>(loc, dstType, val);
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

bool isCompatibleType(mlir::Type type) {
  if (auto tuple_type = type.dyn_cast<mlir::TupleType>())
    return llvm::all_of(tuple_type, &isCompatibleType);

  return type.isa<mlir::IntegerType, mlir::IndexType, mlir::FloatType,
                  mlir::RankedTensorType, mlir::NoneType, plier::LiteralType,
                  plier::TypeVar>();
}

template <typename R> bool isCompatibleTypes(R &&vals) {
  return llvm::all_of(vals,
                      [](auto val) { return isCompatibleType(val.getType()); });
}

template <typename T> py::capsule wrapMlir(T val) {
  return py::capsule(val.getAsOpaquePointer());
}

template <typename T> T unwrapMlir(py::capsule obj) {
  return T::getFromOpaquePointer(static_cast<const void *>(obj));
}

auto unwrapSsaVal(py::handle obj) {
  return unwrapMlir<mlir::Value>(obj.attr("_ssa_val").cast<py::capsule>());
}

auto unwrapType(py::handle obj) {
  if (py::hasattr(obj, "_ssa_val")) {
    auto val = unwrapSsaVal(obj);
    if (auto type = val.getType().dyn_cast<plier::TypeVar>())
      return type.getType();
  } else if (py::hasattr(obj, "_mlir_type")) {
    return unwrapMlir<mlir::Type>(obj.attr("_mlir_type").cast<py::capsule>());
  }
  plier::reportError(llvm::Twine("Invalid type object: ") +
                     toStr(obj.get_type()));
}

size_t containerSize(py::handle obj) {
  if (py::isinstance<py::tuple>(obj))
    return obj.cast<py::tuple>().size();

  if (py::isinstance<py::list>(obj))
    return obj.cast<py::list>().size();

  return 1;
}

template <typename F> void containerIterate(py::handle obj, F &&func) {
  auto impl = [&](auto cont) {
    for (auto it : llvm::enumerate(cont))
      func(it.index(), it.value());
  };
  if (py::isinstance<py::tuple>(obj)) {
    impl(obj.cast<py::tuple>());
  } else if (py::isinstance<py::list>(obj)) {
    impl(obj.cast<py::list>());
  } else {
    func(std::size_t(0), obj);
  }
}

template <typename RetType = llvm::SmallVector<mlir::Value>,
          typename UnwrapFunc>
auto toValues(py::handle obj, UnwrapFunc &&unwrapFunc) {
  RetType ret(containerSize(obj));
  containerIterate(
      obj, [&](auto index, py::handle elem) { ret[index] = unwrapFunc(elem); });
  return ret;
}

llvm::Optional<py::object> getPyLiteral(mlir::Attribute attr) {
  assert(attr);
  if (auto intAttr = attr.dyn_cast<mlir::IntegerAttr>()) {
    if (auto intType = attr.getType().dyn_cast<mlir::IntegerType>()) {
      // Ignore index type
      if (intType.getWidth() == 1)
        return py::bool_(intAttr.getInt() != 0);
    }
    return py::int_(plier::getIntAttrValue(intAttr));
  }
  if (auto floatAttr = attr.dyn_cast<mlir::FloatAttr>())
    return py::float_(floatAttr.getValueAsDouble());

  return {};
}

llvm::Optional<py::object> makePyLiteral(mlir::Value val) {
  assert(val);
  if (auto literal = val.getType().dyn_cast<plier::LiteralType>())
    return getPyLiteral(literal.getValue());

  if (auto cast = val.getDefiningOp<plier::SignCastOp>())
    val = cast.value();

  if (auto attr = plier::getConstVal<mlir::Attribute>(val))
    return getPyLiteral(attr);

  return {};
}

static mlir::Value doCast(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value val, mlir::Type type) {
  if (val.getType() != type)
    return builder.createOrFold<plier::CastOp>(loc, type, val);

  return val;
}

bool cmpCapsule(py::capsule a1, py::capsule a2) {
  return static_cast<void *>(a1) == static_cast<void *>(a2);
}

py::object printTypeCapsule(py::capsule t) {
  auto type = unwrapMlir<mlir::Type>(t);
  return py::str("Type: \"" + toStr(type) + "\"");
}

void setupPyVar(py::handle var);
} // namespace

struct PyLinalgResolver::Context {
  py::object var;
  py::object type;
  py::object builder;
  py::object inspect;
  py::object types_mod;
  py::object compile_func;
  py::object lookup_func;

  py::object createVar(py::capsule context, mlir::Value value) {
    assert(value);
    auto type = value.getType();
    if (type.isa<mlir::NoneType>())
      return py::none();

    if (auto typevar = type.dyn_cast<plier::TypeVar>())
      return createType(typevar.getType());

    if (auto literal = makePyLiteral(value))
      return *literal;

    auto ret = var(context, wrapMlir(value));
    setupPyVar(ret);
    return ret;
  }

  py::object createType(mlir::Type t) {
    return type(wrapMlir(t), py::cpp_function(&cmpCapsule),
                py::cpp_function(&printTypeCapsule));
  }

  mlir::FuncOp compileBody(py::handle body, py::list arg_types) {
    auto func = compile_func(body, arg_types).cast<py::capsule>();
    auto mlirFunc =
        mlir::cast<mlir::FuncOp>(static_cast<mlir::Operation *>(func));
    mlirFunc.setPrivate();
    mlirFunc->setAttr(plier::attributes::getForceInlineName(),
                      mlir::UnitAttr::get(mlirFunc->getContext()));
    return mlirFunc;
  }

  py::object wrapResult(py::capsule context, mlir::ValueRange values) {
    if (values.empty())
      return py::none();

    if (values.size() == 1)
      return createVar(context, values.front());

    py::tuple ret(values.size());
    for (auto it : llvm::enumerate(values))
      ret[it.index()] = createVar(context, it.value());

    return std::move(ret);
  }

  mlir::Value unwrapVal(mlir::Location loc, mlir::OpBuilder &builder,
                        py::handle obj) {
    if (py::isinstance(obj, var))
      return unwrapSsaVal(obj);

    if (py::isinstance(obj, type)) {
      auto type = plier::TypeVar::get(unwrapType(obj));
      return builder.create<plier::UndefOp>(loc, type);
    }

    if (obj.is_none()) {
      auto type = mlir::NoneType::get(builder.getContext());
      return builder.create<plier::UndefOp>(loc, type);
    }

    if (py::isinstance<py::iterable>(obj)) {
      llvm::SmallVector<mlir::Value> elems(py::len(obj));
      for (auto it : llvm::enumerate(obj))
        elems[it.index()] = unwrapVal(loc, builder, it.value());

      mlir::ValueRange vr(elems);
      auto resType = mlir::TupleType::get(builder.getContext(), vr.getTypes());
      return builder.create<plier::BuildTupleOp>(loc, resType, elems);
    }

    if (py::isinstance<py::bool_>(obj)) {
      auto type = builder.getI1Type();
      auto attr = builder.getIntegerAttr(type, (obj.cast<bool>() ? 1 : 0));
      return builder.create<mlir::arith::ConstantOp>(loc, attr);
    }

    if (py::isinstance<py::int_>(obj)) {
      auto attr = builder.getI64IntegerAttr(obj.cast<int64_t>());
      auto res = builder.create<mlir::arith::ConstantOp>(loc, attr);
      auto intType = builder.getIntegerType(64, true);
      return builder.create<plier::SignCastOp>(loc, intType, res);
    }

    if (py::isinstance<py::float_>(obj)) {
      auto attr = builder.getF64FloatAttr(obj.cast<double>());
      return builder.create<mlir::arith::ConstantOp>(loc, attr);
    }

    plier::reportError(llvm::Twine("Invalid element type: ") +
                       toStr(obj.get_type()));
  }
};

namespace {
py::object
getArgs(py::handle inspect, py::handle func,
        llvm::function_ref<py::object(mlir::Value)> createVar,
        mlir::ValueRange args,
        llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs) {
  auto sigFunc = inspect.attr("signature");
  auto sig = sigFunc(func);
  auto params = sig.attr("parameters");
  auto paramsList = py::list(params);
  auto paramsListSize = static_cast<unsigned>(paramsList.size());
  paramsList =
      paramsList[py::slice(1, paramsListSize, 1)]; // skip builder param
  auto paramAttr = inspect.attr("Parameter");
  auto empty = paramAttr.attr("empty");
  auto vararg = paramAttr.attr("VAR_POSITIONAL");

  llvm::SmallVector<py::object> retArgs;
  retArgs.reserve(paramsListSize);
  for (auto paramName : paramsList) {
    auto param = params[paramName];
    if (param.attr("kind").is(vararg)) {
      while (!args.empty()) {
        retArgs.emplace_back(createVar(args.front()));
        args = args.drop_front();
      }
      continue;
    } else {
      if (!args.empty()) {
        retArgs.emplace_back(createVar(args.front()));
        args = args.drop_front();
        continue;
      }
      if (!kwargs.empty()) {
        auto name = paramName.cast<std::string>();
        auto val = [&]() -> mlir::Value {
          for (auto kwarg : kwargs) {
            if (kwarg.first == name)
              return kwarg.second;
          }
          return {};
        }();
        if (val) {
          retArgs.emplace_back(createVar(val));
          continue;
        }
      }
    }
    auto defVal = param.attr("default");
    if (!defVal.is(empty)) {
      retArgs.emplace_back(defVal);
    } else {
      return py::none();
    }
  }

  if (!args.empty())
    return py::none();

  py::tuple ret(retArgs.size());
  for (auto it : llvm::enumerate(retArgs))
    ret[it.index()] = std::move(it.value());

  return std::move(ret);
}

PyBuilderContext &getPyContext(py::capsule &ctx) {
  return *static_cast<PyBuilderContext *>(ctx);
}

auto getAgrsFromTuple(py::handle args,
                      llvm::function_ref<mlir::Value(py::handle)> unpack) {
  llvm::SmallVector<mlir::Value> ret;
  if (args.is_none())
    return ret;

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
  for (auto it : llvm::enumerate(iterators))
    ret[it.index()] =
        mlir::StringAttr::get(&ctx, it.value().cast<std::string>()).getValue();

  return ret;
}

mlir::AffineMapAttr getAffineMapAttr(py::handle obj, mlir::MLIRContext &ctx) {
  auto str = (llvm::Twine("affine_map<") + obj.cast<std::string>() + ">").str();
  return mlir::parseAttribute(str, &ctx).cast<mlir::AffineMapAttr>();
}

auto getAffineMaps(py::list maps, mlir::MLIRContext &ctx) {
  llvm::SmallVector<mlir::AffineMap> ret(maps.size());
  for (auto it : llvm::enumerate(maps))
    ret[it.index()] = getAffineMapAttr(it.value(), ctx).getValue();

  return ret;
}

auto getGenericOpBodyTypes(mlir::ValueRange inputs, mlir::ValueRange outputs) {
  llvm::SmallVector<mlir::Type> ret;
  ret.reserve(inputs.size() + outputs.size());
  for (auto r : {inputs, outputs}) {
    for (auto type : r.getTypes()) {
      auto elemType = [&]() {
        if (auto tensor = type.dyn_cast<mlir::RankedTensorType>())
          return tensor.getElementType();

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
    auto elemType = type.cast<mlir::RankedTensorType>().getElementType();
    ret.emplace_back(elemType);
  }
  return ret;
}

bool isInt(mlir::Type type) {
  return type.isa<mlir::IntegerType, mlir::IndexType>();
}

bool isFloat(mlir::Type type) { return type.isa<mlir::FloatType>(); }

unsigned getIntBitWidth(mlir::Type type) {
  if (type.isa<mlir::IntegerType>())
    return type.cast<mlir::IntegerType>().getWidth();

  if (type.isa<mlir::IndexType>())
    return 64; // TODO

  llvm_unreachable("No an integer type");
}

unsigned getFloatBitWidth(mlir::Type type) {
  return type.cast<mlir::FloatType>().getWidth();
}

mlir::Type broadcast_type(mlir::Type type1, mlir::Type type2) {
  if (type1 == type2) {
    return type1;
  }
  // TODO
  if (isInt(type1) && isInt(type2)) {
    bool isSigned = type1.isSignedInteger() || type2.isSignedInteger();
    auto signess =
        (isSigned ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned);
    auto width = std::max(getIntBitWidth(type1), getIntBitWidth(type2));
    return mlir::IntegerType::get(type1.getContext(), width, signess);
  }
  if (isFloat(type1) && isFloat(type2))
    return (getFloatBitWidth(type1) > getFloatBitWidth(type2) ? type1 : type2);

  if (isFloat(type1) && isInt(type2))
    return type1;

  if (isInt(type1) && isFloat(type2))
    return type2;

  llvm_unreachable("Unable to broadcast type");
}

mlir::Value broadcastDim(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value val1, mlir::Value val2) {
  assert(val1.getType().isa<mlir::IndexType>());
  assert(val2.getType().isa<mlir::IndexType>());
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto cond = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, val1, one);
  return builder.create<mlir::SelectOp>(loc, cond, val2, val1);
}

mlir::Value expandDim(mlir::OpBuilder &builder, mlir::Location loc,
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
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  mlir::Value cond = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, one, dimVal);
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

mlir::Value expandDims(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value val, unsigned numDims,
                       mlir::ValueRange targetShape) {
  assert(numDims <= targetShape.size());
  if (numDims < targetShape.size()) {
    targetShape = targetShape.drop_front(targetShape.size() - numDims);
  }
  mlir::Value current = val;
  for (unsigned i = 0; i < numDims; ++i)
    current = expandDim(builder, loc, val, current, i, targetShape);

  current = builder.create<plier::EnforceShapeOp>(loc, current, targetShape);
  return current;
}

py::object broadcastImpl(py::capsule context, py::tuple args) {
  if (1 == args.size())
    return args[0];

  auto &ctx = getPyContext(context);
  auto loc = ctx.loc;
  auto &builder = ctx.builder;
  llvm::SmallVector<mlir::Value> mlirArgs(args.size());
  for (auto it : llvm::enumerate(args)) {
    auto val = ctx.context.unwrapVal(loc, builder, it.value());
    mlirArgs[it.index()] = val;
  }
  using shape_t = llvm::SmallVector<mlir::Value>;
  auto getShape =
      [&](mlir::Value val) -> llvm::Optional<std::pair<shape_t, mlir::Type>> {
    auto type = val.getType();
    if (auto shaped = type.dyn_cast<mlir::ShapedType>()) {
      if (!shaped.hasRank())
        return {};

      shape_t ret(static_cast<size_t>(shaped.getRank()));
      for (auto it : llvm::enumerate(ret)) {
        auto dim = builder.create<mlir::tensor::DimOp>(loc, val, it.index());
        ret[it.index()] = dim;
      }
      return std::make_pair(ret, shaped.getElementType());
    }
    if (type.isa<mlir::IntegerType, mlir::IndexType, mlir::FloatType>())
      return std::make_pair(shape_t{}, type);

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
    if (!shapeAndType)
      return py::none();

    resType = broadcast_type(resType, shapeAndType->second);
    auto newShapeVals = shapeAndType->first;
    for (auto it :
         llvm::zip(llvm::reverse(shapeVals), llvm::reverse(newShapeVals))) {
      auto &oldVal = std::get<0>(it);
      auto newVal = std::get<1>(it);
      oldVal = broadcastDim(builder, loc, oldVal, newVal);
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
      if (val.getType() != resType)
        val = builder.create<plier::CastOp>(loc, resType, val);

      ret[it.index()] = ctx.context.createVar(context, val);
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
      val = expandDims(builder, loc, val,
                       static_cast<unsigned>(srcType.getRank()), shapeVals);
    }
    if (val.getType() != signlessTensorType) {
      auto type = val.getType();
      if (auto srcType = type.dyn_cast<mlir::ShapedType>()) {
        assert(srcType.hasRank());
        auto srcNumDims = static_cast<unsigned>(srcType.getRank());
        auto numDims = static_cast<unsigned>(signlessTensorType.getRank());
        auto init = builder
                        .create<mlir::linalg::InitTensorOp>(
                            loc, shapeVals, signlessTensorType.getElementType())
                        .getResult();
        mlir::AffineMap maps[] = {
            mlir::AffineMap::getMinorIdentityMap(numDims, srcNumDims,
                                                 builder.getContext()),
            //                    mlir::AffineMap::getMultiDimIdentityMap(num_dims,
            //                    builder.getContext()).getMajorSubMap(src_num_dims),
            mlir::AffineMap::getMultiDimIdentityMap(numDims,
                                                    builder.getContext()),
        };
        llvm::SmallVector<llvm::StringRef> iterators(numDims, "parallel");
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
    ret[it.index()] = ctx.context.createVar(context, val);
  }
  return std::move(ret);
}

py::object initTensorImpl(py::capsule context, py::handle shape,
                          py::handle dtype, py::handle initVal) {
  auto &ctx = getPyContext(context);
  auto loc = ctx.loc;
  auto &builder = ctx.builder;
  auto elemType = unwrapType(dtype);
  auto signlessElemType = makeSignlessType(elemType);
  mlir::Value init;
  auto indexType = builder.getIndexType();
  auto count = py::len(shape);
  llvm::SmallVector<mlir::Value> shapeVal(count);
  llvm::SmallVector<int64_t> staticShape(count, -1);
  for (size_t i = 0; i < count; ++i) {
    auto elem = shape[py::int_(i)];
    if (py::isinstance<py::int_>(elem))
      staticShape[i] = elem.cast<int64_t>();

    auto elemVal = ctx.context.unwrapVal(loc, builder, elem);
    elemVal = doSignCast(builder, loc, elemVal);
    shapeVal[i] = doCast(builder, loc, elemVal, indexType);
  }

  if (initVal.is_none()) {
    init = builder.create<mlir::linalg::InitTensorOp>(loc, shapeVal,
                                                      signlessElemType);
  } else {
    auto val =
        doCast(builder, loc, ctx.context.unwrapVal(loc, builder, initVal),
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
  return ctx.context.createVar(context, init);
}

py::object fillTensorImpl(py::capsule context, py::handle tensor,
                          py::handle value) {
  auto &ctx = getPyContext(context);
  auto loc = ctx.loc;
  auto &builder = ctx.builder;
  auto tensorVal = ctx.context.unwrapVal(loc, builder, tensor);
  auto tensorType = tensorVal.getType().cast<mlir::ShapedType>();
  auto initVal = ctx.context.unwrapVal(loc, builder, value);
  if (initVal.getType() != tensorType.getElementType())
    initVal = builder.create<plier::CastOp>(loc, tensorType.getElementType(),
                                            initVal);

  //    auto val = builder.create<mlir::linalg::FillOp>(loc, tensor_type,
  //    tensor_val, init_val);
  auto rank = static_cast<unsigned>(tensorType.getRank());
  mlir::AffineMap affine_maps[] = {
      mlir::AffineMap::getMultiDimIdentityMap(rank, builder.getContext()),
  };
  llvm::SmallVector<llvm::StringRef> iterators(rank, "parallel");
  auto body = [&](mlir::OpBuilder &builder, mlir::Location loc,
                  mlir::ValueRange values) {
    assert(values.size() == 1);
    builder.create<mlir::linalg::YieldOp>(loc, initVal);
  };
  auto val = builder.create<mlir::linalg::GenericOp>(
      loc, tensorType, llvm::None, tensorVal, affine_maps, iterators, body);
  return ctx.context.createVar(context, val.getResult(0));
}

py::object genericImpl(py::capsule context, py::handle inputs,
                       py::handle outputs, py::list iterators, py::list maps,
                       py::handle body) {
  auto &ctx = getPyContext(context);
  auto loc = ctx.loc;
  auto &builder = ctx.builder;
  auto &mlirContext = *builder.getContext();

  auto unpack = [&](py::handle obj) -> mlir::Value {
    return ctx.context.unwrapVal(loc, builder, obj);
  };

  auto inputsArgs = getAgrsFromTuple(inputs, unpack);
  auto outputArgs = getAgrsFromTuple(outputs, unpack);
  auto mlirIterators = getIterators(iterators, mlirContext);

  auto bodyTypes = getGenericOpBodyTypes(inputsArgs, outputArgs);
  auto funcTypes = mapTypesToNumbaChecked(ctx.context.types_mod, bodyTypes);
  auto bodyFunc = ctx.context.compileBody(body, funcTypes);

  auto castValues = [&](mlir::ValueRange vals, mlir::TypeRange types) {
    assert(vals.size() == types.size());
    llvm::SmallVector<mlir::Value> ret(vals.size());
    for (auto it : llvm::enumerate(vals)) {
      auto index = static_cast<unsigned>(it.index());
      ret[index] = doCast(builder, loc, it.value(), types[index]);
    }
    return ret;
  };

  auto affineMaps = getAffineMaps(maps, mlirContext);
  auto bodyBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
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

  auto genericOp = builder.create<mlir::linalg::GenericOp>(
      loc, retTypes, inputsArgsSignless, outputArgsSignless, affineMaps,
      mlirIterators, bodyBuilder);
  auto results =
      doSignCast(builder, loc, genericOp.getResults(), getTypes(outputArgs));
  return ctx.context.wrapResult(context, results);
}

py::object indexImpl(py::capsule context, py::int_ dimObj) {
  auto &ctx = getPyContext(context);
  auto loc = ctx.loc;
  auto &builder = ctx.builder;
  auto val = static_cast<int64_t>(dimObj);
  if (val < 0)
    plier::reportError("Index cannot be negative");

  auto dimVal =
      builder.create<mlir::linalg::IndexOp>(loc, static_cast<uint64_t>(val));
  auto resType = builder.getIntegerType(64, /*signed*/ true);
  auto res = doCast(builder, loc, dimVal, resType);
  return ctx.context.wrapResult(context, res);
}

py::object fromElementsImpl(py::capsule context, py::handle values,
                            py::handle dtype) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto type = unwrapType(dtype);

  llvm::SmallVector<mlir::Value> vals(containerSize(values));
  containerIterate(values, [&](auto index, py::handle obj) {
    if (py::isinstance(obj, ctx.context.var)) {
      vals[index] = unwrapSsaVal(obj);
    } else if (py::isinstance<py::int_>(obj) ||
               py::isinstance<py::float_>(obj)) {
      auto attr = [&]() -> mlir::Attribute {
        if (type.isa<mlir::IntegerType>()) {
          auto signless = makeSignlessType(type);
          return mlir::IntegerAttr::get(signless, obj.cast<int64_t>());
        }
        if (type.isa<mlir::FloatType>())
          return mlir::FloatAttr::get(type, obj.cast<double>());

        plier::reportError("Invalid dtype");
      }();
      auto res = builder.create<mlir::arith::ConstantOp>(loc, attr);
      vals[index] = doSignCast(builder, loc, res, type);
    } else {
      plier::reportError("Invalid element type");
    }
  });

  if (vals.empty())
    plier::reportError("Invalid from_elemets size");

  auto resTensorType =
      mlir::RankedTensorType::get(mlir::ShapedType::kDynamicSize, type);
  for (auto &val : vals)
    val = doSignCast(builder, loc, doCast(builder, loc, val, type));

  auto res =
      builder.create<mlir::tensor::FromElementsOp>(loc, vals).getResult();
  auto sizelessTensorType = mlir::RankedTensorType::get(
      mlir::ShapedType::kDynamicSize, makeSignlessType(type));
  res =
      builder.createOrFold<mlir::tensor::CastOp>(loc, sizelessTensorType, res);

  return ctx.context.createVar(context,
                               doSignCast(builder, loc, res, resTensorType));
}

py::object extractImpl(py::capsule context, py::handle value,
                       py::handle indices) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;

  llvm::SmallVector<mlir::Value> ind(containerSize(indices));
  containerIterate(indices, [&](auto index, py::handle obj) {
    if (py::isinstance(obj, ctx.context.var)) {
      ind[index] = doSignCast(builder, loc, unwrapSsaVal(obj));
    } else if (py::isinstance<py::int_>(obj)) {
      ind[index] = builder.create<mlir::arith::ConstantIndexOp>(
          loc, obj.cast<int64_t>());
    } else {
      plier::reportError("Invalid element type");
    }
  });
  auto tensor = ctx.context.unwrapVal(loc, builder, value);
  auto tensorType = tensor.getType().dyn_cast<mlir::RankedTensorType>();
  if (!tensorType)
    plier::reportError(llvm::Twine("extract: invalid source type ") +
                       toStr(tensor.getType()));

  auto origElement = tensorType.getElementType();
  auto res = builder
                 .create<mlir::tensor::ExtractOp>(
                     loc, doSignCast(builder, loc, tensor), ind)
                 .getResult();
  res = doSignCast(builder, loc, res, origElement);
  return ctx.context.createVar(context, res);
}

py::object reshapeImpl(py::capsule context, py::handle src,
                       py::iterable newDims) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;

  auto unwrapVal = [&](py::handle obj) {
    return ctx.context.unwrapVal(loc, builder, obj);
  };

  auto srcVal = unwrapVal(src);
  auto srcType = srcVal.getType().dyn_cast<mlir::RankedTensorType>();
  if (!srcType)
    plier::reportError(llvm::Twine("invalid reshape argument: ") +
                       toStr(srcVal.getType()));

  auto newDimsVals = [&]() {
    auto dimType = builder.getIndexType();
    auto dims = unwrapVal(newDims);
    llvm::SmallVector<mlir::Value> ret;
    if (auto tupleType = dims.getType().dyn_cast<mlir::TupleType>()) {
      auto dimsCount = tupleType.size();
      ret.resize(dimsCount);
      for (size_t i = 0; i < dimsCount; ++i) {
        auto ind = builder.create<mlir::arith::ConstantIndexOp>(loc, i);
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

  return ctx.context.createVar(context, reshaped);
}

py::object externalCallImpl(py::capsule context, py::str funcName,
                            py::handle inputs, py::handle outputs,
                            py::bool_ decorate) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto unwrapVal = [&](py::handle obj) {
    return ctx.context.unwrapVal(loc, builder, obj);
  };
  auto inputVals = toValues(inputs, unwrapVal);
  auto outputVals = toValues(outputs, unwrapVal);

  inputVals.reserve(inputVals.size() + outputVals.size());

  llvm::SmallVector<mlir::Type, 1> retTypes;
  for (auto val : outputVals) {
    auto type = val.getType();
    if (auto tensorType = type.dyn_cast<mlir::TensorType>()) {
      auto memrefType = mlir::MemRefType::get(tensorType.getShape(),
                                              tensorType.getElementType());
      auto memref =
          builder.create<mlir::bufferization::ToMemrefOp>(loc, memrefType, val);
      inputVals.emplace_back(memref);
    } else {
      retTypes.emplace_back(type);
    }
  }

  auto func = [&]() {
    auto argTypes = getTypes(inputVals);
    auto funcType =
        mlir::FunctionType::get(builder.getContext(), argTypes, retTypes);
    auto name = static_cast<std::string>(funcName);
    assert(!name.empty());
    auto mod =
        builder.getBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>();
    assert(mod);
    auto f = mod.lookupSymbol<mlir::FuncOp>(name);
    if (f) {
      if (f.getType() != funcType) {
        plier::reportError(llvm::Twine("linalg_builder::external_call: "
                                       "invalid function redefinition: ") +
                           name);
      }
    } else {
      f = plier::add_function(builder, mod, name, funcType);
      if (decorate)
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
      val = builder.create<mlir::bufferization::ToTensorOp>(loc, val);
      results.emplace_back(val);
    }
  }

  results.append(res.begin(), res.end());

  if (results.empty())
    return py::none();

  if (results.size() == 1)
    return ctx.context.createVar(context, results.front());

  py::tuple ret(results.size());
  for (auto it : llvm::enumerate(results))
    ret[it.index()] = ctx.context.createVar(context, it.value());

  return std::move(ret);
}

py::object insertImpl(py::capsule context, py::handle src, py::handle dst,
                      py::handle offsets, py::handle sizes,
                      py::handle strides) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto unwrapVal = [&](py::handle obj) {
    return ctx.context.unwrapVal(loc, builder, obj);
  };
  auto indexType = builder.getIndexType();
  auto unwrapList = [&](py::handle obj) {
    auto len = py::len(obj);
    llvm::SmallVector<mlir::Value> res(len);
    for (auto it : llvm::enumerate(obj))
      res[it.index()] = doCast(builder, loc, unwrapVal(it.value()), indexType);

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
  return ctx.context.createVar(context, res);
}

py::object inlineFuncImpl(py::capsule context, py::handle func,
                          py::handle retType, py::tuple args) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;

  auto argsValues = [&]() {
    auto unwrapVal = [&](py::handle obj) {
      return ctx.context.unwrapVal(loc, builder, obj);
    };
    llvm::SmallVector<mlir::Value> ret(args.size());
    for (auto it : llvm::enumerate(args))
      ret[it.index()] = unwrapVal(it.value());

    return ret;
  }();
  auto funcTypes =
      mapTypesToNumbaChecked(ctx.context.types_mod, getTypes(argsValues));
  auto bodyFunc = ctx.context.compileBody(func, funcTypes);
  auto funcType = bodyFunc.getType();
  auto funcArgsTypes = funcType.getInputs();
  if (funcArgsTypes.size() != argsValues.size())
    plier::reportError(
        llvm::Twine("Invalid function arguments count, expected ") +
        llvm::Twine(argsValues.size()) + ", got" +
        llvm::Twine(funcArgsTypes.size()));

  if (funcType.getNumResults() != 1)
    plier::reportError(llvm::Twine("Invalid number of return values: ") +
                       llvm::Twine(funcType.getNumResults()));

  auto castValues = [&](mlir::ValueRange vals, mlir::TypeRange types) {
    assert(vals.size() == types.size());
    llvm::SmallVector<mlir::Value> ret(vals.size());
    for (auto it : llvm::enumerate(vals)) {
      auto index = static_cast<unsigned>(it.index());
      ret[index] = doCast(builder, loc, it.value(), types[index]);
    }
    return ret;
  };

  auto castedArgs = castValues(argsValues, funcArgsTypes);

  auto resValue =
      builder.create<mlir::CallOp>(loc, bodyFunc, castedArgs).getResult(0);

  auto mlirRetType = unwrapType(retType);
  resValue = doCast(builder, loc, resValue, mlirRetType);
  return ctx.context.createVar(context, resValue);
}

py::object castImpl(py::capsule context, py::handle src, py::handle dtype) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto unwrapVal = [&](py::handle obj) {
    return ctx.context.unwrapVal(loc, builder, obj);
  };
  auto val = unwrapVal(src);
  auto type = unwrapType(dtype);
  auto ret = builder.createOrFold<plier::CastOp>(loc, type, val);
  return ctx.context.createVar(context, ret);
}

py::object undefImpl(py::capsule context, py::handle dtype) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto type = unwrapType(dtype);
  auto ret = builder.createOrFold<plier::UndefOp>(loc, type);
  return ctx.context.createVar(context, ret);
}

py::object subviewImpl(py::capsule context, py::handle src, py::handle offsets,
                       py::handle sizes, py::handle strides, py::handle rank) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto origSrcVal = ctx.context.unwrapVal(loc, builder, src);
  auto origSrcType = origSrcVal.getType().cast<mlir::ShapedType>();
  auto srcVal = doSignCast(builder, loc, origSrcVal);
  auto srcType = srcVal.getType().cast<mlir::TensorType>();
  auto memrefType =
      mlir::MemRefType::get(srcType.getShape(), srcType.getElementType());
  auto memref =
      builder.create<mlir::bufferization::ToMemrefOp>(loc, memrefType, srcVal);

  auto indexType = builder.getIndexType();
  auto indexCast = [&](mlir::Value val) -> mlir::OpFoldResult {
    while (auto parent = val.getDefiningOp<plier::SignCastOp>())
      val = parent.value();

    if (auto constVal = mlir::getConstantIntValue(val))
      return builder.getIndexAttr(*constVal);

    return doCast(builder, loc, val, indexType);
  };

  auto unwrapVal = [&](py::handle obj) {
    return ctx.context.unwrapVal(loc, builder, obj);
  };

  using ValsArray = llvm::SmallVector<mlir::OpFoldResult>;
  auto unpackValues = [&](py::handle obj) -> ValsArray {
    auto input = toValues<ValsArray>(obj, unwrapVal);
    assert(!input.empty());
    if (input.size() > 1) {
      for (auto i : llvm::seq(size_t(0), input.size()))
        input[i] = indexCast(input[i].get<mlir::Value>());

      return input;
    }

    auto val = input.front().get<mlir::Value>();
    ValsArray ret;
    if (auto tupleType = val.getType().dyn_cast<mlir::TupleType>()) {
      ret.resize(tupleType.size());
      for (auto i : llvm::seq(size_t(0), tupleType.size())) {
        auto ind = builder.create<mlir::arith::ConstantIndexOp>(loc, i);
        ret[i] = indexCast(builder.createOrFold<plier::GetItemOp>(
            loc, tupleType.getType(i), val, ind));
      }
    } else {
      ret.emplace_back(indexCast(val));
    }

    return ret;
  };

  auto offsetVals = unpackValues(offsets);
  auto sizeVals = [&]() -> ValsArray {
    if (sizes.is_none()) {
      ValsArray ret(offsetVals.size());
      for (auto i : llvm::seq(size_t(0), ret.size())) {
        auto dim = builder.createOrFold<mlir::tensor::DimOp>(
            loc, origSrcVal, static_cast<int64_t>(i));
        auto offset = offsetVals[i].get<mlir::Value>();
        auto size = builder.createOrFold<mlir::arith::SubIOp>(loc, dim, offset);
        ret[i] = size;
      }
      return ret;
    } else {
      return unpackValues(sizes);
    }
  }();
  auto strideVals = [&]() -> ValsArray {
    if (strides.is_none()) {
      return ValsArray(offsetVals.size(), builder.getIndexAttr(1));
    } else {
      return unpackValues(strides);
    }
  }();
  auto viewMemrefType = [&]() -> mlir::MemRefType {
    if (rank.is_none()) {
      return mlir::memref::SubViewOp::inferResultType(memrefType, offsetVals,
                                                      sizeVals, strideVals)
          .cast<mlir::MemRefType>();
    } else {
      auto rankVal = rank.cast<unsigned>();
      return mlir::memref::SubViewOp::inferRankReducedResultType(
                 rankVal, memrefType, offsetVals, sizeVals, strideVals)
          .cast<mlir::MemRefType>();
    }
  }();
  auto view = builder.createOrFold<mlir::memref::SubViewOp>(
      loc, viewMemrefType, memref, offsetVals, sizeVals, strideVals);

  auto getDynShape = [](int64_t r) {
    return llvm::SmallVector<int64_t>(r, mlir::ShapedType::kDynamicSize);
  };

  auto flatViewType = mlir::MemRefType::get(
      getDynShape(viewMemrefType.getRank()), viewMemrefType.getElementType());
  view = builder.createOrFold<plier::ChangeLayoutOp>(loc, flatViewType, view);
  auto ret =
      builder.create<mlir::bufferization::ToTensorOp>(loc, view).getResult();
  auto resType = ret.getType().cast<mlir::ShapedType>();
  auto resSignlessType = resType.clone(getDynShape(resType.getRank()));
  if (resSignlessType != resType)
    ret = builder.create<mlir::tensor::CastOp>(loc, resSignlessType, ret);
  auto resSignedType = resSignlessType.clone(origSrcType.getElementType());
  return ctx.context.createVar(context,
                               doSignCast(builder, loc, ret, resSignedType));
}

py::object arrayTypeImpl(py::capsule context, py::iterable dims,
                         py::handle dtype) {
  auto &ctx = getPyContext(context);
  auto elemType = unwrapType(dtype);
  llvm::SmallVector<int64_t> shape(py::len(dims));
  for (auto it : llvm::enumerate(dims))
    shape[it.index()] = it.value().cast<int64_t>();

  auto arrayType = mlir::RankedTensorType::get(shape, elemType);
  return ctx.context.createType(arrayType);
}

void setupPyBuilder(py::handle builder, mlir::OpBuilder &b,
                    llvm::function_ref<py::object(mlir::Type)> createType) {
  py::setattr(builder, "_broadcast", py::cpp_function(&broadcastImpl));
  py::setattr(builder, "_init_tensor", py::cpp_function(&initTensorImpl));
  py::setattr(builder, "_fill_tensor", py::cpp_function(&fillTensorImpl));
  py::setattr(builder, "_linalg_generic", py::cpp_function(&genericImpl));
  py::setattr(builder, "_linalg_index", py::cpp_function(&indexImpl));
  py::setattr(builder, "_from_elements", py::cpp_function(&fromElementsImpl));
  py::setattr(builder, "_extract", py::cpp_function(&extractImpl));
  py::setattr(builder, "_reshape", py::cpp_function(&reshapeImpl));
  py::setattr(builder, "_external_call", py::cpp_function(&externalCallImpl));
  py::setattr(builder, "_insert", py::cpp_function(&insertImpl));
  py::setattr(builder, "_inline_func", py::cpp_function(&inlineFuncImpl));
  py::setattr(builder, "_cast", py::cpp_function(&castImpl));
  py::setattr(builder, "_undef", py::cpp_function(&undefImpl));
  py::setattr(builder, "_subview", py::cpp_function(&subviewImpl));

  py::setattr(builder, "_array_type", py::cpp_function(&arrayTypeImpl));

  auto addType = [&](const char *name, mlir::Type type) {
    py::setattr(builder, name, createType(type));
  };

  addType("bool", b.getIntegerType(1));

  addType("int8", b.getIntegerType(8, true));
  addType("int16", b.getIntegerType(16, true));
  addType("int32", b.getIntegerType(32, true));
  addType("int64", b.getIntegerType(64, true));

  addType("uint8", b.getIntegerType(8, false));
  addType("uint16", b.getIntegerType(16, false));
  addType("uint32", b.getIntegerType(32, false));
  addType("uint64", b.getIntegerType(64, false));

  addType("int8_signless", b.getIntegerType(8));
  addType("int16_signless", b.getIntegerType(16));
  addType("int32_signless", b.getIntegerType(32));
  addType("int64_signless", b.getIntegerType(64));

  addType("index", b.getIndexType());

  addType("float16", b.getF16Type());
  addType("float32", b.getF32Type());
  addType("float64", b.getF64Type());
}

py::object shapeImpl(py::capsule context, py::capsule ssaVal) {
  auto &ctx = getPyContext(context);
  auto value = unwrapMlir<mlir::Value>(ssaVal);
  if (auto mlirType = value.getType().dyn_cast<mlir::RankedTensorType>()) {
    auto &builder = ctx.builder;
    auto loc = ctx.loc;
    auto shape = mlirType.getShape();
    llvm::SmallVector<mlir::Value> shapeVals(shape.size());
    for (auto it : llvm::enumerate(shape)) {
      auto i = it.index();
      mlir::Value mlir_dim = builder.create<mlir::tensor::DimOp>(loc, value, i);
      shapeVals[i] = mlir_dim;
    }
    llvm::SmallVector<mlir::Type> shape_types(shape.size(),
                                              builder.getIndexType());
    auto shapeType = mlir::TupleType::get(builder.getContext(), shape_types);
    auto shapeVar =
        builder.create<plier::BuildTupleOp>(loc, shapeType, shapeVals);
    return ctx.context.createVar(context, shapeVar.getResult());
  }
  return py::list();
}

py::object dtypeImpl(py::capsule context, py::capsule ssaVal) {
  auto &ctx = getPyContext(context);
  auto value = unwrapMlir<mlir::Value>(ssaVal);
  auto type = value.getType();
  if (auto tensorType = type.dyn_cast<mlir::ShapedType>())
    type = tensorType.getElementType();

  return ctx.context.createType(type);
}

py::object typeImpl(py::capsule context, py::capsule ssaVal) {
  auto &ctx = getPyContext(context);
  auto value = unwrapMlir<mlir::Value>(ssaVal);
  auto type = value.getType();
  return ctx.context.createType(type);
}

py::object lenImpl(py::capsule /*context*/, py::capsule ssaVal) {
  auto value = unwrapMlir<mlir::Value>(ssaVal);
  auto type = value.getType();
  if (auto tupleType = type.dyn_cast<mlir::TupleType>())
    return py::int_(tupleType.size());

  return py::none();
}

py::object getitemImpl(py::capsule context, py::capsule ssaVal,
                       py::handle index) {
  auto &ctx = getPyContext(context);
  auto value = unwrapMlir<mlir::Value>(ssaVal);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto type = value.getType();
  if (auto tensor = type.dyn_cast<mlir::TensorType>()) {
    auto indexVal = ctx.context.unwrapVal(loc, builder, index);
    auto elemType = tensor.getElementType();
    auto res = builder.create<plier::GetItemOp>(loc, elemType, value, indexVal);
    return ctx.context.createVar(context, res.getResult());
  }

  auto indexVal = index.cast<int64_t>();
  if (auto tupleType = type.dyn_cast<mlir::TupleType>()) {
    auto maxIndex = static_cast<int64_t>(tupleType.size());
    if (indexVal < 0 || indexVal >= maxIndex)
      throw py::index_error(("Invalid getitem index: " + llvm::Twine(indexVal) +
                             ", expected [0:" + llvm::Twine(maxIndex) + ")")
                                .str());

    if (auto parentOp = value.getDefiningOp<plier::BuildTupleOp>())
      return ctx.context.createVar(
          context, parentOp.getOperand(static_cast<unsigned>(indexVal)));

    auto elemType = tupleType.getType(static_cast<size_t>(indexVal));
    auto ind = builder.create<mlir::arith::ConstantIndexOp>(loc, indexVal);
    auto item = builder.create<plier::GetItemOp>(loc, elemType, value, ind);
    return ctx.context.createVar(context, item.getResult());
  } else {
    throw py::index_error("Invalid getitem");
  }
}

template <typename Op>
static mlir::Value binopFunc(mlir::Location loc, mlir::OpBuilder &builder,
                             mlir::Value lhs, mlir::Value rhs) {
  return builder.create<Op>(loc, lhs, rhs);
}

template <typename Op>
static mlir::Value rbinopFunc(mlir::Location loc, mlir::OpBuilder &builder,
                              mlir::Value lhs, mlir::Value rhs) {
  return builder.create<Op>(loc, rhs, lhs);
}

static mlir::Value binopFuncIdiv(mlir::Location loc, mlir::OpBuilder &builder,
                                 mlir::Value lhs, mlir::Value rhs) {
  auto lhsVar = doCast(builder, loc, lhs, builder.getF64Type());
  auto rhsVar = doCast(builder, loc, rhs, builder.getF64Type());
  return builder.create<mlir::arith::DivFOp>(loc, lhsVar, rhsVar);
}

py::object binopImpl(py::capsule context, py::capsule ssaVal, py::handle rhs,
                     py::str op) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto lhs = unwrapMlir<mlir::Value>(ssaVal);

  auto type = lhs.getType();
  if (!type.isa<mlir::IntegerType, mlir::IndexType, mlir::FloatType,
                mlir::ShapedType>())
    plier::reportError("Invalid binop arg type");

  auto isFloat = [&]() -> bool {
    if (auto shapedType = type.dyn_cast<mlir::ShapedType>())
      return shapedType.getElementType().isa<mlir::FloatType>();

    return type.isa<mlir::FloatType>();
  }();

  using binop_func_t =
      mlir::Value (*)(mlir::Location loc, mlir::OpBuilder & builder,
                      mlir::Value lhs, mlir::Value rhs);
  const std::tuple<llvm::StringRef, binop_func_t, binop_func_t> funcs[] = {
      {"+", &binopFunc<mlir::arith::AddIOp>, &binopFunc<mlir::arith::AddFOp>},
      {"-", &binopFunc<mlir::arith::SubIOp>, &binopFunc<mlir::arith::SubFOp>},
      {"r-", &rbinopFunc<mlir::arith::SubIOp>,
       &rbinopFunc<mlir::arith::SubFOp>},
      {"*", &binopFunc<mlir::arith::MulIOp>, &binopFunc<mlir::arith::MulFOp>},
      {"/", &binopFuncIdiv, &binopFunc<mlir::arith::DivFOp>},
  };

  auto opName = static_cast<std::string>(op);
  for (auto f : funcs) {
    auto name = std::get<0>(f);
    auto func = (isFloat ? std::get<2>(f) : std::get<1>(f));
    if (name == opName) {
      auto rhsVar =
          doCast(builder, loc, ctx.context.unwrapVal(loc, builder, rhs), type);
      auto res = func(loc, builder, lhs, rhsVar);
      return ctx.context.createVar(context, res);
    }
  }
  plier::reportError("Unhandled binop type");
}

py::object strImpl(py::capsule /*context*/, py::capsule ssaVal) {
  return py::str("Var: \"" + toStr(unwrapMlir<mlir::Value>(ssaVal)) + "\"");
}

void setupPyVar(pybind11::handle var) {
  py::setattr(var, "_shape", py::cpp_function(&shapeImpl));
  py::setattr(var, "_dtype", py::cpp_function(&dtypeImpl));
  py::setattr(var, "_type", py::cpp_function(&typeImpl));
  py::setattr(var, "_len", py::cpp_function(&lenImpl));
  py::setattr(var, "_getitem", py::cpp_function(&getitemImpl));
  py::setattr(var, "_binop", py::cpp_function(&binopImpl));
  py::setattr(var, "_str", py::cpp_function(&strImpl));
}

PyLinalgResolver::Values unpackResults(PyBuilderContext &ctx,
                                       py::handle object) {
  PyLinalgResolver::Values ret;
  if (object.is_none())
    return ret;

  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto unwrapVal = [&](py::handle obj) {
    return ctx.context.unwrapVal(loc, builder, obj);
  };
  if (py::isinstance<py::tuple>(object)) {
    auto tuple = object.cast<py::tuple>();
    llvm::SmallVector<mlir::Value> vals(tuple.size());
    for (auto it : llvm::enumerate(tuple))
      vals[it.index()] = unwrapVal(it.value());

    mlir::ValueRange vr(vals);

    auto tupleType = mlir::TupleType::get(builder.getContext(), vr.getTypes());
    ret.emplace_back(builder.create<plier::BuildTupleOp>(loc, tupleType, vr));
  } else {
    ret.emplace_back(unwrapVal(object));
  }
  return ret;
}
} // namespace

PyLinalgResolver::PyLinalgResolver(const char *modName, const char *regName)
    : context(std::make_unique<Context>()) {
  assert(modName != nullptr);
  assert(regName != nullptr);
  auto builderMod = py::module::import("numba_dpcomp.mlir.linalg_builder");
  auto registryMod = py::module::import(modName);
  auto registry = registryMod.attr(regName);

  context->var = builderMod.attr("Var");
  context->type = builderMod.attr("Type");
  context->builder = builderMod.attr("Builder");
  context->inspect = py::module::import("inspect");
  context->types_mod = py::module::import("numba.core.types");
  context->compile_func = builderMod.attr("compile_func");
  context->lookup_func = registry.attr("lookup_func");
}

PyLinalgResolver::~PyLinalgResolver() {}

llvm::Optional<PyLinalgResolver::Values>
PyLinalgResolver::rewriteFunc(llvm::Twine name, mlir::Location loc,
                              mlir::OpBuilder &builder, mlir::ValueRange args,
                              KWArgs kwargs) const {
  return rewrite((name + "()").str(), loc, builder, args, kwargs);
}

llvm::Optional<PyLinalgResolver::Values>
PyLinalgResolver::rewriteAttr(llvm::Twine name, mlir::Location loc,
                              mlir::OpBuilder &builder, mlir::Value arg) const {
  return rewrite(name.str(), loc, builder, arg, {});
}

llvm::Optional<PyLinalgResolver::Values>
PyLinalgResolver::rewrite(llvm::StringRef name, mlir::Location loc,
                          mlir::OpBuilder &builder, mlir::ValueRange args,
                          KWArgs kwargs) const {
  assert(!name.empty());
  if (!isCompatibleTypes(args) ||
      !isCompatibleTypes(llvm::make_second_range(kwargs)))
    return {};

  auto builderFunc = context->lookup_func(py::str(name.data(), name.size()));
  if (builderFunc.is_none())
    return {};

  PyBuilderContext pyBuilderContext{loc, builder, *context};
  auto pyContext = py::capsule(&pyBuilderContext);
  auto pyArgs = getArgs(
      context->inspect, builderFunc,
      [&](auto val) { return context->createVar(pyContext, val); }, args,
      kwargs);
  if (pyArgs.is_none())
    return {};

  auto pyBuilder = context->builder(pyContext);
  setupPyBuilder(pyBuilder, builder,
                 [&](auto type) { return context->createType(type); });

  auto result = builderFunc(pyBuilder, *pyArgs);
  if (result.is_none())
    return {};

  return unpackResults(pyBuilderContext, result);
}
