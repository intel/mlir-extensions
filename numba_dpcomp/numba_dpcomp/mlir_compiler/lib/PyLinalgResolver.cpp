// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PyLinalgResolver.hpp"

#include <pybind11/pybind11.h>

#include <mlir/AsmParser/AsmParser.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Parser/Parser.h>

#include "imex/Dialect/imex_util/Dialect.hpp"
#include "imex/Dialect/ntensor/IR/NTensorOps.hpp"
#include "imex/Transforms/CastUtils.hpp"
#include "imex/Transforms/ConstUtils.hpp"
#include "imex/Transforms/FuncUtils.hpp"
#include "imex/Utils.hpp"

#include "PyMapTypes.hpp"

namespace py = pybind11;

struct PyBuilderContext {
  mlir::Location loc;
  mlir::OpBuilder &builder;
  PyLinalgResolver::Context &context;
};

namespace {
template <typename C> static auto getTempShape(const C &container) {
  return llvm::SmallVector<mlir::OpFoldResult>(std::begin(container),
                                               std::end(container));
}

static std::string toStr(mlir::Value val) {
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  ss << val;
  ss.flush();
  return ret;
}

static std::string toStr(mlir::Type type) {
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  ss << type;
  ss.flush();
  return ret;
}

static std::string toStr(mlir::TypeRange typesRange) {
  std::string str;
  llvm::raw_string_ostream ss(str);
  for (auto type : typesRange)
    ss << type << " ";

  ss.flush();
  return str;
}

static std::string toStr(py::handle obj) {
  return py::str(obj).cast<std::string>();
}

static py::object mapTypesToNumbaChecked(py::handle typesMod,
                                         mlir::TypeRange typesRange) {
  auto funcTypes = mapTypesToNumba(typesMod, typesRange);
  if (funcTypes.is_none())
    imex::reportError(llvm::Twine("map_types_to_numba failed: ") +
                      toStr(typesRange));
  return funcTypes;
}

static mlir::Type makeSignlessType(mlir::Type type) {
  if (auto shaped = type.dyn_cast<mlir::ShapedType>()) {
    auto origElemType = shaped.getElementType();
    auto signlessElemType = makeSignlessType(origElemType);
    return shaped.clone(signlessElemType);
  } else if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
    if (!intType.isSignless())
      return mlir::IntegerType::get(intType.getContext(), intType.getWidth());
  }
  return type;
}

static mlir::Value doSignCast(mlir::OpBuilder &builder, mlir::Location &loc,
                              mlir::Value val) {
  auto origType = val.getType();
  auto signlessType = makeSignlessType(origType);
  if (signlessType != origType)
    val = builder.createOrFold<imex::util::SignCastOp>(loc, signlessType, val);

  return val;
}

static mlir::Value doSignCast(mlir::OpBuilder &builder, mlir::Location &loc,
                              mlir::Value val, mlir::Type dstType) {
  auto origType = val.getType();
  if (dstType != origType)
    val = builder.createOrFold<imex::util::SignCastOp>(loc, dstType, val);

  return val;
}

static auto getTypes(mlir::ValueRange values) {
  auto types = values.getTypes();
  llvm::SmallVector<mlir::Type> ret(types.begin(), types.end());
  return ret;
}

static bool isCompatibleType(mlir::Type type) {
  if (auto tupleType = type.dyn_cast<mlir::TupleType>())
    return llvm::all_of(tupleType, &isCompatibleType);

  return type.isa<mlir::IntegerType, mlir::IndexType, mlir::FloatType,
                  mlir::ComplexType, mlir::RankedTensorType, mlir::MemRefType,
                  mlir::NoneType, imex::util::TypeVarType,
                  imex::ntensor::NTensorType>();
}

static bool isCompatibleTypeVal(mlir::Value val) {
  return isCompatibleType(val.getType());
}

template <typename R> static bool isCompatibleTypes(R &&vals) {
  return llvm::all_of(vals, &isCompatibleTypeVal);
}

template <typename T> static py::capsule wrapMlir(T val) {
  return py::capsule(val.getAsOpaquePointer());
}

template <typename T> static T unwrapMlir(py::capsule obj) {
  return T::getFromOpaquePointer(static_cast<const void *>(obj));
}

static auto unwrapSsaVal(py::handle obj) {
  return unwrapMlir<mlir::Value>(obj.attr("_ssa_val").cast<py::capsule>());
}

static auto unwrapType(py::handle obj) {
  if (py::hasattr(obj, "_ssa_val")) {
    auto val = unwrapSsaVal(obj);
    if (auto type = val.getType().dyn_cast<imex::util::TypeVarType>())
      return type.getType();
  } else if (py::hasattr(obj, "_mlir_type")) {
    return unwrapMlir<mlir::Type>(obj.attr("_mlir_type").cast<py::capsule>());
  }
  imex::reportError(llvm::Twine("Invalid type object: ") +
                    toStr(obj.get_type()));
}

static size_t containerSize(py::handle obj) {
  if (py::isinstance<py::tuple>(obj))
    return obj.cast<py::tuple>().size();

  if (py::isinstance<py::list>(obj))
    return obj.cast<py::list>().size();

  return 1;
}

template <typename F> static void containerIterate(py::handle obj, F &&func) {
  auto impl = [&](auto cont) {
    for (auto [i, val] : llvm::enumerate(cont))
      func(i, val);
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
static auto toValues(py::handle obj, UnwrapFunc &&unwrapFunc) {
  RetType ret(containerSize(obj));
  containerIterate(
      obj, [&](auto index, py::handle elem) { ret[index] = unwrapFunc(elem); });
  return ret;
}

static llvm::Optional<py::object> getPyLiteral(mlir::Attribute attr) {
  assert(attr);
  if (auto intAttr = attr.dyn_cast<mlir::IntegerAttr>()) {
    if (auto intType = attr.cast<mlir::TypedAttr>()
                           .getType()
                           .dyn_cast<mlir::IntegerType>()) {
      // Ignore index type
      if (intType.getWidth() == 1)
        return py::bool_(intAttr.getInt() != 0);
    }
    return py::int_(imex::getIntAttrValue(intAttr));
  }
  if (auto floatAttr = attr.dyn_cast<mlir::FloatAttr>())
    return py::float_(floatAttr.getValueAsDouble());

  return {};
}

static mlir::Value doCast(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value val, mlir::Type dstType) {
  auto srcType = val.getType();
  if (srcType == dstType)
    return val;

  if (imex::canConvert(srcType, dstType))
    return imex::doConvert(builder, loc, val, dstType);

  if (srcType.isa<mlir::MemRefType>() &&
      dstType.isa<imex::ntensor::NTensorType>())
    return builder.createOrFold<imex::ntensor::FromMemrefOp>(loc, dstType, val);

  if (srcType.isa<imex::ntensor::NTensorType>() &&
      dstType.isa<mlir::MemRefType>())
    return builder.createOrFold<imex::ntensor::ToMemrefOp>(loc, dstType, val);

  if (srcType.isa<mlir::RankedTensorType>() &&
      dstType.isa<imex::ntensor::NTensorType>())
    return builder.createOrFold<imex::ntensor::FromTensorOp>(loc, dstType, val);

  if (srcType.isa<imex::ntensor::NTensorType>() &&
      dstType.isa<mlir::RankedTensorType>())
    return builder.createOrFold<imex::ntensor::ToTensorOp>(loc, dstType, val);

  if (srcType.isa<imex::ntensor::NTensorType>() &&
      dstType.isa<imex::ntensor::NTensorType>())
    return builder.createOrFold<imex::ntensor::CastOp>(loc, dstType, val);

  imex::reportError(llvm::Twine("Cannot cast types :") + toStr(srcType) +
                    " to " + toStr(dstType));
}

static bool cmpCapsule(py::capsule a1, py::capsule a2) {
  return static_cast<void *>(a1) == static_cast<void *>(a2);
}

static py::object printTypeCapsule(py::capsule t) {
  auto type = unwrapMlir<mlir::Type>(t);
  return py::str("Type: \"" + toStr(type) + "\"");
}

static void setupPyVar(py::handle var);
} // namespace

struct PyLinalgResolver::Context {
  py::object var;
  py::object type;
  py::object builder;
  py::object inspect;
  py::object typesMod;
  py::object compileFunc;
  py::object lookupFunc;

  py::object createVar(py::capsule context, mlir::Value value,
                       bool makeLiteral = false) {
    assert(value);
    auto type = value.getType();
    if (type.isa<mlir::NoneType>())
      return py::none();

    if (auto typevar = type.dyn_cast<imex::util::TypeVarType>())
      return createType(typevar.getType());

    if (makeLiteral) {
      if (auto literal = makePyLiteral(context, value))
        return *literal;
    }

    auto ret = var(context, wrapMlir(value));
    setupPyVar(ret);
    return ret;
  }

  py::object createType(mlir::Type t) {
    return type(wrapMlir(t), py::cpp_function(&cmpCapsule),
                py::cpp_function(&printTypeCapsule));
  }

  mlir::func::FuncOp compileBody(py::handle body, py::list arg_types) {
    auto func = compileFunc(body, arg_types).cast<py::capsule>();
    auto mlirFunc =
        mlir::cast<mlir::func::FuncOp>(static_cast<mlir::Operation *>(func));
    mlirFunc.setPrivate();
    mlirFunc->setAttr(imex::util::attributes::getForceInlineName(),
                      mlir::UnitAttr::get(mlirFunc->getContext()));
    return mlirFunc;
  }

  py::object wrapResult(py::capsule context, mlir::ValueRange values) {
    if (values.empty())
      return py::none();

    if (values.size() == 1)
      return createVar(context, values.front());

    py::tuple ret(values.size());
    for (auto [i, val] : llvm::enumerate(values))
      ret[i] = createVar(context, val);

    return std::move(ret);
  }

  mlir::Value unwrapVal(mlir::Location loc, mlir::OpBuilder &builder,
                        py::handle obj) {
    if (py::isinstance(obj, var))
      return unwrapSsaVal(obj);

    if (py::isinstance(obj, type)) {
      auto type = imex::util::TypeVarType::get(unwrapType(obj));
      return builder.create<imex::util::UndefOp>(loc, type);
    }

    if (obj.is_none()) {
      auto type = mlir::NoneType::get(builder.getContext());
      return builder.create<imex::util::UndefOp>(loc, type);
    }

    if (py::isinstance<py::iterable>(obj)) {
      llvm::SmallVector<mlir::Value> elems(py::len(obj));
      for (auto [i, val] : llvm::enumerate(obj))
        elems[i] = unwrapVal(loc, builder, val);

      mlir::ValueRange vr(elems);
      auto resType = mlir::TupleType::get(builder.getContext(), vr.getTypes());
      return builder.create<imex::util::BuildTupleOp>(loc, resType, elems);
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
      return builder.create<imex::util::SignCastOp>(loc, intType, res);
    }

    if (py::isinstance<py::float_>(obj)) {
      auto attr = builder.getF64FloatAttr(obj.cast<double>());
      return builder.create<mlir::arith::ConstantOp>(loc, attr);
    }

    imex::reportError(llvm::Twine("Invalid element type: ") +
                      toStr(obj.get_type()));
  }

  mlir::Value unwrapVal(mlir::Location loc, mlir::OpBuilder &builder,
                        py::handle obj, mlir::Type resultType) {
    assert(resultType);
    if (resultType.isa<mlir::IndexType>() && py::isinstance<py::int_>(obj)) {
      return builder.create<mlir::arith::ConstantIndexOp>(loc,
                                                          obj.cast<int64_t>());
    } else if (resultType.isa<mlir::IntegerType>() &&
               py::isinstance<py::int_>(obj)) {
      auto intType = resultType.cast<mlir::IntegerType>();
      auto intSignlessType = makeSignlessType(intType);
      mlir::Value res = builder.create<mlir::arith::ConstantIntOp>(
          loc, obj.cast<int64_t>(), intSignlessType);
      if (intSignlessType != intType)
        res = builder.create<imex::util::SignCastOp>(loc, intType, res);

      return res;
    } else if (resultType.isa<mlir::FloatType>() &&
               py::isinstance<py::float_>(obj)) {
      auto floatVal = [&]() {
        auto v = obj.cast<double>();
        if (resultType.isF32())
          return llvm::APFloat(static_cast<float>(v));
        if (resultType.isF64())
          return llvm::APFloat(static_cast<double>(v));
        llvm_unreachable("Unhandled float type");
      }();
      return builder.create<mlir::arith::ConstantFloatOp>(
          loc, floatVal, resultType.cast<mlir::FloatType>());
    }

    return doCast(builder, loc, unwrapVal(loc, builder, obj), resultType);
  }

private:
  llvm::Optional<py::object> makePyLiteral(py::capsule context,
                                           mlir::Value val) {
    assert(val);
    if (auto buildTuple = val.getDefiningOp<imex::util::BuildTupleOp>()) {
      auto args = buildTuple.getArgs();
      auto count = static_cast<unsigned>(args.size());
      py::tuple ret(count);
      for (auto i : llvm::seq(0u, count))
        ret[i] = createVar(context, args[i]);

      return ret;
    }

    if (auto cast = val.getDefiningOp<imex::util::SignCastOp>())
      val = cast.getSource();

    if (auto attr = imex::getConstVal<mlir::Attribute>(val))
      return getPyLiteral(attr);

    return {};
  }
};

namespace {
static mlir::Value toTensor(mlir::Location loc, mlir::OpBuilder &builder,
                            mlir::Value val) {
  auto srcType = val.getType();
  if (auto memrefType = srcType.dyn_cast<mlir::MemRefType>())
    return builder.create<mlir::bufferization::ToTensorOp>(loc, val);

  if (auto ntensorType = srcType.dyn_cast<imex::ntensor::NTensorType>()) {
    auto tensorType = mlir::RankedTensorType::get(ntensorType.getShape(),
                                                  ntensorType.getElementType());
    return builder.create<imex::ntensor::ToTensorOp>(loc, tensorType, val);
  }

  return val;
}

static mlir::Type toNTensorType(mlir::ShapedType type) {
  return imex::ntensor::NTensorType::get(type.getShape(),
                                         type.getElementType());
}

static mlir::Value toNTensor(mlir::Location loc, mlir::OpBuilder &builder,
                             mlir::Value val) {
  auto srcType = val.getType();
  if (auto memrefType = srcType.dyn_cast<mlir::MemRefType>())
    return builder.create<imex::ntensor::FromMemrefOp>(
        loc, toNTensorType(memrefType), val);

  if (auto tensorType = srcType.dyn_cast<mlir::RankedTensorType>())
    return builder.create<imex::ntensor::FromTensorOp>(
        loc, toNTensorType(tensorType), val);

  return val;
}

template <typename T> static auto getDynShape(T r) {
  return llvm::SmallVector<int64_t>(static_cast<size_t>(r),
                                    mlir::ShapedType::kDynamic);
};

static mlir::MemRefType getMemrefType(mlir::ShapedType srcType, bool strided) {
  if (strided) {
    auto ctx = srcType.getContext();
    auto layout = mlir::StridedLayoutAttr::get(ctx, mlir::ShapedType::kDynamic,
                                               getDynShape(srcType.getRank()));
    return mlir::MemRefType::get(srcType.getShape(), srcType.getElementType(),
                                 layout);
  } else {
    return mlir::MemRefType::get(srcType.getShape(), srcType.getElementType());
  }
}

static mlir::Value toMemref(mlir::Location loc, mlir::OpBuilder &builder,
                            mlir::Value val, float strided = false) {
  auto srcType = val.getType();
  if (auto tensorType = srcType.dyn_cast<mlir::RankedTensorType>()) {
    auto type = getMemrefType(tensorType, strided);
    return builder.create<mlir::bufferization::ToMemrefOp>(loc, type, val);
  }

  if (auto ntensorType = srcType.dyn_cast<imex::ntensor::NTensorType>()) {
    auto type = getMemrefType(ntensorType, strided);
    return builder.create<imex::ntensor::ToMemrefOp>(loc, type, val);
  }

  return val;
}

static py::object
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
  for (auto [i, val] : llvm::enumerate(retArgs))
    ret[i] = std::move(val);

  return std::move(ret);
}

static PyBuilderContext &getPyContext(py::capsule &ctx) {
  return *static_cast<PyBuilderContext *>(ctx);
}

static auto
getAgrsFromTuple(py::handle args,
                 llvm::function_ref<mlir::Value(py::handle)> unpack) {
  llvm::SmallVector<mlir::Value> ret;
  if (args.is_none())
    return ret;

  if (py::isinstance<py::tuple>(args)) {
    auto tuple = args.cast<py::tuple>();
    ret.resize(tuple.size());
    for (auto [i, val] : llvm::enumerate(tuple))
      ret[i] = unpack(val);

  } else {
    ret.emplace_back(unpack(args));
  }
  return ret;
}

static auto getIterators(py::list iterators, mlir::MLIRContext &ctx) {
  llvm::SmallVector<mlir::utils::IteratorType> ret(iterators.size());
  for (auto [i, iter] : llvm::enumerate(iterators)) {
    auto str = iter.cast<std::string>();
    ret[i] = [&]() -> mlir::utils::IteratorType {
      if (str == "parallel")
        return mlir::utils::IteratorType::parallel;

      if (str == "reduction")
        return mlir::utils::IteratorType::reduction;

      imex::reportError(llvm::Twine("Invalid linalg iterator type: ") + str);
    }();
  }

  return ret;
}

static mlir::AffineMapAttr getAffineMapAttr(py::handle obj,
                                            mlir::MLIRContext &ctx) {
  auto str = (llvm::Twine("affine_map<") + obj.cast<std::string>() + ">").str();
  return mlir::parseAttribute(str, &ctx).cast<mlir::AffineMapAttr>();
}

static auto getAffineMaps(py::list maps, mlir::MLIRContext &ctx) {
  llvm::SmallVector<mlir::AffineMap> ret(maps.size());
  for (auto [i, val] : llvm::enumerate(maps))
    ret[i] = getAffineMapAttr(val, ctx).getValue();

  return ret;
}

static auto getGenericOpBodyTypes(mlir::ValueRange inputs,
                                  mlir::ValueRange outputs) {
  llvm::SmallVector<mlir::Type> ret;
  ret.reserve(inputs.size() + outputs.size());
  for (auto r : {inputs, outputs}) {
    for (auto type : r.getTypes()) {
      auto elemType = [&]() {
        if (auto shaped = type.dyn_cast<mlir::ShapedType>())
          return shaped.getElementType();

        return type;
      }();
      ret.emplace_back(elemType);
    }
  }
  return ret;
}

static auto genericOpBodyResultTypes(mlir::ValueRange outputs) {
  llvm::SmallVector<mlir::Type> ret;
  ret.reserve(outputs.size());
  for (auto type : outputs.getTypes()) {
    auto elemType = type.cast<mlir::ShapedType>().getElementType();
    ret.emplace_back(elemType);
  }
  return ret;
}

static py::object broadcastImpl(py::capsule context, py::tuple args,
                                py::handle resultType) {
  auto &ctx = getPyContext(context);
  auto loc = ctx.loc;
  auto &builder = ctx.builder;

  llvm::SmallVector<mlir::Value> mlirArgs(args.size());

  int64_t rank = -1;
  for (auto [i, obj] : llvm::enumerate(args)) {
    auto val =
        toNTensor(loc, builder, ctx.context.unwrapVal(loc, builder, obj));

    auto srcType = val.getType();
    if (!srcType.isa<imex::ntensor::NTensorType>()) {
      if (!imex::ntensor::NTensorType::isValidElementType(srcType))
        imex::reportError(llvm::Twine("broadcast: invalid source type ") +
                          toStr(srcType));

      auto retType = imex::ntensor::NTensorType::get(std::nullopt, srcType);
      val = builder.create<imex::ntensor::FromElementsOp>(loc, retType, val);
    } else {
      rank = std::max(rank, srcType.cast<mlir::ShapedType>().getRank());
    }

    mlirArgs[i] = val;
  }

  llvm::SmallVector<mlir::Type> resTypes(args.size());
  llvm::SmallVector<int64_t> resShape(rank >= 0 ? static_cast<size_t>(rank) : 0,
                                      mlir::ShapedType::kDynamic);
  for (auto [i, arg] : llvm::enumerate(mlirArgs))
    resTypes[i] = arg.getType().cast<mlir::ShapedType>().clone(resShape);

  auto broadcast =
      builder.create<imex::ntensor::BroadcastOp>(loc, resTypes, mlirArgs);

  llvm::SmallVector<mlir::Value> results(broadcast->getNumResults());
  if (!resultType.is_none()) {
    auto resType = unwrapType(resultType);
    for (auto [i, res] :
         llvm::enumerate(mlir::ValueRange(broadcast.getResults()))) {
      auto srcType = res.getType().cast<mlir::ShapedType>();
      auto dstType = srcType.clone(resType);
      if (srcType != dstType) {
        if (!imex::canConvert(srcType.getElementType(), resType))
          imex::reportError(llvm::Twine("Cannont convert from ") +
                            toStr(srcType.getElementType()) + " to " +
                            toStr(resType));

        auto bodyBuilder = [&](mlir::OpBuilder &b, mlir::Location l,
                               mlir::ValueRange vals) {
          assert(vals.size() == 1);
          auto res = imex::doConvert(b, l, vals.front(), resType);
          assert(res);
          b.create<imex::ntensor::ElementwiseYieldOp>(l, res);
        };

        res = builder
                  .create<imex::ntensor::ElementwiseOp>(loc, dstType, res,
                                                        bodyBuilder)
                  .getResult(0);
      }
      results[i] = res;
    }
  } else {
    auto res = broadcast.getResults();
    results.assign(res.begin(), res.end());
  }

  py::tuple ret(results.size());
  for (auto [i, res] : llvm::enumerate(results)) {
    auto srcType = res.getType().cast<mlir::ShapedType>();
    auto dstType = mlir::RankedTensorType::get(srcType.getShape(),
                                               srcType.getElementType());
    if (rank >= 0) {
      res = builder.create<imex::ntensor::ToTensorOp>(loc, dstType, res);
    } else {
      res = builder.create<imex::ntensor::LoadOp>(loc, res);
    }
    ret[i] = ctx.context.createVar(context, res);
  }

  return std::move(ret);
}

static py::object initTensorImpl(py::capsule context, py::iterable shape,
                                 py::handle dtype, py::handle initVal) {
  auto &ctx = getPyContext(context);
  auto loc = ctx.loc;
  auto &builder = ctx.builder;
  auto elemType = unwrapType(dtype);

  auto indexType = builder.getIndexType();
  auto count = py::len(shape);
  llvm::SmallVector<mlir::Value> dynamicShape;
  auto staticShape = getDynShape(count);
  for (auto [i, elem] : llvm::enumerate(shape)) {
    auto elemVal = ctx.context.unwrapVal(loc, builder, elem, indexType);
    if (auto constVal = mlir::getConstantIntValue(elemVal)) {
      staticShape[i] = *constVal;
    } else {
      dynamicShape.emplace_back(elemVal);
    }
  }

  mlir::Value init;
  if (!initVal.is_none()) {
    init = doCast(builder, loc, ctx.context.unwrapVal(loc, builder, initVal),
                  elemType);
  }

  auto resType = imex::ntensor::NTensorType::get(staticShape, elemType);
  mlir::Value res = builder.create<imex::ntensor::CreateArrayOp>(
      loc, resType, dynamicShape, init);
  return ctx.context.createVar(context, res);
}

static py::object fillTensorImpl(py::capsule context, py::handle tensor,
                                 py::handle value) {
  auto &ctx = getPyContext(context);
  auto loc = ctx.loc;
  auto &builder = ctx.builder;
  auto tensorVal = ctx.context.unwrapVal(loc, builder, tensor);
  auto tensorType = tensorVal.getType().cast<mlir::ShapedType>();
  auto initVal = ctx.context.unwrapVal(loc, builder, value);
  initVal = doCast(builder, loc, initVal, tensorType.getElementType());

  auto rank = static_cast<unsigned>(tensorType.getRank());
  mlir::AffineMap affine_maps[] = {
      mlir::AffineMap::getMultiDimIdentityMap(rank, builder.getContext()),
  };
  llvm::SmallVector<mlir::utils::IteratorType> iterators(
      rank, mlir::utils::IteratorType::parallel);
  auto body = [&](mlir::OpBuilder &builder, mlir::Location loc,
                  mlir::ValueRange values) {
    assert(values.size() == 1);
    builder.create<mlir::linalg::YieldOp>(loc, initVal);
  };
  auto val = builder.create<mlir::linalg::GenericOp>(
      loc, tensorType, std::nullopt, tensorVal, affine_maps, iterators, body);
  return ctx.context.createVar(context, val.getResult(0));
}

static py::object genericImpl(py::capsule context, py::handle inputs,
                              py::handle outputs, py::list iterators,
                              py::list maps, py::handle body) {
  auto &ctx = getPyContext(context);
  auto loc = ctx.loc;
  auto &builder = ctx.builder;
  auto &mlirContext = *builder.getContext();

  auto unpack = [&](py::handle obj) -> mlir::Value {
    return toTensor(loc, builder, ctx.context.unwrapVal(loc, builder, obj));
  };

  auto inputsArgs = getAgrsFromTuple(inputs, unpack);
  auto outputArgs = getAgrsFromTuple(outputs, unpack);
  auto mlirIterators = getIterators(iterators, mlirContext);

  auto bodyTypes = getGenericOpBodyTypes(inputsArgs, outputArgs);
  auto funcTypes = mapTypesToNumbaChecked(ctx.context.typesMod, bodyTypes);
  auto bodyFunc = ctx.context.compileBody(body, funcTypes);

  auto castValues = [&](mlir::ValueRange vals, mlir::TypeRange types) {
    assert(vals.size() == types.size());
    llvm::SmallVector<mlir::Value> ret(vals.size());
    for (auto [i, val] : llvm::enumerate(vals))
      ret[i] = doCast(builder, loc, val, types[i]);

    return ret;
  };

  auto affineMaps = getAffineMaps(maps, mlirContext);
  auto bodyBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::ValueRange args) {
    auto funcType = bodyFunc.getFunctionType();
    auto newArgs = castValues(args, funcType.getInputs());
    auto call = builder.create<mlir::func::CallOp>(loc, bodyFunc, newArgs);
    auto newResults =
        castValues(call.getResults(), genericOpBodyResultTypes(outputArgs));
    builder.create<mlir::linalg::YieldOp>(loc, newResults);
  };

  auto retTypes = getTypes(outputArgs);

  auto genericOp = builder.create<mlir::linalg::GenericOp>(
      loc, retTypes, inputsArgs, outputArgs, affineMaps, mlirIterators,
      bodyBuilder);
  return ctx.context.wrapResult(context, genericOp.getResults());
}

static py::object indexImpl(py::capsule context, py::int_ dimObj) {
  auto &ctx = getPyContext(context);
  auto loc = ctx.loc;
  auto &builder = ctx.builder;
  auto val = static_cast<int64_t>(dimObj);
  if (val < 0)
    imex::reportError("Index cannot be negative");

  auto dimVal =
      builder.create<mlir::linalg::IndexOp>(loc, static_cast<uint64_t>(val));
  auto resType = builder.getIntegerType(64, /*signed*/ true);
  auto res = doCast(builder, loc, dimVal, resType);
  return ctx.context.wrapResult(context, res);
}

static py::object fromElementsImpl(py::capsule context, py::handle values,
                                   py::handle dtype) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  mlir::Type type;
  if (!dtype.is_none())
    type = unwrapType(dtype);

  llvm::SmallVector<mlir::Value> vals(containerSize(values));
  containerIterate(values, [&](auto index, py::handle obj) {
    auto val = [&]() -> mlir::Value {
      if (type)
        return ctx.context.unwrapVal(loc, builder, obj, type);

      auto v = ctx.context.unwrapVal(loc, builder, obj);
      type = v.getType();
      return v;
    }();
    vals[index] = val;
  });

  if (vals.empty())
    imex::reportError("Invalid from_elemets size");

  auto resTensorType =
      mlir::RankedTensorType::get(mlir::ShapedType::kDynamic, type);
  for (auto &val : vals)
    val = doCast(builder, loc, val, type);

  auto res =
      builder.create<mlir::tensor::FromElementsOp>(loc, vals).getResult();
  res = builder.createOrFold<mlir::tensor::CastOp>(loc, resTensorType, res);

  return ctx.context.createVar(context, res);
}

static py::object extractImpl(py::capsule context, py::handle value,
                              py::handle indices) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;

  auto indexType = builder.getIndexType();
  llvm::SmallVector<mlir::Value> ind(containerSize(indices));
  containerIterate(indices, [&](auto index, py::handle obj) {
    ind[index] = ctx.context.unwrapVal(loc, builder, obj, indexType);
  });
  auto tensor =
      toTensor(loc, builder, ctx.context.unwrapVal(loc, builder, value));
  auto tensorType = tensor.getType().dyn_cast<mlir::RankedTensorType>();
  if (!tensorType)
    imex::reportError(llvm::Twine("extract: invalid source type ") +
                      toStr(tensor.getType()));

  auto res =
      builder.create<mlir::tensor::ExtractOp>(loc, tensor, ind).getResult();
  return ctx.context.createVar(context, res);
}

static py::object reshapeImpl(py::capsule context, py::handle src,
                              py::handle newDims) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;

  auto unwrapVal = [&](py::handle obj) {
    return ctx.context.unwrapVal(loc, builder, obj);
  };
  auto dimType = builder.getIndexType();
  auto unwrapDim = [&](py::handle obj) {
    return ctx.context.unwrapVal(loc, builder, obj, dimType);
  };

  auto srcVal = toTensor(loc, builder, unwrapVal(src));
  auto srcType = srcVal.getType().dyn_cast<mlir::RankedTensorType>();
  if (!srcType)
    imex::reportError(llvm::Twine("invalid reshape argument: ") +
                      toStr(srcVal.getType()));

  auto newDimsVals = [&]() {
    auto dimCast = [&](mlir::Value val) {
      return doCast(builder, loc, val, dimType);
    };
    llvm::SmallVector<mlir::OpFoldResult> ret;
    if (py::isinstance<py::tuple>(newDims)) {
      auto t = newDims.cast<py::tuple>();
      ret.resize(t.size());
      for (auto [i, val] : llvm::enumerate(t))
        ret[i] = unwrapDim(val);

      return ret;
    }
    auto dims = unwrapVal(newDims);
    if (auto tupleType = dims.getType().dyn_cast<mlir::TupleType>()) {
      auto dimsCount = tupleType.size();
      ret.resize(dimsCount);
      for (size_t i = 0; i < dimsCount; ++i) {
        auto ind = builder.create<mlir::arith::ConstantIndexOp>(loc, i);
        auto elemType = tupleType.getType(i);
        auto item = builder.createOrFold<imex::util::TupleExtractOp>(
            loc, elemType, dims, ind);
        ret[i] = dimCast(item);
      }
    } else {
      ret.emplace_back(dimCast(dims));
    }
    return ret;
  }();

  auto srcRank = static_cast<unsigned>(srcType.getRank());
  auto dstRank = static_cast<unsigned>(newDimsVals.size());

  if (srcRank == 1 && dstRank == 1) {
    mlir::OpFoldResult offset = builder.getIndexAttr(0);
    mlir::OpFoldResult size = newDimsVals.front();
    mlir::OpFoldResult stride = builder.getIndexAttr(1);

    mlir::Value slice = builder.create<mlir::tensor::ExtractSliceOp>(
        loc, srcVal, offset, size, stride);
    return ctx.context.createVar(context, slice);
  }

  auto isUnitDim = [](mlir::OpFoldResult v) {
    if (auto intVal = mlir::getConstantIntValue(v))
      return *intVal == 1;

    return false;
  };

  auto unitDimsCount = [&]() {
    unsigned ret = 0;
    for (auto v : newDimsVals)
      if (isUnitDim(v))
        ++ret;
    return ret;
  }();

  auto shape = getDynShape(dstRank);

  auto resultType = srcType.clone(shape);

  // TODO: Limit to 1D case for now
  if ((srcRank == 1) && (dstRank == (srcRank + unitDimsCount)) &&
      (unitDimsCount != 0)) {
    llvm::SmallVector<mlir::ReassociationIndices> reassoc(srcRank);
    llvm::SmallVector<int64_t> expandShape = shape;
    int currInd = -1;
    for (auto i : llvm::seq(0u, dstRank)) {
      if (!isUnitDim(newDimsVals[i])) {
        ++currInd;
      } else {
        expandShape[i] = 1;
      }

      reassoc[std::max(0, currInd)].emplace_back(i);
    }

    shape.resize(static_cast<unsigned>(srcType.getRank()),
                 mlir::ShapedType::kDynamic);
    auto dynShapeType = srcType.clone(shape);
    if (dynShapeType != srcType)
      srcVal = builder.create<mlir::tensor::CastOp>(loc, dynShapeType, srcVal);

    auto expandType = resultType.clone(expandShape);
    mlir::Value res = builder.create<mlir::tensor::ExpandShapeOp>(
        loc, expandType, srcVal, reassoc);
    if (expandType != resultType)
      res = builder.create<mlir::tensor::CastOp>(loc, resultType, res);

    return ctx.context.createVar(context, res);
  }

  auto toValues = [&](mlir::ArrayRef<mlir::OpFoldResult> src) {
    auto size = src.size();
    llvm::SmallVector<mlir::Value> values(size);
    for (auto i : llvm::seq<size_t>(0, size)) {
      auto v = src[i];
      if (auto val = v.dyn_cast<mlir::Value>()) {
        values[i] = val;
      } else {
        auto constVal = v.get<mlir::Attribute>()
                            .cast<mlir::IntegerAttr>()
                            .getValue()
                            .getSExtValue();
        values[i] = builder.create<mlir::arith::ConstantIndexOp>(loc, constVal);
      }
    }
    return values;
  };

  auto shapeTensor =
      builder.create<mlir::tensor::FromElementsOp>(loc, toValues(newDimsVals));

  mlir::Value reshaped = builder.create<mlir::tensor::ReshapeOp>(
      loc, resultType, srcVal, shapeTensor);

  return ctx.context.createVar(context, reshaped);
}

static py::object externalCallImpl(py::capsule context, py::str funcName,
                                   py::handle inputs, py::handle outputs,
                                   py::bool_ decorate, py::bool_ returnTensor,
                                   py::handle attrs) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto unwrapVal = [&](py::handle obj) {
    auto tensor =
        toNTensor(loc, builder, ctx.context.unwrapVal(loc, builder, obj));
    auto tensorType = tensor.getType().dyn_cast<imex::ntensor::NTensorType>();
    if (tensorType) {
      auto commonType = imex::ntensor::NTensorType::get(
          getDynShape(tensorType.getRank()), tensorType.getElementType());
      if (commonType != tensorType)
        tensor = builder.create<imex::ntensor::CastOp>(loc, commonType, tensor);
    }
    return tensor;
  };
  auto inputVals = toValues(inputs, unwrapVal);
  auto outputVals = toValues(outputs, unwrapVal);

  inputVals.reserve(inputVals.size() + outputVals.size());

  llvm::SmallVector<mlir::Type, 1> retTypes;
  for (auto val : outputVals) {
    auto type = val.getType();
    auto shapedType = type.dyn_cast<mlir::ShapedType>();
    if (!returnTensor && shapedType) {
      if (!shapedType.isa<mlir::MemRefType>())
        val = toMemref(loc, builder, val);

      inputVals.emplace_back(val);
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
    auto f = mod.lookupSymbol<mlir::func::FuncOp>(name);
    if (f) {
      auto existingType = f.getFunctionType();
      if (existingType != funcType) {
        imex::reportError(llvm::Twine("linalg_builder::external_call: "
                                      "invalid function redefinition: ") +
                          name + " " + toStr(existingType) + " and " +
                          toStr(funcType));
      }
    } else {
      f = imex::addFunction(builder, mod, name, funcType);
      if (decorate)
        f->setAttr("llvm.emit_c_interface",
                   mlir::UnitAttr::get(builder.getContext()));

      if (!attrs.is_none()) {
        auto attrDict = attrs.cast<py::dict>();
        for (auto [name, val] : attrDict) {
          auto nameStr = name.cast<std::string>();
          auto attrVal = [&, v = val]() -> mlir::Attribute {
            if (py::isinstance<py::int_>(v))
              return builder.getI64IntegerAttr(v.cast<int64_t>());

            if (py::isinstance<py::float_>(v))
              return builder.getF64FloatAttr(v.cast<double>());

            if (py::isinstance<py::str>(v))
              return builder.getStringAttr(v.cast<std::string>());

            imex::reportError(llvm::Twine("Unsupported attr type: ") +
                              py::str(v).cast<std::string>());
          }();

          f->setAttr(nameStr, attrVal);
        }
      }
    }
    return f;
  }();

  auto call = builder.create<mlir::func::CallOp>(loc, func, inputVals);

  auto res = call->getResults();

  llvm::SmallVector<mlir::Value> results;
  results.reserve(outputVals.size() + res.size());

  for (auto [i, val] : llvm::enumerate(
           llvm::ArrayRef(inputVals).take_back(outputVals.size()))) {
    mlir::Value newVal = val;
    if (!returnTensor && outputVals[i].getType().isa<mlir::ShapedType>()) {
      if (newVal.getType().isa<mlir::MemRefType>())
        newVal = builder.create<mlir::bufferization::ToTensorOp>(loc, newVal);

      assert(newVal.getType().isa<mlir::TensorType>());
      results.emplace_back(newVal);
    }
  }

  results.append(res.begin(), res.end());

  if (results.empty())
    return py::none();

  if (results.size() == 1)
    return ctx.context.createVar(context, results.front());

  py::tuple ret(results.size());
  for (auto [i, val] : llvm::enumerate(results))
    ret[i] = ctx.context.createVar(context, val);

  return std::move(ret);
}

static py::object insertImpl(py::capsule context, py::handle src,
                             py::handle dst, py::handle offsets,
                             py::handle strides) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto unwrapVal = [&](py::handle obj) {
    return toTensor(loc, builder, ctx.context.unwrapVal(loc, builder, obj));
  };
  auto indexType = builder.getIndexType();
  auto unwrapIndex = [&](py::handle obj) {
    return ctx.context.unwrapVal(loc, builder, obj, indexType);
  };
  auto unwrapList = [&](py::handle obj) {
    auto len = py::len(obj);
    llvm::SmallVector<mlir::Value> res(len);
    for (auto [i, val] : llvm::enumerate(obj))
      res[i] = unwrapIndex(val);

    return res;
  };
  auto srcTensor = unwrapVal(src);
  auto dstTensor = unwrapVal(dst);
  auto offsetsVec = unwrapList(offsets);
  auto stridesVec = unwrapList(strides);

  llvm::SmallVector<mlir::Value> sizesVec(offsetsVec.size());
  for (auto i : llvm::seq<size_t>(0, sizesVec.size()))
    sizesVec[i] = builder.createOrFold<mlir::tensor::DimOp>(loc, srcTensor, i);

  auto srcShapedType = srcTensor.getType().cast<mlir::ShapedType>();
  auto rank = static_cast<unsigned>(srcShapedType.getRank());
  auto dynShape = getDynShape(rank);
  auto newShapedType = srcShapedType.clone(dynShape);
  if (srcShapedType != newShapedType)
    srcTensor =
        builder.create<mlir::tensor::CastOp>(loc, newShapedType, srcTensor);

  auto res =
      builder
          .create<mlir::tensor::InsertSliceOp>(loc, srcTensor, dstTensor,
                                               offsetsVec, sizesVec, stridesVec)
          .getResult();
  return ctx.context.createVar(context, res);
}

static py::object inlineFuncImpl(py::capsule context, py::handle func,
                                 py::handle retType, py::tuple args) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;

  auto argsValues = [&]() {
    auto unwrapVal = [&](py::handle obj) {
      return ctx.context.unwrapVal(loc, builder, obj);
    };
    llvm::SmallVector<mlir::Value> ret(args.size());
    for (auto [i, val] : llvm::enumerate(args))
      ret[i] = unwrapVal(val);

    return ret;
  }();
  auto funcTypes =
      mapTypesToNumbaChecked(ctx.context.typesMod, getTypes(argsValues));
  auto bodyFunc = ctx.context.compileBody(func, funcTypes);
  auto funcType = bodyFunc.getFunctionType();
  auto funcArgsTypes = funcType.getInputs();
  if (funcArgsTypes.size() != argsValues.size())
    imex::reportError(
        llvm::Twine("Invalid function arguments count, expected ") +
        llvm::Twine(argsValues.size()) + ", got" +
        llvm::Twine(funcArgsTypes.size()));

  if (funcType.getNumResults() != 1)
    imex::reportError(llvm::Twine("Invalid number of return values: ") +
                      llvm::Twine(funcType.getNumResults()));

  auto castValues = [&](mlir::ValueRange vals, mlir::TypeRange types) {
    assert(vals.size() == types.size());
    llvm::SmallVector<mlir::Value> ret(vals.size());
    for (auto [i, val] : llvm::enumerate(vals)) {
      auto index = static_cast<unsigned>(i);
      ret[index] = doCast(builder, loc, val, types[index]);
    }
    return ret;
  };

  auto castedArgs = castValues(argsValues, funcArgsTypes);

  auto resValue = builder.create<mlir::func::CallOp>(loc, bodyFunc, castedArgs)
                      .getResult(0);

  auto mlirRetType = unwrapType(retType);
  resValue = doCast(builder, loc, resValue, mlirRetType);
  return ctx.context.createVar(context, resValue);
}

static py::object castImpl(py::capsule context, py::handle src,
                           py::handle dtype) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto unwrapVal = [&](py::handle obj) {
    return toTensor(loc, builder, ctx.context.unwrapVal(loc, builder, obj));
  };
  auto val = unwrapVal(src);
  auto type = unwrapType(dtype);
  auto ret = doCast(builder, loc, val, type);
  return ctx.context.createVar(context, ret);
}

static py::object undefImpl(py::capsule context, py::handle dtype) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto type = unwrapType(dtype);
  auto ret = builder.createOrFold<imex::util::UndefOp>(loc, type);
  return ctx.context.createVar(context, ret);
}

static py::object subviewImpl(py::capsule context, py::handle src,
                              py::handle offsets, py::handle sizes,
                              py::handle strides, py::handle rank) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;

  auto unwrapVal = [&](py::handle obj) {
    return toTensor(loc, builder, ctx.context.unwrapVal(loc, builder, obj));
  };

  auto srcVal = unwrapVal(src);
  auto srcType = srcVal.getType().cast<mlir::RankedTensorType>();

  auto indexType = builder.getIndexType();
  auto indexCast = [&](mlir::Value val) -> mlir::OpFoldResult {
    while (auto parent = val.getDefiningOp<imex::util::SignCastOp>())
      val = parent.getSource();

    if (auto constVal = mlir::getConstantIntValue(val))
      return builder.getIndexAttr(*constVal);

    return doCast(builder, loc, val, indexType);
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
        ret[i] = indexCast(builder.createOrFold<imex::util::TupleExtractOp>(
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
            loc, srcVal, static_cast<int64_t>(i));
        auto offset = [&]() -> mlir::Value {
          auto off = offsetVals[i];
          if (off.is<mlir::Value>())
            return off.get<mlir::Value>();

          auto val = off.get<mlir::Attribute>()
                         .cast<mlir::IntegerAttr>()
                         .getValue()
                         .getSExtValue();
          return builder.create<mlir::arith::ConstantIndexOp>(loc, val);
        }();
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
  auto viewType = [&]() -> mlir::RankedTensorType {
    if (rank.is_none()) {
      return mlir::tensor::ExtractSliceOp::inferResultType(srcType, offsetVals,
                                                           sizeVals, strideVals)
          .cast<mlir::RankedTensorType>();
    } else {
      auto rankVal = rank.cast<unsigned>();
      return mlir::tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                 rankVal, srcType, offsetVals, sizeVals, strideVals)
          .cast<mlir::RankedTensorType>();
    }
  }();
  auto view = builder.createOrFold<mlir::tensor::ExtractSliceOp>(
      loc, viewType, srcVal, offsetVals, sizeVals, strideVals);

  auto resType = view.getType().cast<mlir::ShapedType>();
  auto resDynamicType = resType.clone(getDynShape(resType.getRank()));
  if (resDynamicType != resType)
    view = builder.create<mlir::tensor::CastOp>(loc, resDynamicType, view);
  return ctx.context.createVar(context, view);
}

static py::object forceCopyImpl(py::capsule context, py::handle src) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto srcVal =
      toNTensor(loc, builder, ctx.context.unwrapVal(loc, builder, src));

  auto valType = srcVal.getType().cast<imex::ntensor::NTensorType>();
  llvm::SmallVector<mlir::Value> dynamicSizes;
  for (auto [i, dim] : llvm::enumerate(valType.getShape())) {
    if (mlir::ShapedType::isDynamic(dim)) {
      mlir::Value dimVal = builder.create<imex::ntensor::DimOp>(
          loc, srcVal, static_cast<int64_t>(i));
      dynamicSizes.emplace_back(dimVal);
    }
  }

  mlir::Value res = builder.create<imex::ntensor::CreateArrayOp>(
      loc, valType, dynamicSizes, /*initVal*/ mlir::Value{});
  builder.create<imex::ntensor::CopyOp>(loc, srcVal, res);
  return ctx.context.createVar(context, res);
}

static py::object selectImpl(py::capsule context, py::handle cond,
                             py::handle trueV, py::handle falseV) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto condVal = ctx.context.unwrapVal(loc, builder, cond);
  auto trueVal = ctx.context.unwrapVal(loc, builder, trueV);
  auto falseVal = ctx.context.unwrapVal(loc, builder, falseV);
  auto res =
      builder.create<mlir::arith::SelectOp>(loc, condVal, trueVal, falseVal);
  return ctx.context.createVar(context, res);
}

static py::object arrayTypeImpl(py::capsule context, py::iterable dims,
                                py::handle dtype) {
  auto &ctx = getPyContext(context);
  auto elemType = unwrapType(dtype);
  llvm::SmallVector<int64_t> shape(py::len(dims));
  for (auto [i, val] : llvm::enumerate(dims))
    shape[i] = val.cast<int64_t>();

  auto arrayType = mlir::RankedTensorType::get(shape, elemType);
  return ctx.context.createType(arrayType);
}

static void
setupPyBuilder(py::handle builder, mlir::OpBuilder &b,
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
  py::setattr(builder, "_force_copy", py::cpp_function(&forceCopyImpl));
  py::setattr(builder, "_select", py::cpp_function(&selectImpl));

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

  addType("complex64", mlir::ComplexType::get(b.getF32Type()));
  addType("complex128", mlir::ComplexType::get(b.getF64Type()));
}

static py::object shapeImpl(py::capsule context, py::capsule ssaVal) {
  auto &ctx = getPyContext(context);
  auto value = unwrapMlir<mlir::Value>(ssaVal);
  if (auto mlirType = value.getType().dyn_cast<mlir::ShapedType>()) {
    if (!mlirType.hasRank())
      imex::reportError("Unranked shaped are not supported");

    bool isTensor = mlirType.isa<mlir::TensorType>();
    bool isNTensor = mlirType.isa<imex::ntensor::NTensorType>();
    auto &builder = ctx.builder;
    auto loc = ctx.loc;
    auto rank = static_cast<unsigned>(mlirType.getRank());
    llvm::SmallVector<mlir::Value> shapeVals(rank);
    for (auto i : llvm::seq(0u, rank)) {
      mlir::Value mlirDim;
      if (isTensor) {
        mlirDim = builder.create<mlir::tensor::DimOp>(loc, value, i);
      } else if (isNTensor) {
        mlirDim = builder.create<imex::ntensor::DimOp>(loc, value, i);
      } else {
        mlirDim = builder.create<mlir::memref::DimOp>(loc, value, i);
      }
      shapeVals[i] = mlirDim;
    }
    llvm::SmallVector<mlir::Type> shapeTypes(rank, builder.getIndexType());
    auto shapeType = mlir::TupleType::get(builder.getContext(), shapeTypes);
    auto shapeVar =
        builder.create<imex::util::BuildTupleOp>(loc, shapeType, shapeVals);
    return ctx.context.createVar(context, shapeVar.getResult());
  }
  return py::tuple();
}

static py::object stridesImpl(py::capsule context, py::capsule ssaVal) {
  auto &ctx = getPyContext(context);
  auto value = unwrapMlir<mlir::Value>(ssaVal);
  if (auto mlirType = value.getType().dyn_cast<mlir::ShapedType>()) {
    if (!mlirType.hasRank())
      imex::reportError("Unranked strides are not supported");

    auto &builder = ctx.builder;
    auto loc = ctx.loc;

    auto memref = toMemref(loc, builder, value, /*strided*/ true);

    auto metadata =
        builder.create<mlir::memref::ExtractStridedMetadataOp>(loc, memref);

    auto rank = static_cast<unsigned>(mlirType.getRank());
    llvm::SmallVector<mlir::Type> shapeTypes(rank, builder.getIndexType());
    auto shapeType = mlir::TupleType::get(builder.getContext(), shapeTypes);
    auto shapeVar = builder.create<imex::util::BuildTupleOp>(
        loc, shapeType, metadata.getStrides());
    return ctx.context.createVar(context, shapeVar.getResult());
  }
  return py::tuple();
}

static py::object dtypeImpl(py::capsule context, py::capsule ssaVal) {
  auto &ctx = getPyContext(context);
  auto value = unwrapMlir<mlir::Value>(ssaVal);
  auto type = value.getType();
  if (auto shapedType = type.dyn_cast<mlir::ShapedType>())
    type = shapedType.getElementType();

  return ctx.context.createType(type);
}

static py::object typeImpl(py::capsule context, py::capsule ssaVal) {
  auto &ctx = getPyContext(context);
  auto value = unwrapMlir<mlir::Value>(ssaVal);
  auto type = value.getType();
  return ctx.context.createType(type);
}

static py::object lenImpl(py::capsule /*context*/, py::capsule ssaVal) {
  auto value = unwrapMlir<mlir::Value>(ssaVal);
  auto type = value.getType();
  if (auto tupleType = type.dyn_cast<mlir::TupleType>())
    return py::int_(tupleType.size());

  return py::none();
}

static py::object getitemImpl(py::capsule context, py::capsule ssaVal,
                              py::handle index) {
  auto &ctx = getPyContext(context);
  auto value = unwrapMlir<mlir::Value>(ssaVal);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto type = value.getType();
  if (auto shaped = type.dyn_cast<mlir::ShapedType>()) {
    auto indexVal = ctx.context.unwrapVal(loc, builder, index);
    auto elemType = shaped.getElementType();

    mlir::Value array;
    auto ntensorType = imex::ntensor::NTensorType::get(shaped.getShape(),
                                                       shaped.getElementType());
    if (shaped.isa<mlir::MemRefType>()) {
      array =
          builder.create<imex::ntensor::FromMemrefOp>(loc, ntensorType, value);
    } else if (shaped.isa<mlir::RankedTensorType>()) {
      array =
          builder.create<imex::ntensor::FromTensorOp>(loc, ntensorType, value);
    } else if (shaped.isa<imex::ntensor::NTensorType>()) {
      array = value;
    } else {
      throw py::index_error("Invalid shaped type");
    }
    auto res = builder.create<imex::ntensor::GetitemOp>(loc, elemType, array,
                                                        indexVal);
    return ctx.context.createVar(context, res.getResult());
  }

  auto indexVal = index.cast<int64_t>();
  if (auto tupleType = type.dyn_cast<mlir::TupleType>()) {
    auto maxIndex = static_cast<int64_t>(tupleType.size());
    if (indexVal < 0 || indexVal >= maxIndex)
      throw py::index_error(("Invalid getitem index: " + llvm::Twine(indexVal) +
                             ", expected [0:" + llvm::Twine(maxIndex) + ")")
                                .str());

    if (auto parentOp = value.getDefiningOp<imex::util::BuildTupleOp>())
      return ctx.context.createVar(
          context, parentOp.getOperand(static_cast<unsigned>(indexVal)));

    auto elemType = tupleType.getType(static_cast<size_t>(indexVal));
    auto ind = builder.create<mlir::arith::ConstantIndexOp>(loc, indexVal);
    auto item =
        builder.create<imex::util::TupleExtractOp>(loc, elemType, value, ind);
    return ctx.context.createVar(context, item.getResult());
  } else {
    throw py::index_error("Invalid getitem");
  }
}

template <typename Op>
static mlir::Value binopFunc(mlir::Location loc, mlir::OpBuilder &builder,
                             mlir::Value lhs, mlir::Value rhs) {
  auto lhsVar = doSignCast(builder, loc, lhs);
  auto rhsVar = doSignCast(builder, loc, rhs);
  auto res = builder.create<Op>(loc, lhsVar, rhsVar);
  return doSignCast(builder, loc, res, lhs.getType());
}

template <typename Op>
static mlir::Value rbinopFunc(mlir::Location loc, mlir::OpBuilder &builder,
                              mlir::Value lhs, mlir::Value rhs) {
  auto lhsVar = doSignCast(builder, loc, lhs);
  auto rhsVar = doSignCast(builder, loc, rhs);
  auto res = builder.create<Op>(loc, rhsVar, lhsVar);
  return doSignCast(builder, loc, res, lhs.getType());
}

static mlir::Value binopFuncIdiv(mlir::Location loc, mlir::OpBuilder &builder,
                                 mlir::Value lhs, mlir::Value rhs) {
  auto lhsVar = doCast(builder, loc, lhs, builder.getF64Type());
  auto rhsVar = doCast(builder, loc, rhs, builder.getF64Type());
  return builder.create<mlir::arith::DivFOp>(loc, lhsVar, rhsVar);
}

static mlir::Value binopFFloorDiv(mlir::Location loc, mlir::OpBuilder &builder,
                                  mlir::Value lhs, mlir::Value rhs) {
  auto lhsVar = doCast(builder, loc, lhs, builder.getF64Type());
  auto rhsVar = doCast(builder, loc, rhs, builder.getF64Type());
  auto res = builder.create<mlir::arith::DivFOp>(loc, lhsVar, rhsVar);
  return builder.create<mlir::math::FloorOp>(loc, res);
}

template <mlir::arith::CmpIPredicate Pred>
static mlir::Value binopCmpI(mlir::Location loc, mlir::OpBuilder &builder,
                             mlir::Value lhs, mlir::Value rhs) {
  assert(lhs.getType() == rhs.getType());
  lhs = doSignCast(builder, loc, lhs);
  rhs = doSignCast(builder, loc, rhs);
  return builder.create<mlir::arith::CmpIOp>(loc, Pred, lhs, rhs);
}

template <mlir::arith::CmpFPredicate Pred>
static mlir::Value binopCmpF(mlir::Location loc, mlir::OpBuilder &builder,
                             mlir::Value lhs, mlir::Value rhs) {
  return builder.create<mlir::arith::CmpFOp>(loc, Pred, lhs, rhs);
}

static py::object binopImpl(py::capsule context, py::capsule ssaVal,
                            py::handle rhs, py::str op) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto lhs = unwrapMlir<mlir::Value>(ssaVal);

  auto type = lhs.getType();
  if (!type.isa<mlir::IntegerType, mlir::IndexType, mlir::FloatType,
                mlir::ShapedType>())
    imex::reportError("Invalid binop arg type");

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
      {"//", &binopFunc<mlir::arith::DivSIOp>, &binopFFloorDiv},

      {"lt", &binopCmpI<mlir::arith::CmpIPredicate::slt>,
       &binopCmpF<mlir::arith::CmpFPredicate::OLT>},
      {"le", &binopCmpI<mlir::arith::CmpIPredicate::sle>,
       &binopCmpF<mlir::arith::CmpFPredicate::OLE>},
      {"gt", &binopCmpI<mlir::arith::CmpIPredicate::sgt>,
       &binopCmpF<mlir::arith::CmpFPredicate::OGT>},
      {"ge", &binopCmpI<mlir::arith::CmpIPredicate::sge>,
       &binopCmpF<mlir::arith::CmpFPredicate::OGE>},
      {"eq", &binopCmpI<mlir::arith::CmpIPredicate::eq>,
       &binopCmpF<mlir::arith::CmpFPredicate::OEQ>},
      {"ne", &binopCmpI<mlir::arith::CmpIPredicate::ne>,
       &binopCmpF<mlir::arith::CmpFPredicate::ONE>},
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
  imex::reportError("Unhandled binop type");
}

static py::object unopImpl(py::capsule context, py::capsule ssaVal,
                           py::str op) {
  auto &ctx = getPyContext(context);
  auto &builder = ctx.builder;
  auto loc = ctx.loc;
  auto val = unwrapMlir<mlir::Value>(ssaVal);

  auto type = val.getType();
  if (!type.isa<mlir::IntegerType, mlir::IndexType, mlir::FloatType>())
    imex::reportError("Invalid unop arg type");

  auto opName = static_cast<std::string>(op);
  mlir::Value res;
  if (opName == "+") {
    res = val;
  } else if (opName == "-") {
    if (type.isa<mlir::FloatType>()) {
      res = builder.create<mlir::arith::NegFOp>(loc, val);
    } else {
      auto signlessType = makeSignlessType(type);
      auto zero = builder.getIntegerAttr(signlessType, 0);
      auto zeroVal = builder.create<mlir::arith::ConstantOp>(loc, zero);
      val = doSignCast(builder, loc, val);
      res = builder.create<mlir::arith::SubIOp>(loc, zeroVal, val);
      res = doSignCast(builder, loc, res, type);
    }
  } else {
    imex::reportError("Unhandled unop type");
  }
  return ctx.context.createVar(context, res);
}

static py::object strImpl(py::capsule /*context*/, py::capsule ssaVal) {
  return py::str("Var: \"" + toStr(unwrapMlir<mlir::Value>(ssaVal)) + "\"");
}

static py::object literalImpl(py::capsule context, py::capsule ssaVal) {
  auto &ctx = getPyContext(context);
  auto value = unwrapMlir<mlir::Value>(ssaVal);
  return ctx.context.createVar(context, value, /*makeLiteral*/ true);
}

static void setupPyVar(pybind11::handle var) {
  py::setattr(var, "_shape", py::cpp_function(&shapeImpl));
  py::setattr(var, "_strides", py::cpp_function(&stridesImpl));
  py::setattr(var, "_dtype", py::cpp_function(&dtypeImpl));
  py::setattr(var, "_type", py::cpp_function(&typeImpl));
  py::setattr(var, "_len", py::cpp_function(&lenImpl));
  py::setattr(var, "_getitem", py::cpp_function(&getitemImpl));
  py::setattr(var, "_binop", py::cpp_function(&binopImpl));
  py::setattr(var, "_unop", py::cpp_function(&unopImpl));
  py::setattr(var, "_str", py::cpp_function(&strImpl));
  py::setattr(var, "_literal", py::cpp_function(&literalImpl));
}

static PyLinalgResolver::Values unpackResults(PyBuilderContext &ctx,
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
    for (auto [i, val] : llvm::enumerate(tuple))
      vals[i] = unwrapVal(val);

    mlir::ValueRange vr(vals);

    auto tupleType = mlir::TupleType::get(builder.getContext(), vr.getTypes());
    ret.emplace_back(
        builder.create<imex::util::BuildTupleOp>(loc, tupleType, vr));
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
  context->typesMod = py::module::import("numba.core.types");
  context->compileFunc = builderMod.attr("compile_func");
  context->lookupFunc = registry.attr("lookup_func");
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

  auto builderFunc = context->lookupFunc(py::str(name.data(), name.size()));
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
