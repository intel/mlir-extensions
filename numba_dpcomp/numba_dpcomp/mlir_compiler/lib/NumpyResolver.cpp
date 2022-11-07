// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "NumpyResolver.hpp"

#include "imex/Dialect/imex_util/Dialect.hpp"

#include <pybind11/pybind11.h>

#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>

namespace py = pybind11;

class NumpyResolver::Impl {
public:
  py::module mod;
  py::object map;
  py::object empty;
  py::object vararg;

  py::object getFuncDesc(llvm::StringRef name) {
    return map(py::str(name.data(), name.size()));
  }
};

NumpyResolver::NumpyResolver(const char *modName, const char *mapName)
    : impl(std::make_unique<Impl>()) {
  impl->mod = py::module::import(modName);
  impl->map = impl->mod.attr(mapName);

  auto inspect = py::module::import("inspect");
  auto param = inspect.attr("Parameter");
  impl->empty = param.attr("empty");
  impl->vararg = param.attr("VAR_POSITIONAL");
}

NumpyResolver::~NumpyResolver() {}

bool NumpyResolver::hasFunc(llvm::StringRef name) const {
  return !impl->getFuncDesc(name).is_none();
}

static llvm::Optional<mlir::Value>
parseDefault(mlir::OpBuilder &builder, mlir::Location loc, py::handle obj) {
  if (py::isinstance<py::int_>(obj)) {
    auto val = obj.cast<int64_t>();
    auto type = builder.getI64Type();
    return builder.create<mlir::arith::ConstantIntOp>(loc, val, type)
        .getResult();
  }
  if (py::isinstance<py::float_>(obj)) {
    auto val = llvm::APFloat(obj.cast<double>());
    auto type = builder.getF64Type();
    return builder.create<mlir::arith::ConstantFloatOp>(loc, val, type)
        .getResult();
  }
  if (py::isinstance<py::none>(obj)) {
    auto type = builder.getNoneType();
    return builder.create<imex::util::UndefOp>(loc, type).getResult();
  }
  if (py::isinstance<py::tuple>(obj)) {
    auto val = obj.cast<py::tuple>();
    llvm::SmallVector<mlir::Value> elems(val.size());
    for (auto [i, elem] : llvm::enumerate(val)) {
      auto elemVal = parseDefault(builder, loc, elem);
      if (!elemVal)
        return llvm::None;

      elems[i] = *elemVal;
    }
    mlir::ValueRange values(elems);
    auto tupleType = builder.getTupleType(values.getTypes());
    return builder.create<imex::util::BuildTupleOp>(loc, tupleType, values)
        .getResult();
  }

  return llvm::None;
}

mlir::LogicalResult
NumpyResolver::resolveFuncArgs(mlir::OpBuilder &builder, mlir::Location loc,
                               llvm::StringRef name, mlir::ValueRange args,
                               mlir::ArrayAttr argsNames,
                               llvm::SmallVectorImpl<mlir::Value> &resultArgs,
                               llvm::SmallVectorImpl<mlir::Value> &outArgs) {
  assert(args.size() == argsNames.size() && "args and names count mismatch");
  auto res = impl->getFuncDesc(name);
  if (res.is_none())
    return mlir::failure();

  llvm::SmallVector<std::pair<mlir::Value, mlir::StringAttr>>
      argsNamesAndValues;
  argsNamesAndValues.reserve(args.size());
  for (auto [val, name] : llvm::zip(args, argsNames))
    argsNamesAndValues.emplace_back(val, name.cast<mlir::StringAttr>());

  auto findArg = [&](llvm::StringRef argName) -> llvm::Optional<mlir::Value> {
    if (argsNamesAndValues.empty())
      return llvm::None;

    for (auto [i, it] : llvm::enumerate(argsNamesAndValues)) {
      auto origArgName = it.second.getValue();

      if (origArgName == argName) {
        auto res = it.first;
        argsNamesAndValues.erase(argsNamesAndValues.begin() + i);
        return res;
      }
    }
    return llvm::None;
  };

  auto outTuple = res.attr("out").cast<py::tuple>();
  llvm::SmallBitVector outProcessed(outTuple.size(), false);
  for (auto [i, out] : llvm::enumerate(outTuple)) {
    auto outName = out.cast<py::tuple>()[0];
    auto nameStr = outName.cast<std::string>();
    if (auto arg = findArg(nameStr)) {
      outArgs.emplace_back(*arg);
      outProcessed[i] = true;
    }
  }

  auto funcArgs = res.attr("params").cast<py::list>();
  resultArgs.reserve(funcArgs.size());
  for (auto arg : funcArgs) {
    auto tup = arg.cast<py::tuple>();
    auto name = tup[0].cast<std::string>();
    auto param = tup[1];

    if (param.attr("kind").cast<py::object>().is(impl->vararg)) {
      for (auto arg : llvm::make_first_range(argsNamesAndValues))
        resultArgs.emplace_back(arg);

      argsNamesAndValues.clear();
      continue;
    }

    if (!argsNamesAndValues.empty()) {
      auto argName = argsNamesAndValues.front().second.getValue();
      if (argName.empty()) {
        resultArgs.emplace_back(argsNamesAndValues.front().first);
        argsNamesAndValues.erase(argsNamesAndValues.begin());
        continue;
      }
    }

    auto origArg = findArg(name);
    if (origArg) {
      resultArgs.emplace_back(*origArg);
    } else {
      auto defAttr = param.attr("default").cast<py::object>();
      if (defAttr.is(impl->empty))
        return mlir::failure();

      auto defVal = parseDefault(builder, loc, defAttr);
      if (!defVal)
        return mlir::failure();

      resultArgs.emplace_back(*defVal);
    }
  }

  for (auto [i, out] : llvm::enumerate(outTuple)) {
    if (argsNamesAndValues.empty())
      break;

    if (outProcessed[i])
      continue;

    auto outIdx = out.cast<py::tuple>()[1];
    outArgs.emplace_back(argsNamesAndValues.front().first);
    argsNamesAndValues.erase(argsNamesAndValues.begin());
  }

  // Not all arguments were processed.
  if (!argsNamesAndValues.empty())
    return mlir::failure();

  return mlir::success();
}
