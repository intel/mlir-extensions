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

#include "NumpyResolver.hpp"

#include "imex/Dialect/imex_util/dialect.hpp"

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

  py::object getFuncDesc(llvm::StringRef name) {
    return map(py::str(name.data(), name.size()));
  }
};

NumpyResolver::NumpyResolver(const char *modName, const char *mapName)
    : impl(std::make_unique<Impl>()) {
  impl->mod = py::module::import(modName);
  impl->map = impl->mod.attr(mapName);

  auto inspect = py::module::import("inspect");
  impl->empty = inspect.attr("Parameter").attr("empty");
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
                               llvm::SmallVectorImpl<mlir::Value> &resultArgs) {
  assert(args.size() == argsNames.size() && "args and names count misnatch");
  auto res = impl->getFuncDesc(name);
  if (res.is_none())
    return mlir::failure();

  auto argsNamesArr = argsNames.getValue();

  auto findArg = [&](llvm::StringRef argName) -> llvm::Optional<mlir::Value> {
    if (argsNamesArr.empty())
      return llvm::None;

    for (auto [i, nameAttr] : llvm::enumerate(argsNamesArr)) {
      auto origArgName = nameAttr.cast<mlir::StringAttr>().getValue();

      if (origArgName == argName) {
        auto res = args[i];
        argsNamesArr = argsNamesArr.drop_front();
        args = args.drop_front();
        return res;
      }
    }
    return llvm::None;
  };

  auto funcArgs = res.cast<py::list>();
  resultArgs.resize(funcArgs.size());
  for (auto [i, arg] : llvm::enumerate(funcArgs)) {
    assert(args.size() == argsNamesArr.size() &&
           "args and names count misnatch");

    if (!argsNamesArr.empty()) {
      auto argName = argsNamesArr.front().cast<mlir::StringAttr>().getValue();
      if (argName.empty()) {
        resultArgs[i] = args.front();
        argsNamesArr = argsNamesArr.drop_front();
        args = args.drop_front();
        continue;
      }
    }

    auto tup = arg.cast<py::tuple>();
    auto name = tup[0].cast<std::string>();
    auto param = tup[1];

    auto origArg = findArg(name);
    if (origArg) {
      resultArgs[i] = *origArg;
    } else {
      auto defAttr = param.attr("default").cast<py::object>();
      if (defAttr.is(impl->empty))
        return mlir::failure();

      auto defVal = parseDefault(builder, loc, defAttr);
      if (!defVal)
        return mlir::failure();

      resultArgs[i] = *defVal;
    }
  }

  // Not all arguments were processed.
  if (!argsNamesArr.empty())
    return mlir::failure();

  return mlir::success();
}
