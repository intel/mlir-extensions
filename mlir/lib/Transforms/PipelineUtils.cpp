// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/PipelineUtils.hpp"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>

#include "imex/Dialect/imex_util/Dialect.hpp"

mlir::ArrayAttr imex::getPipelineJumpMarkers(mlir::ModuleOp module) {
  return module->getAttrOfType<mlir::ArrayAttr>(
      imex::util::attributes::getJumpMarkersName());
}

void imex::addPipelineJumpMarker(mlir::ModuleOp module, mlir::StringAttr name) {
  assert(name);
  assert(!name.getValue().empty());

  auto jumpMarkers = imex::util::attributes::getJumpMarkersName();
  llvm::SmallVector<mlir::Attribute, 16> nameList;
  if (auto oldAttr = module->getAttrOfType<mlir::ArrayAttr>(jumpMarkers))
    nameList.assign(oldAttr.begin(), oldAttr.end());

  auto it = llvm::lower_bound(
      nameList, name, [](mlir::Attribute lhs, mlir::StringAttr rhs) {
        return lhs.cast<mlir::StringAttr>().getValue() < rhs.getValue();
      });
  if (it == nameList.end()) {
    nameList.emplace_back(name);
  } else if (*it != name) {
    nameList.insert(it, name);
  }
  module->setAttr(jumpMarkers,
                  mlir::ArrayAttr::get(module.getContext(), nameList));
}

void imex::removePipelineJumpMarker(mlir::ModuleOp module,
                                    mlir::StringAttr name) {
  assert(name);
  assert(!name.getValue().empty());

  auto jumpMarkers = imex::util::attributes::getJumpMarkersName();
  llvm::SmallVector<mlir::Attribute, 16> nameList;
  if (auto oldAttr = module->getAttrOfType<mlir::ArrayAttr>(jumpMarkers))
    nameList.assign(oldAttr.begin(), oldAttr.end());

  auto it = llvm::lower_bound(
      nameList, name, [](mlir::Attribute lhs, mlir::StringAttr rhs) {
        return lhs.cast<mlir::StringAttr>().getValue() < rhs.getValue();
      });
  assert(it != nameList.end());
  nameList.erase(it);
  module->setAttr(jumpMarkers,
                  mlir::ArrayAttr::get(module.getContext(), nameList));
}
