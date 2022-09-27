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

#include "imex/Transforms/pipeline_utils.hpp"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>

#include "imex/Dialect/imex_util/dialect.hpp"

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
