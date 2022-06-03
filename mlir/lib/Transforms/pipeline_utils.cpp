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

#include "mlir-extensions/Transforms/pipeline_utils.hpp"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>

#include "mlir-extensions/Dialect/plier/dialect.hpp"

mlir::ArrayAttr plier::getPipelineJumpMarkers(mlir::ModuleOp module) {
  return module->getAttrOfType<mlir::ArrayAttr>(
      plier::attributes::getJumpMarkersName());
}

void plier::addPipelineJumpMarker(mlir::ModuleOp module,
                                  mlir::StringAttr name) {
  assert(name);
  assert(!name.getValue().empty());

  auto jump_markers = plier::attributes::getJumpMarkersName();
  llvm::SmallVector<mlir::Attribute, 16> name_list;
  if (auto old_attr = module->getAttrOfType<mlir::ArrayAttr>(jump_markers)) {
    name_list.assign(old_attr.begin(), old_attr.end());
  }
  auto it = llvm::lower_bound(
      name_list, name, [](mlir::Attribute lhs, mlir::StringAttr rhs) {
        return lhs.cast<mlir::StringAttr>().getValue() < rhs.getValue();
      });
  if (it == name_list.end()) {
    name_list.emplace_back(name);
  } else if (*it != name) {
    name_list.insert(it, name);
  }
  module->setAttr(jump_markers,
                  mlir::ArrayAttr::get(module.getContext(), name_list));
}

void plier::removePipelineJumpMarker(mlir::ModuleOp module,
                                     mlir::StringAttr name) {
  assert(name);
  assert(!name.getValue().empty());

  auto jump_markers = plier::attributes::getJumpMarkersName();
  llvm::SmallVector<mlir::Attribute, 16> name_list;
  if (auto old_attr = module->getAttrOfType<mlir::ArrayAttr>(jump_markers)) {
    name_list.assign(old_attr.begin(), old_attr.end());
  }
  auto it = llvm::lower_bound(
      name_list, name, [](mlir::Attribute lhs, mlir::StringAttr rhs) {
        return lhs.cast<mlir::StringAttr>().getValue() < rhs.getValue();
      });
  assert(it != name_list.end());
  name_list.erase(it);
  module->setAttr(jump_markers,
                  mlir::ArrayAttr::get(module.getContext(), name_list));
}
