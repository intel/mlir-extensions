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

#pragma once

namespace plier {
class PipelineRegistry;
}

namespace llvm {
class StringRef;
}

namespace mlir {
class MLIRContext;
class TypeConverter;
} // namespace mlir

void populate_std_type_converter(mlir::MLIRContext &context,
                                 mlir::TypeConverter &converter);

void register_plier_to_std_pipeline(plier::PipelineRegistry &registry);

llvm::StringRef plier_to_std_pipeline_name();
