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

namespace mlir {
class Value;
class Location;
class OpBuilder;
class Type;
class IntegerType;
} // namespace mlir

namespace plier {
mlir::Value indexCast(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value src, mlir::Type dst_type);
mlir::Value indexCast(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value src);

mlir::Type makeSignlessType(mlir::Type type);
mlir::IntegerType makeSignlessType(mlir::IntegerType type);
} // namespace plier
