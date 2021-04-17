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

#include "plier/transforms/cast_utils.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>

mlir::Value plier::index_cast(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value src, mlir::Type dst_type)
{
    auto src_type = src.getType();
    assert(src_type.isa<mlir::IndexType>() || dst_type.isa<mlir::IndexType>());
    if (src_type != dst_type)
    {
        return builder.create<mlir::IndexCastOp>(loc, src, dst_type);
    }
    return src;
}

mlir::Value plier::index_cast(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value src)
{
    return index_cast(builder, loc, src, mlir::IndexType::get(builder.getContext()));
}
