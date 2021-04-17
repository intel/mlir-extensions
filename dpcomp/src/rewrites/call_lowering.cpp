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

#include "plier/rewrites/call_lowering.hpp"

plier::CallOpLowering::CallOpLowering(
    mlir::TypeConverter&, mlir::MLIRContext* context,
    CallOpLowering::resolver_t resolver):
    OpRewritePattern(context), resolver(resolver) {}

mlir::LogicalResult plier::CallOpLowering::matchAndRewrite(plier::PyCallOp op, mlir::PatternRewriter& rewriter) const
{
    auto operands = op.getOperands();
    if (operands.empty())
    {
        return mlir::failure();
    }
    auto func_type = operands[0].getType();
    if (!func_type.isa<plier::PyType>())
    {
        return mlir::failure();
    }

    llvm::SmallVector<mlir::Value> args;
    llvm::SmallVector<std::pair<llvm::StringRef, mlir::Value>> kwargs;
    auto getattr = mlir::dyn_cast_or_null<plier::GetattrOp>(operands[0].getDefiningOp());
    if (getattr)
    {
        args.push_back(getattr.getOperand());
    }
    auto kw_start = op.kw_start();
    operands = operands.drop_front();
    llvm::copy(operands.take_front(kw_start), std::back_inserter(args));
    for (auto it : llvm::zip(operands.drop_front(kw_start), op.kw_names()))
    {
        auto arg = std::get<0>(it);
        auto name = std::get<1>(it).cast<mlir::StringAttr>();
        kwargs.emplace_back(name.getValue(), arg);
    }

    return resolver(op, op.func_name(), args, kwargs, rewriter);
}
