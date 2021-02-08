#pragma once

#include <mlir/IR/Dialect.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace plier
{
// TODO: needed for LoopLikeInterface
using Value = ::mlir::Value;
using Region = ::mlir::Region;
using LogicalResult = ::mlir::LogicalResult;
using Operation = ::mlir::Operation;

template<typename T>
using ArrayRef = ::mlir::ArrayRef<T>;
}

#include "plier/PlierOpsEnums.h.inc"
#include "plier/PlierOpsDialect.h.inc"
#define GET_OP_CLASSES
#include "plier/PlierOps.h.inc"

namespace plier
{
namespace attributes
{
llvm::StringRef getFastmathName();
llvm::StringRef getJumpMarkersName();
llvm::StringRef getParallelName();
llvm::StringRef getMaxConcurrencyName();
llvm::StringRef getForceInlineName();
}

namespace detail
{
struct PyTypeStorage;
}

class PyType : public mlir::Type::TypeBase<::plier::PyType, mlir::Type,
                                           ::plier::detail::PyTypeStorage>
{
public:
    using Base::Base;

    static PyType get(mlir::MLIRContext *context, mlir::StringRef name);
    static PyType getUndefined(mlir::MLIRContext *context);
    static PyType getNone(mlir::MLIRContext *context);

    mlir::StringRef getName() const;
};


}
