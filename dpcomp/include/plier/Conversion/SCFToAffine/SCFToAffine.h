#pragma once

#include "mlir/Support/LLVM.h"

namespace mlir {

class Pass;

/// Uplifts scf to affine. Supports:
/// 1. scf.parallel to affine.parallel.
std::unique_ptr<Pass> createSCFToAffinePass();

} // namespace mlir