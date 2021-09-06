#pragma once

#include "mlir/Support/LLVM.h"

namespace mlir {

class Pass;

/// Uplifts scf.parallel to affine.parallel.
std::unique_ptr<Pass> createSCFToAffinePass();

} // namespace mlir