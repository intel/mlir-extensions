
#ifndef IMEX_BLOCKING_ANALYSIS_H
#define IMEX_BLOCKING_ANALYSIS_H

#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>

#include <llvm/ADT/SetVector.h>

#include "imex/Utils/XeArch.h"

namespace imex {

/// a class representing a inner block size, provides some
/// convinient methods for manipulation.
class Block {
public:
  Block() : values{0, 0} {}

  Block(int64_t h, int64_t w) : values{h, w} {}

  int64_t &operator[](size_t index);
  const int64_t &operator[](size_t index) const;

  bool operator==(Block &other) const;
  bool operator==(const Block &other) const;

  bool operator!=(Block &other) const { return !(*this == other); }
  bool operator!=(const Block &other) const { return !(*this == other); }

  void print(llvm::raw_ostream &os) const;

  llvm::ArrayRef<int64_t> asArrayRef() const;

  operator bool() const { return values[0] != 0 && values[1] != 0; }

private:
  int64_t values[2];
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, Block blk);

// A pair of operator and operand index number representing
// the use point of a value.
typedef std::pair<mlir::Operation *, int64_t> UsePoint;

class BlockingAnalysis {
public:
  explicit BlockingAnalysis(std::shared_ptr<XeuArchInterface> uArch) {
    this->uArch = uArch;
  };

  mlir::LogicalResult run(mlir::Operation *op);

  Block getUseBlockSize(mlir::Value val, UsePoint point) const;
  Block getDefBlockSize(mlir::Value val) const;
  void printAnalysisResult();

private:
  mlir::DataFlowSolver solver;
  std::shared_ptr<XeuArchInterface> uArch;
  mlir::Operation *target;
};

} // namespace imex

#endif // IMEX_BLOCKING_ANALYSIS_H
