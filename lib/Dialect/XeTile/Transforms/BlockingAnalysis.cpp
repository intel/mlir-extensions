#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/Debug.h>

#include "BlockingAnalysis.h"

namespace llvm {
using imex::Block;
// Implementation of llvm::DenseMapInfo for Block, required for
// using Block as a value in DenseMap.
template <> struct DenseMapInfo<Block> {
  static inline Block getEmptyKey() {
    return Block(-1, -1); // the empty key
  }

  static inline Block getTombstoneKey() {
    return Block(-2, -2); // the tombstone key
  }

  static unsigned getHashValue(const Block &b) {
    return hash_combine(b[0], b[1]);
  }

  static bool isEqual(const Block &lhs, const Block &rhs) { return lhs == rhs; }
};
} // namespace llvm

namespace imex {

// ===------------------ Block Implementation --------------------------===//

int64_t &Block::operator[](size_t index) {
  assert(index < 2 && "Index out of bounds");
  return values[index];
}

const int64_t &Block::operator[](size_t index) const {
  assert(index < 2 && "Index out of bounds");
  return values[index];
}

bool Block::operator==(Block &other) const {
  return values[0] == other.values[0] && values[1] == other.values[1];
}

bool Block::operator==(const Block &other) const {
  return values[0] == other.values[0] && values[1] == other.values[1];
}

void Block::print(llvm::raw_ostream &os) const {
  os << "[" << values[0] << ", " << values[1] << "]";
}

llvm::ArrayRef<int64_t> Block::asArrayRef() const {
  return llvm::ArrayRef<int64_t>(values, 2);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, Block blk) {
  blk.print(os);
  return os;
}

// ===------------------ BlockRequests Implementation --------------------===//
// A class holding all blocking requests for a given mlir::Value.
// For convience, it also tracks the UsePoint of the value.
class BlockingRequests {
public:
  BlockingRequests() = default;
  BlockingRequests(int64_t h, int64_t w, mlir::Operation *user, int64_t pos)
      : BlockingRequests(h, w, UsePoint(user, pos)) {}

  BlockingRequests(int64_t h, int64_t w, UsePoint point)
      : BlockingRequests(Block(h, w), point) {}

  BlockingRequests(llvm::ArrayRef<int64_t> shape, UsePoint point)
      : BlockingRequests(shape[0], shape[1], point) {
    assert(shape.size() == 2 && "Invalid block size.");
  }

  BlockingRequests(Block block, UsePoint point);

  bool operator==(const BlockingRequests &other) const;
  bool operator!=(const BlockingRequests &other) const;

  Block getDefBlock() const;
  Block getUseBlock(UsePoint point) const;

  void print(llvm::raw_ostream &os) const;

  static BlockingRequests meet(const BlockingRequests &lhs,
                               const BlockingRequests &rhs);

  static BlockingRequests join(const BlockingRequests &lhs,
                               const BlockingRequests &rhs);

  // indicate that one use of the result operand
  // has decided on the inner block size.
  bool isInitialized() const { return requests.size() != 0; }

  int64_t getNumUniqRequests() const { return getRequests().size(); }

  llvm::SmallVector<Block> getRequests() const {
    llvm::SmallDenseSet<Block, 8> reqs;
    for (auto [point, block] : requests)
      reqs.insert(block);
    return llvm::SmallVector<Block>(reqs.begin(), reqs.end());
  }

  void updateDefBlock(Block block) { def = block; }

private:
  Block def;
  llvm::DenseMap<UsePoint, Block> requests;
};

BlockingRequests::BlockingRequests(Block block, UsePoint point) {
  assert(block && "Invalid block.");
  requests.try_emplace(point, block);
}

Block BlockingRequests::getDefBlock() const {
  if (def)
    return def;
  if (requests.size())
    return (requests.begin()->second);
  return Block();
}

Block BlockingRequests::getUseBlock(UsePoint point) const {
  return requests.lookup(point);
}

void BlockingRequests::print(llvm::raw_ostream &os) const {
  if (!isInitialized()) {
    os << "Uninitialized";
  } else {
    os << "Requests (" << requests.size() << ", "
       << "def: " << def << "): [";
    for (auto [i, iter] : llvm::enumerate(requests)) {
      auto point = iter.first;
      auto block = iter.second;
      os << "{Point(" << *point.first << ", " << point.second << "), blk("
         << block << ")}";
      if (i != requests.size() - 1)
        os << ", ";
      else
        os << "]";
    }
  }
}

bool BlockingRequests::operator==(const BlockingRequests &other) const {
  return requests == other.requests;
}

bool BlockingRequests::operator!=(const BlockingRequests &other) const {
  return !(*this == other);
}

BlockingRequests BlockingRequests::meet(const BlockingRequests &lhs,
                                        const BlockingRequests &rhs) {
  return join(lhs, rhs);
}

BlockingRequests BlockingRequests::join(const BlockingRequests &lhs,
                                        const BlockingRequests &rhs) {
  BlockingRequests newReq;
  if (lhs.isInitialized()) {
    for (auto [point, block] : lhs.requests) {
      newReq.requests.try_emplace(point, block);
    }
  }
  if (rhs.isInitialized()) {
    for (auto [point, block] : rhs.requests) {
      newReq.requests.try_emplace(point, block);
    }
  }
  return newReq;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              BlockingRequests requests) {
  requests.print(os);
  return os;
}

// ===---------------- BlockingLattice Implementation -----------------===//
// A lattice wrapper for BlockingRequests
struct BlockingLattice : public mlir::dataflow::Lattice<BlockingRequests> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BlockingLattice)
  using Lattice::Lattice;

  mlir::ChangeResult join(const AbstractSparseLattice &rhs) override {
    return join(static_cast<const BlockingLattice &>(rhs).getValue());
  }

  mlir::ChangeResult join(const BlockingRequests &other) {
    auto &val = getValue();
    BlockingRequests newValue = BlockingRequests::join(val, other);
    if (newValue == val)
      return mlir::ChangeResult::NoChange;
    val = newValue;
    return mlir::ChangeResult::Change;
  }
};

// ===----------------------BlockingAnalysisImpl ---------------------===//
class BlockingAnalysisImpl
    : public mlir::dataflow::SparseBackwardDataFlowAnalysis<BlockingLattice> {
public:
  BlockingAnalysisImpl(mlir::DataFlowSolver &solver,
                       mlir::SymbolTableCollection &symbolTable,
                       std::shared_ptr<XeuArchInterface> uArch)
      : SparseBackwardDataFlowAnalysis(solver, symbolTable), uArch(uArch) {}

  void visitOperation(mlir::Operation *op,
                      mlir::ArrayRef<BlockingLattice *> operands,
                      mlir::ArrayRef<const BlockingLattice *> results) override;

  void visitBranchOperand(mlir::OpOperand &operand) override {}

  void visitCallOperand(mlir::OpOperand &operand) override {}

  void setToExitState(BlockingLattice *lattice) override {}

private:
  void visitPrefetchTileOp(xetile::PrefetchTileOp op,
                           mlir::ArrayRef<BlockingLattice *> operands,
                           mlir::ArrayRef<const BlockingLattice *> results);

  void visitLoadTileOp(xetile::LoadTileOp op,
                       mlir::ArrayRef<BlockingLattice *> operands,
                       mlir::ArrayRef<const BlockingLattice *> results);

  void visitStoreTileOp(xetile::StoreTileOp op,
                        mlir::ArrayRef<BlockingLattice *> operands,
                        mlir::ArrayRef<const BlockingLattice *> results);

  void visitUpdateTileOp(xetile::UpdateTileOffsetOp op,
                         mlir::ArrayRef<BlockingLattice *> operands,
                         mlir::ArrayRef<const BlockingLattice *> results);

  void visitTileMMAOp(xetile::TileMMAOp op,
                      mlir::ArrayRef<BlockingLattice *> operands,
                      mlir::ArrayRef<const BlockingLattice *> results);

  void visitVectorizableOp(mlir::Operation *op,
                           mlir::ArrayRef<BlockingLattice *> operands,
                           mlir::ArrayRef<const BlockingLattice *> results);

  void visitShapecastOp(mlir::vector::ShapeCastOp op,
                        mlir::ArrayRef<BlockingLattice *> operands,
                        mlir::ArrayRef<const BlockingLattice *> results);

  void visitCreateMaskOp(mlir::vector::CreateMaskOp op,
                         mlir::ArrayRef<BlockingLattice *> operands,
                         mlir::ArrayRef<const BlockingLattice *> results);

  void visitReductionOp(xetile::ReductionOp op,
                        mlir::ArrayRef<BlockingLattice *> operands,
                        mlir::ArrayRef<const BlockingLattice *> results);

  void visitBroadcastOp(xetile::BroadcastOp op,
                        mlir::ArrayRef<BlockingLattice *> operands,
                        mlir::ArrayRef<const BlockingLattice *> results);

  void visitTransposeOp(xetile::TransposeOp op,
                        mlir::ArrayRef<BlockingLattice *> operands,
                        mlir::ArrayRef<const BlockingLattice *> results);

  int getMaxSLMBlockSize(int elemBitWidth, int height);

  template <typename Integertype>
  Block getInnerBlockSize(mlir::Operation *op, mlir::Type elemTy,
                          llvm::ArrayRef<Integertype> &shape,
                          int memorySpace = 0);

  llvm::SmallVector<unsigned int>
  getMMASize(mlir::Type elemTy, const int APrecision, const int BPrecision,
             const int CPrecision, const int DPrecision);

private:
  std::shared_ptr<XeuArchInterface> uArch = nullptr;
};

void BlockingAnalysisImpl::visitOperation(
    mlir::Operation *op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {

  if (auto updateTileOp = mlir::dyn_cast<xetile::UpdateTileOffsetOp>(op))
    visitUpdateTileOp(updateTileOp, operands, results);

  if (auto prefetchOp = mlir::dyn_cast<xetile::PrefetchTileOp>(op))
    visitPrefetchTileOp(prefetchOp, operands, results);

  if (auto loadOp = mlir::dyn_cast<xetile::LoadTileOp>(op))
    visitLoadTileOp(loadOp, operands, results);

  if (auto storeOp = mlir::dyn_cast<xetile::StoreTileOp>(op))
    visitStoreTileOp(storeOp, operands, results);

  if (auto tileMMAOp = mlir::dyn_cast<xetile::TileMMAOp>(op))
    visitTileMMAOp(tileMMAOp, operands, results);

  if (auto reductionOp = mlir::dyn_cast<xetile::ReductionOp>(op))
    visitReductionOp(reductionOp, operands, results);

  if (auto transposeOp = mlir::dyn_cast<xetile::TransposeOp>(op))
    visitTransposeOp(transposeOp, operands, results);

  if (auto broadcastOp = mlir::dyn_cast<xetile::BroadcastOp>(op))
    visitBroadcastOp(broadcastOp, operands, results);

  if (op->hasTrait<mlir::OpTrait::Vectorizable>())
    visitVectorizableOp(op, operands, results);

  if (auto shapecastOp = mlir::dyn_cast<mlir::vector::ShapeCastOp>(op))
    visitShapecastOp(shapecastOp, operands, results);

  if (auto createMaskOp = mlir::dyn_cast<mlir::vector::CreateMaskOp>(op))
    visitCreateMaskOp(createMaskOp, operands, results);
}

void BlockingAnalysisImpl::visitPrefetchTileOp(
    xetile::PrefetchTileOp op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {
  auto tileTy = op.getTile().getType();
  auto elemTy = tileTy.getElementType();
  auto shape = tileTy.getShape();
  auto memSpace = tileTy.getMemoryScopeAsInt();
  // initialized with a default size queried from the architecture
  auto size = getInnerBlockSize(op, elemTy, shape, memSpace);
  if (!size)
    return; // do nothing if didnot get a valid block size
  auto BlockingRequest = BlockingRequests(size, UsePoint(op, 0));
  propagateIfChanged(operands[0], operands[0]->join(BlockingRequest));
}

void BlockingAnalysisImpl::visitLoadTileOp(
    xetile::LoadTileOp op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {
  auto lattice = results[0]->getValue();

  if (lattice.getNumUniqRequests() > 1)
    op.emitWarning("multiple users requesting different blocking sizes.");

  auto tileTy = op.getSource().getType();
  auto elemTy = tileTy.getElementType();
  auto bitWidth = elemTy.getIntOrFloatBitWidth();
  auto shape = tileTy.getShape();
  auto memSpace = tileTy.getMemoryScopeAsInt();
  // initialized with a default size queried from the architecture
  Block block = getInnerBlockSize(op, elemTy, shape, memSpace);

  // It has users but users' requirements are not available yet.
  // Worth to wait until all users are visited.
  if (!op.getValue().use_empty() && !lattice.isInitialized())
    return;

  // adjust according to user's requirements if it is available
  if (lattice.isInitialized()) {
    // Always align the width dimension.
    // NOTE: For transpose usecase, we still align the width dimension. This is
    // because loads with transpose cannot have array_length > 1, plus it has HW
    // limitations on supported width. If we align the height dimension (for
    // reducing reg data movement), it will lead to multiple smaller loads.
    for (auto rq : lattice.getRequests())
      if (rq[1] && ((rq[1] * bitWidth) % 32 == 0)) // has to be 32-bit aligned
        block[1] = std::min(block[1], rq[1]);
  }

  if (!block)
    return; // do nothing if didnot get a valid block size

  auto BlockingRequest = BlockingRequests(block, UsePoint({op, 0}));
  // propagate the blocking size to its def op
  propagateIfChanged(operands[0], operands[0]->join(BlockingRequest));

  // update the def block size for the result value
  BlockingRequests &def = getLatticeElement(op.getValue())->getValue();
  def.updateDefBlock(block);
}

void BlockingAnalysisImpl::visitStoreTileOp(
    xetile::StoreTileOp op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {
  auto tileTy = op.getTile().getType();
  auto elemTy = tileTy.getElementType();
  auto shape = tileTy.getShape();
  auto memSpace = tileTy.getMemoryScopeAsInt();
  auto size = getInnerBlockSize(op, elemTy, shape, memSpace);

  if (!size)
    return; // do nothing if didnot get a valid block size

  for (auto &&[i, inputOpr] : llvm::enumerate(operands)) {
    auto blockingRequest = BlockingRequests(size, UsePoint(op, i));
    propagateIfChanged(inputOpr, inputOpr->join(blockingRequest));
  }
}

void BlockingAnalysisImpl::visitUpdateTileOp(
    xetile::UpdateTileOffsetOp op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {
  auto lattice = results[0]->getValue();
  if (lattice.isInitialized()) {
    auto block = lattice.getRequests()[0];
    auto request = BlockingRequests(block, UsePoint(op, 0));
    propagateIfChanged(operands[0], operands[0]->join(request));
  }
}

void BlockingAnalysisImpl::visitTileMMAOp(
    xetile::TileMMAOp op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {

  auto getElemBitWidth = [](mlir::VectorType vecTy) {
    return vecTy.getElementType().getIntOrFloatBitWidth();
  };

  auto C = op.getC();
  auto aPrecision = getElemBitWidth(op.getAType());
  auto bPrecision = getElemBitWidth(op.getBType());
  auto dPrecision = getElemBitWidth(op.getOutputType());
  auto cPrecision = !C ? dPrecision : getElemBitWidth(C.getType());

  auto mmaSize = getMMASize(op.getElementType(), aPrecision, bPrecision,
                            cPrecision, dPrecision);

  auto blockSizeForA =
      BlockingRequests(mmaSize[0], mmaSize[1], UsePoint({op, 0}));
  auto blockSizeForB =
      BlockingRequests(mmaSize[1], mmaSize[2], UsePoint({op, 1}));

  propagateIfChanged(operands[0], operands[0]->join(blockSizeForA));
  propagateIfChanged(operands[1], operands[1]->join(blockSizeForB));
  if (C) {
    auto blockSizeForC =
        BlockingRequests(mmaSize[0], mmaSize[2], UsePoint(op, 2));
    propagateIfChanged(operands[2], operands[2]->join(blockSizeForC));
  }

  // update the def block size for the result value
  BlockingRequests &def = getLatticeElement(op.getOutput())->getValue();
  def.updateDefBlock(Block(mmaSize[0], mmaSize[2]));
}

void BlockingAnalysisImpl::visitReductionOp(
    xetile::ReductionOp op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {
  auto srcTy = op.getSource().getType();
  auto dims = op.getReductionDims();
  // We only support reduction on 2D types now.
  if (srcTy.getRank() != 2 || dims.size() != 1)
    return;

  auto elemTy = srcTy.getElementType();
  auto shape = srcTy.getShape();
  // ReductionOp is special. Its blocking size is fixed to {1,
  // min(subgroupSize, width)}
  auto size = getInnerBlockSize(op, elemTy, shape);
  if (!size)
    return; // do nothing if didnot get a valid block size

  auto blockingRequest = BlockingRequests(size, UsePoint(op, 0));
  propagateIfChanged(operands[0], operands[0]->join(blockingRequest));
}

void BlockingAnalysisImpl::visitBroadcastOp(
    xetile::BroadcastOp op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {
  auto srcTy = op.getSource().getType();
  auto dims = op.getBroadcastDim();
  // We only support reduction on 2D types now.
  if (srcTy.getRank() != 2 || dims.size() != 1)
    return;

  // broadcast is special. It is currently handled in a hacking way,
  // and need to be generilized. It is not blocked if its users have
  // not requested a blocking size. Otherwize, its blocking size has
  // to be [1, 1] if broadcast along dim 1, or [1, requestedSize[1]]
  // if broadcast along dim 0.
  auto lattice = results[0]->getValue();
  if (!lattice.isInitialized())
    return;

  auto dim = dims[0];
  Block blockSize;

  if (dim == 0) {
    auto req = lattice.getRequests()[0];
    blockSize = Block(1, req[1]);
  } else if (dim == 1) {
    blockSize = Block(1, 1);
  } else {
    return;
  }
  auto blockingRequest = BlockingRequests(blockSize, UsePoint(op, 0));
  propagateIfChanged(operands[0], operands[0]->join(blockingRequest));
}

void BlockingAnalysisImpl::visitTransposeOp(
    xetile::TransposeOp op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {

  auto permutation = op.getPermutation();
  auto resType = op.getResult().getType();
  // we only support true 2D transpose now
  if (resType.getRank() != 2 || permutation != mlir::ArrayRef<int64_t>({1, 0}))
    return;

  auto lattice = results[0]->getValue();

  // Wait for requests from users.
  if (!op->use_empty() && !lattice.isInitialized())
    return;

  Block block;
  auto srcTy = op.getVector().getType();
  auto shape = srcTy.getShape();

  // use the default size if no users
  if (op->use_empty()) {
    block = getInnerBlockSize(op, srcTy.getElementType(), shape);
  }

  // TransposeOp determines its blocking size based on requests from
  // its users, by swapping the blocking size of its users.
  if (lattice.isInitialized()) {
    // TODO: handle multiple users
    if (lattice.getNumUniqRequests() == 1) {
      auto req = lattice.getRequests()[0];
      if (req[0] == 1 && req[1] == 1) {
        // use default size if the request is [1, 1]
        block = getInnerBlockSize(op, srcTy.getElementType(), shape);
      } else {
        block = Block(req[1], req[0]);
      }
    }
  }

  if (!block)
    return; // do nothing if didnot get a valid block size

  auto request = BlockingRequests(block, UsePoint(op, 0));
  propagateIfChanged(operands[0], operands[0]->join(request));

  // update the def block size for the result value
  BlockingRequests &def = getLatticeElement(op.getResult())->getValue();
  def.updateDefBlock(Block(block[1], block[0]));
}

void BlockingAnalysisImpl::visitVectorizableOp(
    mlir::Operation *op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {
  // Currently only supports simple elementwise math ops.
  if (op->getNumResults() != 1)
    return;

  auto type = mlir::dyn_cast<mlir::VectorType>(op->getResult(0).getType());
  if (!type)
    return;

  auto lattice = results[0]->getValue();

  // Wait for requests from users.
  if (!op->use_empty() && !lattice.isInitialized())
    return;

  auto elemTy = type.getElementType();
  auto shape = type.getShape();
  Block block = getInnerBlockSize(op, elemTy, shape);

  // elementwise operations are not sensitive to the block size.
  // It will use the block size requested by its users.
  if (lattice.isInitialized()) {
    block[0] = 0;
    for (auto &req : lattice.getRequests()) {
      block[0] = std::max(block[0], req[0]);
      block[1] = std::min(block[1], req[1]);
    }
  }

  // do nothing if get an invalid block
  if (!block)
    return;

  // propagate the block size on its operands
  for (auto &&[i, inputOpr] : llvm::enumerate(operands)) {
    auto req = BlockingRequests(block, UsePoint(op, i));
    propagateIfChanged(inputOpr, inputOpr->join(req));
  }

  // update the def block size for the result value
  BlockingRequests &def = getLatticeElement(op->getResult(0))->getValue();
  def.updateDefBlock(block);
}

void BlockingAnalysisImpl::visitShapecastOp(
    mlir::vector::ShapeCastOp op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {
  auto shape = op.getSource().getType().getShape();
  if (shape.size() == 2) {
    auto BlockingRequest = BlockingRequests(shape, UsePoint(op, 0));
    propagateIfChanged(operands[0], operands[0]->join(BlockingRequest));
  }
}

void BlockingAnalysisImpl::visitCreateMaskOp(
    mlir::vector::CreateMaskOp op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {
  auto vecTy = op.getVectorType();
  auto shape = vecTy.getShape();
  auto elemTy = vecTy.getElementType();

  auto lattice = results[0]->getValue();
  BlockingRequests &def = getLatticeElement(op->getResult(0))->getValue();
  // TODO: following the Antonio's implementation and use the default size
  // [1, subgroupSize] for CreateMaskOp, but it can be more general.
  Block block = getInnerBlockSize(op, elemTy, shape);
  def.updateDefBlock(block);
}

int BlockingAnalysisImpl::getMaxSLMBlockSize(int elemBitWidth, int height) {
  // TODO: use uArch to get max vec size?
  const int lscConstraint = 512; // lsc supports upto 512 bytes per load/store
  int numElems = (lscConstraint * 8) / elemBitWidth;
  int width = numElems / height;
  return width;
}

// Determine the inner block size for the given operation based on the
// operand's element data type, shape, and also memory space.
template <typename Integertype>
Block BlockingAnalysisImpl::getInnerBlockSize(
    mlir::Operation *op, mlir::Type elemTy, llvm::ArrayRef<Integertype> &shape,
    int memorySpace) {
  assert(elemTy.isIntOrFloat() && "only support int or float element type.");

  int elemSize = elemTy.getIntOrFloatBitWidth();
  const int64_t subgroupSize = uArch->getOneGRFSizeBits() / elemSize;

  int maxHeight = 0, minHeight = 0, maxWidth = 0, minWidth = 0;
  if (mlir::isa<xetile::ReductionOp>(op) ||
      mlir::isa<xetile::BroadcastOp>(op)) {
    // for reduction and broadcast ops, we simply using
    // [1, subgroupSize] as innerblock size
    maxWidth = subgroupSize;
    minWidth = 1;
    maxHeight = 1;
    minHeight = 1;
  } else if (op->hasTrait<mlir::OpTrait::Vectorizable>()) {
    // for elementwise operations, they are pretty flexiable
    // on the block size. But we expect its second dimension
    // is register size aligned.
    minWidth = 1;
    minHeight = 1;
    maxWidth = std::min<int>(shape[1], subgroupSize);
    maxHeight = shape[0];
  } else if (mlir::isa<xetile::TransposeOp>(op)) {
    // for transpose op, we will use the original shape
    // as the default size, and adjust it if it is defined
    // by a load op
    minWidth = 1;
    minHeight = 1;
    maxWidth = shape[1];
    maxHeight = shape[0];

    // if the transpose follows a load op, and data element is 32-bit
    // or 64-bit, it is expected to be folded with a load, and need to
    // be aligned to hardware constraints.
    auto defOp = op->getOperand(0).getDefiningOp<xetile::LoadTileOp>();
    if (defOp && elemSize >= 32) {
      auto params = uArch->get2DLoadConfig(defOp, elemSize, false, true);
      minHeight = params->blockHeight.min;
      minWidth = params->blockWidth.min;
      // to be compatible with the SIMT instrinsic, the maximum height is
      // limited to 16, which is maximum supported value by SIMT instrinsic.
      maxHeight = std::min<int>(params->blockHeight.max, 16);
      maxWidth = params->blockWidth.max;
    }
  } else if (mlir::isa<mlir::vector::CreateMaskOp>(op)) {
    minWidth = 1;
    minHeight = 1;
    maxWidth = std::min<int>(shape[1], subgroupSize);
    maxHeight = 1;
  } else if (memorySpace == 3) {
    // this is supposed for load/store from/to SLM, they will use regular
    // load/store instructions with chunk size. lsc instrinsic and hardware
    // has serveral limits on the size per load/store.
    minHeight = minWidth = 1;
    // If shape[0] is divisible by subgroup size, we use regular load (with
    // chunk size) with XeGPU.load_gather (maxHeight = 16). Otherwise, we
    // use 1D load with XeGPU.load_nd(1d, maxHeight = 1).
    maxHeight = shape[0] % subgroupSize == 0 ? subgroupSize : 1;
    maxWidth = getMaxSLMBlockSize(elemSize, maxHeight);
  } else { // for load/store from/to global memory
    mlir::FailureOr<LoadStore2DConfig> params;
    if (mlir::isa<xetile::StoreTileOp>(op))
      params = uArch->get2DStoreConfig(elemSize);
    if (mlir::isa<xetile::PrefetchTileOp>(op) ||
        mlir::isa<xetile::LoadTileOp>(op)) {
      bool transpose = false;
      // if its user is a transpose op, and data element is 32-bit
      // or 64-bit, we will use the transpose supported size.
      if (auto loadOp = mlir::dyn_cast<xetile::LoadTileOp>(op)) {
        auto value = loadOp.getValue();
        transpose = elemSize >= 32 && value.hasOneUse() &&
                    mlir::isa<xetile::TransposeOp>(*(value.user_begin()));
      }
      params = uArch->get2DLoadConfig(op, elemSize, false, transpose);
    }
    if (mlir::succeeded(params)) {
      maxHeight = params->blockHeight.max;
      minHeight = params->blockHeight.min;
      maxWidth = params->blockWidth.max;
      minWidth = params->blockWidth.min;
    }
  }

  auto findLargestDivisorInRange = [&](int64_t v, int64_t l, int64_t h) {
    for (int i = h; i >= l; i--) {
      if (v % i == 0)
        return i;
    }
    // irregular shape or shape is not in the supported range.
    return 0;
  };

  auto height = findLargestDivisorInRange(shape[0], minHeight, maxHeight);
  auto width = findLargestDivisorInRange(shape[1], minWidth, maxWidth);
  return Block(height, width);
}

llvm::SmallVector<unsigned int>
BlockingAnalysisImpl::getMMASize(mlir::Type elemTy, const int APrecision,
                                 const int BPrecision, const int CPrecision,
                                 const int DPrecision) {
  assert(elemTy.isIntOrFloat() && "only support int or float data type.");
  auto dpasParams =
      uArch->getDPASConfig(APrecision, BPrecision, CPrecision, DPrecision);
  return llvm::SmallVector<unsigned int>(
      {dpasParams.m, dpasParams.k, dpasParams.n});
}

// ===--------------------------------BlockingAnalysis---------------------------------===//

mlir::LogicalResult BlockingAnalysis::run(mlir::Operation *op) {
  mlir::SymbolTableCollection symbolTable;
  // BlockingAnalysisImpl is using default initialize method
  // provided by SparseBackwardDataFlowAnalysis. And this default
  // initialize method relies on results of DeadCodeAnalysis to
  // skip analysis on the dead code.
  solver.load<mlir::dataflow::DeadCodeAnalysis>();
  solver.load<mlir::dataflow::SparseConstantPropagation>();
  solver.load<BlockingAnalysisImpl>(symbolTable, uArch);
  target = op;
  return solver.initializeAndRun(op);
}

void BlockingAnalysis::printAnalysisResult() {
  llvm::dbgs() << "\n\nBlockingAnalysis Results:\n";
  target->walk([&](mlir::Operation *op) {
    if (op->getNumRegions() == 0 && op->getNumResults() == 1) {
      auto resTy = op->getResult(0).getType();
      if (mlir::isa<mlir::VectorType>(resTy) ||
          mlir::isa<xetile::TileType>(resTy)) {
        llvm::dbgs() << "\nOp: " << *op;
        for (auto [i, inputOpr] : llvm::enumerate(op->getOperands())) {
          if (mlir::isa<mlir::VectorType>(inputOpr.getType()) ||
              mlir::isa<xetile::TileType>(inputOpr.getType())) {
            UsePoint p(op, i);
            llvm::dbgs() << "\n   opr[" << i << "]: " << inputOpr
                         << " --> blkSZ: " << getUseBlockSize(inputOpr, p);
          }
        }

        for (auto [i, res] : llvm::enumerate(op->getResults()))
          llvm::dbgs() << "\n   res[" << i << "]: " << res
                       << " --> blkSZ: " << getDefBlockSize(res);
        llvm::dbgs() << "\n";
      }
    } else if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      llvm::dbgs() << "\nOp: " << op->getName();
      for (auto [i, arg] : llvm::enumerate(forOp.getRegionIterArgs()))
        llvm::dbgs() << "\n   arg[" << i << "]: "
                     << " --> blkSZ: " << getDefBlockSize(arg);

      for (auto [i, res] : llvm::enumerate(forOp.getResults()))
        llvm::dbgs() << "\n   res[" << i << "]: "
                     << " --> blkSZ: " << getDefBlockSize(res);
      llvm::dbgs() << "\n";
    } else if (auto YieldOp = mlir::dyn_cast<mlir::scf::YieldOp>(op)) {
      llvm::dbgs() << "\nOp: " << op->getName();
      for (auto [i, res] : llvm::enumerate(YieldOp.getResults()))
        llvm::dbgs() << "\n   res[" << i << "]: " << res
                     << " --> blkSZ: " << getDefBlockSize(res) << ", "
                     << getUseBlockSize(res, UsePoint(op, i));
      llvm::dbgs() << "\n";
    } else if (auto StoreOp = mlir::dyn_cast<xetile::StoreTileOp>(op)) {
      llvm::dbgs() << "\nOp: " << *op;
      for (auto [i, inputOpr] : llvm::enumerate(op->getOperands())) {
        llvm::dbgs() << "\n   opr[" << i << "]: " << inputOpr << " --> blkSZ: "
                     << getUseBlockSize(inputOpr, UsePoint(StoreOp, i));
      }
      llvm::dbgs() << "\n";
    }
  });
}

Block BlockingAnalysis::getUseBlockSize(mlir::Value val, UsePoint point) const {
  auto *state = solver.lookupState<BlockingLattice>(val);
  if (!state)
    return Block();
  return state->getValue().getUseBlock(point);
}

Block BlockingAnalysis::getDefBlockSize(mlir::Value val) const {
  auto *state = solver.lookupState<BlockingLattice>(val);
  if (!state)
    return Block();
  return state->getValue().getDefBlock();
}

} // namespace imex
