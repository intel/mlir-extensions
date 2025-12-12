#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/Debug.h>

#include "imex/Dialect/XeTile/Transforms/BlockingAnalysis.h"
#include "imex/Utils/XeCommon.h"

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
// For convience, it also tracks the use point (OpOperand) of the value.
class BlockingRequests {
public:
  BlockingRequests() = default;

  BlockingRequests(int64_t h, int64_t w, mlir::OpOperand &point)
      : BlockingRequests(Block(h, w), point) {}

  BlockingRequests(llvm::ArrayRef<int64_t> shape, mlir::OpOperand &point)
      : BlockingRequests(shape[0], shape[1], point) {
    assert(shape.size() == 2 && "Invalid block size.");
  }

  BlockingRequests(Block block, mlir::OpOperand &point);

  bool operator==(const BlockingRequests &other) const;
  bool operator!=(const BlockingRequests &other) const;

  Block getDefBlock() const;
  Block getUseBlock(mlir::OpOperand &point) const;

  void print(llvm::raw_ostream &os) const;

  static BlockingRequests meet(const BlockingRequests &lhs,
                               const BlockingRequests &rhs);

  static BlockingRequests join(const BlockingRequests &lhs,
                               const BlockingRequests &rhs);

  // indicate that one use of the result operand
  // has decided on the inner block size.
  bool isInitialized() const { return requests.size() != 0; }

  int64_t getNumUniqRequests() const { return getRequests().size(); }

  template <typename OpTy> bool hasOneUserOfType() const {
    return requests.size() == 1 && mlir::isa<OpTy>(getUser(0));
  }

  mlir::Operation *getUser(unsigned index) const {
    assert(index < requests.size() && "Index out of bounds.");
    auto it = requests.begin();
    std::advance(it, index);
    return it->first->getOwner();
  }

  llvm::SmallVector<Block> getRequests() const {
    llvm::SmallDenseSet<Block, 8> reqs;
    for (auto [point, block] : requests)
      reqs.insert(block);
    return llvm::SmallVector<Block>(reqs.begin(), reqs.end());
  }

  void updateDefBlock(Block block) { def = block; }
  void updateArrayLength(int length) { array_length = length; }
  int getArrayLength() const { return array_length; }

private:
  int array_length = 1;
  Block def;
  llvm::DenseMap<mlir::OpOperand *, Block> requests {};
};

BlockingRequests::BlockingRequests(Block block, mlir::OpOperand &point) {
  assert(block && "Invalid block.");
  requests.try_emplace(&point, block);
}

Block BlockingRequests::getDefBlock() const {
  if (def)
    return def;
  for (auto [p, req] : requests)
    if (req)
      return req;
  return Block();
}

Block BlockingRequests::getUseBlock(mlir::OpOperand &point) const {
  return requests.lookup(&point);
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
      os << "{User(" << *(point->getOwner()) << "), blk(" << block << ")}";
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
  BlockingRequests newReq;
  if (lhs.isInitialized()) {
    for (auto [point, block] : lhs.requests) {
      newReq.requests.try_emplace(point, block);
    }
    newReq.array_length = lhs.array_length;
  }
  if (rhs.isInitialized()) {
    for (auto [point, block] : rhs.requests) {
      newReq.requests.try_emplace(point, block);
    }
    newReq.array_length = std::max(newReq.array_length, rhs.array_length);
  }
  return newReq;
}

BlockingRequests BlockingRequests::join(const BlockingRequests &lhs,
                                        const BlockingRequests &rhs) {
  return meet(lhs, rhs);
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

  template <typename OpTy> bool hasOneUserOfType() const {
    return getValue().hasOneUserOfType<OpTy>();
  }
};

// ===----------------------BlockingAnalysisImpl ---------------------===//

static int64_t getBitWidth(mlir::Type elemTy) {
  assert(elemTy.isIntOrIndexOrFloat() &&
         "Expecting an int, index or float type.");
  // TODO: is it safe to treat index as 32 bit integer?
  // Assuming index vector is mainly used for gather/scatter ops on SLM.
  // in which the address is 32-bit.
  return elemTy.isIntOrFloat() ? elemTy.getIntOrFloatBitWidth() : 32;
}

// Find the largest divisor of v in the range [l, h] if there is any.
// Otherwise it will return 0. If u != 1, and v % u == 0, the returned
// divisor will be also multiple of u too.
static int64_t getDivisorInRange(int64_t v, int64_t l, int64_t h,
                                 int64_t u = 1) {
  auto divisable = u && v % u == 0;
  for (int i = h; i >= l; i--) {
    if (v % i == 0 && (!divisable || i % u == 0))
      return i;
  }
  // irregular shape or shape is not in the supported range.
  return 0;
}

// TODO: currently, we only support optimal cases for SLM access,
// and the block width is fixed to 16. That means the shape[1]
// has to be multiple of 16. The block shape for SLM is [h, 16].
// For colMajor, h/vnni is one of supported chunk sizes: 8, 4,
// 3, 2, 1. For rowMajor, h * 16 /vnni is one of supported chunk
// sizes: 64, 32, 16.
static Block getSLMBlock(xetile::TileType tileTy) {
  // the fallback pass would have already converted all
  // unsupported cases into scattered ops.
  assert(isSupportedOptimalSLMAccess(tileTy) && "Unsupported SLM access case.");
  auto elemTy = tileTy.getElementType();
  auto shape = tileTy.getShape();
  const int w = 16;
  auto vnni = getVnniFactor(elemTy);
  auto colMajor = isColMajorOrder(tileTy.getOrder());
  auto h = getHeightForSLMBlock(shape, w, vnni, colMajor);
  return Block(h, w);
}

class BlockingAnalysisImpl
    : public mlir::dataflow::SparseBackwardDataFlowAnalysis<BlockingLattice> {
public:
  BlockingAnalysisImpl(mlir::DataFlowSolver &solver,
                       mlir::SymbolTableCollection &symbolTable,
                       std::shared_ptr<XeuArchInterface> uArch)
      : SparseBackwardDataFlowAnalysis(solver, symbolTable), uArch(uArch) {}

  mlir::LogicalResult
  visitOperation(mlir::Operation *op,
                 mlir::ArrayRef<BlockingLattice *> operands,
                 mlir::ArrayRef<const BlockingLattice *> results) override;

  void visitBranchOperand(mlir::OpOperand &operand) override {}

  void visitCallOperand(mlir::OpOperand &operand) override {}

  void setToExitState(BlockingLattice *lattice) override {}

private:
  void visitInitTileOp(xetile::InitTileOp op,
                       mlir::ArrayRef<BlockingLattice *> operands,
                       mlir::ArrayRef<const BlockingLattice *> results);

  void visitPrefetchTileOp(xetile::PrefetchTileOp op,
                           mlir::ArrayRef<BlockingLattice *> operands,
                           mlir::ArrayRef<const BlockingLattice *> results);

  void visitLoadTileOp(xetile::LoadTileOp op,
                       mlir::ArrayRef<BlockingLattice *> operands,
                       mlir::ArrayRef<const BlockingLattice *> results);

  void visitStoreTileOp(xetile::StoreTileOp op,
                        mlir::ArrayRef<BlockingLattice *> operands,
                        mlir::ArrayRef<const BlockingLattice *> results);

  void visitLoadGatherOp(xetile::LoadGatherOp op,
                         mlir::ArrayRef<BlockingLattice *> operands,
                         mlir::ArrayRef<const BlockingLattice *> results);

  void visitStoreScatterOp(xetile::StoreScatterOp op,
                           mlir::ArrayRef<BlockingLattice *> operands,
                           mlir::ArrayRef<const BlockingLattice *> results);

  void visitAtomicRMWOp(xetile::AtomicRMWOp op,
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

  template <typename Integertype>
  Block getDefaultSize(mlir::Type elemTy, llvm::ArrayRef<Integertype> &shape);

  llvm::SmallVector<unsigned int>
  getMMASize(mlir::Type elemTy, const int APrecision, const int BPrecision,
             const int CPrecision, const int DPrecision);

private:
  std::shared_ptr<XeuArchInterface> uArch = nullptr;
};

mlir::LogicalResult BlockingAnalysisImpl::visitOperation(
    mlir::Operation *op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {

  if (auto initTileOp = mlir::dyn_cast<xetile::InitTileOp>(op))
    visitInitTileOp(initTileOp, operands, results);

  if (auto updateTileOp = mlir::dyn_cast<xetile::UpdateTileOffsetOp>(op))
    visitUpdateTileOp(updateTileOp, operands, results);

  if (auto prefetchOp = mlir::dyn_cast<xetile::PrefetchTileOp>(op))
    visitPrefetchTileOp(prefetchOp, operands, results);

  if (auto loadOp = mlir::dyn_cast<xetile::LoadTileOp>(op))
    visitLoadTileOp(loadOp, operands, results);

  if (auto gatherOp = mlir::dyn_cast<xetile::LoadGatherOp>(op))
    visitLoadGatherOp(gatherOp, operands, results);

  if (auto storeOp = mlir::dyn_cast<xetile::StoreTileOp>(op))
    visitStoreTileOp(storeOp, operands, results);

  if (auto scatterOp = mlir::dyn_cast<xetile::StoreScatterOp>(op))
    visitStoreScatterOp(scatterOp, operands, results);

  if (auto atomicrmwOp = mlir::dyn_cast<xetile::AtomicRMWOp>(op))
    visitAtomicRMWOp(atomicrmwOp, operands, results);

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

  return mlir::success();
}

void BlockingAnalysisImpl::visitInitTileOp(
    xetile::InitTileOp op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {
  auto lattice = results[0]->getValue();

  if (!lattice.isInitialized())
    return;

  assert(lattice.getNumUniqRequests() == 1 &&
         "InitTileOp should have only one request.");

  auto block = lattice.getRequests()[0];

  BlockingRequests &def = getLatticeElement(op.getTile())->getValue();
  def.updateDefBlock(block);

  // only work on scattered init_tile, which has indices
  if (op.getIndices()) {
    auto req = BlockingRequests(block, op->getOpOperand(1));
    propagateIfChanged(operands[1], operands[1]->join(req));
  }
}

void BlockingAnalysisImpl::visitPrefetchTileOp(
    xetile::PrefetchTileOp op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {
  auto tileTy = op.getTile().getType();
  auto elemBits = getBitWidth(tileTy.getElementType());
  auto config = uArch->get2DPrefetchConfig(op, elemBits);
  assert(mlir::succeeded(config) && "Failed to get prefetch config.");
  auto maxH = config->blockHeight.max;
  auto minH = config->blockHeight.min;
  auto maxW = config->blockWidth.max;
  auto minW = config->blockWidth.min;
  auto shape = tileTy.getShape();
  auto h = getDivisorInRange(shape[0], minH, maxH);
  auto w = getDivisorInRange(shape[1], minW, maxW);
  Block block(h, w);
  if (!block)
    return; // do nothing if didnot get a valid block size
  auto BlockingRequest = BlockingRequests(block, op->getOpOperand(0));
  propagateIfChanged(operands[0], operands[0]->join(BlockingRequest));
}

void BlockingAnalysisImpl::visitLoadTileOp(
    xetile::LoadTileOp op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {
  // temporary skip the load_tile with multiple results
  if (op.getValues().size() > 1)
    return;

  // It has users but users' requirements are not available yet.
  // Worth to wait until all users are visited.

  auto value = op.getValues()[0];
  auto lattice = results[0]->getValue();
  if (!value.use_empty() && !lattice.isInitialized())
    return;

  // TODO: currently, we only support one user requesting the blocking size.
  assert(lattice.getNumUniqRequests() <= 1 &&
         "multiple users requesting different blocking sizes.");

  auto tileTy = op.getTileType();
  auto shape = tileTy.getShape();
  auto memSpace = tileTy.getMemorySpaceAsInt();

  Block block;
  if (memSpace == 3) {
    block = getSLMBlock(tileTy);
  } else {
    // for global memory access, the block size majorly contrainted by the
    // hardware block load capability. For dim 1, it will try to get the
    // largest divisor for rq[1] if there is or shape[1] otherwise in the
    // range [minW, maxW]. For dim 0, it will try to get the largest divisor
    // of shape[0] in the range [minH, maxH] that is divisible by rq[0] if
    // there is, otherwise 1,
    auto elemTy = tileTy.getElementType();
    auto elemBits = getBitWidth(elemTy);
    bool hasTransposeUser = value.hasOneUse() &&
                            mlir::isa<xetile::TransposeOp>(*(op->user_begin()));
    auto transpose = elemBits >= 32 && hasTransposeUser;
    auto config = uArch->get2DLoadConfig(op, elemBits, false, transpose);
    assert(mlir::succeeded(config) && "Failed to get load config.");
    auto maxH = config->blockHeight.max;
    auto minH = config->blockHeight.min;
    auto maxW = config->blockWidth.max;
    auto minW = config->blockWidth.min;

    Block rq =
        lattice.isInitialized() ? lattice.getRequests()[0] : Block(1, shape[1]);
    int64_t w =
        std::min<int64_t>(rq[1], getDivisorInRange(shape[1], minW, maxW));
    int64_t h = getDivisorInRange(shape[0], minH, maxH, rq[0]);
    // for cases of load+transpose+dpas, the block height should be aligned
    // to minimize the data movement for dpas.
    if (hasTransposeUser)
      h = std::min<int64_t>(h, rq[0]);
    block = Block(h, w);
  }

  if (!block)
    return; // do nothing if didnot get a valid block size

  auto BlockingRequest = BlockingRequests(block, op->getOpOperand(0));

  if (memSpace != 3) {
    auto computeArrayLength = [&]() -> int {
      auto elemBits = getBitWidth(tileTy.getElementType());
      auto config = uArch->get2DLoadConfig(op, elemBits, false, false);
      if (mlir::succeeded(config) && block) {
        auto availableArrayLengths = config->array_length;
        // Do not let an inner block get array_length'ed to blocks
        // finer than one GRF.
        if (block[0] * block[1] * elemBits < uArch->getOneGRFSizeBits()) {
          return 1;
        }
        const int maxBlockWidth = std::min<int>(config->restriction, shape[1]);
        for (auto len : llvm::reverse(availableArrayLengths)) {
          if (len * block[1] <= maxBlockWidth &&
              (shape[1] / block[1]) % len == 0) {
            return len;
          }
        }
      }
      return 1;
    };
    BlockingRequest.updateArrayLength(computeArrayLength());
  }

  // propagate the blocking size to its def op
  propagateIfChanged(operands[0], operands[0]->join(BlockingRequest));

  // update the def block size for the result value
  BlockingRequests &def = getLatticeElement(op.getValues()[0])->getValue();
  def.updateDefBlock(block);
}

void BlockingAnalysisImpl::visitLoadGatherOp(
    xetile::LoadGatherOp op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {

  auto tileTy = op.getTile().getType();
  auto elemTy = tileTy.getElementType();
  auto shape = tileTy.getShape();

  // TODO: currently 1D gather is not considered.
  if (shape.size() == 1)
    return;

  auto size = getDefaultSize(elemTy, shape);
  if (!size)
    return;

  for (auto &&[i, inputOpr] : llvm::enumerate(operands)) {
    auto blockingRequest = BlockingRequests(size, op->getOpOperand(i));
    propagateIfChanged(inputOpr, inputOpr->join(blockingRequest));
  }
}

void BlockingAnalysisImpl::visitStoreTileOp(
    xetile::StoreTileOp op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {
  auto tileTy = op.getTile().getType();
  auto memSpace = tileTy.getMemorySpaceAsInt();
  Block block;
  if (memSpace == 3) {
    block = getSLMBlock(tileTy);
  } else {
    auto shape = tileTy.getShape();
    auto elemBits = getBitWidth(tileTy.getElementType());
    auto config = uArch->get2DStoreConfig(elemBits);
    assert(mlir::succeeded(config) && "Failed to get store config.");
    auto maxH = config->blockHeight.max;
    auto minH = config->blockHeight.min;
    auto maxW = config->blockWidth.max;
    auto minW = config->blockWidth.min;
    auto h = getDivisorInRange(shape[0], minH, maxH);
    auto w = getDivisorInRange(shape[1], minW, maxW);
    block = Block(h, w);
  }

  if (!block)
    return; // do nothing if didnot get a valid block size

  for (auto &&[i, opr] : llvm::enumerate(operands)) {
    auto blockingRequest = BlockingRequests(block, op->getOpOperand(i));
    propagateIfChanged(opr, opr->join(blockingRequest));
  }
}

void BlockingAnalysisImpl::visitStoreScatterOp(
    xetile::StoreScatterOp op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {
  auto tileTy = op.getTile().getType();
  auto elemTy = tileTy.getElementType();
  auto shape = tileTy.getShape();

  // TODO: currently 1D scatter is not considered.
  if (shape.size() == 1)
    return;

  auto size = getDefaultSize(elemTy, shape);
  if (!size)
    return;

  for (auto &&[i, inputOpr] : llvm::enumerate(operands)) {
    auto blockingRequest = BlockingRequests(size, op->getOpOperand(i));
    propagateIfChanged(inputOpr, inputOpr->join(blockingRequest));
  }
}

void BlockingAnalysisImpl::visitAtomicRMWOp(
    xetile::AtomicRMWOp op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {
  auto tileTy = op.getTile().getType();
  auto elemTy = tileTy.getElementType();
  auto shape = tileTy.getShape();

  auto size = getDefaultSize(elemTy, shape);
  if (!size)
    return;

  for (auto &&[i, inputOpr] : llvm::enumerate(operands)) {
    auto blockingRequest = BlockingRequests(size, op->getOpOperand(i));
    propagateIfChanged(inputOpr, inputOpr->join(blockingRequest));
  }
}

void BlockingAnalysisImpl::visitUpdateTileOp(
    xetile::UpdateTileOffsetOp op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {
  auto lattice = results[0]->getValue();
  if (lattice.isInitialized()) {
    auto block = lattice.getRequests()[0];
    auto request = BlockingRequests(block, op->getOpOperand(0));
    propagateIfChanged(operands[0], operands[0]->join(request));
    if (op.getIndices()) {
      auto request = BlockingRequests(block, op->getOpOperand(1));
      propagateIfChanged(operands[1], operands[1]->join(request));
    }
  }
}

void BlockingAnalysisImpl::visitTileMMAOp(
    xetile::TileMMAOp op, mlir::ArrayRef<BlockingLattice *> operands,
    mlir::ArrayRef<const BlockingLattice *> results) {

  auto C = op.getC();
  auto aPrecision = getBitWidth(op.getAType().getElementType());
  auto bPrecision = getBitWidth(op.getBType().getElementType());
  auto dPrecision = getBitWidth(op.getOutputType().getElementType());
  auto cPrecision = !C ? dPrecision : getBitWidth(C.getType().getElementType());

  auto mmaSize = getMMASize(op.getElementType(), aPrecision, bPrecision,
                            cPrecision, dPrecision);
  auto M = op.getAType().getShape()[0];
  auto blkM = getDivisorInRange(M, 1, mmaSize[0]);
  auto blockSizeForA = BlockingRequests(blkM, mmaSize[1], op->getOpOperand(0));
  auto blockSizeForB =
      BlockingRequests(mmaSize[1], mmaSize[2], op->getOpOperand(1));

  propagateIfChanged(operands[0], operands[0]->join(blockSizeForA));
  propagateIfChanged(operands[1], operands[1]->join(blockSizeForB));
  if (C) {
    auto blockSizeForC =
        BlockingRequests(blkM, mmaSize[2], op->getOpOperand(2));
    propagateIfChanged(operands[2], operands[2]->join(blockSizeForC));
  }

  // update the def block size for the result value
  BlockingRequests &def = getLatticeElement(op.getOutput())->getValue();
  def.updateDefBlock(Block(blkM, mmaSize[2]));
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
  auto size = getDefaultSize(elemTy, shape);
  if (!size)
    return; // do nothing if didnot get a valid block size

  auto blockingRequest = BlockingRequests(size, op->getOpOperand(0));
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

  auto req = lattice.getRequests()[0];
  auto dim = dims[0];
  Block blockSize;

  if (dim == 0) {
    blockSize = Block(1, req[1]);
  } else if (dim == 1) {
    blockSize = Block(1, 1);
  } else {
    return;
  }
  auto blockingRequest = BlockingRequests(blockSize, op->getOpOperand(0));
  propagateIfChanged(operands[0], operands[0]->join(blockingRequest));

  // update the def block size for the result value
  BlockingRequests &def = getLatticeElement(op.getResult())->getValue();
  def.updateDefBlock(Block(1, req[1]));
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

  // Wait for requests from users, unless all users are terminators
  if (!op->use_empty() && !lattice.isInitialized()) {
    for (auto user : op->getUsers()) {
      if (!user->hasTrait<mlir::OpTrait::ReturnLike>())
        return;
    }
  }

  auto srcTy = op.getVector().getType();
  auto shape = srcTy.getShape();
  auto elemTy = srcTy.getElementType();
  auto elemBits = getBitWidth(elemTy);

  int64_t minH = 1, maxH = shape[0];
  int64_t minW = 1, maxW = shape[1];
  auto defOp = op.getVector().getDefiningOp<xetile::LoadTileOp>();
  if (defOp && elemBits >= 32) {
    auto config = uArch->get2DLoadConfig(defOp, elemBits, false, true);
    minH = config->blockHeight.min;
    minW = config->blockWidth.min;
    // to be compatible with the SIMT instrinsic, the maximum height is
    // limited to 16, which is maximum supported value by SIMT instrinsic.
    maxH = std::min<int>(config->blockHeight.max, 16);
    maxW = config->blockWidth.max;
  }

  auto h = getDivisorInRange(shape[0], minH, maxH);
  auto w = getDivisorInRange(shape[1], minW, maxW);
  Block block(h, w);

  // TransposeOp determines its blocking size based on requests from
  // its users, by swapping the blocking size of its users.
  if (lattice.isInitialized()) {
    // TODO: handle multiple users
    if (lattice.getNumUniqRequests() == 1) {
      auto req = lattice.getRequests()[0];
      if (req[0] != 1 || req[1] != 1) {
        block = Block(req[1], req[0]);
      }
    }
  }

  if (!block)
    return; // do nothing if didnot get a valid block size

  auto request = BlockingRequests(block, op->getOpOperand(0));
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
  // TODO: only consider 2D shape for now
  if (!type || type.getRank() != 2)
    return;

  auto lattice = results[0]->getValue();
  auto elemTy = type.getElementType();
  auto shape = type.getShape();

  // elementwise operations are pretty flexiable on the block size.
  // But we expect its second dimension is register size aligned.
  Block block = getDefaultSize(elemTy, shape);
  block[0] = shape[0];

  // Wait for requests from users, unless all of its users are terminators.
  if (!op->use_empty() && !lattice.isInitialized()) {
    for (auto user : op->getUsers()) {
      if (!user->hasTrait<mlir::OpTrait::ReturnLike>() ||
          !mlir::isa<mlir::FunctionOpInterface>(user->getParentOp()))
        return;
    }
  }

  // TODO: special hack is needed for select op, becasue mismatch with
  // creat_mask will generate vector shuffles on i1 type, which is not
  // well supported by IGC yet. Using default size (same as CreateMask)
  // could help to avoid this. Remove it when lowering of create_mask
  // and IGC get matured.
  if (mlir::isa<mlir::arith::SelectOp>(op)) {
    block = Block(1, block[1]);
  }

  // elementwise operations are not sensitive to the block size.
  // It will use the block size requested by its users, except SelectOp
  if (lattice.isInitialized() && !mlir::isa<mlir::arith::SelectOp>(op)) {
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
    auto req = BlockingRequests(block, op->getOpOperand(i));
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
    auto BlockingRequest = BlockingRequests(shape, op->getOpOperand(0));
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

  if (!op->use_empty() && !lattice.isInitialized())
    return;

  BlockingRequests &def = getLatticeElement(op->getResult(0))->getValue();
  // TODO: following the Antonio's implementation and use the default size
  // [1, subgroupSize] for CreateMaskOp if 2D transform is not enabled.
  // If 2D transform is enabled, it will aligned with its users.
  Block block = getDefaultSize(elemTy, shape);

  // TODO: need to enable the following code after 2D lowering in
  // GPUToSPIRV is enabled.
  // for (auto &req : lattice.getRequests()) {
  //   block[0] = std::max(block[0], req[0]);
  //   block[1] = std::min(block[1], req[1]);
  // }
  def.updateDefBlock(block);
}

template <typename Integertype>
Block BlockingAnalysisImpl::getDefaultSize(mlir::Type elemTy,
                                           llvm::ArrayRef<Integertype> &shape) {
  const int64_t bits = getBitWidth(elemTy);
  const int64_t maxElems = uArch->getOneGRFSizeBits() / bits;
  auto width = getDivisorInRange(shape[1], 1, maxElems);
  return Block(1, width);
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
            mlir::OpOperand &p = op->getOpOperand(i);
            llvm::dbgs() << "\n   opr[" << i << "]: " << inputOpr
                         << " --> blkSZ: " << getUseBlockSize(inputOpr, p)
                         << ", arrayLen: " << getArrayLength(inputOpr);
          }
        }

        for (auto [i, res] : llvm::enumerate(op->getResults()))
          llvm::dbgs() << "\n   res[" << i << "]: " << res
                       << " --> blkSZ: " << getDefBlockSize(res)
                       << ", arrayLen: " << getArrayLength(res);
        llvm::dbgs() << "\n";
      }
    } else if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      llvm::dbgs() << "\nOp: " << op->getName();
      for (auto [i, arg] : llvm::enumerate(forOp.getRegionIterArgs()))
        llvm::dbgs() << "\n   arg[" << i << "]: "
                     << " --> blkSZ: " << getDefBlockSize(arg)
                     << ", arrayLen: " << getArrayLength(arg);

      for (auto [i, res] : llvm::enumerate(forOp.getResults()))
        llvm::dbgs() << "\n   res[" << i << "]: "
                     << " --> blkSZ: " << getDefBlockSize(res)
                     << ", arrayLen: " << getArrayLength(res);
      llvm::dbgs() << "\n";
    } else if (auto YieldOp = mlir::dyn_cast<mlir::scf::YieldOp>(op)) {
      llvm::dbgs() << "\nOp: " << op->getName();
      for (auto [i, res] : llvm::enumerate(YieldOp.getResults()))
        llvm::dbgs() << "\n   res[" << i << "]: " << res
                     << " --> blkSZ: " << getDefBlockSize(res) << ", "
                     << getUseBlockSize(res, op->getOpOperand(i))
                     << ", arrayLen: " << getArrayLength(res);
      llvm::dbgs() << "\n";
    } else if (auto StoreOp = mlir::dyn_cast<xetile::StoreTileOp>(op)) {
      llvm::dbgs() << "\nOp: " << *op;
      for (auto [i, inputOpr] : llvm::enumerate(op->getOperands())) {
        llvm::dbgs() << "\n   opr[" << i << "]: " << inputOpr << " --> blkSZ: "
                     << getUseBlockSize(inputOpr, op->getOpOperand(i))
                     << ", arrayLen: " << getArrayLength(inputOpr);
      }
      llvm::dbgs() << "\n";
    } else if (auto WhileOp = mlir::dyn_cast<mlir::scf::WhileOp>(op)) {
      llvm::dbgs() << "\nOp: " << op->getName();
      for (auto [i, arg] : llvm::enumerate(WhileOp.getBefore().getArguments()))
        llvm::dbgs() << "\n   before arg[" << i << "]: "
                     << " --> blkSZ: " << getDefBlockSize(arg)
                     << ", arrayLen: " << getArrayLength(arg);
      for (auto [i, arg] : llvm::enumerate(WhileOp.getAfter().getArguments()))
        llvm::dbgs() << "\n   after arg[" << i << "]: "
                     << " --> blkSZ: " << getDefBlockSize(arg)
                     << ", arrayLen: " << getArrayLength(arg);
      llvm::dbgs() << "\n";
    }
  });
}

Block BlockingAnalysis::getUseBlockSize(mlir::Value val,
                                        mlir::OpOperand &point) const {
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

int BlockingAnalysis::getArrayLength(mlir::Value val) const {
  auto *state = solver.lookupState<BlockingLattice>(val);
  if (!state)
    return 1;
  return state->getValue().getArrayLength();
}

} // namespace imex
