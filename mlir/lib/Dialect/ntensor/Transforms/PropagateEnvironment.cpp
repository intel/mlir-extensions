// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Dialect/ntensor/Transforms/PropagateEnvironment.hpp"

#include "imex/Dialect/ntensor/IR/NTensorOps.hpp"

#include <llvm/Support/Debug.h>
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#define DEBUG_TYPE "env-propagation"

namespace {
static bool needUpdate(mlir::Operation *op) {
  assert(op && "Invalid op");
  return mlir::isa<imex::ntensor::NTensorDialect>(op->getDialect());
}

static bool needPropagation(mlir::Operation *op) {
  assert(op && "Invalid op");
  return needUpdate(op) || mlir::isa<mlir::arith::SelectOp>(op);
}

static llvm::Optional<mlir::Attribute> getTensorEnv(mlir::Value val) {
  if (auto tensor = val.getType().dyn_cast<imex::ntensor::NTensorType>())
    return tensor.getEnvironment();

  return std::nullopt;
}

class EnvValue {
  using Storage = llvm::PointerUnion<mlir::Attribute, mlir::StringAttr>;

public:
  EnvValue() = default;

  explicit EnvValue(mlir::Attribute env_) : env(env_) {}

  explicit EnvValue(mlir::StringAttr desc) : env(desc) {}

  static EnvValue getInvalid(Storage lhs, Storage rhs) {
    auto lhsEnv = lhs.get<mlir::Attribute>();
    assert(lhsEnv);
    auto rhsEnv = rhs.get<mlir::Attribute>();
    assert(rhsEnv);

    std::string str;
    llvm::raw_string_ostream os(str);
    os << "Incompatible envs: ";
    os << lhsEnv << " and " << rhsEnv;
    os.flush();

    auto ctx = lhsEnv.getContext();
    return EnvValue(mlir::StringAttr::get(ctx, str));
  }

  /// Whether the state is uninitialized.
  bool isUninitialized() const { return env.isNull(); }
  bool isInvalid() const { return env.is<mlir::StringAttr>(); }

  mlir::Attribute getEnv() const {
    assert(!isUninitialized());
    assert(!isInvalid());
    return env.get<mlir::Attribute>();
  }

  llvm::StringRef getInvalidReason() const {
    assert(isInvalid());
    return env.get<mlir::StringAttr>().getValue();
  }

  static EnvValue join(const EnvValue &lhs, const EnvValue &rhs) {
    if (lhs.isInvalid() || rhs.isInvalid())
      return lhs.isInvalid() ? lhs.env : rhs.env;

    if (lhs.isUninitialized())
      return rhs;

    if (rhs.isUninitialized())
      return lhs;

    if (!lhs.getEnv())
      return rhs;

    if (!rhs.getEnv())
      return lhs;

    return lhs.env == rhs.env ? lhs : getInvalid(lhs.env, rhs.env);
  }

  bool operator==(const EnvValue &rhs) const { return env == rhs.env; }

  void print(llvm::raw_ostream &os) const {
    if (isUninitialized()) {
      os << "None";
    } else if (isInvalid()) {
      os << "Invalid: " << getInvalidReason();
    } else {
      os << env.get<mlir::Attribute>();
    }
  }

private:
  EnvValue(Storage env_) : env(env_) {}

  Storage env;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const EnvValue &state) {
  state.print(os);
  return os;
}

struct EnvValueLattice : public mlir::dataflow::Lattice<EnvValue> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EnvValueLattice)
  using Lattice::Lattice;
};

class EnvValueAnalysis
    : public mlir::dataflow::SparseDataFlowAnalysis<EnvValueLattice> {
public:
  using SparseDataFlowAnalysis::SparseDataFlowAnalysis;

  void visitOperation(mlir::Operation *op,
                      llvm::ArrayRef<const EnvValueLattice *> operands,
                      llvm::ArrayRef<EnvValueLattice *> results) override {
    LLVM_DEBUG(llvm::dbgs()
               << "EnvValueAnalysis: Visiting operation: " << *op << "\n");

    if (!needPropagation(op))
      return setAllToEntryStates(results);

    assert(operands.size() == op->getNumOperands() && "Invalid operands count");
    EnvValue env(mlir::Attribute{});
    for (auto [argLattice, origArg] : llvm::zip(operands, op->getOperands())) {
      if (auto tensorEnv = getTensorEnv(origArg)) {
        auto &latticeVal = argLattice->getValue();
        if (!latticeVal.isUninitialized())
          env = EnvValue::join(env, latticeVal);

        env = EnvValue::join(env, EnvValue(*tensorEnv));
      }
    }

    assert(results.size() == op->getNumResults() && "Invalid results count");
    for (auto result : op->getResults()) {
      if (auto tensorEnv = getTensorEnv(result))
        env = EnvValue::join(env, EnvValue(*tensorEnv));
    }

    LLVM_DEBUG(llvm::dbgs()
               << "EnvValueAnalysis: Operation deduced env: " << env << "\n");

    for (auto [resultLattice, result] : llvm::zip(results, op->getResults())) {
      if (getTensorEnv(result)) {
        auto changed = resultLattice->join(env);
        propagateIfChanged(resultLattice, changed);
      }
    }
  }

  void setToEntryState(EnvValueLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(EnvValue{}));
  }
};

struct PropagateEnvironmentPass
    : public mlir::PassWrapper<PropagateEnvironmentPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PropagateEnvironmentPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<imex::ntensor::NTensorDialect>();
  }

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "PropagateEnvironmentPass:\n");
    auto *root = getOperation();

    mlir::DataFlowSolver solver;
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<mlir::dataflow::SparseConstantPropagation>();
    solver.load<EnvValueAnalysis>();
    if (mlir::failed(solver.initializeAndRun(root)))
      return signalPassFailure();

    bool failed = false;
    llvm::SmallVector<std::pair<mlir::Operation *, mlir::Attribute>, 0>
        opsToProcess;
    root->walk([&](mlir::Operation *op) {
      if (!needUpdate(op))
        return;

      mlir::Attribute env;
      for (auto args : {mlir::ValueRange(op->getOperands()),
                        mlir::ValueRange(op->getResults())}) {
        for (auto arg : args) {
          auto *state = solver.lookupState<EnvValueLattice>(arg);
          if (!state)
            continue;

          if (auto srcEnv = getTensorEnv(arg))
            env = *srcEnv;

          auto &val = state->getValue();
          if (val.isUninitialized())
            continue;

          if (val.isInvalid()) {
            op->emitError(val.getInvalidReason());
            failed = true;
            return signalPassFailure();
          }

          if (auto newEnv = val.getEnv()) {
            if (!env) {
              env = newEnv;
            } else if (env != newEnv) {
              op->emitError("Enviroment type conflict: ")
                  << env << " " << newEnv;
              failed = true;
              return signalPassFailure();
            }
          }
        }
      }

      opsToProcess.emplace_back(op, env);
    });

    if (failed)
      return;

    mlir::OpBuilder builder(&getContext());
    for (auto [op, env] : opsToProcess) {
      assert(op);
      if (mlir::isa<imex::ntensor::CastOp>(op))
        continue;

      auto loc = op->getLoc();
      builder.setInsertionPoint(op);
      for (auto &operand : op->getOpOperands()) {
        auto arg = operand.get();
        auto tensor = arg.getType().dyn_cast<imex::ntensor::NTensorType>();
        if (!tensor)
          continue;

        if (tensor.getEnvironment() == env)
          continue;

        auto newType = imex::ntensor::NTensorType::get(tensor.getShape(),
                                                       tensor.getElementType(),
                                                       env, tensor.getLayout());
        mlir::Value newVal =
            builder.createOrFold<imex::ntensor::CastOp>(loc, newType, arg);
        operand.set(newVal);
      }

      builder.setInsertionPointAfter(op);
      for (auto res : op->getResults()) {
        auto tensor = res.getType().dyn_cast<imex::ntensor::NTensorType>();
        if (!tensor)
          continue;

        if (tensor.getEnvironment() == env)
          continue;

        auto newType = imex::ntensor::NTensorType::get(tensor.getShape(),
                                                       tensor.getElementType(),
                                                       env, tensor.getLayout());
        res.setType(newType);
        auto newRes = builder.create<imex::ntensor::CastOp>(loc, tensor, res);
        res.replaceAllUsesExcept(newRes.getResult(), newRes);
      }
    }

    root->walk([&](mlir::FunctionOpInterface func) {
      if (func.isExternal())
        return;

      auto funcType = func.getFunctionType().cast<mlir::FunctionType>();
      auto argsTypes = funcType.getInputs();

      llvm::SmallVector<mlir::Type> newArgsTypes(argsTypes.begin(),
                                                 argsTypes.end());

      auto &entryBlock = func.getFunctionBody().front();

      bool changed = false;
      builder.setInsertionPointToStart(&entryBlock);
      for (auto [i, arg] : llvm::enumerate(entryBlock.getArguments())) {
        auto oldType = arg.getType().dyn_cast<imex::ntensor::NTensorType>();
        if (!oldType)
          continue;

        auto *state = solver.lookupState<EnvValueLattice>(arg);
        if (!state)
          continue;

        auto &val = state->getValue();
        if (val.isUninitialized())
          continue;

        if (val.isInvalid()) {
          func->emitError(val.getInvalidReason());
          return signalPassFailure();
        }

        auto newEnv = val.getEnv();
        if (newEnv == getTensorEnv(arg))
          continue;

        auto newType = imex::ntensor::NTensorType::get(
            oldType.getShape(), oldType.getElementType(), newEnv,
            oldType.getLayout());

        auto cast =
            builder.create<imex::ntensor::CastOp>(arg.getLoc(), oldType, arg);
        arg.replaceAllUsesExcept(cast.getResult(), cast);
        arg.setType(newType);
        newArgsTypes[i] = newType;
        changed = true;
      }

      if (!changed)
        return;

      auto newFuncType = mlir::FunctionType::get(&getContext(), newArgsTypes,
                                                 funcType.getResults());
      func.setType(newFuncType);

      auto uses = mlir::SymbolTable::getSymbolUses(func, root);
      if (!uses)
        return;

      for (auto use : *uses) {
        auto user = use.getUser();
        auto callOp = mlir::dyn_cast<mlir::CallOpInterface>(user);
        if (!callOp) {
          user->emitError("Not a call op");
          return signalPassFailure();
          failed = true;
        }

        auto loc = user->getLoc();
        builder.setInsertionPoint(user);
        for (auto [i, arg] : llvm::enumerate(callOp.getArgOperands())) {
          auto oldType = arg.getType();
          auto newType = newArgsTypes[i];
          if (oldType == newType)
            continue;

          auto cast = builder.create<imex::ntensor::CastOp>(loc, newType, arg);
          callOp->setOperand(i, cast.getResult());
        }
      }
    });

    if (failed)
      return;

    // Run some cleanup
    mlir::RewritePatternSet patterns(&getContext());
    imex::ntensor::CastOp::getCanonicalizationPatterns(patterns, &getContext());
    (void)mlir::applyPatternsAndFoldGreedily(root, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<mlir::Pass> imex::ntensor::createPropagateEnvironmentPass() {
  return std::make_unique<PropagateEnvironmentPass>();
}
