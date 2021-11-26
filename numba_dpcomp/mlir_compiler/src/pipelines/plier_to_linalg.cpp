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

#include "pipelines/plier_to_linalg.hpp"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include <mlir/Analysis/AffineAnalysis.h>

#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/Passes.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/SCF/Transforms.h>
#include <mlir/Dialect/StandardOps/Transforms/FuncConversions.h>
#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tensor/Transforms/Passes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/LoopUtils.h>
#include <mlir/Transforms/Passes.h>

#include "plier/dialect.hpp"

#include "pipelines/plier_to_scf.hpp"
#include "pipelines/plier_to_std.hpp"
#include "pipelines/pre_low_simplifications.hpp"

#include "plier/Conversion/SCFToAffine/SCFToAffine.h"
#include "plier/pass/rewrite_wrapper.hpp"
#include "plier/rewrites/call_lowering.hpp"
#include "plier/rewrites/canonicalize_reductions.hpp"
#include "plier/rewrites/common_opts.hpp"
#include "plier/rewrites/cse.hpp"
#include "plier/rewrites/force_inline.hpp"
#include "plier/rewrites/loop_rewrites.hpp"
#include "plier/rewrites/memory_rewrites.hpp"
#include "plier/rewrites/promote_to_parallel.hpp"
#include "plier/rewrites/type_conversion.hpp"
#include "plier/rewrites/uplift_math_calls.hpp"
#include "plier/transforms/cast_utils.hpp"
#include "plier/transforms/const_utils.hpp"
#include "plier/transforms/loop_utils.hpp"
#include "plier/transforms/pipeline_utils.hpp"

#include "base_pipeline.hpp"
#include "loop_utils.hpp"
#include "mangle.hpp"
#include "plier/compiler/pipeline_registry.hpp"
#include "py_func_resolver.hpp"
#include "py_linalg_resolver.hpp"

#include <cctype>

namespace {
int64_t getOptLevel(mlir::Operation *op) {
  assert(op);
  auto attr = op->getAttr(plier::attributes::getOptLevelName())
                  .dyn_cast_or_null<mlir::IntegerAttr>();
  if (!attr) {
    return 0;
  }
  return std::max(static_cast<int64_t>(0), attr.getInt());
}

mlir::LogicalResult applyOptimizations(
    mlir::FuncOp op, const mlir::FrozenRewritePatternSet &patterns,
    mlir::AnalysisManager am,
    llvm::function_ref<mlir::LogicalResult(mlir::FuncOp)> additionalOpts =
        nullptr) {
  bool repeat = false;
  do {
    repeat = false;
    (void)mlir::applyPatternsAndFoldGreedily(op, patterns);
    if (mlir::succeeded(plier::applyCSE(op.getRegion(), false))) {
      repeat = true;
    }

    auto memOptRes = plier::optimizeMemoryOps(am);
    if (!memOptRes) {
      op.emitError() << "Failed to build memssa analysis";
      return mlir::failure();
    }
    if (mlir::succeeded(*memOptRes)) {
      repeat = true;
    }

    if (additionalOpts && mlir::succeeded(additionalOpts(op))) {
      repeat = true;
    }
    if (repeat) {
      am.invalidate({});
    }
  } while (repeat);
  return mlir::success();
}

enum class ArrayLayout { C, F, A };

bool parse_layout(llvm::StringRef &name, ArrayLayout &layout) {
  if (name.consume_back("C")) {
    layout = ArrayLayout::C;
    return true;
  }
  if (name.consume_back("F")) {
    layout = ArrayLayout::F;
    return true;
  }
  if (name.consume_back("A")) {
    layout = ArrayLayout::A;
    return true;
  }
  return false;
}

template <typename T> bool consume_int_back(llvm::StringRef &name, T &result) {
  unsigned len = 0;
  auto tmp_name = name;
  while (!tmp_name.empty() && std::isdigit(tmp_name.back())) {
    ++len;
    tmp_name = tmp_name.drop_back();
  }
  tmp_name = name.substr(name.size() - len);
  if (!tmp_name.consumeInteger<T>(10, result)) {
    name = name.substr(0, name.size() - len);
    return true;
  }
  return false;
}

struct ArrayDesc {
  unsigned dims = 0;
  ArrayLayout layout = {};
  llvm::StringRef name;
};

llvm::Optional<ArrayDesc> parse_array_desc(llvm::StringRef &name) {
  unsigned num_dims = 0;
  ArrayLayout layout = {};
  if (name.consume_front("array(") && name.consume_back(")") &&
      parse_layout(name, layout) && name.consume_back(", ") &&
      name.consume_back("d") && consume_int_back(name, num_dims) &&
      name.consume_back(", ") && !name.empty()) {
    return ArrayDesc{num_dims, layout, name};
  }
  return {};
}

mlir::Type map_array_type(mlir::MLIRContext &ctx, mlir::TypeConverter &conveter,
                          llvm::StringRef &name) {
  if (auto desc = parse_array_desc(name)) {
    if (desc->layout == ArrayLayout::C || desc->layout == ArrayLayout::F ||
        desc->layout == ArrayLayout::A) {
      if (auto type =
              conveter.convertType(plier::PyType::get(&ctx, desc->name))) {
        llvm::SmallVector<int64_t> shape(desc->dims, -1);
        return mlir::RankedTensorType::get(shape, type);
      }
    }
  }
  return nullptr;
}

mlir::Type map_plier_type(mlir::TypeConverter &converter, mlir::Type type) {
  if (auto pyType = type.dyn_cast<plier::PyType>()) {
    auto name = pyType.getName();
    return map_array_type(*type.getContext(), converter, name);
  }
  return nullptr;
}

void rerun_scf_pipeline(mlir::Operation *op) {
  assert(nullptr != op);
  auto marker =
      mlir::StringAttr::get(op->getContext(), plierToScfPipelineName());
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  assert(nullptr != mod);
  plier::add_pipeline_jump_marker(mod, marker);
}

static mlir::Value skipCasts(mlir::Value val) {
  auto getArg = [](mlir::Value arg) -> mlir::Value {
    auto cast = arg.getDefiningOp<mlir::UnrealizedConversionCastOp>();
    if (!cast)
      return {};

    auto inputs = cast.inputs();
    if (inputs.size() != 1)
      return {};

    return inputs.front();
  };
  while (auto arg = getArg(val))
    val = arg;

  return val;
};

mlir::LogicalResult
lowerPrange(plier::PyCallOp op, mlir::ValueRange operands,
            llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs,
            mlir::PatternRewriter &rewriter) {
  auto parent = op->getParentOp();
  auto setAttr = [](mlir::scf::ForOp op) {
    op->setAttr(plier::attributes::getParallelName(),
                mlir::UnitAttr::get(op->getContext()));
  };
  if (mlir::succeeded(lowerRange(op, operands, kwargs, rewriter, setAttr))) {
    rerun_scf_pipeline(parent);
    return mlir::success();
  }
  return mlir::failure();
}

mlir::LogicalResult
lowerLen(plier::PyCallOp op, mlir::ValueRange operands,
         llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs,
         mlir::PatternRewriter &rewriter) {
  if (operands.size() != 1 || !kwargs.empty())
    return mlir::failure();

  auto arg = skipCasts(operands.front());
  if (!arg.getType().isa<mlir::RankedTensorType>())
    return mlir::failure();

  rerun_scf_pipeline(op);

  auto loc = op.getLoc();
  auto dim = rewriter.createOrFold<mlir::tensor::DimOp>(loc, arg, 0);
  rewriter.replaceOpWithNewOp<plier::CastOp>(op, op.getType(), dim);
  return mlir::success();
}

using kwargs_t = llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>>;
using func_t = mlir::LogicalResult (*)(plier::PyCallOp, mlir::ValueRange,
                                       kwargs_t, mlir::PatternRewriter &);
static const std::pair<llvm::StringRef, func_t> builtinFuncsHandlers[] = {
    // clang-format off
    {"numba.prange", lowerPrange},
    {"len", lowerLen},
    // clang-format on
};

struct NumpyCallsLowering final : public plier::CallOpLowering {
  NumpyCallsLowering(mlir::MLIRContext *context)
      : CallOpLowering(context),
        resolver("numba_dpcomp.mlir.numpy.funcs", "registry") {}

protected:
  virtual mlir::LogicalResult
  resolveCall(plier::PyCallOp op, mlir::StringRef name, mlir::Location loc,
              mlir::PatternRewriter &rewriter, mlir::ValueRange args,
              KWargs kwargs) const override {
    for (auto &handler : builtinFuncsHandlers)
      if (handler.first == name)
        return handler.second(op, args, kwargs, rewriter);

    auto res = resolver.rewriteFunc(name, loc, rewriter, args, kwargs);
    if (!res)
      return mlir::failure();

    auto results = std::move(res).getValue();
    assert(results.size() == op->getNumResults());
    for (auto it : llvm::enumerate(results)) {
      auto i = it.index();
      auto r = it.value();
      auto dstType = op->getResultTypes()[i];
      if (dstType != r.getType())
        results[i] = rewriter.create<plier::CastOp>(loc, dstType, r);
    }

    rerun_scf_pipeline(op);
    rewriter.replaceOp(op, results);
    return mlir::success();
  }

private:
  PyLinalgResolver resolver;
};

struct NumpyAttrsLowering : public mlir::OpRewritePattern<plier::GetattrOp> {
  NumpyAttrsLowering(mlir::MLIRContext *context)
      : OpRewritePattern(context),
        resolver("numba_dpcomp.mlir.numpy.funcs", "registry") {}

  mlir::LogicalResult
  matchAndRewrite(plier::GetattrOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto arg = skipCasts(op.value());

    if (!arg.getType().isa<mlir::ShapedType>())
      return mlir::failure();

    auto loc = op.getLoc();
    auto res = resolver.rewriteAttr(llvm::Twine("array.") + op.name(), loc,
                                    rewriter, arg);
    if (!res)
      return mlir::failure();

    auto results = *res;
    assert(results.size() == op->getNumResults());
    for (auto it : llvm::enumerate(results)) {
      auto i = it.index();
      auto r = it.value();
      auto dstType = op->getResultTypes()[i];
      if (dstType != r.getType())
        results[i] = rewriter.create<plier::CastOp>(loc, dstType, r);
    }

    rerun_scf_pipeline(op);
    rewriter.replaceOp(op, results);
    return mlir::success();
  }

private:
  PyLinalgResolver resolver;
};

struct NumpyBinOpLowering : public mlir::OpRewritePattern<plier::BinOp> {
  NumpyBinOpLowering(mlir::MLIRContext *context)
      : OpRewritePattern(context),
        resolver("numba_dpcomp.mlir.numpy.funcs", "registry") {}

  mlir::LogicalResult
  matchAndRewrite(plier::BinOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto lhs = skipCasts(op.lhs());
    auto rhs = skipCasts(op.rhs());

    if (!lhs.getType().isa<mlir::ShapedType>() &&
        !rhs.getType().isa<mlir::ShapedType>())
      return mlir::failure();

    auto name = op.op();

    for (auto it : plier::getOperators()) {
      if (it.op == name) {
        auto res = resolver.rewriteFunc(llvm::Twine("operator.") + it.name,
                                        op.getLoc(), rewriter, {lhs, rhs}, {});
        if (!res)
          return mlir::failure();

        rerun_scf_pipeline(op);
        rewriter.replaceOp(op, *res);
        return mlir::success();
      }
    }
    return mlir::failure();
  }

private:
  PyLinalgResolver resolver;
};

// TODO: remove
struct ExternalCallsLowering final : public plier::CallOpLowering {
  using CallOpLowering::CallOpLowering;

protected:
  virtual mlir::LogicalResult
  resolveCall(plier::PyCallOp op, mlir::StringRef name, mlir::Location loc,
              mlir::PatternRewriter &rewriter, mlir::ValueRange args,
              KWargs kwargs) const override {
    if (!kwargs.empty())
      return mlir::failure(); // TODO: kwargs support

    auto types = args.getTypes();
    auto mangledName = mangle(name, types);
    if (mangledName.empty())
      return mlir::failure();

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    assert(mod);
    auto externalFunc = mod.lookupSymbol<mlir::FuncOp>(mangledName);
    if (!externalFunc) {
      externalFunc = resolver.getFunc(name, types);
      if (externalFunc) {
        externalFunc.setPrivate();
        externalFunc.setName(mangledName);
      }
    }
    if (!externalFunc)
      return mlir::failure();

    assert(externalFunc.getType().getNumResults() == op->getNumResults());

    llvm::SmallVector<mlir::Value> castedArgs(args.size());
    auto funcTypes = externalFunc.getType().getInputs();
    for (auto it : llvm::enumerate(args)) {
      auto arg = it.value();
      auto i = it.index();
      auto dstType = funcTypes[i];
      if (arg.getType() != dstType)
        castedArgs[i] = rewriter.createOrFold<plier::CastOp>(loc, dstType, arg);
      else
        castedArgs[i] = arg;
    }

    auto newFuncCall =
        rewriter.create<mlir::CallOp>(loc, externalFunc, castedArgs);

    auto results = newFuncCall.getResults();
    llvm::SmallVector<mlir::Value> castedResults(results.size());

    for (auto it : llvm::enumerate(results)) {
      auto i = static_cast<unsigned>(it.index());
      auto res = it.value();
      auto oldResType = op->getResult(i).getType();
      if (res.getType() != oldResType)
        castedResults[i] =
            rewriter.createOrFold<plier::CastOp>(loc, oldResType, res);
      else
        castedResults[i] = res;
    }

    rerun_scf_pipeline(op);
    rewriter.replaceOp(op, castedResults);
    return mlir::success();
  }

private:
  PyFuncResolver resolver;
};

struct UnrankedToElementCasts
    : public mlir::OpConversionPattern<plier::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::CastOp op, plier::CastOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto value = adaptor.value();
    auto srcType = value.getType();
    auto converter = *getTypeConverter();
    auto dstType = converter.convertType(op.getType());
    if (!dstType)
      return mlir::failure();

    auto isCompatible = [](mlir::Type tensor, mlir::Type element) {
      if (auto tensorType = tensor.dyn_cast<mlir::RankedTensorType>())
        return tensorType.getRank() == 0 &&
               tensorType.getElementType() == element;

      return false;
    };
    if (isCompatible(srcType, dstType)) {
      rewriter.replaceOpWithNewOp<mlir::tensor::ExtractOp>(op, value);
      return mlir::success();
    }
    if (isCompatible(dstType, srcType)) {
      auto singleElemTensor = rewriter.create<mlir::tensor::FromElementsOp>(
          op.getLoc(), op.value());
      rewriter.replaceOpWithNewOp<mlir::linalg::TensorCollapseShapeOp>(
          op, dstType, singleElemTensor,
          llvm::ArrayRef<mlir::ReassociationExprs>{});
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct NumpyCallsLoweringPass
    : public plier::RewriteWrapperPass<
          NumpyCallsLoweringPass, void, void, NumpyCallsLowering,
          NumpyAttrsLowering, NumpyBinOpLowering, ExternalCallsLowering> {};

static mlir::Value index_cast(mlir::Value value, mlir::Location loc,
                              mlir::OpBuilder &builder) {
  if (!value.getType().isa<mlir::IndexType>()) {
    auto indexType = mlir::IndexType::get(value.getContext());
    auto res = builder.create<plier::CastOp>(loc, indexType, value);
    rerun_scf_pipeline(res);
    return res;
  }
  return value;
}

static bool isValidGetitemIndex(mlir::Type type) {
  if (type.isa<plier::SliceType>())
    return true;

  if (auto tupleType = type.dyn_cast<mlir::TupleType>())
    return llvm::all_of(tupleType.getTypes(), &isValidGetitemIndex);

  return type.isa<mlir::IntegerType, mlir::IndexType, plier::LiteralType>();
}

struct GetitemOpLowering : public mlir::OpConversionPattern<plier::GetItemOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::GetItemOp op, plier::GetItemOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto value = adaptor.value();
    auto index = adaptor.index();
    auto type = value.getType();
    bool isMemref = type.isa<mlir::MemRefType>();
    bool isTensor = type.isa<mlir::TensorType>();

    if (!isMemref && !isTensor)
      return mlir::failure();

    if (!isValidGetitemIndex(index.getType()))
      return mlir::failure();

    auto loc = op.getLoc();
    auto indexType = rewriter.getIndexType();
    auto getPos =
        [&](mlir::Value indexVal,
            unsigned dim) -> std::tuple<mlir::OpFoldResult, mlir::OpFoldResult,
                                        mlir::OpFoldResult, bool> {
      auto valType = indexVal.getType();
      if (auto sliceType = valType.dyn_cast<plier::SliceType>()) {
        auto getItemOrConst = [&](unsigned i) -> mlir::Value {
          assert(i < 3);
          auto createInd = [&](int64_t i) {
            return rewriter.create<mlir::arith::ConstantIndexOp>(loc, i);
          };
          return rewriter.create<plier::SliceGetItemOp>(
              loc, indexType, indexVal, value, createInd(i), dim);
        };
        auto offset = getItemOrConst(0);
        auto end = getItemOrConst(1);
        auto stride = getItemOrConst(2);
        auto size =
            rewriter.create<mlir::arith::SubIOp>(loc, end, offset).getResult();
        return {offset, size, stride, true};
      } else if (auto literal = valType.dyn_cast<plier::LiteralType>()) {
        auto offset = literal.getValue();
        return {offset, rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                false};
      } else {
        auto offset = index_cast(indexVal, loc, rewriter);
        return {offset, rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                false};
      }
    };

    auto makeFullSlice =
        [&](unsigned dim) -> std::tuple<mlir::OpFoldResult, mlir::OpFoldResult,
                                        mlir::OpFoldResult> {
      auto begin = rewriter.getIndexAttr(0);
      auto end = rewriter.createOrFold<mlir::tensor::DimOp>(loc, value, dim);
      auto step = rewriter.getIndexAttr(1);
      return {begin, end, step};
    };

    auto shapedType = type.cast<mlir::ShapedType>();
    auto rank = static_cast<unsigned>(shapedType.getRank());
    llvm::SmallVector<mlir::OpFoldResult> offsets(rank);
    llvm::SmallVector<mlir::OpFoldResult> sizes(rank);
    llvm::SmallVector<mlir::OpFoldResult> strides(rank);
    llvm::SmallVector<unsigned> dimsIndices;
    if (auto tupleType = index.getType().dyn_cast<mlir::TupleType>()) {
      auto count = static_cast<unsigned>(tupleType.size());
      if (count > rank)
        return mlir::failure();

      for (auto it : llvm::enumerate(tupleType)) {
        auto i = it.index();
        auto getitemInd =
            rewriter.create<mlir::arith::ConstantIndexOp>(loc, it.index());
        auto ind = rewriter.create<plier::GetItemOp>(loc, it.value(), index,
                                                     getitemInd);
        bool isSlice = false;
        std::tie(offsets[i], sizes[i], strides[i], isSlice) =
            getPos(ind.getResult(), static_cast<unsigned>(i));
        if (isSlice)
          dimsIndices.emplace_back(i);
      }

      for (auto i : llvm::seq(count, rank)) {
        std::tie(offsets[i], sizes[i], strides[i]) = makeFullSlice(i);
        dimsIndices.emplace_back(i);
      }
    } else {
      bool isSlice = false;
      std::tie(offsets[0], sizes[0], strides[0], isSlice) = getPos(index, 0);
      if (isSlice)
        dimsIndices.emplace_back(0);

      for (auto i : llvm::seq(1u, rank)) {
        std::tie(offsets[i], sizes[i], strides[i]) = makeFullSlice(i);
        dimsIndices.emplace_back(i);
      }
    }

    mlir::Value res;
    auto elemType = shapedType.getElementType();
    auto elemTypeSignless = plier::makeSignlessType(elemType);
    if (elemType != elemTypeSignless) {
      if (isMemref) {
        auto memrefType = type.cast<mlir::MemRefType>();
        auto signlessType = mlir::MemRefType::get(
            memrefType.getShape(), elemTypeSignless, memrefType.getLayout());
        value = rewriter.create<plier::SignCastOp>(loc, signlessType, value);
      } else if (isTensor) {
        auto tensorType = type.cast<mlir::RankedTensorType>();
        auto signlessType = mlir::RankedTensorType::get(
            tensorType.getShape(), elemTypeSignless, tensorType.getEncoding());
        value = rewriter.create<plier::SignCastOp>(loc, signlessType, value);
      } else {
        llvm_unreachable("Invalid getitem");
      }
    }

    if (!dimsIndices.empty()) {
      auto numDims = static_cast<unsigned>(dimsIndices.size());
      auto needReshape = (numDims != type.cast<mlir::ShapedType>().getRank());
      if (isMemref) {
        if (needReshape)
          return mlir::failure(); // TODO: not implemented

        res = rewriter.create<mlir::memref::SubViewOp>(loc, value, offsets,
                                                       sizes, strides);
      } else if (isTensor) {
        res = rewriter.create<mlir::tensor::ExtractSliceOp>(loc, value, offsets,
                                                            sizes, strides);
        if (needReshape) {
          auto resultType = mlir::RankedTensorType::get(
              llvm::SmallVector<int64_t>(numDims, -1), elemType);
          auto resultTypeSignless = mlir::RankedTensorType::get(
              llvm::SmallVector<int64_t>(numDims, -1), elemTypeSignless);
          llvm::SmallVector<mlir::Value> elements(numDims);
          for (auto it : llvm::enumerate(dimsIndices)) {
            auto dim =
                rewriter.create<mlir::tensor::DimOp>(loc, value, it.value());
            elements[it.index()] = dim;
          }
          auto shape =
              rewriter.create<mlir::tensor::FromElementsOp>(loc, elements);
          res = rewriter.create<mlir::tensor::ReshapeOp>(
              loc, resultTypeSignless, res, shape);
          if (resultType != resultTypeSignless) {
            res = rewriter.create<plier::SignCastOp>(loc, resultType, res);
          }
        }
      } else {
        llvm_unreachable("Invalid getitem");
      }
    } else {
      auto toValues = [&](auto vals) {
        llvm::SmallVector<mlir::Value> ret(vals.size());
        for (auto it : llvm::enumerate(vals)) {
          auto i = it.index();
          auto val = it.value();
          if (auto v = val.template dyn_cast<mlir::Value>()) {
            ret[i] = v;
          } else {
            auto attr = val.template get<mlir::Attribute>();
            auto attrVal = attr.template cast<mlir::IntegerAttr>()
                               .getValue()
                               .getSExtValue();
            ret[i] =
                rewriter.create<mlir::arith::ConstantIndexOp>(loc, attrVal);
          }
        }

        return ret;
      };
      if (isMemref) {
        res = rewriter.create<mlir::memref::LoadOp>(loc, value,
                                                    toValues(offsets));
      } else if (isTensor) {
        res = rewriter.create<mlir::tensor::ExtractOp>(loc, value,
                                                       toValues(offsets));
      } else {
        llvm_unreachable("Invalid getitem");
      }

      if (elemType != elemTypeSignless)
        res = rewriter.create<plier::SignCastOp>(loc, elemType, res);
    }

    rerun_scf_pipeline(op);
    rewriter.replaceOp(op, res);
    return mlir::success();
  }
};

struct SetitemOpLowering : public mlir::OpConversionPattern<plier::SetItemOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::SetItemOp op, plier::SetItemOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto target = adaptor.target();
    auto targetType = target.getType().dyn_cast<mlir::ShapedType>();
    if (!targetType)
      return mlir::failure();

    auto index = adaptor.index();
    if (!isValidGetitemIndex(index.getType()))
      return mlir::failure();

    auto elemType = targetType.getElementType();
    auto signlessElemType = plier::makeSignlessType(elemType);
    if (auto targetTensorType =
            targetType.template dyn_cast<mlir::RankedTensorType>()) {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      if (auto parentOp = target.getDefiningOp()) {
        rewriter.setInsertionPointAfter(parentOp);
      } else {
        rewriter.setInsertionPointToStart(target.getParentBlock());
      }

      auto loc = target.getLoc();
      if (elemType != signlessElemType) {
        auto tensorType = mlir::RankedTensorType::get(
            targetTensorType.getShape(), signlessElemType,
            targetTensorType.getEncoding());
        target = rewriter.create<plier::SignCastOp>(loc, tensorType, target);
      }
      auto memrefType =
          mlir::MemRefType::get(targetTensorType.getShape(), signlessElemType);
      auto memref =
          rewriter.create<mlir::memref::BufferCastOp>(loc, memrefType, target);
      target = memref;
      for (auto &use : llvm::make_early_inc_range(target.getUses())) {
        auto useOp = use.getOwner();
        assert(nullptr != useOp);
        if (useOp != memref) {
          if (mlir::isa<plier::SetItemOp>(useOp)) {
            rewriter.updateRootInPlace(useOp, [&]() {
              useOp->setOperand(use.getOperandNumber(), memref);
            });
          } else {
            mlir::OpBuilder::InsertionGuard g(rewriter);
            rewriter.setInsertionPoint(useOp);
            auto newVal = rewriter.create<mlir::memref::TensorLoadOp>(
                useOp->getLoc(), memref);
            rewriter.updateRootInPlace(useOp, [&]() {
              useOp->setOperand(use.getOperandNumber(), newVal);
            });
          }
        }
      }
    } else if (targetType.isa<mlir::MemRefType>()) {
      // nothing
    } else {
      return mlir::failure();
    }

    auto value = adaptor.value();
    auto loc = op.getLoc();
    if (value.getType() != elemType) {
      // TODO
      value = rewriter.createOrFold<plier::CastOp>(loc, elemType, value);
      rerun_scf_pipeline(op);
    }

    llvm::SmallVector<mlir::Value> indices;
    if (auto tupleType = index.getType().template dyn_cast<mlir::TupleType>()) {
      indices.resize(tupleType.size());
      for (auto it : llvm::enumerate(tupleType)) {
        auto i = it.index();
        auto getitemInd = rewriter.create<mlir::arith::ConstantIndexOp>(loc, i);
        auto ind = rewriter
                       .create<plier::GetItemOp>(loc, tupleType.getType(i),
                                                 index, getitemInd)
                       .getResult();
        auto indType = tupleType.getType(i);
        auto signlessIndType = plier::makeSignlessType(indType);
        if (signlessIndType != indType) {
          ind = rewriter.create<plier::SignCastOp>(loc, signlessIndType, ind);
        }
        indices[it.index()] = index_cast(ind, loc, rewriter);
      }
      rerun_scf_pipeline(op);
    } else {
      indices.push_back(index_cast(index, loc, rewriter));
    }

    if (elemType != signlessElemType)
      value = rewriter.create<plier::SignCastOp>(loc, signlessElemType, value);

    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, value, target,
                                                       indices);
    return mlir::success();
  }
};

mlir::Value unstride(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value src, mlir::MemRefType newType) {
  auto srcType = src.getType().cast<mlir::MemRefType>();
  if (newType == srcType)
    return src;

  if (srcType.getLayout().isIdentity())
    return builder.createOrFold<mlir::memref::CastOp>(loc, src, newType);

  auto rank = static_cast<unsigned>(srcType.getRank());
  llvm::SmallVector<mlir::Value> sizes(rank);
  for (unsigned i = 0; i < rank; ++i)
    sizes[i] = builder.createOrFold<mlir::memref::DimOp>(loc, src, i);

  auto allocType = mlir::MemRefType::get(
      llvm::SmallVector<int64_t>(rank, mlir::ShapedType::kDynamicSize),
      srcType.getElementType());
  auto result =
      builder.create<mlir::memref::AllocOp>(loc, allocType, sizes).getResult();
  if (result.getType() != newType)
    result = builder.createOrFold<mlir::memref::CastOp>(loc, result, newType);

  builder.create<mlir::linalg::CopyOp>(loc, src, result);
  return result;
}

struct FixStridedClone : public mlir::OpRewritePattern<mlir::memref::CloneOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::CloneOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.input().getType() != op.getType()) {
      rewriter.replaceOpWithNewOp<mlir::memref::CloneOp>(op, op.input());
      return mlir::success();
    }
    return mlir::failure();
  }
};

llvm::Optional<unsigned> getSingleDynamicDim(mlir::ShapedType type) {
  if (!type.hasRank()) {
    return llvm::None;
  }

  int dimIndex = -1;
  for (auto it : llvm::enumerate(type.getShape())) {
    auto i = static_cast<int>(it.index());
    auto dim = it.value();
    if (dim == mlir::ShapedType::kDynamicSize) {
      if (dimIndex != -1) {
        return llvm::None;
      }
      dimIndex = i;
    } else if (dim != 1) {
      return llvm::None;
    }
  }
  if (dimIndex != -1) {
    return static_cast<unsigned>(dimIndex);
  }
  return llvm::None;
}

struct FixStridedReshape
    : public mlir::OpRewritePattern<mlir::memref::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::ReshapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto source = op.source();
    auto shape = op.shape();
    auto srcType = source.getType().cast<mlir::MemRefType>();
    auto dstType = op.getType().cast<mlir::MemRefType>();
    if (dstType.getRank() == 1) {
      if (auto srcDimIndex = getSingleDynamicDim(srcType)) {
        auto srcRank = static_cast<unsigned>(srcType.getRank());
        assert(*srcDimIndex < srcRank);
        auto loc = op.getLoc();
        auto zero =
            rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0).getResult();
        llvm::SmallVector<mlir::OpFoldResult> offsets(srcRank,
                                                      rewriter.getIndexAttr(0));
        llvm::SmallVector<mlir::OpFoldResult> sizes(srcRank,
                                                    rewriter.getIndexAttr(1));
        sizes[*srcDimIndex] =
            rewriter.createOrFold<mlir::memref::LoadOp>(loc, shape, zero);
        llvm::SmallVector<mlir::OpFoldResult> strides(srcRank,
                                                      rewriter.getIndexAttr(1));
        auto view = rewriter.createOrFold<mlir::memref::SubViewOp>(
            loc, source, offsets, sizes, strides);
        if (view.getType().cast<mlir::MemRefType>().getRank() >
            dstType.getRank()) {
          std::array<int32_t, 1> mapping;
          mapping[0] = static_cast<int32_t>(*srcDimIndex);
          rewriter.replaceOpWithNewOp<plier::ReduceRankOp>(op, view, mapping);
        } else {
          rewriter.replaceOp(op, view);
        }
        return mlir::success();
      }
    }

    auto newType =
        mlir::MemRefType::get(srcType.getShape(), srcType.getElementType());
    auto newSource = unstride(rewriter, op.getLoc(), source, newType);
    if (newSource == source) {
      return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<mlir::memref::ReshapeOp>(op, op.getType(),
                                                         newSource, shape);
    return mlir::success();
  }
};

struct FixStridedIf : public mlir::OpRewritePattern<mlir::scf::YieldOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::YieldOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getNumOperands() == 0)
      return mlir::failure();

    auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op->getParentOp());
    if (!ifOp)
      return mlir::failure();

    llvm::SmallVector<mlir::Type> resultTypes(ifOp.getNumResults());
    auto trueYield = ifOp.thenYield();
    auto falseYield = ifOp.elseYield();

    bool changed = false;
    for (auto it : llvm::enumerate(llvm::zip(trueYield.getOperandTypes(),
                                             falseYield.getOperandTypes(),
                                             ifOp.getResultTypes()))) {
      auto index = static_cast<unsigned>(it.index());
      auto trueType = std::get<0>(it.value());
      auto falseType = std::get<1>(it.value());
      auto origType = std::get<2>(it.value());

      if (origType.isa<mlir::MemRefType>() || origType != trueType ||
          origType != falseType) {
        changed = true;
        auto trueMemref = trueType.cast<mlir::MemRefType>();
        auto falseMemref = falseType.cast<mlir::MemRefType>();
        bool isTrueIdentity = trueMemref.getLayout().isIdentity();
        auto resultType = (isTrueIdentity ? falseMemref : trueMemref);

        for (auto yield : {trueYield, falseYield}) {
          auto arg = yield.getOperand(index);
          if (resultType != arg.getType()) {
            rewriter.setInsertionPoint(yield);
            auto newVal =
                unstride(rewriter, trueYield.getLoc(), arg, resultType);
            if (newVal != arg) {
              yield.setOperand(index, newVal);
            }
          }
        }

        resultTypes[index] = resultType;
      } else {
        resultTypes[index] = origType;
      }
    }

    if (changed)
      rewriter.updateRootInPlace(ifOp, [&]() {
        for (auto it : llvm::enumerate(ifOp->getResults()))
          it.value().setType(resultTypes[it.index()]);
      });

    return mlir::success(changed);
  }
};

struct FixStridedSubview
    : public mlir::OpRewritePattern<mlir::memref::SubViewOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::SubViewOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto source = op.source();
    auto srcType = source.getType().cast<mlir::MemRefType>();
    auto dstType = op.getType().cast<mlir::MemRefType>();
    auto offsets = op.getMixedOffsets();
    auto sizes = op.getMixedSizes();
    auto strides = op.getMixedStrides();
    auto inferredType =
        [&]() {
          auto dstRank = static_cast<unsigned>(dstType.getRank());
          if (srcType.getRank() != dstRank) {
            return mlir::memref::SubViewOp::inferRankReducedResultType(
                dstRank, srcType, offsets, sizes, strides);
          } else {
            return mlir::memref::SubViewOp::inferResultType(srcType, offsets,
                                                            sizes, strides);
          }
        }()
            .cast<mlir::MemRefType>();

    if (inferredType == dstType) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<mlir::memref::SubViewOp>(
        op, inferredType, source, offsets, sizes, strides);
    return mlir::success();
  }
};

struct FixStridedReturn : public mlir::OpRewritePattern<mlir::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::ReturnOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto func = op->getParentOfType<mlir::FuncOp>();
    auto argsCount = op.getNumOperands();
    if (!func || func.getNumResults() != argsCount) {
      return mlir::failure();
    }

    auto loc = op.getLoc();
    auto retTypes = func.getType().getResults();
    bool changed = false;
    llvm::SmallVector<mlir::Value> newArgs(argsCount);
    for (auto it : llvm::enumerate(op.getOperands())) {
      auto arg = it.value();
      auto i = it.index();
      auto retType = retTypes[i];
      if (arg.getType() != retType) {
        auto srcMemrefType = arg.getType().dyn_cast<mlir::MemRefType>();
        auto dstMemrefType = retType.dyn_cast<mlir::MemRefType>();
        if (srcMemrefType && dstMemrefType) {
          auto newMemrefType = dstMemrefType;
          if (!dstMemrefType.getLayout().isIdentity())
            newMemrefType = mlir::MemRefType::get(
                dstMemrefType.getShape(), dstMemrefType.getElementType());

          if (newMemrefType != dstMemrefType) {
            arg =
                rewriter.create<mlir::memref::CastOp>(loc, arg, dstMemrefType);
            changed = true;
          }
        }
      }
      newArgs[i] = arg;
    }

    if (changed) {
      rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op, newArgs);
    }
    return mlir::success(changed);
  }
};

struct FixStridedCall : public mlir::OpRewritePattern<mlir::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    auto func = mod.lookupSymbol<mlir::FuncOp>(op.callee());
    if (!func)
      return mlir::failure();

    auto funcType = func.getType();
    if (funcType.getNumInputs() != op.operands().size())
      return mlir::failure();

    auto loc = op.getLoc();
    bool changed = false;
    llvm::SmallVector<mlir::Value> newArgs(funcType.getNumInputs());
    for (auto it : llvm::enumerate(op.operands())) {
      auto i = static_cast<unsigned>(it.index());
      auto arg = it.value();
      newArgs[i] = arg;
      if (auto srcMemref = arg.getType().dyn_cast<mlir::MemRefType>()) {
        if (auto dstMemref =
                funcType.getInput(i).dyn_cast<mlir::MemRefType>()) {
          if (srcMemref.getShape() == dstMemref.getShape() &&
              srcMemref.getElementType() == dstMemref.getElementType() &&
              srcMemref.getLayout() != dstMemref.getLayout()) {
            changed = true;
            newArgs[i] =
                rewriter.create<mlir::memref::CastOp>(loc, arg, dstMemref);
          }
        }
      }
    }

    if (changed) {
      rewriter.replaceOpWithNewOp<mlir::CallOp>(op, op.calleeAttr(),
                                                funcType.getResults(), newArgs);
    }
    return mlir::success(changed);
  }
};

struct CleanupLoads : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto block = op->getBlock();
    auto it = mlir::Block::iterator(op);
    if (it == block->begin())
      return mlir::failure();

    --it;
    auto store = mlir::dyn_cast<mlir::memref::StoreOp>(*it);
    if (!store)
      return mlir::failure();

    if (store.memref() != op.memref() || store.indices() != op.indices())
      return mlir::failure();

    rewriter.replaceOp(op, store.value());
    return mlir::success();
  }
};

struct MakeStridedLayoutPass
    : public mlir::PassWrapper<MakeStridedLayoutPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override;
};

void MakeStridedLayoutPass::runOnOperation() {
  auto context = &getContext();
  auto mod = getOperation();

  bool changed = false;
  llvm::SmallVector<mlir::Type> newArgTypes;
  llvm::SmallVector<mlir::Type> newResTypes;
  llvm::SmallVector<mlir::Value> newOperands;
  for (auto func : mod.getOps<mlir::FuncOp>()) {
    if (!func.isPublic() && !func.getBody().empty())
      continue;

    mlir::OpBuilder builder(func.body());
    auto loc = builder.getUnknownLoc();
    auto funcType = func.getType();
    auto argTypes = funcType.getInputs();
    auto resTypes = funcType.getResults();
    newArgTypes.assign(argTypes.begin(), argTypes.end());
    newResTypes.assign(resTypes.begin(), resTypes.end());
    bool hasBody = !func.getBody().empty();
    for (auto it : llvm::enumerate(argTypes)) {
      auto i = static_cast<unsigned>(it.index());
      auto type = it.value();
      if (auto memrefType = type.dyn_cast<mlir::MemRefType>()) {
        auto rank = static_cast<unsigned>(memrefType.getRank());
        auto makeShape = [&](int64_t val) {
          return llvm::SmallVector<int64_t>(rank, val);
        };
        auto strideVal = mlir::ShapedType::kDynamicStrideOrOffset;
        auto affineMap = mlir::makeStridedLinearLayoutMap(makeShape(strideVal),
                                                          strideVal, context);
        auto newMemrefType =
            mlir::MemRefType::get(makeShape(mlir::ShapedType::kDynamicSize),
                                  memrefType.getElementType(), affineMap);

        if (newMemrefType != memrefType) {
          newArgTypes[i] = newMemrefType;

          if (hasBody) {
            auto arg = func.getBody().front().getArgument(i);
            arg.setType(newMemrefType);
            auto dst =
                builder.create<plier::ChangeLayoutOp>(loc, memrefType, arg);
            arg.replaceAllUsesExcept(dst, dst);
          }
        }
      }
    }

    for (auto it : llvm::enumerate(resTypes)) {
      auto type = it.value();
      if (auto memrefType = type.dyn_cast<mlir::MemRefType>()) {
        auto rank = static_cast<unsigned>(memrefType.getRank());
        auto makeShape = [&](int64_t val) {
          return llvm::SmallVector<int64_t>(rank, val);
        };
        auto strideVal = mlir::ShapedType::kDynamicStrideOrOffset;
        auto affineMap = mlir::makeStridedLinearLayoutMap(
            makeShape(strideVal), strideVal, builder.getContext());
        auto newmemrefType =
            mlir::MemRefType::get(makeShape(mlir::ShapedType::kDynamicSize),
                                  memrefType.getElementType(), affineMap);
        newResTypes[it.index()] = newmemrefType;
      }
    }

    auto newFuncType =
        mlir::FunctionType::get(&getContext(), newArgTypes, newResTypes);
    if (newFuncType != funcType) {
      changed = true;
      func.setType(newFuncType);
      func.walk([&](mlir::ReturnOp ret) {
        builder.setInsertionPoint(ret);
        auto count = static_cast<unsigned>(newResTypes.size());
        for (auto i : llvm::seq(0u, count)) {
          auto arg = ret.getOperand(i);
          auto newType = newResTypes[i];
          if (arg.getType() != newType) {
            assert(arg.getType().isa<mlir::MemRefType>());
            assert(newType.isa<mlir::MemRefType>());
            auto newArg = builder.createOrFold<mlir::memref::CastOp>(
                ret.getLoc(), arg, newType);
            ret.setOperand(i, newArg);
          }
        }
      });
      auto funcUses = mlir::SymbolTable::getSymbolUses(func, mod);
      if (funcUses) {
        for (auto use : *funcUses) {
          auto call = mlir::cast<mlir::CallOp>(use.getUser());
          auto loc = call.getLoc();

          builder.setInsertionPoint(call);
          assert(newArgTypes.size() == call.operands().size());
          auto argsCount = static_cast<unsigned>(newArgTypes.size());
          newOperands.resize(argsCount);
          for (auto i : llvm::seq(0u, argsCount)) {
            auto arg = call.operands()[i];
            auto oldType = arg.getType();
            auto newType = newArgTypes[i];
            if (oldType != newType) {
              assert(oldType.isa<mlir::MemRefType>());
              assert(newType.isa<mlir::MemRefType>());
              auto newArg =
                  builder.create<plier::ChangeLayoutOp>(loc, newType, arg)
                      .getResult();
              newOperands[i] = newArg;
            } else {
              newOperands[i] = arg;
            }
          }
          call.operandsMutable().assign(newOperands);

          builder.setInsertionPointAfter(call);
          assert(resTypes.size() == call.getNumResults());
          auto numResults = call.getNumResults();
          for (auto i : llvm::seq(0u, numResults)) {
            auto res = call.getResult(i);
            auto oldType = res.getType();
            auto newType = resTypes[i];
            if (oldType != newType) {
              assert(oldType.isa<mlir::MemRefType>());
              assert(newType.isa<mlir::MemRefType>());
              auto newRes =
                  builder.create<plier::ChangeLayoutOp>(loc, oldType, res);
              res.replaceAllUsesExcept(newRes, newRes);
            }
          }
        }
      }
    }
  }

  //  if (changed) {
  //    mlir::OwningRewritePatternList patterns(context);

  ////    plier::ChangeLayoutOp::getCanonicalizationPatterns(patterns, context);

  //    //    plier::populateCommonOptsPatterns(*context, patterns);
  //    //    patterns.insert<
  //    //        // clang-format off
  //    //        FixStridedIf,
  //    //        FixStridedClone,
  //    //        FixStridedReshape,
  //    //        FixStridedSubview,
  //    //        FixStridedReturn,
  //    //        FixStridedCall,
  //    //        CleanupLoads
  //    //        // clang-format on
  //    //        >(context);

  //    (void)mlir::applyPatternsAndFoldGreedily(mod, std::move(patterns));

  //    mod.walk([&](plier::ChangeLayoutOp op) {
  //      op.emitError("Layout change failed");
  //      signalPassFailure();
  //    });
  //  }
}

struct CheckStridedLayoutPass
    : public mlir::PassWrapper<CheckStridedLayoutPass, mlir::OperationPass<>> {
  void runOnOperation() override;
};

void CheckStridedLayoutPass::runOnOperation() {
  auto op = getOperation();
  op->walk([&](plier::ChangeLayoutOp cl) {
    cl.emitError("Layout change failed");
    signalPassFailure();
  });
}

struct PlierToLinalgPass
    : public mlir::PassWrapper<PlierToLinalgPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<plier::PlierDialect>();
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
  }

  void runOnOperation() override;
};

template <typename T> bool hasCompatibleShape(T &&a1, T &&a2) {
  if (a1.getRank() != a2.getRank()) {
    return false;
  }
  for (auto it : llvm::zip(a1.getShape(), a2.getShape())) {
    auto s1 = std::get<0>(it);
    auto s2 = std::get<1>(it);
    if (s1 != mlir::ShapedType::kDynamicSize &&
        s2 != mlir::ShapedType::kDynamicSize && s1 != s2) {
      return false;
    }
  }
  return true;
}

struct LowerTensorCasts : public mlir::OpConversionPattern<plier::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::CastOp op, plier::CastOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = *getTypeConverter();

    auto value = adaptor.value();
    auto srcType = value.getType();
    auto dstType = converter.convertType(op.getType());
    if (!dstType)
      return mlir::failure();

    if (srcType == dstType) {
      rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(op, dstType,
                                                                    value);
      return mlir::success();
    }

    if (srcType.isa<mlir::RankedTensorType>() &&
        dstType.isa<mlir::RankedTensorType>()) {
      auto src = srcType.cast<mlir::RankedTensorType>();
      auto dst = dstType.cast<mlir::RankedTensorType>();
      auto srcElem = src.getElementType();
      auto dstElem = dst.getElementType();
      if (!hasCompatibleShape(src, dst))
        return mlir::failure();

      auto signlessSrcType = mlir::RankedTensorType::get(
          src.getShape(), plier::makeSignlessType(srcElem), src.getEncoding());
      auto signlessDstType = mlir::RankedTensorType::get(
          dst.getShape(), plier::makeSignlessType(dstElem), dst.getEncoding());
      auto loc = op.getLoc();
      if (signlessSrcType != src)
        value = rewriter.createOrFold<plier::SignCastOp>(loc, signlessSrcType,
                                                         value);

      value = rewriter.createOrFold<mlir::tensor::CastOp>(loc, signlessDstType,
                                                          value);
      if (signlessDstType != dst)
        value = rewriter.createOrFold<plier::SignCastOp>(loc, dst, value);

      rewriter.replaceOp(op, value);
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct SimplifyExpandDims
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  using mlir::OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::GenericOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op.hasTensorSemantics())
      return mlir::failure();

    if (op.getNumInputs() != 1 || op.getNumOutputs() != 1)
      return mlir::failure();

    auto context = op.getContext();
    auto parallelAttr = mlir::StringAttr::get(context, "parallel");
    if (llvm::any_of(op.iterator_types(),
                     [&](auto attr) { return attr != parallelAttr; }))
      return mlir::failure();

    auto maps = op.indexing_maps();
    assert(maps.size() == 2);
    auto outMap = maps[1].cast<mlir::AffineMapAttr>().getValue();
    if (!outMap.isIdentity())
      return mlir::failure();

    auto inMap = maps[0].cast<mlir::AffineMapAttr>().getValue();
    auto numDims = op.getNumLoops();
    if (inMap.getNumResults() != numDims)
      return mlir::failure();

    bool changed = false;
    auto outShape = op.getOutputOperand(0)
                        ->get()
                        .getType()
                        .cast<mlir::RankedTensorType>()
                        .getShape();
    llvm::SmallVector<mlir::AffineExpr> exprs(numDims);
    for (unsigned i = 0; i < numDims; ++i) {
      auto prevExpr = inMap.getResult(i);
      bool canConvert = [&]() {
        if (outShape[i] == 1) {
          auto constExpr = prevExpr.dyn_cast<mlir::AffineConstantExpr>();
          if (constExpr && constExpr.getValue() == 0)
            return true;
        }
        return false;
      }();
      if (canConvert) {
        changed = true;
        exprs[i] = mlir::getAffineDimExpr(i, context);
      } else {
        exprs[i] = prevExpr;
      }
    }

    if (changed) {
      const mlir::Attribute newMaps[] = {
          mlir::AffineMapAttr::get(
              mlir::AffineMap::get(numDims, 0, exprs, context)),
          maps[1]};
      auto newMapsAttr = mlir::ArrayAttr::get(context, newMaps);
      rewriter.updateRootInPlace(op,
                                 [&]() { op.indexing_mapsAttr(newMapsAttr); });
    }

    return mlir::success(changed);
  }
};

struct LowerEnforceShape
    : public mlir::OpRewritePattern<plier::EnforceShapeOp> {
  using mlir::OpRewritePattern<plier::EnforceShapeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::EnforceShapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto type = op.getType();
    auto src = op.value();
    rewriter.replaceOpWithNewOp<mlir::tensor::CastOp>(op, type, src);
    return mlir::success();
  }
};

static bool isTensor(mlir::TypeConverter &converter, mlir::Type type) {
  return !!converter.convertType(type).dyn_cast_or_null<mlir::TensorType>();
}

void PlierToLinalgPass::runOnOperation() {
  auto &context = getContext();

  mlir::TypeConverter typeConverter;
  // Convert unknown types to itself
  typeConverter.addConversion([](mlir::Type type) { return type; });
  populateStdTypeConverter(context, typeConverter);
  populateTupleTypeConverter(context, typeConverter);
  populateArrayTypeConverter(context, typeConverter);

  auto materializeCast = [](mlir::OpBuilder &builder, mlir::Type type,
                            mlir::ValueRange inputs,
                            mlir::Location loc) -> llvm::Optional<mlir::Value> {
    if (inputs.size() == 1)
      return builder
          .create<mlir::UnrealizedConversionCastOp>(loc, type, inputs.front())
          .getResult(0);

    return llvm::None;
  };
  typeConverter.addArgumentMaterialization(materializeCast);
  typeConverter.addSourceMaterialization(materializeCast);
  typeConverter.addTargetMaterialization(materializeCast);

  mlir::RewritePatternSet patterns(&context);
  mlir::ConversionTarget target(context);

  target.addDynamicallyLegalOp<plier::GetItemOp>(
      [&typeConverter](plier::GetItemOp op) -> llvm::Optional<bool> {
        if (isTensor(typeConverter, op.value().getType()))
          return false;

        return llvm::None;
      });

  target.addDynamicallyLegalOp<plier::SetItemOp>(
      [&typeConverter](plier::SetItemOp op) -> bool {
        return !isTensor(typeConverter, op.target().getType());
      });

  target.addDynamicallyLegalOp<plier::CastOp>([&](plier::CastOp op) -> bool {
    auto inputType = op.value().getType();
    auto srcType = typeConverter.convertType(inputType);
    auto dstType = typeConverter.convertType(op.getType());
    if (!srcType || !dstType)
      return true;

    auto isZeroRankTensor = [](mlir::Type t) -> bool {
      auto tensor = t.dyn_cast<mlir::RankedTensorType>();
      return tensor && tensor.getRank() == 0;
    };

    if ((isZeroRankTensor(srcType) && dstType.isIntOrIndexOrFloat()) ||
        (isZeroRankTensor(dstType) && srcType.isIntOrIndexOrFloat()))
      return false;

    if (srcType == dstType && inputType != op.getType())
      return false;

    return srcType == dstType || !(srcType.isa<mlir::ShapedType>() &&
                                   dstType.isa<mlir::ShapedType>());
  });

  plier::populateControlFlowTypeConversionRewritesAndTarget(typeConverter,
                                                            patterns, target);
  plier::populateTupleTypeConversionRewritesAndTarget(typeConverter, patterns,
                                                      target);

  patterns.insert<
      // clang-format off
      GetitemOpLowering,
      SetitemOpLowering,
      LowerTensorCasts,
      UnrankedToElementCasts
      // clang-format on
      >(typeConverter, &context);

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns))))
    signalPassFailure();
}

struct LowerLinalgPass
    : public mlir::PassWrapper<LowerLinalgPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::AffineDialect>();
  }

  void runOnOperation() override;
};

void LowerLinalgPass::runOnOperation() {
  mlir::OwningRewritePatternList patterns(&getContext());

  patterns.insert<mlir::linalg::LinalgLoweringPattern<mlir::linalg::GenericOp>,
                  mlir::linalg::LinalgLoweringPattern<mlir::linalg::CopyOp>>(
      &getContext(), mlir::linalg::LinalgLoweringType::ParallelLoops);

  (void)mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

struct OptimizeGlobalsConstsLoad
    : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using mlir::OpRewritePattern<mlir::memref::LoadOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // We access data outside function, but doesnt change it, lets hope it
    // is safe.
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod) {
      return mlir::failure();
    }
    mlir::SymbolTable symbolTable(mod);

    llvm::SmallVector<uint64_t> indices(op.indices().size());
    for (auto it : llvm::enumerate(op.indices())) {
      auto constIndex =
          it.value().getDefiningOp<mlir::arith::ConstantIndexOp>();
      if (!constIndex)
        return mlir::failure();

      auto val = constIndex.value();
      if (val < 0)
        return mlir::failure();

      indices[it.index()] = static_cast<uint64_t>(val);
    }
    auto getGlobal = op.memref().getDefiningOp<mlir::memref::GetGlobalOp>();
    if (!getGlobal)
      return mlir::failure();

    auto sym = symbolTable.lookup<mlir::memref::GlobalOp>(getGlobal.name());
    if (!sym)
      return mlir::failure();

    if (!sym.constant())
      return mlir::failure();

    auto initAttr = sym.initial_value();
    if (!initAttr)
      return mlir::failure();

    auto elements = initAttr->dyn_cast<mlir::ElementsAttr>();
    if (!elements)
      return mlir::failure();

    if (elements.getType().getElementType() != op.getType() ||
        !elements.isValidIndex(indices))
      return mlir::failure();

    auto vals = elements.tryGetValues<mlir::Attribute>();
    if (!vals)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, (*vals)[indices]);
    return mlir::success();
  }
};

struct ForceInlinePass
    : public plier::RewriteWrapperPass<ForceInlinePass, void, void,
                                       plier::ForceInline> {};

struct PostPlierToLinalgPass
    : public mlir::PassWrapper<PostPlierToLinalgPass, mlir::FunctionPass> {
  void runOnFunction() override;
};

void PostPlierToLinalgPass::runOnFunction() {
  auto &context = getContext();
  mlir::OwningRewritePatternList patterns(&context);

  plier::populateCommonOptsPatterns(context, patterns);

  patterns.insert<SimplifyExpandDims>(&context);

  (void)mlir::applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
}

struct MakeTensorsSignlessPass
    : public mlir::PassWrapper<MakeTensorsSignlessPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override;
};

void MakeTensorsSignlessPass::runOnOperation() {
  auto module = getOperation();
  auto *context = &getContext();

  mlir::TypeConverter typeConverter;
  typeConverter.addConversion([](mlir::Type type) { return type; });
  typeConverter.addConversion(
      [](mlir::RankedTensorType type) -> llvm::Optional<mlir::Type> {
        auto elemType = type.getElementType().dyn_cast<mlir::IntegerType>();
        if (elemType && !elemType.isSignless()) {
          auto signless =
              mlir::IntegerType::get(type.getContext(), elemType.getWidth());
          return mlir::RankedTensorType::get(type.getShape(), signless,
                                             type.getEncoding());
        }
        return llvm::None;
      });
  populateTupleTypeConverter(*context, typeConverter);

  auto materializeSignCast = [](mlir::OpBuilder &builder, mlir::Type type,
                                mlir::ValueRange inputs,
                                mlir::Location loc) -> mlir::Value {
    assert(inputs.size() == 1);
    return builder.create<plier::SignCastOp>(loc, type, inputs[0]);
  };
  typeConverter.addArgumentMaterialization(materializeSignCast);
  typeConverter.addSourceMaterialization(materializeSignCast);
  typeConverter.addTargetMaterialization(materializeSignCast);

  mlir::RewritePatternSet patterns(context);
  mlir::ConversionTarget target(*context);

  plier::populateControlFlowTypeConversionRewritesAndTarget(typeConverter,
                                                            patterns, target);
  plier::populateTupleTypeConversionRewritesAndTarget(typeConverter, patterns,
                                                      target);

  target.addLegalOp<mlir::ModuleOp, plier::SignCastOp>();

  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

struct TensorFusionPass
    : public mlir::PassWrapper<TensorFusionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override;
};

void TensorFusionPass::runOnOperation() {
  auto &context = getContext();
  mlir::OwningRewritePatternList patterns(&context);

  plier::populateCommonOptsPatterns(context, patterns);

  patterns.insert<SimplifyExpandDims, LowerEnforceShape>(&context);

  mlir::linalg::populateElementwiseOpsFusionPatterns(patterns);

  (void)mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

struct LoopInvariantCodeMotion
    : public mlir::OpRewritePattern<mlir::scf::ForOp> {
  using mlir::OpRewritePattern<mlir::scf::ForOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto parentOp = op->getParentOp();
    rewriter.startRootUpdate(parentOp);
    auto res = mlir::moveLoopInvariantCode(op);
    if (mlir::succeeded(res)) {
      rewriter.finalizeRootUpdate(parentOp);
    } else {
      rewriter.cancelRootUpdate(parentOp);
    }
    return res;
  }
};

struct BufferizeReshape
    : public mlir::OpConversionPattern<mlir::tensor::ReshapeOp> {
  using mlir::OpConversionPattern<mlir::tensor::ReshapeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::ReshapeOp op,
                  mlir::tensor::ReshapeOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto getType = [&](mlir::Type type) {
      auto shapedType = type.cast<mlir::ShapedType>();
      return mlir::MemRefType::get(shapedType.getShape(),
                                   shapedType.getElementType());
    };
    auto source = adaptor.source();
    auto shape = adaptor.shape();
    auto resType = getType(op.getType());
    rewriter.replaceOpWithNewOp<mlir::memref::ReshapeOp>(op, resType, source,
                                                         shape);
    return mlir::success();
  }
};

struct FixDeallocPlacement
    : public mlir::OpRewritePattern<mlir::memref::DeallocOp> {
  using mlir::OpRewritePattern<mlir::memref::DeallocOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::DeallocOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto block = op->getBlock();
    auto blockIt = mlir::Block::iterator(op);
    mlir::Operation *newPos = op;
    ++blockIt;
    auto memref = op.memref();
    mlir::BufferViewFlowAnalysis analysis(op->getParentOfType<mlir::FuncOp>());
    auto aliases = analysis.resolve(memref);
    auto blockEnd = block->without_terminator().end();
    for (auto &it : llvm::make_range(blockIt, blockEnd)) {
      auto visitor = [&](mlir::Operation *inner) {
        for (auto arg : inner->getOperands()) {
          if (aliases.count(arg)) {
            return mlir::WalkResult::interrupt();
          }
        }
        return mlir::WalkResult::advance();
      };
      if (it.walk(visitor).wasInterrupted()) {
        newPos = &it;
      }
    }

    if (newPos != op) {
      rewriter.setInsertionPointAfter(newPos);
      rewriter.create<mlir::memref::DeallocOp>(op.getLoc(), memref);
      rewriter.eraseOp(op);
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct AdditionalBufferize
    : public mlir::PassWrapper<AdditionalBufferize, mlir::FunctionPass> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<plier::PlierDialect>();
  }

  void runOnFunction() override;
};

void AdditionalBufferize::runOnFunction() {
  auto module = getOperation();
  auto *context = &getContext();

  mlir::BufferizeTypeConverter typeConverter;
  populateTupleTypeConverter(*context, typeConverter);

  auto materializeTupleCast =
      [](mlir::OpBuilder &builder, mlir::Type type, mlir::ValueRange inputs,
         mlir::Location loc) -> llvm::Optional<mlir::Value> {
    if (inputs.size() != 1)
      return llvm::None;

    auto input = inputs.front();
    if (input.getType().isa<mlir::TupleType>() || type.isa<mlir::TupleType>())
      return builder.createOrFold<plier::CastOp>(loc, type, input);

    return llvm::None;
  };
  typeConverter.addArgumentMaterialization(materializeTupleCast);
  typeConverter.addSourceMaterialization(materializeTupleCast);
  typeConverter.addTargetMaterialization(materializeTupleCast);

  mlir::RewritePatternSet patterns(context);
  mlir::ConversionTarget target(*context);

  plier::populateControlFlowTypeConversionRewritesAndTarget(typeConverter,
                                                            patterns, target);
  plier::populateTupleTypeConversionRewritesAndTarget(typeConverter, patterns,
                                                      target);
  target.addIllegalOp<mlir::tensor::ReshapeOp>();
  target.addLegalOp<mlir::memref::ReshapeOp>();

  patterns.insert<BufferizeReshape>(typeConverter, context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

struct CloneArgsPass
    : public mlir::PassWrapper<CloneArgsPass, mlir::FunctionPass> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<plier::PlierDialect>();
  }

  void runOnFunction() override;
};

void CloneArgsPass::runOnFunction() {
  auto func = getFunction();
  if (func.isPrivate() || func.isDeclaration() || func.body().empty()) {
    return;
  }

  mlir::OpBuilder builder(&getContext());
  //  auto loc = builder.getUnknownLoc();
  //  auto block = &func.body().front();
  //  builder.setInsertionPointToStart(block);
  //  for (auto arg : block->getArguments()) {
  //    if (auto type = arg.getType().dyn_cast<mlir::MemRefType>()) {
  //      auto retained = builder.create<mlir::memref::CloneOp>(loc, type, arg);
  //      arg.replaceAllUsesExcept(retained, retained);
  //    }
  //  }

  for (auto &block : func.getBody()) {
    auto ret = mlir::dyn_cast_or_null<mlir::ReturnOp>(block.getTerminator());
    if (!ret)
      continue;

    auto loc = ret.getLoc();
    bool needReplace = false;
    llvm::SmallVector<mlir::Value> newArgs(ret.operands().size());
    builder.setInsertionPoint(ret);
    for (auto it : llvm::enumerate(ret.operands())) {
      auto i = it.index();
      auto arg = it.value();
      if (arg.getType().isa<mlir::MemRefType>()) {
        newArgs[i] = builder.create<mlir::memref::CloneOp>(loc, arg);
        needReplace = true;
      } else {
        newArgs[i] = arg;
      }
    }

    if (needReplace) {
      builder.create<mlir::ReturnOp>(loc, newArgs);
      ret.erase();
    }
  }
}

struct ReplaceClones : public mlir::OpRewritePattern<mlir::memref::CloneOp> {
  using mlir::OpRewritePattern<mlir::memref::CloneOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::CloneOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<plier::RetainOp>(op, op.getSource());
    return mlir::success();
  }
};

struct LowerCloneOpsPass
    : public plier::RewriteWrapperPass<LowerCloneOpsPass, mlir::FuncOp, void,
                                       ReplaceClones> {};

struct UpliftMathCallsPass
    : public plier::RewriteWrapperPass<UpliftMathCallsPass, void, void,
                                       plier::UpliftMathCalls> {};

struct PostLinalgOptPass
    : public mlir::PassWrapper<PostLinalgOptPass, mlir::FunctionPass> {
  void runOnFunction() override;
};

void PostLinalgOptPass::runOnFunction() {
  auto func = getFunction();
  auto optLevel = getOptLevel(func);
  if (0 == optLevel)
    return;

  auto &context = getContext();
  mlir::OwningRewritePatternList patterns(&context);

  plier::populateCommonOptsPatterns(context, patterns);

  patterns.insert<OptimizeGlobalsConstsLoad, plier::CanonicalizeReduction,
                  plier::PromoteToParallel, plier::MergeNestedForIntoParallel>(
      &context);

  auto additionalOpt = [](mlir::FuncOp op) {
    (void)plier::prepareForFusion(op.getRegion());
    return plier::naivelyFuseParallelOps(op.getRegion());
  };
  if (mlir::failed(applyOptimizations(func, std::move(patterns),
                                      getAnalysisManager(), additionalOpt))) {
    signalPassFailure();
  }
}

struct FixDeallocPlacementPass
    : public plier::RewriteWrapperPass<FixDeallocPlacementPass, mlir::FuncOp,
                                       void, FixDeallocPlacement> {};

void populatePlierToLinalgGenPipeline(mlir::OpPassManager &pm) {
  pm.addPass(std::make_unique<PlierToLinalgPass>());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<NumpyCallsLoweringPass>());
  pm.addPass(std::make_unique<ForceInlinePass>());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addNestedPass<mlir::FuncOp>(std::make_unique<PostPlierToLinalgPass>());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCSEPass());
}

void populatePlierToLinalgOptPipeline(mlir::OpPassManager &pm) {
  pm.addPass(std::make_unique<MakeTensorsSignlessPass>());

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<TensorFusionPass>());

  pm.addPass(mlir::createTensorConstantBufferizePass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createSCFBufferizePass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createLinalgBufferizePass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createStdBufferizePass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createTensorBufferizePass());
  pm.addPass(mlir::createFuncBufferizePass());
  pm.addNestedPass<mlir::FuncOp>(std::make_unique<AdditionalBufferize>());
  pm.addNestedPass<mlir::FuncOp>(mlir::createFinalizingBufferizePass());

  pm.addNestedPass<mlir::FuncOp>(mlir::createBufferHoistingPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createBufferLoopHoistingPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createPromoteBuffersToStackPass());

  pm.addNestedPass<mlir::FuncOp>(std::make_unique<CloneArgsPass>());
  pm.addPass(std::make_unique<MakeStridedLayoutPass>());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::FuncOp>(std::make_unique<CheckStridedLayoutPass>());
  pm.addNestedPass<mlir::FuncOp>(mlir::createBufferDeallocationPass());
  pm.addPass(mlir::createCanonicalizerPass());

  pm.addNestedPass<mlir::FuncOp>(std::make_unique<LowerCloneOpsPass>());

  pm.addPass(std::make_unique<LowerLinalgPass>());
  pm.addPass(std::make_unique<ForceInlinePass>());
  pm.addPass(mlir::createSymbolDCEPass());

  pm.addNestedPass<mlir::FuncOp>(std::make_unique<UpliftMathCallsPass>());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createLoopInvariantCodeMotionPass());

  // ToDo: This pass also tries to do some simple fusion, whic should be split
  // in separate pass
  pm.addNestedPass<mlir::FuncOp>(std::make_unique<PostLinalgOptPass>());

  pm.addNestedPass<mlir::FuncOp>(std::make_unique<FixDeallocPlacementPass>());

  pm.addPass(mlir::createSymbolDCEPass());
}
} // namespace

void populateArrayTypeConverter(mlir::MLIRContext & /*context*/,
                                mlir::TypeConverter &converter) {
  converter.addConversion(
      [&](plier::PyType type) -> llvm::Optional<mlir::Type> {
        auto ret = map_plier_type(converter, type);
        if (!ret)
          return llvm::None;

        return ret;
      });
}

// ToDo: how does this sink stuff actually works?
void registerPlierToLinalgPipeline(plier::PipelineRegistry &registry) {
  registry.register_pipeline([](auto sink) {
    auto stage = getHighLoweringStage();
    sink(plierToLinalgGenPipelineName(), {plierToStdPipelineName()},
         {plierToLinalgOptPipelineName()}, {plierToScfPipelineName()},
         &populatePlierToLinalgGenPipeline);
    sink(plierToLinalgOptPipelineName(),
         {plierToLinalgGenPipelineName(), untuplePipelineName()},
         {removeSignPipelineName(), stage.end}, {},
         &populatePlierToLinalgOptPipeline);
  });
}

llvm::StringRef plierToLinalgGenPipelineName() { return "plier_to_linalg_gen"; }

llvm::StringRef plierToLinalgOptPipelineName() { return "plier_to_linalg_opt"; }
