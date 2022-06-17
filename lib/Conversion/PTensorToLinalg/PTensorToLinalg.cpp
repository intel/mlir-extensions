// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Converting PTensor to Linalg

#include <imex/Conversion/PTensorToLinalg/PTensorToLinalg.h>
#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

// return type without a sign
// copied from py_linalg_resolver.cpp
static mlir::Type makeSignlessType(mlir::Type type) {
  if (auto shaped = type.dyn_cast<mlir::ShapedType>()) {
    auto origElemType = shaped.getElementType();
    return makeSignlessType(origElemType);
  } else if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
    if (!intType.isSignless())
      return mlir::IntegerType::get(intType.getContext(), intType.getWidth());
  }
  return type;
}

// creating operand cast to signless type if needed
// copied from py_linalg_resolver.cpp
static mlir::Value doSignCast(mlir::OpBuilder &builder, mlir::Location &loc,
                              mlir::Value val) {
  auto origType = val.getType();
  auto signlessType = makeSignlessType(origType);
  if (signlessType != origType) {
    val =
        builder
            .create<::mlir::UnrealizedConversionCastOp>(loc, signlessType, val)
            .getResult(0);
  }
  return val;
}

#if 0
// creating operand cast to given type if needed
// copied from py_linalg_resolver.cpp
static mlir::Value doSignCast(mlir::OpBuilder &builder, mlir::Location &loc,
                              mlir::Value val, mlir::Type dstType) {
  auto origType = val.getType();
  if (dstType != origType) {
      val = builder.create<::mlir::UnrealizedConversionCastOp>(loc, dstType, val).getResult(0);
  }
  return val;
}
#endif

// Initialze a distributed Tensor:
// 1. register tensor with runtime
// 2. get local shape
// 3. init local tensor
// returns pair of tensor and id as assigned by runtime
// If not distributed, simply init tensor
static auto initDTensor(mlir::Location &loc,
                        ::mlir::ConversionPatternRewriter &rewriter, bool dist,
                        uint64_t rank, ::mlir::Value shp, ::mlir::Type eltyp,
                        ::llvm::SmallVector<mlir::Value> &lshp /* out */) {
  if (dist) {
    auto ityp = rewriter.getI64Type();
    auto idxtyp = rewriter.getIndexType();
    auto shptyp = mlir::RankedTensorType::get(
        llvm::SmallVector<int64_t>(1, rank), idxtyp);

    // Register with runtime
    ::mlir::Value id =
        rewriter.create<::dist::RegisterPTensorOp>(loc, ityp, shp);
    // and get local shape
    auto lshp_mr = rewriter.create<::dist::LocalShapeOp>(loc, shptyp, id);

    // get shape as SmallVector<mlir::Value>
    // why can't we just use the existing tensor?
    lshp.resize(rank);
    for (auto i : ::llvm::seq(0lu, rank)) {
      auto ia = rewriter.getIndexAttr(i);
      auto idx = rewriter.create<::mlir::arith::ConstantOp>(loc, ia);
      lshp[i] = rewriter.create<::mlir::tensor::ExtractOp>(
          loc, idxtyp, lshp_mr, ::mlir::ValueRange({idx}));
    }
    // create a 1d tensor of local shape
    auto ltnsr =
        rewriter.create<::mlir::linalg::InitTensorOp>(loc, lshp, eltyp);
    return std::make_pair(ltnsr.getResult(), id);
  } else { // not distributed, simply init
    auto ltnsr = rewriter.create<::mlir::linalg::InitTensorOp>(loc, shp, eltyp);
    return std::make_pair(ltnsr.getResult(), ::mlir::Value());
  }
}

// *******************************
// ***** Individual patterns *****
// *******************************

// convert PTensor's arange and its return type to Linalg/tensor
// we also need some arith and affine (for linalg::genericop)
struct ARangeLowering
    : public ::mlir::OpConversionPattern<::ptensor::ARangeOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::ptensor::ARangeOp op, ::ptensor::ARangeOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto converter = *getTypeConverter();

    // Get Operands
    auto start = adaptor.start();
    auto stop = adaptor.stop();
    auto step = adaptor.step();
    auto orgrtyp = op.getType().dyn_cast<::ptensor::PTensorType>();
    assert(orgrtyp);

    // we operator on signless integers
    auto ityp = rewriter.getI64Type();
    if (start.getType() != ityp) {
      start =
          rewriter.create<::mlir::UnrealizedConversionCastOp>(loc, ityp, start)
              .getResult(0);
    }
    if (stop.getType() != ityp) {
      stop =
          rewriter.create<::mlir::UnrealizedConversionCastOp>(loc, ityp, stop)
              .getResult(0);
    }
    if (step.getType() != ityp) {
      step =
          rewriter.create<::mlir::UnrealizedConversionCastOp>(loc, ityp, step)
              .getResult(0);
    }

    // Create constants 0, 1, -1 for later
    auto zattr = rewriter.getI64IntegerAttr(0);
    auto zero =
        rewriter.create<mlir::arith::ConstantOp>(loc, zattr).getResult();
    auto oattr = rewriter.getI64IntegerAttr(1);
    auto one = rewriter.create<mlir::arith::ConstantOp>(loc, oattr).getResult();
    auto mattr = rewriter.getI64IntegerAttr(-1);
    auto mone =
        rewriter.create<mlir::arith::ConstantOp>(loc, mattr).getResult();

    // Compute number of elements as (stop - start + step + (step < 0 ? 1 : -1))
    // / step
    auto cnd = rewriter.create<mlir::arith::CmpIOp>(
        loc, ::mlir::arith::CmpIPredicate::ult, step, zero);
    auto inc = rewriter.create<mlir::arith::SelectOp>(loc, cnd, one, mone);
    auto tmp1 = rewriter.create<mlir::arith::AddIOp>(loc, stop, step);
    auto tmp2 = rewriter.create<mlir::arith::AddIOp>(loc, tmp1, inc);
    auto tmp3 = rewriter.create<mlir::arith::SubIOp>(loc, tmp2, start);
    auto cnt =
        rewriter.create<mlir::arith::DivUIOp>(loc, tmp3, step).getResult();
    cnt = rewriter
              .create<::mlir::UnrealizedConversionCastOp>(
                  loc, ::mlir::IndexType::get(cnt.getType().getContext()), cnt)
              .getResult(0);

    // create shape vector
    auto ttyp = converter.convertType(op.getType())
                    .dyn_cast<::mlir::RankedTensorType>();
    assert(ttyp);
    auto typ = ttyp.getElementType();
    llvm::SmallVector<mlir::Value> shp(1, cnt);

    // register and init tensor
    llvm::SmallVector<mlir::Value> lshp(1);
    auto tmp_tnsr =
        rewriter.create<::mlir::linalg::InitTensorOp>(loc, shp, typ);
    auto shape = rewriter.create<::mlir::shape::ShapeOfOp>(loc, tmp_tnsr);
    auto tnsr_id =
        initDTensor(loc, rewriter, orgrtyp.getDist(), 1, shape, typ, lshp);

    // compute start index of local partition
    if (orgrtyp.getDist()) {
      auto offtyp =
          rewriter
              .getIndexType(); // mlir::MemRefType::get(llvm::SmallVector<int64_t>(1,
                               // mlir::ShapedType::kDynamicSize), ityp);
      auto offs =
          rewriter.create<::dist::LocalOffsetsOp>(loc, offtyp, tnsr_id.second);
      // auto _off = rewriter.create<::mlir::memref::DimOp>(loc, offs, 0);
      auto off = rewriter.create<mlir::arith::IndexCastOp>(loc, ityp, offs);
      auto tmp =
          rewriter.create<mlir::arith::MulIOp>(loc, off, step); // off * step
      start =
          rewriter.create<mlir::arith::AddIOp>(loc, start,
                                               tmp); // start + (off * stride)
    }

    // fill with arange values
    // map needed for output only (we have no input tensor)
    const ::mlir::AffineMap maps[] = {
        ::mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext())};
    llvm::SmallVector<mlir::StringRef> iterators(1, "parallel");

    // The body; accepting no input, the lambda simply captures start and step
    auto body = [&start, &step, &typ, &ityp](mlir::OpBuilder &builder,
                                             ::mlir::Location loc,
                                             ::mlir::ValueRange args) {
      auto dim = builder.getI64IntegerAttr(0);
      auto idx = builder.create<mlir::linalg::IndexOp>(loc, dim);
      auto _idx = builder.create<mlir::arith::IndexCastOp>(loc, ityp, idx);
      auto tmp = builder.create<mlir::arith::MulIOp>(loc, step, _idx);
      auto val = builder.create<mlir::arith::AddIOp>(loc, start, tmp);
      auto ret = builder.create<::mlir::UnrealizedConversionCastOp>(
          loc, typ, val.getResult());
      // auto _val = builder.create<mlir::arith::SIToFPOp>(loc, typ, val);
      (void)builder.create<mlir::linalg::YieldOp>(loc, ret.getResult(0));
    };

    (void)rewriter.replaceOpWithNewOp<mlir::linalg::GenericOp>(
        op, ttyp, llvm::None, tnsr_id.first, maps, iterators, body);
    return ::mlir::success();
  }
};

// function type for building body for linalg::generic
using BodyType = std::function<void(
    mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange args)>;

// any genericOp body needs to close with a yield
// we also add a cast op to "typ" if needed
template <typename T>
static void yield(mlir::OpBuilder &builder, ::mlir::Location loc,
                  ::mlir::Type typ, T val) {
  auto res = val;
  if (typ != res.getType()) {
    res = builder.create<::mlir::UnrealizedConversionCastOp>(loc, typ, res)
              .getResult(0);
  }
  (void)builder.create<mlir::linalg::YieldOp>(loc, res);
}

// trivial builders have simple arith equivalents
// the arith ops are template arguments, one for ints and one for floats
// currently only integers and floats are supported
// currently unsigned int ops are not supported
template <typename IOP, typename FOP = IOP>
static BodyType buildTrivial(::mlir::Type typ) {
  return [typ](mlir::OpBuilder &builder, ::mlir::Location loc,
               ::mlir::ValueRange args) -> void {
    auto lhs = doSignCast(builder, loc, args[0]);
    auto rhs = doSignCast(builder, loc, args[1]);
    if (lhs.getType().isIntOrIndex()) {
      yield(builder, loc, typ, builder.create<IOP>(loc, lhs, rhs).getResult());
    } else if (lhs.getType().isIntOrIndexOrFloat()) {
      yield(builder, loc, typ, builder.create<FOP>(loc, lhs, rhs).getResult());
    } else {
      assert("Only integers and floats supported for binary ops" == nullptr);
    }
  };
}

// get a body builder for given binary operation and result type
// we accept a result type to insert a cast after the operation if needed
static BodyType getBodyBuilder(::ptensor::EWBinOpId bop, ::mlir::Type typ) {
  switch (bop) {
  case ptensor::ADD:
    return buildTrivial<mlir::arith::AddIOp, mlir::arith::AddFOp>(typ);
  // case ptensor::ATAN2] =
  case ptensor::FLOOR_DIVIDE:
    return buildTrivial<mlir::arith::FloorDivSIOp>(typ);
  // case ptensor::LOGADDEXP] =
  // case ptensor::LSHIFT] =
  // case ptensor::MATMUL] =
  case ptensor::MAXIMUM:
    return buildTrivial<mlir::arith::MaxSIOp, mlir::arith::MaxFOp>(typ);
  case ptensor::MINIMUM:
    return buildTrivial<mlir::arith::MinSIOp, mlir::arith::MinFOp>(typ);
  case ptensor::MODULO:
    return buildTrivial<mlir::arith::RemSIOp, mlir::arith::RemFOp>(typ);
  case ptensor::MULTIPLY:
    return buildTrivial<mlir::arith::MulIOp, mlir::arith::MulFOp>(typ);
  // case ptensor::POW] =
  case ptensor::SUBTRACT:
    return buildTrivial<mlir::arith::SubIOp, mlir::arith::SubFOp>(typ);
  // case ptensor::TRUE_DIVIDE] =
  // case ptensor::BITWISE_AND] =
  // case ptensor::BITWISE_LEFT_SHIFT] =
  // case ptensor::BITWISE_OR] =
  // case ptensor::BITWISE_RIGHT_SHIFT] =
  // case ptensor::BITWISE_XOR] =

  // case ptensor::EQUAL] =
  // case ptensor::GREATER] =
  // case ptensor::GREATER_EQUAL] =
  // case ptensor::LESS] =
  // case ptensor::LESS_EQUAL] =
  // case ptensor::LOGICAL_AND] =
  // case ptensor::LOGICAL_OR] =
  // case ptensor::LOGICAL_XOR] =
  // case ptensor::NOT_EQUAL] =
  default:
    assert("unsupported elementwise binary operation" == nullptr);
  };
}

// convert PTensor's elementwise binary operations and their return type to
// Linalg/tensor the given op's type is expected to convert to the apprioprate
// type (shape and element-type) we also need some arith and affine (for
// linalg::genericop)
// Convert PTensor's elementwise binary operations to Linalg
struct EWBinOpLowering
    : public ::mlir::OpConversionPattern<::ptensor::EWBinOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::ptensor::EWBinOp op, ::ptensor::EWBinOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // We expect to lower PTensors
    auto lhsorgtyp = op.lhs().getType().dyn_cast<::ptensor::PTensorType>();
    auto rhsorgtyp = op.rhs().getType().dyn_cast<::ptensor::PTensorType>();
    // we expect RankedTensorType as operands
    auto lhstyp = adaptor.lhs().getType().dyn_cast<::mlir::RankedTensorType>();
    auto rhstyp = adaptor.rhs().getType().dyn_cast<::mlir::RankedTensorType>();
    if (!lhstyp || !rhstyp || !lhsorgtyp || !rhsorgtyp) {
      // fail if not, will be retired if operands get converted elsewhere
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto converter = *getTypeConverter();

    // input tensors might have compatible but different types
    assert(adaptor.lhs().getType() == adaptor.rhs().getType());
    assert(adaptor.lhs().getType() == adaptor.rhs().getType());

    // the element type of a binop depends on the input arguments and the
    // operation itself we assume this had beeen taken care of and simply use
    // the op's converted type
    auto _t = converter.convertType(op.getType());
    auto shaped = _t.dyn_cast<::mlir::RankedTensorType>();
    assert(shaped);
    auto typ = shaped.getElementType();

    // build tensor using the resulting element type
    // the shape is not statically known, we need to retrieve it (it's the same
    // as the input shapes)
    // FIXME shape broadcasting: input tensors might have compatible but
    // different shapes
    auto lhs = adaptor.lhs();
    auto rank = static_cast<unsigned>(shaped.getRank());
    llvm::SmallVector<mlir::Value> shp(rank);
    llvm::SmallVector<mlir::StringRef> iterators(rank);
    for (auto i : llvm::seq(0u, rank)) {
      shp[i] = rewriter.create<::mlir::tensor::DimOp>(loc, lhs, i);
      // iterate in parallel
      iterators[i] = "parallel";
    }

    // register and init tensor
    llvm::SmallVector<mlir::Value> lshp(rank);
    auto tmp_tnsr =
        rewriter.create<::mlir::linalg::InitTensorOp>(loc, shp, typ);
    auto shape = rewriter.create<::mlir::shape::ShapeOfOp>(loc, tmp_tnsr);
    auto tnsr_id =
        initDTensor(loc, rewriter, lhsorgtyp.getDist(), rank, shape, typ, lshp);

    // Get signless operands into vec
    llvm::SmallVector<mlir::Value, 2> oprnds = {adaptor.lhs(), adaptor.rhs()};

    // all maps are identity maps
    auto imap =
        ::mlir::AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    const ::mlir::AffineMap maps[] = {imap, imap, imap};

    // create binop as linalg::generic
    const ::ptensor::EWBinOpId bopid =
        (::ptensor::EWBinOpId)adaptor.op().cast<::mlir::IntegerAttr>().getInt();
    auto bodyBuilder = getBodyBuilder(bopid, typ);
    (void)rewriter
        .replaceOpWithNewOp<::mlir::linalg::GenericOp>(
            op, tnsr_id.first.getType(), oprnds, tnsr_id.first, maps, iterators,
            bodyBuilder)
        .getResult(0);
    return ::mlir::success();
  }
};

// get a body builder for giben binary operation and result type
// we accept a result type to insert a cast after the operation if needed
static BodyType getBodyBuilder(::ptensor::ReduceOpId rop, ::mlir::Type typ) {
  switch (rop) {
  case ::ptensor::PROD:
    return getBodyBuilder(::ptensor::MULTIPLY, typ);
  case ::ptensor::SUM:
    return getBodyBuilder(::ptensor::ADD, typ);
  case ::ptensor::MAX:
    return getBodyBuilder(::ptensor::MAXIMUM, typ);
  case ::ptensor::MIN:
    return getBodyBuilder(::ptensor::MINIMUM, typ);
  case ::ptensor::MEAN:
  case ::ptensor::STD:
  case ::ptensor::VAR:
  default:
    assert("unsupported reduction operation" == nullptr);
  };
}

// convert PTensor's reduction operations and their return type to Linalg/tensor
// the given op's type is expected to convert to the apprioprate type (shape and
// element-type) we also need some arith and affine (for linalg::genericop)
// FIXME reduction over a subset of dimensionsstruct ReductionOpLowering
struct ReductionOpLowering
    : public ::mlir::OpConversionPattern<::ptensor::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::ptensor::ReductionOp op,
                  ::ptensor::ReductionOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // we expect RankedTensorType as operands
    auto inptyp =
        adaptor.input().getType().dyn_cast<::mlir::RankedTensorType>();
    auto orginptyp = op.input().getType().dyn_cast<::ptensor::PTensorType>();
    if (!inptyp || !orginptyp) {
      // fail if not, will be retired if operands get converted elsewhere
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto converter = *getTypeConverter();

    // Get signless operands into vec
    llvm::SmallVector<mlir::Value, 1> oprnds = {adaptor.input()};

    // determine resulting element type from converted op-type
    auto _t = converter.convertType(op.getType());
    auto shaped = _t.dyn_cast<::mlir::RankedTensorType>();
    assert(shaped);
    auto typ = shaped.getElementType();
    auto sltyp = makeSignlessType(typ);

    // build tensor using the resulting element type and shape
    // FIXME support reduction dimensions
    auto rank = static_cast<unsigned>(shaped.getRank());
    assert(rank == 0);
    llvm::SmallVector<mlir::Value> shp(0); //::mlir::ShapedType::kDynamicSize;
    // create new tensor
    auto zattr = rewriter.getI64IntegerAttr(0);
    auto zero =
        rewriter.create<mlir::arith::ConstantOp>(loc, zattr).getResult();
    llvm::SmallVector<mlir::Value> lshp(rank);
    auto tmp_tnsr =
        rewriter.create<::mlir::linalg::InitTensorOp>(loc, shp, typ);
    auto shape = rewriter.create<::mlir::shape::ShapeOfOp>(loc, tmp_tnsr);
    auto tnsr_id = initDTensor(loc, rewriter, orginptyp.getDist(), rank, shape,
                               sltyp, lshp);
    auto tnsr =
        rewriter.create<::mlir::linalg::FillOp>(loc, zero, tnsr_id.first);

    // rank/num-dims of input
    auto irank = static_cast<unsigned>(inptyp.getRank());
    // input maps are identity maps
    auto imap =
        ::mlir::AffineMap::getMultiDimIdentityMap(irank, rewriter.getContext());
    // output map is "*->()"
    auto omap = ::mlir::AffineMap::get(irank, 0, rewriter.getContext());
    const ::mlir::AffineMap maps[] = {imap, omap};
    llvm::SmallVector<mlir::StringRef> iterators(irank, "reduction");

    // create reduction op as linalg::generic
    const ::ptensor::ReduceOpId ropid = (::ptensor::ReduceOpId)adaptor.op()
                                            .cast<::mlir::IntegerAttr>()
                                            .getInt();
    auto bodyBuilder = getBodyBuilder(ropid, sltyp);
    auto rtnsr = rewriter
                     .create<::mlir::linalg::GenericOp>(
                         loc, tnsr.getType(0), oprnds, tnsr.getResult(0), maps,
                         iterators, bodyBuilder)
                     .getResult(0);

    // we reduced the local part, now we reduce across processes
    if (orginptyp.getDist()) {
      rtnsr = rewriter.create<::dist::AllReduceOp>(loc, tnsr.getType(0),
                                                   adaptor.op(), rtnsr);
    }

    // For now we only support reduction over all dims and return a scalar
    auto rval = rewriter.create<::mlir::tensor::ExtractOp>(
        loc, sltyp, rtnsr, ::mlir::ValueRange());
    (void)rewriter.replaceOpWithNewOp<::mlir::UnrealizedConversionCastOp>(
        op, typ, rval.getResult());

    return ::mlir::success();
  }
};

// *******************************
// ***** Pass infrastructure *****
// *******************************

namespace imex {

/// Populate the given list with patterns that eliminate Dist ops
void populateDistElimConversionPatterns(::mlir::LLVMTypeConverter &converter,
                                        ::mlir::RewritePatternSet &patterns);

// Converting PTensor to Linalg
// After success, no more PTensor should be left, replaced by Linalg & Affine &
// Arith We use a type converter to get rid of PTensorType
struct PTensorToLinalgPass
    : public ::mlir::PassWrapper<PTensorToLinalgPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  virtual void
  getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<::ptensor::PTensorDialect>();
    registry.insert<::dist::DistDialect>();
    registry.insert<::mlir::linalg::LinalgDialect>();
    registry.insert<::mlir::AffineDialect>();
    registry.insert<::mlir::func::FuncDialect>();
    registry.insert<::mlir::tensor::TensorDialect>();
    registry.insert<::mlir::arith::ArithmeticDialect>();
  }

  void runOnOperation() override {
    auto &ctxt = getContext();
    ::mlir::ConversionTarget target(ctxt);
    ::mlir::TypeConverter typeConverter;
    // Convert unknown types to itself
    typeConverter.addConversion([](::mlir::Type type) { return type; });
    // Convert PTensorType to its RankedTensorType
    typeConverter.addConversion(
        [&typeConverter](::ptensor::PTensorType type)
            -> llvm::Optional<::mlir::Type> { return type.getRtensor(); });

#if 1
    // In theory we should not need any materialization
    // if we use a hybrid conversion (plier->ptensor->linalg and direct
    // plier->linalg) we might need it, though
    auto materializeCast =
        [](::mlir::OpBuilder &builder, ::mlir::Type type,
           ::mlir::ValueRange inputs,
           ::mlir::Location loc) -> ::llvm::Optional<::mlir::Value> {
      if (inputs.size() == 1) {
        return builder
            .create<::mlir::UnrealizedConversionCastOp>(loc, type,
                                                        inputs.front())
            .getResult(0);
      }
      return ::llvm::None;
    };
    // typeConverter.addArgumentMaterialization(materializeCast);
    typeConverter.addSourceMaterialization(materializeCast);
    // typeConverter.addTargetMaterialization(materializeCast);
#endif
    // We convert all PTensor stuff...
    target.addIllegalDialect<::ptensor::PTensorDialect>();
    // ...into Linalg, Affine, Tensor, Arith, Dist
    target.addLegalDialect<::dist::DistDialect>();
    target.addLegalDialect<::mlir::linalg::LinalgDialect>();
    target.addLegalDialect<::mlir::AffineDialect>();
    target.addLegalDialect<::mlir::tensor::TensorDialect>();
    target.addLegalDialect<::mlir::arith::ArithmeticDialect>();

    ::mlir::RewritePatternSet patterns(&ctxt);
#define FIXME 0
#if FIXME
    // For now, we also use plier's SignCastOp
    target.addLegalOp<::plier::SignCastOp>();

    // add rewrites/conversions for return types/ops and other control flow
    // stuff
    plier::populateControlFlowTypeConversionRewritesAndTarget(typeConverter,
                                                              patterns, target);
#endif
    patterns.insert<ARangeLowering, EWBinOpLowering, ReductionOpLowering>(
        typeConverter, &ctxt);

    if (::mlir::failed(::mlir::applyPartialConversion(getOperation(), target,
                                                      ::std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertDistElimPass() {
  return std::make_unique<PTensorToLinalgPass>();
}

} // namespace imex
