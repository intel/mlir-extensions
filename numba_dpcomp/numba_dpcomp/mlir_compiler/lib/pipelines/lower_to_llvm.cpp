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

#include "pipelines/lower_to_llvm.hpp"

#include <mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/LoweringOptions.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MathToLibm/MathToLibm.h>
#include <mlir/Conversion/MemRefToLLVM/AllocLikeConversion.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Arithmetic/Transforms/Passes.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/Triple.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>

#include "base_pipeline.hpp"
#include "pipelines/plier_to_std.hpp"

#include "mlir-extensions/compiler/pipeline_registry.hpp"
#include "mlir-extensions/Dialect/plier/dialect.hpp"
#include "mlir-extensions/Dialect/plier_util/dialect.hpp"
#include "mlir-extensions/Transforms/func_utils.hpp"
#include "mlir-extensions/Transforms/type_conversion.hpp"
#include "mlir-extensions/utils.hpp"

namespace {
mlir::LowerToLLVMOptions getLLVMOptions(mlir::MLIRContext &context) {
  static llvm::DataLayout dl = []() {
    llvm::InitializeNativeTarget();
    auto triple = llvm::sys::getProcessTriple();
    std::string err_str;
    auto target = llvm::TargetRegistry::lookupTarget(triple, err_str);
    if (nullptr == target) {
      plier::reportError(llvm::Twine("Unable to get target: ") + err_str);
    }
    llvm::TargetOptions target_opts;
    std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
        triple, llvm::sys::getHostCPUName(), "", target_opts, llvm::None));
    return machine->createDataLayout();
  }();
  mlir::LowerToLLVMOptions opts(&context);
  opts.dataLayout = dl;
  opts.useBarePtrCallConv = false;
  opts.emitCWrappers = false;
  opts.allocLowering = mlir::LowerToLLVMOptions::AllocLowering::None;
  return opts;
}

mlir::Value doCast(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value src, mlir::Type dstType) {
  if (src.getType() == dstType)
    return src;

  return builder.create<mlir::UnrealizedConversionCastOp>(loc, dstType, src)
      .getResult(0);
}

mlir::Type convertTupleTypes(mlir::MLIRContext &context,
                             mlir::TypeConverter &converter,
                             mlir::TypeRange types) {
  if (types.empty())
    return mlir::LLVM::LLVMStructType::getLiteral(&context, llvm::None);

  auto unitupleType = [&]() -> mlir::Type {
    assert(!types.empty());
    auto elemType = types.front();
    auto tail = types.drop_front();
    if (llvm::all_of(tail, [&](auto t) { return t == elemType; }))
      return elemType;
    return nullptr;
  }();

  auto count = static_cast<unsigned>(types.size());
  if (unitupleType) {
    auto newType = converter.convertType(unitupleType);
    if (!newType)
      return {};
    return mlir::LLVM::LLVMArrayType::get(newType, count);
  }
  llvm::SmallVector<mlir::Type> newTypes;
  newTypes.reserve(count);
  for (auto type : types) {
    auto newType = converter.convertType(type);
    if (!newType)
      return {};
    newTypes.emplace_back(newType);
  }

  return mlir::LLVM::LLVMStructType::getLiteral(&context, newTypes);
}

mlir::Type convertTuple(mlir::MLIRContext &context,
                        mlir::TypeConverter &converter, mlir::TupleType tuple) {
  return convertTupleTypes(context, converter, tuple.getTypes());
}

void populateToLLVMAdditionalTypeConversion(
    mlir::LLVMTypeConverter &converter) {
  converter.addConversion(
      [&converter](mlir::TupleType type) -> llvm::Optional<mlir::Type> {
        auto res = convertTuple(*type.getContext(), converter, type);
        if (!res)
          return llvm::None;
        return res;
      });
  auto voidPtrType = mlir::LLVM::LLVMPointerType::get(
      mlir::IntegerType::get(&converter.getContext(), 8));
  converter.addConversion(
      [voidPtrType](mlir::NoneType) -> llvm::Optional<mlir::Type> {
        return voidPtrType;
      });
  converter.addConversion(
      [voidPtrType](plier::OpaqueType) -> llvm::Optional<mlir::Type> {
        return voidPtrType;
      });
  converter.addConversion(
      [voidPtrType](plier::TypeVar) -> llvm::Optional<mlir::Type> {
        return voidPtrType;
      });
}

struct LLVMTypeHelper {
  LLVMTypeHelper(mlir::MLIRContext &ctx) : type_converter(&ctx) {
    populateToLLVMAdditionalTypeConversion(type_converter);
  }

  mlir::Type i(unsigned bits) {
    return mlir::IntegerType::get(&type_converter.getContext(), bits);
  }

  mlir::Type ptr(mlir::Type type) {
    assert(static_cast<bool>(type));
    auto ll_type = type_converter.convertType(type);
    assert(static_cast<bool>(ll_type));
    return mlir::LLVM::LLVMPointerType::get(ll_type);
  }

  mlir::MLIRContext &get_context() { return type_converter.getContext(); }

  mlir::LLVMTypeConverter &get_type_converter() { return type_converter; }

private:
  mlir::LLVMTypeConverter type_converter;
};

mlir::Type getExceptInfoType(LLVMTypeHelper &type_helper) {
  mlir::Type elems[] = {
      type_helper.ptr(type_helper.i(8)),
      type_helper.i(32),
      type_helper.ptr(type_helper.i(8)),
  };
  return mlir::LLVM::LLVMStructType::getLiteral(&type_helper.get_context(),
                                                elems);
}

mlir::LLVM::LLVMStructType get_array_type(mlir::TypeConverter &converter,
                                          mlir::MemRefType type) {
  assert(type);
  auto ctx = type.getContext();
  auto i8p = mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(ctx, 8));
  auto i64 = mlir::IntegerType::get(ctx, 64);
  auto dataType = converter.convertType(type.getElementType());
  assert(dataType);
  if (type.getRank() > 0) {
    auto shapeType = mlir::LLVM::LLVMArrayType::get(
        i64, static_cast<unsigned>(type.getRank()));
    const mlir::Type members[] = {
        i8p,                                        // 0, meminfo
        i8p,                                        // 1, parent
        i64,                                        // 2, nitems
        i64,                                        // 3, itemsize
        mlir::LLVM::LLVMPointerType::get(dataType), // 4, data
        shapeType,                                  // 5, shape
        shapeType,                                  // 6, strides
    };
    return mlir::LLVM::LLVMStructType::getLiteral(ctx, members);
  } else {
    const mlir::Type members[] = {
        i8p,                                        // 0, meminfo
        i8p,                                        // 1, parent
        i64,                                        // 2, nitems
        i64,                                        // 3, itemsize
        mlir::LLVM::LLVMPointerType::get(dataType), // 4, data
    };
    return mlir::LLVM::LLVMStructType::getLiteral(ctx, members);
  }
}

template <typename F> void flatten_type(mlir::Type type, F &&func) {
  if (auto struct_type = type.dyn_cast<mlir::LLVM::LLVMStructType>()) {
    for (auto elem : struct_type.getBody()) {
      flatten_type(elem, std::forward<F>(func));
    }
  } else if (auto arr_type = type.dyn_cast<mlir::LLVM::LLVMArrayType>()) {
    auto elem = arr_type.getElementType();
    auto size = arr_type.getNumElements();
    for (unsigned i = 0; i < size; ++i) {
      flatten_type(elem, std::forward<F>(func));
    }
  } else {
    func(type);
  }
}

template <typename F>
mlir::Value unflatten(mlir::Type type, mlir::Location loc,
                      mlir::OpBuilder &builder, F &&next_func) {
  namespace mllvm = mlir::LLVM;
  if (auto struct_type = type.dyn_cast<mlir::LLVM::LLVMStructType>()) {
    mlir::Value val = builder.create<mllvm::UndefOp>(loc, struct_type);
    for (auto elem : llvm::enumerate(struct_type.getBody())) {
      auto elem_index =
          builder.getI64ArrayAttr(static_cast<int64_t>(elem.index()));
      auto elem_type = elem.value();
      auto elem_val =
          unflatten(elem_type, loc, builder, std::forward<F>(next_func));
      val = builder.create<mlir::LLVM::InsertValueOp>(loc, val, elem_val,
                                                      elem_index);
    }
    return val;
  } else if (auto arr_type = type.dyn_cast<mlir::LLVM::LLVMArrayType>()) {
    auto elem_type = arr_type.getElementType();
    auto size = arr_type.getNumElements();
    mlir::Value val = builder.create<mllvm::UndefOp>(loc, arr_type);
    for (unsigned i = 0; i < size; ++i) {
      auto elem_index = builder.getI64ArrayAttr(static_cast<int64_t>(i));
      auto elem_val =
          unflatten(elem_type, loc, builder, std::forward<F>(next_func));
      val = builder.create<mlir::LLVM::InsertValueOp>(loc, val, elem_val,
                                                      elem_index);
    }
    return val;
  } else {
    return next_func();
  }
}

void write_memref_desc(llvm::raw_ostream &os, mlir::MemRefType memref_type) {
  if (memref_type.hasRank()) {
    auto rank = memref_type.getRank();
    assert(rank >= 0);
    if (rank > 0) {
      os << memref_type.getRank() << "x";
    }
  } else {
    os << "?x";
  }
  memref_type.getElementType().print(os);
}

std::string gen_to_memref_conversion_func_name(mlir::MemRefType memref_type) {
  assert(memref_type);
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  ss << "__convert_to_memref_";
  write_memref_desc(ss, memref_type);
  ss.flush();
  return ret;
}

std::string gen_from_memref_conversion_func_name(mlir::MemRefType memref_type) {
  assert(memref_type);
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  ss << "__convert_from_memref_";
  write_memref_desc(ss, memref_type);
  ss.flush();
  return ret;
}

mlir::Value div_strides(mlir::Location loc, mlir::OpBuilder &builder,
                        mlir::Value strides, mlir::Value m) {
  auto array_type = strides.getType().cast<mlir::LLVM::LLVMArrayType>();
  mlir::Value array = builder.create<mlir::LLVM::UndefOp>(loc, array_type);
  auto count = array_type.getNumElements();
  for (unsigned i = 0; i < count; ++i) {
    auto index = builder.getI64ArrayAttr(i);
    auto prev = builder.create<mlir::LLVM::ExtractValueOp>(
        loc, array_type.getElementType(), strides, index);
    auto val = builder.create<mlir::LLVM::SDivOp>(loc, prev, m);
    array = builder.create<mlir::LLVM::InsertValueOp>(loc, array, val, index);
  }
  return array;
}

mlir::Value mul_strides(mlir::Location loc, mlir::OpBuilder &builder,
                        mlir::Value strides, mlir::Value m) {
  auto array_type = strides.getType().cast<mlir::LLVM::LLVMArrayType>();
  mlir::Value array = builder.create<mlir::LLVM::UndefOp>(loc, array_type);
  auto count = array_type.getNumElements();
  for (unsigned i = 0; i < count; ++i) {
    auto index = builder.getI64ArrayAttr(i);
    auto prev = builder.create<mlir::LLVM::ExtractValueOp>(
        loc, array_type.getElementType(), strides, index);
    auto val = builder.create<mlir::LLVM::MulOp>(loc, prev, m);
    array = builder.create<mlir::LLVM::InsertValueOp>(loc, array, val, index);
  }
  return array;
}

unsigned item_size(mlir::Type type) {
  if (auto inttype = type.dyn_cast<mlir::IntegerType>()) {
    assert((inttype.getWidth() % 8) == 0);
    return inttype.getWidth() / 8;
  }
  if (auto floattype = type.dyn_cast<mlir::FloatType>()) {
    assert((floattype.getWidth() % 8) == 0);
    return floattype.getWidth() / 8;
  }
  llvm_unreachable("item_size: invalid type");
}

mlir::func::FuncOp
get_to_memref_conversion_func(mlir::ModuleOp module, mlir::OpBuilder &builder,
                              mlir::MemRefType memrefType,
                              mlir::LLVM::LLVMStructType src_type,
                              mlir::LLVM::LLVMStructType dst_type) {
  assert(memrefType);
  assert(src_type);
  assert(dst_type);
  auto func_name = gen_to_memref_conversion_func_name(memrefType);
  if (auto func = module.lookupSymbol<mlir::func::FuncOp>(func_name)) {
    assert(func.getFunctionType().getNumResults() == 1);
    assert(func.getFunctionType().getResult(0) == dst_type);
    return func;
  }
  auto func_type =
      mlir::FunctionType::get(builder.getContext(), src_type, dst_type);
  auto loc = builder.getUnknownLoc();
  auto new_func = plier::add_function(builder, module, func_name, func_type);
  auto alwaysinline =
      mlir::StringAttr::get(builder.getContext(), "alwaysinline");
  new_func->setAttr("passthrough",
                    mlir::ArrayAttr::get(builder.getContext(), alwaysinline));
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto block = new_func.addEntryBlock();
  builder.setInsertionPointToStart(block);
  namespace mllvm = mlir::LLVM;
  mlir::Value arg = block->getArgument(0);
  auto extract = [&](unsigned index) {
    auto res_type = src_type.getBody()[index];
    auto i = builder.getI64ArrayAttr(index);
    return builder.create<mllvm::ExtractValueOp>(loc, res_type, arg, i);
  };
  auto meminfo = extract(0);
  auto ptr = extract(4);
  auto rank = memrefType.getRank();
  auto shape = (rank > 0 ? extract(5) : mlir::Value());
  auto strides = (rank > 0 ? extract(6) : mlir::Value());
  auto i64 = mlir::IntegerType::get(builder.getContext(), 64);
  auto offset =
      builder.create<mllvm::ConstantOp>(loc, i64, builder.getI64IntegerAttr(0));
  mlir::Value res = builder.create<mllvm::UndefOp>(loc, dst_type);
  auto meminfo_casted =
      builder.create<mllvm::BitcastOp>(loc, ptr.getType(), meminfo);
  auto itemsize = builder.create<mllvm::ConstantOp>(
      loc, i64,
      builder.getI64IntegerAttr(item_size(memrefType.getElementType())));
  auto insert = [&](unsigned index, mlir::Value val) {
    auto i = builder.getI64ArrayAttr(index);
    res = builder.create<mllvm::InsertValueOp>(loc, res, val, i);
  };
  insert(0, meminfo_casted);
  insert(1, ptr);
  insert(2, offset);
  if (rank > 0) {
    insert(3, shape);
    insert(4, div_strides(loc, builder, strides, itemsize));
  }
  builder.create<mllvm::ReturnOp>(loc, res);
  return new_func;
}

mlir::func::FuncOp
get_from_memref_conversion_func(mlir::ModuleOp module, mlir::OpBuilder &builder,
                                mlir::MemRefType memrefType,
                                mlir::LLVM::LLVMStructType src_type,
                                mlir::LLVM::LLVMStructType dst_type) {
  assert(memrefType);
  assert(src_type);
  assert(dst_type);
  auto func_name = gen_from_memref_conversion_func_name(memrefType);
  if (auto func = module.lookupSymbol<mlir::func::FuncOp>(func_name)) {
    assert(func.getFunctionType().getNumResults() == 1);
    assert(func.getFunctionType().getResult(0) == dst_type);
    return func;
  }
  auto func_type =
      mlir::FunctionType::get(builder.getContext(), src_type, dst_type);
  auto loc = builder.getUnknownLoc();
  auto new_func = plier::add_function(builder, module, func_name, func_type);
  auto alwaysinline =
      mlir::StringAttr::get(builder.getContext(), "alwaysinline");
  new_func->setAttr("passthrough",
                    mlir::ArrayAttr::get(builder.getContext(), alwaysinline));
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto block = new_func.addEntryBlock();
  builder.setInsertionPointToStart(block);
  namespace mllvm = mlir::LLVM;
  mlir::Value arg = block->getArgument(0);
  auto i8ptr_type = mllvm::LLVMPointerType::get(builder.getIntegerType(8));
  auto i64_type = builder.getIntegerType(64);
  auto extract = [&](unsigned index) {
    auto res_type = src_type.getBody()[index];
    auto i = builder.getI64ArrayAttr(index);
    return builder.create<mllvm::ExtractValueOp>(loc, res_type, arg, i);
  };
  auto meminfo = builder.create<mllvm::BitcastOp>(loc, i8ptr_type, extract(0));
  auto orig_ptr = extract(1);
  auto offset = extract(2);
  auto rank = memrefType.getRank();
  auto shape = (rank > 0 ? extract(3) : mlir::Value());
  auto strides = (rank > 0 ? extract(4) : mlir::Value());
  auto ptr = builder.create<mllvm::GEPOp>(loc, orig_ptr.getType(), orig_ptr,
                                          offset.getResult());
  mlir::Value res = builder.create<mllvm::UndefOp>(loc, dst_type);
  auto null = builder.create<mllvm::NullOp>(loc, i8ptr_type);
  mlir::Value nitems = builder.create<mllvm::ConstantOp>(
      loc, i64_type, builder.getI64IntegerAttr(1));
  for (int64_t i = 0; i < rank; ++i) {
    auto dim = builder.create<mllvm::ExtractValueOp>(
        loc, nitems.getType(), shape, builder.getI64ArrayAttr(i));
    nitems = builder.create<mllvm::MulOp>(loc, nitems, dim);
  }
  auto itemsize = builder.create<mllvm::ConstantOp>(
      loc, i64_type,
      builder.getI64IntegerAttr(item_size(memrefType.getElementType())));
  auto insert = [&](unsigned index, mlir::Value val) {
    auto i = builder.getI64ArrayAttr(index);
    res = builder.create<mllvm::InsertValueOp>(loc, res, val, i);
  };
  insert(0, meminfo);
  insert(1, null); // parent
  insert(2, nitems);
  insert(3, itemsize);
  insert(4, ptr);
  if (rank > 0) {
    insert(5, shape);
    insert(6, mul_strides(loc, builder, strides, itemsize));
  }
  builder.create<mllvm::ReturnOp>(loc, res);
  return new_func;
}

mlir::Attribute get_fastmath_attrs(mlir::MLIRContext &ctx) {
  auto add_pair = [&](auto name, auto val) {
    const mlir::Attribute attrs[] = {mlir::StringAttr::get(&ctx, name),
                                     mlir::StringAttr::get(&ctx, val)};
    return mlir::ArrayAttr::get(&ctx, attrs);
  };
  const mlir::Attribute attrs[] = {
      add_pair("denormal-fp-math", "preserve-sign,preserve-sign"),
      add_pair("denormal-fp-math-f32", "ieee,ieee"),
      add_pair("no-infs-fp-math", "true"),
      add_pair("no-nans-fp-math", "true"),
      add_pair("no-signed-zeros-fp-math", "true"),
      add_pair("unsafe-fp-math", "true"),
      add_pair(plier::attributes::getFastmathName(), "1"),
  };
  return mlir::ArrayAttr::get(&ctx, attrs);
}

mlir::Type getFunctionResType(mlir::MLIRContext &context,
                              mlir::TypeConverter &converter,
                              mlir::TypeRange types) {
  if (types.empty())
    return mlir::LLVM::LLVMPointerType::get(
        mlir::IntegerType::get(&context, 8));

  llvm::SmallVector<mlir::Type> newResTypes(types.size());
  for (auto it : llvm::enumerate(types)) {
    auto i = it.index();
    auto type = it.value();
    if (auto memreftype = type.dyn_cast<mlir::MemRefType>()) {
      newResTypes[i] = get_array_type(converter, memreftype);
    } else {
      newResTypes[i] = type;
    }
  }

  if (newResTypes.size() == 1)
    return newResTypes.front();

  return convertTupleTypes(context, converter, newResTypes);
}

mlir::LogicalResult fixFuncSig(LLVMTypeHelper &typeHelper,
                               mlir::func::FuncOp func) {
  if (func.isPrivate())
    return mlir::success();

  if (func->getAttr(plier::attributes::getFastmathName()))
    func->setAttr("passthrough", get_fastmath_attrs(*func.getContext()));

  auto oldType = func.getFunctionType();
  auto &ctx = *oldType.getContext();
  llvm::SmallVector<mlir::Type> args;

  auto ptr = [&](auto arg) { return typeHelper.ptr(arg); };

  mlir::OpBuilder builder(&ctx);
  auto uloc = builder.getUnknownLoc();
  unsigned index = 0;
  auto addArg = [&](mlir::Type type) {
    args.push_back(type);
    auto ret = func.getBody().insertArgument(index, type, uloc);
    ++index;
    return ret;
  };

  auto &typeConverter = typeHelper.get_type_converter();
  auto &context = typeConverter.getContext();
  auto origRetType =
      getFunctionResType(context, typeConverter, oldType.getResults());
  if (!origRetType)
    return mlir::failure();

  if (!typeConverter.convertType(origRetType)) {
    func->emitError("fixFuncSig: couldn't convert return type: ")
        << origRetType;
    return mlir::failure();
  }

  builder.setInsertionPointToStart(&func.getBody().front());

  auto loc = builder.getUnknownLoc();
  llvm::SmallVector<mlir::Value> newArgs;
  auto processArg = [&](mlir::Type type) {
    if (auto memrefType = type.dyn_cast<mlir::MemRefType>()) {
      newArgs.clear();
      auto arrType = get_array_type(typeConverter, memrefType);
      flatten_type(arrType, [&](mlir::Type new_type) {
        newArgs.push_back(addArg(new_type));
      });
      auto it = newArgs.begin();
      mlir::Value desc = unflatten(arrType, loc, builder, [&]() {
        auto ret = *it;
        ++it;
        return ret;
      });

      auto mod = mlir::cast<mlir::ModuleOp>(func->getParentOp());
      auto dstType = typeConverter.convertType(memrefType);
      assert(dstType);
      auto convFunc = get_to_memref_conversion_func(
          mod, builder, memrefType, arrType,
          dstType.cast<mlir::LLVM::LLVMStructType>());
      auto converted =
          builder.create<mlir::func::CallOp>(loc, convFunc, desc).getResult(0);
      auto casted = doCast(builder, loc, converted, memrefType);
      func.getBody().getArgument(index).replaceAllUsesWith(casted);
      func.getBody().eraseArgument(index);
    } else {
      args.push_back(type);
      ++index;
    }
  };

  addArg(ptr(origRetType));
  addArg(ptr(ptr(getExceptInfoType(typeHelper))));

  auto oldArgs = oldType.getInputs();
  for (auto arg : oldArgs) {
    processArg(arg);
  }
  auto retType = mlir::IntegerType::get(&ctx, 32);
  func.setType(mlir::FunctionType::get(&ctx, args, retType));
  return mlir::success();
}

struct ReturnOpLowering : public mlir::OpRewritePattern<mlir::func::ReturnOp> {
  ReturnOpLowering(mlir::MLIRContext *ctx, mlir::TypeConverter &converter)
      : OpRewritePattern(ctx), typeConverter(converter) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::func::ReturnOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto parent = op->getParentOfType<mlir::func::FuncOp>();
    if (nullptr == parent || parent.isPrivate())
      return mlir::failure();

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto convertVal = [&](mlir::Value val) -> mlir::Value {
      auto origType = val.getType();
      auto llRetType = typeConverter.convertType(origType);
      if (!llRetType)
        return {};

      if (origType.isa<mlir::NoneType>())
        return rewriter.create<mlir::LLVM::NullOp>(loc, llRetType);

      val = doCast(rewriter, loc, val, llRetType);
      if (auto memrefType = origType.dyn_cast<mlir::MemRefType>()) {
        auto dstType = get_array_type(typeConverter, memrefType)
                           .cast<mlir::LLVM::LLVMStructType>();
        auto func = get_from_memref_conversion_func(
            mod, rewriter, memrefType,
            llRetType.cast<mlir::LLVM::LLVMStructType>(), dstType);
        val = rewriter.create<mlir::func::CallOp>(loc, func, val).getResult(0);
      }
      return val;
    };
    rewriter.setInsertionPoint(op);
    auto addr = op->getParentRegion()->front().getArgument(0);
    if (op.getNumOperands() == 0) {
      assert(addr.getType().isa<mlir::LLVM::LLVMPointerType>());
      auto nullType =
          addr.getType().cast<mlir::LLVM::LLVMPointerType>().getElementType();
      auto llVal = rewriter.create<mlir::LLVM::NullOp>(op.getLoc(), nullType);
      rewriter.create<mlir::LLVM::StoreOp>(loc, llVal, addr);
    } else if (op.getNumOperands() == 1) {
      mlir::Value val = convertVal(op.getOperand(0));
      if (!val)
        return mlir::failure();
      rewriter.create<mlir::LLVM::StoreOp>(loc, val, addr);
    } else {
      auto resType =
          getFunctionResType(*ctx, typeConverter, op.getOperandTypes());
      auto val = rewriter.create<mlir::LLVM::UndefOp>(loc, resType).getResult();
      for (auto it : llvm::enumerate(op.operands())) {
        auto arg = convertVal(it.value());
        if (!arg)
          return mlir::failure();

        auto index = rewriter.getI64ArrayAttr(static_cast<int64_t>(it.index()));
        val = rewriter.create<mlir::LLVM::InsertValueOp>(loc, val, arg, index);
      }
      rewriter.create<mlir::LLVM::StoreOp>(loc, val, addr);
    }
    auto retType = mlir::IntegerType::get(ctx, 32);
    mlir::Value ret = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, retType, mlir::IntegerAttr::get(retType, 0));
    rewriter.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(op, ret);
    return mlir::success();
  }

private:
  mlir::TypeConverter &typeConverter;
};

template <typename Op>
struct ApplyFastmathFlags : public mlir::OpRewritePattern<Op> {
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    auto parent = mlir::cast<mlir::LLVM::LLVMFuncOp>(op->getParentOp());
    bool changed = false;

    rewriter.startRootUpdate(op);
    auto fmf = op.getFastmathFlags();
    getFastmathFlags(parent, [&](auto flag) {
      if (!mlir::LLVM::bitEnumContains(fmf, flag)) {
        fmf = fmf | flag;
        changed = true;
      }
    });
    if (changed) {
      op.setFastmathFlagsAttr(mlir::LLVM::FMFAttr::get(op.getContext(), fmf));
      rewriter.finalizeRootUpdate(op);
    } else {
      rewriter.cancelRootUpdate(op);
    }

    return mlir::success(changed);
  }

private:
  template <typename F>
  static void getFastmathFlags(mlir::LLVM::LLVMFuncOp func, F &&sink) {
    if (func->hasAttr(plier::attributes::getFastmathName())) {
      sink(mlir::LLVM::FastmathFlags::fast);
    }
  }
};

struct AllocOpLowering : public mlir::AllocLikeOpLLVMLowering {
  AllocOpLowering(mlir::LLVMTypeConverter &converter)
      : AllocLikeOpLLVMLowering(mlir::memref::AllocOp::getOperationName(),
                                converter) {}

  std::tuple<mlir::Value, mlir::Value>
  allocateBuffer(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                 mlir::Value sizeBytes, mlir::Operation *op) const override {
    auto allocOp = mlir::cast<mlir::memref::AllocOp>(op);
    auto memRefType = allocOp.getType();
    mlir::Value alignment;
    if (auto alignmentAttr = allocOp.alignment()) {
      alignment = createIndexConstant(rewriter, loc, *alignmentAttr);
    } else if (!memRefType.getElementType().isSignlessIntOrIndexOrFloat()) {
      // In the case where no alignment is specified, we may want to override
      // `malloc's` behavior. `malloc` typically aligns at the size of the
      // biggest scalar on a target HW. For non-scalars, use the natural
      // alignment of the LLVM type given by the LLVM DataLayout.
      alignment = getSizeInBytes(loc, memRefType.getElementType(), rewriter);
    } else {
      alignment = createIndexConstant(
          rewriter, loc, 32 /*item_size(memRefType.getElementType())*/);
    }
    alignment = rewriter.create<mlir::LLVM::TruncOp>(
        loc, rewriter.getIntegerType(32), alignment);

    auto mod = allocOp->getParentOfType<mlir::ModuleOp>();
    auto meminfo_ptr =
        createAllocCall(loc, "NRT_MemInfo_alloc_safe_aligned", getVoidPtrType(),
                        {sizeBytes, alignment}, mod, rewriter);
    auto data_ptr =
        createAllocCall(loc, "NRT_MemInfo_data_fast", getVoidPtrType(),
                        {meminfo_ptr}, mod, rewriter);

    auto elem_ptr_type =
        mlir::LLVM::LLVMPointerType::get(memRefType.getElementType());
    auto bitcast = [&](mlir::Value val) {
      return rewriter.create<mlir::LLVM::BitcastOp>(loc, elem_ptr_type, val);
    };

    return std::make_tuple(bitcast(meminfo_ptr), bitcast(data_ptr));
  }

private:
  mlir::Value createAllocCall(mlir::Location loc, mlir::StringRef name,
                              mlir::Type ptrType,
                              mlir::ArrayRef<mlir::Value> params,
                              mlir::ModuleOp module,
                              mlir::ConversionPatternRewriter &rewriter) const {
    using namespace mlir;
    SmallVector<Type, 2> paramTypes;
    auto allocFuncOp = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
    if (!allocFuncOp) {
      for (Value param : params)
        paramTypes.push_back(param.getType());
      auto allocFuncType =
          LLVM::LLVMFunctionType::get(getVoidPtrType(), paramTypes);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      allocFuncOp = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                      name, allocFuncType);
    }

    auto allocFuncSymbol = mlir::SymbolRefAttr::get(allocFuncOp);
    auto allocatedPtr = rewriter
                            .create<LLVM::CallOp>(loc, getVoidPtrType(),
                                                  allocFuncSymbol, params)
                            .getResult(0);
    return rewriter.create<LLVM::BitcastOp>(loc, ptrType, allocatedPtr);
  }
};

struct DeallocOpLowering
    : public mlir::ConvertOpToLLVMPattern<mlir::memref::DeallocOp> {
  using ConvertOpToLLVMPattern<mlir::memref::DeallocOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::DeallocOp op,
                  mlir::memref::DeallocOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Insert the `free` declaration if it is not already present.
    auto freeFunc = op->getParentOfType<mlir::ModuleOp>()
                        .lookupSymbol<mlir::LLVM::LLVMFuncOp>("NRT_decref");
    if (!freeFunc) {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(
          op->getParentOfType<mlir::ModuleOp>().getBody());
      freeFunc = rewriter.create<mlir::LLVM::LLVMFuncOp>(
          rewriter.getUnknownLoc(), "NRT_decref",
          mlir::LLVM::LLVMFunctionType::get(getVoidType(), getVoidPtrType()));
    }

    mlir::MemRefDescriptor memref(adaptor.memref());
    mlir::Value casted = rewriter.create<mlir::LLVM::BitcastOp>(
        op.getLoc(), getVoidPtrType(),
        memref.allocatedPtr(rewriter, op.getLoc()));
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, mlir::TypeRange(), mlir::SymbolRefAttr::get(freeFunc), casted);
    return mlir::success();
  }
};

struct ReshapeLowering
    : public mlir::ConvertOpToLLVMPattern<mlir::memref::ReshapeOp> {
  using ConvertOpToLLVMPattern<mlir::memref::ReshapeOp>::ConvertOpToLLVMPattern;

  explicit ReshapeLowering(mlir::LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<mlir::memref::ReshapeOp>(converter) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::ReshapeOp op,
                  mlir::memref::ReshapeOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto dstType = converter->convertType(op.getType());
    if (!dstType)
      return mlir::failure();

    mlir::MemRefDescriptor source(adaptor.source());
    mlir::MemRefDescriptor shape(adaptor.shape());

    auto loc = op.getLoc();
    auto result = mlir::MemRefDescriptor::undef(rewriter, loc, dstType);
    result.setAllocatedPtr(rewriter, loc, source.allocatedPtr(rewriter, loc));
    result.setAlignedPtr(rewriter, loc, source.alignedPtr(rewriter, loc));
    result.setOffset(rewriter, loc, source.offset(rewriter, loc));

    auto memRefType = op.getType().cast<mlir::MemRefType>();
    auto numDims = memRefType.getRank();
    llvm::SmallVector<mlir::Value> sizes(static_cast<unsigned>(numDims));
    auto indexType = getIndexType();
    for (unsigned i = 0; i < numDims; ++i) {
      auto ind = createIndexConstant(rewriter, loc, i);
      mlir::Value dataPtr =
          getStridedElementPtr(loc, memRefType, shape, ind, rewriter);
      auto size = rewriter.create<mlir::LLVM::LoadOp>(loc, dataPtr).getResult();
      if (size.getType() != indexType)
        size = rewriter.create<mlir::LLVM::ZExtOp>(loc, indexType, size);

      result.setSize(rewriter, loc, i, size);
      sizes[i] = size;
    }

    // Strides: iterate sizes in reverse order and multiply.
    int64_t stride = 1;
    mlir::Value runningStride = createIndexConstant(rewriter, loc, 1);
    for (auto i = static_cast<unsigned>(memRefType.getRank()); i-- > 0;) {
      result.setStride(rewriter, loc, i, runningStride);

      int64_t size = memRefType.getShape()[i];
      if (size == 0)
        continue;
      bool useSizeAsStride = stride == 1;
      if (size == mlir::ShapedType::kDynamicSize)
        stride = mlir::ShapedType::kDynamicSize;
      if (stride != mlir::ShapedType::kDynamicSize)
        stride *= size;

      if (useSizeAsStride)
        runningStride = sizes[i];
      else if (stride == mlir::ShapedType::kDynamicSize)
        runningStride =
            rewriter.create<mlir::LLVM::MulOp>(loc, runningStride, sizes[i]);
      else
        runningStride =
            createIndexConstant(rewriter, loc, static_cast<uint64_t>(stride));
    }

    rewriter.replaceOp(op, static_cast<mlir::Value>(result));
    return mlir::success();
  }
};

struct ExpandShapeLowering
    : public mlir::ConvertOpToLLVMPattern<mlir::memref::ExpandShapeOp> {
  using ConvertOpToLLVMPattern<
      mlir::memref::ExpandShapeOp>::ConvertOpToLLVMPattern;

  explicit ExpandShapeLowering(mlir::LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<mlir::memref::ExpandShapeOp>(converter) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::ExpandShapeOp op,
                  mlir::memref::ExpandShapeOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto dstType = op.getType().cast<mlir::MemRefType>();
    auto converter = getTypeConverter();
    auto dstLlvmType = converter->convertType(dstType);
    if (!dstLlvmType)
      return mlir::failure();

    auto inds = op.getReassociationIndices();

    int64_t currentDstInd = 0;
    auto dstShape = dstType.getShape();
    for (auto &ind : inds) {
      int64_t reassocInd = -1;
      for (auto i : ind) {
        assert(i >= 0);
        if (dstShape[currentDstInd] != 1) {
          if (reassocInd != -1)
            return mlir::failure();
          reassocInd = i;
        }
        ++currentDstInd;
      }
    }

    mlir::MemRefDescriptor src(adaptor.src());

    auto loc = op->getLoc();
    auto result = mlir::MemRefDescriptor::undef(rewriter, loc, dstLlvmType);
    result.setAllocatedPtr(rewriter, loc, src.allocatedPtr(rewriter, loc));
    result.setAlignedPtr(rewriter, loc, src.alignedPtr(rewriter, loc));
    result.setOffset(rewriter, loc, src.offset(rewriter, loc));

    auto indexType = getIndexType();
    auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, indexType, rewriter.getIntegerAttr(indexType, 0));
    auto one = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, indexType, rewriter.getIntegerAttr(indexType, 1));

    currentDstInd = 0;
    for (auto it : llvm::enumerate(inds)) {
      auto &ind = it.value();
      auto srcInd = it.index();
      for (auto i : ind) {
        assert(i >= 0);
        mlir::Value dim;
        mlir::Value stride;
        if (dstShape[currentDstInd] != 1) {
          dim = src.size(rewriter, loc, srcInd);
          stride = src.stride(rewriter, loc, srcInd);
        } else {
          dim = one;
          stride = zero;
        }
        result.setSize(rewriter, loc, currentDstInd, dim);
        result.setStride(rewriter, loc, currentDstInd, stride);
        ++currentDstInd;
      }
    }

    rewriter.replaceOp(op, static_cast<mlir::Value>(result));
    return mlir::success();
  }
};

class LLVMFunctionPass : public mlir::OperationPass<mlir::LLVM::LLVMFuncOp> {
public:
  using OperationPass<mlir::LLVM::LLVMFuncOp>::OperationPass;

  /// The polymorphic API that runs the pass over the currently held function.
  virtual void runOnFunction() = 0;

  /// The polymorphic API that runs the pass over the currently held operation.
  void runOnOperation() final {
    if (!getFunction().isExternal())
      runOnFunction();
  }

  /// Return the current function being transformed.
  mlir::LLVM::LLVMFuncOp getFunction() { return this->getOperation(); }
};

void copyAttrs(mlir::Operation *src, mlir::Operation *dst) {
  const mlir::StringRef attrs[] = {
      plier::attributes::getFastmathName(),
      plier::attributes::getParallelName(),
      plier::attributes::getMaxConcurrencyName(),
  };
  for (auto name : attrs) {
    if (auto attr = src->getAttr(name)) {
      dst->setAttr(name, attr);
    }
  }
}

struct LowerParallel : public mlir::OpRewritePattern<plier::ParallelOp> {
  LowerParallel(mlir::MLIRContext *context)
      : OpRewritePattern(context), converter(context) {}

  mlir::LogicalResult
  matchAndRewrite(plier::ParallelOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto num_loops = op.getNumLoops();
    llvm::SmallVector<mlir::Value> contextVars;
    llvm::SmallVector<mlir::Operation *> contextConstants;
    llvm::DenseSet<mlir::Value> context_vars_set;
    auto addContextVar = [&](mlir::Value value) {
      if (0 != context_vars_set.count(value)) {
        return;
      }
      context_vars_set.insert(value);
      if (auto op = value.getDefiningOp()) {
        if (op->hasTrait<mlir::OpTrait::ConstantLike>()) {
          contextConstants.emplace_back(op);
          return;
        }
      }
      contextVars.emplace_back(value);
    };

    auto isDefinedInside = [&](mlir::Value value) {
      auto &thisRegion = op.getLoopBody();
      auto opRegion = value.getParentRegion();
      assert(nullptr != opRegion);
      do {
        if (opRegion == &thisRegion) {
          return true;
        }
        opRegion = opRegion->getParentRegion();
      } while (nullptr != opRegion);
      return false;
    };

    if (op->walk([&](mlir::Operation *inner) -> mlir::WalkResult {
            if (op != inner) {
              for (auto arg : inner->getOperands()) {
                if (!isDefinedInside(arg)) {
                  addContextVar(arg);
                }
              }
            }
            return mlir::WalkResult::advance();
          })
            .wasInterrupted()) {
      return mlir::failure();
    }

    auto contextType = [&]() -> mlir::LLVM::LLVMStructType {
      llvm::SmallVector<mlir::Type> fields;
      fields.reserve(contextVars.size());
      for (auto var : contextVars) {
        auto type = converter.convertType(var.getType());
        if (!type) {
          return {};
        }
        fields.emplace_back(type);
      }
      return mlir::LLVM::LLVMStructType::getLiteral(op.getContext(), fields);
    }();

    if (!contextType) {
      return mlir::failure();
    }

    plier::AllocaInsertionPoint allocaInsertionPoint(op);

    auto contextPtrType = mlir::LLVM::LLVMPointerType::get(contextType);

    auto loc = op.getLoc();
    auto indexType = rewriter.getIndexType();
    auto llvmIndexType = converter.getIndexType();
    auto toLLVMIndex = [&](mlir::Value val) -> mlir::Value {
      if (val.getType() != llvmIndexType) {
        return rewriter
            .create<mlir::UnrealizedConversionCastOp>(loc, llvmIndexType, val)
            .getResult(0);
      }
      return val;
    };
    auto fromLLVMIndex = [&](mlir::Value val) -> mlir::Value {
      if (val.getType() != indexType) {
        return doCast(rewriter, loc, val, indexType);
      }
      return val;
    };
    auto llvmI32Type = mlir::IntegerType::get(op.getContext(), 32);
    auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmI32Type, rewriter.getI32IntegerAttr(0));
    auto context = allocaInsertionPoint.insert(rewriter, [&]() {
      auto one = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmI32Type, rewriter.getI32IntegerAttr(1));
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, contextPtrType, one, 0);
    });

    for (auto it : llvm::enumerate(contextVars)) {
      auto type = contextType.getBody()[it.index()];
      auto llvmVal = doCast(rewriter, loc, it.value(), type);
      auto i = rewriter.getI32IntegerAttr(static_cast<int32_t>(it.index()));
      mlir::Value indices[] = {
          zero, rewriter.create<mlir::LLVM::ConstantOp>(loc, llvmI32Type, i)};
      auto pointerType = mlir::LLVM::LLVMPointerType::get(type);
      auto ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, pointerType, context,
                                                    indices);
      rewriter.create<mlir::LLVM::StoreOp>(loc, llvmVal, ptr);
    }
    auto voidPtrType = mlir::LLVM::LLVMPointerType::get(
        mlir::IntegerType::get(op.getContext(), 8));
    auto contextAbstract =
        rewriter.create<mlir::LLVM::BitcastOp>(loc, voidPtrType, context);

    auto inputRangeType = [&]() {
      const mlir::Type members[] = {
          llvmIndexType, // lower_bound
          llvmIndexType, // upper_bound
          llvmIndexType, // step
      };
      return mlir::LLVM::LLVMStructType::getLiteral(op.getContext(), members);
    }();
    auto inputRangePtr = mlir::LLVM::LLVMPointerType::get(inputRangeType);
    auto rangeType = [&]() {
      const mlir::Type members[] = {
          llvmIndexType, // lower_bound
          llvmIndexType, // upper_bound
      };
      return mlir::LLVM::LLVMStructType::getLiteral(op.getContext(), members);
    }();
    auto rangePtr = mlir::LLVM::LLVMPointerType::get(rangeType);
    auto funcType = [&]() {
      const mlir::Type args[] = {
          rangePtr,   // bounds
          indexType,  // thread index
          voidPtrType // context
      };
      return mlir::FunctionType::get(op.getContext(), args, {});
    }();

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    auto outlinedFunc = [&]() -> mlir::func::FuncOp {
      auto func = [&]() {
        auto parentFunc = op->getParentOfType<mlir::func::FuncOp>();
        assert(parentFunc);
        auto func_name = [&]() {
          auto old_name = parentFunc.getName();
          for (int i = 0;; ++i) {
            auto name =
                (0 == i
                     ? (llvm::Twine(old_name) + "_outlined").str()
                     : (llvm::Twine(old_name) + "_outlined_" + llvm::Twine(i))
                           .str());
            if (!mod.lookupSymbol<mlir::func::FuncOp>(name)) {
              return name;
            }
          }
        }();

        auto func = plier::add_function(rewriter, mod, func_name, funcType);
        copyAttrs(parentFunc, func);
        return func;
      }();
      mlir::BlockAndValueMapping mapping;
      auto &oldEntry = op.getLoopBody().front();
      auto entry = func.addEntryBlock();
      auto loc = rewriter.getUnknownLoc();
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(entry);
      auto pos0 = rewriter.getI64ArrayAttr(0);
      auto pos1 = rewriter.getI64ArrayAttr(1);
      for (unsigned i = 0; i < num_loops; ++i) {
        auto arg = entry->getArgument(0);
        const mlir::Value indices[] = {rewriter.create<mlir::LLVM::ConstantOp>(
            loc, llvmI32Type,
            rewriter.getI32IntegerAttr(static_cast<int32_t>(i)))};
        auto ptr =
            rewriter.create<mlir::LLVM::GEPOp>(loc, rangePtr, arg, indices);
        auto dims = rewriter.create<mlir::LLVM::LoadOp>(loc, ptr);
        auto lower = rewriter.create<mlir::LLVM::ExtractValueOp>(
            loc, llvmIndexType, dims, pos0);
        auto upper = rewriter.create<mlir::LLVM::ExtractValueOp>(
            loc, llvmIndexType, dims, pos1);
        mapping.map(oldEntry.getArgument(i), fromLLVMIndex(lower));
        mapping.map(oldEntry.getArgument(i + num_loops), fromLLVMIndex(upper));
      }
      mapping.map(oldEntry.getArgument(2 * num_loops),
                  entry->getArgument(1)); // thread index
      for (auto arg : contextConstants)
        rewriter.clone(*arg, mapping);

      auto contextPtr = rewriter.create<mlir::LLVM::BitcastOp>(
          loc, contextPtrType, entry->getArgument(2));
      auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmI32Type, rewriter.getI32IntegerAttr(0));
      for (auto it : llvm::enumerate(contextVars)) {
        auto index = it.index();
        auto oldVal = it.value();
        const mlir::Value indices[] = {
            zero, rewriter.create<mlir::LLVM::ConstantOp>(
                      loc, llvmI32Type,
                      rewriter.getI32IntegerAttr(static_cast<int32_t>(index)))};
        auto pointerType =
            mlir::LLVM::LLVMPointerType::get(contextType.getBody()[index]);
        auto ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, pointerType,
                                                      contextPtr, indices);
        auto llvmVal = rewriter.create<mlir::LLVM::LoadOp>(loc, ptr);
        auto val = doCast(rewriter, loc, llvmVal, oldVal.getType());
        mapping.map(oldVal, val);
      }
      op.getLoopBody().cloneInto(&func.getBody(), mapping);
      auto &orig_entry = *std::next(func.getBody().begin());
      rewriter.create<mlir::cf::BranchOp>(loc, &orig_entry);
      for (auto &block : func.getBody()) {
        if (auto term = mlir::dyn_cast<plier::YieldOp>(block.getTerminator())) {
          rewriter.eraseOp(term);
          rewriter.setInsertionPointToEnd(&block);
          rewriter.create<mlir::func::ReturnOp>(loc);
        }
      }
      return func;
    }();

    auto parallelFor = [&]() {
      auto func_name = "dpcompParallelFor";
      if (auto sym = mod.lookupSymbol<mlir::func::FuncOp>(func_name)) {
        return sym;
      }
      const mlir::Type args[] = {
          inputRangePtr, // bounds
          indexType,     // num_loops
          funcType,      // func
          voidPtrType    // context
      };
      auto parallelFuncType =
          mlir::FunctionType::get(op.getContext(), args, {});
      return plier::add_function(rewriter, mod, func_name, parallelFuncType);
    }();
    auto funcAddr = rewriter.create<mlir::func::ConstantOp>(
        loc, funcType, mlir::SymbolRefAttr::get(outlinedFunc));

    auto inputRanges = allocaInsertionPoint.insert(rewriter, [&]() {
      auto numLoopsAttr = rewriter.getIntegerAttr(llvmIndexType, num_loops);
      auto numLoopsVar = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmIndexType, numLoopsAttr);
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, inputRangePtr,
                                                   numLoopsVar, 0);
    });
    for (unsigned i = 0; i < num_loops; ++i) {
      mlir::Value inputRange =
          rewriter.create<mlir::LLVM::UndefOp>(loc, inputRangeType);
      auto insert = [&](mlir::Value val, unsigned index) {
        inputRange = rewriter.create<mlir::LLVM::InsertValueOp>(
            loc, inputRange, val, rewriter.getI64ArrayAttr(index));
      };
      insert(toLLVMIndex(op.lowerBounds()[i]), 0);
      insert(toLLVMIndex(op.upperBounds()[i]), 1);
      insert(toLLVMIndex(op.steps()[i]), 2);
      const mlir::Value indices[] = {rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmI32Type, rewriter.getI32IntegerAttr(static_cast<int>(i)))};
      auto ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, inputRangePtr,
                                                    inputRanges, indices);
      rewriter.create<mlir::LLVM::StoreOp>(loc, inputRange, ptr);
    }

    auto numLoopsVar =
        rewriter.create<mlir::arith::ConstantIndexOp>(loc, num_loops);
    const mlir::Value pfArgs[] = {inputRanges, numLoopsVar, funcAddr,
                                  contextAbstract};
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(op, parallelFor, pfArgs);
    return mlir::success();
  }

private:
  mutable mlir::LLVMTypeConverter converter; // TODO
};

struct LowerParallelToCFGPass
    : public mlir::PassWrapper<LowerParallelToCFGPass,
                               mlir::OperationPass<void>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override final {
    auto &context = getContext();
    mlir::RewritePatternSet patterns(&context);
    patterns.insert<LowerParallel>(&getContext());

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};

struct PreLLVMLowering
    : public mlir::PassWrapper<PreLLVMLowering,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override final {
    auto &context = getContext();
    LLVMTypeHelper type_helper(context);

    mlir::RewritePatternSet patterns(&context);
    auto func = getOperation();
    if (mlir::failed(fixFuncSig(type_helper, func))) {
      signalPassFailure();
      return;
    }

    patterns.insert<ReturnOpLowering>(&context,
                                      type_helper.get_type_converter());

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};

struct PostLLVMLowering
    : public mlir::PassWrapper<PostLLVMLowering, LLVMFunctionPass> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  void runOnFunction() override final {
    auto &context = getContext();
    mlir::RewritePatternSet patterns(&context);

    patterns.insert<ApplyFastmathFlags<mlir::LLVM::FAddOp>,
                    ApplyFastmathFlags<mlir::LLVM::FSubOp>,
                    ApplyFastmathFlags<mlir::LLVM::FMulOp>,
                    ApplyFastmathFlags<mlir::LLVM::FDivOp>,
                    ApplyFastmathFlags<mlir::LLVM::FRemOp>,
                    ApplyFastmathFlags<mlir::LLVM::FCmpOp>,
                    ApplyFastmathFlags<mlir::LLVM::CallOp>>(&context);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};

struct LowerRetainOp : public mlir::ConvertOpToLLVMPattern<plier::RetainOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::RetainOp op, plier::RetainOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto arg = adaptor.source();
    if (!arg.getType().isa<mlir::LLVM::LLVMStructType>())
      return mlir::failure();

    auto llvmVoidPointerType = getVoidPtrType();
    auto incref_func = [&]() {
      auto mod = op->getParentOfType<mlir::ModuleOp>();
      assert(mod);
      auto func = mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>("NRT_incref");
      if (!func) {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        auto llvmVoidType = getVoidType();
        func = rewriter.create<mlir::LLVM::LLVMFuncOp>(
            rewriter.getUnknownLoc(), "NRT_incref",
            mlir::LLVM::LLVMFunctionType::get(llvmVoidType,
                                              llvmVoidPointerType));
      }
      return func;
    }();

    mlir::MemRefDescriptor source(arg);

    auto loc = op.getLoc();
    mlir::Value ptr = source.allocatedPtr(rewriter, loc);
    ptr = rewriter.create<mlir::LLVM::BitcastOp>(loc, llvmVoidPointerType, ptr);
    rewriter.create<mlir::LLVM::CallOp>(loc, incref_func, ptr);
    rewriter.replaceOp(op, arg);

    return mlir::success();
  }
};

struct LowerExtractMemrefMetadataOp
    : public mlir::ConvertOpToLLVMPattern<plier::ExtractMemrefMetadataOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::ExtractMemrefMetadataOp op,
                  plier::ExtractMemrefMetadataOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto arg = adaptor.source();
    if (!arg.getType().isa<mlir::LLVM::LLVMStructType>())
      return mlir::failure();

    auto loc = op.getLoc();
    mlir::MemRefDescriptor src(arg);
    auto val = [&]() -> mlir::Value {
      auto index = op.dimIndex().getSExtValue();
      if (index == -1)
        return src.offset(rewriter, loc);

      return src.stride(rewriter, loc, static_cast<unsigned>(index));
    }();

    rewriter.replaceOp(op, static_cast<mlir::Value>(val));
    return mlir::success();
  }
};

struct LowerBuildTuple
    : public mlir::ConvertOpToLLVMPattern<plier::BuildTupleOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::BuildTupleOp op, plier::BuildTupleOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto type = converter->convertType(op.getType());
    if (!type)
      return mlir::failure();

    auto loc = op.getLoc();
    mlir::Value init = rewriter.create<mlir::LLVM::UndefOp>(loc, type);
    for (auto it : llvm::enumerate(adaptor.args())) {
      auto arg = it.value();
      auto newType = arg.getType();
      assert(newType);
      auto casted = doCast(rewriter, loc, arg, newType);
      auto index = rewriter.getI64ArrayAttr(static_cast<int64_t>(it.index()));
      init =
          rewriter.create<mlir::LLVM::InsertValueOp>(loc, init, casted, index);
    }

    rewriter.replaceOp(op, init);
    return mlir::success();
  }
};

struct LowerUndef : public mlir::ConvertOpToLLVMPattern<plier::UndefOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::UndefOp op, plier::UndefOp::Adaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto type = converter->convertType(op.getType());
    if (!type)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::LLVM::UndefOp>(op, type);
    return mlir::success();
  }
};

static void addToGlobalDtors(mlir::ConversionPatternRewriter &rewriter,
                             mlir::ModuleOp mod, mlir::SymbolRefAttr attr,
                             int32_t priority) {
  auto loc = mod->getLoc();
  auto dtorOps = mod.getOps<mlir::LLVM::GlobalDtorsOp>();
  auto prioAttr = rewriter.getI32IntegerAttr(priority);
  mlir::OpBuilder::InsertionGuard g(rewriter);
  if (dtorOps.empty()) {
    rewriter.setInsertionPoint(mod.getBody(), std::prev(mod.getBody()->end()));
    auto syms = rewriter.getArrayAttr(attr);
    auto priorities = rewriter.getArrayAttr(prioAttr);
    rewriter.create<mlir::LLVM::GlobalDtorsOp>(loc, syms, priorities);
    return;
  }
  assert(llvm::hasSingleElement(dtorOps));
  auto dtorOp = *dtorOps.begin();

  auto addpendArray = [&](mlir::ArrayAttr arr,
                          mlir::Attribute attr) -> mlir::ArrayAttr {
    auto vals = arr.getValue();
    llvm::SmallVector<mlir::Attribute> ret(vals.begin(), vals.end());
    ret.emplace_back(attr);
    return rewriter.getArrayAttr(ret);
  };
  auto newDtors = addpendArray(dtorOp.getDtors(), attr);
  auto newPrioritiess = addpendArray(dtorOp.getPriorities(), prioAttr);
  rewriter.setInsertionPoint(dtorOp);
  rewriter.create<mlir::LLVM::GlobalDtorsOp>(loc, newDtors, newPrioritiess);
  rewriter.eraseOp(dtorOp);
}

struct LowerTakeContextOp
    : public mlir::ConvertOpToLLVMPattern<plier::TakeContextOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::TakeContextOp op,
                  plier::TakeContextOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter);
    auto ctx = op.context();
    auto ctxType = converter->convertType(ctx.getType());
    if (!ctxType)
      return mlir::failure();

    mlir::ValueRange results = op.results();
    auto resultsCount = static_cast<unsigned>(results.size());
    llvm::SmallVector<mlir::Type> resultTypes(resultsCount);
    for (auto i : llvm::seq(0u, resultsCount)) {
      auto type = converter->convertType(results[i].getType());
      if (!type)
        return mlir::failure();

      resultTypes[i] = type;
    }

    auto ctxStructType =
        mlir::LLVM::LLVMStructType::getLiteral(getContext(), resultTypes);
    auto ctxStructPtrType = mlir::LLVM::LLVMPointerType::get(ctxStructType);

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    assert(mod);

    auto unknownLoc = rewriter.getUnknownLoc();
    auto loc = op->getLoc();
    auto wrapperType =
        mlir::LLVM::LLVMFunctionType::get(getVoidType(), ctxType);
    auto wrapperPtrType = mlir::LLVM::LLVMPointerType::get(wrapperType);
    mlir::Value initFuncPtr;

    auto insertFunc = [&](mlir::StringRef name, mlir::Type type,
                          mlir::LLVM::Linkage linkage) {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      return rewriter.create<mlir::LLVM::LLVMFuncOp>(unknownLoc, name, type,
                                                     linkage);
    };

    auto lookupFunc = [&](mlir::StringRef name, mlir::Type type) {
      // TODO: fix and use lookupOrCreateFn
      if (auto func = mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name))
        return func;

      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      return rewriter.create<mlir::LLVM::LLVMFuncOp>(
          unknownLoc, name, type, mlir::LLVM::Linkage::External);
    };

    if (auto initFuncSym = adaptor.initFuncAttr()) {
      auto funcName = initFuncSym.getLeafReference().getValue();
      auto wrapperName = (funcName + "_wrapper").str();

      auto initFunc = [&]() {
        mlir::OpBuilder::InsertionGuard g(rewriter);
        auto func =
            insertFunc(wrapperName, wrapperType, mlir::LLVM::Linkage::Private);
        auto block = rewriter.createBlock(
            &func.getBody(), mlir::Region::iterator{}, ctxType, unknownLoc);
        rewriter.setInsertionPointToStart(block);

        auto innerResults = rewriter
                                .create<mlir::func::CallOp>(
                                    unknownLoc, initFuncSym, results.getTypes())
                                ->getResults();

        mlir::Value ctxStruct =
            rewriter.create<mlir::LLVM::UndefOp>(unknownLoc, ctxStructType);
        for (auto i : llvm::seq(0u, resultsCount)) {
          auto pos = rewriter.getI64ArrayAttr(i);
          auto val = rewriter.getRemappedValue(innerResults[i]);
          ctxStruct = rewriter.create<mlir::LLVM::InsertValueOp>(
              unknownLoc, ctxStruct, val, pos);
        }
        auto ptr = rewriter.create<mlir::LLVM::BitcastOp>(
            unknownLoc, ctxStructPtrType, block->getArgument(0));
        rewriter.create<mlir::LLVM::StoreOp>(unknownLoc, ctxStruct, ptr);
        rewriter.create<mlir::LLVM::ReturnOp>(unknownLoc, llvm::None);
        return func;
      }();

      initFuncPtr = rewriter.create<mlir::LLVM::AddressOfOp>(loc, initFunc);
    } else {
      initFuncPtr = rewriter.create<mlir::LLVM::NullOp>(loc, wrapperPtrType);
    }

    mlir::Value deinitFuncPtr;
    if (auto deinitFuncSym = adaptor.releaseFuncAttr()) {
      auto funcName = deinitFuncSym.getLeafReference().getValue();
      auto wrapperName = (funcName + "_wrapper").str();

      auto deinitFunc = [&]() {
        mlir::OpBuilder::InsertionGuard g(rewriter);
        auto func =
            insertFunc(wrapperName, wrapperType, mlir::LLVM::Linkage::Private);
        auto block = rewriter.createBlock(
            &func.getBody(), mlir::Region::iterator{}, ctxType, unknownLoc);
        rewriter.setInsertionPointToStart(block);

        auto ptr = rewriter.create<mlir::LLVM::BitcastOp>(
            unknownLoc, ctxStructPtrType, block->getArgument(0));
        auto ctxStruct =
            rewriter.create<mlir::LLVM::LoadOp>(unknownLoc, ctxStructType, ptr);

        llvm::SmallVector<mlir::Value> args(resultsCount);
        for (auto i : llvm::seq(0u, resultsCount)) {
          auto pos = rewriter.getI64ArrayAttr(i);
          mlir::Value val = rewriter.create<mlir::LLVM::ExtractValueOp>(
              unknownLoc, resultTypes[i], ctxStruct, pos);
          auto resType = results[i].getType();
          if (val.getType() != resType)
            val = rewriter
                      .create<mlir::UnrealizedConversionCastOp>(unknownLoc,
                                                                resType, val)
                      .getResult(0);

          args[i] = val;
        }

        rewriter.create<mlir::func::CallOp>(unknownLoc, deinitFuncSym,
                                            llvm::None, args);
        rewriter.create<mlir::LLVM::ReturnOp>(unknownLoc, llvm::None);
        return func;
      }();

      deinitFuncPtr = rewriter.create<mlir::LLVM::AddressOfOp>(loc, deinitFunc);
    } else {
      deinitFuncPtr = rewriter.create<mlir::LLVM::NullOp>(loc, wrapperPtrType);
    }

    auto takeCtxFunc = [&]() -> mlir::LLVM::LLVMFuncOp {
      llvm::StringRef name("dpcompTakeContext");
      auto retType = getVoidPtrType();
      const mlir::Type argTypes[] = {
          mlir::LLVM::LLVMPointerType::get(getVoidPtrType()),
          getIndexType(),
          wrapperPtrType,
          wrapperPtrType,
      };
      auto funcType = mlir::LLVM::LLVMFunctionType::get(retType, argTypes);
      return lookupFunc(name, funcType);
    }();

    auto purgeCtxFunc = [&]() -> mlir::LLVM::LLVMFuncOp {
      llvm::StringRef name("dpcompPurgeContext");
      auto retType = getVoidType();
      auto argType = mlir::LLVM::LLVMPointerType::get(getVoidPtrType());
      auto funcType = mlir::LLVM::LLVMFunctionType::get(retType, argType);
      return lookupFunc(name, funcType);
    }();

    auto ctxHandle = [&]() {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      llvm::StringRef name("context_handle"); // TODO: unique name
      auto handle = rewriter.create<mlir::LLVM::GlobalOp>(
          unknownLoc, ctxType, /*isConstant*/ false,
          mlir::LLVM::Linkage::Internal, name, mlir::Attribute());

      llvm::StringRef cleanupFuncName(".dpcomp_context_cleanup");
      auto cleanupFunc =
          mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>(cleanupFuncName);
      if (!cleanupFunc) {
        auto cleanupFuncType =
            mlir::LLVM::LLVMFunctionType::get(getVoidType(), llvm::None);
        cleanupFunc = rewriter.create<mlir::LLVM::LLVMFuncOp>(
            unknownLoc, cleanupFuncName, cleanupFuncType);
        auto block = rewriter.createBlock(&cleanupFunc.getBody());
        rewriter.setInsertionPointToStart(block);
        rewriter.create<mlir::LLVM::ReturnOp>(unknownLoc, llvm::None);

        addToGlobalDtors(rewriter, mod, mlir::SymbolRefAttr::get(cleanupFunc),
                         0);
      }

      assert(llvm::hasSingleElement(cleanupFunc.getBody()));
      rewriter.setInsertionPointToStart(&cleanupFunc.getBody().front());
      mlir::Value addr =
          rewriter.create<mlir::LLVM::AddressOfOp>(unknownLoc, handle);
      rewriter.create<mlir::LLVM::CallOp>(unknownLoc, purgeCtxFunc, addr);

      return handle;
    }();

    auto ctxHandlePtr =
        rewriter.create<mlir::LLVM::AddressOfOp>(loc, ctxHandle);
    auto contextSize = getSizeInBytes(loc, ctxStructType, rewriter);

    const mlir::Value takeCtxArgs[] = {
        ctxHandlePtr,
        contextSize,
        initFuncPtr,
        deinitFuncPtr,
    };
    auto ctxPtr =
        rewriter.create<mlir::LLVM::CallOp>(loc, takeCtxFunc, takeCtxArgs)
            .getResult(0);

    llvm::SmallVector<mlir::Value> takeCtxResults;
    takeCtxResults.emplace_back(ctxPtr);

    auto ctxStructPtr =
        rewriter.create<mlir::LLVM::BitcastOp>(loc, ctxStructPtrType, ctxPtr);
    auto ctxStruct =
        rewriter.create<mlir::LLVM::LoadOp>(loc, ctxStructType, ctxStructPtr);

    for (auto i : llvm::seq(0u, resultsCount)) {
      auto pos = rewriter.getI64ArrayAttr(i);
      auto res = rewriter.create<mlir::LLVM::ExtractValueOp>(
          loc, resultTypes[i], ctxStruct, pos);
      takeCtxResults.emplace_back(res);
    }

    rewriter.replaceOp(op, takeCtxResults);
    return mlir::success();
  }
};

struct LowerReleaseContextOp
    : public mlir::ConvertOpToLLVMPattern<plier::ReleaseContextOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::ReleaseContextOp op,
                  plier::ReleaseContextOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    assert(mod);

    auto unknownLoc = rewriter.getUnknownLoc();
    auto loc = op->getLoc();

    auto lookupFunc = [&](mlir::StringRef name, mlir::Type type) {
      // TODO: fix and use lookupOrCreateFn
      if (auto func = mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name))
        return func;

      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      return rewriter.create<mlir::LLVM::LLVMFuncOp>(
          unknownLoc, name, type, mlir::LLVM::Linkage::External);
    };

    auto releaseCtxFunc = [&]() -> mlir::LLVM::LLVMFuncOp {
      llvm::StringRef name("dpcompReleaseContext");
      auto voidPtr = getVoidPtrType();
      auto funcType = mlir::LLVM::LLVMFunctionType::get(voidPtr, voidPtr);
      return lookupFunc(name, funcType);
    }();

    rewriter.create<mlir::LLVM::CallOp>(loc, releaseCtxFunc, adaptor.context());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// Copypasted from mlir
struct LLVMLoweringPass
    : public mlir::PassWrapper<LLVMLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {

  /// Run the dialect converter on the module.
  void runOnOperation() override {
    using namespace mlir;
    auto &context = getContext();
    auto options = getLLVMOptions(context);
    if (options.useBarePtrCallConv && options.emitCWrappers) {
      getOperation().emitError()
          << "incompatible conversion options: bare-pointer calling convention "
             "and C wrapper emission";
      signalPassFailure();
      return;
    }
    if (failed(LLVM::LLVMDialect::verifyDataLayoutString(
            options.dataLayout.getStringRepresentation(),
            [this](const Twine &message) {
              getOperation().emitError() << message.str();
            }))) {
      signalPassFailure();
      return;
    }

    ModuleOp m = getOperation();

    LLVMTypeConverter typeConverter(&context, options);
    populateToLLVMAdditionalTypeConversion(typeConverter);
    RewritePatternSet patterns(&context);
    populateFuncToLLVMFuncOpConversionPattern(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
    populateLinalgToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);

    patterns.insert<
        // clang-format off
        LowerUndef,
        LowerBuildTuple,
        LowerRetainOp,
        AllocOpLowering,
        DeallocOpLowering,
        ReshapeLowering,
        ExpandShapeLowering,
        LowerExtractMemrefMetadataOp,
        LowerTakeContextOp,
        LowerReleaseContextOp
        // clang-format on
        >(typeConverter);

    LLVMConversionTarget target(context);
    target.addIllegalOp<plier::TakeContextOp, plier::ReleaseContextOp>();
    target.addIllegalDialect<mlir::func::FuncDialect>();

    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();

    m->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
               StringAttr::get(m.getContext(),
                               options.dataLayout.getStringRepresentation()));
  }

private:
};

static void populateLowerToLlvmPipeline(mlir::OpPassManager &pm) {
  pm.addPass(std::make_unique<LowerParallelToCFGPass>());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::arith::createArithmeticExpandOpsPass());
  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<PreLLVMLowering>());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertMathToLLVMPass());
  pm.addPass(mlir::createConvertMathToLibmPass());
  pm.addPass(std::make_unique<LLVMLoweringPass>());
  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(
      std::make_unique<PostLLVMLowering>());
  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
}
} // namespace

void registerLowerToLLVMPipeline(plier::PipelineRegistry &registry) {
  registry.registerPipeline([](auto sink) {
    auto stage = getLowerLoweringStage();
    sink(lowerToLLVMPipelineName(), {stage.begin}, {stage.end}, {},
         &populateLowerToLlvmPipeline);
  });
}

llvm::StringRef lowerToLLVMPipelineName() { return "lower_to_llvm"; }
