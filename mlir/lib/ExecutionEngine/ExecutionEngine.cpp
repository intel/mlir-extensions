// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/ExecutionEngine/ExecutionEngine.hpp"

#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ExecutionEngine/ObjectCache.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/IRTransformLayer.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/SubtargetFeature.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Target/TargetMachine.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>

#define DEBUG_TYPE "imex-execution-engine"

static llvm::OptimizationLevel mapToLevel(llvm::CodeGenOpt::Level level) {
  unsigned optimizeSize = 0; // TODO: unhardcode

  switch (level) {
  default:
    llvm_unreachable("Invalid optimization level!");

  case 0:
    return llvm::OptimizationLevel::O0;

  case 1:
    return llvm::OptimizationLevel::O1;

  case 2:
    switch (optimizeSize) {
    default:
      llvm_unreachable("Invalid optimization level for size!");

    case 0:
      return llvm::OptimizationLevel::O2;

    case 1:
      return llvm::OptimizationLevel::Os;

    case 2:
      return llvm::OptimizationLevel::Oz;
    }

  case 3:
    return llvm::OptimizationLevel::O3;
  }
}

static llvm::PipelineTuningOptions
getPipelineTuningOptions(llvm::CodeGenOpt::Level optLevelVal) {
  llvm::PipelineTuningOptions pto;

  pto.LoopUnrolling = optLevelVal > 0;
  pto.LoopVectorization = optLevelVal > 1;
  pto.SLPVectorization = optLevelVal > 1;
  return pto;
}

static void runOptimizationPasses(llvm::Module &M, llvm::TargetMachine &TM) {
  llvm::CodeGenOpt::Level optLevelVal = TM.getOptLevel();

  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  llvm::PassInstrumentationCallbacks pic;
  llvm::PrintPassOptions ppo;
  ppo.Indent = false;
  ppo.SkipAnalyses = false;
  llvm::StandardInstrumentations si(/*debugLogging*/ false, /*verifyEach*/ true,
                                    ppo);

  si.registerCallbacks(pic, &fam);

  llvm::PassBuilder pb(&TM, getPipelineTuningOptions(optLevelVal));

  llvm::ModulePassManager mpm;

  if (/*verify*/ true) {
    pb.registerPipelineStartEPCallback(
        [&](llvm::ModulePassManager &mpm, llvm::OptimizationLevel level) {
          mpm.addPass(createModuleToFunctionPassAdaptor(llvm::VerifierPass()));
        });
  }

  // Register all the basic analyses with the managers.
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::OptimizationLevel level = mapToLevel(optLevelVal);

  if (optLevelVal == 0) {
    mpm = pb.buildO0DefaultPipeline(level);
  } else {
    mpm = pb.buildPerModuleDefaultPipeline(level);
  }

  mpm.run(M, mam);
}

/// A simple object cache following Lang's LLJITWithObjectCache example.
class imex::ExecutionEngine::SimpleObjectCache : public llvm::ObjectCache {
public:
  void notifyObjectCompiled(const llvm::Module *m,
                            llvm::MemoryBufferRef objBuffer) override {
    cachedObjects[m->getModuleIdentifier()] =
        llvm::MemoryBuffer::getMemBufferCopy(objBuffer.getBuffer(),
                                             objBuffer.getBufferIdentifier());
  }

  std::unique_ptr<llvm::MemoryBuffer>
  getObject(const llvm::Module *m) override {
    auto i = cachedObjects.find(m->getModuleIdentifier());
    if (i == cachedObjects.end()) {
      LLVM_DEBUG(llvm::dbgs() << "No object for " << m->getModuleIdentifier()
                              << " in cache. Compiling.\n");
      return nullptr;
    }
    LLVM_DEBUG(llvm::dbgs() << "Object for " << m->getModuleIdentifier()
                            << " loaded from cache.\n");
    return llvm::MemoryBuffer::getMemBuffer(i->second->getMemBufferRef());
  }

  /// Dump cached object to output file `filename`.
  void dumpToObjectFile(llvm::StringRef outputFilename) {
    // Set up the output file.
    std::string errorMessage;
    auto file = mlir::openOutputFile(outputFilename, &errorMessage);
    if (!file) {
      llvm::errs() << errorMessage << "\n";
      return;
    }

    // Dump the object generated for a single module to the output file.
    assert(cachedObjects.size() == 1 && "Expected only one object entry.");
    auto &cachedObject = cachedObjects.begin()->second;
    file->os() << cachedObject->getBuffer();
    file->keep();
  }

private:
  llvm::StringMap<std::unique_ptr<llvm::MemoryBuffer>> cachedObjects;
};

/// Wrap a string into an llvm::StringError.
static llvm::Error makeStringError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

// Setup LLVM target triple from the current machine.
static void setupModule(llvm::Module &M, llvm::TargetMachine &TM) {
  M.setDataLayout(TM.createDataLayout());
  M.setTargetTriple(TM.getTargetTriple().normalize());
  for (auto &&func : M.functions()) {
    if (!func.hasFnAttribute("target-cpu"))
      func.addFnAttr("target-cpu", TM.getTargetCPU());

    if (!func.hasFnAttribute("target-features")) {
      auto featStr = TM.getTargetFeatureString();
      if (!featStr.empty())
        func.addFnAttr("target-features", featStr);
    }
  }
}

namespace {
class CustomCompiler : public llvm::orc::SimpleCompiler {
public:
  using Transformer = std::function<llvm::Error(llvm::Module &)>;
  using AsmPrinter = std::function<void(llvm::StringRef)>;

  CustomCompiler(Transformer t, AsmPrinter a,
                 std::unique_ptr<llvm::TargetMachine> TM,
                 llvm::ObjectCache *ObjCache = nullptr)
      : SimpleCompiler(*TM, ObjCache), TM(std::move(TM)),
        transformer(std::move(t)), printer(std::move(a)) {}

  llvm::Expected<CompileResult> operator()(llvm::Module &M) override {
    if (transformer) {
      auto err = transformer(M);
      if (err)
        return err;
    }

    setupModule(M, *TM);
    runOptimizationPasses(M, *TM);

    if (printer) {
      llvm::SmallVector<char, 0> buffer;
      llvm::raw_svector_ostream os(buffer);

      llvm::legacy::PassManager PM;
      if (TM->addPassesToEmitFile(PM, os, nullptr,
                                  llvm::CodeGenFileType::CGFT_AssemblyFile))
        return makeStringError("Target does not support Asm emission");

      PM.run(M);
      printer(llvm::StringRef(buffer.data(), buffer.size()));
    }

    return llvm::orc::SimpleCompiler::operator()(M);
  }

private:
  std::shared_ptr<llvm::TargetMachine> TM;
  Transformer transformer;
  AsmPrinter printer;
};
} // namespace

imex::ExecutionEngine::ExecutionEngine(ExecutionEngineOptions options)
    : cache(options.enableObjectCache ? new SimpleObjectCache() : nullptr),
      gdbListener(options.enableGDBNotificationListener
                      ? llvm::JITEventListener::createGDBRegistrationListener()
                      : nullptr),
      perfListener(nullptr) {
  if (options.enablePerfNotificationListener) {
    if (auto *listener = llvm::JITEventListener::createPerfJITEventListener())
      perfListener = listener;
    else if (auto *listener =
                 llvm::JITEventListener::createIntelJITEventListener())
      perfListener = listener;
  }

  // Callback to create the object layer with symbol resolution to current
  // process and dynamically linked libraries.
  auto objectLinkingLayerCreator = [this](llvm::orc::ExecutionSession &session,
                                          const llvm::Triple &targetTriple) {
    auto objectLayer =
        std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(session, []() {
          return std::make_unique<llvm::SectionMemoryManager>();
        });

    // Register JIT event listeners if they are enabled.
    if (gdbListener)
      objectLayer->registerJITEventListener(*gdbListener);
    if (perfListener)
      objectLayer->registerJITEventListener(*perfListener);

    // COFF format binaries (Windows) need special handling to deal with
    // exported symbol visibility.
    // cf llvm/lib/ExecutionEngine/Orc/LLJIT.cpp LLJIT::createObjectLinkingLayer
    if (targetTriple.isOSBinFormatCOFF()) {
      objectLayer->setOverrideObjectFlagsWithResponsibilityFlags(true);
      objectLayer->setAutoClaimResponsibilityForObjectSymbols(true);
    }

    return objectLayer;
  };

  // Callback to inspect the cache and recompile on demand. This follows Lang's
  // LLJITWithObjectCache example.
  auto compileFunctionCreator =
      [this, jitCodeGenOptLevel = options.jitCodeGenOptLevel,
       transformer = options.lateTransformer,
       asmPrinter = options.asmPrinter](llvm::orc::JITTargetMachineBuilder jtmb)
      -> llvm::Expected<
          std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
    if (jitCodeGenOptLevel)
      jtmb.setCodeGenOptLevel(*jitCodeGenOptLevel);
    auto tm = jtmb.createTargetMachine();
    if (!tm)
      return tm.takeError();
    return std::make_unique<CustomCompiler>(transformer, asmPrinter,
                                            std::move(*tm), cache.get());
  };

  auto tmBuilder =
      llvm::cantFail(llvm::orc::JITTargetMachineBuilder::detectHost());

  // Create the LLJIT by calling the LLJITBuilder with 2 callbacks.
  jit = cantFail(llvm::orc::LLJITBuilder()
                     .setCompileFunctionCreator(compileFunctionCreator)
                     .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
                     .setJITTargetMachineBuilder(tmBuilder)
                     .create());

  symbolMap = std::move(options.symbolMap);
  transformer = std::move(options.transformer);
}

imex::ExecutionEngine::~ExecutionEngine() {}

llvm::Expected<imex::ExecutionEngine::ModuleHandle>
imex::ExecutionEngine::loadModule(mlir::ModuleOp m) {
  assert(m);

  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext);
  auto llvmModule = mlir::translateModuleToLLVMIR(m, *ctx);
  if (!llvmModule)
    return makeStringError("could not convert to LLVM IR");

  // Add a ThreadSafemodule to the engine and return.
  llvm::orc::ThreadSafeModule tsm(std::move(llvmModule), std::move(ctx));
  if (transformer)
    cantFail(tsm.withModuleDo(
        [this](llvm::Module &module) { return transformer(module); }));

  llvm::orc::JITDylib *dylib;
  while (true) {
    auto uniqueName =
        (llvm::Twine("module") + llvm::Twine(uniqueNameCounter++)).str();
    if (jit->getJITDylibByName(uniqueName))
      continue;

    auto res = jit->createJITDylib(std::move(uniqueName));
    if (!res)
      return res.takeError();

    dylib = &res.get();
    break;
  }
  assert(dylib);

  auto dataLayout = jit->getDataLayout();
  dylib->addGenerator(
      cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          dataLayout.getGlobalPrefix())));

  if (symbolMap)
    cantFail(
        dylib->define(absoluteSymbols(symbolMap(llvm::orc::MangleAndInterner(
            dylib->getExecutionSession(), jit->getDataLayout())))));

  llvm::cantFail(jit->addIRModule(*dylib, std::move(tsm)));
  llvm::cantFail(jit->initialize(*dylib));
  return static_cast<ModuleHandle>(dylib);
}

void imex::ExecutionEngine::releaseModule(ModuleHandle handle) {
  assert(handle);
  auto dylib = static_cast<llvm::orc::JITDylib *>(handle);
  llvm::cantFail(jit->deinitialize(*dylib));
  dylib->Release();
}

llvm::Expected<void *>
imex::ExecutionEngine::lookup(imex::ExecutionEngine::ModuleHandle handle,
                              llvm::StringRef name) const {
  assert(handle);
  auto dylib = static_cast<llvm::orc::JITDylib *>(handle);
  auto expectedSymbol = jit->lookup(*dylib, name);

  // JIT lookup may return an Error referring to strings stored internally by
  // the JIT. If the Error outlives the ExecutionEngine, it would want have a
  // dangling reference, which is currently caught by an assertion inside JIT
  // thanks to hand-rolled reference counting. Rewrap the error message into a
  // string before returning. Alternatively, ORC JIT should consider copying
  // the string into the error message.
  if (!expectedSymbol) {
    std::string errorMessage;
    llvm::raw_string_ostream os(errorMessage);
    llvm::handleAllErrors(expectedSymbol.takeError(),
                          [&os](llvm::ErrorInfoBase &ei) { ei.log(os); });
    return makeStringError(os.str());
  }

  if (void *fptr = expectedSymbol->toPtr<void *>())
    return fptr;

  return makeStringError("looked up function is null");
}

void imex::ExecutionEngine::dumpToObjectFile(llvm::StringRef filename) {
  if (cache == nullptr) {
    llvm::errs() << "cannot dump ExecutionEngine object code to file: "
                    "object cache is disabled\n";
    return;
  }
  cache->dumpToObjectFile(filename);
}
