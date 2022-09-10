// Copyright 2022 Intel Corporation
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

#include "mlir-extensions/ExecutionEngine/execution_engine.hpp"

#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ExecutionEngine/ObjectCache.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/IRTransformLayer.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
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

#define DEBUG_TYPE "execution-engine"

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
static bool setupTargetTriple(llvm::Module *llvmModule) {
  // Setup the machine properties from the current architecture.
  auto targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string errorMessage;
  const auto *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if (!target) {
    llvm::errs() << "NO target: " << errorMessage << "\n";
    return true;
  }

  std::string cpu(llvm::sys::getHostCPUName());
  llvm::SubtargetFeatures features;
  llvm::StringMap<bool> hostFeatures;

  if (llvm::sys::getHostCPUFeatures(hostFeatures))
    for (auto &f : hostFeatures)
      features.AddFeature(f.first(), f.second);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetTriple, cpu, features.getString(), {}, {}));
  if (!machine) {
    llvm::errs() << "Unable to create target machine\n";
    return true;
  }
  llvmModule->setDataLayout(machine->createDataLayout());
  llvmModule->setTargetTriple(targetTriple);
  return false;
}

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

    // Resolve symbols from shared libraries.
    //    for (auto libPath : options.sharedLibPaths) {
    //      auto mb = llvm::MemoryBuffer::getFile(libPath);
    //      if (!mb) {
    //        llvm::errs() << "Failed to create MemoryBuffer for: " << libPath
    //               << "\nError: " << mb.getError().message() << "\n";
    //        continue;
    //      }
    //      auto &jd = session.createBareJITDylib(std::string(libPath));
    //      auto loaded = DynamicLibrarySearchGenerator::Load(
    //          libPath.data(), dataLayout.getGlobalPrefix());
    //      if (!loaded) {
    //        llvm::errs() << "Could not load " << libPath << ":\n  " <<
    //        loaded.takeError()
    //               << "\n";
    //        continue;
    //      }
    //      jd.addGenerator(std::move(*loaded));
    //      cantFail(objectLayer->add(jd, std::move(mb.get())));
    //    }

    return objectLayer;
  };

  // Callback to inspect the cache and recompile on demand. This follows Lang's
  // LLJITWithObjectCache example.
  auto compileFunctionCreator = [this, jitCodeGenOptLevel =
                                           options.jitCodeGenOptLevel](
                                    llvm::orc::JITTargetMachineBuilder jtmb)
      -> llvm::Expected<
          std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
    if (jitCodeGenOptLevel)
      jtmb.setCodeGenOptLevel(*jitCodeGenOptLevel);
    auto tm = jtmb.createTargetMachine();
    if (!tm)
      return tm.takeError();
    return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(std::move(*tm),
                                                               cache.get());
  };

  // Create the LLJIT by calling the LLJITBuilder with 2 callbacks.
  jit = cantFail(llvm::orc::LLJITBuilder()
                     .setCompileFunctionCreator(compileFunctionCreator)
                     .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
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

  if (setupTargetTriple(llvmModule.get()))
    return makeStringError("Failed to setup module targetTriple");

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
