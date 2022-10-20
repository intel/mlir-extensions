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

#include "imex/Compiler/PipelineRegistry.hpp"

#include "imex/Utils.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/StringSaver.h>

#include <set>
#include <unordered_map>
#include <utility>

void imex::PipelineRegistry::registerPipeline(
    PipelineRegistry::registry_entry_t func) {
  assert(nullptr != func);
  pipelines.push_back(std::move(func));
}

namespace {
template <typename T, typename IterF, typename VisitF>
static void topoVisit(T &elem, IterF &&iterFunc, VisitF &&func) {
  if (elem.visited)
    return;

  elem.visited = true;
  iterFunc(elem, [&](T &next) {
    topoVisit(next, std::forward<IterF>(iterFunc), std::forward<VisitF>(func));
  });
  func(elem);
}
} // namespace

void imex::PipelineRegistry::populatePassManager(
    populate_pass_manager_t resultSink) const {
  llvm::BumpPtrAllocator allocator;
  llvm::UniqueStringSaver stringSet(allocator);

  using name_id = const void *;
  auto getId = [](llvm::StringRef name) -> name_id {
    assert(!name.empty());
    return name.data();
  };
  std::set<llvm::StringRef>
      pipelinesOrdered; // sorted set to make order consistent

  auto getPipeline = [&](llvm::StringRef name) -> llvm::StringRef {
    if (name.empty())
      reportError("Empty pipeline name");

    auto str = stringSet.save(name);
    pipelinesOrdered.insert(str);
    return str;
  };

  struct PipelineSet : protected llvm::SmallVector<llvm::StringRef, 4> {
    using Base = llvm::SmallVector<llvm::StringRef, 4>;
    using Base::begin;
    using Base::end;
    using Base::value_type;
    void push_back(llvm::StringRef id) {
      auto it = std::equal_range(begin(), end(), id);
      if (it.first == it.second)
        insert(it.first, id);
    }
  };

  struct PipelineInfo {
    llvm::StringRef name;
    PipelineSet prevPipelines;
    PipelineSet nextPipelines;
    pipeline_funt_t func = nullptr;
    PipelineInfo *next = nullptr;
    llvm::ArrayRef<llvm::StringRef> jumps;
    bool visited = false;
    bool iterating = false;
    bool jumpTarget = false;
  };

  std::unordered_map<name_id, PipelineInfo> pipelinesMap;

  auto sink = [&](llvm::StringRef pipelineName,
                  llvm::ArrayRef<llvm::StringRef> prevPipelines,
                  llvm::ArrayRef<llvm::StringRef> nextPipelines,
                  llvm::ArrayRef<llvm::StringRef> jumps, pipeline_funt_t func) {
    assert(!pipelineName.empty());
    assert(nullptr != func);
    auto i = getPipeline(pipelineName);
    auto it = pipelinesMap.insert({getId(i), {}});
    if (!it.second)
      reportError("Duplicated pipeline name");

    auto &info = it.first->second;
    info.name = i;
    info.func = func;
    llvm::transform(prevPipelines, std::back_inserter(info.prevPipelines),
                    getPipeline);
    llvm::transform(nextPipelines, std::back_inserter(info.nextPipelines),
                    getPipeline);
    if (!jumps.empty()) {
      auto data = allocator.Allocate<llvm::StringRef>(jumps.size());
      llvm::transform(jumps, data, [&](llvm::StringRef str) {
        assert(!str.empty());
        return stringSet.save(str);
      });
      info.jumps = {data, jumps.size()};
    }
  };

  for (auto &p : pipelines) {
    assert(nullptr != p);
    p(sink);
  }

  auto getPipelineInfo = [&](llvm::StringRef name) -> PipelineInfo & {
    auto id = getId(name);
    auto it = pipelinesMap.find(id);
    if (it == pipelinesMap.end())
      reportError(llvm::Twine("Pipeline not found") + name);

    return it->second;
  };

  // Make all deps bidirectional
  for (auto name : pipelinesOrdered) {
    auto &info = getPipelineInfo(name);
    for (auto prev : info.prevPipelines) {
      auto &prevInfo = getPipelineInfo(prev);
      prevInfo.nextPipelines.push_back(name);
    }
    for (auto next : info.nextPipelines) {
      auto &nextInfo = getPipelineInfo(next);
      nextInfo.prevPipelines.push_back(name);
    }
  }

  // toposort
  PipelineInfo *firstPipeline = nullptr;
  PipelineInfo *currentPipeline = nullptr;
  for (auto name : pipelinesOrdered) {
    auto iterFunc = [&](PipelineInfo &elem, auto func) {
      elem.iterating = true;
      for (auto it : elem.prevPipelines) {
        auto &info = getPipelineInfo(it);
        if (info.iterating)
          reportError(llvm::Twine("Pipeline depends on itself: ") + elem.name);

        func(info);
      }
      elem.iterating = false;
    };
    auto visitFunc = [&](PipelineInfo &elem) {
      assert(nullptr == elem.next);
      auto current = &elem;
      if (nullptr == firstPipeline) {
        firstPipeline = current;
      } else {
        assert(nullptr != currentPipeline);
        currentPipeline->next = current;
      }
      currentPipeline = current;
    };
    topoVisit(getPipelineInfo(name), iterFunc, visitFunc);
  }

  assert(nullptr != firstPipeline);

  auto iteratePipelines = [&](auto func) {
    for (auto current = firstPipeline; nullptr != current;
         current = current->next)
      func(*current);
  };

  iteratePipelines([&](PipelineInfo &pipeline) {
    if (!pipeline.jumps.empty()) {
      for (auto jump : pipeline.jumps)
        getPipelineInfo(jump).jumpTarget = true;

      if (nullptr != pipeline.next)
        pipeline.next->jumpTarget = true;
    }
  });

  llvm::SmallVector<pipeline_funt_t, 32> funcs;
  llvm::StringRef currentName = firstPipeline->name;
  llvm::ArrayRef<llvm::StringRef> currentJumps;
  resultSink([&](auto addStage) {
    auto flushStages = [&]() {
      if (!funcs.empty()) {
        assert(!currentName.empty());
        auto flusher = [&](mlir::OpPassManager &pm) {
          for (auto f : funcs)
            f(pm);
        };
        addStage(currentName, currentJumps, flusher);
        funcs.clear();
        currentName = {};
        currentJumps = {};
      }
      assert(currentName.empty());
      assert(currentJumps.empty());
    };
    iteratePipelines([&](PipelineInfo &pipeline) {
      if (pipeline.jumpTarget) {
        flushStages();
        currentName = pipeline.name;
      }
      funcs.emplace_back(pipeline.func);
      currentJumps = pipeline.jumps;
    });
    flushStages();
  });
}
