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

#ifdef IMEX_ENABLE_TBB_SUPPORT

#include <array>
#include <cassert>
#include <cstdio>
#include <mutex>

#define TBB_PREVIEW_WAITING_FOR_WORKERS 1
#define TBB_PREVIEW_BLOCKED_RANGE_ND 1

#include <tbb/blocked_rangeNd.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include "dpcomp-runtime_export.h"

#define DEBUG 0

namespace {
static std::mutex &getDebugMutex() {
  static std::mutex mut;
  return mut;
}

static tbb::task_scheduler_handle tbbTshAttach() {
#if TBB_INTERFACE_VERSION >= 12060
  return tbb::attach();
#else
  return tbb::task_scheduler_handle::get();
#endif
}

static void tbbTshRelease(tbb::task_scheduler_handle &tsh) {
#if TBB_INTERFACE_VERSION >= 12060
  tsh.release();
#else
  tbb::task_scheduler_handle::release(tsh);
#endif
}

struct TBBContext {
  TBBContext(int numThreads)
      : numThreads(numThreads), schedulerHandle(tbbTshAttach()),
        arena(numThreads) {}

  ~TBBContext() {
    arena.terminate();
    if (!tbb::finalize(schedulerHandle, std::nothrow)) {
      if (DEBUG) {
        fprintf(stderr, "dpcomp: failed to finalize tbb runtime\n");
        fflush(stderr);
      }
      tbbTshRelease(schedulerHandle);
    }
  }

  int numThreads;
  tbb::task_scheduler_handle schedulerHandle;
  tbb::task_arena arena;
};

std::unique_ptr<TBBContext> globalContext;

TBBContext &getContext() {
  if (globalContext == nullptr) {
    fprintf(stderr, "dpcomp: tbb runtime is not initialized\n");
    fflush(stderr);
    abort();
  }
  return *globalContext;
}

struct InputRange {
  size_t lower;
  size_t upper;
  size_t step;
};

struct Range {
  size_t lower;
  size_t upper;
};

struct Dim {
  Range val;
  Dim *prev;
};

using ParallelForFptr = void (*)(const Range *, size_t, void *);

static void parallelForNested(const InputRange *inputRanges, size_t depth,
                              size_t numThreads, size_t numLoops, Dim *prevDim,
                              ParallelForFptr func, void *ctx);

template <unsigned N, bool Term, size_t... Is>
static void runParallelFor(const InputRange *inputRanges, size_t depth,
                           size_t numThreads, size_t numLoops, Dim *prevDim,
                           ParallelForFptr func, void *ctx) {
  std::array<InputRange, N> tempRanges;
  std::copy_n(inputRanges + depth, N, tempRanges.begin());

  if (DEBUG) {
    std::lock_guard<std::mutex> lock(getDebugMutex());
    fprintf(stderr, "parallel_for_nested: depth=%d", static_cast<int>(depth));
    for (unsigned i = 0; i < N; ++i) {
      auto &input = tempRanges[i];
      auto lowerBound = input.lower;
      auto upperBound = input.upper;
      auto step = input.step;
      fprintf(stderr, " (lower_bound=%d, upper_bound=%d, step=%d)",
              static_cast<int>(lowerBound), static_cast<int>(upperBound),
              static_cast<int>(step));
    }
    fprintf(stderr, "\n");
  }

  auto getRange = [&](size_t i) {
    auto &input = tempRanges[i];
    auto lowerBound = input.lower;
    auto upperBound = input.upper;
    auto step = input.step;
    size_t count = (upperBound - lowerBound + step - 1) / step;
    size_t grain =
        std::max(size_t(1), std::min(count / numThreads / 2, size_t(64)));
    return tbb::blocked_range<size_t>(0, count, grain);
  };

  tbb::blocked_rangeNd<size_t, N> range(getRange(Is)...);

  auto runFunc = [&](Dim *current) {
    auto threadIndex =
        static_cast<size_t>(tbb::this_task_arena::current_thread_index());
    assert(threadIndex >= 0);
    std::array<Range, 8> staticRanges;
    std::unique_ptr<Range[]> dynRanges;
    auto *rangePtr = [&]() -> Range * {
      if (numLoops <= staticRanges.size())
        return staticRanges.data();

      dynRanges.reset(new Range[numLoops]);
      return dynRanges.get();
    }();

    for (size_t i = 0; i < numLoops; ++i) {
      assert(current);
      rangePtr[numLoops - i - 1] = current->val;
      current = current->prev;
    }

    if (DEBUG) {
      std::lock_guard<std::mutex> lock(getDebugMutex());
      fprintf(stderr, "parallel_for func: thread_index=%d",
              static_cast<int>(threadIndex));
      for (size_t i = 0; i < numLoops; ++i) {
        auto &input = rangePtr[i];
        auto lowerBound = input.lower;
        auto upperBound = input.upper;
        fprintf(stderr, " (lower_bound=%d, upper_bound=%d)",
                static_cast<int>(lowerBound), static_cast<int>(upperBound));
      }
      fprintf(stderr, "\n");
    }
    func(rangePtr, threadIndex, ctx);
  };

  auto loopBody = [&](const tbb::blocked_rangeNd<size_t, N> &r) {
    std::array<Dim, N> dims;
    auto prev = prevDim;
    for (unsigned i = 0; i < N; ++i) {
      auto &input = tempRanges[i];
      auto lower_bound = input.lower;
      auto step = input.step;
      auto rDim = r.dim(i);
      auto begin = lower_bound + rDim.begin() * step;
      auto end = lower_bound + rDim.end() * step;
      dims[i] = Dim{Range{begin, end}, prev};
      prev = &dims[i];
    }

    if (Term) {
      runFunc(prev);
    } else {
      auto next = depth + N;
      parallelForNested(inputRanges, next, numThreads, numLoops, prev, func,
                        ctx);
    }
  };

  tbb::parallel_for(range, loopBody, tbb::auto_partitioner());
}

static void parallelForNested(const InputRange *inputRanges, size_t depth,
                              size_t numThreads, size_t numLoops, Dim *prevDim,
                              ParallelForFptr func, void *ctx) {
  assert(numLoops > depth);
  auto rem = numLoops - depth;
  if (rem == 1) {
    runParallelFor<1, true, 0>(inputRanges, depth, numThreads, numLoops,
                               prevDim, func, ctx);
  } else if (rem == 2) {
    runParallelFor<2, true, 0, 1>(inputRanges, depth, numThreads, numLoops,
                                  prevDim, func, ctx);
  } else if (rem == 3) {
    runParallelFor<3, true, 0, 1, 2>(inputRanges, depth, numThreads, numLoops,
                                     prevDim, func, ctx);
  } else {
    runParallelFor<3, false, 0, 1, 2>(inputRanges, depth, numThreads, numLoops,
                                      prevDim, func, ctx);
  }
}
} // namespace

extern "C" {
DPCOMP_RUNTIME_EXPORT void dpcompParallelFor(const InputRange *inputRanges,
                                             size_t numLoops,
                                             ParallelForFptr func, void *ctx) {
  auto &context = getContext();
  auto numThreads = static_cast<size_t>(context.numThreads);
  if (DEBUG) {
    std::lock_guard<std::mutex> lock(getDebugMutex());
    fprintf(stderr, "parallel_for num_loops=%d: ", static_cast<int>(numLoops));
    for (size_t i = 0; i < numLoops; ++i) {
      auto r = inputRanges[i];
      fprintf(stderr, "(%d, %d, %d) ", static_cast<int>(r.lower),
              static_cast<int>(r.upper), static_cast<int>(r.step));
    }
    fprintf(stderr, "\n");
  }

  context.arena.execute([&] {
    parallelForNested(inputRanges, 0, numThreads, numLoops, nullptr, func, ctx);
  });
}

DPCOMP_RUNTIME_EXPORT void dpcompParallelInit(int numThreads) {
  if (DEBUG)
    fprintf(stderr, "dpcomp_parallel_init %d\n", numThreads);

  if (nullptr == globalContext)
    globalContext = std::make_unique<TBBContext>(numThreads);
}

DPCOMP_RUNTIME_EXPORT void dpcompParallelFinalize() {
  if (DEBUG)
    fprintf(stderr, "dpcomp_parallel_finalize\n");

  globalContext.reset();
}
}
#endif // IMEX_ENABLE_TBB_SUPPORT
