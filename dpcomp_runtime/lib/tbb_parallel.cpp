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
static std::mutex &getDebugMutext() {
  static std::mutex mut;
  return mut;
}

struct TBBContext {
  TBBContext(int numThreads)
      : numThreads(numThreads),
        scheduler_handle(tbb::task_scheduler_handle::get()), arena(numThreads) {

  }

  ~TBBContext() {
    arena.terminate();
    (void)tbb::finalize(scheduler_handle, std::nothrow);
  }

  int numThreads;
  tbb::task_scheduler_handle scheduler_handle;
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

using parallel_for_fptr = void (*)(const Range *, size_t, void *);

static void parallel_for_nested(const InputRange *input_ranges, size_t depth,
                                size_t num_threads, size_t num_loops,
                                Dim *prev_dim, parallel_for_fptr func,
                                void *ctx);

template <unsigned N, bool Term, size_t... Is>
static void run_parallel_for(const InputRange *input_ranges, size_t depth,
                             size_t num_threads, size_t num_loops,
                             Dim *prev_dim, parallel_for_fptr func, void *ctx) {
  std::array<InputRange, N> tempRanges;
  std::copy_n(input_ranges + depth, N, tempRanges.begin());

  if (DEBUG) {
    std::lock_guard<std::mutex> lock(getDebugMutext());
    fprintf(stderr, "parallel_for_nested: depth=%d", static_cast<int>(depth));
    for (unsigned i = 0; i < N; ++i) {
      auto &input = tempRanges[i];
      auto lower_bound = input.lower;
      auto upper_bound = input.upper;
      auto step = input.step;
      fprintf(stderr, " (lower_bound=%d, upper_bound=%d, step=%d)",
              static_cast<int>(lower_bound), static_cast<int>(upper_bound),
              static_cast<int>(step));
    }
    fprintf(stderr, "\n");
  }

  auto getRange = [&](size_t i) {
    auto &input = tempRanges[i];
    auto lower_bound = input.lower;
    auto upper_bound = input.upper;
    auto step = input.step;
    size_t count = (upper_bound - lower_bound + step - 1) / step;
    size_t grain =
        std::max(size_t(1), std::min(count / num_threads / 2, size_t(64)));
    return tbb::blocked_range<size_t>(0, count, grain);
  };

  tbb::blocked_rangeNd<size_t, N> range(getRange(Is)...);

  auto runFunc = [&](Dim *current) {
    auto thread_index =
        static_cast<size_t>(tbb::this_task_arena::current_thread_index());
    assert(thread_index >= 0);
    std::array<Range, 8> static_ranges;
    std::unique_ptr<Range[]> dyn_ranges;
    auto *range_ptr = [&]() -> Range * {
      if (num_loops <= static_ranges.size())
        return static_ranges.data();

      dyn_ranges.reset(new Range[num_loops]);
      return dyn_ranges.get();
    }();

    for (size_t i = 0; i < num_loops; ++i) {
      assert(current);
      range_ptr[num_loops - i - 1] = current->val;
      current = current->prev;
    }

    if (DEBUG) {
      std::lock_guard<std::mutex> lock(getDebugMutext());
      fprintf(stderr, "parallel_for func: thread_index=%d",
              static_cast<int>(thread_index));
      for (size_t i = 0; i < num_loops; ++i) {
        auto &input = range_ptr[i];
        auto lower_bound = input.lower;
        auto upper_bound = input.upper;
        fprintf(stderr, " (lower_bound=%d, upper_bound=%d)",
                static_cast<int>(lower_bound), static_cast<int>(upper_bound));
      }
      fprintf(stderr, "\n");
    }
    func(range_ptr, thread_index, ctx);
  };

  auto loopBody = [&](const tbb::blocked_rangeNd<size_t, N> &r) {
    std::array<Dim, N> dims;
    auto prev = prev_dim;
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
      parallel_for_nested(input_ranges, next, num_threads, num_loops, prev,
                          func, ctx);
    }
  };

  tbb::parallel_for(range, loopBody, tbb::auto_partitioner());
}

static void parallel_for_nested(const InputRange *input_ranges, size_t depth,
                                size_t num_threads, size_t num_loops,
                                Dim *prev_dim, parallel_for_fptr func,
                                void *ctx) {
  assert(num_loops > depth);
  auto rem = num_loops - depth;
  if (rem == 1) {
    run_parallel_for<1, true, 0>(input_ranges, depth, num_threads, num_loops,
                                 prev_dim, func, ctx);
  } else if (rem == 2) {
    run_parallel_for<2, true, 0, 1>(input_ranges, depth, num_threads, num_loops,
                                    prev_dim, func, ctx);
  } else if (rem == 3) {
    run_parallel_for<3, true, 0, 1, 2>(input_ranges, depth, num_threads,
                                       num_loops, prev_dim, func, ctx);
  } else {
    run_parallel_for<3, false, 0, 1, 2>(input_ranges, depth, num_threads,
                                        num_loops, prev_dim, func, ctx);
  }
}
} // namespace

extern "C" {
DPCOMP_RUNTIME_EXPORT void dpcompParallelFor(const InputRange *input_ranges,
                                             size_t num_loops,
                                             parallel_for_fptr func,
                                             void *ctx) {
  auto &context = getContext();
  auto num_threads = static_cast<size_t>(context.numThreads);
  if (DEBUG) {
    std::lock_guard<std::mutex> lock(getDebugMutext());
    fprintf(stderr, "parallel_for num_loops=%d: ", static_cast<int>(num_loops));
    for (size_t i = 0; i < num_loops; ++i) {
      auto r = input_ranges[i];
      fprintf(stderr, "(%d, %d, %d) ", static_cast<int>(r.lower),
              static_cast<int>(r.upper), static_cast<int>(r.step));
    }
    fprintf(stderr, "\n");
  }

  context.arena.execute([&] {
    parallel_for_nested(input_ranges, 0, num_threads, num_loops, nullptr, func,
                        ctx);
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
