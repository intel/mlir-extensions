
#include <array>
#include <cstdio>

#define TBB_PREVIEW_WAITING_FOR_WORKERS 1

#include <tbb/task_arena.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

#include "dpcomp-runtime_export.h"

#define DEBUG 0

namespace
{
struct TBBContext
{
    TBBContext(int numThreads):
        numThreads(numThreads),
        scheduler_handle(tbb::task_scheduler_handle::get()),
        arena(numThreads)
    {

    }

    ~TBBContext()
    {
        (void)tbb::finalize(scheduler_handle, std::nothrow);
    }

    int numThreads;
    tbb::task_scheduler_handle scheduler_handle;
    tbb::task_arena arena;
};

std::unique_ptr<TBBContext> globalContext;

TBBContext& getContext()
{
    if (globalContext == nullptr)
    {
        fprintf(stderr, "dpcomp: tbb runtime is not initialized\n");
        fflush(stderr);
        abort();
    }
    return *globalContext;
}

struct InputRange
{
    size_t lower;
    size_t upper;
    size_t step;
};

struct Range
{
    size_t lower;
    size_t upper;
};

struct Dim
{
    Range val;
    Dim* prev;
};

using parallel_for_fptr = void(*)(const Range*, size_t, void*);

static void parallel_for_nested(const InputRange* input_ranges, size_t depth, size_t num_threads, size_t num_loops, Dim* prev_dim, parallel_for_fptr func, void* ctx)
{
    auto input = input_ranges[depth];
    auto lower_bound = input.lower;
    auto upper_bound = input.upper;
    auto step = input.step;

    if(DEBUG)
    {
        printf("parallel_for_nested: lower_bound=%d, upper_bound=%d, step=%d, depth=%d\n",
               static_cast<int>(lower_bound),
               static_cast<int>(upper_bound),
               static_cast<int>(step),
               static_cast<int>(depth));
    }

    size_t count = (upper_bound - lower_bound + step - 1) / step;
    size_t grain = std::max(size_t(1), std::min(count / num_threads / 2, size_t(64)));
    tbb::parallel_for(tbb::blocked_range<size_t>(0, count, grain),
        [&](const tbb::blocked_range<size_t>& r)
        {
            auto begin = lower_bound + r.begin() * step;
            auto end = lower_bound + r.end() * step;
            if(DEBUG)
            {
                printf("parallel_for_nested body: begin=%d, end=%d, depth=%d\n\n",
                       static_cast<int>(begin),
                       static_cast<int>(end),
                       static_cast<int>(depth));
            }
            auto next = depth + 1;
            Dim dim{Range{begin, end}, prev_dim};
            if (next == num_loops)
            {
                auto thread_index = static_cast<size_t>(tbb::this_task_arena::current_thread_index());
                std::array<Range, 8> static_ranges;
                std::unique_ptr<Range[]> dyn_ranges;
                auto* range_ptr = [&]()->Range*
                {
                    if (num_loops <= static_ranges.size())
                    {
                        return static_ranges.data();
                    }
                    dyn_ranges.reset(new Range[num_loops]);
                    return dyn_ranges.get();
                }();

                Dim* current = &dim;
                for (size_t i = 0; i < num_loops; ++i)
                {
                    range_ptr[num_loops - i - 1] = current->val;
                    current = current->prev;
                }
                func(range_ptr, thread_index, ctx);
            }
            else
            {
                parallel_for_nested(input_ranges, next, num_threads, num_loops, &dim, func, ctx);
            }
        }, tbb::auto_partitioner());
}
}

extern "C"
{
DPCOMP_RUNTIME_EXPORT void dpcomp_parallel_for(const InputRange* input_ranges, size_t num_loops, parallel_for_fptr func, void* ctx)
{
    auto& context = getContext();
    auto num_threads = static_cast<size_t>(context.numThreads);
    if(DEBUG)
    {
        printf("parallel_for num_loops=%d: ", static_cast<int>(num_loops));
        for (size_t i = 0; i < num_loops; ++i)
        {
            auto r = input_ranges[i];
            printf("(%d, %d, %d) ",
                   static_cast<int>(r.lower),
                   static_cast<int>(r.upper),
                   static_cast<int>(r.step));
        }
        puts("\n");
    }

    context.arena.execute([&]
    {
        parallel_for_nested(input_ranges, 0, num_threads, num_loops, nullptr, func, ctx);
    });
}

DPCOMP_RUNTIME_EXPORT void dpcomp_parallel_init(int numThreads)
{
    if(DEBUG)
    {
        printf("dpcomp_parallel_init %d\n", numThreads);
    }
    if (nullptr == globalContext)
    {
        globalContext = std::make_unique<TBBContext>(numThreads);
    }
}

DPCOMP_RUNTIME_EXPORT void dpcomp_parallel_finalize()
{
    if(DEBUG)
    {
        puts("dpcomp_parallel_finalize\n");
    }
    globalContext.reset();
}
}
