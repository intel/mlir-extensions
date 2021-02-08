#include "plier/compiler/pipeline_registry.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/StringSaver.h>

#include "plier/utils.hpp"

#include <set>
#include <unordered_map>
#include <utility>

void plier::PipelineRegistry::register_pipeline(PipelineRegistry::registry_entry_t func)
{
    assert(nullptr != func);
    pipelines.push_back(std::move(func));
}

namespace
{
template<typename T, typename IterF, typename VisitF>
void topo_visit(T& elem, IterF&& iter_func, VisitF&& func)
{
    if (elem.visited)
    {
        return;
    }
    elem.visited = true;
    iter_func(elem, [&](T& next)
    {
        topo_visit(next, std::forward<IterF>(iter_func), std::forward<VisitF>(func));
    });
    func(elem);
}
}

void plier::PipelineRegistry::populate_pass_manager(populate_pass_manager_t result_sink) const
{
    llvm::BumpPtrAllocator allocator;
    llvm::UniqueStringSaver string_set(allocator);

    using name_id = const void*;
    auto get_id = [](llvm::StringRef name)->name_id
    {
        assert(!name.empty());
        return name.data();
    };
    std::set<llvm::StringRef> pipelines_ordered; // sorted set to make order consistent

    auto get_pipeline = [&](llvm::StringRef name)->llvm::StringRef
    {
        if (name.empty())
        {
            report_error("Empty pipeline name");
        }
        auto str = string_set.save(name);
        pipelines_ordered.insert(str);
        return str;
    };

    struct PipelineSet : protected llvm::SmallVector<llvm::StringRef, 4>
    {
        using Base = llvm::SmallVector<llvm::StringRef, 4>;
        using Base::begin;
        using Base::end;
        using Base::value_type;
        void push_back(llvm::StringRef id)
        {
            auto it = std::equal_range(begin(), end(), id);
            if (it.first == it.second)
            {
                insert(it.first, id);
            }
        }
    };

    struct PipelineInfo
    {
        llvm::StringRef name;
        PipelineSet prev_pipelines;
        PipelineSet next_pipelines;
        pipeline_funt_t func = nullptr;
        PipelineInfo* next = nullptr;
        llvm::ArrayRef<llvm::StringRef> jumps;
        bool visited = false;
        bool iterating = false;
        bool jump_target = false;
    };

    std::unordered_map<name_id, PipelineInfo> pipelines_map;

    auto sink = [&](llvm::StringRef pipeline_name,
                    llvm::ArrayRef<llvm::StringRef> prev_pipelines,
                    llvm::ArrayRef<llvm::StringRef> next_pipelines,
                    llvm::ArrayRef<llvm::StringRef> jumps,
                    pipeline_funt_t func)
    {
        assert(!pipeline_name.empty());
        assert(nullptr != func);
        auto i = get_pipeline(pipeline_name);
        auto it = pipelines_map.insert({get_id(i), {}});
        if (!it.second)
        {
            report_error("Duplicated pipeline name");
        }
        auto& info = it.first->second;
        info.name = i;
        info.func = func;
        llvm::transform(prev_pipelines, std::back_inserter(info.prev_pipelines), get_pipeline);
        llvm::transform(next_pipelines, std::back_inserter(info.next_pipelines), get_pipeline);
        if (!jumps.empty())
        {
            auto data = allocator.Allocate<llvm::StringRef>(jumps.size());
            llvm::transform(jumps, data, [&](llvm::StringRef str)
            {
                assert(!str.empty());
                return string_set.save(str);
            });
            info.jumps = { data, jumps.size() };
        }
    };

    for (auto& p : pipelines)
    {
        assert(nullptr != p);
        p(sink);
    }

    auto get_pipeline_info = [&](llvm::StringRef name)->PipelineInfo&
    {
        auto id = get_id(name);
        auto it = pipelines_map.find(id);
        if (it == pipelines_map.end())
        {
            report_error(llvm::Twine("Pipeline not found") + name);
        }
        return it->second;
    };

    // Make all deps bidirectional
    for (auto name : pipelines_ordered)
    {
        auto& info = get_pipeline_info(name);
        for (auto prev : info.prev_pipelines)
        {
            auto& prev_info = get_pipeline_info(prev);
            prev_info.next_pipelines.push_back(name);
        }
        for (auto next : info.next_pipelines)
        {
            auto& next_info = get_pipeline_info(next);
            next_info.prev_pipelines.push_back(name);
        }
    }

    // toposort
    PipelineInfo* first_pipeline = nullptr;
    PipelineInfo* current_pipeline = nullptr;
    for (auto name : pipelines_ordered)
    {
        auto iter_func = [&](PipelineInfo& elem, auto func)
        {
            elem.iterating = true;
            for (auto it : elem.prev_pipelines)
            {
                auto& info = get_pipeline_info(it);
                if (info.iterating)
                {
                    report_error(llvm::Twine("Pipeline depends on itself: ") + elem.name);
                }
                func(info);
            }
            elem.iterating = false;
        };
        auto visit_func = [&](PipelineInfo& elem)
        {
            assert(nullptr == elem.next);
            auto current = &elem;
            if (nullptr == first_pipeline)
            {
                first_pipeline = current;
            }
            else
            {
                assert(nullptr != current_pipeline);
                current_pipeline->next = current;
            }
            current_pipeline = current;
        };
        topo_visit(get_pipeline_info(name), iter_func, visit_func);
    }

    assert(nullptr != first_pipeline);

    auto iterate_pipelines = [&](auto func)
    {
        for (auto current = first_pipeline; nullptr != current;
             current = current->next)
        {
            func(*current);
        }
    };

    iterate_pipelines([&](PipelineInfo& pipeline)
    {
        if (!pipeline.jumps.empty())
        {
            for (auto jump : pipeline.jumps)
            {
                get_pipeline_info(jump).jump_target = true;
            }
            if (nullptr != pipeline.next)
            {
                pipeline.next->jump_target = true;
            }
        }
    });

    llvm::SmallVector<pipeline_funt_t, 32> funcs;
    llvm::StringRef current_name = first_pipeline->name;
    llvm::ArrayRef<llvm::StringRef> current_jumps;
    result_sink([&](auto add_stage)
    {
        auto flush_stages = [&]()
        {
            if (!funcs.empty())
            {
                assert(!current_name.empty());
                auto flusher = [&](mlir::OpPassManager& pm)
                {
                    for (auto f : funcs)
                    {
                        f(pm);
                    }
                };
                add_stage(current_name, current_jumps, flusher);
                funcs.clear();
                current_name = {};
                current_jumps = {};
            }
            assert(current_name.empty());
            assert(current_jumps.empty());
        };
        iterate_pipelines([&](PipelineInfo& pipeline)
        {
            if (pipeline.jump_target)
            {
                flush_stages();
                current_name = pipeline.name;
            }
            funcs.emplace_back(pipeline.func);
            current_jumps = pipeline.jumps;
        });
        flush_stages();
    });
}
