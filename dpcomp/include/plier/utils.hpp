#pragma once

#include <utility>

#include <llvm/ADT/ScopeExit.h>

namespace llvm
{
class Twine;
}

namespace plier
{
[[noreturn]] void report_error(const llvm::Twine& msg);

template<typename T, typename H, typename F>
void scoped_diag_handler(T& ctx, H&& diag_handler, F&& func)
{
    auto& diag_engine = ctx.getDiagEngine();
    auto diag_id = diag_engine.registerHandler(std::forward<H>(diag_handler));
    auto diag_guard = llvm::make_scope_exit([&]()
    {
        diag_engine.eraseHandler(diag_id);
    });
    func();
}
}
