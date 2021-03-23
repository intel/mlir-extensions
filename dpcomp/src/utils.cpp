#include "plier/utils.hpp"

#include <stdexcept>

#include "llvm/ADT/Twine.h"

void plier::report_error(const llvm::Twine& msg)
{
    throw std::runtime_error(msg.str());
}
