#include "plier/transforms/pipeline_utils.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Attributes.h>

#include "plier/dialect.hpp"

mlir::ArrayAttr plier::get_pipeline_jump_markers(mlir::ModuleOp module)
{
    return module->getAttrOfType<mlir::ArrayAttr>(plier::attributes::getJumpMarkersName());
}

void plier::add_pipeline_jump_marker(mlir::ModuleOp module, mlir::StringAttr name)
{
    assert(name);
    assert(!name.getValue().empty());

    auto jump_markers = plier::attributes::getJumpMarkersName();
    llvm::SmallVector<mlir::Attribute, 16> name_list;
    if (auto old_attr = module->getAttrOfType<mlir::ArrayAttr>(jump_markers))
    {
        name_list.assign(old_attr.begin(), old_attr.end());
    }
    auto it = llvm::lower_bound(name_list, name,
    [](mlir::Attribute lhs, mlir::StringAttr rhs)
    {
        return lhs.cast<mlir::StringAttr>().getValue() < rhs.getValue();
    });
    if (it == name_list.end())
    {
        name_list.emplace_back(name);
    }
    else if (*it != name)
    {
        name_list.insert(it, name);
    }
    module->setAttr(jump_markers, mlir::ArrayAttr::get(module.getContext(), name_list));
}


void plier::remove_pipeline_jump_marker(mlir::ModuleOp module, mlir::StringAttr name)
{
    assert(name);
    assert(!name.getValue().empty());

    auto jump_markers = plier::attributes::getJumpMarkersName();
    llvm::SmallVector<mlir::Attribute, 16> name_list;
    if (auto old_attr = module->getAttrOfType<mlir::ArrayAttr>(jump_markers))
    {
        name_list.assign(old_attr.begin(), old_attr.end());
    }
    auto it = llvm::lower_bound(name_list, name,
    [](mlir::Attribute lhs, mlir::StringAttr rhs)
    {
        return lhs.cast<mlir::StringAttr>().getValue() < rhs.getValue();
    });
    assert(it != name_list.end());
    name_list.erase(it);
    module->setAttr(jump_markers, mlir::ArrayAttr::get(module.getContext(), name_list));
}
