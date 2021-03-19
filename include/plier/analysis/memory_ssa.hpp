#pragma once

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/simple_ilist.h>
#include <llvm/Support/Allocator.h>

namespace mlir
{
struct LogicalResult;
class Operation;
class Region;
}

namespace plier
{

class MemorySSA
{
public:
    struct Node;

    MemorySSA() = default;
    MemorySSA(const MemorySSA&) = delete;
    MemorySSA(MemorySSA&&) = default;

    Node* createDef(mlir::Operation* op, Node* arg);
    Node* createUse(mlir::Operation* op, Node* arg);
    Node* createPhi(mlir::Operation* op, llvm::ArrayRef<Node*> args);

    void eraseNode(Node* node);

    Node* getRoot();
    Node* getTerm();
    Node* getNode(mlir::Operation* op) const;

    auto& getNodes()
    {
        return nodes;
    }

    void print(llvm::raw_ostream& os);
    void print(Node* node, llvm::raw_ostream& os);

    mlir::LogicalResult optimizeUses(llvm::function_ref<bool(mlir::Operation*, mlir::Operation*)> mayAlias);

private:
    Node* root = nullptr;
    Node* term = nullptr;
    llvm::DenseMap<mlir::Operation*, Node*> nodesMap;
    llvm::BumpPtrAllocator allocator;
    llvm::simple_ilist<Node> nodes;

    enum class NodeType
    {
        Root,
        Def,
        Use,
        Phi,
        Term
    };
    Node* createNode(mlir::Operation* op, NodeType type, llvm::ArrayRef<Node*> args);
};

llvm::Optional<plier::MemorySSA> buildMemorySSA(mlir::Region& region);
}
