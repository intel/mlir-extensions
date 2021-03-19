#pragma once

#include <iterator>

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
    enum class NodeType
    {
        Root,
        Def,
        Use,
        Phi,
        Term
    };
    struct Node;

    MemorySSA() = default;
    MemorySSA(const MemorySSA&) = delete;
    MemorySSA(MemorySSA&&) = default;

    Node* createDef(mlir::Operation* op, Node* arg);
    Node* createUse(mlir::Operation* op, Node* arg);
    Node* createPhi(mlir::Operation* op, llvm::ArrayRef<Node*> args);

    void eraseNode(Node* node);
    NodeType getNodeType(Node* node) const;
    mlir::Operation* getNodeOperation(Node* node) const;
    Node* getNodeDef(Node* node) const;

    Node* getRoot();
    Node* getTerm();
    Node* getNode(mlir::Operation* op) const;

    struct NodesIterator
    {
        using iterator_category = std::bidirectional_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = Node;
        using pointer = value_type*;
        using reference = value_type&;

        using internal_iterator = llvm::simple_ilist<Node>::iterator;

        NodesIterator(internal_iterator iter);
        NodesIterator(const NodesIterator&) = default;
        NodesIterator(NodesIterator&&) = default;

        NodesIterator& operator=(const NodesIterator&) = default;
        NodesIterator& operator=(NodesIterator&&) = default;

        bool operator==(const NodesIterator& rhs) const { return iterator == rhs.iterator; }
        bool operator!=(const NodesIterator& rhs) const { return iterator != rhs.iterator; }

        NodesIterator& operator++();
        NodesIterator operator++(int);

        NodesIterator& operator--();
        NodesIterator operator--(int);

        reference operator*();
        pointer operator->();

    private:
        internal_iterator iterator;
    };

    llvm::iterator_range<NodesIterator> getNodes();

    void print(llvm::raw_ostream& os);
    void print(Node* node, llvm::raw_ostream& os);

    mlir::LogicalResult optimizeUses(llvm::function_ref<bool(mlir::Operation*, mlir::Operation*)> mayAlias);

private:
    Node* root = nullptr;
    Node* term = nullptr;
    llvm::DenseMap<mlir::Operation*, Node*> nodesMap;
    llvm::BumpPtrAllocator allocator;
    llvm::simple_ilist<Node> nodes;

    Node* createNode(mlir::Operation* op, NodeType type, llvm::ArrayRef<Node*> args);
};

llvm::Optional<plier::MemorySSA> buildMemorySSA(mlir::Region& region);
}
