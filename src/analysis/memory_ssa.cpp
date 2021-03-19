#include "plier/analysis/memory_ssa.hpp"

#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

struct plier::MemorySSA::Node : public llvm::ilist_node<Node>
{
    using Type = plier::MemorySSA::NodeType;

    mlir::Operation* getOperation() const
    {
        return operation;
    }

    Type getType() const
    {
        return type;
    }

    unsigned getNumArguments() const
    {
        return argCount;
    }

    auto getArguments()
    {
        return llvm::map_range(llvm::makeArrayRef(&args[0], argCount), [](auto& a)->Node*
        {
            return a.arg;
        });
    }

    auto getUsers()
    {
        return llvm::map_range(users, [](auto& a)->Node*
        {
            return a.getParent();
        });
    }

    auto getUses()
    {
        struct Use
        {
            Node* user;
            unsigned index;
        };

        return llvm::map_range(users, [](auto& a)->Use
        {
            return {a.getParent(), a.index};
        });
    }

    void setDominator(Node* node)
    {
        dominator = node;
    }

    Node* getDominator() const
    {
        if (nullptr != dominator)
        {
            return dominator;
        }
        if (getType() == Type::Root)
        {
            return nullptr;
        }
        assert(getNumArguments() == 1);
        return getArgument(0);
    }

    void setPostDominator(Node* node)
    {
        postDominator = node;
    }

    Node* getPostDominator()
    {
        if (nullptr != postDominator)
        {
            return postDominator;
        }
        if (getType() == Type::Term || getType() == Type::Use)
        {
            return nullptr;
        }
        auto isNotUse = [](Node* n)
        {
            return n->getType() != Type::Use;
        };
        // TODO: cache?
        assert(llvm::count_if(getUsers(), isNotUse) == 1);
        for (auto user : getUsers())
        {
            if (isNotUse(user))
            {
                return user;
            }
        }
        llvm_unreachable("");
    }

    Node* getArgument(unsigned i) const
    {
        assert(i < argCount);
        return args[i].arg;
    }

    void setArgument(unsigned i, Node* node)
    {
        assert(i < argCount);
        if (nullptr != args[i].arg)
        {
            args[i].arg->users.erase(args[i].getIterator());
        }
        args[i].arg = node;
        if (nullptr != node)
        {
            node->users.push_back(args[i]);
        }
    }

private:
    Node() = default;
    Node(const Node&) = delete;
    Node(mlir::Operation* op, Type t, llvm::ArrayRef<Node*> a)
    {
        assert(a.size() == 1 || t == Type::Phi);
        operation = op;
        argCount = static_cast<unsigned>(a.size());
        type = t;
        for (auto it : llvm::enumerate(a))
        {
            auto i = it.index();
            if (i >= 1)
            {
                new(&args[i]) Arg();
            }
            auto arg = it.value();
            args[i].index = static_cast<unsigned>(i);
            if (nullptr != arg)
            {
                args[i].arg = arg;
                arg->users.push_back(args[i]);
            }
        }
    }
    ~Node()
    {
        for (unsigned i = 0; i < argCount; ++i)
        {
            if (args[i].arg != nullptr)
            {
                args[i].arg->users.erase(args[i].getIterator());
            }
            if (i >= 1)
            {
                args[i].~Arg();
            }
        }
    }
    friend class MemorySSA;

    static size_t computeSize(size_t numArgs)
    {
        return sizeof(Node) + (numArgs > 1 ? sizeof(Arg) * (numArgs - 1) : 0);
    }

    mlir::Operation* operation = nullptr;
    Node* dominator = nullptr;
    Node* postDominator = nullptr;
    Type type = Type::Root;
    unsigned argCount = 0;

    struct Arg : public llvm::ilist_node<Arg>
    {
        Node* arg = nullptr;
        unsigned index = 0;

        Node* getParent()
        {
            auto offset = static_cast<unsigned>(offsetof(Node, args) + sizeof(Arg) * index);
            return reinterpret_cast<Node*>(reinterpret_cast<char*>(this) - offset);
        }
    };
    llvm::simple_ilist<Arg> users;

    Arg args[1]; // Variadic size
};

plier::MemorySSA::Node* plier::MemorySSA::createNode(mlir::Operation* op, NodeType type, llvm::ArrayRef<plier::MemorySSA::Node*> args)
{
    auto ptr = allocator.Allocate(Node::computeSize(args.size()), std::alignment_of<Node>::value);
    auto node = new(ptr) Node(op, type, args);
    nodesMap[op] = node;
    nodes.push_back(*node);
    return node;
}

plier::MemorySSA::Node* plier::MemorySSA::createDef(mlir::Operation* op, plier::MemorySSA::Node* arg)
{
    return createNode(op, NodeType::Def, arg);
}

plier::MemorySSA::Node* plier::MemorySSA::createUse(mlir::Operation* op, plier::MemorySSA::Node* arg)
{
    return createNode(op, NodeType::Use, arg);
}

plier::MemorySSA::Node* plier::MemorySSA::createPhi(mlir::Operation* op, llvm::ArrayRef<plier::MemorySSA::Node*> args)
{
    return createNode(op, NodeType::Phi, args);
}

void plier::MemorySSA::eraseNode(plier::MemorySSA::Node* node)
{
    assert(nullptr != node);
    assert(node->getUsers().empty());
    auto op = node->getOperation();
    if (op != nullptr)
    {
        nodesMap.erase(op);
    }
    nodes.erase(node->getIterator());
    node->~Node();
}

plier::MemorySSA::Node* plier::MemorySSA::getRoot()
{
    if (nullptr == root)
    {
        root = new(allocator.Allocate(Node::computeSize(0), std::alignment_of<Node>::value)) Node();
        nodes.push_back(*root);
    }
    return root;
}

plier::MemorySSA::Node* plier::MemorySSA::getTerm()
{
    if (nullptr == term)
    {
        Node* temp = nullptr;
        term = new(allocator.Allocate(Node::computeSize(1), std::alignment_of<Node>::value)) Node(nullptr, NodeType::Term, temp);
        nodes.push_back(*term);
    }
    return term;
}

plier::MemorySSA::Node* plier::MemorySSA::getNode(mlir::Operation* op) const
{
    assert(nullptr != op);
    auto it = nodesMap.find(op);
    return it != nodesMap.end() ? it->second : nullptr;
}

void plier::MemorySSA::print(llvm::raw_ostream& os)
{
    for (auto& node : getNodes())
    {
        os << "\n";
        print(&node, os);
        if (node.getOperation())
        {
            node.getOperation()->print(os);
            os << "\n";
        }
    }
}

void plier::MemorySSA::print(plier::MemorySSA::Node* node, llvm::raw_ostream& os) /*const*/ // TODO: identifyObject const
{
    const llvm::StringRef types[] = {
        "MemoryRoot",
        "MemoryDef",
        "MemoryUse",
        "MemoryPhi",
        "MemoryTerm",
    };
    auto writeId = [&](const Node* node)
    {
        if (nullptr != node)
        {
            os << allocator.identifyKnownObject(node);
        }
        else
        {
            os << "null";
        }
    };
    auto type = node->getType();
    writeId(node);
    os << " = ";
    auto typeInd = static_cast<int>(type);
    assert(typeInd >= 0 && typeInd < static_cast<int>(llvm::array_lengthof(types)));
    os << types[typeInd] << "(";
    auto args = node->getArguments();
    llvm::interleaveComma(args, os, writeId);
    os << ") ";
    if (auto dom = node->getDominator())
    {
        os << "dom ";
        writeId(dom);
    }
    if (auto postDom = node->getPostDominator())
    {
        os << "; post-dom ";
        writeId(postDom);
    }
    auto users = node->getUsers();
    if (!users.empty())
    {
        os << "; users: ";
        llvm::interleaveComma(users, os, writeId);
    }
    os << "\n";
}

namespace
{
template<typename C, typename F>
bool checkPhisAlias(C& phiCache, plier::MemorySSA::Node* phi, plier::MemorySSA::Node* stop, mlir::Operation* useOp, F&& mayAlias)
{
    assert(nullptr != phi);
    assert(nullptr != stop);
    assert(nullptr != useOp);
    using NodeType = plier::MemorySSA::Node::Type;
    assert(phi->getType() == NodeType::Phi);
    if (phiCache.count(phi) == 0)
    {
        phiCache.insert(phi);
        for (auto node : phi->getArguments())
        {
            assert(nullptr != node);
            while(node != stop)
            {
                auto type = node->getType();
                if (type == NodeType::Def)
                {
                    assert(nullptr != node->getOperation());
                    if (mayAlias(node->getOperation(), useOp))
                    {
                        return false;
                    }
                }
                else if(type == NodeType::Phi)
                {
                    if (!checkPhisAlias(phiCache, node, stop, useOp, std::forward<F>(mayAlias)))
                    {
                        return false;
                    }
                }
                else
                {
                    llvm_unreachable("");
                }
                node = node->getDominator();
                if (nullptr == node)
                {
                    break;
                }
            }
        }
    }
    return true;
}

template<typename F>
plier::MemorySSA::Node* getDef(plier::MemorySSA::Node* def, mlir::Operation* useOp, F&& mayAlias)
{
    useOp->dump();
    while(true)
    {
        assert(nullptr != def);
        auto dom = def->getDominator();
        if (nullptr == dom)
        {
            return def;
        }

        auto type = def->getType();
        if (type == plier::MemorySSA::Node::Type::Phi)
        {
            llvm::SmallDenseSet<plier::MemorySSA::Node*> phiCache;
            if (!checkPhisAlias(phiCache, def, dom, useOp, std::forward<F>(mayAlias)))
            {
                return def;
            }
        }
        else if (type == plier::MemorySSA::Node::Type::Def)
        {
            assert(nullptr != def->getOperation());
            if (mayAlias(def->getOperation(), useOp))
            {
                return def;
            }
        }
        else
        {
            llvm_unreachable("");
        }
        def = dom;
    }
}
}

mlir::LogicalResult plier::MemorySSA::optimizeUses(llvm::function_ref<bool (mlir::Operation*, mlir::Operation*)> mayAlias)
{
    assert(mayAlias);
    bool changed = false;
    for (auto& node : getNodes())
    {
        if (node.getType() == NodeType::Use)
        {
            assert(node.getNumArguments() == 1);
            auto def = node.getDominator();
            assert(nullptr != def);
            auto op = node.getOperation();
            assert(nullptr != op);
            auto newDef = getDef(def, op, mayAlias);
            assert(nullptr != newDef);
            if (newDef != def)
            {
                node.setArgument(0, newDef);
                changed = true;
            }
        }
    }
    return mlir::success(changed);
}

namespace
{
auto hasMemEffect(mlir::Operation& op)
{
    struct Result
    {
        bool read = false;
        bool write = false;
    };

    Result ret;
    if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op))
    {
        if (effects.hasEffect<mlir::MemoryEffects::Write>())
        {
            ret.write = true;
        }
        if (effects.hasEffect<mlir::MemoryEffects::Read>())
        {
            ret.read = true;
        }
    }
    else if(op.hasTrait<mlir::OpTrait::HasRecursiveSideEffects>())
    {
        ret.write = true;
    }
    return ret;
}

plier::MemorySSA::Node* memSSAProcessRegion(mlir::Region& region, plier::MemorySSA::Node* entryNode, plier::MemorySSA& memSSA)
{
    assert(nullptr != entryNode);
    if (!llvm::hasSingleElement(region))
    {
        // Only structured control flow is supported for now
        return nullptr;
    }

    auto& block = region.front();
    plier::MemorySSA::Node* currentNode = entryNode;
    for (auto& op : block)
    {
        if (!op.getRegions().empty())
        {
            if (auto loop = mlir::dyn_cast<mlir::LoopLikeOpInterface>(op))
            {
                std::array<plier::MemorySSA::Node*, 2> phiArgs = {nullptr, currentNode};
                auto phi = memSSA.createPhi(&op, phiArgs);
                auto result = memSSAProcessRegion(loop.getLoopBody(), phi, memSSA);
                if (nullptr == result)
                {
                    return nullptr;
                }

                if (result != phi)
                {
                    phi->setArgument(0, result);
                    phi->setDominator(currentNode);
                    currentNode->setPostDominator(phi);
                    currentNode = phi;
                }
                else
                {
                    for (auto use : llvm::make_early_inc_range(phi->getUses()))
                    {
                        assert(use.user != nullptr);
                        use.user->setArgument(use.index, currentNode);
                    }
                    memSSA.eraseNode(phi);
                }
            }
            else
            {
                // Unsupported op, check if it has any mem effects
                if (op.walk([](mlir::Operation* nestedOp)
                    {
                        auto res = hasMemEffect(*nestedOp);
                        if (res.read || res.write)
                        {
                            return mlir::WalkResult::interrupt();
                        }
                        return mlir::WalkResult::advance();
                    }).wasInterrupted())
                {
                    return nullptr;
                }
            }
        }
        else
        {
            auto res = hasMemEffect(op);
            if (res.write)
            {
                auto newNode = memSSA.createDef(&op, currentNode);
                newNode->setDominator(currentNode);
                currentNode->setPostDominator(newNode);
                currentNode = newNode;
            }
            if (res.read)
            {
                memSSA.createUse(&op, currentNode);
            }
        }

    }

    return currentNode;
}
}

llvm::Optional<plier::MemorySSA> plier::buildMemorySSA(mlir::Region& region)
{
    plier::MemorySSA ret;
    if (auto last = memSSAProcessRegion(region, ret.getRoot(), ret))
    {
        ret.getTerm()->setArgument(0, last);
    }
    else
    {
        return {};
    }
    return std::move(ret);
}
