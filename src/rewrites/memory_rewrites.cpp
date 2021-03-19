#include "plier/rewrites/memory_rewrites.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <llvm/Support/Allocator.h>
#include <mlir/Dialect/SCF/SCF.h>

namespace
{
bool isWrite(mlir::Operation& op)
{
    if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op))
    {
        return effects.hasEffect<mlir::MemoryEffects::Write>();
    }
    return false;
}

bool isRead(mlir::Operation& op)
{
    if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op))
    {
        return effects.hasEffect<mlir::MemoryEffects::Read>();
    }
    return false;
}

struct Result
{
    bool changed;
    bool hasWrites;
    bool hasReads;
};

Result promoteLoadsImpl(mlir::Region& region, mlir::PatternRewriter& rewriter)
{
    bool changed = false;
    bool hasWrites = false;
    bool hasReads = false;
    bool storeDead = false;
    for (auto& block : region.getBlocks())
    {
        mlir::StoreOp currentStore;
        for (auto& op : llvm::make_early_inc_range(block))
        {
            if (!op.getRegions().empty())
            {
                for (auto& nestedRegion : op.getRegions())
                {
                    auto res = promoteLoadsImpl(nestedRegion, rewriter);
                    if (res.changed)
                    {
                        changed = true;
                    }
                    if (res.hasWrites)
                    {
                        currentStore = {};
                    }
                    if (res.hasReads)
                    {
                        storeDead = false;
                    }
                }
                continue;
            }

            if (auto load = mlir::dyn_cast<mlir::LoadOp>(op))
            {
                hasReads = true;
                if (currentStore)
                {
                    if (load.memref() == currentStore.memref() &&
                        load.indices() == currentStore.indices())
                    {
                        rewriter.replaceOp(&op, currentStore.value());
                        changed = true;
                    }
                    else
                    {
                        storeDead = false;
                    }
                }
            }
            else if (auto store = mlir::dyn_cast<mlir::StoreOp>(op))
            {
                if (currentStore && storeDead &&
                    currentStore.memref() == store.memref() &&
                    currentStore.indices() == store.indices())
                {
                    rewriter.eraseOp(currentStore);
                }
                hasWrites = true;
                currentStore = store;
                storeDead = true;
            }
            else if (isWrite(op))
            {
                hasWrites = true;
                currentStore = {};
            }
            else if (isRead(op))
            {
                hasReads = true;
                storeDead = false;
            }
            else if(op.hasTrait<mlir::OpTrait::HasRecursiveSideEffects>())
            {
                currentStore = {};
                hasWrites = true;
                hasReads = true;
                storeDead = false;
            }
        }
    }
    return Result{changed, hasWrites, hasReads};
}

bool checkIsSingleElementsMemref(mlir::ShapedType type)
{
    if (!type.hasRank())
    {
        return false;
    }
    return llvm::all_of(type.getShape(), [](auto val) { return val == 1; });
}
}

mlir::LogicalResult plier::promoteLoads(mlir::Region& region, mlir::PatternRewriter& rewriter)
{
    return mlir::success(promoteLoadsImpl(region, rewriter).changed);
}

mlir::LogicalResult plier::promoteLoads(mlir::Region& region)
{
    class MyPatternRewriter : public mlir::PatternRewriter
    {
    public:
        MyPatternRewriter(mlir::MLIRContext *ctx) : PatternRewriter(ctx) {}
    };

    MyPatternRewriter dummyRewriter(region.getContext());
    return mlir::success(promoteLoadsImpl(region, dummyRewriter).changed);
}

mlir::LogicalResult plier::PromoteLoads::matchAndRewrite(mlir::FuncOp op, mlir::PatternRewriter& rewriter) const
{
    return promoteLoads(op.getRegion(), rewriter);
}

mlir::LogicalResult plier::SingeWriteMemref::matchAndRewrite(mlir::StoreOp op, mlir::PatternRewriter& rewriter) const
{
    auto memref = op.memref();
    if (!checkIsSingleElementsMemref(memref.getType().cast<mlir::ShapedType>()))
    {
        return mlir::failure();
    }
    auto parent = memref.getDefiningOp();
    if (!mlir::isa_and_nonnull<mlir::AllocOp, mlir::AllocaOp>(parent))
    {
        return mlir::failure();
    }

    mlir::StoreOp valueStore;
    llvm::SmallVector<mlir::Operation*> loads;
    for (auto user : memref.getUsers())
    {
        if (auto store = mlir::dyn_cast<mlir::StoreOp>(user))
        {
            if (valueStore)
            {
                // More than one store
                return mlir::failure();
            }
            valueStore = store;
        }
        else if (auto load = mlir::dyn_cast<mlir::LoadOp>(user))
        {
            loads.emplace_back(load);
        }
        else if (mlir::isa<mlir::DeallocOp>(user))
        {
            // nothing
        }
        else
        {
            // Unsupported op
            return mlir::failure();
        }
    }

    auto parentBlock = parent->getBlock();
    if (!valueStore || valueStore->getBlock() != parentBlock)
    {
        return mlir::failure();
    }

    auto val = valueStore.value();
    for (auto load : loads)
    {
        rewriter.replaceOp(load, val);
    }
    for (auto user : llvm::make_early_inc_range(parent->getUsers()))
    {
        rewriter.eraseOp(user);
    }
    rewriter.eraseOp(parent);
    return mlir::success();
}

namespace
{
class MemorySSA
{
public:
    struct Node : public llvm::ilist_node<Node>
    {
        enum class Type
        {
            Root,
            Def,
            Use,
            Phi
        };

        mlir::Operation* getOperation() const
        {
            return operation;
        }

        Type getType() const
        {
            return type;
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
            assert(nullptr != op);
            assert(a.size() == 1 || t == Type::Phi);
            operation = op;
            argCount = static_cast<unsigned>(a.size());
            type = t;
            for (auto it : llvm::enumerate(a))
            {
                auto i = it.index();
                if (i > 1)
                {
                    new(&args[i]) Arg();
                }
                auto arg = it.value();
                args[i].offset = static_cast<unsigned>(offsetof(Node, args) + sizeof(Arg) * i);
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
                args[i].~Arg();
            }
        }
        friend class MemorySSA;

        static size_t computeSize(size_t numArgs)
        {
            return sizeof(Node) + (numArgs > 1 ? sizeof(Arg) * (numArgs - 1) : 0);
        }

        mlir::Operation* operation = nullptr;
        Type type = Type::Root;
        unsigned argCount = 0;

        struct Arg : public llvm::ilist_node<Arg>
        {
            Node* arg = nullptr;
            unsigned offset = 0;

            Node* getParent()
            {
                return reinterpret_cast<Node*>(reinterpret_cast<char*>(this) - offset);
            }
        };
        llvm::simple_ilist<Arg> users;

        Arg args[1]; // Variadic size
    };

    Node* createNode(mlir::Operation* op, Node::Type type, llvm::ArrayRef<Node*> args)
    {
        assert(nullptr != op);
        auto ptr = allocator.Allocate(Node::computeSize(args.size()), std::alignment_of<Node>::value);
        auto node = new(ptr) Node(op, type, args);
        nodesMap[op] = node;
        nodes.push_back(*node);
        return node;
    }

    Node* getRoot()
    {
        if (nullptr == root)
        {
            root = new(allocator.Allocate(Node::computeSize(0), std::alignment_of<Node>::value)) Node();
            nodes.push_back(*root);
        }
        return root;
    }

    MemorySSA() = default;
    MemorySSA(const MemorySSA&) = delete;
    MemorySSA(MemorySSA&&) = default;

    Node* getNode(mlir::Operation* op) const
    {
        assert(nullptr != op);
        auto it = nodesMap.find(op);
        return it != nodesMap.end() ? it->second : nullptr;
    }

    auto& getNodes()
    {
        return nodes;
    }

    void print(Node* node, llvm::raw_ostream& os) /*const*/ // TODO: identifyObject const
    {
        const llvm::StringRef types[] = {
            "MemoryRoot",
            "MemoryDef",
            "MemoryUse",
            "MemoryPhi",
        };
        auto getId = [this](const Node* node)
        {
            assert(nullptr != node);
            return *allocator.identifyObject(node);
        };
        auto type = node->getType();
        os << getId(node) << " = ";
        os << types[static_cast<int>(type)] << "(";
        auto args = node->getArguments();
        llvm::interleaveComma(args, os, [&](const Node* n) { os << getId(n); });
        os << ")";
        auto users = node->getUsers();
        if (!users.empty())
        {
            os << " users: ";
            llvm::interleaveComma(users, os, [&](const Node* n) { os << getId(n); });
        }
        os << "\n";
    }

private:
    Node* root = nullptr;
    llvm::DenseMap<mlir::Operation*, Node*> nodesMap;
    llvm::BumpPtrAllocator allocator;
    llvm::simple_ilist<Node> nodes;
};

MemorySSA::Node* memSSAProcessRegion(mlir::Region& region, MemorySSA::Node* entryNode, MemorySSA& memSSA)
{
    assert(nullptr != entryNode);
    if (!llvm::hasSingleElement(region))
    {
        // Only structured control flow is supported for now
        return nullptr;
    }

    auto& block = region.front();
    MemorySSA::Node* currentNode = entryNode;
    using NodeType = MemorySSA::Node::Type;
    for (auto& op : block)
    {
        auto createNode = [&](NodeType type, auto args)
        {
            return memSSA.createNode(&op, type, args);
        };
        if (!op.getRegions().empty())
        {
            if (mlir::isa<mlir::scf::ForOp, mlir::scf::ParallelOp>(op))
            {
                assert(llvm::hasSingleElement(op.getRegions()));
                std::array<MemorySSA::Node*, 2> phiArgs = {nullptr, currentNode};
                auto phi = createNode(NodeType::Phi, phiArgs);
                currentNode = memSSAProcessRegion(op.getRegions().front(), phi, memSSA);
                phi->setArgument(0, currentNode);
            }
            else if (mlir::isa<mlir::scf::ReduceOp>(op))
            {
                // TODO: handle reduce
            }
            else
            {
                // Unsupported op
                return nullptr;
            }
        }
        else
        {
            if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op))
            {
                if (effects.hasEffect<mlir::MemoryEffects::Write>())
                {
                    currentNode = createNode(NodeType::Def, currentNode);
                }
                if (effects.hasEffect<mlir::MemoryEffects::Read>())
                {
                    createNode(NodeType::Use, currentNode);
                }
            }
            else if(op.hasTrait<mlir::OpTrait::HasRecursiveSideEffects>())
            {
                currentNode = createNode(NodeType::Def, currentNode);
            }
        }

    }

    return currentNode;
}

llvm::Optional<MemorySSA> buildMemorySSAImpl(mlir::FuncOp func)
{
    MemorySSA ret;
    if (nullptr == memSSAProcessRegion(func.getRegion(), ret.getRoot(), ret))
    {
        return {};
    }
    return std::move(ret);
}
}

mlir::LogicalResult plier::buildMemorySSA(mlir::FuncOp func)
{
    llvm::errs() << "buildMemorySSA1\n";
    auto res = buildMemorySSAImpl(func);
    if (!res)
    {
        llvm::errs() << "buildMemorySSA2\n";
        return mlir::failure();
    }
    auto& memSSA = *res;
    func.dump();
    llvm::errs() << "buildMemorySSA3\n";
    for (auto& node : memSSA.getNodes())
    {
        llvm::errs() << "\n";
        memSSA.print(&node, llvm::errs());
        if (node.getOperation())
        {
            node.getOperation()->dump();
        }
    }

    llvm::errs() << "buildMemorySSAend\n";
    return mlir::success();
}
