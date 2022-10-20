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

#pragma once

#include <iterator>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/simple_ilist.h>
#include <llvm/Support/Allocator.h>

namespace mlir {
struct LogicalResult;
class Operation;
class Region;
} // namespace mlir

namespace imex {

class MemorySSA {
public:
  enum class NodeType { Root, Def, Use, Phi, Term };
  struct Node;

  MemorySSA() = default;
  MemorySSA(const MemorySSA &) = delete;
  MemorySSA(MemorySSA &&) = default;

  MemorySSA &operator=(const MemorySSA &) = delete;
  MemorySSA &operator=(MemorySSA &&) = default;

  Node *createDef(mlir::Operation *op, Node *arg);
  Node *createUse(mlir::Operation *op, Node *arg);
  Node *createPhi(mlir::Operation *op, llvm::ArrayRef<Node *> args);

  void eraseNode(Node *node);
  NodeType getNodeType(Node *node) const;
  mlir::Operation *getNodeOperation(Node *node) const;
  Node *getNodeDef(Node *node) const;
  llvm::SmallVector<Node *> getUsers(Node *node);

  Node *getRoot();
  Node *getTerm();
  Node *getNode(mlir::Operation *op) const;

  struct NodesIterator {
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = Node;
    using pointer = value_type *;
    using reference = value_type &;

    using internal_iterator = llvm::simple_ilist<Node>::iterator;

    NodesIterator(internal_iterator iter);
    NodesIterator(const NodesIterator &) = default;
    NodesIterator(NodesIterator &&) = default;

    NodesIterator &operator=(const NodesIterator &) = default;
    NodesIterator &operator=(NodesIterator &&) = default;

    bool operator==(const NodesIterator &rhs) const {
      return iterator == rhs.iterator;
    }
    bool operator!=(const NodesIterator &rhs) const {
      return iterator != rhs.iterator;
    }

    NodesIterator &operator++();
    NodesIterator operator++(int);

    NodesIterator &operator--();
    NodesIterator operator--(int);

    reference operator*();
    pointer operator->();

  private:
    internal_iterator iterator;
  };

  llvm::iterator_range<NodesIterator> getNodes();

  void print(llvm::raw_ostream &os);
  void print(Node *node, llvm::raw_ostream &os);

  mlir::LogicalResult optimizeUses(
      llvm::function_ref<bool(mlir::Operation *, mlir::Operation *)> mayAlias);

private:
  Node *root = nullptr;
  Node *term = nullptr;
  llvm::DenseMap<mlir::Operation *, Node *> nodesMap;
  llvm::BumpPtrAllocator allocator;
  llvm::simple_ilist<Node> nodes;

  Node *createNode(mlir::Operation *op, NodeType type,
                   llvm::ArrayRef<Node *> args);
};

llvm::Optional<imex::MemorySSA> buildMemorySSA(mlir::Region &region);
} // namespace imex
