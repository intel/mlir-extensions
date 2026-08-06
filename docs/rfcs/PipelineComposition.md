# RFC: High Level MLIR Pass Pipeline Composition

Ivan Butygin, Renato Golin

## Summary

We propose a new API to build a dependencies-based graph of pass pipelines.

## Motivation

At this point MLIR pass infrastruture only allows to define a linear sequence of passes. While this approach works
for simple cases, more complex compiler pipelines may require more control over passes execution order.

Two main usecases we are considering:
* Extensibility. We want to have the abilty for users to extend existing pipeline by inserting theirs custom passes
into various places while reusing most of the exiting pipeline. With current approach the most common way way to achieve this is
for pipeline developer to add a fixed 'extension point' during initial pipeline design.
* Dynamic pipeline control flow. It's often required to have an ability select specific sequence of passes based on some info which
is only becomes available durin compilation process. It can be either branching to some separate sequnces of passes based on selected codegen path
or running sequence of passes in the loop until some runtime condition is reached.

## Proposal

This API also allows a dynamic control flow between pipelines in graph, controlled by passes inside said pipelines.

## New APIs

### PipelineGraph
```C++
class PipelineGraph {
public:
    LogicalResult registerPipeline(
        StringRef name,
        ArrayRef<StringRef> predecessors,
        ArrayRef<StringRef> successors,
        ArrayRef<StringRef> jumpTargets,
        std::function<void(OpPassManager &)> populateFunc,
        raw_ostream &errorStream);

    LogicalResult registerPipeline(
        StringRef name,
        ArrayRef<StringRef> predecessors,
        ArrayRef<StringRef> successors,
        ArrayRef<StringRef> jumpTargets,
        std::function<void(PipelineGraph &)> populateFunc,
        raw_ostream &errorStream);

    FailureOr<PipelineSchedule> createPipelineSchedule(raw_ostream &errorStream) const;
};
```
`PipelineGraph` is main entry point for this new API.
User is adding pipelines into graph via `registerPipeline` function, each pipeline have the following properties:
* `name` - Name of thepipeline
* `predecessors` - List of names of pipelines which must be run before current pipeline.
* `successors` - List of the names of pipelines which must be run after current pipeline.
* `jumpTargets` - List of the names of pipelines to which we can dynamically jump after current pipeline.
* `populateFunc` - Callback to populate pipeline with passes.

User can either register linear sequence of passes via `OpPassManager` variant or a subgraph via `PipelineGraph`.
Registered subgraphs are isolated from the current graph and always executed as single entity (i.e. control flow can't jump into or from the middle of subgraph).

After user populated the graph object they must call `createPipelineSchedule` method to compile the resulted graph into runnable schedule.
`createPipelineSchedule` will build a DAG from pipelines dependencies provided by user, and will try to get linear execution order to satify these dependencies.

If two pipelines doesn't have direct and indirect dependencies, order in which they will be executed is not specified.

Implicit cycles in graph are not allowed and will result in `createPipelineSchedule` returning error. But cycles via `jumpTargets` are allowed (see later).

Empty pipelines (i.e. pipelines without passes, when `populateFunc` does nothing) are allowed and sometimes desirable.

One usecase is using empty pipelines as anchors for other pipelines. Let's say use wants to split hist entire compiler pipeline into 3 stages: `high`, `middle` and `low`.
They can create a two empty pipelines `high-to-middle` and `middle-to-low` with appropriate dependencies and the use those as anchors to specify at which compiler stage insert stages which do actual work.

### PipelineSchedule
```C++
class PipelineSchedule {
public:
    LogicalResult run(Operation *op);
};
```
`PipelineSchedule` object encapsulates compiled pipeline graph.
Main method is `LogicalResult run(Operation *op);` which follows existing MLIR `PassManager::run` semantics.
`run` will execute

### Dynamic pipeline control flow

If pipeline has `jumpTargets` populated, it can possibly jump to one of those `jumpTargets` after finishing instead of continuing normally.

Jump is controlled via special MLIR attribute `pipeline.jump_target`, attached to top-level op (usually `builtin.module`).
```
builtin.module {pipeline.jump_target="Foo"} {
...
}
```

Passes inside pipeline can set this attribute to indicate they want compilatin flow to jump to the specific point.
After current pipeline is finished, runtime will check if module object have attribute set and if it does, jump to the selected pipeline and clear the attribute.

Setting attribute to the value, which wasnt in `jumpTargets` for the current pipeline will result in error and abort the compilation flow.


### Pipeline examples

Here is some examples where non-trivial pipeline dependencies are needed.

#### Numba-mlir

https://github.com/numba/numba-mlir

```
          frontend
              |
              V
          cfg-to-scf
            /    ^
           /      \
          V        \
     python-to-std  |
          \        /
           \      /
            V    /
      numpy-to-linalg
              |
              V
        bufferization
              |
              V
        optimization
           /      \
          /        \
         V          \
    lower-to-gpu     |
         \          /
          \        /
           V      V
         lower-to-llvm
```
In this pipeline we are lowering scalar python ops in `python-to-std` stage and
numpy ops in `numpy-to-linalg` and we may need to jump backwards to `cfg-to-scf`/`python-to-std`
multiple times to lower generated `linalg.generic` body, which are represented as
normal python function in our numpy->linalg conversion.

Pipeline description for this will looks like (pseudocode):
```
# pipeline format:
# name: [predecessors], [successors], [jumps]
frontend: [], [], []
cfg-to-scf: [frontend], [optimization], []
python-to-std: [cfg-to-scf], [optimization], []
numpy-to-linalg: [python-to-std], [bufferization, optimization], [cfg-to-scf]
bufferization: [], [optimization], []
optimization: [], [], []
lower-to-gpu: [optimization], [lower-to-llvm], []
lower-to-llvm: [optimization], [], []
```

#### TPP

https://github.com/plaidml/tpp-mlir
```
          frontend
          /       \
         V         V
  gpu-pipeline  default-pipeline
         \         /
          V       V
        bufferization
              |
              V
       linalg-lowering
              |
              V
        lower-to-llvm
```

Pipeline will looks like:
```
frontend: [], [], []
gpu-pipeline: [frontend], [bufferization], []
default-pipeline: [frontend], [bufferization], []
bufferization: [], [linalg-lowering], []
linalg-lowering: [], [lower-to-llvm], []
lower-to-llvm: [], [], []
```
