# RFC: Region Dialect

Ivan Butygin, Frank Schlimbach

## Summary

We propose a basic region concept which allows groups of operations to carry annotations, provided as attributes and/or SSA values.

## Motivation

At this point there is not standard and convenient way in MLIR to annotate operations or groups of operations in way that can by default persists unrelated passes. Attributes often do not survive passes like the canonicalizer. Annotations using use-def chains are more persistent but using them can become relatively involved.

One example for the need of persistent annotations is the compute-follows-data model: data is allocated in a specific location, like a GPU, and operations on that data are expected to be executed in the same location (e.g. on the same device). The location might be assigned in a higher-level tensor dialect but the actual lowering for the specific device will happen much later when the information of the higher level dialect is gone. This requires some mechanism to persists annotations from a higher level dialect to lower levels.

## Proposal

We propose a simple region operation which accepts attributes and operands and groups operations. The attributes and operands can be seen as annotations. Lower/later passes will either use the region and adequately interpret its attributes and operands, or simply ignore them. A yield operation allows returning values needed lexically after the region.

A region operation has the following properties

- it is not isolated from above
- it has its own lexical scope
- it can be nested
- it has a single block

### Operations

#### region.env_region (region::EnvironmentRegionOp)

Operation that executes its region with a specific environment.

Syntax:

```
operation ::= attr-dict $environment ($args^ `:` type($args))? (`->` type($results)^)? $region
```

`env_region` executes operations inside its region within a specific
environment. Operations are executed exactly once. All SSA values that
dominate the op can be accessed inside the op.

`env_region` takes "environment" attribute and zero or more SSA arguments.

Actual interpretation of the "environment" attribute and arguments is not
specified and is left to the lowering passes.

Values can be yielded from `env_region` region using `env_region_yield` op.
Values yielded from this op's region define the op's results.

Regions can be arbitrarily nested into each other, it is up to specific
passes how to interpret nested env regions.

#### region.env_region_yield (region::EnvironmentRegionYieldOp)

Environment region yield and termination operation.

Syntax:

```
operation ::= attr-dict ($results^ `:` type($results))?
```

`env_region_yield` yields SSA values from the `env_region` op region and
terminates it.

If `env_region_yield` has any operands, the operands must match the parent
operation's results.

## Examples

A simple example:

```
%0 = ...
%1 = region.env_region "GPU" -> i64 {
    %inner_0 = op1 (%0) // This is fine, not isolated from above
    %inner_1 = op2 (%inner_0) -> i64 // This is in the same lexical context
    region.env_region_yield %inner_1 : i64 // Goes into %1
}
// This is fine, access return value
%2 = op3 (%1)  // OK

// This is not, since the lexical context is now dead
%3 = op4 (%inner_0) // ERROR
```

A simple, dedicated environment attribute for indicating GPU envs could look like this:
`#region.gpu_env<device = "opencl:gpu:0">`
and could be used like this
```
%0 = ...
%1 = region.env_region #region.gpu_env<device = "opencl:gpu:0"> -> i64 {
    %inner_0 = op1 (%0) // This is fine, not isolated from above
    %inner_1 = op2 (%inner_0) -> i64 // This is in the same lexical context
    region.env_region_yield %inner_1 : i64 // Goes into %1
}
```

## Passes

We consider the following passes, others are certainly possible:
- `add-gpu-regions`: adding `region.env_region` around operations which operate on or return arrays that are targeted for GPUs
- canonicalizers: simplify yield operations and fuse adjacent regions
- `insert-gpu-allocs`: convert `memref.alloc` into `gpu.alloc` for occurrences of `memref.alloc` which are inside a GPU region. Convert `memref.dealloc` to `gpu.dealloc` similarly.
- `remove-gpu-regions`: remove GPU regions by moving operations in the region to their outer lexical scope

## Upstreaming Plans

We intend to create an RFC for adding the operations suggested here, probably into some standard dialect like SCF.
