# RFC: DistRuntime Dialect

Frank Schlimbach, Tuomas Karna

## Summary

We propose a dialect representing operations that typically require a
distributed runtime.

## Motivation

Some domain-specific dialects, like HLO, include operations related to
communication between members of distributed teams (such as MPI processes).
There is no separation of concerns, domain-specific operations are mixed with
runtime and communication primitives. The tight integration makes it
inconvenient to share the domain-unrelated functionality with other dialects –
in addition to a potentially unnecessarily large machinery and runtime (like the
XLA universe).  

In complex cases, possible lower-level dialects, like the proposed MPI dialect,
are too low level to realistically allow lowering fully in MLIR.

A semantically rich and general dialect allows combining with various other
dialects and the use of small and special-purpose runtime libraries written in
higher-level languages. Of course, lowering such a dialect entirely in MLIR is
also possible for those who dare.

## Proposal

We propose a dialect that supports communication in SPMD environments. The
general assumption is that teams (e.g., groups) of processes/threads execute the
same set of operations but operate on different data. Most operations in this
dialect hence require the team argument. It is up to the lowering/runtime to
interpret the value of a given team argument.

Two kinds of operations are provided:

1. A set of basic operations providing information about the state of the
   runtime, like the id of the caller within a team or the number of members
   within a team. These operations do not require the participation of other
   team members.
2. A set of high-level primitives that typically require communication and,
   therefore, involve - implicitly or explicitly - other team members.

Some operations are defined to enable asynchronous communication which allows
for overlapping communication and computation. Such operations return a value of
the opaque type `AsyncHandle`. It is up to the runtime and related lowering
passes to give life to the type. The results of such an asynchronous
operation, whether they are written in-place or returned as values, must be
preceded with a call to `distruntime.wait` operation before their first
use/consumption.

### Basic Operations

#### `distruntime.team_size` (`distruntime::TeamSizeOp`)

Operation that returns the number of team members of a given team.

Syntax:

```
operation ::= "distruntime.team_size"() {team=$team} : () -> index
```

#### `distruntime.team_member ` (`distruntime::TeamMemberOp`)

Operation that returns the member of the given team that the caller represents.
Members of a team are represented as Index types.

Syntax:

```
operation ::= "distruntime.team_member"() {team=$team} : () -> index
```

#### `distruntime.wait` (`distruntime::WaitOp`)

Wait for asynchronous operation to finish; accepts an `AsyncHandle`.

Syntax:

```
operation ::= "distruntime.wait"($asynchandle) : (AsyncHandle)
```

### Operations with Collective Communication

#### `distruntime.allreduce` (`distruntime::AllReduceOp`)

Operation that performs an in-place all-reduce with a given operation. The shape
of the data argument must be identical for all members of the team. Reduction
happens for each element of the argument over all team members. The meaning of
the 'op' attribute is defined by the lowering passes.

Syntax:

```
operation ::= "distruntime.allreduce"($data) {team=$team, op=$op} : (AnyMemRef)
```

#### `distruntime.get_halo` (`distruntime::GetHaloOp`)

For a given, distributed array, compute and return the left and right halo  
as implied by the locally owned data and requested bounding box.

Data that is not locally owned will be provided in the left or right halo,
depending on if the data is from before or after the local part in the first
dimension of the global array. Hence it is possible that one or both returned
halos are empty.

The local data is not modified.

Arguments:

- `local`: the locally owned data
- `gShape`: the global shape of the distributed array
- `lOffsets`: the offset of the local data within the global array
- `bbOffsets`: the offsets of the requested data part
- `bbSizes`: the shape of the requested data part
- `team`: the distributed team owning the distributed array
- `key` [optional]: a statically assigned id for the given operation (to allow caching)

`gShape`, `lOffsets`, `bbOffsets` and `bbSizes` are variadic arguments with same
size `r` where `r` is the rank of the global array (e.g., one number for each
dimension of the global array). 1trim Returns an `AsyncHandle`, the left and the
right halo.

Syntax:

```
operation ::= "distruntime.get_halo"($local, $gShape, $lOffsets, bbOffsets, $bbSizes)\
 {team=$team, key=$key : I64} \
 : (AnyType, Variadic<Index>, Variadic<Index>, Variadic<Index>, Variadic<Index>) \
 -> (AsyncHandle, AnyType, AnyType)
