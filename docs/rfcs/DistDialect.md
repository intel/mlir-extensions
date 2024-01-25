# RFC: Dist Dialect

Frank Schlimbach

## Summary

We propose a dialect for distributing NDArrays in a SPMD fashion, allowing
automatic scale-out on systems with non-shared memory, such as HPC clusters or
systems with multiple GPUs.

## Motivation

Within a well defined scope automatic scale-out to large systems
with distributed memory can be possible. The array-API (numpy-like arrays) has
such a well-defined scope. The semantics of its small set of operations are
known. Additionally, data partitioning and distribution are straightforward and
so make it a good target not only for automatic SPMD parallelization but also
distribution. Of course some use cases can still be hard to automatically
parallelize/distribute with good performance.

A distributed dialect which supports basic array operations (similarly to the
NDArray) can be used by various front ends targeting MLIR without the need to
(re-)implement distribution separately.

## Proposal

We propose a dialect that provides basic features to allow automatic
partitioning and distribution of NDArrays. The dialect assumes SPMD execution
model. More specifically, each execution unit (or process) executes the same
program but locally owns only a part of the globally distributed data. There is
no central entity which partitions data or assigns work to workers.

The Dist dialect is related to the NDArray and DistRuntime dialect. It is
expected that the Dist dialect will eventually get lowered to NDArray and
DistRuntime. Being familiar with NDArray and DistRuntime will greatly help
understanding the remainder of the proposal.

### Attaching Distribution Information To An NDArray

The Dist dialect defines a new environment attribute `DistEnvAttr` that can be
attached to NDArrays. `DistEnvAttr` carries the information required to describe
the partitioning of the global NDArray.

As an example, let's assume we want to equally distribute `ndarray.ndarray<33xi64>` across a team of 4. The third member of the team would attach label the type as a distributed array by attaching the following `DistEnvAttr`:

`ndarray.ndarray<44xi64, #dist.dist_env<team = 37416 loffs = 22 lparts = 0,11,0>>`

This defines the following:

* the array has global size `44`
* the array is distributed across team `37416`
* the local data starts at global index `22`
* the local part is of size `11`
* the size of the right and left halos is `0`

Notice that `lparts` encodes the shapes of 3 parts that are held locally:

1. left halo
2. locally owned data
3. right halo

All parts are of type `ndarray.NDArray`. Halos parts are copies of data owned by
remote team members. Parts always represent pieces of the global array resulting
from block-partitioning, i.e. they represent a contiguous block of the global
index space. Furthermore, the concatenation of left halo, local data and right
halo also represents a contiguous subset of the global index space.

At this point, arrays are split only in the first dimension. A more general
scheme can be added once required. However, when more than one dimension is cut
it requires more than two halo parts and 'left' and 'right' are no longer
sufficient to describe their position relative to the locally owned data.

Notice that any part can be empty - even the locally owned part. For example: a
subview of a global array might not intersect with the locally owned part.

Parts and offsets are omitted (only) for 0d arrays:
`ndarray.ndarray<f64, #dist.dist_env<team = 1>`

The local offset represents offsets in all dimensions, so in principle allows
partitions across multiple dimensions. For each dimension, the offset is
provided to the first part (in most cases that's the left halo).

The offsets and sizes in `DistEnvAttr` can be static as in the example.
Alternatively, they can be partially or fully dynamic - even if the global size
is static. The above example with fully dynamic local offsets and sizes would become:

`ndarray.ndarray<44xi64, #dist.dist_env<team = 37416 loffs = ? lparts = ?,?,?>>`

There is no placeholder for unknown teams.

The distribution metadata generalizes to arrays of arbitrary dimensions. Here is
an example of a distributed 2d array type:

`ndarray.ndarray<44x55xi64, #dist.dist_env<team = 1 loffs = 22,0 lparts = 0x0,11x55,0x0>>`

To indicate that the array is distributed across devices/GPUs, an additional
environment gets attached to `ndarray.ndarray`. The additional environment
defines on which device/GPU the local data should be stored. See NDArray spec
for details about GPU support.

As an example, consider distributing an array across two GPUs in the same computer. The types could look like this

* team member 0:
  `ndarray.ndarray<22xi64, #dist.dist_env<team = 1 loffs = 22 lparts = 0,22,0>, #region.gpu_env<device = "sycl:gpu:0">>`
* team member 1:
  `ndarray.ndarray<22xi64, #dist.dist_env<team = 1 loffs = 22 lparts = 0,22,0>, #region.gpu_env<device = "sycl:gpu:1">>`

#### dist.init_dist_array (dist::InitDistArrayOp)

Instantiate a distributed array, binding  to distributed meta information.

Accepted dynamic distributed meta information:
    - the local offset
    - local parts

The team and resulting global shape is encoded in the result type.

### Operations For Querying The Current State Of The Distributed NDArray

#### dist.local_offsets_of (dist::LocalOffsetsOfOp)

Get local offsets of a distributed array.

Returns `rank`-many values, one for each dimension of `$array`.

`dist.local_offsets_of($array) : (ndarray.ndarray) -> Variadic<index>`

#### dist.parts_of (dist::PartsOfOp)

Get local parts of a distributed array. Returns either one (0d array) or 3 parts
(all other cases: left halo, locally owned data, right halo) as
`ndarray.ndarray`. Returned arrays have the same rank as the input array.

`dist.local_offsets_of($array) : (ndarray.ndarray) -> Variadic<ndarray.ndarray>`

### Operations Supporting Partitioning

#### dist.repartition (dist::RePartitionOp)

Repartition an array so that each process holds the requested data locally.

Creates a new NDArray by repartitioning the input array. It is assumed to be a
collective call. All participating processes request which part of the global
array they need. The halo parts of the returned array get filled with data that
is owned by remote team members. The local data is not modified, the returned
local part is a subview of the local part of the input.

Target offset and target shape are optional arguments. If missing the operations
returns a default-partitioned array.

`$array [loffs $target_offsets lsizes $target_sizes] attr-dict : (ndarray.ndarray[,Variadic<index>, Variadic<index>]) to ndarray.ndarray`

#### dist.default_partition (dist::DefaultPartitionOp)

Compute the default shape and offsets of the local partition.

All input and output shapes/offsets are vectors with same length.

Arrays are cut along the first dimension and partitions are equally distributed
among all members of the team. Member "i" of the team gets assigned to part "i".
Odd elements in the cut dimension are equally distributed among the last team
members. This guarantees that the sizes of local parts differ by at most one
element in the cut dimension.

For example, an array of size 8 will yield the local part sizes (2, 2, 2, 2) if
the team has 4 members. For a team of 3 it will render (2, 3, 3).

Other partition strategies could be added later.

```MLIR
$loffs, $lshape = dist.default_partition($nprocs, $tid, $gshape) \
  : (index, index, Variadic<index>) \
  -> (Variadic<index>, Variadic<index>)
```

#### dist.local_target_of_slice (dist::LocalTargetOfSliceOp)

Compute local intersection of a distributed array with a slice.

This operation computes the intersection of the local part of the array and the
provided slice. The slice is provided as a triplet of offsets, sizes and strides
(similar to a subview). While the slice refers to the global index space of the
distributed array, the operation returns local offsets and sizes, relative to
the local part (e.g. these are not global indices).

All input and output shapes/offsets/strides are `$array.rank()`-long vectors.

```MLIR
dist.local_target_of_slice $array \[$offsets\]\[$sizes\]\[$strides\] \
   : (ndarray.ndarray) to (Variadic<index>, Variadic<index>)
```

### Utility Operations Supporting Optimizations

#### dist.local_bounding_box (dist::LocalBoundingBoxOp)

Compute (or extend) bounding box for data locally required by given view and
target.

The locally required view is the intersection of the given view and target.

If an existing bounding box is provided, update the bounding box. The update strategy is determined by the `inner` attribute:

* if `inner` is unset (default) return the convex hull of given bounding box and
  locally required view.
* else return the intersection of given bounding box and locally required view.

If no bounding box is provided (through `b_b_offsets` and `b_b_sizes`) return the offset and shape of the locally required view.

The bounding box is returned as global offsets and shape.

All input and output shapes/offsets/strides are vectors with same length.

```MLIR
dist.local_bounding_box $inner \[$offsets\]\[$sizes\]\[$strides\] \
  \[$target_offsets\]\[$target_sizes\] \
  [bboffs $b_b_offsets bb_sizes $b_b_sizes] \
  : (Variadic<index>, Variadic<index>)
```

#### dist.local_core (dist::LocalCoreOp)

Compute or update overlap of given core, locally owned data and locally required
data.

The locally required view is the intersection of the given slice and target.

If no local core is provided, return the intersection of locally owned data and
locally required data. Otherwise return the intersection of given core, locally
owned data and locally required data.

The intersection is returned as global offsets and shape.

```MLIR
$src toffs $targetOffsets tsizes $targetSizes \
  soffs $sliceOffsets ssizes $sliceSizes sstrides $sliceStrides \
  [coffs $coreOffsets csizes $coreSizes]
  : (ndarray.ndarray) to (Variadic<index>, Variadic<index>)
```

### Operations Extending NDArray Operations

For effective distribution and optimization some operations in the NDArray
dialect require additional information. Extended versions of such operations are
provided in the Dist dialect. They are expected to eventually lower to NDArray.

Operations are extended by two concepts:

1. Target view

   A target view is the view (or region) of the result to which the locally
   computed result is mapped to. It is defined by global view offsets and sizes.
   A series of dependent operations can share the same target.

2. Core view

   A core view is a region of the target view that is shared across a
   series of dependent operations. The core is always a subset of the locally
   owned data. It is defined by global view offsets and sizes. The core is used
   to generate loops of matching loop bounds to facilitate loop fusion. This is
   non-trivial if the combined views are shifted views of the same array
   because the shifted views have different intersections with halos and the
   locally owned data.

#### dist.subview (dist::SubviewOp)

Extends the ndarray.SubviewOp with optional target offsets and target sizes.

#### dist.ewbin (dist::EWBinOp)

Extends the ndarray.SubviewOp with optional core offsets, core sizes and target offsets.

#### dist.ewuny (dist::EWUnyOp)

Extends the ndarray.SubviewOp with optional core offsets, core sizes and target offsets.
