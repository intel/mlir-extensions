# RFC: Parallel Array Dialect (NDArray)

Frank Schlimbach, Ivan Butygin, Diptorup Deb

## Summary

We propose a solution which can provide a high-level dialect for tensors and (numpy-like) tensor-operations allowing passes to create parallelism for GPU, OpenMP and distributed memory while following the compute-follows-data concept.

In this document, we describe several dialects and passes to provide a good high-level view on the overall flow. We might want to split this into more than one document.

## Motivation

MLIR provides a few tensor dialects (like tensor and TOSA) and separate dialects/conversions for targeting devices and parallelism. An extended type is needed to keep the required analysis in the compiler pipeline to an acceptable extent. Even moderately complicated flows, for example when device and host tensors exist at the same time, complicated analysis is needed to correctly implement compute-follows. This is particularly true when converting to distributed parallelism, such as MPI-based; this is difficult to get right with existing dialects if possible at all. The here suggested type `ndarray` enables such analysis by providing the necessary information, like the location of the data (like the device or distribution team).

Additionally, the current architecture in IMEX does not have an explicit MLIR representation of arrays and so makes it very hard to re-use its tensor functionality in MLIR alone. Currently, potential front-ends (e.g. numba-like ro torch-like packages) require functionality to emit MLIR which converts array semantics into tensor semantics. `NDArray` provides array semantics in a form which is usable directly by front-ends.

1. (numpy-like) Python arrays are mutable. MLIR's tensor semantics are not sufficient to express in-place semantics and using memrefs directly makes it impossible to effectively existing MLIR optimization passes such as `linalg-fuse-elementwise-ops`. A dialect tailored towards the needs of such arrays would allow python libraries to express also mutable semantics on the same level as immutable ones.
2. A dedicated dialect for such numpy-like arrays and operations can accept hints for targets (like devices, MPI etc) and enable compute-follows-data in libraries independent of front-ends like numba or torch. One example of such a use case is a distributed and JIT-compiler based numpy implementation.

## Proposal

The main purpose of this proposal is to enable compute-follows-data for a large variety of platforms. We strive to enable it not only different kinds of shared memory systems (like CPUs, GPUs, FPGAs, ...) but also (potentially heterogenous) distributed memory systems. In its core, compute-follows-data means that operations on an tensor will happen where the tensor was allocated.

The back-bone in our proposal is a tensor dialect which we call `NDArray` providing a tensor type (`ndarray.ndarray`) and operations on them. Additionally, we accompany `NDArray` with a few more dialects to express concerns related to device handling and distribution:

1. `Region` dialect: use simple regions to carry annotation on operations across passes (which is not possible with attribute)
2. `Dist` dialect: dealing with the aspects of distributed data management, such as distributed memory allocation, GC and communication
3. `DistRuntime` dialect: features for distributed operation which are expected to be lowered to runtime library calls

Additionally we propose appropriate passes. Similar to the accompanying dialects they will be outlined in more more detail elsewhere.

1. `add-gpu-regions`: Adding GPU Regions when high-level NDArray operations operate on GPU-allocated tensors
2. `ndarray-dist`: Propagating distributed env attributes on `NDArray` operations when data was assigned to a distributed team and add necessary ops from `Dist`
3. `dist-coalesce`: coalesce communication primitives, which reduces the number of communication calls
4. `dist-infer-elementwise-cores`: Identify intersecting loop bounds of dependent elementwise cores and annotate related operations
5. `convert-dist-to-standard`: Lower `Dist` dialect to `NDArray`, `DistRuntime` and lower dialects
6. `overlap-comm-and-compute`: make async communication overlap with local computation if possible
7. `add-comm-cache-keys`: Provide optimization potential to runtimes by assigning unique keys to instances of `DistRuntime` operations
8. `lower-distruntime-to-idtr`: lower `DistRuntime` to "Intel Distributed Tensor Runtime" library
9. `convert-ndarray-to-linalg`: lower `NDArray` to `linalg` and lower dialects
10. `convert-region-to-gpu`: use annotations from GPU regions, for example converting `memref.allocs` into `gpu.allocs`

### `NDArray` Type

Since operations are expected to execute in the same location as its input tensors, it is necessary to carry the tensor-location from the point of its allocation to the point of the operation. For this, we introduce a type which logically extends the `tensor::RankedTensorType` with a variable number of _environments_. An _environment_ can be of any attribute type. We introduce two environment attributes:

* `GPUEnvAttr`: indicates the location on a local device. It can hold arbitrary information about the device and can be as simple as a string. The exact type/syntax of device-identifiers is defined by the GPU runtime. For example, SYCL runtime will accept SYCL filter strings as defined [here](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter). In this case, requesting the first GPU device through the OpenCL backend would look like this: `#region.gpu_env<device = "opencl:gpu:0">`, attaching this to a 6x6 array of 32bit ints gives `!ndarray.ndarray<6x6xi32, #region.gpu_env<device = "opencl:gpu:0">>`
* `DistEnvAttr`: indicates the tensor is distributed. The attribute holds information necessary to correctly identify the local part of the tensor, such as local part shapes and local offsets. In addition, it holds a `team` attribute which indicates a team (of processes) among which the tensor is partitioned-distributed. The type of the `team` depends on the underlying runtime; for MPI and Intel's distributed runtime this would be a `int64`. A team which evaluates to "false" indicates a non-distributed environment but allows uniform function signatures. As an example, let's look at a global 6x6 array of 32bit ints. Assuming it locally owns the last 3 rows locally, the corresponding type would be `!ndarray.ndarray<6x6xi32, #dist.dist_env<team = 22 : i64 loffs = 3,0 lparts = 3x6>>` (here the team is identified by `22`).

The actual GPU and Dist environments are assigned by the appropriate `NDArray` creation operations (or block arguments) and will be propagated by suitable passes.

The arrays themselves are assumed to eventually lower to `memrefs`.

Notice: By default device and distribution support is disabled and so renders conventional host operations.

### `NDArray` Operations

The initial set of operations matches the requirements of the core of [array-API](https://data-apis.org/array-api/latest/API_specification/index.html). The operations in the NDArray dialect operate on `NDArray`s.

Notice: some of the operations mutate existing `NDArray`s (which makes them logically more arrays than tensors).

It constitutes an error if an operation has multiple (input and output) arguments of type `NDArrayType` and their `GPUEnvAttr` attribute is not the same on all arguments. Similarly, it is invalid to provide multiple operands of `NDArrayType` with different `team` attributes (in their `DistEnvAttr`).

#### In-place Semantics

`NDArray` operates in normal `RankedTensor` semantics with the exception of the operation `ndarray.insert_slice` and `ndarray.subview`. This is necessary to conform with array api.

* `ndarray.insert_slice` __guarantees__ an update __in-place__. Here the array/tensor behaves like a `memref`.
* In contrast to mlir's tensor semantics, `ndarray.subview` __guarantees__ to always provide a view, never a copy.

The `NDArray` dialect also has ops following tensor dialect semantics: `immutable_insert_slice`, `extract_slice`.

#### Broadcasting/Ranked Arrays

In most cases the output shape can be determined at compile time if dependent input shapes are known. In rare cases the shape of input tensor(s) needs to be known in addition to the rank. Unranked `NDArray`s are not supported.

NDArray operations follow the [broadcasting semantics](https://data-apis.org/array-api/latest/API_specification/broadcasting.html) and the [type promotion rules](https://data-apis.org/array-api/2022.12/API_specification/type_promotion.html) of the array-API.

Broadcasting happens implicitly, meaning it is the responsibility of the implementation of affected operations to perform the necessary (logical) expansion. In addition, an explicit broadcast operation is available.

#### `NDArray` Operation details

There are many operations defined in the array API and even more in the numpy API. It does not seem feasible to have one MLIR operation per numpy function (we'd get a hundred or more). The `Linalg` dialect addresses this by 'classifying' them, for example into `elementwise_unary` and `elementwise_binary` operations and having the actual per-element-operation as a parameter. On the contrary, the `TOSA` dialect decided to have have an exact 1:1 mapping for every tensor-operation (`add`, `abs`, `ceil` `arithmetic_right_shift` etc.).

The below set of operations allows expressing the full array-API and accrues from the following rules:

1. classify/group where semantically justifiable and so reduce the interface's surface. Such a operation is a generalization and allows direct/easy mapping of array-API functions to the same (parameterized) operation.
2. do not classify when parameters differ and so allow early type checking

* Array creation
  * `linspace(start, stop, n, endpoint) : (number, number, number, bool) -> ndarray.ndarray`
    * covers `arange`
  * `create(shape, value) : (shape.shape, anytype) -> ndarray.ndarray`
    * covers `empty`, `ones`, `zeros`, `full`
  * `create_like(rsh, value) : (shape.shape, anytype) -> ndarray.ndarray`
    * covers `empty_like`, `ones_like`, `zeros_like`, `full_like`
  * `eye(n_rows, n_cols, k) : (int, int, int) -> ndarray.ndarray`
  * `meshgrid(arrays) : (list) -> list`
  * `extract_triangle{$side}(t, k) : (ndarray.ndarray, int) -> ndarray.ndarray`
    * `$side = ['lower', 'upper']`
  * `delete(tensor) : (ndarray) -> void`
  * `from_dlpack(obj) : (ptr) -> ndarray.ndarray`
* Array attributes
  * `dim(t, i) : (ndarray.ndarray, Index) -> Index`
  * `shape(t) : (ndarray.ndarray) -> Variadic<Index>`
  * `size(t) : (ndarray.ndarray) -> Index`
* Data Type functions
  * `cast(t) {copy : bool} : (ndarray.ndarray) -> ndarray.ndarray`
  * `broadcast(t, shape) : (ndarray.ndarray, Variadic<Index>) -> ndarray.ndarray`
* Indexing
  * `subview(t, slice) : (ndarray.ndarray, Variadic<Index, Index, Index>) -> ndarray.ndarray`
    * returns a view
  * `extract_slice(t, slice) : (ndarray.ndarray, Variadic<Index, Index, Index>) -> ndarray.ndarray`
    * returns a copy
  * `extract_mask(t, mask) : (ndarray.ndarray, ndarray.ndarray) -> ndarray.ndarray`
    * resulting shape cannot be determined at compile time
  * `insert_slice(t, slice, val) : (ndarray.ndarray, Variadic<Index, Index, Index>, ndarray.ndarray)`
  * `immutable_insert_slice(t, slice, val) : (ndarray.ndarray, Variadic<Index, Index, Index>, ndarray.ndarray) -> ndarray.ndarray`
* Manipulation
  * `combine (tensors, axis) {$op : str} : (Variadic<NDArray>, int) -> ndarray.ndarray`
    * `$op = ['concat', 'stack']`
  * `expand_dims(t, axis) : (ndarray.ndarray, Variadic<Index>) -> ndarray.ndarray`
  * `flip(t, axis) : (ndarray.ndarray, Variadic<Index>) -> ndarray.ndarray`
  * `permute_dims(t, axis) : (ndarray.ndarray, Variadic<Index>) -> ndarray.ndarray`
  * `reshape(t, shape) {copy : bool} : (ndarray.ndarray, shape.shape, bool) -> ndarray.ndarray`
  * `roll(t, shift, axis) : (ndarray.ndarray, int, Variadic<Index>) -> ndarray.ndarray`
  * `squeeze(t, axis) : (ndarray.ndarray, Variadic<Index>) -> ndarray.ndarray`
    * Requires shaped tensor as input, number of ranks is not sufficient.
* Elementwise operations
  * `elementwise_unary_op (t) {op : i8} : (ndarray.ndarray) -> ndarray`
    * `$op = ['abs', 'acos', 'acosh', 'asin', 'bitwise_invert', 'isnan', 'isinf', 'logical_not', ...]`
  * `elementwise_binary_op (lhs, rhs) {$op: i8}: (ndarray.ndarray, ndarray.ndarray) -> ndarray`
    * `$ebop = ['add', 'sub', 'bitwise_left_shift', 'greater', 'less', 'equal', 'logical_and', ...]`
* Linear Algebra
  * `matmul(rhs, lhs) : (ndarray.ndarray, ndarray.ndarray) -> ndarray.ndarray`
  * `matrix_transpose(t) : (ndarray.ndarray) -> ndarray.ndarray`
  * `vecdot(rhs, lhs, axis) : (ndarray.ndarray, ndarray.ndarray, Index) -> ndarray.ndarray`
  * `tensor_dot(rhs, lhs, axis) : (ndarray.ndarray, ndarray.ndarray, Variadic<Index>) -> ndarray.ndarray`
* Searching
  * `find_index (t, axis) {op : str} : (ndarray.ndarray, Optional<Index>) -> ndarray.ndarray`
    * `op = ['argmax', 'argmin']`
  * `nonzero(t) : (ndarray.ndarray) -> Variadic<NDArray>`
  * `where(cond, rhs, lhs) : (scf.condition, ndarray.ndarray, ndarray.ndarray) -> ndarray.ndarray`
    * resulting shape cannot be determined at compile time
* Set Functions
  * `unique_all(t) : (ndarray.ndarray) -> [ndarray.ndarray, ndarray.ndarray, ndarray.ndarray, ndarray.ndarray]`
    * resulting shape cannot be determined at compile time
  * `unique_counts(t) : (ndarray.ndarray) -> [ndarray.ndarray, ndarray.ndarray]`
    * resulting shape cannot be determined at compile time
  * `unique_inverse(t) : (ndarray.ndarray) -> [ndarray.ndarray, ndarray.ndarray]`
    * resulting shape cannot be determined at compile time
  * `unique_values(t) : (ndarray.ndarray) -> ndarray.ndarray`
    * resulting shape cannot be determined at compile time
* Sorting Functions
  * `sort(t, descending, stable) : (ndarray.ndarray, bool, bool) -> ndarray.ndarray`
  * `argsort(t, descending, stable) : (ndarray.ndarray, bool, bool) -> ndarray.ndarray`
* Statistical Functions
  * `reduce (t, axis, correction) {op : str}: (ndarray.ndarray, Variadic<Index>, Optional<number>) -> ndarray.ndarray`
    * `op = ['max', 'min', 'mean', 'prod', 'sum', 'var', 'std']`
* Utility Functions
  * `test (t, axis) {op : str}: (ndarray.ndarray, Variadic<Index>) -> ndarray.ndarray`
    * `op = ['any', 'all']`

## Alternatives

### TOSA

We could also consider the `TOSA` dialect, either by extending it directly or lowering to it instead of to `Linalg`. The benefit of he latter seems small because it does not offer significantly richer functionality.

Extending TOSA could be a viable alternative to `NDArray`. TOSA's design requires any operation to be fundamental in the sense that it cannot be build out of others. We can add the `team` and `device` information to the TOSA tensor but we'll need to add higher-level wrapper/generation functions to establish the desired level of abstraction. This is more complicated than `NDArray` and so only useful if upstream TOSA is onboard with the idea. It is also thinkable to start with a separate dialect and merge it into `TOSA` once stabilized.

### Operation classification options

There are several aspects to consider and each approach has its pros and cons

Classification

* \+ in theory a single operation suffices and so the operation spec will be small
* \+ easier to use when coming from numba
* \- some late error checking
* \- manual dispatch needed (potentially at various places)

One Op per tensor-operation

* \+ MLIR handles dispatch
* \+ clearer syntax and documentation
* \+ more early error checking (like type checking)
* \- large spec, likely including code duplication
* \- potentially more verbose code when coming from numba/Python

In the end, the crucial question probably is if `NDArray` is to be positioned as a generic (parallel) tensor dialect or primarily as an entry point into MLIR from Python. Also, prioritization of aspects of SW design practices play a role.
