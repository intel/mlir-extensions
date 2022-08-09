# RFC: Parallel Tensor Dialect (PTensor)
Frank Schlimbach, Ivan Butygin

## Summary
We propose a solution which can provide a high-level dialect for tensors and (numpy-like) tensor-operations allowing passes to create parallelism for GPU, OpenMP but also distributed memory while following the compute-follows-data concept.

In this document, we describe several dialects and passes to provide a good high-level view on the overall flow. We might want to split this into more than one document.

## Motivation
MLIR provides a few tensor dialects (like tensor and TOSA) and separate dialects/conversions for targeting devices and parallelism. More complicated flows, for example when device and host tensors exist at the same time, complicated analysis is needed to correctly implement compute-follows data. Similarly, converting to distributed parallelism, such as MPI-based, is difficult to get right with existing dialects if possible at all.
Additionally, the current architecture in Plier does not have an explicit MLIR representation of arrays/tensors and so makes it very hard to re-use its tensor functionality without coming from python/numba. Currently conversions are even relying the Python runtime. A dialect for such numpy-like tensors and -operations can accept hints for targets (like devices, MPI etc) and enable its use in libraries independent of numba and Python.

## Proposal
We propose three new Dialects:
1. __PTensor__: providing a tensor type (ptensor) and operations on such ptensors
2. __Dist__: dealing with the aspects of distributed data management, such as distributed memory alloction, GC and communication
3. __CFD__: providing utilities to facilitate compute follows data

Additionally we propose appropriate passes
1. Converting __PTensor__ operations into dialects __Linalg__, __GPU__, __CFD__ and __Dist__
2. Converting __Dist__ operations into appropriate runtime calls
3. Converting __CFD__ to __GPU__ Where?

### ptensor Type
The ptensor type extends the `mlir::tensor` with `device` and `dist` attributes. It also carries
The `device` indicates where the tensor lives, e.g. on which device.
The target device is represented as a plain string which will be forwarded to the gpu/distributed runtimes. The exact syntax of device-identifiers is defined by the GPU runtime. The default value is `device="default"`.
The `dist` attribute indicates whether the tensor is partitioned-distributed or not (boolean `UnitAttr`) and its default is `dist=false`.

### __PTensor__ Operations
The initial set of operations matches the requirements of the core of [array-API](https://data-apis.org/array-api/latest/API_specification/index.html). Notice, since we focus on compute-follows-data, only the creation functions/operations will require the `device` and `dist` attributes. Operations consuming ptensors react on the the `device` and `dist` attribute of their input.

There are many operations defined in the array API and even more in the numpy API. It does not seem feasable to have one MLIR operation per numpy function (we'd get a hundred or more). The Linalg dialect addresses this by 'classifying' them, for example into 'elementwise_unary' and 'elementwise_binary' operations and having the actual per-element-operation as a parameter. On the contrary, the TOSA dialect decided to have have an excat 1:1 mapping for every tensor-operation ('add', 'abs', 'ceil' 'arithmetic_right_shift' etc.).

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

In the end, the crucial question probably is if PTensor is to be positioned as a generic (parallel) tensor dialect or primarily as an entry point into MLIR from Python. Also, priorization of aspects of SW design practices play a role.

We suggest
* classify/group where semantically justifyable and so reduce the interface's surface
* do not classify when parameters differ and so allow early type checking

#### Operation details
* Tensor creation
  * `arange(start, stop, step, dtype, device) : (int64, int64, int64, type, str) -> ptensor.ptensor`
  * `asarray(??) : (??) -> ptensor.ptensor`
  * `create(shape, value, dtype, device) : (shape.shape, anytype, type, str) -> ptensor.ptensor`
    * covers `empty, ones, zeros, full`
  * `create_like(rsh, value, dtype, device) : (shape.shape, anytype, type, str) -> ptensor.ptensor`
    * covers `empty_like, ones_like, zeros_like, full_like`
  * `eye(n_rows, n_cols, k, dtype, device) : (int63, int64, int64, type, str) -> ptensor.ptensor`
  * `from_dlpack(obj) : (ptr) -> ptensor.ptensor`
  * `linspace(start, stop, n, dtype, device) : (number, number, number, type, str) -> ptensor.ptensor`
  * `meshgrid(arrays) : (list) -> list`
  * `extract_triangle{$side}(rhs, k) : (ptensor.ptensor, int64) -> ptensor.ptensor`
    * `$side = ['lower', 'upper']`
* Tensor attributes
  * `shape(rhs) : (ptensor.ptensor) -> shape.shape`
  * `rank(rhs) : (ptensor.ptensor) -> int64`
  * `size(rhs) : (ptensor.ptensor) -> int64`
* Data Type functions
  * `cast(rhs, dtype, do_copy) : (ptensor.ptensor, type, bool) -> ptensor.ptensor`
  * `broadcast(rhs, shape) : (ptensor.ptensor, shape.shape) -> ptensor.ptensor`
  * `result_type(inpts) : (list) -> type`
* Indexing
  * `extract_slice(rhs, slice) : (ptensor.ptensor, list) -> ptensor.ptensor`
  * `extract_mask(rhs, mask) : (ptensor.ptensor, ptensor.ptensor) -> ptensor.ptensor`
* Manipulation
  * `combine{$cop}(tensors, axis) : (list, int64) -> ptensor.ptensor`
    * `$cop = ['concat', 'stack']`
  * `expand_dims(rhs, axis) : (ptensor.ptensor, array) -> ptensor.ptensor`
  * `flip(rhs, axis) : (ptensor.ptensor, array) -> ptensor.ptensor`
  * `permute_dims(rhs, axis) : (ptensor.ptensor, array) -> ptensor.ptensor`
  * `reshape(rhs, shape, copy) : (ptensor.ptensor, shape.shape, bool) -> ptensor.ptensor`
  * `roll(rhs, shift, axis) : (ptensor.ptensor, int64, array) -> ptensor.ptensor`
  * `squeeze(rhs, axis) : (ptensor.ptensor, array) -> ptensor.ptensor`
* Elementwise operations
  * `elementwise_unary_op{$euop}(rhs) : (ptensor.ptensor) -> ptensor`
    * `$euop = ['abs', 'acos', 'acosh', 'asin', 'bitwise_invert', ...]`
  * `elementwise_unary_test{$euop}(rhs) : (ptensor.ptensor) -> ptensor`
    * `$euop = ['isnan', 'isinf', 'logical_not', ...]`
  * `elementwise_binary_op{$ebop}(lhs, rhs) : (ptensor.ptensor, ptensor.ptensor) -> ptensor`
    * `$ebop = ['add', 'sub', 'greater', 'bitwise_left_shift', ...]`
  * `elementwise_binary_test{$ebop}(lhs, rhs) : (ptensor.ptensor, ptensor.ptensor) -> ptensor`
    * `$ebop = ['greater', 'less', 'equal', 'logical_and', ...]`
* Linear Algebra
  * `matmul(rhs, lhs) : (ptensor.ptensor, ptensor.ptensor) -> ptensor.ptensor`
  * `matrix_transpose(rhs) : (ptensor.ptensor) -> ptensor.ptensor`
  * `vecdot(rhs, lhs, axis) : (ptensor.ptensor, ptensor.ptensor, int64) -> ptensor.ptensor`
  * `tensor_dot(rhs, lhs, axis) : (ptensor.ptensor, ptensor.ptensor, array) -> ptensor.ptensor`
* Searching
  * `find_index{$fop}(rhs, axis) : (ptensor.ptensor, int64, bool) -> ptensor.ptensor`
    * `$fop = ['argmax', 'argmin']`
  * `nonzero(rhs) : (ptensor.ptensor) -> tuple`
  * `where(cond, rhs, lhs) : (scf.condition, ptensor.ptensor, ptensor.ptensor) -> ptensor.ptensor`
* Set Functions
  * `unique_all(rhs) : (ptensor.ptensor) -> [ptensor.ptensor, ptensor.ptensor, ptensor.ptensor, ptensor.ptensor]`
  * `unique_counts(rhs) : (ptensor.ptensor) -> [ptensor.ptensor, ptensor.ptensor]`
  * `unique_inverse(rhs) : (ptensor.ptensor) -> [ptensor.ptensor, ptensor.ptensor]`
  * `unique_values(rhs) : (ptensor.ptensor) -> ptensor.ptensor`
* Sorting Functions
  * `sort{$sop}(rhs, descending, stable) : (ptensor.ptensor, bool, bool) -> ptensor.ptensor`
  * `argsort{$sop}(rhs, descending, stable) : (ptensor.ptensor, bool, bool) -> ptensor.ptensor`
* Statistical Functions
  * `reduce{$rop}(rhs, axis, correction) : (ptensor.ptensor, int64) -> ptensor.ptensor`
    * `$rop = ['max', 'min', 'mean', 'prod', 'sum', 'var', 'std']`
* Utility Functions
  * `test{$top}(rhs, axis) : (ptensor.ptensor, int64) -> ptensor.ptensor`
    * `$rop = ['any', 'all']`

### __Dist__ Operations
The Dist dialect provides operations dealing with tensors which are partitioned and distributed across multiple processes. The operations assume soem kind of a runtime which handles aspects like communication and partitioning.
- `init_dtensor(shape) : (ValueRange) -> (int64)`
- `fini_dtensor(dtensor_id) : (int64) -> void`
- `init_view(dtensor_id, slice) : (int64, slice) -> (int64)`
- `local_shape(dtensor_id) : (int64) -> shape.shape`
- `local_slice(dtensor_id) : (int64) -> slice`
- `copy(from_id, to_id, from_ltensor, to_ltensor) : (int64, int64, RankedTensor, RankedTensor) -> void`
- `finalize_reduce(dtensor_id, ltensor) : (int64, RankedTensor) -> void`

### __CFD__ Operations
The CFD dialect provides region operation which defines the target on which operations within the region should be executed on.

### Passes
All passes which consume `ptensor`s and -operations comply to compute-follows-data: Generation of GPU-, distribution-, parallelization- et al. operations depend on the `dist` and `device` attributes of input tensors. For example, if a tensor has the attribute "device='GPU'" then memory allocation operations must target the GPU.

#### --ptensor-gpu
This pass inserts GPU regions around all operations which are expected to execute on GPU devices. This enables compute-follows-data and allows later passes to identify what goes to non-default devices and what stays on the default target device.

It constitutes an error if an operation has multiple (input and output) arguments of type ptensor and their `device` argument is not the same on all ptensor arguments. If the `device` attribute is not the default device, a region is created and populated with the input op.

This pass needs to be executed before --lower-ptensor to be effective.

Example:
```
ptensor.add(%pt, %1): (ptensor<..., dist='GPU'>, int64) -> ptensor.ptensor<..., dist='GPU'>
```
will become
```
cfd.region{'dist'}(team) {
  cfd.region{'GPU'}() {
    ptensor.add(%pt, %1): (ptensor<..., dist='GPU'>, int64) -> ptensor.ptensor<..., dist='GPU'>
  }
}
```

#### --lower-ptensor
This pass completely lowers ptensor operations to
- __Tensor__: Tensor types will convert to plain tensors. `dist` and `device` attributes get stripped off and used to appropriately create necessary operations in __GPU__ and __Dist__ dialects.
- __Linalg__: The actual functionality will be represnted by one or more operations of the Linalg dialect.
- __Dist__: new dialect which deals with primitives to manage operations on distributed tensors. Similar to the __GPU__ dialect the primitives might lower to interaction with a runtime (library).
- utility dialects like __memref__, __shape__, __affine__, __func__ and __arith__

The ptensor type will be replaced by `RankedTensor` if if `dist==false`. Distributed tensors will be replaced by 2 values: The `RankedTensor` and an `int64` representing the a unique identifier of the 'global' tensor entity. The latter is used in for interacting with the __Dist__ dialect.

Combining conversions to multiple dialects keeps analysis simple and allows effective code generation. Some operations require interleaving operations from various dialects. Separate passes would become unnecessarily complicated because they need to understand the context (which we know while we convert ptensor). For instance, distributed `arange` requires various interactions with __Dist__ interleaving process-local operation in __Linalg__; a simple wrap of the local operation is not feasible:
```
%res = ptensor.arange(%1, %2, %3){dist} : (int64, int64, int64)  -> ptensor.ptensor<..., dist=true>
```
would decompose into the following high-level ops:
```
%start = $1
%stop = $compute_stop_index(%1, %2, %3)
%step = $3
%gshape = $compute_global_shape(%1, %2, %3)
%did = dist.init_dtensor(%gshape)
%lshape = dist.local_shape(%did)
%lslice = dist.local_slice(%did)
%ltensor = linalg.init_tensor(%lshape)
%rtensor = linalg.generic($compute_arange, %lslice, %lshape, %ltensor)
%res = %rtensor, %did
```

Complicated computation kernels might be simply replaced with a an appropriate library call to MKL or alike.

Possible parameters to this pass could be flags to ignore `dist` attributes.

#### --dist-to-intel-runtime
FIXME: name and flow should be similar to what happens on the GPU side
This pass converts all operations from __Dist__ into appropriate calls to the "Intel distributed runtime for MLIR (C)". This library provides the necessary hooks to deal with aspects like data/shape partitioning, tensor registration, GC, communication and more.

## Alternatives
We could also consider the `TOSA` dialect, either by extending it directly or lowering to it instead of to Linalg. The benefit of the latter is unclear and the first seems reasonable only if we can get the MLIR community's/TOSA maintainers' buy in.

## Remarks
- most of the __Dist__-unrelated work is planned independent of this in some form or another (elminating dependence on Python runtime, streamlining compile pipeline) or can be copied from existing passes in Plier. PTensor and its passes will provide a structured and re-usable context for them.
- The detailed API of the distributed runtime is to be defined.

## Questions
- target: mapping queue <-> device
- how are differences in HW handled in GPU pipeline/SPIR-V?
