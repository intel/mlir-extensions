# RFC: Parallel Tensor Dialect (PTensor)
Frank Schlimbach, Ivan Butygin, Diptorup Deb

## Summary
We propose a solution which can provide a high-level dialect for tensors and (numpy-like) tensor-operations allowing passes to create parallelism for GPU, OpenMP but also distributed memory while following the compute-follows-data concept.

In this document, we describe several dialects and passes to provide a good high-level view on the overall flow. We might want to split this into more than one document.

## Motivation
MLIR provides a few tensor dialects (like tensor and TOSA) and separate dialects/conversions for targeting devices and parallelism. An annotated type is needed to keep the required analysis in the compiler pipeline to an acceptable extent. Even moderately complicated flows, for example when device and host tensors exist at the same time, complicated analysis is needed to correctly implement compute-follows data. This is particularly true when converting to distributed parallelism, such as MPI-based; this is difficult to get right with existing dialects if possible at all.

Additionally, the current architecture in Plier does not have an explicit MLIR representation of arrays/tensors and so makes it very hard to re-use its tensor functionality without coming from python/numba. Currently conversions are even relying the Python runtime. A dialect for such numpy-like tensors and -operations can accept hints for targets (like devices, MPI etc) and enable its use in libraries independent of numba and Python. One example of such a use case is a distributed and JIT-compiler based numpy implementation.

## Proposal
The main purpose of this proposal is to enable compute-follows-data for a large variety of platforms. We strive to enable it not only different kinds of shared memory systems (like CPUs, GPUs, FPGAs, ...) but also (potentially heterogenous) distributed memory systems. In its core, compute-follows-data means that operations on an tensor will happen where the tensor was allocated.

We propose two new Dialects and one dialect extension:
1. __PTensor__ dialect: providing a tensor type (ptensor) and operations on tensors
2. __Dist__ dialect: dealing with the aspects of distributed data management, such as distributed memory allocation, GC and communication
3. __intel_sycl.device_region__ operation in __intel_sycl__ dialect (RFC #291)

Additionally we propose appropriate passes
1. Converting __PTensor__ operations into dialects __Linalg__, __GPU__, __intel_sycl__ and __Dist__
2. Converting __Dist__ operations into appropriate runtime calls
3. Converting __intel-sycl.device_region__ to appropriate runtime calls.

### ptensor Type
Since operations are expected to execute in the same location as its input tensors, it is necessary to carry the tensor-location from the point of its allocation to the point of the operation. For this, we introduce a type which logically extends the `mlir::tensor` type with the necessary location information. The location information includes:
* the `device`: The optional `device` indicates where the tensor lives, e.g. on which device. The target device is represented as a generic type which will be forwarded to the gpu/distributed runtimes. The exact type/syntax of device-identifiers is defined by the GPU runtime. For example, SYCL runtime will accept SYCL filter strings as defined [here](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter). The default `NoneType` which disables device support.
* a distributed `team`:  The optional `team` indicates a team (of processes) among which the tensor is partitioned-distributed. The type of the `team` depends on the underlying runtime; for MPI and Intel's distributed runtime this would be a `int64`. It defaults to `NoneType` which disables support for distributed operation.
* a globally unique id `gid`: `int64`

The tensors themselves are assumed to eventually lower to memrefs.

Notice: By default device and distribution support is disabled and so renders conventional host operations.

### __PTensor__ Operations
The initial set of operations matches the requirements of the core of [array-API](https://data-apis.org/array-api/latest/API_specification/index.html). Notice, since we focus on compute-follows-data, only the creation functions/operations will require the `device` and `team` attributes. The operations in the PTensor dialect operate on ptensors. To allow operations on standard tensors and memrefs the PTensor dialect provides the operation `from_ranked` to convert MemRefs and RankedTensors to ptensors with default `device` and `team`.

Notice: some of the operations mutate existing ptensors.

#### Broadcasting/Ranked Tensors
PTensor operates on ranked tensors. In rare cases the shape of input tensor(s) needs to be known as well. Unranked tensors are not supported.

PTensor operations follow the [broadcasting semantics of the array-API](https://data-apis.org/array-api/latest/API_specification/broadcasting.html).

Broadcasting happens implicitly, it is the responsibility of the implementation of affected operations to perform the necessary (logical) expansion. In addition, an explicit broadcast operation is available.

#### Operation details
There are many operations defined in the array API and even more in the numpy API. It does not seem feasible to have one MLIR operation per numpy function (we'd get a hundred or more). The Linalg dialect addresses this by 'classifying' them, for example into 'elementwise_unary' and 'elementwise_binary' operations and having the actual per-element-operation as a parameter. On the contrary, the TOSA dialect decided to have have an exact 1:1 mapping for every tensor-operation ('add', 'abs', 'ceil' 'arithmetic_right_shift' etc.).

The below set of operations accrues from the following rules:
1. classify/group where semantically justifiable and so reduce the interface's surface
2. do not classify when parameters differ and so allow early type checking

* Tensor creation
  * `arange(start, stop, step, dtype, device, team) : (int, int, int, type, str, int) -> ptensor.ptensor`
  * `create(shape, value, dtype, device, team) : (shape.shape, anytype, type, str, int) -> ptensor.ptensor`
    * covers `empty, ones, zeros, full`
  * `create_like(rsh, value, dtype, device, team) : (shape.shape, anytype, type, str, int) -> ptensor.ptensor`
    * covers `empty_like, ones_like, zeros_like, full_like`
  * `eye(n_rows, n_cols, k, dtype, device, team) : (int, int, int, type, str, int) -> ptensor.ptensor`
  * `linspace(start, stop, n, dtype, device, team) : (number, number, number, type, str, int) -> ptensor.ptensor`
  * `meshgrid(arrays) : (list) -> list`
  * `extract_triangle{$side}(rhs, k) : (ptensor.ptensor, int) -> ptensor.ptensor`
    * `$side = ['lower', 'upper']`
  * `delete(tensor) : (ptensor) -> void`
  * `from_dlpack(obj) : (ptr) -> ptensor.ptensor`
  * `from_ranked(ranked) : (Memref|RankedTensor) -> ptensor.ptensor`
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
  * `combine{$cop}(tensors, axis) : (list, int) -> ptensor.ptensor`
    * `$cop = ['concat', 'stack']`
  * `expand_dims(rhs, axis) : (ptensor.ptensor, array) -> ptensor.ptensor`
  * `flip(rhs, axis) : (ptensor.ptensor, array) -> ptensor.ptensor`
  * `permute_dims(rhs, axis) : (ptensor.ptensor, array) -> ptensor.ptensor`
  * `reshape(rhs, shape, copy) : (ptensor.ptensor, shape.shape, bool) -> ptensor.ptensor`
  * `roll(rhs, shift, axis) : (ptensor.ptensor, int, array) -> ptensor.ptensor`
  * `squeeze(rhs, axis) : (ptensor.ptensor, array) -> ptensor.ptensor` Requires shaped tensor as input, number of ranks is not sufficient.
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
  * `vecdot(rhs, lhs, axis) : (ptensor.ptensor, ptensor.ptensor, int) -> ptensor.ptensor`
  * `tensor_dot(rhs, lhs, axis) : (ptensor.ptensor, ptensor.ptensor, array) -> ptensor.ptensor`
* Searching
  * `find_index{$fop}(rhs, axis) : (ptensor.ptensor, int, bool) -> ptensor.ptensor`
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
  * `reduce{$rop}(rhs, axis, correction) : (ptensor.ptensor, int) -> ptensor.ptensor`
    * `$rop = ['max', 'min', 'mean', 'prod', 'sum', 'var', 'std']`
* Utility Functions
  * `test{$top}(rhs, axis) : (ptensor.ptensor, int) -> ptensor.ptensor`
    * `$rop = ['any', 'all']`

### __Dist__ Operations
The Dist dialect provides operations dealing with tensors which are partitioned and distributed across multiple processes. The operations assume some kind of a runtime which handles aspects like communication and partitioning.
- `init_dtensor(team, shape) : (int, ValueRange) -> (int64)`
- `fini_dtensor(team, dtensor_id) : (int, int) -> void`
- `init_view(team, dtensor_id, slice) : (int, int, slice) -> (int64)`
- `local_shape(team, dtensor_id) : (int, int) -> shape.shape`
- `local_slice(team, dtensor_id) : (int, int) -> slice`
- `copy(team, from_id, to_id, from_ltensor, to_ltensor) : (int, int, int, RankedTensor, RankedTensor) -> void`
- `finalize_reduce(team, dtensor_id, ltensor) : (int, int, RankedTensor) -> void`

For details watch out for a separate RFC.

### __intel_sycl.gpu-region__
The `intel_sycl.device_region` operation defines the target on which operations within the region should be executed on. The region carries a parameter defining the concrete target device. The parameter meets the same requirements as the `device` attribute of a `ptensor.ptensor`. The main purpose of this region is 1. to provide an easy way for lower passes to comply to compute-follows-data operation and 2. to shared runtime information among operations on devices.

A `intel_sycl.device_region` must not be nested within another `intel_sycl.device_region`.

Example
```
intel_sycl.device_region('level_zero:gpu:1') {
  tensor.add(%pt1, %pt2): (tensor<...>, tensor<...>) -> tensor.tensor<...>
}
```

### Passes
All passes which consume `ptensor`s and -operations comply to compute-follows-data: Generation of GPU-, distribution-, parallelization- et al. operations depend on the `team` and `device` attributes of input tensors. For example, if a tensor has the attribute "device='GPU'" then memory allocation operations must target the GPU.

#### --lower-ptensor
This pass completely lowers ptensor operations to
- __Tensor__: `ptensor.ptensor` will be type-converted to `tuple(tensor, str, int64, int64)` representing the tensor, the device, the team and a globally unique id.
  - the tuple elements `team`, `gid` and `device` are used to appropriately create necessary operations in __intel_sycl__ and __Dist__ dialects
- __Linalg__: The actual functionality will be represented by one or more operations of the Linalg dialect.
- __intel_sycl__: Appropriate `intel_sycl.device_region` will be put around operations which have inputs of type `ptensor.ptensor` with a non-null `device` attribute.
- __Dist__: new dialect which deals with primitives to manage operations on distributed tensors. Operations from the `team` dialect will generated if input tensors have non-none `team` attributes.
- utility dialects like __memref__, __shape__, __affine__, __func__ and __arith__

Combining conversions to multiple dialects keeps analysis simple and allows effective code generation. Some operations require interleaving operations from various dialects. Separate passes would become unnecessarily complicated because they need to understand the context (which we know while we convert ptensor). For instance, distributed `arange` requires various interactions with __Dist__ interleaving process-local operation in __Linalg__; a simple wrap of the local operation is not feasible.

It constitutes an error if an operation has multiple (input and output) arguments of type ptensor and their `device` attribute is not the same on all ptensor arguments. If the `device` attribute is not the default device, a region is created and populated with the input op.

Similarly, it constitutes an error if an operation has multiple (input and output) arguments of type ptensor and their `team` attribute is not the same on all ptensor arguments.

Complicated computation kernels might be simply replaced with a an appropriate library call to MKL or alike.

Possible parameters to this pass could be flags to ignore `team` attributes.

##### Example:
```
%res = ptensor.arange(%1, %2, %3, device, team) : (int64, int64, int64, str, int64)  -> ptensor.ptensor<...>
```

If device==null and team==none this would decompose into the following high-level ops:
```
%start = $1
%stop = $compute_stop_index(%1, %2, %3)
%step = $3
%gshape = $compute_global_shape(%1, %2, %3)
%ltensor = linalg.init_tensor(%gshape)
%rtensor = linalg.generic($compute_arange, %lslice, %lshape, %ltensor)
%res = %rtensor, %device, %team, None
```

If device==null but team!=none this would decompose into the following high-level ops:
```
%start = $1
%stop = $compute_stop_index(%1, %2, %3)
%step = $3
%gshape = $compute_global_shape(%1, %2, %3)
%gid = dist.init_dtensor(%gshape)
%lshape = dist.local_shape(%gid)
%lslice = dist.local_slice(%gid)
%ltensor = linalg.init_tensor(%lshape)
%rtensor = linalg.generic($compute_arange, %lslice, %lshape, %ltensor)
%res = %rtensor, %device, %team, %gid
```

If device!=null but team==none this would decompose into the following
```
%start = $1
%stop = $compute_stop_index(%1, %2, %3)
%step = $3
%gshape = $compute_global_shape(%1, %2, %3)
%gslice = slice($start, $stop, $step)
intel_sycl.device_region(%device) {
  %ltensor = linalg.init_tensor(%gshape)
  %rtensor = linalg.generic($compute_arange, %gslice, %gshape, %ltensor)
}
%res = %rtensor, %device, %team, None
```

If device!=null and team!=none this would become something like:
```
%start = $1
%stop = $compute_stop_index(%1, %2, %3)
%step = $3
%gshape = $compute_global_shape(%1, %2, %3)
%gid = dist.init_dtensor(%gshape)
%lshape = dist.local_shape(%gid)
%lslice = dist.local_slice(%gid)
intel_sycl.device_region(%device) {
  %ltensor = linalg.init_tensor(%lshape)
  %rtensor = linalg.generic($compute_arange, %lslice, %lshape, %ltensor)
}
%res = %rtensor, %device, %team, %gid
```

#### --dist-to-intel-runtime
FIXME: name and flow should be similar to what happens on the GPU side
This pass converts all operations from __Dist__ into appropriate calls to the "Intel distributed runtime for MLIR (TM)". This library provides the necessary hooks to deal with aspects like data/shape partitioning, tensor registration, GC, communication and more.

## Alternatives
### TOSA
We could also consider the `TOSA` dialect, either by extending it directly or lowering to it instead of to Linalg. The benefit of he latter seems small because it does not offer significantly richer functionality.

Extending TOSA could be a viable alternative to PTensor. TOSA's design requires any operation to be fundamental in the sense that it cannot be build out of others. We can add the `team` and `device` information to the TOSA tensor but we'll need to add higher-level wrapper/generation functions to establish the desired level of abstraction. This is more complicated than PTensor and so only useful if upstream TOSA is onboard with the idea. It is also thinkable to start with a separate dialect and merge it into TOSA once stabilized.

The `intel_sycl.device_region` operation could also live in the PTensor dialect or in a separate dialect. A dialect with only one operation doesn't seem justifyable. The downside of having it in PTensor is that we would not be able to fully convert PTensor with `--ower-ptensor` because we need to keep the region for lower dialects which need to detect the region. Having the region in the dialect which actually uses it seems most apprpriate.

### Wrapper/generator functions
A generator function for creating distributed operations like the above `arange` example is possible as long as the information about the distributed aspect of the input tensor(s) is available. This could be handled by a dialect providing a dist-region and and so would not require a new tensor dialect. However, it is unclear how operations on the created tensors can be wrapped with such a dist-region without a type which contains the appropriate information. In particular interchanging distributed- (and maybe even device-)tensors across function boundaries requires additional information embeded in the type/argument. The same is true for device aspect. Basically the ptensor type removes the analysis burden if compute-follows-data is desired.

Is the ptensor-type-less option possible with reasonable effort?

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

In the end, the crucial question probably is if PTensor is to be positioned as a generic (parallel) tensor dialect or primarily as an entry point into MLIR from Python. Also, priorization of aspects of SW design practices play a role.

## Remarks
- most of the __Dist__-unrelated work is planned independent of this in some form or another (elminating dependence on Python runtime, streamlining compile pipeline) or can be copied from existing passes in Plier. PTensor and its passes will provide a structured and re-usable context for them.
- The detailed API the distributed runtime is to be defined, possibly in an iterative process during implementation.

## Other Questions
- target: mapping queue <-> device
- how are differences in HW handled in GPU pipeline/SPIR-V?
