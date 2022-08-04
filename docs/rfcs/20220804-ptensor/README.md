# RFC: Parallel Tensor Dialect (PTensor)
## Summary
We propose a solution which can provide a high-level dialect for tensors and (numpy-like) tensor-operations allowing passes to create parallelism for GPU, OpenMP but also distributed memory while following the compute-follows-data concept.

## Motivation
MLIR provides a few tensor dialects (like tensor and TOSA) and separate dialects/conversions for targeting devices and parallelism. More complicated flows, for example when device and host tensors exist at the same time, complicated analysis is needed to correctly implement compute-follows data. Similarly, converting to distributed parallelism, such as MPI-based, is difficult to get right with existing dialects if possible at all.
Additionally, the current architecture in Plier does not have an explicit MLIR representation of arrays/tensors and so makes it very hard to re-use its tensor functionality without coming from python/numba. Currently conversions are even relying the Python runtime. A dialect for such numpy-like tensors and -operations can accept hints for targets (like devices, MPI etc) and enable its use in libraries independent of numba and Python.

## Proposal
We propose two new Dialects:
1. __PTensor__: providing a tensor type (ptensor) and operations on such ptensors
2. __Dist__: dealing with the aspects of distributed data management, such as distributed memory alloction, GC and communication
3. FIXME: I hope we do not need an extra GPU dialect and can use what already exists

Additionally we propose appropriate passes
1. Converting __PTensor__ operations into dialects __Linalg__, __GPU__ and __Dist__
2. Converting __Dist__ operations into appropriate runtime calls

### ptensor Type
The ptensor type extends the `mlir::tensor` with `device` and `dist` attributes.
The `device` indicates where the tensor lives, e.g. on which device.
The target device is represented as a plain string which will be forwarded to the gpu/distributed runtimes. The exact syntax of device-identifiers is defined by the GPU runtime. The default value is `device="default"`.
The `dist` attribute indicates whether the tensor is partitioned-distributed or not (boolean `UnitAttr`) and its default is `dist=false`.

### ptensor Operations
The initial set of operations matches the requirements of the core of [array-API](https://data-apis.org/array-api/latest/API_specification/index.html). Notice, since we focus in compute-follows-data, only the creation functions/operations will require the `device` and `dist` attributes. Operations consuming ptensors react on the the `device` and `dist` attribute of their input.

### dist Operations
- `alloc_dtensor(shape) : (ValueRange) -> (int64, RankedTensor)`
- `dealloc_dtensor(dtensor) : (int64) -> void`
- `create_view(dtensor, slice) : (int64, slice) -> (int64, RankedTensor)`
- `local_shape(dtensor) : (int64) -> shape.shape`
- `finalize_reduce(dtensor) : (int64) -> void`
- `copy_view(from, to) : (int64, int64) -> void`

### Passes
All passes which consume `ptensor`s and -operations comply to compute-follows-data: Generation of GPU-, distribution-, parallelization- et al. operations depend on the `dist` and `device` attributes of input tensors. For example, if a tensor has the attribute "device='GPU'" then memory allocation operations must target the GPU.

#### --lower-ptensor
This pass completely lowers ptensor operations to
- __Tensor__: Tensor types will convert to plain tensors. `dist` and `device` attributes get stripped off and used to appropriately create necessary operations in __GPU__ and __Dist__ dialects.
- __Linalg__: The actual functionality will be represnted by one or more operations of the Linalg dialect.
- __GPU__: highest level GPU dialect to indicate what needs to go to GPUs
- __Dist__: new dialect which deals with primitives to manage operations on distributed tensors. Similar to the __GPU__ dialect the primitives might lower to interaction with a runtime (library).
- utility dialects like __shape__, __affine__, __func__ and __arith__

Combining conversions to multiple dialects keeps analysis simple and allows effective code generation. Some operations require interleaving operations from various dialects. Separate passes would become unnecessarily complicated because they need to understand the context (which we know while we convert ptensor). For instance, distributed `arange` requires various interactions with __Dist__ interleaving process-local operation in __Linalg__; a simple wrap of the local operation is not feasible.

Complicated computation kernels might be simply replaced with a an appropriate library call to MKL or alike.

Possible parameters to this pass could be flags to ignore `dist` and/or `device` attributes.

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
