# SetSPIRVAbiAttribute Pass


The SetSPIRVAbiAttribute pass, adds a kernel attribute called spv.entry_point_abi to the kernel function. Since SPIR-V programs themselves are not enough for running workloads on GPU; a companion host application is needed to manage the resources referenced by SPIR-V programs and dispatch the workload. It is also quite possible that both those programs are written by different frond-end languages.Hence the need to add the entry point abi.
spv.entry_point_abi is a struct attribute that should be attached to the entry function. Some of the lowering passes expect this attribute to perform the lowering.

# Example

```
// -----// IR Dump Before {anonymous}::SetSPIRVAbiAttribute () //----- //
gpu.module @main_kernel {
  gpu.func @main_kernel(%arg0: memref<8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>) kernel {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = gpu.block_id  x
    %1 = memref.load %arg0[%0] : memref<8xf32>
    %2 = memref.load %arg1[%0] : memref<8xf32>
    %3 = arith.addf %1, %2 : f32
    memref.store %3, %arg2[%0] : memref<8xf32>
    gpu.return
  }
}
```

The Pass will change the IR to:

```
// -----// IR Dump After {anonymous}::SetSPIRVAbiAttribute () //----- //
gpu.module @main_kernel {
  gpu.func @main_kernel(%arg0: memref<8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>) kernel attributes {spv.entry_point_abi = #spv.entry_point_abi<>} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = gpu.block_id  x
    %1 = memref.load %arg0[%0] : memref<8xf32>
    %2 = memref.load %arg1[%0] : memref<8xf32>
    %3 = arith.addf %1, %2 : f32
    memref.store %3, %arg2[%0] : memref<8xf32>
    gpu.return
  }
}
```


As shown in the example above, the kernel attribute is added after the pass.


## Reason for this Custom Pass:

Upstream does not have a pass which does these conversions. This is a very small pass, so, maybe we can have it as a custom pass rather than upstreaming.
