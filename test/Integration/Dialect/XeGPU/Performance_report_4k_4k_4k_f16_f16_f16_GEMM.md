# 4Kx4Kx4K GEMM Performance Study

### Machine-specific information:
-----------------------------
**Machine:** `BMG B580`

**EU count:** `320`

**L3 cache size:** `18874368 Byte (18 MB)`

### Workload-specific information:
-----------------------------
**Input range:** Random number between `0.0` and `1.0`

**Barrier settings:** No barrier inside the K loop.

**Workgroup Tile Size :** `256x256` `C` tile per worgroup

**Subgroup Tile Size:** `32x64` `C` tile per-subgroup

**K-stepping:** `32`

**Subgroup Size:** `16`

**We use constant C-tile initialized to `0.0` for all our experiments (i.e., no load for C matrix).**

**Prefetch Strategy:** Co-operative prefetch

### IMEX tests used for our analysis:
--------------------------------

- Lane level 4K GEMM: https://github.com/intel/mlir-extensions/blob/main/test/Integration/Dialect/XeGPU/SIMT/gemm_4kx4kx4k_f16_f16_f16.mlir
- Subgroup level 4K GEMM: https://github.com/intel/mlir-extensions/blob/main/test/Integration/Dialect/XeGPU/SG/gemm_4kx4kx4k_f16_f16_f16.mlir
- Workgroup level 4K GEMM: https://github.com/intel/mlir-extensions/blob/main/test/Integration/Dialect/XeGPU/WG/gemm_4kx4kx4k_f16_f16_f16.mlir

### Profiling strategy

We used the Level-Zero event based profiling strategy available in IMEX. We used 100 warm-up runs and 100 profiling runs (default settings in IMEX for profiling) with cache flushing enabled. To enable these settings we used the following environment variables:

```
export IMEX_ENABLE_PROFILING=1
export IMEX_ENABLE_CACHE_FLUSHING=1
```

The actual command used to run the tests are available in the test files `RUN` command.

### Experimentation criteria:
------------------------

We perform this study to observe the code-generation prowess of the MLIR pipeline on Intel GPUs. We utilize the features offered by the Intel MLIR dialects `xegpu, xevm`, other upstream dialects, transformation, and conversion passes. Through this study, we wanted to understand the current capability of the MLIR path in reaching the peak performance. Therefore we experimented with all 3-levels of `XeGPU` kernels:
- [**Lane level**](#lane-level-implementation:) (gives us the most freedom, but need to write the most code)
- **Subgroup level** (subgroup level kernel implementation, need to write much less code, but may loose some performance in abstraction)
- **Workgroup level** (workgroup level kernel, least amount of code, more chance of performance loss in abstraction)

We also played with different loading and prefetch strategies that are used by other frameworks (e.g., OneDNN, Triton). The co-operative prefetch strategy is explained at the end of this report.


## Lane-level implementation:

We experimented with 3 different tuning mechanisms at lane-level:
- ### Load strategy:
  When loading data from the GPU memory to register, there are two things that we need to be aware of,
    - getting the maximum throughput &
    - minimizing the effect stalls due to cache misses

  Now, these two optimization criteria may sometime be contradictory to each other. The reason is, to get the maximum throughput one would load the maximum possible data that the hardware support in one go (i.e., 2KB or `32x32xf16` tile in B580). However, loading these large blocks increases the chance of cache misses as opposed to smaller load (i.e., `8x16xf16` or 256B). It also reduces the chance of doing software pipelining to hide the load latency. However, the problem is, too small of a load seriously affects the throughput of the GPU (i.e., one needs 8 load instructions as opposed to 1 load instructions to load the same 2KB data). Therefore, finding a middle ground may often lead to good performance. For example, in XeTLA, for GEMM, they don't load the maximum possible `2KB`/`32x32xf16`, rather, `1KB`/`16x32xf16`. In our experimentation, we use these two loading strategies:
  - Large Load: loads the largest possible tile that the hardware supports. For example, in B580, we are able to load a total of `32x32xf16` tile in one go (it essentially creates a `32x16xf16` load with `array_length = 2`).
  - XeTLA like smaller load: In this strategy, we don't load the largest possible size, but a slightly smaller size. For example, in B580, we are loading a total of `16x32xf16` tile in one go (it essentially creates a `16x16xf16` load with `array_length = 2`).
- ### Cache controls:
  Cache controls are caching mechanism used at different levels of caches for certain instructions such as prefetch, load, and store. Prefetch and load are controlled by same cache controls (`Uncached, Cached, Streaming, InvalidateAfterRead, ConstCached`); store has separate cache controls (`Uncached, WriteThrough, WriteBack, Streaming`). A system usually have default values for load/prefetch and store cache controls. However, for specific use cases, user can pass external cache control hints, these passed cache controls will overrides the default values. This allows user to use different cache controls for different use cases. For example, for `gemm` cases, having `cached` load/prefetch cache control value for both `L1`, and `L3` proved to be useful. So passing these explicitly ensures that we are using these values irrespective of whatever is the default setting in that system (default cache control values may differ across different machines). This provides a stable performance irrespective of the system settings. We experiment with both:
  - Default cache controls
  - User-provided cache controls
- ### Prefetch:
  Prefetching allows one to bring data closer to registers before they are being loaded. Using this strategy, we can bring data to `L1`, before they are being loaded. That way, the load would have low-latency due to a cache hit `L1`, rather than a miss in `L1` and `L3` if the data was not prefetched and had to be loaded from the memory directly. [Cache-controls](#cache-controls) decides at what levels the data will be cached (i.e., cache control value `cached` in `L1` and `L3`, would mean that the data will be cached in both `L1` and `L3`).

  However, we need to be careful to not prefetch too much since the caches are scarce resources. Therefore, if we prefetch too much and too early, by the time a load is done on that piece of data, it may already be evicted. Therefore finding a balance as to how to prefetch and how much to prefetch is very important. For the how question, we use [co-operative prefetch strategy](#co-operative-prefetch), as for the 'how much' question, we experiment with a few values:
  - No prefetch
  - 1-stage prefetch: prefetch data 1-iteration (of k-loop) before the actual load.
  - 3-stage prefetch: prefetch data 3-iteration (of k-loop) before the actual load.


Now, let's look at the results of our tuning. We start with tuning the [prefetch](#prefetch) first. For load strategy we use, Large load, and for Cache controls we use the default system settings.

### Impact of prefetch:
------------------------

We try 3 different prefetch settings:
- No prefetch
- 1-stage prefetch
- 3-stage prefetch


**Table 1: Prefetch Performance Comparison**
| Metric  | No Prefetch  | 1-Stage Prefetch | 3-Stage Prefetch |
| ------- | ------------ | ---------------- | ---------------- |
| Max     | 66.08 TFLOPS | 92.93 TFLOPS     | 93.89 TFLOPS     |
| Min     | 64.34 TFLOPS | 86.78 TFLOPS     | 86.60 TFLOPS     |
| Avg     | 65.12 TFLOPS | 89.43 TFLOPS     | 89.86 TFLOPS     |
| Median  | 65.09 TFLOPS | 89.27 TFLOPS     | 89.66 TFLOPS     |
| Std Dev | 0.34 TFLOPS  | 1.43 TFLOPS      | 1.76 TFLOPS      |


Table 1, shows the impact of prefetch. As we can see, prefetch improves performance significantly. Let's take average result for comparison; it improves the performance by `~37.33%` for 1-stage prefetch, and `~37.99%` for the 3-stage prefetch. This shows just how much prefetch is important for the GEMM performance in BMG. It also shows that 1-stage prefetch is enough for 4K GEMM. As doing extra prefetch does not really improve performance as much (less than 1%).



### Impact of cache controls:
-------------------------------

Now that we established the impact of prefetch, let's move our focus on cache controls. In this experiments, we want to see the impact of cache controls in performance. We are using two different cache controls:
- Default cache controls (system default)
- User-provided cache controls (cache control values used by other frameworks like triton)

For user provided cache control values, for `load/prefetch`, we use `cached` for both `L1` and `L3`. For `store`, we use `writeback` for both `L1` and `L3`.

**Table 2: User provided cache control values**
| Operation     | L1        | L3        |
| ------------- | --------- | --------- |
| Load/Prefetch | Cached    | Cached    |
| Store         | Writeback | Writeback |

Now, both `load` and `prefetch` uses the same cache control values. Hence, to get the impact of user-provided cache control over the default settings, we need two different experiments.

#### Impact of user-provided cache controls in `load`:
To understand the impact of user-provided cache controls in `load` operation (i.e., 2d block load operation), we have to take out the impact of prefetch. Hence, we use the [no-prefetch](#no-prefetch) version with both default and user-provided cache control values. Table 3 shows the results. As we can see there is virtually no difference. This leads us to believe that the default cache controls in this system is similar to the user-provided ones.

**Table 3: Impact of user-provided cache controls in `load`**
| Metric  | Default Cache Control | User-Provided Cache Control |
| ------- | --------------------- | --------------------------- |
| Max     | 66.08 TFLOPS          | 65.85 TFLOPS                |
| Min     | 64.34 TFLOPS          | 64.25 TFLOPS                |
| Avg     | 65.12 TFLOPS          | 65.00 TFLOPS                |
| Median  | 65.09 TFLOPS          | 65.01 TFLOPS                |
| Std Dev | 0.34 TFLOPS           | 0.34 TFLOPS                 |


#### Impact of user-provided cache controls in `prefetch`:
To understand the impact of user-provided cache controls in `prefetch` operation, we use the 1-stage prefetch version with both default and user-provided cache control values. We only change the `prefetch` cache controls, in other words, the user-provided cache control only affects the `prefetch` operations. Given what we saw in our previous experiment with `load`, we should see similar numbers with `prefetch` as well, given that `load` and `prefetch` uses the same cache control values. And we do see similar numbers as shown in Table 4.

**Table 4: Impact of user-provided cache controls in `prefetch`**
| Metric  | Default Cache Control | User-Provided Cache Control |
| ------- | --------------------- | --------------------------- |
| Max     | 92.93 TFLOPS          | 92.52 TFLOPS                |
| Min     | 86.78 TFLOPS          | 86.78 TFLOPS                |
| Avg     | 89.43 TFLOPS          | 89.44 TFLOPS                |
| Median  | 89.27 TFLOPS          | 89.39 TFLOPS                |
| Std Dev | 1.43 TFLOPS           | 1.40 TFLOPS                 |


Therefore, it shows that the system default and user-provided values for load/prefetch cache control is same in our experimental settings. We saw similar result with `store` instruction as well.

**Please note that, if you are doing this experiment in PVC, PVC does not have a default cache control for `prefetch`. In other words, for PVC, the hardware expects user-provided cache controls, if default is used, the prefetch instructions are simply ignored. For `load` and `store`, default cache control works in PVC.**


### Impact of Load:
--------------------
To understand the impact of the different load strategies described in the [Load Strategy](#load-strategy) section, we take the best performing large load version (large loads + 3-stage prefetch + default cache controls), compare it with the similarly set-up (input range, similar barrier settings, similar cache controls) smaller Xe-TLA like load-version.
- Large Load version: loads 2KB or `32x32xf16` tile (`32x16xf16` load with `array_length = 2`).
- XeTLA like load version: loads 1KB or `16x32xf16` tile (`16x16xf16` load with `array_length = 2`)

**Table 5: Impact Load Strategies**

| Metric  | Large Load   | XeTLA Like Load |
| ------- | ------------ | --------------- |
| Max     | 93.89 TFLOPS | 93.36 TFLOPS    |
| Min     | 86.60 TFLOPS | 86.28 TFLOPS    |
| Avg     | 89.86 TFLOPS | 89.38 TFLOPS    |
| Median  | 89.66 TFLOPS | 89.39 TFLOPS    |
| Std Dev | 1.76 TFLOPS  | 1.74 TFLOPS     |

Table 5 shows the result. As we can see there is virtually no difference in performance. Therefore, for 4K GEMM in B580, we can stick to largest possible loads as it leads to simpler code-generation for MLIR stack (only optimize for size, not multi-objective optimization).




Lane-level vs Subgroup-level:
==============================
We take the best performing lane-level implementation (large load + 3-stage prefetch + default cache control), compare it against the similarly set-up (input range, similar barrier settings, similar cache controls) Subgroup-level version. It shows the impact of Hand-coded lane-level implementation vs. the lowered from subgroup implementation. This is to help us figure out where we can improve in our lowering.

**Table 6: Lane-level vs Subgroup-level**

| Metric  | Lane Level   | Subgroup Level |
| ------- | ------------ | -------------- |
| Max     | 93.89 TFLOPS | 93.05 TFLOPS   |
| Min     | 86.60 TFLOPS | 85.18 TFLOPS   |
| Avg     | 89.86 TFLOPS | 88.85 TFLOPS   |
| Median  | 89.66 TFLOPS | 88.74 TFLOPS   |
| Std Dev | 1.76 TFLOPS  | 1.94 TFLOPS    |


Table 6 shows the result. The performance is on par with Lane-level implementation. We also checked the SPIR-V and assembly generated by the SG-level implementation. The assembly and SPIR-V is very similar to Lane-level implementation. There are very minor difference in some ALU instructions (address computation); otherwise, very similar to the Lane-level implementation. So it seems, performance-wise, our lowerings from subgroup and downwards are working well (at least for this gemm case).

Lane-level vs Workgroup-level:
==============================
We take the best performing lane-level implementation (large load + 3-stage prefetch + default cache control), compare it against the similarly set-up (input range, similar barrier settings, similar cache controls) Workgroup-level version. It shows the impact of Hand-coded lane-level implementation vs. the lowered from workgroup implementation. This is to help us figure out where we can improve in our lowering. We would like to be close in performance in this two version. However, as we can see in Table 7, we are about ~7% away from the lane-level performance.

**Table 7: Lane-level vs Workgroup-level**

| Metric  | Lane Level   | Workgroup Level |
| ------- | ------------ | --------------- |
| Max     | 93.89 TFLOPS | 87.99 TFLOPS    |
| Min     | 86.60 TFLOPS | 79.54 TFLOPS    |
| Avg     | 89.86 TFLOPS | 83.49 TFLOPS    |
| Median  | 89.66 TFLOPS | 83.40 TFLOPS    |
| Std Dev | 1.76 TFLOPS  | 2.19 TFLOPS     |

The question is why is this performance gap:
After investigating the generated codes (MLIR IR), one of the major issues I found is that the generated load, store, and prefetch instructions are not optimized to maximize the use of the hardware. After XeGPUBlocking pass, we generate DPAS sized loads, prefetches, and stores. This causes the under-utilization of our load/prefetch and store engines. We can support larger load, prefetch, store sizes. The issue seems to be originating from our lack of support of `array_length`. We are working on this.

XeGPU layout used by the load instruction in WG-level implementation:

```
#a = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [8, 16]>
#b = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [16, 16]>
```

With array_length support, the layout we may be able support for load would look like this:

```
#a_load = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [32, 32]>
#b_load = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [32, 32]>
```


## Conclusion:

With the lane-level implementation we are able to achieve around `~94` TFLOPs. This is close to the performance achieved by Triton and OneDNN. Unfortunately, they don't have a `f16, f16, and f16` (data type of A, B, C matrices respectively) version available. Triton only have bf16, bf16, f32 implementation of 4K GEMM available, and they reach about ~96 TFLOPs. This number is very close to what we achieve. In the near future, we plan to report number with the similar settings.


**Table 8: Lane-level vs Subgroup-level vs Workgroup-level**
| Metric  | Lane Level   | Subgroup Level | Workgroup Level |
| ------- | ------------ | -------------- | --------------- |
| Max     | 93.89 TFLOPS | 93.05 TFLOPS   | 87.99 TFLOPS    |
| Min     | 86.60 TFLOPS | 85.18 TFLOPS   | 79.54 TFLOPS    |
| Avg     | 89.86 TFLOPS | 88.85 TFLOPS   | 83.49 TFLOPS    |
| Median  | 89.66 TFLOPS | 88.74 TFLOPS   | 83.40 TFLOPS    |
| Std Dev | 1.76 TFLOPS  | 1.94 TFLOPS    | 2.19 TFLOPS     |

There are few things that we can use to improve our performance even more such as persistent kernel implementation, simultaneous launch. We plan to use them in the future.

As for WG-level implementation, the lowest hanging fruit is the `array_length` support, this should reduce the performance gap between the lane-level implementation and WG-level implementation significantly.



Appendix:
==========

TFLOPS to ms:
-------------
Here is the calculation used to convert execution time (ms) to TFLOPS.

### ms to TFLOPS Conversion Formula for 4K GEMM

#### General GEMM FLOPs formula

For matrix multiplication of size `M x K x N`:

```text
FLOPs = 2 × M x K x N
```

#### TFLOPS formula

If execution time is in **seconds**:

```text
TFLOPS = FLOPs / (time_in_seconds × 10^12)
```

If execution time is in **milliseconds**:

```text
TFLOPS = FLOPs / (time_in_ms × 10^9)
```

#### For 4K GEMM

For `M, K, N = 4096`:

```text
FLOPs = 2 × 4096^3
      = 137,438,953,472
```

So the direct conversion becomes:

```text
TFLOPS = 137,438,953,472 / (time_in_ms × 10^9)
```

Which simplifies to:

```text
TFLOPS = 137.438953472 / time_in_ms
```

#### Example

For `time = 1.53835 ms`:

```text
TFLOPS = 137.438953472 / 1.53835
       ≈ 89.34
```

#### Final shortcut

Use this for **4K GEMM**:

```text
TFLOPS ≈ 137.438953472 / time_in_ms
```

Co-operative Prefetch:
----------------------

We use the following co-operative prefetch strategy for A and B matrix.
In this case, we prefetch one 256x32 tile of A, and 32x256 tile of B.

**Prefetching A tile (256x32):**

prefetch the entire `256x32` slice of A WG tile, this means each subgroups needs to prefetch `8x32` slice:
```
      // each 1x4 row of SGs do a colloborative prefetch of 8x32 slice of the 32x32 tile
      // SG 0 -> slice 0 |
      // SG 1 -> slice 1 |
      // SG 2 -> slice 2  > SG 0,1,2,3 share data prefetch from the top 32x32 tile.
      // SG 3 -> slice 3 |
      // SG 4 -> slice 4
      // ....
      // SG 31 -> slice 31
```

**Prefetching B tile (32x256):**

Prefetch the entire `32x256` slice of B WG tile, we still use the prefetch size `8x32`.
SGs have `8x4` layout. In this case `8` subgroups must do a colloborative  prefetch of `32x64` tile.
This because the B tile arrangement within the `32x256` slice is as follows
```
      // 32x64 | 32x64 | 32x64 | 32x64
      // in terms of 8x32 slices the arrangement is,
      // 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32
      // 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32
      // 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32
      // 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32
      // So SGs 0,1,2,3,....31 prefetch in following fashion
      // | 0  | 16||  1 | 17 || 2  | 18 || 3 | 19 |
      // | 4  | 20||  5 | 21 || 6  | 22 || 7 | 23 |
      // | 8  | 24||  9 | 25 || 10 | 26 || 11| 27 |
      // | 12 | 28|| 13 | 29 || 14 | 30 || 15| 31 |
```
For example, SGs `0,4,8,12,16,20,24,28` share the data in left `32x64` tile of B slice.
