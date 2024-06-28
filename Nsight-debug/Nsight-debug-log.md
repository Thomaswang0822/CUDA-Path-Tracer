# Debugging Performance with Nsight Compute

I found that my code was signficantly slower than the ref code (by @HummaWhite), even when we are "on the same page".
Under the same scene setting, his path-tracer achieves ~10 FPS and mine has only ~6 FPS.
Thus, I learned Nvidia Nsight Compute to perform some performance profiling on my and his code.

## Setting: The Cornel Box

- Light: original light (a square area light on the ceiling)
- Object: a dielectric (glass) teapot other than the CBox
- Max Depth: 4
- SPP: 50

In addition, the profiling takes place on the same laptop, and the runtime CUDA configurations (blockSize, gridSize, etc.)
are the same.

## computerIntersections()

My code:

<img src="PT-computeIntersections.png" width="800">

Reference code:

<img src="REF-computeIntersections.png" width="800">

The root cause of performance difference for this kernel is number of registers used per thread (#reg).
The ref code only uses #reg = 52, but mine uses #reg = 115.

Higher #reg leads to lower occupancy (both theoretical and achieved), lower throughput (both memory and compute), and ultimately
longer time to finish.

## shadeSegment()

Note that I called this kernel `shadeSegment()` while the reference code calls it `pathIntegSamplerSurface()`, and they
perform same operations on high level.

My code:

<img src="PT-shade.png" width="800">

Reference code:

<img src="REF-shade.png" width="800">

The root cause, not surprisingly, is also #reg, 164 vs 80.

## Debug Result

The ref code puts all the definitions of `DevScene` member `__device__` functions in *scene.h*, but I moved the
definitions to a separate *scene.cu* file. Also, for other remaining `__host__` definitions, the ref code stores
them in *scene.cpp* instead of *scene.cu*.

The huge difference, in the end, is solely caused by inlining `__device__` function. When moving them to the header file,
the performance become consistent, and according to Nsight Compute, both kernels utlize same #reg as the reference code.

TODO: elaborate this in main README.