# CUDA Path Tracer

## Quick Intro

This project serves to:

1. refresh my memory on path tracing and related techniques;
2. gain more insight on shipping a path tracer to GPU;

It is forked from the starter code of [**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**](https://github.com/CIS565-Fall-2022/Project3-CUDA-Path-Tracer), and will not be submitted as any schoolwork.

## Primary Reference

Since I have learned and (painfully) debugged those low-level interfaces in a past class project, it's more efficient to use someone else's code.

The primary reference code I used is by [@HummaWhite](https://github.com/HummaWhite/Project3-CUDA-Path-Tracer).

For code irrelevant to CUDA, I mostly copy after understanding them. For CUDA-related code, I spent more time worship the technical beauty.
The parallizable BVH is the biggest reason I picked this repo as my reference.

## Other References

- [CIS 565 Discussion Slides](https://onedrive.live.com/view.aspx?resid=A6B78147D66DD722%2196250&authkey=!AHM5o0OIig5tENc)
- [CPU Path Tracer](https://github.com/Thomaswang0822/torrey_renderer): my own class project

## Step 1. A Minimal CUDA Path Tracer

It has no fancy materials, no mesh loader, and even no triangle supported. But it's not trivial at all. 
The biggest difference (from a traditional CPU PT) is the memory management behind the massive parallelism.

In addition to changing the path-trace function, which is recursive in nature, into iterative, there is an smarter way than pixel parallelization.
Briefly, we do ray parallelization by maintaining a pool of active rays. Terminated rays (miss, hitting light, etc.) get "picked out" from the pool.
They are moved to another pool and their data get updated to the rendering. This is called ***Stream Compaction***. Memory management is handled by
*thrust* API. More info can be found in the Discussion Slides starting from page 21.

Here is the illustration: the rendering converges from ~50 spp (top) to ~5000 spp (bottom).
![50spp](img/step1_50spp.png)
![5000spp](img/step1_5000spp.png)
