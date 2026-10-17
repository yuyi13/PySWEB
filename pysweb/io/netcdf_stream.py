"""
Script: netcdf_stream.py
Objective: Match compressed NetCDF chunks and bounded read caches to streamed array copies.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Variable dimensions, shapes and existing NetCDF chunk metadata.
Outputs: Bounded chunk layouts and temporary per-variable read-cache settings.
Usage: from pysweb.io.netcdf_stream import stream_chunks, sequential_read_cache
Dependencies: contextlib, math, numpy, netCDF4-compatible variables.
"""
from contextlib import contextmanager
from math import ceil, prod
import numpy as np


def stream_chunks(dimensions, shape, *, block_dim=None, block_size=32, spatial_size=64):
    """Keep compressed chunks within the slabs written by streaming callers."""
    if block_size < 1 or spatial_size < 1:
        raise ValueError('Positive streaming chunk sizes required')
    if block_dim is None:
        block_dim = next((d for d in ('time', 'scene_time') if d in dimensions), None)
    if block_dim is None or block_dim not in dimensions:
        return None
    return tuple(max(1, min(int(size), block_size if dim == block_dim else
        32 if dim in ('time', 'scene_time') else 1 if dim == 'layer' else spatial_size))
        for dim, size in zip(dimensions, shape))


@contextmanager
def sequential_read_cache(variable, block_dim='time', *, max_cache_bytes=512*1024**2):
    """Cache one or two source chunk slabs, capped independently of host memory.

    Old files may have multi-year temporal chunks. A small time slice then
    touches more spatial chunks than the default cache can retain. Reusing the
    slab prevents repeated decompression while callers still read small blocks.
    Restore the original cache after each variable, including on failure.
    """
    chunks = variable.chunking()
    if chunks is None or isinstance(chunks, str) or block_dim not in variable.dimensions:
        yield
        return
    if max_cache_bytes < 1:
        raise ValueError('Positive NetCDF cache limit required')
    previous = variable.get_var_chunk_cache()
    axis = variable.dimensions.index(block_dim)
    slab_chunks = prod(ceil(size/chunk) for i, (size, chunk) in
                       enumerate(zip(variable.shape, chunks)) if i != axis)
    slab_bytes = prod(chunks)*np.dtype(variable.dtype).itemsize*slab_chunks
    size = min(int(max_cache_bytes), max(int(previous[0]), 2*slab_bytes))
    slots = max(int(previous[1]), 2*slab_chunks+1)
    variable.set_var_chunk_cache(size=size, nelems=slots, preemption=1.0)
    try:
        yield
    finally:
        variable.set_var_chunk_cache(*previous)
