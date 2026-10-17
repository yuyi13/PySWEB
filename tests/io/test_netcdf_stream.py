"""
Script: test_netcdf_stream.py
Objective: Verify bounded compressed layouts, cache restoration and exact streamed auxiliary values.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Small numeric NetCDF fixtures and unpopulated long-calendar variable metadata.
Outputs: Streaming memory/layout and value-preservation assertions.
Usage: python -m pytest tests/io/test_netcdf_stream.py
Dependencies: netCDF4, numpy, pytest, pysweb.
"""
import netCDF4 as nc
import numpy as np
import pytest
from pysweb.io.netcdf_stream import stream_chunks, sequential_read_cache
from pysweb.swb.paired import _copy_variable


def test_long_calendar_chunks_follow_temporal_and_row_write_blocks():
    assert stream_chunks(('time', 'y', 'x'), (5844, 167, 167)) == (32, 64, 64)
    assert stream_chunks(('scene_time', 'y', 'x'), (577, 202, 221)) == (32, 64, 64)
    assert stream_chunks(('time', 'layer', 'y', 'x'), (12414, 5, 167, 167)) == (32, 1, 64, 64)
    assert stream_chunks(('time', 'y', 'x'), (5844, 167, 167),
        block_dim='y', block_size=8) == (32, 8, 64)
    assert stream_chunks(('y', 'x'), (167, 167)) is None
    with pytest.raises(ValueError, match='Positive'):
        stream_chunks(('time',), (10,), block_size=0)


def test_source_cache_is_capped_and_restored_after_failure(tmp_path):
    with nc.Dataset(tmp_path/'old_layout.nc', 'w') as ds:
        for dim, size in [('time', 6000), ('y', 128), ('x', 128)]:
            ds.createDimension(dim, size)
        # Metadata only: model the real long-calendar chunk geometry without
        # allocating or writing a large fixture in the test suite.
        variable = ds.createVariable('ET', 'f4', ('time', 'y', 'x'),
            zlib=True, chunksizes=(1500, 32, 32))
        variable.set_var_chunk_cache(size=16*1024**2, nelems=1009, preemption=0.75)
        original = variable.get_var_chunk_cache()
        with pytest.raises(RuntimeError, match='copy failed'):
            with sequential_read_cache(variable, max_cache_bytes=128*1024**2):
                assert original[0] < variable.get_var_chunk_cache()[0] <= 128*1024**2
                raise RuntimeError('copy failed')
        assert variable.get_var_chunk_cache() == original


@pytest.mark.parametrize('format', ['NETCDF3_CLASSIC', 'NETCDF4'])
def test_unchunked_sources_remain_readable(tmp_path, format):
    with nc.Dataset(tmp_path/'contiguous.nc', 'w', format=format) as ds:
        ds.createDimension('time', 3)
        variable = ds.createVariable('ET', 'f4', ('time',))
        variable[:] = [1., 2., 3.]
        with sequential_read_cache(variable):
            np.testing.assert_array_equal(variable[:], [1., 2., 3.])


@pytest.mark.parametrize('indices', [np.arange(1, 5), np.array([0, 2, 3, 5])])
def test_auxiliary_copy_preserves_nonleading_time_values_and_metadata(tmp_path, indices):
    values = np.arange(36, dtype='float32').reshape(3, 6, 2)
    values[1, 3, 0] = np.nan
    with nc.Dataset(tmp_path/'source.nc', 'w') as source, nc.Dataset(tmp_path/'copy.nc', 'w') as target:
        for dim, size in [('y', 3), ('time', 6), ('x', 2)]:
            source.createDimension(dim, size)
            target.createDimension(dim, len(indices) if dim == 'time' else size)
        variable = source.createVariable('forcing', 'f4', ('y', 'time', 'x'),
            fill_value=np.nan, zlib=True)
        variable.units = 'mm day-1'
        variable[:] = values
        copied = _copy_variable(target, variable, time_indices=indices)
        np.testing.assert_equal(np.ma.filled(copied[:], np.nan), values[:, indices, :])
        assert copied.units == variable.units
        assert copied.chunking()[1] <= 32
