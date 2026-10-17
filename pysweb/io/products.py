"""
Script: products.py
Objective: Publish exact spatial subsets and verify product and auxiliary grid contracts.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: NetCDF source products and an exact Landsat template window.
Outputs: Streaming subsets, strict final grid checks and auxiliary metadata provenance.
Usage: from pysweb.io.products import crop_product, assert_identical_grids
Dependencies: netCDF4, numpy, rasterio, xarray, pysweb
"""
from pathlib import Path
import os
import netCDF4 as nc
import numpy as np
from rasterio import Affine
import xarray as xr
from pysweb.io.grids import open_template, spatial
from pysweb.io.provenance import input_manifest, save_output_manifest
from pysweb.io.netcdf_stream import stream_chunks, sequential_read_cache


def _declared_transform(path, template):
    """Read the stored affine too; coordinate inference can hide a bad origin."""
    if Path(path).suffix.lower() not in {'.tif', '.tiff'}:
        with xr.open_dataset(path, decode_coords='all') as dataset:
            mapping = dataset.rio.grid_mapping
            if mapping in dataset and 'GeoTransform' in dataset[mapping].attrs:
                values = [float(value) for value in dataset[mapping].attrs['GeoTransform'].split()]
                if len(values) != 6 or not np.isfinite(values).all():
                    raise ValueError('Invalid declared affine transform')
                return Affine.from_gdal(*values)
    return template.rio.transform()


def assert_identical_grids(first_path, second_path):
    first, second = open_template(first_path), open_template(second_path)
    if first.rio.crs != second.rio.crs:
        raise ValueError('Final product CRS differs')
    transform = _declared_transform(first_path, first)
    if not np.array_equal(tuple(transform), tuple(_declared_transform(second_path, second))):
        raise ValueError('Final product affine transform differs')
    for dim in ('x', 'y'):
        if not np.array_equal(first[dim].values, second[dim].values):
            raise ValueError(f'Final product {dim} coordinates differ')
    return {'crs': str(first.rio.crs), 'transform': list(transform)[:6],
            'shape': [first.sizes['y'], first.sizes['x']], 'identical': True}


def auxiliary_grid_contract(output_path, auxiliary_path):
    """Require exact centres and CRS before copying onto the output metadata.

    Reconstructing an affine from floating-point centres can change its last
    bits. Only coordinate-scale roundoff is admissible for an auxiliary whose
    centres match exactly. Final product comparisons remain bitwise strict.
    """
    output, auxiliary = open_template(output_path), open_template(auxiliary_path)
    if output.rio.crs != auxiliary.rio.crs:
        raise ValueError('Auxiliary CRS differs')
    for dim in ('x', 'y'):
        if not np.array_equal(output[dim].values, auxiliary[dim].values):
            raise ValueError(f'Auxiliary {dim} coordinates differ')
    target = np.asarray(_declared_transform(output_path, output))[:6]
    source = np.asarray(_declared_transform(auxiliary_path, auxiliary))[:6]
    scale = max(1., *(float(np.max(np.abs(output[dim].values))) for dim in ('x', 'y')))
    tolerance = 8*np.finfo('float64').eps*scale
    if (target[[1, 3]] != 0).any() or (source[[1, 3]] != 0).any():
        raise ValueError('Auxiliary rotated affine is unsupported')
    if not np.isfinite(source).all() or not np.allclose(source, target, rtol=0, atol=tolerance):
        raise ValueError('Auxiliary affine differs beyond coordinate roundoff')
    return dict(identical_coordinates=True, crs=str(output.rio.crs),
        source_transform=source.tolist(), output_transform=target.tolist(),
        affine_metadata_normalized=not np.array_equal(source, target),
        maximum_affine_adjustment=float(np.max(np.abs(source-target))),
        coordinate_roundoff_tolerance=tolerance)


def crop_product(source_path, output_path, template):
    """Copy every auxiliary in bounded slabs; crop only exact x/y source cells."""
    template = spatial(template)
    source_grid = open_template(source_path)
    if source_grid.rio.crs != template.rio.crs:
        raise ValueError('Final SEB crop must retain its selected Landsat CRS')
    slices = {}
    for dim in ('x', 'y'):
        indices = source_grid.indexes[dim].get_indexer(template[dim].values)
        if np.any(indices < 0) or (len(indices) > 1 and np.any(np.diff(indices) != 1)):
            raise ValueError('Final grid must be an exact contiguous SEB pixel window')
        slices[dim] = slice(int(indices[0]), int(indices[-1])+1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f'.{output_path.name}.{os.getpid()}.tmp')
    try:
        with nc.Dataset(source_path) as source, nc.Dataset(temporary, 'w') as output:
            output.setncatts({name: source.getncattr(name) for name in source.ncattrs()})
            for dim, values in source.dimensions.items():
                output.createDimension(dim, template.sizes[dim] if dim in slices else len(values))
            for name, variable in source.variables.items():
                fill = variable.getncattr('_FillValue') if '_FillValue' in variable.ncattrs() else None
                options = {'zlib': True, 'complevel': 3} if variable.ndim else {}
                block_dim = next((d for d in variable.dimensions if d not in slices), None)
                shape = tuple(len(output.dimensions[d]) for d in variable.dimensions)
                chunks = stream_chunks(variable.dimensions, shape, block_dim=block_dim)
                if chunks is not None:
                    options['chunksizes'] = chunks
                target = output.createVariable(name, variable.datatype, variable.dimensions, fill_value=fill, **options)
                target.setncatts({k: variable.getncattr(k) for k in variable.ncattrs() if k != '_FillValue'})
                count = len(source.dimensions[block_dim]) if block_dim else 1
                with sequential_read_cache(variable, block_dim):
                    for begin in range(0, count, 32):
                        src = tuple(slices.get(d, slice(begin, begin+32) if d == block_dim else slice(None))
                                    for d in variable.dimensions)
                        dst = tuple(slice(begin, begin+32) if d == block_dim else slice(None)
                                    for d in variable.dimensions)
                        target[dst or Ellipsis] = variable[src or Ellipsis]
            output['spatial_ref'].GeoTransform = ' '.join(str(v) for v in template.rio.transform().to_gdal())
            output.spatial_grid = 'Exact subset of selected Landsat 30 m UTM scene grid'
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    save_output_manifest(output_path, input_manifest({'operation': 'exact spatial crop',
        'transform': list(template.rio.transform())[:6]}, [source_path]))
    return output_path
