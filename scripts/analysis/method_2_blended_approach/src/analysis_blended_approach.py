import numpy as np
import metobs_toolkit 
import os
from pathlib import Path
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import glob
import cartopy.crs as ccrs
import cartopy
import osmnx as ox
import rasterio



def _km_to_degree_bounds(radius_km, lon_city, lat_city, margin_km=5.0):
    """Return approximate latitude and longitude half-widths in degrees."""
    half_extent_km = radius_km + margin_km
    lat_delta = half_extent_km / 111.0
    cos_lat = max(np.cos(np.deg2rad(lat_city)), 1e-6)
    lon_delta = half_extent_km / (111.0 * cos_lat)
    return lat_delta, lon_delta


def _nearest_grid_index(lon, lat, lon_city, lat_city):
    """Return the nearest grid index for 1D or 2D lon/lat coordinates."""
    if lon.ndim == 1 and lat.ndim == 1:
        ilon = int(np.nanargmin(np.abs(lon.values - lon_city)))
        ilat = int(np.nanargmin(np.abs(lat.values - lat_city)))
        return ilat, ilon

    if lon.ndim == 2 and lat.ndim == 2:
        dist = (lon - lon_city) ** 2 + (lat - lat_city) ** 2
        return np.unravel_index(np.nanargmin(dist.values), dist.shape)

    raise ValueError("lon and lat must both be 1D or both be 2D.")


def crop_area_city_radius(ds, radius_km, lon_city, lat_city, margin_km=5.0, lon_name='lon', lat_name='lat'):
    """Crop a dataset around a city center using a radius-based buffer.

    The returned crop extends to ``radius_km  + margin_km`` in each
    direction from the city center.
    """
    if lon_name not in ds.coords or lat_name not in ds.coords:
        raise KeyError(f"Expected coordinates {lon_name!r} and {lat_name!r} in the dataset.")

    lon = ds[lon_name]
    lat = ds[lat_name]
    lat_delta, lon_delta = _km_to_degree_bounds(radius_km, lon_city, lat_city, margin_km=margin_km)

    lat_min = lat_city - lat_delta
    lat_max = lat_city + lat_delta
    lon_min = lon_city - lon_delta
    lon_max = lon_city + lon_delta

    if lon.ndim == 1 and lat.ndim == 1:
        lon_dim = lon.dims[0]
        lat_dim = lat.dims[0]

        lon_slice = slice(lon_min, lon_max) if lon.values[0] <= lon.values[-1] else slice(lon_max, lon_min)
        lat_slice = slice(lat_min, lat_max) if lat.values[0] <= lat.values[-1] else slice(lat_max, lat_min)
        return ds.sel({lon_dim: lon_slice, lat_dim: lat_slice})

    if lon.ndim == 2 and lat.ndim == 2:
        within_bounds = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        if not bool(within_bounds.any()):
            raise ValueError("No grid cells found inside the requested city crop bounds.")

        y_idx, x_idx = np.where(within_bounds.values)
        y_dim, x_dim = lon.dims
        y0 = max(0, int(y_idx.min()))
        y1 = min(ds.sizes[y_dim], int(y_idx.max()) + 1)
        x0 = max(0, int(x_idx.min()))
        x1 = min(ds.sizes[x_dim], int(x_idx.max()) + 1)
        return ds.isel({y_dim: slice(y0, y1), x_dim: slice(x0, x1)})

    raise ValueError("lon and lat must both be 1D or both be 2D.")


def create_rural_mask(
    static_crp,
    lon_city,
    lat_city,
    buffer_radius_km,
    nature_threshold=0.5,
    orography_threshold=750,
    lon_name='lon',
    lat_name='lat',
    nature_var='nature_fraction',
    orography_var='orog',
    mask_name='rural_mask',
):
    """Create a reusable rural mask from cropped static fields.

    The mask keeps points that are within ``buffer_radius_km`` of the city
    center, have ``nature_fraction`` greater than or equal to
    ``nature_threshold`` and differ by at most ``orography_threshold`` meters
    from the orography value at the city center.
    """
    missing = [name for name in (lon_name, lat_name, nature_var, orography_var) if name not in static_crp]
    if missing:
        raise KeyError(f"Missing required variables/coordinates in static_crp: {missing}")

    lon = static_crp[lon_name]
    lat = static_crp[lat_name]
    radius_km = 6371.0

    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    lon_city_rad = np.deg2rad(lon_city)
    lat_city_rad = np.deg2rad(lat_city)

    a = (
        np.sin((lat_rad - lat_city_rad) / 2.0) ** 2
        + np.cos(lat_city_rad) * np.cos(lat_rad) * np.sin((lon_rad - lon_city_rad) / 2.0) ** 2
    )
    distance_km = 2.0 * radius_km * np.arcsin(np.sqrt(a))

    iy, ix = _nearest_grid_index(lon, lat, lon_city, lat_city)
    if lon.ndim == 1:
        center_orography = static_crp[orography_var].isel({lat.dims[0]: iy, lon.dims[0]: ix})
    else:
        center_orography = static_crp[orography_var].isel({lat.dims[0]: iy, lon.dims[1]: ix})

    mask = (
        (distance_km <= buffer_radius_km)
        & (static_crp[nature_var] >= nature_threshold)
        & (np.abs(static_crp[orography_var] - center_orography) <= orography_threshold)
    )

    mask_ds = xr.Dataset({mask_name: mask.astype(bool)})
    mask_ds[mask_name].attrs.update({
        "long_name": "Rural mask for model and static data",
        "city_center_lon": lon_city,
        "city_center_lat": lat_city,
        "buffer_radius_km": buffer_radius_km,
        "nature_threshold": nature_threshold,
        "orography_threshold_m": orography_threshold,
    })
    return mask_ds

def add_weighted_timeseries(wc, obs_ds, buffers=[50,100,200], rural_components=None, var_label='temperature'):
    """Add rural/urban weighted time series to obs_ds for given buffer radii.

    Parameters
    ----------
    wc : pandas.DataFrame
        Landcover fractions indexed by (name, buffer_radius) with columns for
        landcover classes. If `rural_fraction`/`urban_fraction` are missing they
        will be constructed from `rural_components` and the `Built-up` column.
    obs_ds : xarray.Dataset
        Observation dataset with station variables named by station identifier.
    buffers : list of int
        Buffer radii (meters) to compute weights for, e.g. [50,100,200].
    rural_components : list of str or None
        Landcover columns to consider rural. Defaults to ['Grassland','Cropland','Bare / sparse vegetation'].
    var_label : str
        Short label used in created variable names.

    Returns
    -------
    xarray.Dataset
        The input obs_ds with new data variables added.
    """
    import numpy as _np

    if rural_components is None:
        rural_components = ["Grassland", "Cropland", "Bare / sparse vegetation"]

    # Ensure rural/urban fraction columns exist
    if "rural_fraction" not in wc.columns:
        wc = wc.copy()
        wc["rural_fraction"] = 0.0
        for comp in rural_components:
            if comp in wc.columns:
                wc["rural_fraction"] = wc["rural_fraction"] + wc[comp]
    if "urban_fraction" not in wc.columns and "Built-up" in wc.columns:
        wc = wc.copy()
        wc["urban_fraction"] = wc["Built-up"]

    for buf in buffers:
        try:
            weights = wc.xs(buf, level="buffer_radius")[["rural_fraction", "urban_fraction"]]
        except Exception:
            print(f"Buffer {buf} not found in wc index; skipping.")
            continue

        # determine overlapping stations
        station_vars = [station for station in obs_ds.data_vars if station in weights.index]
        if not station_vars:
            print(f"No overlapping stations for buffer {buf}; skipping.")
            continue

        station_da = obs_ds[station_vars].to_array(dim="station")

        rural_w = xr.DataArray(
            weights.loc[station_vars, "rural_fraction"].to_numpy(),
            dims=["station"],
            coords={"station": station_vars},
        )
        urban_w = xr.DataArray(
            weights.loc[station_vars, "urban_fraction"].to_numpy(),
            dims=["station"],
            coords={"station": station_vars},
        )

        # weighted numerator and denominator (NaN-safe)
        rural_num = (station_da * rural_w).sum(dim="station", skipna=True)
        rural_den = (station_da.notnull() * rural_w).sum(dim="station")
        rural_ts = xr.where(rural_den > 0, rural_num / rural_den, _np.nan)

        urban_num = (station_da * urban_w).sum(dim="station", skipna=True)
        urban_den = (station_da.notnull() * urban_w).sum(dim="station")
        urban_ts = xr.where(urban_den > 0, urban_num / urban_den, _np.nan)

        # variable names include buffer in meters
        rv_name = f"rural_weighted_{var_label}_{buf}m"
        uv_name = f"urban_weighted_{var_label}_{buf}m"

        obs_ds = obs_ds.assign({rv_name: rural_ts, uv_name: urban_ts})

        obs_ds[rv_name].attrs.update({
            "long_name": f"Rural-fraction-weighted station mean {var_label} ({buf}m)",
            "units": "degC",
        })
        obs_ds[uv_name].attrs.update({
            "long_name": f"Urban-fraction-weighted station mean {var_label} ({buf}m)",
            "units": "degC",
        })

    return obs_ds


def crop_area_city(ds, res, lon_city, lat_city, lon_lim, lat_lim, lon_name='lon', lat_name='lat'):
    """Crop a dataset around a city center for either 2D curvilinear or 1D lon/lat grids."""

    dlon = int(111 * lon_lim / res)
    dlat = int(111 * lat_lim / res)

    if lon_name not in ds.coords or lat_name not in ds.coords:
        raise KeyError(f"Expected coordinates {lon_name!r} and {lat_name!r} in the dataset.")

    lon = ds[lon_name]
    lat = ds[lat_name]

    if lon.ndim == 1 and lat.ndim == 1:
        lon_dim = lon.dims[0]
        lat_dim = lat.dims[0]

        ilon = int(np.nanargmin(np.abs(lon.values - lon_city)))
        ilat = int(np.nanargmin(np.abs(lat.values - lat_city)))

        x0 = max(0, ilon - dlon)
        x1 = min(ds.sizes[lon_dim], ilon + dlon + 1)
        y0 = max(0, ilat - dlat)
        y1 = min(ds.sizes[lat_dim], ilat + dlat + 1)

        return ds.isel({lat_dim: slice(y0, y1), lon_dim: slice(x0, x1)})

    if lon.ndim == 2 and lat.ndim == 2:
        dist = (lon - lon_city) ** 2 + (lat - lat_city) ** 2
        iy, ix = np.unravel_index(np.nanargmin(dist.values), dist.shape)

        y_dim, x_dim = lon.dims
        y0 = max(0, iy - dlat)
        y1 = min(ds.sizes[y_dim], iy + dlat + 1)
        x0 = max(0, ix - dlon)
        x1 = min(ds.sizes[x_dim], ix + dlon + 1)

        return ds.isel({y_dim: slice(y0, y1), x_dim: slice(x0, x1)})

    raise ValueError("lon and lat must both be 1D or both be 2D.")

def align_to_reference_grid(dataset_to_align, reference_ds, tolerance=1e-6, coord_names=['x', 'y']):
    """Align dataset coordinates to reference dataset grid if differences are within tolerance.
    
    Parameters
    ----------
    dataset_to_align : xr.Dataset
        Dataset whose coordinates will be aligned.
    reference_ds : xr.Dataset
        Reference dataset with target coordinates.
    tolerance : float
        Maximum allowed difference to treat as numeric error (default 1e-9).
    coord_names : list of str
        Coordinate names to align (default ['x', 'y']).
    
    Returns
    -------
    xr.Dataset
        Dataset with coordinates aligned to reference if within tolerance.
    str
        Status message describing what was done.
    """
    result = dataset_to_align.copy(deep=True)
    status_lines = []
    coords_to_assign = {}
    
    for coord in coord_names:
        if coord not in dataset_to_align.coords or coord not in reference_ds.coords:
            status_lines.append(f"Coordinate '{coord}' not found in one or both datasets, skipping.")
            continue
        
        dataset_coord = dataset_to_align[coord].values
        reference_coord = reference_ds[coord].values
        
        if len(dataset_coord) != len(reference_coord):
            status_lines.append(f"Coordinate '{coord}' has different lengths: dataset={len(dataset_coord)}, reference={len(reference_coord)}. Skipping.")
            continue
        
        diff = np.abs(dataset_coord - reference_coord)
        max_diff = np.nanmax(diff)
        
        if max_diff < tolerance:
            coords_to_assign[coord] = reference_coord
            status_lines.append(f"Coordinate '{coord}': max diff={max_diff:.2e} (< {tolerance:.2e}). ALIGNED to reference.")
        else:
            status_lines.append(f"Coordinate '{coord}': max diff={max_diff:.2e} (≥ {tolerance:.2e}). NOT aligned (diff too large).")
    
    # Assign all aligned coordinates at once
    if coords_to_assign:
        result = result.assign_coords(coords_to_assign)
    
    return result, "\n".join(status_lines)


def compute_spatial_mean_over_mask(dataset, mask_da, variables=None, spatial_dims=None):
    """Compute spatial mean of dataset variables over a mask region.
    
    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset with variables to average.
    mask_da : xr.DataArray
        Binary mask (1 for region to include, 0 or NaN for region to exclude).
    variables : list of str or None
        List of variable names to compute means for. If None, uses all data variables.
    spatial_dims : list of str or None
        Spatial dimensions to average over (e.g., ['x', 'y']). If None, auto-detects
        from common dimension names ['x', 'y', 'lat', 'lon'].
    
    Returns
    -------
    xr.Dataset
        Dataset with same variables, spatially averaged over mask (time dimension preserved).
    """
    # Determine variables to process
    if variables is None:
        variables = list(dataset.data_vars)
    
    # Auto-detect spatial dimensions if not provided
    if spatial_dims is None:
        spatial_dims = [d for d in mask_da.dims if d in dataset.dims]
        if not spatial_dims:
            spatial_dims = [d for d in ['x', 'y', 'lat', 'lon'] if d in dataset.dims]
    
    if not spatial_dims:
        raise ValueError("Could not determine spatial dimensions. Please provide spatial_dims explicitly.")
    
    result_vars = {}
    for var in variables:
        if var not in dataset.data_vars:
            print(f"Warning: variable '{var}' not found in dataset, skipping.")
            continue
        
        # Apply mask and compute spatial mean (preserving other dimensions like time)
        masked_var = dataset[var].where(mask_da)
        spatial_mean = masked_var.mean(dim=spatial_dims, skipna=True)
        result_vars[var] = spatial_mean
    
    # Combine all variables into a single dataset
    result_ds = xr.Dataset(result_vars)
    return result_ds

def from_latlon_to_xy( lat, lon, proj):
    xy = rasterio.warp.transform(ccrs.PlateCarree(), proj, [lon], [lat])
    x, y = xy[0][0], xy[1][0]
    #print(f"Coordinates : x = {x:.2f} m, y = {y:.2f} m")
    return (x, y)