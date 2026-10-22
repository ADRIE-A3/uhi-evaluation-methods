# This file is part of URCLIMASK (https://github.com/FPS-URB-RCC/URCLIMASK),
# Copyright 2023, European Union. Licensed under the Apache License, Version 2.0
# (see LICENSE in this directory).
#
# Modified by Leon Adriaensen (Ghent University), 2025-2026, from URCLIMASK commit b779ebc (2025-10-01):
#   - station classification from the UCDB urban centre and from LCZ
#     (classify_stations_from_ucdb_city, classify_stations_from_LCZ);
#   - point-to-point UHI (compute_UHI_p2p, plot_UHI_p2p_cycle) and p2a plotting (plot_blend_on_ax);
#   - grid alignment and mask-polygon helpers (align_to_reference_grid, build_polygons_from_urmask,
#     plot_polygon_boundaries, from_latlon_to_xy) and plot_ensemble.

import xarray as xr
import geopandas as gpd
import os
import numpy as np
from pathlib import Path
import pandas as pd
from shapely.geometry import Point, Polygon, box, MultiPolygon
from shapely.ops import unary_union
from rasterio import warp
import rasterio
import cartopy.crs as ccrs
from matplotlib.lines import Line2D



def load_ucdb_city(root, city):
    """
    Load and filter a city shapefile from the Urban Centre Database (UCDB).

    Parameters:
    root (str): The root directory where the shapefile is located.
    city (str): The name of the city to load.

    Returns:
    gpd.GeoDataFrame: A GeoDataFrame containing the filtered city shapefile.
    """
    root_path = Path(root)
    default_gpkg = root_path / "GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2" / "GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg"

    if default_gpkg.exists():
        gpkg_path = default_gpkg
    else:
        # Support alternate extracted directory layouts by searching recursively.
        matches = sorted(root_path.rglob("GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg"))
        if not matches:
            raise FileNotFoundError(
                f"UCDB geopackage not found under '{root}'. "
                "Expected GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg in any subdirectory."
            )
        gpkg_path = matches[0]

    ucdb_info = gpd.read_file(gpkg_path)
    ucdb_city = ucdb_info.query(f'UC_NM_MN =="{city}"').to_crs(crs='EPSG:4326')
    if city == 'London' or city == 'Birmingham':
        ucdb_city = ucdb_city[ucdb_city['CTR_MN_NM'] == 'United Kingdom']
    if city == 'Riga':
        ucdb_city = ucdb_city[ucdb_city['CTR_MN_NM'] == 'Latvia']
    if city == 'Santiago':
        ucdb_city = ucdb_city[ucdb_city['CTR_MN_NM'] == 'Chile']
    if city == 'Barcelona':
        ucdb_city = ucdb_city[ucdb_city['CTR_MN_NM'] == 'Spain']
    if city == 'Dhaka':
        ucdb_city = ucdb_city[ucdb_city['CTR_MN_NM'] == 'Bangladesh']
    if city == 'Naples':
        ucdb_city = ucdb_city[ucdb_city['CTR_MN_NM'] == 'Italy']
    return ucdb_city


def traverseDir(root, end):
    for (dirpath, dirnames, filenames) in os.walk(root):
        for file in filenames:
            if file.endswith((end)):
                yield os.path.join(dirpath, file)

def fix_360_longitudes(
    dataset: xr.Dataset, lonname: str = "lon"
) -> xr.Dataset:
    """
    Fix longitude values.

    Function to transform datasets where longitudes are in (0, 360) to (-180, 180).

    Parameters
    ----------
    dataset (xarray.Dataset): data stored by dimensions
    lonname (str): name of the longitude dimension

    Returns
    -------
    dataset (xarray.Dataset): data with the new longitudes
    """
    lon = dataset[lonname]
    if lon.max().values > 180 and lon.min().values >= 0:
        dataset[lonname] = dataset[lonname].where(lon <= 180, other=lon - 360)
    return dataset

def kelvin2degC(ds, variable):
    """
    Convert a variable's units from Kelvin to Celsius in an xarray Dataset.

    Parameters:
    ds (xr.Dataset): The dataset containing the variable to convert.
    variable (str): The name of the variable to convert.

    Returns:
    xr.Dataset: The dataset with the variable converted to Celsius.
    """
    if ds[variable].attrs.get('units') == 'K':
        ds[variable] = ds[variable] -273.15
        ds[variable].attrs['units'] = 'degC'
        
    return ds


def fix_360_longitudes(
    dataset: xr.Dataset, lonname: str = "lon"
) -> xr.Dataset:
    """
    Fix longitude values.

    Function to transform datasets where longitudes are in (0, 360) to (-180, 180).

    Parameters
    ----------
    dataset (xarray.Dataset): data stored by dimensions
    lonname (str): name of the longitude dimension

    Returns
    -------
    dataset (xarray.Dataset): data with the new longitudes
    """
    lon = dataset[lonname]
    if lon.max().values > 180 and lon.min().values >= 0:
        dataset[lonname] = dataset[lonname].where(lon <= 180, other=lon - 360)
    return dataset

def plot_urban_polygon(ds, ax):
    '''
    Plots urban and non-urban polygons from a mask dataset on the given axis.
    Returns GeoDataFrames for urban and non-urban areas.
    '''
    # Assume the mask is in the 'urmask' variable
    mask = ds['urmask']
    lon2d = mask.lon.values
    lat2d = mask.lat.values
    
    # Create lists to store polygons for urban areas (mask == 1) and non-urban areas (mask == 0)
    urban_polygons = []
    non_urban_polygons = []
    
    if lon2d.ndim == 1:
        dist_lat = abs(lat2d[1] - lat2d[0]) / 2
        dist_lon = abs(lon2d[1] - lon2d[0]) / 2
        for lon in range(len(lon2d)):
            for lat in range(len(lat2d)):
                square = Polygon([
                    (round(lon2d[lon] - dist_lon, 3), round(lat2d[lat] - dist_lat, 3)),  # bottom-left corner
                    (round(lon2d[lon] + dist_lon, 3), round(lat2d[lat] - dist_lat, 3)),  # bottom-right corner
                    (round(lon2d[lon] + dist_lon, 3), round(lat2d[lat] + dist_lat, 3)),  # top-right corner
                    (round(lon2d[lon] - dist_lon, 3), round(lat2d[lat] + dist_lat, 3)),  # top-left corner
                ])

                # Add the polygon to the corresponding list
                if mask[lat, lon] == 1:
                    urban_polygons.append(square)
                elif mask[lat, lon] == 0:
                    non_urban_polygons.append(square)
    else:
        # Reconstruct cell corners from the 2D center coordinates.
        # This preserves the original grid shape used by pcolormesh while also
        # extending the outer boundary so edge cells are not skipped.
        nlat, nlon = mask.shape

        def _edge_coords(coord2d, axis):
            if axis == 1:
                edge = np.empty((coord2d.shape[0], coord2d.shape[1] + 1), dtype=float)
                edge[:, 1:-1] = 0.5 * (coord2d[:, :-1] + coord2d[:, 1:])
                edge[:, 0] = coord2d[:, 0] - 0.5 * (coord2d[:, 1] - coord2d[:, 0]) if coord2d.shape[1] > 1 else coord2d[:, 0]
                edge[:, -1] = coord2d[:, -1] + 0.5 * (coord2d[:, -1] - coord2d[:, -2]) if coord2d.shape[1] > 1 else coord2d[:, -1]
                return edge
            edge = np.empty((coord2d.shape[0] + 1, coord2d.shape[1]), dtype=float)
            edge[1:-1, :] = 0.5 * (coord2d[:-1, :] + coord2d[1:, :])
            edge[0, :] = coord2d[0, :] - 0.5 * (coord2d[1, :] - coord2d[0, :]) if coord2d.shape[0] > 1 else coord2d[0, :]
            edge[-1, :] = coord2d[-1, :] + 0.5 * (coord2d[-1, :] - coord2d[-2, :]) if coord2d.shape[0] > 1 else coord2d[-1, :]
            return edge

        lon_edge = _edge_coords(lon2d, axis=1)
        lon_edge = _edge_coords(lon_edge, axis=0)
        lat_edge = _edge_coords(lat2d, axis=1)
        lat_edge = _edge_coords(lat_edge, axis=0)

        for lat in range(nlat):
            for lon in range(nlon):
                if pd.isnull(mask.lon[lat, lon]) or pd.isnull(mask.lat[lat, lon]):
                    continue

                square = Polygon([
                    (lon_edge[lat, lon], lat_edge[lat, lon]),
                    (lon_edge[lat, lon + 1], lat_edge[lat, lon + 1]),
                    (lon_edge[lat + 1, lon + 1], lat_edge[lat + 1, lon + 1]),
                    (lon_edge[lat + 1, lon], lat_edge[lat + 1, lon]),
                ])

                if mask[lat, lon] == 1:
                    urban_polygons.append(square)
                elif mask[lat, lon] == 0:
                    non_urban_polygons.append(square)
                    
    # Unite all adjacent polygons for urban (mask == 1) and non-urban (mask == 0)
    unified_urban_polygon = unary_union(urban_polygons)
    unified_non_urban_polygon = unary_union(non_urban_polygons)
    # Create GeoDataFrames for the urban and non-urban polygons
    # CRS 'EPSG:4326' specifies the WGS 84 coordinate system, which is widely used for global GPS coordinates (lat/lon)
    gdf_urban = gpd.GeoDataFrame(geometry=[unified_urban_polygon])
    gdf_non_urban = gpd.GeoDataFrame(geometry=[unified_non_urban_polygon])
    # Plot the boundary of the unified non-urban polygon (in blue)
    gdf_non_urban.boundary.plot(ax=ax,color='#8A8D28', zorder=1, linewidth=2)
    # Plot the boundary of the unified urban polygon (in red) on top of the non-urban
    gdf_urban.boundary.plot(ax=ax,  color='#A52A2A', zorder=100, linewidth=2)

    return(gdf_urban, gdf_non_urban)




def plot_urban_borders(ds, ax, alpha = 1, linewidth = 2):
    """
    Plot the borders of urban areas on a map.

    Parameters:
    ds (xr.Dataset): The dataset containing longitude, latitude, and urban area data.
    ax (matplotlib.axes._subplots.AxesSubplot): The matplotlib axes on which to plot.

    """
    lon2d = ds.lon.values
    lat2d = ds.lat.values
    dist_lat = lat2d[1, 0] - lat2d[0, 0]
    dist_lon = lon2d[0, 1] - lon2d[0, 0]
    dist_latlon = lat2d[0 ,1] - lat2d[0, 0]
    dist_lonlat = lon2d[1, 0] - lon2d[0, 0]
    # Overlay the cell borders and handle NaNs
    for i in range(len(ds.lat)-1):
        for j in range(len(ds.lon)-1):
            lons = [lon2d[i, j], lon2d[i, j+1], lon2d[i+1, j+1], lon2d[i+1, j], lon2d[i, j]]
            lats = [lat2d[i, j], lat2d[i, j+1], lat2d[i+1, j+1], lat2d[i+1, j], lat2d[i, j]]

            lons = lons - abs(lon2d[i, j] - lon2d[i, j+1])/2              
            lats = lats - abs(lat2d[i, j] - lat2d[i+1, j])/2
            
            data_cell = ds['urmask'].values[i, j]
            
            if data_cell == 1:
                ax.plot(lons, lats, color='grey', zorder = 100, linewidth = linewidth, alpha = alpha)
            elif data_cell == 0:
                ax.plot(lons, lats, color='green', zorder = 1, linewidth = linewidth, alpha = alpha)

     # Plot the rightmost column
    for i in range(len(ds.lat) - 1):
        lons = [lon2d[i, -1], lon2d[i + 1, -1], lon2d[i + 1, -1] + dist_lon, lon2d[i, -1] + dist_lon, lon2d[i, -1]]
        lats = [lat2d[i, -1], lat2d[i + 1, -1], lat2d[i + 1, -1] + dist_latlon, lat2d[i, -1] + dist_latlon, lat2d[i, -1]]

        lons = lons - abs(lon2d[i, -1] - lon2d[i, -1])/2 - dist_lon/2            
        lats = lats - abs(lat2d[i, -1] - lat2d[i+1, -1])/2 
        
        data_cell = ds['urmask'].values[i, -1]
    
        if data_cell == 1:
            ax.plot(lons, lats, color='grey', zorder=100, linewidth = linewidth, alpha = alpha)
        elif data_cell == 0:
            ax.plot(lons, lats, color='green', zorder=1, linewidth = linewidth, alpha = alpha)
    
    # Plot the topmost row
    for j in range(len(ds.lon) - 1):
        lons = [lon2d[-1, j], lon2d[-1, j + 1], lon2d[-1, j + 1]  + dist_lonlat, lon2d[-1, j]  + dist_lonlat, lon2d[-1, j]]
        lats = [lat2d[-1, j], lat2d[-1, j + 1], lat2d[-1, j + 1] + dist_lat, lat2d[-1, j] + dist_lat, lat2d[-1, j]]

        
        lons = lons - abs(lon2d[-1, j] - lon2d[-1, j+1])/2           
        lats = lats - abs(lat2d[-1, j] - lat2d[-1, j])/2 - dist_lat/2  
        
        data_cell = ds['urmask'].values[-1, j]
    
        if data_cell == 1:
            ax.plot(lons, lats, color='red', zorder=100, linewidth=2)
        elif data_cell == 0:
            ax.plot(lons, lats, color='b', zorder=1, linewidth=2)
    
    # Plot the bottom right corner
    lons = [
        lon2d[-1, -1],
        lon2d[-1, -1] + dist_lon,
        lon2d[-1, -1] + dist_lon  + dist_lonlat,
        lon2d[-1, -1]  + dist_lonlat,
        lon2d[-1, -1]
    ]
    lats = [
        lat2d[-1, -1],
        lat2d[-1, -1] + dist_latlon,
        lat2d[-1, -1] + dist_lat + dist_latlon,
        lat2d[-1, -1] + dist_lat,
        lat2d[-1, -1]
    ]
        
    data_cell = ds['urmask'].values[-1, -1]

    lons = lons - abs(lon2d[-1, -1] - lon2d[-1, -1])/2 - dist_lon/2
    lats = lats - abs(lat2d[-1, -1] - lat2d[-1, -1])/2 - dist_lat/2  
    
    if data_cell == 1:
        ax.plot(lons, lats, color='#8A8D28', zorder=100, linewidth=2)
    elif data_cell == 0:
        ax.plot(lons, lats, color='#A52A2A', zorder=1, linewidth=2)


        
RCM_DICT = {
    'EUR-11': 
    { 
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-6',
    },
    'EUR-22': 
    {
        'REMO': 'GERICS_REMO2015',
    },
    'WAS-22': {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-7',
    },
    'EAS-22':
     {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'KNU_RegCM4-0',
    },
    'CAM-22':
     {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-7',
    },
    'SAM-22':
     {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-7',
    },
    'NAM-22':
     {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-6',
    },
    'AUS-22':
     {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-7',
    },
    'AFR-22':
     {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-7',
    },
    'SEA-22':
    {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-7',
    },
}


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


def classify_stations_from_ucdb_city(obs_attr_df, ucdb_city):
	"""Classify stations as urban/rural using the UCDB city polygon."""
	from shapely.geometry import Point

	if ucdb_city is None:
		raise ValueError("ucdb_city must be provided when using the UCDB-based classification.")

	city_geometry = getattr(ucdb_city, "geometry", ucdb_city)
	if hasattr(city_geometry, "unary_union"):
		city_geometry = city_geometry.unary_union

	if city_geometry.is_empty:
		raise ValueError("ucdb_city geometry is empty.")

	out = obs_attr_df.copy()
	station_lon = out["lon"].astype(float).to_numpy()
	station_lat = out["lat"].astype(float).to_numpy()

	inside_city = []
	for lon, lat in zip(station_lon, station_lat):
		inside_city.append(1 if city_geometry.covers(Point(lon, lat)) else 0)

	out["inside_city"] = np.asarray(inside_city, dtype=int)
	out["station_class"] = np.where(out["inside_city"] == 1, "urban", "rural")
	return out

def classify_stations_from_LCZ(obs_attr_df):
    """Classify stations as urban/rural using the LCZ classification."""
    if "LCZ" not in obs_attr_df.columns:
        raise ValueError("LCZ column not found in obs_attr_df.")

    out = obs_attr_df.copy()
    lczs = []
    for lcz in out["LCZ"]:
        if 'LCZ' in str(lcz):
            lczs.append(0)  # rural
        else:
            lczs.append(1)  # urban

    out["inside_city"] = lczs
    out["station_class"] =  np.where(out["inside_city"] == 1, "urban", "rural")

    return out    

def from_latlon_to_xy( lat, lon, proj):
    xy = rasterio.warp.transform(ccrs.PlateCarree(), proj, [lon], [lat])
    x, y = xy[0][0], xy[1][0]
    #print(f"Coordinates : x = {x:.2f} m, y = {y:.2f} m")
    return (x, y)

def compute_UHI_p2p(obs_attri, obs_ts, ds_RCM_urb, proj,
                   variable='tasmin',
                   urban_col='station_class',
                   urban_label='urban',
                   rural_label='rural',
                   mean=True,
                   ):
    """
    Compute point-to-point UHI from observations and RCM simulations.

    Parameters
    ----------
    obs_attri : dataframe
        Station metadata containing lat, lon and urban/rural classification.
    obs_ts : dataframe
        Time series dataframe (time x stations).
    ds_RCM_urb : xarray.Dataset
        RCM dataset containing variable on x/y grid.
    variable : str
        Variable name in ds_RCM_urb.
    mean : bool
        If True, average all urban/rural stations and gridcells.
        If False, return individual station/gridcell series.

    Returns
    -------
    dict
        Contains observational and simulated urban/rural series and UHI.
    """

    # --------------------------------------------------
    # Identify stations
    # --------------------------------------------------

    urban_stations = obs_attri.loc[
        obs_attri[urban_col] == urban_label, 'code'
    ].values

    rural_stations = obs_attri.loc[
        obs_attri[urban_col] == rural_label, 'code'
    ].values


    # --------------------------------------------------
    # Observations
    # --------------------------------------------------

    urb_obs = obs_ts[urban_stations]
    rur_obs = obs_ts[rural_stations]


    if mean:
        urb_obs = urb_obs.mean(axis=1)
        rur_obs = rur_obs.mean(axis=1)


    # --------------------------------------------------
    # Simulations
    # --------------------------------------------------

    urb_sim_list = []
    rur_sim_list = []


    for station in urban_stations:

        lat, lon = obs_attri.loc[
            obs_attri.code == station,
            ['lat','lon']
        ].values[0]

        x, y = from_latlon_to_xy(lat, lon, proj)

        urb_sim_list.append(
            ds_RCM_urb[variable]
            .sel(x=x, y=y, method='nearest')
        )


    for station in rural_stations:

        lat, lon = obs_attri.loc[
            obs_attri.code == station,
            ['lat','lon']
        ].values[0]

        x, y = from_latlon_to_xy(lat, lon, proj)

        rur_sim_list.append(
            ds_RCM_urb[variable]
            .sel(x=x, y=y, method='nearest')
        )


    urb_sim = xr.concat(
        urb_sim_list,
        dim='station'
    )

    rur_sim = xr.concat(
        rur_sim_list,
        dim='station'
    )


    if mean:
        urb_sim = urb_sim.mean(dim='station')
        rur_sim = rur_sim.mean(dim='station')


    return {
        'urb_obs': urb_obs,
        'rur_obs': rur_obs,
        'urb_sim': urb_sim,
        'rur_sim': rur_sim
    }

def plot_UHI_p2p_cycle(
        ax,
        uhi_data,
        mean=False,
        percentiles=[10],
        city_c='red',
        rural_c='green',
        plot_individual=True,
        label=None
    ):

    months = np.arange(1, 13)


    # =====================================================
    # Mean mode
    # =====================================================

    if mean:

        obs_uhi = (
            uhi_data['urb_obs'] -
            uhi_data['rur_obs']
        )

        sim_uhi = (
            uhi_data['urb_sim'] -
            uhi_data['rur_sim']
        )


        obs_cycle = (
            obs_uhi
            .groupby(obs_uhi.index.month)
            .mean()
        )

        sim_cycle = (
            sim_uhi
            .groupby('time.month')
            .mean()
        )


        ax.plot(
            months,
            obs_cycle,
            color=city_c,
            ls='--',
            marker='o',
            label='OBS UHI' if label is None else f'OBS UHI ({label})',
            lw=4,
        )


        ax.plot(
            months,
            sim_cycle,
            color=city_c,
            label='RCM UHI' if label is None else f'RCM UHI ({label})',
            lw=4
        )

        return obs_cycle, sim_cycle


    # =====================================================
    # Ensemble mode
    # =====================================================

    else:


        # -------------------------------------------------
        # Observations
        # -------------------------------------------------

        urb_obs = uhi_data['urb_obs']
        rur_obs = uhi_data['rur_obs']


        urb_obs_month = (
            urb_obs
            .groupby(urb_obs.index.month)
            .mean()
        )

        rur_obs_month = (
            rur_obs
            .groupby(rur_obs.index.month)
            .mean()
        )


        rural_obs_mean = (
            rur_obs_month.mean(axis=1)
        )


        # urban stations relative to rural mean
        obs_urban_uhi = (
            urb_obs_month.T -
            rural_obs_mean.values
        ).T


        # rural stations relative to rural mean
        obs_rural_diff = (
            rur_obs_month.T -
            rural_obs_mean.values
        ).T



        # -------------------------------------------------
        # Simulations
        # -------------------------------------------------

        urb_sim = uhi_data['urb_sim']
        rur_sim = uhi_data['rur_sim']


        urb_sim_month = (
            urb_sim
            .groupby('time.month')
            .mean()
        )

        rur_sim_month = (
            rur_sim
            .groupby('time.month')
            .mean()
        )


        rural_sim_mean = (
            rur_sim_month.mean(dim='station')
        )


        # urban gridcells relative to rural mean
        sim_urban_uhi = (
            urb_sim_month -
            rural_sim_mean
        )


        # rural gridcells relative to rural mean
        sim_rural_diff = (
            rur_sim_month -
            rural_sim_mean
        )



        # =================================================
        # Helper plotting routine
        # =================================================

        def plot_ensemble(data, color, label):

            if isinstance(data, xr.DataArray):
                values = data.values
            else:
                values = data.values.T


            # ensure shape = (members, months)
            if values.shape[0] == 12:
                values = values.T
       
            mean_val = np.nanmean(
                values,
                axis=0
            )


            lower = np.nanpercentile(
                values,
                percentiles[0],
                axis=0
            )

            upper = np.nanpercentile(
                values,
                100-percentiles[0],
                axis=0
            )


            ax.fill_between(
                months,
                lower,
                upper,
                color=color,
                alpha=0.05
            )
            #plot the percentile lines
            for bound in [lower, upper]:
                ax.plot(
                    months,
                    bound,
                    color=color,
                    alpha=0.6,
                    linewidth=0.8,
                    ls='--' if 'OBS' in label else '-'
                )

            # individual station/gridcell curves
            if plot_individual:
                for member in values:
                    ax.plot(
                        months,
                        member,
                        color='purple' if 'urban' in label else 'lime',
                        alpha=1,
                        linewidth=0.8, 
                        ls='--' if 'OBS' in label else '-'
                    )


            # ensemble mean
            ax.plot(
                months,
                mean_val,
                color=color,
                linewidth=4,
                label=label, 
                ls='--' if 'OBS' in label else '-', 
            )
            return mean_val


        # =================================================
        # Plot urban UHI
        # =================================================

        urb_obs_mean = plot_ensemble(
            obs_urban_uhi,
            city_c,
            'OBS urban - rural mean'
        )


        urb_sim_mean = plot_ensemble(
            sim_urban_uhi,
            city_c,
            'RCM urban - rural mean'
        )



        # =================================================
        # Plot rural deviation from rural mean
        # =================================================



        rur_sim_mean = plot_ensemble(
            sim_rural_diff,
            rural_c,
            'RCM rural - rural mean'
        )


        # observational mean line style requested
        rur_obs_mean = np.nanmean(
            obs_urban_uhi,
            axis=0
        )





    ax.axhline(
        0,
        color='black',
        linewidth=0.8
    )


    ax.set_xticks(months)

    ax.set_xlabel(
        'Month'
    )

    ax.set_ylabel(
        'Temperature difference (K)'
    )

    ax.legend()

    return urb_obs_mean, urb_sim_mean 


def build_polygons_from_urmask(urmask, mask_name='urmask'):
    from shapely.geometry import Point, Polygon, box, MultiPolygon

    x = urmask["x"].values
    y = urmask["y"].values

    dx = np.diff(x).mean()
    dy = np.diff(y).mean()

    x_edges = np.concatenate(([x[0]-dx/2], x[:-1]+dx/2, [x[-1]+dx/2]))
    y_edges = np.concatenate(([y[0]-dy/2], y[:-1]+dy/2, [y[-1]+dy/2]))

        
    urban_mask = urmask[mask_name].values == 1
    rural_mask = urmask[mask_name].values == 0
    masks = [urban_mask, rural_mask]

    polygons = []

    for mask in masks:
        cells = []
        ny, nx = mask.shape

        for j in range(ny):
            for i in range(nx):
                if mask[j, i]:
                    cells.append(
                        box(
                            x_edges[i],
                            y_edges[j],
                            x_edges[i+1],
                            y_edges[j+1],
                        )
                    )
        polygons.append(unary_union(cells))
    return polygons[0], polygons[1]  # urban_polygon, rural_polygon


def plot_polygon_boundaries(geom, ax, plot_proj=ccrs.PlateCarree(), **kwargs):
    from shapely.geometry import Point, Polygon, box, MultiPolygon

    def _plot_boundary(geometry):
        if isinstance(geometry, Polygon):
            x, y = geometry.exterior.xy
            ax.plot(x, y, transform=plot_proj, **kwargs)

            # Interior boundaries (holes)
            for interior in geometry.interiors:
                x, y = interior.xy
                ax.plot(x, y, transform=plot_proj, **kwargs)

        elif isinstance(geometry, MultiPolygon):
            for poly in geometry.geoms:
                _plot_boundary(poly)

    _plot_boundary(geom)
def plot_blend_on_ax(
    plot_proj, ax, da_anom, vmaxxx,
    rural_polygon, urban_polygon_blend,
    x_station, y_station, uhi_blend_obs,
    ucdb_city=None, extent=None, top_labels=True, left_labels=True
):
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import matplotlib.ticker as mticker

    mesh = ax.pcolormesh(
        da_anom['x'],
        da_anom['y'],
        da_anom.values,
        vmin=-vmaxxx,
        vmax=vmaxxx,
        shading='auto',
        cmap='bwr',
        transform=plot_proj,
    )

    if ucdb_city is not None:
        ucdb_base = (
            ucdb_city if ucdb_city.crs is not None
            else ucdb_city.set_crs(epsg=4326)
        )
        ucdb_plot = ucdb_base.to_crs(plot_proj.proj4_init)

        ax.add_geometries(
            ucdb_plot.geometry,
            crs=plot_proj,
            facecolor="none",
            edgecolor="#ff66ff",
            linewidth=2,
        )

    plot_polygon_boundaries(
        rural_polygon, ax,
        color='#8A8D28', linewidth=2,
        plot_proj=plot_proj
    )

    plot_polygon_boundaries(
        urban_polygon_blend[0], ax,
        color='#A52A2A', linewidth=2,
        plot_proj=plot_proj
    )

    ax.scatter(
        x_station, y_station,
        c=uhi_blend_obs.mean(),
        cmap='bwr',
        vmin=-vmaxxx, vmax=vmaxxx,
        marker='o', s=60,
        linewidths=2,
        edgecolors='#A52A2A',
        zorder=10000
    )

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        linewidth=0.5,
        linestyle='--',
        draw_labels=True,
        x_inline=False,
        y_inline=False
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = top_labels
    gl.left_labels = left_labels
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlocator = mticker.MultipleLocator(0.5)
    gl.ylocator = mticker.MultipleLocator(0.5)

    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    return ax
        