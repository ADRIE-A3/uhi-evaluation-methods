# This file is part of URCLIMASK (https://github.com/FPS-URB-RCC/URCLIMASK),
# Copyright 2023, European Union. Licensed under the Apache License, Version 2.0
# (see LICENSE in this directory).
#
# Modified by Leon Adriaensen (Ghent University), 2025-2026, from URCLIMASK commit b779ebc (2025-10-01):
#   - urban/rural station grouping via a station_class column and obs/sim time-overlap checks
#     (_get_obs_codes_by_group, _filter_codes_in_timeseries, check_obs_sim_overlap);
#   - observed vs simulated UHI time series and seasonal subsetting
#     (_compute_uhi_timeseries_obs_sim, _season_months, _align_ds_with_mask);
#   - additional plots (plot_reduced_annual_cycle, plot_UI_urclim_visualisation,
#     plot_uhi_quantile_comparison, plot_diurnal_cycles_and_uhi_from_precomputed).

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import os
import pandas as pd
import xarray as xr
from icecream import ic
from itertools import product
from urclimask.urban_areas import plot_urban_polygon
from shapely.geometry import Point, Polygon, box, MultiPolygon
from urclimask.utils import build_polygons_from_urmask, plot_polygon_boundaries
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker



class UrbanIsland:
    def __init__(
        self,
        *,
        ds = None, 
        urban_vicinity = None, 
        rcm = None,
        anomaly = 'abs',
        period= 'Annual',
        obs_attributes = pd.DataFrame(), 
        obs_timeseries = pd.DataFrame(), 
        ):
        """
        Parameters:
        - ds (xarray.Dataset): Dataset containing the variable of interest.
        - urban_vicinity (xarray.DataArray or ndarray): Urban mask identifying urban grid cells.
        - anomaly (str): Type of anomaly to compute; either 'rel' (relative) or 'abs' (absolute).
        - period (str): Time period to filter ('Annual', 'jja', 'djf').
        - obs_attributes (list): Metadata for the observation points (e.g., locations).
        - obs_timeseries (list): Time series data for each observation point.
        """       
        self.ds = ds
        self.urban_vicinity = urban_vicinity
        self.rcm = rcm
        self.anomaly = anomaly
        self.period = period
        self.obs_attr = obs_attributes
        self.obs_time = obs_timeseries

    def _get_obs_codes_by_group(self):
        """Return station codes classified as urban and rural.

        Preferred classification is `station_class` with values {'urban','rural','neither'}.
        Backward compatibility: if `station_class` is absent, uses legacy boolean
        `inside_city` where True->urban and False->rural.
        """
        if self.obs_attr.empty:
            return pd.Index([], dtype=object), pd.Index([], dtype=object)

        obs_codes = self.obs_attr['code'].astype(str)

        if 'station_class' in self.obs_attr.columns:
            classes = self.obs_attr['station_class'].astype(str).str.lower()
            urban_codes = obs_codes[classes == 'urban']
            rural_codes = obs_codes[classes == 'rural']
            return urban_codes, rural_codes

        if 'inside_city' in self.obs_attr.columns:
            urban_codes = obs_codes[self.obs_attr['inside_city'] == True]
            rural_codes = obs_codes[self.obs_attr['inside_city'] == False]
            return urban_codes, rural_codes

        return pd.Index([], dtype=object), pd.Index([], dtype=object)

    def _filter_codes_in_timeseries(self, codes):
        """Keep station codes present in observation timeseries columns."""
        if self.obs_time is None or getattr(self.obs_time, 'empty', True):
            return []
        available = set(self.obs_time.columns.astype(str))
        return [code for code in codes.astype(str) if code in available]

    def check_obs_sim_overlap(self, drop_unaligned=True, drop_empty_stations=True, verbose=True):
        """Check temporal overlap between observations and simulation.

        Parameters
        ----------
        drop_unaligned : bool, default=True
            If True, keeps only observation timestamps that also exist in the
            simulation time axis and updates ``self.obs_time`` in place.
        drop_empty_stations : bool, default=True
            When ``drop_unaligned`` is True, drop observation columns that
            become fully missing after temporal alignment.
        verbose : bool, default=True
            If True, print a compact overlap report.

        Returns
        -------
        dict
            Summary with counts, fractions, and dropped elements.
        """
        report = {
            'status': 'ok',
            'drop_unaligned': bool(drop_unaligned),
            'drop_empty_stations': bool(drop_empty_stations),
        }

        if self.ds is None or 'time' not in self.ds.coords:
            report.update({'status': 'missing-simulation-time', 'message': 'Simulation has no time coordinate.'})
            if verbose:
                print('[OVERLAP] Simulation has no time coordinate.')
            return report

        if self.obs_time is None or getattr(self.obs_time, 'empty', True):
            report.update({'status': 'missing-observations', 'message': 'Observation timeseries is empty.'})
            if verbose:
                print('[OVERLAP] Observation timeseries is empty.')
            return report

        if not isinstance(self.obs_time, pd.DataFrame):
            self.obs_time = pd.DataFrame(self.obs_time)

        obs_index_raw = self.obs_time.index
        if isinstance(obs_index_raw, pd.DatetimeIndex):
            obs_index = obs_index_raw
        else:
            obs_index = pd.to_datetime(obs_index_raw, errors='coerce')

        sim_index_raw = self.ds.indexes.get('time', pd.Index(self.ds['time'].values))
        if isinstance(sim_index_raw, pd.DatetimeIndex):
            sim_index = sim_index_raw
        else:
            sim_index = pd.to_datetime(sim_index_raw.astype(str), errors='coerce')

        obs_index = pd.DatetimeIndex(obs_index).dropna().unique().sort_values()
        sim_index = pd.DatetimeIndex(sim_index).dropna().unique().sort_values()

        common_times = obs_index.intersection(sim_index)
        obs_only = obs_index.difference(sim_index)
        sim_only = sim_index.difference(obs_index)

        report.update({
            'obs_n_times': int(len(obs_index)),
            'sim_n_times': int(len(sim_index)),
            'overlap_n_times': int(len(common_times)),
            'obs_only_n_times': int(len(obs_only)),
            'sim_only_n_times': int(len(sim_only)),
            'obs_overlap_fraction': float(len(common_times) / len(obs_index)) if len(obs_index) else np.nan,
            'sim_overlap_fraction': float(len(common_times) / len(sim_index)) if len(sim_index) else np.nan,
            'obs_start': str(obs_index.min()) if len(obs_index) else None,
            'obs_end': str(obs_index.max()) if len(obs_index) else None,
            'sim_start': str(sim_index.min()) if len(sim_index) else None,
            'sim_end': str(sim_index.max()) if len(sim_index) else None,
            'overlap_start': str(common_times.min()) if len(common_times) else None,
            'overlap_end': str(common_times.max()) if len(common_times) else None,
        })

        if verbose:
            print('[OVERLAP] Observation vs simulation time coverage')
            print(f"  obs times: {report['obs_n_times']} ({report['obs_start']} -> {report['obs_end']})")
            print(f"  sim times: {report['sim_n_times']} ({report['sim_start']} -> {report['sim_end']})")
            print(
                f"  common times: {report['overlap_n_times']} "
                f"(obs overlap: {100 * report['obs_overlap_fraction']:.1f}%, "
                f"sim overlap: {100 * report['sim_overlap_fraction']:.1f}%)"
            )
            print(f"  obs-only times: {report['obs_only_n_times']}")
            print(f"  sim-only times: {report['sim_only_n_times']}")

        if drop_unaligned:
            obs_rows_before = int(self.obs_time.shape[0])
            obs_cols_before = int(self.obs_time.shape[1])

            aligned_obs = self.obs_time.copy()
            aligned_obs.index = pd.to_datetime(aligned_obs.index, errors='coerce')
            aligned_obs = aligned_obs[aligned_obs.index.notna()]
            aligned_obs = aligned_obs.loc[aligned_obs.index.isin(common_times)].sort_index()

            dropped_station_codes = []
            if drop_empty_stations and not aligned_obs.empty:
                keep_cols = aligned_obs.columns[aligned_obs.notna().any(axis=0)]
                dropped_station_codes = [str(c) for c in aligned_obs.columns if c not in keep_cols]
                aligned_obs = aligned_obs[keep_cols]

            self.obs_time = aligned_obs

            report.update({
                'obs_rows_before': obs_rows_before,
                'obs_rows_after': int(self.obs_time.shape[0]),
                'obs_cols_before': obs_cols_before,
                'obs_cols_after': int(self.obs_time.shape[1]),
                'dropped_station_codes': dropped_station_codes,
            })

            if verbose:
                print('[OVERLAP] Applied alignment to observations')
                print(f"  rows: {obs_rows_before} -> {report['obs_rows_after']}")
                print(f"  stations: {obs_cols_before} -> {report['obs_cols_after']}")
                if dropped_station_codes:
                    print(f"  dropped empty stations after alignment: {dropped_station_codes}")

        return report

    def _align_ds_with_mask(self, ds):
        """
        Align dataset to match the urban_vicinity mask's spatial grid.
        Uses nearest-neighbor interpolation on lat/lon values to handle different projections.
        
        Parameters:
        - ds: xarray DataArray or Dataset to alignerr
        
        Returns:
        - Aligned dataset/dataarray with same x/y coordinates as urban_vicinity
        """
        mask = self.urban_vicinity['urmask']
        
        print(f"[ALIGN] Aligning dataset shape {ds.shape} to mask shape {mask.shape}")
        
        # Get coordinate names upfront
        mask_x_name = mask.cf['x'].name if hasattr(mask, 'cf') else 'x'
        mask_y_name = mask.cf['y'].name if hasattr(mask, 'cf') else 'y'
        
        try:
            # Use scipy for more robust nearest-neighbor interpolation
            from scipy.spatial import cKDTree
            
            # Get lat/lon from both datasets
            ds_lons = ds['lon'].values.ravel()
            ds_lats = ds['lat'].values.ravel()
            mask_lons = mask['lon'].values.ravel()
            mask_lats = mask['lat'].values.ravel()
            
            # Build KDTree for efficient nearest neighbor search
            tree = cKDTree(np.column_stack([ds_lons, ds_lats]))
            distances, indices = tree.query(np.column_stack([mask_lons, mask_lats]))
            
            # Reshape dataset to (time, lat*lon) and get aligned values
            ds_reshaped = ds.stack(grid=ds.dims[-2:])  # Stack x,y dims to single 'grid' dim
            ds_aligned = ds_reshaped.isel(grid=indices)
            ds_aligned = ds_aligned.unstack('grid')  # Unstack back to 2D spatial
            
            # Copy over lat/lon AND x/y coordinates from mask
            ds_aligned = ds_aligned.assign_coords({
                'lon': (ds_aligned.dims[-2:], mask['lon'].values),
                'lat': (ds_aligned.dims[-2:], mask['lat'].values),
                mask_x_name: mask[mask_x_name].values,
                mask_y_name: mask[mask_y_name].values,
            })
            
            print(f"[ALIGN] Successfully aligned using lat/lon (max distance: {distances.max():.6f}°)")
        except ImportError:
            print("[ALIGN] scipy not available, falling back to reindex_like")
            ds_aligned = ds.reindex_like(mask, method='nearest')
        
        return ds_aligned

    def _get_spatial_dim_names(self, da):
        """
        Return the y/x dimension names for a DataArray.
        Prefers cf_xarray when available and falls back to common dim names.
        """
        try:
            x_name = da.cf['x'].name
            y_name = da.cf['y'].name
            return y_name, x_name
        except Exception:
            spatial_dims = [dim for dim in da.dims if dim not in ('time', 'month', 'hour')]
            if len(spatial_dims) >= 2:
                if 'y' in spatial_dims and 'x' in spatial_dims:
                    return 'y', 'x'
                return spatial_dims[0], spatial_dims[1]
            raise KeyError(f"Could not infer spatial dimensions from {da.dims}")

    def compute_spatial_climatology(self, urban_mask_name='urmask', update_ds=True):
        """
        Computes the spatial climatology and corresponding anomaly from both the model 
        dataset and the observations, if available.
    
        Outputs:
        - self.ds_spatial_climatology: Spatial anomaly computed from the model.
        - self.obs_spatial_climatology: Spatial anomaly computed from the observations (if available).
        """
        print(f"\n[COMPUTE] Input shape: {self.ds.shape}, dims: {self.ds.dims}")
        
        # Align dataset with mask before computing
        ds_aligned = self._align_ds_with_mask(self.ds)
        print(f"[COMPUTE] After alignment: shape={ds_aligned.shape}")
        
        # calculate climatology and anomaly from the model
        ds_var_period_mean = ds_aligned.mean('time').compute()
        print(f"[COMPUTE] After mean('time'): shape={ds_var_period_mean.shape}")
        print(f"[COMPUTE] After mean - sample values: min={float(ds_var_period_mean.min()):.4f}, max={float(ds_var_period_mean.max()):.4f}, mean={float(ds_var_period_mean.mean()):.4f}")
        
        print(f"[COMPUTE] Mask shape: {self.urban_vicinity[urban_mask_name].shape}")
        
        # Check rural mask
        rural_mask = self.urban_vicinity[urban_mask_name] == 0
        print(f"[COMPUTE] Mask unique values: {np.unique(self.urban_vicinity[urban_mask_name].values)}")
        print(f"[COMPUTE] rural_mask (urmask==0) True count: {rural_mask.sum().item() if hasattr(rural_mask.sum(), 'item') else rural_mask.sum()}")
        
        rural_count = int(rural_mask.sum().item() if hasattr(rural_mask.sum(), 'item') else rural_mask.sum())
        print(f"[COMPUTE] Rural cells count: {rural_count}")
        
        # Apply where operation
        print(f"[COMPUTE] Before where: ds shape={ds_var_period_mean.shape}, all finite count={np.isfinite(ds_var_period_mean.values).sum()}")
        print(f"[COMPUTE] rural_mask shape={rural_mask.shape}, True count={rural_mask.sum().item()}")
        print(f"[COMPUTE] ds dims={ds_var_period_mean.dims}, coords={list(ds_var_period_mean.coords)}")
        print(f"[COMPUTE] mask dims={rural_mask.dims}, coords={list(rural_mask.coords)}")
        
        ds_rural = ds_var_period_mean.where(rural_mask)
        print(f"[COMPUTE] After where: ds shape={ds_rural.shape}, size={ds_rural.size}")
        rural_finite = int(np.isfinite(ds_rural.values).sum())
        print(f"[COMPUTE] After .where(rural_mask): finite values={rural_finite}/{ds_rural.size}")
        
        rural_mean = ds_rural.mean().compute()
        print(f"[COMPUTE] Rural mean value: {float(rural_mean):.6f}")
        
        ds_anomaly = ds_var_period_mean - rural_mean        
        print(f"[COMPUTE] Final anomaly: shape={ds_anomaly.shape}, min={float(ds_anomaly.min()):.6f}, max={float(ds_anomaly.max()):.6f}")
        
        ds_anomaly.attrs['units'] = self.ds.attrs.get('units', 'unknown')

        # calculate climatology from the observations
        obs_anomaly = None
        if not self.obs_attr.empty:
            obs_period_mean = self.obs_time.mean()
            _, rural_codes = self._get_obs_codes_by_group()
            rural_codes = self._filter_codes_in_timeseries(rural_codes)

            if len(rural_codes) > 0:
                obs_rur_mean = obs_period_mean[rural_codes].mean(skipna=True)
                obs_anomaly = (obs_period_mean - obs_rur_mean)
            else:
                print("[COMPUTE] No rural stations available for observation anomaly computation.")
        else:
            print("[COMPUTE] No observation anomaly computed (obs_attr is empty).")
        if self.anomaly == 'rel':
            ds_anomaly = (ds_anomaly/ds_var_period_mean)*100
            ds_anomaly.attrs['units'] = "%"
            if obs_anomaly is not None:
                obs_anomaly = (obs_anomaly/obs_period_mean)*100
        if update_ds:
            self.ds_spatial_climatology = ds_anomaly.compute()
            if obs_anomaly is not None:
                self.obs_spatial_climatology = obs_anomaly
        else:
            return ds_anomaly.compute(), obs_anomaly
        
        
    def compute_annual_cycle(self):
        """
        Computes the hourly climatology (annual cycle) and corresponding anomaly 
        from the model dataset and the observation, if available.

        Output:
        - self.ds_annual_cycle: Monthly anomaly for each month of the year from the model.
        - self.obs_annual_cycle: Monthly anomaly for each month of the year from the observation (if available).
        """
        # Align dataset with mask before computing
        ds_aligned = self._align_ds_with_mask(self.ds)
        y_dim, x_dim = self._get_spatial_dim_names(ds_aligned)
        
        is_rural = self.urban_vicinity['urmask'] == 0
        is_urban = self.urban_vicinity['urmask'] == 1
        rural_mean = (ds_aligned
            .where(is_rural)
            .groupby('time.month')
            .mean(dim = [y_dim, x_dim, 'time'])
            .compute()
        )       
        ds_period_mean = ds_aligned.groupby('time.month').mean('time')                  
        ds_anomaly = (ds_period_mean - rural_mean).compute()
        ds_anomaly.attrs['units'] = self.ds.attrs.get('units', 'unknown')

        # calculate climatology from the observations
        obs_anomaly = None
        if not self.obs_attr.empty:
            codes_ins_city, codes_out_city = self._get_obs_codes_by_group()
            codes_ins_city = self._filter_codes_in_timeseries(codes_ins_city)
            codes_out_city = self._filter_codes_in_timeseries(codes_out_city)
            obs_month = self.obs_time.groupby(self.obs_time.index.month).mean()
            obs_month_mean = pd.DataFrame(index=obs_month.index)
            obs_month_mean['rural_mean'] = obs_month[codes_out_city].mean(axis=1).values if len(codes_out_city) > 0 else np.nan
            obs_month_mean['urban_mean'] = obs_month[codes_ins_city].mean(axis=1).values if len(codes_ins_city) > 0 else np.nan
            obs_anomaly = obs_month_mean.sub(obs_month_mean['rural_mean'], axis = 0)
            raw_anomaly = obs_month.sub(obs_month_mean['rural_mean'], axis=0)
            for code in obs_month.columns:
                obs_anomaly[code] = raw_anomaly[code]

        if self.anomaly == 'rel':
            ds_anomaly = (ds_anomaly / ds_period_mean) * 100
            ds_anomaly.attrs['units'] = "%"
            if obs_anomaly is not None:
                obs_anomaly = obs_anomaly.div(obs_month_mean['rural_mean'], axis=0) * 100

        self.ds_annual_cycle = ds_anomaly
        self.obs_annual_cycle = obs_anomaly
    
    
    def compute_daily_cycle(self):
        """
        Computes the hourly climatology (daily cycle) and corresponding anomaly for the specified period
        from the model dataset and the observations, if available.
    
        Output:
        - self.ds_daily_cycle: Hourly anomaly for each hour of the day from the model.
        - self.obs_daily_cycle: Hourly anomaly for each hour of the day from the observations (if available).
        """
        # Align dataset with mask before computing
        ds_var = self._align_ds_with_mask(self.ds)
        y_dim, x_dim = self._get_spatial_dim_names(ds_var)
        
        if self.period == 'jja':
            ds_var = ds_var.sel(time=ds_var['time'].dt.month.isin([6, 7, 8]))
        elif self.period == 'djf':
            ds_var = ds_var.sel(time=ds_var['time'].dt.month.isin([12, 1, 2]))
    
        is_rural = self.urban_vicinity['urmask'] == 0
        is_urban = self.urban_vicinity['urmask'] == 1
        rounded_time = ds_var.time.dt.round('h')
        ds_var = ds_var.assign_coords(hour=rounded_time.dt.hour)
        rural_mean = (
            ds_var
            .where(is_rural)
            .groupby('time.hour')
            .mean(dim=[y_dim, x_dim, 'time'])
            .compute()
        )
    
        ds_period_mean = ds_var.groupby('time.hour').mean('time')
    
        ds_anomaly = (ds_period_mean - rural_mean).compute()
        ds_anomaly.attrs['units'] = ds_var.attrs.get('units', 'unknown')
    
        obs_anomaly = None
    
        if not self.obs_attr.empty:    
            codes_ins_city, codes_out_city = self._get_obs_codes_by_group()
            codes_ins_city = self._filter_codes_in_timeseries(codes_ins_city)
            codes_out_city = self._filter_codes_in_timeseries(codes_out_city)
    
            obs_hour = self.obs_time.groupby(self.obs_time.index.hour).mean()
    
            obs_hour_mean = pd.DataFrame(index=obs_hour.index)
            obs_hour_mean['rural_mean'] = obs_hour[codes_out_city].mean(axis=1).values if len(codes_out_city) > 0 else np.nan
            obs_hour_mean['urban_mean'] = obs_hour[codes_ins_city].mean(axis=1).values if len(codes_ins_city) > 0 else np.nan
    
            obs_anomaly = obs_hour_mean.sub(obs_hour_mean['rural_mean'], axis=0)
            raw_anomaly = obs_hour.sub(obs_hour_mean['rural_mean'], axis=0)
            for code in obs_hour.columns:
                obs_anomaly[code] = raw_anomaly[code]
    
        if self.anomaly == 'rel':
            ds_anomaly = (ds_anomaly / ds_period_mean) * 100
            ds_anomaly.attrs['units'] = "%"
            if obs_anomaly is not None:
                obs_anomaly = obs_anomaly.div(obs_hour_mean['rural_mean'], axis=0) * 100
    
        self.ds_daily_cycle = ds_anomaly
        self.obs_daily_cycle = obs_anomaly

    
    def plot_UI_urclim_visualisation(
        self,
        proj,
        ax,
        mask_name = 'urmask',
        p2p_plot = False,
        ds_anomaly = None,
        obs_anomaly = None,
        city_name = None,
        ucdb_city = None, 
        alpha_urb_borders = 1, 
        linewidth_urb_borders = 2, 
        vmax = None,
        bottom_ticks = False,
        left_ticks = False,
        plot_cbar = False,
        ):
        from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
        import matplotlib.ticker as mticker

        
        ds_anomaly = self.compute_spatial_climatology(urban_mask_name=mask_name, update_ds=False)[0]
            
        if not self.obs_attr.empty and obs_anomaly is None:
            if hasattr(self, 'obs_spatial_climatology'):
                #print(self.obs_spatial_climatology)
                obs_anomaly = self.obs_spatial_climatology
        
    
        # Compute the maximum absolute value
        if vmax:
            max_abs_value = vmax
        else:
            max_abs_value = abs(ds_anomaly).max().item()
        #print(ds_anomaly)
        im1 = ax.pcolormesh(ds_anomaly.x, ds_anomaly.y, 
                            ds_anomaly.values,
                            transform=proj,
                            cmap='bwr', alpha = 0.7,
                            vmin = -max_abs_value, vmax = max_abs_value)

        #make sure the lon lat values correspond to the stations code in the obs_anomaly dataframe
        obs_anomaly = obs_anomaly.reindex(self.obs_attr['code'], axis=1)
        print(obs_anomaly)

        if obs_anomaly is not None:
            sc = ax.scatter(
                self.obs_attr.x, 
                self.obs_attr.y,
                c=obs_anomaly.values, 
                label = self.obs_attr['code'],
                cmap='bwr',
                vmin=-max_abs_value, vmax=max_abs_value,
                marker='o', 
                s=60, 
                linewidths=2,
                edgecolors=['#A52A2A' if self.obs_attr.iloc[x]['inside_city'] == 1  else '#8A8D28' for x in range(len(self.obs_attr))], 
                zorder=10000
            )
        if plot_cbar:
            cbar = plt.colorbar(im1, ax = ax, shrink = 0.8)
            unit = ds_anomaly.attrs.get('units', 'unknown') 
            
            cbar.set_label(f"{unit}", rotation = 90, fontsize = 14)
            
        ax.coastlines(ls='--', color='black', linewidth=0.8)        
        if ucdb_city is not None: 
                ucdb_base = ucdb_city if getattr(ucdb_city, "crs", None) is not None else ucdb_city.set_crs(epsg=4326, allow_override=True) 
                proj4 = getattr(proj, "proj4_init", None) 
                if proj4 is not None: 
                    ucdb_plot = ucdb_base.to_crs(proj4) 
                    ucdb_plot_crs = proj 
                ax.add_geometries( ucdb_plot.geometry, crs=ucdb_plot_crs, facecolor="none", edgecolor="#ff66ff", linewidth=2)

        urban_polygon, rural_polygon = build_polygons_from_urmask(self.urban_vicinity, mask_name=mask_name)
        plot_polygon_boundaries( urban_polygon, ax, plot_proj=proj, color='#A52A2A', linewidth=2,)
        plot_polygon_boundaries( rural_polygon, ax, plot_proj=proj, color='#8A8D28', linewidth=2,)


        gl = ax.gridlines(    crs=ccrs.PlateCarree(),   linewidth=0.5,   linestyle='--', draw_labels=True ,x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False 
        gl.bottom_labels = bottom_ticks
        gl.left_labels = left_ticks
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()

        gl.xlocator = mticker.MultipleLocator(0.1)
        gl.ylocator = mticker.MultipleLocator(0.1)

        #plt.subplots_adjust(wspace=0.1, hspace=0.1)
        
        return ax


    def plot_UI_map(
        self,
        *,
        ds_anomaly = None,
        obs_anomaly = None,
        city_name = None,
        ucdb_city = None, 
        alpha_urb_borders = 1, 
        linewidth_urb_borders = 2, 
        vmax = None,
        ):
        """
        Plots the Urban Island effect using model and optional observation anomalies.
    
        Parameters:
        - ds_anomaly (xarray.DataArray, optional): Spatial anomaly from the model. If not provided, it will be computed.
        - obs_anomaly (numpy.ndarray, optional): Spatial anomaly from the observations. If not provided, it will be computed.
        - city_name (str, optional): Name of the city to include in the plot title.
        - ucdb_city: GeoDataFrame containing the city's geometry.
        - alpha_urb_borders (float): Transparency level of the urban borders (0 to 1).
        - linewidth_urb_borders (float): Line width of the urban borders.
        - vmax (float, optional): Maximum absolute value for the color scale. If not provided, it is computed from the data.
    
        Outputs:
        - fig (matplotlib.figure.Figure): The resulting figure object.
        """
        
        if ds_anomaly is None:
            if not hasattr(self, 'ds_spatial_climatology'):
                self.compute_spatial_climatology()
            ds_anomaly = self.ds_spatial_climatology
            
        if not self.obs_attr.empty and obs_anomaly is None:
            if hasattr(self, 'obs_spatial_climatology'):
                #print(self.obs_spatial_climatology)
                obs_anomaly = self.obs_spatial_climatology
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(12, 6))
    
        # Compute the maximum absolute value
        if vmax:
            max_abs_value = vmax
        else:
            max_abs_value = abs(ds_anomaly).max().item()
        
        im1 = ax.pcolormesh(ds_anomaly.lon, ds_anomaly.lat, 
                            ds_anomaly.values,
                            cmap='bwr', alpha = 0.7,
                            vmin = -max_abs_value, vmax = max_abs_value)
        #make sure the lon lat values correspond to the stations code in the obs_anomaly dataframe
        obs_anomaly = obs_anomaly.reindex(self.obs_attr['code'], axis=1)
        print(obs_anomaly)
        if obs_anomaly is not None:
            sc = ax.scatter(
                self.obs_attr.lon, 
                self.obs_attr.lat,
                c=obs_anomaly.values, 
                label = self.obs_attr['code'],
                cmap='bwr',
                vmin=-max_abs_value, vmax=max_abs_value,
                marker='o', 
                s=40, 
                edgecolors='gray', 
                zorder=10000
            )
        if ucdb_city is not None:
            ucdb_city.plot(ax=ax, facecolor="none", transform=proj, edgecolor="#ff66ff", linewidth=2, zorder=1000)

        cbar = fig.colorbar(im1, ax = ax)
        unit = ds_anomaly.attrs.get('units', 'unknown') 
        
        cbar.set_label(f"{unit}", rotation = 90, fontsize = 14)
        
        if self.rcm:
            ax.set_title(f"Urban Island for {city_name} (variable: {ds_anomaly.name}, rcm: {self.rcm})", fontsize=18)
        else:
            ax.set_title(f"Urban Island for {city_name} (variable: {ds_anomaly.name})", fontsize=18)     

        #ax.coastlines()
        plot_urban_polygon(self.urban_vicinity, ax)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        
        return fig

    def plot_UI_annual_cycle(
        self,
        *,
        ds_anomaly = None,
        obs_anomaly = None,
        percentiles = [5],
        gridcell_series = True, 
        city_name = None, 
        ax = None,            
        vmax = None, 
        vmin = None):
        '''
        Plots the annual cycle of the Urban Island effect using model and optional observation data.
        
        Parameters:
        - ds_anomaly (xarray.DataArray, optional): Monthly anomaly from the model. If not provided, it will be computed.
        - obs_anomaly (pandas.DataFrame, optional): Monthly anomaly from observations. If not provided, it will be computed.
        - percentiles (list of int): Percentiles to shade around the mean values (e.g., [5] for 5th–95th range).
        - gridcell_series (bool): Whether to plot individual grid cell time series as transparent lines.
        - city_name (str, optional): Name of the city to include in the plot title.
        - ax (matplotlib.axes.Axes, optional): Existing axis to plot on. If None, a new figure and axis are created.
        - vmax (float, optional): Upper limit for the y-axis.
        - vmin (float, optional): Lower limit for the y-axis.
        
        Outputs:
        - fig (matplotlib.figure.Figure): The resulting figure object.
        '''

        if ds_anomaly is None:
            if not hasattr(self, 'ds_annual_cycle'):
                self.compute_annual_cycle()
            ds_anomaly = self.ds_annual_cycle
        
        if not self.obs_attr.empty and obs_anomaly is None:
            if hasattr(self, 'obs_annual_cycle'):
                obs_anomaly = self.obs_annual_cycle
        
        y_dim, x_dim = self._get_spatial_dim_names(ds_anomaly)
        
        is_rural = self.urban_vicinity['urmask'] == 0
        is_urban = self.urban_vicinity['urmask'] == 1

        rural_anomaly = ds_anomaly.where(is_rural)
        urban_anomaly = ds_anomaly.where(is_urban)
    
        urban_mean = urban_anomaly.mean(dim = [y_dim, x_dim]).compute()
        rural_mean = rural_anomaly.mean(dim = [y_dim, x_dim]).compute()

        urban_area_legend = False
        not_urban_area_legend = False
        fig, ax = plt.subplots(figsize=(15, 7))

        (urban_mean).plot(ax=ax,  color = '#A52A2A', linestyle='-', 
                                         linewidth = 4, label='Urban mean')
                             
        (rural_mean-rural_mean).plot(ax=ax,  color = '#8A8D28', linestyle='-', 
                                     linewidth = 4, label='Vicinity mean')
                             
        if percentiles:
            # Fill within percentiles
            axis = [rural_anomaly.get_axis_num(x_dim),
                rural_anomaly.get_axis_num(y_dim)]
            colors = ['#8A8D28', '#A52A2A']
            for index, anom in enumerate([ rural_anomaly, urban_anomaly]):
                for percentile in percentiles:
                    lower_percentile = np.nanpercentile(anom, percentile, axis=axis)
                    upper_percentile = np.nanpercentile(anom, 100-percentile, axis=axis)
                    ax.fill_between(
                        rural_anomaly['month'],
                        lower_percentile, upper_percentile,
                        color=colors[index], alpha=0.1, linewidth=1, linestyle = '--',
                    )
        
                    # Plot the lower percentile line
                    ax.plot(
                        rural_anomaly['month'], lower_percentile,
                        color=colors[index], alpha=0.5, linewidth=1, linestyle='--',  label=f'{percentile} to {100-percentile} Percentile')
                    # Plot the upper percentile line
                    ax.plot(
                        rural_anomaly['month'], upper_percentile,
                        color=colors[index], alpha=0.5, linewidth=1, linestyle='--')
                if gridcell_series:
                    for i, j in product(anom[x_dim].values, anom[y_dim].values):
                        anom_val = anom.sel({x_dim:i,
                                             y_dim:j})
                        if not np.isnan(anom_val[0]):
                            anom_val.plot(ax=ax, color=colors[index], linewidth=0.1, alpha = 0.1)
                             
        #Plot the observation if requested
        if not self.obs_attr.empty and obs_anomaly is not None:
            codes_ins_city, codes_out_city = self._get_obs_codes_by_group()
            codes_ins_city = self._filter_codes_in_timeseries(codes_ins_city)
            codes_out_city = self._filter_codes_in_timeseries(codes_out_city)

            if len(codes_ins_city) > 0:
                obs_anomaly[codes_ins_city].plot(ax = ax, marker='o', color = '#A52A2A', 
                                                     linestyle='--', linewidth = 1, alpha=0.4)
            if len(codes_out_city) > 0:
                obs_anomaly[codes_out_city].plot(ax = ax, marker='o', color = 'g', 
                                                     linestyle='--', linewidth = 1, alpha=0.4)

            obs_anomaly['urban_mean'].plot(ax = ax, color='#A52A2A', linestyle='--', marker='o',
                                                         linewidth = 4, label='Urban mean (obs.)', 
                                                         zorder = 2000) 

            obs_anomaly['rural_mean'].plot(ax = ax, color='#8A8D28', linestyle='--', marker='o',
                                                         linewidth = 4, label='Vicinity mean (obs.)', 
                                                         zorder = 2000)
                
            
        ax.set_xticks(np.arange(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 
                            'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize = 18)
        ax.tick_params(axis='y', labelsize=18)
        if vmax is not None and vmin is not None:
            ax.set_ylim(vmin, vmax)

        # Add legend to the plot
        #ax.legend(fontsize = 14, loc='center left', bbox_to_anchor=(0, -0.2), prop={'size': 14})

        if self.rcm:
            ax.set_title(f"Urban Island for {city_name} (variable: {ds_anomaly.name}, rcm: {self.rcm})", fontsize=18)
        else:
            ax.set_title(f"Urban Island for {city_name} (variable: {ds_anomaly.name})", fontsize=18)    

        unit = ds_anomaly.attrs.get('units', 'unknown')         
        ax.set_ylabel(f"{unit}")
      
        return fig



    def plot_reduced_annual_cycle(
        self,
        *,
        ds_anomaly = None,
        obs_anomaly = None,
        percentiles = [5],
        gridcell_series = False, 
        city_name = None, 
        ax = None,            
        vmax = None, 
        vmin = None,
        city_c = '#A52A2A',
        rural_c= '#8A8D28',
        label = None,
        offset_sim = 0,
        offset_obs = 0
        ):
        '''
        Plots the annual cycle of the Urban Island effect using model and optional observation data on a given ax
        
        Parameters:
        - ds_anomaly (xarray.DataArray, optional): Monthly anomaly from the model. If not provided, it will be computed.
        - obs_anomaly (pandas.DataFrame, optional): Monthly anomaly from observations. If not provided, it will be computed.
        - percentiles (list of int): Percentiles to shade around the mean values (e.g., [5] for 5th–95th range).
        - gridcell_series (bool): Whether to plot individual grid cell time series as transparent lines.
        - city_name (str, optional): Name of the city to include in the plot title.
        - ax (matplotlib.axes.Axes, optional): Existing axis to plot on. If None, a new figure and axis are created.
        - vmax (float, optional): Upper limit for the y-axis.
        - vmin (float, optional): Lower limit for the y-axis.
        
        Outputs:
        - fig (matplotlib.figure.Figure): The resulting figure object.
        '''

        if ds_anomaly is None:
            if not hasattr(self, 'ds_annual_cycle'):
                self.compute_annual_cycle()
            ds_anomaly = self.ds_annual_cycle
        
        if not self.obs_attr.empty and obs_anomaly is None:
            if hasattr(self, 'obs_annual_cycle'):
                obs_anomaly = self.obs_annual_cycle
        
        y_dim, x_dim = self._get_spatial_dim_names(ds_anomaly)
        
        is_rural = self.urban_vicinity['urmask'] == 0
        is_urban = self.urban_vicinity['urmask'] == 1

        rural_anomaly = ds_anomaly.where(is_rural)
        urban_anomaly = ds_anomaly.where(is_urban)
    
        urban_mean = urban_anomaly.mean(dim = [y_dim, x_dim]).compute()
        rural_mean = rural_anomaly.mean(dim = [y_dim, x_dim]).compute()

        urban_area_legend = False
        not_urban_area_legend = False
        if ax is None:
            print('creating new figure and axis')
            fig, ax = plt.subplots(figsize=(15, 7))
        else: 
            print('using existing axis')

        (urban_mean+ offset_sim).plot(ax=ax,  color = city_c, linestyle='-', 
                                         linewidth = 4, label=f'{label} (sim)')
                             
        (rural_mean-rural_mean).plot(ax=ax,  color = rural_c, linestyle='-', 
                                     linewidth = 4, label=f'rural ref (sim)')
                             
        if percentiles:
            # Fill within percentiles
            axis = [rural_anomaly.get_axis_num(x_dim),
                rural_anomaly.get_axis_num(y_dim)]
            colors = [rural_c, city_c ]
            for index, anom in enumerate([ rural_anomaly, urban_anomaly]):
                for percentile in percentiles:
                    lower_percentile = np.nanpercentile(anom, percentile, axis=axis)
                    upper_percentile = np.nanpercentile(anom, 100-percentile, axis=axis)
                    ax.fill_between(
                        rural_anomaly['month'],
                        lower_percentile, upper_percentile,
                        color=colors[index], alpha=0.05, linewidth=1, linestyle = '--',
                    )
        
                    # Plot the lower percentile line
                    ax.plot(
                        rural_anomaly['month'], lower_percentile,
                        color=colors[index], alpha=0.5, linewidth=1, linestyle='--')
                    # Plot the upper percentile line
                    ax.plot(
                        rural_anomaly['month'], upper_percentile,
                        color=colors[index], alpha=0.5, linewidth=1, linestyle='--')
                if gridcell_series:
                    for i, j in product(anom[x_dim].values, anom[y_dim].values):
                        anom_val = anom.sel({x_dim:i,
                                             y_dim:j})
                        if not np.isnan(anom_val[0]):
                            anom_val.plot(ax=ax, color=colors[index], linewidth=0.1, alpha = 0.1)
                             
        #Plot the observation if requested
        if not self.obs_attr.empty and obs_anomaly is not None:
            codes_ins_city, codes_out_city = self._get_obs_codes_by_group()
            codes_ins_city = self._filter_codes_in_timeseries(codes_ins_city)
            codes_out_city = self._filter_codes_in_timeseries(codes_out_city)
            (obs_anomaly['urban_mean']+offset_obs).plot(ax = ax, color=city_c, linestyle='--', marker='o',
                                                         linewidth = 4, label=f'{label} urban (obs)', 
                                                         zorder = 2000) 

            obs_anomaly['rural_mean'].plot(ax = ax, color=rural_c, linestyle='--', marker='o',
                                                         linewidth = 4, label=f'rural ref (obs.)', 
                                                         zorder = 2000)
                
            
        ax.set_xticks(np.arange(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 
                            'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize = 18)
        ax.tick_params(axis='y', labelsize=18)
        if vmax is not None and vmin is not None:
            ax.set_ylim(vmin, vmax)

        # Add legend to the plot
        #ax.legend(fontsize = 14, loc='center left', bbox_to_anchor=(0, -0.2), prop={'size': 14})


        if self.rcm:
            ax.set_title(f"Urban Island for {city_name} (variable: {ds_anomaly.name}, rcm: {self.rcm})", fontsize=18)
        else:
            ax.set_title(f"Urban Island for {city_name} (variable: {ds_anomaly.name})", fontsize=18)    

        unit = ds_anomaly.attrs.get('units', 'unknown')         
        ax.set_ylabel(f"{unit}")
      
        return ax


    def plot_daily_cycle(
        self,
        *,
        ds_anomaly = None,
        obs_anomaly = None,
        percentiles = [5],                      
        gridcell_series = True, 
        city_name = None, 
        ax = None,            
        vmax = None, 
        vmin = None):
        '''
        Plots the daily cycle of a variable, with optional anomalies, urban/rural differentiation, and observational overlays.
        
        Parameters:
        - ds_anomaly (xarray.DataArray, optional): Daily anomaly from the model. If not provided, it will be computed.
        - obs_anomaly (pandas.DataFrame, optional): Daily anomaly from observations. If not provided, it will be computed.
        - percentiles (list of int): Percentiles to shade around the mean values (e.g., [5] for 5th–95th range).
        - gridcell_series (bool): Whether to plot individual grid cell time series as transparent lines.
        - city_name (str, optional): Name of the city to include in the plot title.
        - ax (matplotlib.axes.Axes, optional): Existing axis to plot on. If None, a new figure and axis are created.
        - vmax (float, optional): Upper limit for the y-axis.
        - vmin (float, optional): Lower limit for the y-axis.
        
        Outputs:
        - fig (matplotlib.figure.Figure): The resulting figure object.
        '''
        if ds_anomaly is None:
            if not hasattr(self, 'ds_daily_cycle'):
                self.compute_daily_cycle()
            ds_anomaly = self.ds_daily_cycle
        
        if not self.obs_attr.empty and obs_anomaly is None:
            if hasattr(self, 'obs_daily_cycle'):
                obs_anomaly = self.obs_daily_cycle
        
        y_dim, x_dim = self._get_spatial_dim_names(ds_anomaly)
        
        is_rural = self.urban_vicinity['urmask'] == 0
        is_urban = self.urban_vicinity['urmask'] == 1

        rural_anomaly = ds_anomaly.where(is_rural)
        urban_anomaly = ds_anomaly.where(is_urban)
    
        urban_mean = urban_anomaly.mean(dim = [y_dim, x_dim]).compute()
        rural_mean = rural_anomaly.mean(dim = [y_dim, x_dim]).compute()

        urban_area_legend = False
        not_urban_area_legend = False
    
        fig, ax = plt.subplots(figsize=(15, 7))

        (urban_mean).plot(ax=ax,  color = '#A52A2A', linestyle='-', 
                                         linewidth = 4, label='Urban mean')
                             
        (rural_mean-rural_mean).plot(ax=ax,  color = '#8A8D28', linestyle='-', 
                                     linewidth = 4, label='Vicinity mean')

        
        # Plot individual data squares for urban and rural areas if requested
        if percentiles:
            # Fill within percentiles
            axis = [rural_anomaly.get_axis_num(x_dim),
                rural_anomaly.get_axis_num(y_dim)]
            colors = ['#8A8D28', '#A52A2A']
            for index, anom in enumerate([ rural_anomaly, urban_anomaly]):
                for percentile in percentiles:
                    lower_percentile = np.nanpercentile(anom, percentile, axis=axis)
                    upper_percentile = np.nanpercentile(anom, 100-percentile, axis=axis)
                    ax.fill_between(
                        rural_anomaly['hour'],
                        lower_percentile, upper_percentile,
                        color=colors[index], alpha=0.1, linewidth=1, linestyle = '--',
                    )
        
                    # Plot the lower percentile line
                    ax.plot(
                        rural_anomaly['hour'], lower_percentile,
                        color=colors[index], alpha=0.5, linewidth=1, linestyle='--', label=f'{percentile} to {100-percentile} Percentile')
                    # Plot the upper percentile line
                    ax.plot(
                        rural_anomaly['hour'], upper_percentile,
                        color=colors[index], alpha=0.5, linewidth=1, linestyle='--')
                for i, j in product(anom[x_dim].values, anom[y_dim].values):
                    anom_val = anom.sel({x_dim:i,
                                         y_dim:j})
                    if not np.isnan(anom_val[0]):
                        anom_val.plot(ax=ax, color=colors[index], linewidth=0.1, alpha = 0.1)

        # Plot observations if requested
        if not self.obs_attr.empty and obs_anomaly is not None:
            codes_ins_city, codes_out_city = self._get_obs_codes_by_group()
            codes_ins_city = self._filter_codes_in_timeseries(codes_ins_city)
            codes_out_city = self._filter_codes_in_timeseries(codes_out_city)

            if len(codes_ins_city) > 0:
                obs_anomaly[codes_ins_city].plot(ax = ax, marker='o', color = 'k', 
                                                     linestyle='--', linewidth = 2)
            if len(codes_out_city) > 0:
                obs_anomaly[codes_out_city].plot(ax = ax, marker='o', color = 'g', 
                                                     linestyle='--', linewidth = 2)
            obs_anomaly['urban_mean'].plot(ax = ax, color= 'k', linestyle='--', marker='o',
                                                         linewidth = 4, label='Urban mean (obs.)', 
                                                         zorder = 2000) 

            obs_anomaly['rural_mean'].plot(ax = ax, color='g', linestyle='--', marker='o',
                                                         linewidth = 4, label='Vicinity mean (obs.)', 
                                                         zorder = 2000)

        
        ax.set_xticks(np.arange(0, 24))
        ax.set_xticklabels([f'{h}' for h in range(24)], fontsize=12)
        ax.tick_params(axis='y', labelsize=18)
        if vmax is not None and vmin is not None:
            ax.set_ylim(vmin, vmax)

        # Add legend to the plot
        ax.legend(fontsize = 14, loc='center left', bbox_to_anchor=(0, -0.2), prop={'size': 14})

        ax.set_title(f"Urban Island for {city_name} (variable: {ds_anomaly.name})", fontsize=18)

        unit = ds_anomaly.attrs.get('units', 'unknown')         
        ax.set_ylabel(f"{unit}")
      
        return fig

    def _season_months(self, season):
        """Return month numbers for a requested season."""
        if season is None:
            return None

        season_l = str(season).strip().lower()
        if season_l in ("annual", "all", "year", "yearly"):
            return list(range(1, 13))
        if season_l == "djf":
            return [12, 1, 2]
        if season_l == "mam":
            return [3, 4, 5]
        if season_l == "jja":
            return [6, 7, 8]
        if season_l == "son":
            return [9, 10, 11]

        raise ValueError(f"Unsupported season '{season}'. Use one of: DJF, MAM, JJA, SON, Annual.")

    def _compute_uhi_timeseries_obs_sim(self, season=None, min_valid_stations=1):
        """Compute aligned model/observation UHIN timeseries for one season.

        UHIN is computed as urban mean minus rural/vicinity mean at each timestamp.
        """
        if self.ds is None:
            raise ValueError("Simulation data (self.ds) is required.")

        if self.obs_time is None or getattr(self.obs_time, 'empty', True):
            raise ValueError("Observation timeseries (self.obs_time) is empty.")

        # Ensure DataArray input for model diagnostics.
        if isinstance(self.ds, xr.Dataset):
            if len(self.ds.data_vars) == 0:
                raise ValueError("Simulation dataset has no data variables.")
            ds_var = self.ds[list(self.ds.data_vars)[0]]
        else:
            ds_var = self.ds

        # Model UHIN from spatial means of urban and rural masks.
        ds_aligned = self._align_ds_with_mask(ds_var)
        y_dim, x_dim = self._get_spatial_dim_names(ds_aligned)
        is_urban = self.urban_vicinity['urmask'] == 1
        is_rural = self.urban_vicinity['urmask'] == 0

        sim_urban = ds_aligned.where(is_urban).mean(dim=[y_dim, x_dim])
        sim_rural = ds_aligned.where(is_rural).mean(dim=[y_dim, x_dim])
        sim_uhi = (sim_urban - sim_rural).compute()

        sim_index = pd.to_datetime(sim_uhi['time'].values, errors='coerce')
        sim_series = pd.Series(np.asarray(sim_uhi.values).ravel(), index=sim_index, name='uhi_sim')
        sim_series = sim_series[sim_series.index.notna()]
        sim_series = sim_series.groupby(sim_series.index).mean().sort_index()

        # Observation UHIN from station groups.
        codes_urban, codes_rural = self._get_obs_codes_by_group()
        codes_urban = self._filter_codes_in_timeseries(codes_urban)
        codes_rural = self._filter_codes_in_timeseries(codes_rural)

        if len(codes_urban) == 0 or len(codes_rural) == 0:
            raise ValueError(
                f"Need both urban and rural stations in obs_timeseries. Found urban={len(codes_urban)}, rural={len(codes_rural)}."
            )

        obs_df = self.obs_time.copy()
        if not isinstance(obs_df, pd.DataFrame):
            obs_df = pd.DataFrame(obs_df)
        obs_df.index = pd.to_datetime(obs_df.index, errors='coerce')
        obs_df = obs_df[obs_df.index.notna()].sort_index()

        urban_mean = obs_df[codes_urban].mean(axis=1, skipna=True)
        rural_mean = obs_df[codes_rural].mean(axis=1, skipna=True)
        obs_uhi = (urban_mean - rural_mean).rename('uhi_obs')

        if int(min_valid_stations) > 1:
            urban_valid = obs_df[codes_urban].notna().sum(axis=1)
            rural_valid = obs_df[codes_rural].notna().sum(axis=1)
            valid_rows = (urban_valid >= int(min_valid_stations)) & (rural_valid >= int(min_valid_stations))
            obs_uhi = obs_uhi.where(valid_rows)

        obs_uhi = obs_uhi.groupby(obs_uhi.index).mean().sort_index()

        # Keep only matched timestamps for QQ comparison.
        joined = pd.concat([obs_uhi, sim_series], axis=1, join='inner').dropna(how='any')

        months = self._season_months(season)
        if months is not None:
            joined = joined[joined.index.month.isin(months)]

        return joined

    def plot_uhi_quantile_comparison(
        self,
        *,
        seasons=('djf', 'jja'),
        n_quantiles=101,
        city_name=None,
        model_label='Model',
        marker_size=28,
        include_fit=True,
        obs_tolerance_k=1.0,
        min_valid_stations=1,
        figsize=(13, 5),
    ):
        """Create seasonal QQ plots for observation vs model UHIN.

        This reproduces the spirit of the paper's seasonal QQ diagnostics (e.g., DJF/JJA):
        quantiles of model UHIN are compared against quantiles of observed UHIN.

        Parameters
        ----------
        seasons : tuple/list[str]
            Seasons to plot, e.g. ('djf', 'jja') or ('djf',).
        n_quantiles : int
            Number of quantile levels between 0 and 1.
        city_name : str, optional
            City name for titles.
        model_label : str
            Label of the model in legends.
        marker_size : int/float
            Scatter marker size.
        include_fit : bool
            If True, overlays linear fit of QQ points.
        obs_tolerance_k : float
            Half-width (in K) of the gray envelope around the 1:1 line
            representing model-observation differences within +/- obs_tolerance_k.
        min_valid_stations : int
            Minimum non-NaN stations per group to keep an observation timestamp.
        figsize : tuple
            Figure size.

        Returns
        -------
        fig : matplotlib.figure.Figure
            QQ figure.
        diagnostics : dict
            Season-wise diagnostics (sample size, correlation, slope/intercept).
        """
        if isinstance(seasons, str):
            seasons = (seasons,)

        if int(n_quantiles) < 3:
            raise ValueError("n_quantiles must be >= 3.")

        quantiles = np.linspace(0.0, 1.0, int(n_quantiles))
        n_panels = len(seasons)
        fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
        axes = axes.ravel()

        diagnostics = {}

        for i, season in enumerate(seasons):
            ax = axes[i]
            aligned = self._compute_uhi_timeseries_obs_sim(
                season=season,
                min_valid_stations=min_valid_stations,
            )

            if aligned.empty:
                ax.text(0.5, 0.5, f"No aligned data for {str(season).upper()}", ha='center', va='center')
                ax.set_axis_off()
                diagnostics[str(season).lower()] = {'n': 0}
                continue

            obs_vals = aligned['uhi_obs'].to_numpy()
            sim_vals = aligned['uhi_sim'].to_numpy()
            q_obs = np.nanquantile(obs_vals, quantiles)
            q_sim = np.nanquantile(sim_vals, quantiles)

            min_xy = np.nanmin(np.r_[q_obs, q_sim])
            max_xy = np.nanmax(np.r_[q_obs, q_sim])
            pad = 0.05 * (max_xy - min_xy + 1e-12)
            xline = np.array([min_xy - pad, max_xy + pad])

            # Gray envelope: +/- 1 K around the 1:1 line (paper-style tolerance band).
            if obs_tolerance_k is not None and float(obs_tolerance_k) > 0:
                tol = float(obs_tolerance_k)
                ax.fill_between(
                    xline,
                    xline - tol,
                    xline + tol,
                    color='gray',
                    alpha=0.2,
                    zorder=0,
                    label=f'+/-{tol:g} K from observations',
                )

            ax.scatter(q_obs, q_sim, s=marker_size, alpha=0.8, color='tab:blue', label=f"{model_label} quantiles")
            ax.plot(xline, xline, 'k-', lw=1.4, label='1:1')

            slope = np.nan
            intercept = np.nan
            if include_fit and len(q_obs) > 1:
                slope, intercept = np.polyfit(q_obs, q_sim, 1)
                ax.plot(
                    xline,
                    [slope * xline[0] + intercept, slope * xline[1] + intercept],
                    linestyle='--', color='gray', lw=1.2,
                    label=f"fit: y={slope:.2f}x+{intercept:.2f}",
                )

            corr = np.corrcoef(obs_vals, sim_vals)[0, 1] if len(obs_vals) > 1 else np.nan
            season_key = str(season).lower()
            diagnostics[season_key] = {
                'n': int(len(aligned)),
                'corr': float(corr) if np.isfinite(corr) else np.nan,
                'slope': float(slope) if np.isfinite(slope) else np.nan,
                'intercept': float(intercept) if np.isfinite(intercept) else np.nan,
                'obs_min': float(np.nanmin(obs_vals)),
                'obs_max': float(np.nanmax(obs_vals)),
                'sim_min': float(np.nanmin(sim_vals)),
                'sim_max': float(np.nanmax(sim_vals)),
            }

            season_label = str(season).upper()
            title_bits = [f"{season_label}"]
            if city_name:
                title_bits.insert(0, city_name)
            title_bits.append(f"n={len(aligned)}")
            ax.set_title(" - ".join(title_bits))
            ax.set_xlabel("UHIN observation")
            ax.set_ylabel("UHIN model")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_xlim(min_xy - pad, max_xy + pad)
            ax.set_ylim(min_xy - pad, max_xy + pad)

        fig.tight_layout()
        return fig, diagnostics

    def plot_diurnal_cycles_and_uhi_from_precomputed(
        self,
        diurnal_cycle_obs,
        diurnal_cycle_sim,
        obs_attributes,
        urmask,
        *,
        city_name="City",
        sim_label="SIM",
        period="YEAR",
        save_fig=False,
        save_dir=None,
    ):
        """
        Plot 2x2 panels for multi-station diurnal evaluation using pre-computed diurnal cycles:
        - (0,0) Diurnal cycle means with filled spread per (source, land use)
        - (0,1) Bias per station + bold mean bias per land use
        - (1,0) UHI from mean(urban) - mean(rural) for obs and sim
        - (1,1) UHI bias = UHI_sim - UHI_obs

        Parameters:
        ----------
        diurnal_cycle_obs : pd.DataFrame
            Hourly means of observations (rows=hours 0-23, columns=station codes, values=temperatures).
        diurnal_cycle_sim : xr.DataArray or xr.Dataset
            Simulation data grouped by hour (can be spatial or averaged).
        obs_attributes : pd.DataFrame
            Station metadata with columns: 'code', 'inside_city' (or 'station_class').
        urmask : xr.Dataset
            Dataset with 'urmask' variable (1=urban, 0=rural).
        city_name : str, default="City"
            Name of the city for the plot title and file naming.
        sim_label : str, default="SIM"
            Label for the simulation (used in legend and file names).
        period : str, default="YEAR"
            Period description for title (e.g., "YEAR", "JJA", "DJF").
        save_fig : bool, default=False
            If True, save the figure to disk.
        save_dir : str, optional
            Directory where to save figures. If None and save_fig=True, uses current directory.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The resulting 2x2 plot figure.
        """
        import matplotlib.pyplot as plt
        
        # Extract urban/rural station codes from obs_attributes
        codes_urban = [str(c) for c in obs_attributes.loc[obs_attributes["inside_city"] == True, "code"].values
                       if str(c) in diurnal_cycle_obs.columns]
        codes_rural = [str(c) for c in obs_attributes.loc[obs_attributes["inside_city"] == False, "code"].values
                       if str(c) in diurnal_cycle_obs.columns]

        if len(codes_urban) == 0 or len(codes_rural) == 0:
            raise ValueError(
                f"Must have both urban and rural stations. Found urban={len(codes_urban)}, rural={len(codes_rural)}"
            )

        # Extract urban/rural diurnal cycles from observations
        obs_urban_cycles = diurnal_cycle_obs[codes_urban]
        obs_rural_cycles = diurnal_cycle_obs[codes_rural]
        
        obs_urban_mean = obs_urban_cycles.mean(axis=1)
        obs_urban_min = obs_urban_cycles.min(axis=1)
        obs_urban_max = obs_urban_cycles.max(axis=1)
        
        obs_rural_mean = obs_rural_cycles.mean(axis=1)
        obs_rural_min = obs_rural_cycles.min(axis=1)
        obs_rural_max = obs_rural_cycles.max(axis=1)

        # Extract simulation diurnal cycles (spatial mean for urban/rural)
        is_urban = urmask['urmask'] == 1
        is_rural = urmask['urmask'] == 0
        
        # If diurnal_cycle_sim is a Dataset, extract the first data variable
        if isinstance(diurnal_cycle_sim, xr.Dataset):
            sim_var = diurnal_cycle_sim[list(diurnal_cycle_sim.data_vars)[0]]
        else:
            sim_var = diurnal_cycle_sim
        
        y_dim, x_dim = self._get_spatial_dim_names(sim_var)

        # Compute simulation urban/rural means for each hour
        sim_hourly_urban = sim_var.where(is_urban).mean(dim=[y_dim, x_dim])
        sim_hourly_rural = sim_var.where(is_rural).mean(dim=[y_dim, x_dim])
        
        # Compute simulation spreads (spatial min/max)
        sim_urban_min = sim_var.where(is_urban).min(dim=[y_dim, x_dim])
        sim_urban_max = sim_var.where(is_urban).max(dim=[y_dim, x_dim])
        sim_rural_min = sim_var.where(is_rural).min(dim=[y_dim, x_dim])
        sim_rural_max = sim_var.where(is_rural).max(dim=[y_dim, x_dim])

        # Extract numpy arrays from xarray objects for easier manipulation
        sim_hourly_urban_vals = np.asarray(sim_hourly_urban.values).flatten()
        sim_hourly_rural_vals = np.asarray(sim_hourly_rural.values).flatten()
        sim_urban_min_vals = np.asarray(sim_urban_min.values).flatten()
        sim_urban_max_vals = np.asarray(sim_urban_max.values).flatten()
        sim_rural_min_vals = np.asarray(sim_rural_min.values).flatten()
        sim_rural_max_vals = np.asarray(sim_rural_max.values).flatten()
        
        # Compute biases: for each station, compare with appropriate simulation (urban or rural)
        bias_dict = {}
        for code in codes_urban:
            if code in diurnal_cycle_obs.columns:
                bias_dict[code] = diurnal_cycle_obs[code].values - sim_hourly_urban_vals
        for code in codes_rural:
            if code in diurnal_cycle_obs.columns:
                bias_dict[code] = diurnal_cycle_obs[code].values - sim_hourly_rural_vals
        
        # Compute mean bias per hour for urban and rural stations
        hours = np.arange(24)
        bias_urban_hourly_list = [bias_dict[code] for code in codes_urban if code in bias_dict]
        bias_rural_hourly_list = [bias_dict[code] for code in codes_rural if code in bias_dict]
        
        bias_urban_mean_hourly = np.mean(bias_urban_hourly_list, axis=0) if bias_urban_hourly_list else np.zeros(24)
        bias_rural_mean_hourly = np.mean(bias_rural_hourly_list, axis=0) if bias_rural_hourly_list else np.zeros(24)

        # Compute UHI
        uhi_obs = obs_urban_mean.values - obs_rural_mean.values
        uhi_sim = sim_hourly_urban_vals - sim_hourly_rural_vals
        uhi_bias = uhi_sim - uhi_obs

        # Color palettes
        urban_colors = plt.cm.Reds(np.linspace(0.45, 0.9, len(codes_urban)))
        rural_colors = plt.cm.Greens(np.linspace(0.45, 0.9, len(codes_rural)))

        fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True)

        sim_ls = "-"
        obs_ls = "--"
        lw_single = 1.8
        lw_mean = 3.0
        alpha_single = 0.7

        # (0,0) Diurnal cycles: filled spread around means
        ax = axes[0, 0]
        
        ax.fill_between(hours, obs_urban_min.values, obs_urban_max.values, 
                        color="lightcoral", alpha=0.5, label="obs urban spread")
        ax.fill_between(hours, sim_urban_min_vals, sim_urban_max_vals, 
                        color="indianred", alpha=0.5, label=f"{sim_label} urban spread")
        ax.fill_between(hours, obs_rural_min.values, obs_rural_max.values, 
                        color="lightgreen", alpha=0.5, label="obs rural spread")
        ax.fill_between(hours, sim_rural_min_vals, sim_rural_max_vals, 
                        color="mediumseagreen", alpha=0.5, label=f"{sim_label} rural spread")

        ax.plot(hours, obs_urban_mean.values, color="darkred", linestyle=obs_ls, 
                linewidth=lw_mean, label="obs urban mean")
        ax.plot(hours, sim_hourly_urban_vals, color="darkred", linestyle=sim_ls, 
                linewidth=lw_mean, label=f"{sim_label} urban mean")
        ax.plot(hours, obs_rural_mean.values, color="darkgreen", linestyle=obs_ls, 
                linewidth=lw_mean, label="obs rural mean")
        ax.plot(hours, sim_hourly_rural_vals, color="darkgreen", linestyle=sim_ls, 
                linewidth=lw_mean, label=f"{sim_label} rural mean")

        ax.set_title("Diurnal cycle - grouped means and spread")
        ax.set_xlabel("Hour")
        ax.set_ylabel(f"Temperature")
        ax.grid(True)
        ax.legend(fontsize=8, ncol=2)

        # (0,1) Bias per station + mean bias per land use
        ax = axes[0, 1]
        
        for color, code in zip(urban_colors, codes_urban):
            if code in bias_dict:
                ax.plot(hours, bias_dict[code], color=color, linestyle="-", 
                       alpha=alpha_single, linewidth=lw_single, label=f"bias {code}")
        
        for color, code in zip(rural_colors, codes_rural):
            if code in bias_dict:
                ax.plot(hours, bias_dict[code], color=color, linestyle="-", 
                       alpha=alpha_single, linewidth=lw_single, label=f"bias {code}")

        ax.plot(hours, bias_urban_mean_hourly, color="darkred", linestyle="-", 
                linewidth=lw_mean, label="urban bias mean")
        ax.plot(hours, bias_rural_mean_hourly, color="darkgreen", linestyle="-", 
                linewidth=lw_mean, label="rural bias mean")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title("Station bias (sim - obs)")
        ax.set_xlabel("Hour")
        ax.set_ylabel(f"Temperature")
        ax.grid(True)
        ax.legend(fontsize=8, ncol=2)

        # (1,0) UHI from means
        ax = axes[1, 0]
        ax.plot(hours, uhi_obs, label="obs UHI (mean urban - mean rural)", 
               color="black", linestyle=obs_ls, linewidth=lw_mean)
        ax.plot(hours, uhi_sim, label=f"{sim_label} UHI (mean urban - mean rural)", 
               color="tab:blue", linestyle=sim_ls, linewidth=lw_mean)
        ax.set_title("UHI from grouped means")
        ax.set_xlabel("Hour")
        ax.set_ylabel(f"Temperature")
        ax.grid(True)
        ax.legend(fontsize=8)

        # (1,1) UHI bias
        ax = axes[1, 1]
        ax.plot(hours, uhi_bias, label="UHI bias (sim - obs)", 
               color="tab:blue", linewidth=lw_mean)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title("UHI bias")
        ax.set_xlabel("Hour")
        ax.set_ylabel(f"Temperature")
        ax.grid(True)
        ax.legend(fontsize=8)

        plt.suptitle(f"{city_name} Diurnal Cycles and UHI\n{sim_label} vs Observations\n({period})", 
                    fontsize=15)
        plt.tight_layout()

        if save_fig:
            if save_dir is None:
                save_dir = "./figures"
            os.makedirs(save_dir, exist_ok=True)
            fname = f"diurnal_uhi_{city_name}_{sim_label}_{period}.png"
            plt.savefig(f"{save_dir}/{fname}", dpi=300)
            print(f"Saved figure to {save_dir}/{fname}")

        return fig
