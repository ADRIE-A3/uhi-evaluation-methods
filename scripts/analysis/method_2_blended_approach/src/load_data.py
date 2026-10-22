"""Utilities to load simulation and observation datasets for ble workflows."""

from pathlib import Path
import glob

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rasterio


def _years_between(start, stop):
	"""Return an inclusive year range from start to stop."""
	return range(int(str(start)[:4]), int(str(stop)[:4]) + 1)


def _daily_tas_file_variable(variable):
	"""Map supported tas-like variable names to EUROCITIES daily filename tags."""
	name = str(variable).strip().lower()
	mapping = {
		"tmin": "tmin",
		"tasmin": "tmin",
		"min": "tmin",
		"tmean": "tmean",
		"tas": "tmean",
		"tasmean": "tmean",
		"mean": "tmean",
		"tmax": "tmax",
		"tasmax": "tmax",
		"max": "tmax",
	}
	if name not in mapping:
		raise ValueError(
			"For daily EUROCITIES tas files, variable must map to one of: "
			"tmin/tmean/tmax (aliases: tasmin/tas/tasmean/tasmax)."
		)
	return mapping[name]


def _variable_to_daily_stat(variable):
	"""Map tas-like variable names to a pandas daily aggregation stat."""
	name = str(variable).strip().lower()
	mapping = {
		"tmin": "min",
		"tasmin": "min",
		"min": "min",
		"tmean": "mean",
		"tas": "mean",
		"tasmean": "mean",
		"mean": "mean",
		"tmax": "max",
		"tasmax": "max",
		"max": "max",
	}
	if name not in mapping:
		raise ValueError(
			"Cannot infer daily stat from variable. Use one of: "
			"tmin/tmean/tmax (or tasmin/tas/tasmean/tasmax)."
		)
	return mapping[name]


def _simulation_patterns(domain, variable, freq, start, stop):
	"""Build file search patterns for supported domains/frequencies."""
	domain = str(domain).upper()
	freq = str(freq).lower()

	if freq == "monthly":
		patterns = {
			"EUROCORDEX": (
				"/dodrio/scratch/projects/2022_200/project_output/RMIB-UGent/"
				"vsc45381_leon/GIS/analysis/cdo_aggregate_data/monthly_data/"
				f"EUROCORDEX/EUROCORDEX_kobe_cy43_{variable}_monmean_*.nc"
			),
			"EUROCITIES": (
				"/dodrio/scratch/projects/2022_200/project_output/RMIB-UGent/"
				"vsc45381_leon/GIS/analysis/cdo_aggregate_data/monthly_data/"
				f"EUROCITIES/EUROCITIES_operational_cy43_{variable}_monmean_*.nc"
			),
		}
		if domain not in patterns:
			raise ValueError(f"Unsupported domain '{domain}' for monthly data.")
		return [patterns[domain]]

	if freq == "hourly":
		years = _years_between(start, stop)
		if domain == "EUROCORDEX":
			return [
				(
					"/dodrio/scratch/projects/2022_200/project_output/RMIB-UGent/"
					"CORDEXbeII/run_ALARO_sfx/out/netCDF/PROD_ERA5_EUR11/19780101/"
					"CORDEX-CMIP6/DD/EUR-12/RMIB-UGent/ERA5/evaluation/r1i1p1f1/"
					"ALARO1-SFX/v1-r1/1hr/tas/v20241009/"
					"tas_EUR-12_ERA5_evaluation_r1i1p1f1_RMIB-UGent_ALARO1-SFX_"
					f"v1-r1_1hr_{year}010100-{year}123123.nc"
				)
				for year in years
			]

		if domain == "EUROCITIES":
			return [
				(
					"/dodrio/scratch/projects/2022_200/project_output/RMIB-UGent/"
					"vsc45381_leon/GIS/CF_output/conversion_output/output/tas/"
					f"EUROCITIES_operational_cy43_tas_{year}_*.nc"
				)
				for year in years
			]

		raise ValueError(f"Unsupported domain '{domain}' for hourly data.")

	if freq == "daily":
		years = _years_between(start, stop)
		if domain != "EUROCITIES":
			raise ValueError(f"Unsupported domain '{domain}' for daily data.")

		file_var = _daily_tas_file_variable(variable)
		return [
			(
				"/dodrio/scratch/projects/2022_200/project_output/RMIB-UGent/"
				"vsc45381_leon/GIS/analysis/cdo_aggregate_data/daily_data/"
				f"EUROCITIES/EUROCITIES_operational_cy43_{file_var}_{year}.nc"
			)
			for year in years
		]

	raise ValueError("freq must be one of: 'monthly', 'daily', 'hourly'.")


def load_simulation_data(domain, variable, start, stop, freq="monthly"):
	"""Load simulation data for a domain/frequency and return requested period."""
	patterns = _simulation_patterns(
		domain=domain,
		variable=variable,
		freq=freq,
		start=start,
		stop=stop,
	)

	files = []
	for pattern in patterns:
		files.extend(glob.glob(pattern))
	files = sorted(files)

	if not files:
		raise FileNotFoundError(
			f"No simulation files found for domain={domain}, variable={variable}, freq={freq}."
		)

	ds = xr.open_mfdataset(files, combine="nested", concat_dim="time")
	if "tas" in ds.data_vars and variable != "tas":
		print('Renaming "tas" variable to requested variable name for consistency. take care that this only renames the variable dimension, not the data itself.')
		ds = ds.rename({"tas": variable})

	if variable not in ds.data_vars:
		raise KeyError(f"Variable '{variable}' not found in loaded simulation dataset.")

	return ds.sel(time=slice(start, stop))


def load_eurocities_daily_tas_stats(start, stop, stats=("tmin", "tmean", "tmax")):
	"""Load EUROCITIES daily tas-based files for one or more of tmin/tmean/tmax.

	All source files store the data variable as ``tas``; each loaded dataset is renamed
	to the requested stat name before merging.
	"""
	stats = tuple(stats)
	if not stats:
		raise ValueError("stats must contain at least one entry.")

	datasets = []
	for stat in stats:
		datasets.append(
			load_simulation_data(
				domain="EUROCITIES",
				variable=stat,
				start=start,
				stop=stop,
				freq="daily",
			)
		)

	return xr.merge(datasets, compat="override")


def prepare_simulation_for_uhi(
	ds_rcm_period,
	*,
	urban,
	domain_resolution,
	urmask,
	variable=None,
	convert_kelvin=False,
	convert_time_to_month_start=False,
	debug=True,
):
	"""Prepare simulation data for UHI analysis on the urban mask grid.

	Parameters
	----------
	ds_rcm_period : xr.Dataset
		Simulation dataset already sliced to the target time period.
	urban : UrbanVicinity
		Configured UrbanVicinity object used for city cropping.
	domain_resolution : float
		Horizontal resolution used by ``urban.crop_area_city``.
	urmask : xr.Dataset
		Dataset containing ``urmask`` with target x/y/lon/lat grid.
	variable : str | None
		Variable name for optional Kelvin to Celsius conversion.
	convert_kelvin : bool
		Apply ``kelvin2degC`` before further processing.
	convert_time_to_month_start : bool
		Convert time index to month start timestamps.
	debug : bool
		Print overlap and finite-coordinate diagnostics.

	Returns
	-------
	tuple[xr.Dataset, xr.Dataset]
		Prepared simulation dataset and aligned urban mask.
	"""
	if convert_kelvin:
		if variable is None:
			raise ValueError("variable must be provided when convert_kelvin=True")
		ds_rcm_period = kelvin2degC(ds_rcm_period, variable)

	ds_rcm_period = fix_360_longitudes(ds_rcm_period)
	ds_rcm_urb = urban.crop_area_city(ds=ds_rcm_period, res=domain_resolution)

	mask_da = urmask["urmask"]
	x_model = ds_rcm_urb["x"].values
	y_model = ds_rcm_urb["y"].values
	x_mask = mask_da["x"].values
	y_mask = mask_da["y"].values

	common_x = np.intersect1d(x_model, x_mask).size
	common_y = np.intersect1d(y_model, y_mask).size

	if debug:
		print(f"x overlap: {common_x}/{x_model.size} model points ({100 * common_x / x_model.size:.1f}%)")
		print(f"y overlap: {common_y}/{y_model.size} model points ({100 * common_y / y_model.size:.1f}%)")

	if x_model.size == x_mask.size and debug:
		print(f"max |x_model - x_mask|: {np.max(np.abs(x_model - x_mask)):.6f}")
	if y_model.size == y_mask.size and debug:
		print(f"max |y_model - y_mask|: {np.max(np.abs(y_model - y_mask)):.6f}")

	x_target = mask_da["x"].where(
		(mask_da["x"] >= x_model.min()) & (mask_da["x"] <= x_model.max()),
		drop=True,
	)
	y_target = mask_da["y"].where(
		(mask_da["y"] >= y_model.min()) & (mask_da["y"] <= y_model.max()),
		drop=True,
	)

	ds_rcm_urb = ds_rcm_urb.interp(x=x_target, y=y_target, method="nearest")
	mask_da_aligned = mask_da.sel(x=x_target, y=y_target)

	ds_rcm_urb = ds_rcm_urb.assign_coords(lon=mask_da_aligned["lon"], lat=mask_da_aligned["lat"])
	urmask_aligned = urmask.sel(x=x_target, y=y_target)

	if debug:
		lon_finite_pct = 100 * np.isfinite(ds_rcm_urb["lon"].values).sum() / ds_rcm_urb["lon"].values.size
		lat_finite_pct = 100 * np.isfinite(ds_rcm_urb["lat"].values).sum() / ds_rcm_urb["lat"].values.size
		print(f"lon finite: {lon_finite_pct:.1f}%")
		print(f"lat finite: {lat_finite_pct:.1f}%")
		print("Post-alignment exact x match:", ds_rcm_urb["x"].equals(urmask_aligned["x"]))
		print("Post-alignment exact y match:", ds_rcm_urb["y"].equals(urmask_aligned["y"]))

	if convert_time_to_month_start:
		time_new = ds_rcm_urb.indexes["time"].to_period("M").to_timestamp()
		ds_rcm_urb = ds_rcm_urb.assign_coords(time=time_new)

	return ds_rcm_urb, urmask_aligned


def _load_obs_ghent(obs_root, city, start, stop):
	"""Load Ghent station CSV files and convert to an xarray.Dataset."""
	city_dir = Path(obs_root) / city
	stations = ["melle", "sintbavo", "honda", "provinciehuis", "plantentuin", "wondelgem"]

	dfs = []
	for station in stations:
		file_path = city_dir / f"{station}.csv"
		if not file_path.exists():
			continue

		df = pd.read_csv(file_path)
		df["DateTime"] = pd.to_datetime(df["DateTime"])
		df = df.set_index("DateTime")
		df = df[["Temperature"]]
		df.rename(columns={"Temperature": station}, inplace=True)
		dfs.append(df)

	if not dfs:
		raise FileNotFoundError(f"No Ghent station CSV files found in {city_dir}.")

	obs_df = pd.concat(dfs, axis=1)
	obs_df.index.name = "time"
	obs_ds = xr.Dataset.from_dataframe(obs_df)
	return obs_ds.sel(time=slice(start, stop))


def _load_obs_amsterdam(obs_root, start, stop):
	"""Load Amsterdam observational CSV and convert to xarray.Dataset."""
	file_path = Path(obs_root) / "Amsterdam" / "Amsterdam_cleaned_with_WMO.csv"
	if not file_path.exists():
		raise FileNotFoundError(f"Amsterdam file not found: {file_path}")

	df = pd.read_csv(file_path)
	df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
	df = df[df["obstype"] == "temp"]

	df_pivot = df.pivot_table(index="datetime", columns="name", values="value", aggfunc="first")
	obs_ds = xr.Dataset.from_dataframe(df_pivot).rename({"datetime": "time"})
	return obs_ds.sel(time=slice(start, stop))


def _load_obs_rennes(obs_root, start, stop):
	"""Load Rennes observational CSV and convert to xarray.Dataset."""
	file_path = Path(obs_root) / "Rennes" / "Rennes_cleaned.csv"
	if not file_path.exists():
		raise FileNotFoundError(f"Rennes file not found: {file_path}")

	df = pd.read_csv(file_path)
	df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
	df = df[df["obstype"] == "temp"]

	df_pivot = df.pivot_table(index="datetime", columns="name", values="value", aggfunc="first")
	obs_ds = xr.Dataset.from_dataframe(df_pivot).rename({"datetime": "time"})
	return obs_ds.sel(time=slice(start, stop))


def _load_obs_birmingham(obs_root, start, stop):
	"""Load Birmingham observational CSV and convert to xarray.Dataset."""
	file_path = Path(obs_root) / "Birmingham" / "Birmingham_cleaned.csv"
	if not file_path.exists():
		raise FileNotFoundError(f"Birmingham file not found: {file_path}")

	df = pd.read_csv(file_path)
	df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
	df = df[df["obstype"] == "temp"]

	df_pivot = df.pivot_table(index="datetime", columns="name", values="value", aggfunc="first")
	obs_ds = xr.Dataset.from_dataframe(df_pivot).rename({"datetime": "time"})
	return obs_ds.sel(time=slice(start, stop))


def _load_obs_berlin(obs_root, start, stop):
	"""Load Berlin observational CSV and convert to xarray.Dataset."""
	file_path = Path(obs_root) / "Berlin" / "Berlin_cleaned.csv"
	if not file_path.exists():
		raise FileNotFoundError(f"Berlin file not found: {file_path}")

	df = pd.read_csv(file_path)
	if "datetime" not in df.columns:
		raise KeyError(f"Expected a 'datetime' column in {file_path}")

	df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
	df = df.set_index("datetime")
	df.index.name = "time"
	df = df.apply(pd.to_numeric, errors="coerce")

	obs_attributes = _load_obs_metadata(obs_root=obs_root, city="Berlin")
	station_codes = [code for code in obs_attributes["code"].tolist() if code in df.columns]
	df = df.loc[:, station_codes]

	obs_ds = xr.Dataset.from_dataframe(df)
	return obs_ds.sel(time=slice(start, stop))


def _load_obs_zurich(obs_root, start, stop):
	"""Load Zurich observational CSV and convert to xarray.Dataset."""
	file_path = Path(obs_root) / "Zurich" / "Zurich_cleaned.csv"
	if not file_path.exists():
		raise FileNotFoundError(f"Zurich file not found: {file_path}")

	df = pd.read_csv(file_path)
	# Support both 'datetime' and 'DateTime' headers
	if "datetime" not in df.columns and "DateTime" not in df.columns:
		raise KeyError(f"Expected a 'datetime' column in {file_path}")

	dt_col = "datetime" if "datetime" in df.columns else "DateTime"
	df[dt_col] = pd.to_datetime(df[dt_col]).dt.tz_localize(None)
	df = df.set_index(dt_col)
	df.index.name = "time"
	df = df.apply(pd.to_numeric, errors="coerce")

	obs_attributes = _load_obs_metadata(obs_root=obs_root, city="Zurich")
	station_codes = [code for code in obs_attributes["code"].tolist() if code in df.columns]
	if not station_codes:
		# If no station ids match directly, try matching metadata 'station_number' or 'station_id' variants
		station_codes = [c for c in obs_attributes["code"].astype(str).tolist() if c in df.columns]

	df = df.loc[:, station_codes] if station_codes else df

	obs_ds = xr.Dataset.from_dataframe(df)
	return obs_ds.sel(time=slice(start, stop))


def _load_obs_turku(obs_root, start, stop):
	"""Load Turku observational CSV and convert to xarray.Dataset."""
	file_path = Path(obs_root) / "Turku" / "Turku_cleaned.csv"
	if not file_path.exists():
		raise FileNotFoundError(f"Turku file not found: {file_path}")

	df = pd.read_csv(file_path)
	# Support both 'datetime' and 'DateTime' headers
	if "datetime" not in df.columns and "DateTime" not in df.columns:
		raise KeyError(f"Expected a 'datetime' column in {file_path}")

	dt_col = "datetime" if "datetime" in df.columns else "DateTime"
	df[dt_col] = pd.to_datetime(df[dt_col]).dt.tz_localize(None)
	df = df.set_index(dt_col)
	df.index.name = "time"
	df = df.apply(pd.to_numeric, errors="coerce")

	obs_attributes = _load_obs_metadata(obs_root=obs_root, city="Turku")
	station_codes = [code for code in obs_attributes["code"].tolist() if code in df.columns]
	if not station_codes:
		# If no station ids match directly, try matching metadata 'station_number' or 'station_id' variants
		station_codes = [c for c in obs_attributes["code"].astype(str).tolist() if c in df.columns]

	df = df.loc[:, station_codes] if station_codes else df

	obs_ds = xr.Dataset.from_dataframe(df)
	return obs_ds.sel(time=slice(start, stop))


def _load_obs_metadata(obs_root, city):
	"""Load observation station metadata and normalize it to code/lat/lon columns."""
	meta_path = Path(obs_root) / str(city) / f"station_metadata_{city}.csv"
	if not meta_path.exists():
		raise FileNotFoundError(f"Metadata file not found: {meta_path}")

	meta = pd.read_csv(meta_path)
	column_lookup = {str(column).strip().lower(): column for column in meta.columns}

	code_column = None
	for candidate in ("code", "name", "station_id", "station", "site_name"):
		if candidate in column_lookup:
			code_column = column_lookup[candidate]
			break

	lat_column = None
	for candidate in ("lat", "latitude"):
		if candidate in column_lookup:
			lat_column = column_lookup[candidate]
			break

	lon_column = None
	for candidate in ("lon", "longitude"):
		if candidate in column_lookup:
			lon_column = column_lookup[candidate]
			break

	missing = [
		label
		for label, column in (("code", code_column), ("lat", lat_column), ("lon", lon_column))
		if column is None
	]
	if missing:
		raise KeyError(
			f"Unsupported metadata schema in {meta_path}. Missing required columns: {', '.join(missing)}"
		)

	obs_attributes = meta.loc[:, [code_column, lat_column, lon_column]].copy()
	obs_attributes.columns = ["code", "lat", "lon"]
	obs_attributes["code"] = obs_attributes["code"].astype(str).str.strip()
	obs_attributes["lat"] = pd.to_numeric(obs_attributes["lat"], errors="coerce")
	obs_attributes["lon"] = pd.to_numeric(obs_attributes["lon"], errors="coerce")
	return obs_attributes


def load_observations(city, obs_root, start, stop):
	"""Load city-specific observation dataset for the requested time period."""
	city_lc = str(city).strip().lower()
	if city_lc == "ghent":
		return _load_obs_ghent(obs_root=obs_root, city="Ghent", start=start, stop=stop)
	if city_lc == "amsterdam":
		return _load_obs_amsterdam(obs_root=obs_root, start=start, stop=stop)
	if city_lc == "berlin":
		return _load_obs_berlin(obs_root=obs_root, start=start, stop=stop)
	if city_lc == "zurich":
		return _load_obs_zurich(obs_root=obs_root, start=start, stop=stop)
	if city_lc == "rennes":
		return _load_obs_rennes(obs_root=obs_root, start=start, stop=stop)
	if city_lc == "birmingham":
		return _load_obs_birmingham(obs_root=obs_root, start=start, stop=stop)
	if city_lc == "turku":
		return _load_obs_turku(obs_root=obs_root, start=start, stop=stop)

	raise ValueError(f"Unsupported city '{city}'. Add a city-specific loader first.")


def classify_stations_from_urmask(obs_attr_df, urmask_ds):
	"""Classify stations as urban/rural/neither from nearest urmask gridcell."""
	from scipy.spatial import cKDTree

	mask_da = urmask_ds["urmask"]
	out = obs_attr_df.copy()

	station_lon = out["lon"].astype(float).to_numpy()
	station_lat = out["lat"].astype(float).to_numpy()

	grid_lon = np.asarray(mask_da["lon"].values).ravel()
	grid_lat = np.asarray(mask_da["lat"].values).ravel()
	grid_mask = np.asarray(mask_da.values).ravel().astype(float)

	finite_grid = np.isfinite(grid_lon) & np.isfinite(grid_lat)
	if not np.any(finite_grid):
		raise ValueError("urmask grid has no finite lon/lat coordinates.")

	tree = cKDTree(np.column_stack([grid_lon[finite_grid], grid_lat[finite_grid]]))
	_, nearest_idx = tree.query(np.column_stack([station_lon, station_lat]))

	finite_positions = np.flatnonzero(finite_grid)
	nearest_grid_idx = finite_positions[nearest_idx]

	out["urmask_value"] = grid_mask[nearest_grid_idx]
	out["station_class"] = np.where(
		out["urmask_value"] == 1,
		"urban",
		np.where(out["urmask_value"] == 0, "rural", "neither"),
	)

	# Backward-compatible field expected by current UrbanIsland methods.
	out["inside_city"] = np.where(out["station_class"] == "urban", 1, 0)
	return out


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
	out["urmask_value"] = out["inside_city"]
	out["station_class"] = np.where(out["inside_city"] == 1, "urban", "rural")
	return out


def prepare_observation_inputs_for_uhi(
	*,
	ds_obs,
	city,
	obs_root,
	ds_rcm_period,
	urmask,
	variable,
	ucdb_city=None,
	inside_city_method="ucdb",
	start=None,
	stop=None,
	convert_hourly_to_monthly=True,
	obs_downsample=None,
	debug=True,
):
	"""Build obs_attributes and obs_timeseries ready for UrbanIsland workflows.

	The station classification can be based on the UCDB city polygon or the
	urban mask. UCDB-based classification is the default.
	"""
	obs_attributes = _load_obs_metadata(obs_root=obs_root, city=city)

	obs_timeseries = ds_obs.to_dataframe().copy()
	obs_timeseries.columns = obs_timeseries.columns.astype(str).str.strip()
	# Ensure consistent datetime-like index (also when source was grouped by time.date).
	obs_timeseries.index = pd.to_datetime(obs_timeseries.index)

	station_codes = obs_attributes["code"].tolist()
	station_codes = [c for c in station_codes if c in obs_timeseries.columns]
	obs_timeseries = obs_timeseries[station_codes]

	obs_timeseries = obs_timeseries.apply(pd.to_numeric, errors="coerce").sort_index()

	
	model_start = pd.to_datetime(ds_rcm_period.time.min().values)
	model_end = pd.to_datetime(ds_rcm_period.time.max().values)
	if start is None:
		start = model_start

	if stop is None:
		stop = model_end

	# Avoid pandas slice errors from mixed types (e.g., datetime.date index vs string bounds).
	start = pd.to_datetime(start)
	stop = pd.to_datetime(stop)
	

	obs_timeseries = obs_timeseries.loc[start:stop]

	valid_cols = obs_timeseries.columns[obs_timeseries.notna().any(axis=0)]
	obs_timeseries = obs_timeseries[valid_cols]
	obs_attributes = obs_attributes[obs_attributes["code"].isin(valid_cols)].reset_index(drop=True)

	classification_method = str(inside_city_method).strip().lower()
	if classification_method in {"ucdb", "city", "shapefile"}:
		obs_attributes = classify_stations_from_ucdb_city(obs_attributes, ucdb_city)
	elif classification_method in {"urmask", "mask", "urban_mask"}:
		if urmask is None:
			raise ValueError("urmask must be provided when using the urmask-based classification.")
		obs_attributes = classify_stations_from_urmask(obs_attributes, urmask)
	else:
		raise ValueError("inside_city_method must be one of: 'ucdb', 'urmask'.")

	obs_daily_stat = None
	daily_stat = None
	if obs_downsample is None:
		obs_downsample = "monthly" if convert_hourly_to_monthly else "none"
	obs_downsample = str(obs_downsample).strip().lower()
	if obs_downsample not in {"none", "daily", "monthly"}:
		raise ValueError("obs_downsample must be one of: 'none', 'daily', 'monthly'.")

	if obs_downsample in {"daily", "monthly"}:
		daily_stat = _variable_to_daily_stat(variable)
		daily_all = observations_hourly_to_daily_tas_stats(obs_timeseries)
		stat_to_key = {"min": "tmin", "mean": "tmean", "max": "tmax"}
		obs_daily_stat = daily_all[stat_to_key[daily_stat]]

		if obs_downsample == "monthly":
			# Compute per-station monthly means but only keep months where a station
			# has at least half of the days present. Other months become NaN.
			# obs_daily_stat: daily index x stations columns
			# 1) monthly count of non-NaN days per station
			monthly_count = obs_daily_stat.resample("MS").count()
			# 2) monthly mean of daily values per station
			monthly_mean = obs_daily_stat.resample("MS").mean()
			# 3) number of days in each month (calendar-aware)
			months_index = monthly_count.index
			days_in_month = pd.Series([pd.Timestamp(ts).days_in_month for ts in months_index], index=months_index)
			# 4) mask months where station has >= half of the days
			threshold = days_in_month / 2.0
			mask = monthly_count.ge(threshold, axis=0)
			# 5) apply mask to monthly means
			obs_timeseries = monthly_mean.where(mask)
		else:
			obs_timeseries = obs_daily_stat

	if debug:
		print("station class counts:")
		print(obs_attributes["station_class"].value_counts(dropna=False))
		print("obs_timeseries shape:", obs_timeseries.shape)

	return obs_attributes, obs_timeseries, obs_daily_stat, daily_stat


def observations_hourly_to_daily_tas_stats(obs_df):
	"""Convert hourly station data to daily tmin, tmean and tmax DataFrames."""
	obs_df = obs_df.apply(pd.to_numeric, errors="coerce").sort_index()
	return {
		"tmin": obs_df.resample("D").min(),
		"tmean": obs_df.resample("D").mean(),
		"tmax": obs_df.resample("D").max(),
	}


def observations_hourly_to_monthly_mean_daily(obs_df, stat="min", monthly_label="MS"):
	"""Convert hourly station data to daily stats, then monthly means of those daily values."""
	stat = str(stat).lower()
	if stat not in {"min", "max", "mean"}:
		raise ValueError("stat must be one of: 'min', 'max', 'mean'")

	daily_all = observations_hourly_to_daily_tas_stats(obs_df)
	stat_to_key = {"min": "tmin", "mean": "tmean", "max": "tmax"}
	daily = daily_all[stat_to_key[stat]]
	monthly = daily.resample(monthly_label).mean()
	return daily, monthly


__all__ = [
	"load_simulation_data",
	"load_eurocities_daily_tas_stats",
	"prepare_simulation_for_uhi",
	"load_observations",
	"classify_stations_from_urmask",
	"prepare_observation_inputs_for_uhi",
	"observations_hourly_to_daily_tas_stats",
	"observations_hourly_to_monthly_mean_daily",
]

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
