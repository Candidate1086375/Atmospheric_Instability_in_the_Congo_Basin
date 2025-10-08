import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from glob import glob
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# Constants & region bounds (same as main script)
# ---------------------------
Cp = 1004
g = 9.81
Lv = 2.5e6

lat_min, lat_max = -23, 20
lon_min, lon_max = 7, 35
target_levels = [400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]

# Data directories (same as freedom.py)
data_dir = "./"
mse_base_dir = "/soge-home/data/analysis/era5/0.28125x0.28125/hourly/"
cape_dir = "/soge-home/users/catz0220/CAPE/"
output_dir = "scree_plots"
os.makedirs(output_dir, exist_ok=True)

seasons = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
    "ALL": list(range(1, 13))
}

# Full analysis period for comprehensive scree plot analysis
years = np.arange(1982, 2025) 
print(f"Using {len(years)} years for scree plot analysis: {years[0]}-{years[-1]}")

# Range of cluster numbers to test
cluster_range = range(2, 16)  # Test 2 to 15 clusters

# ---------------------------
# Helper functions (copied from main script)
# ---------------------------

def open_grib(fname):
    ds = xr.open_dataset(fname, engine='cfgrib', backend_kwargs={'indexpath': ''})
    return harmonise(ds)

def harmonise(ds):
    rename_map = {}
    for old_name, new_name in [('latitude', 'lat'), ('longitude', 'lon')]:
        if old_name in ds.coords:
            rename_map[old_name] = new_name
    if rename_map:
        ds = ds.rename(rename_map)
    for dim in ['number', 'surface', 'step']:
        if dim in ds.dims:
            ds = ds.isel({dim: 0})
    return ds

def subset_latlon(ds, lat_min, lat_max, lon_min, lon_max):
    """Subset dataset to region of interest."""
    if 'lat' not in ds.coords or 'lon' not in ds.coords:
        print(f"Warning: Missing lat/lon coordinates in dataset")
        return ds
    
    lats = ds['lat']
    lons = ds['lon']
    
    # Handle longitude wrapping (0-360 vs -180-180)
    if lons.max() > 180:
        # Convert region bounds to 0-360
        lon_min_360 = lon_min + 360 if lon_min < 0 else lon_min
        lon_max_360 = lon_max + 360 if lon_max < 0 else lon_max
    else:
        lon_min_360, lon_max_360 = lon_min, lon_max
    
    # Handle both ascending and descending latitude arrays
    if lats[0] > lats[-1]:
        ds_subset = ds.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min_360, lon_max_360))
    else:
        ds_subset = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_360, lon_max_360))
    
    return ds_subset

def pick_var(ds, candidates):
    cand_lower = [c.lower() for c in candidates]
    for v in ds.data_vars:
        if any(c in v.lower() for c in cand_lower):
            return ds[v]
    raise KeyError(f"None of {candidates} found in {list(ds.data_vars)}")

def fix_flux_sign_if_needed(flux, var_name):
    n_total = np.isfinite(flux.values).sum()
    n_neg = np.sum((flux.values < 0) & np.isfinite(flux.values))
    if n_total == 0:
        return flux
    pct_neg = n_neg / n_total * 100
    if pct_neg > 50:
        print(f"Flipping sign of {var_name} (negative values dominate: {pct_neg:.1f}%)")
        return -flux
    return flux

def seasonal_mean_field(flux_ds, year, season):
    months = seasons[season]
    if season == 'DJF':
        if year == years[0]:
            sel = flux_ds.sel(time=((flux_ds['time.year'] == year) & flux_ds['time.month'].isin([1, 2])))
        else:
            sel = flux_ds.sel(time=(
                ((flux_ds['time.year'] == year - 1) & (flux_ds['time.month'] == 12)) |
                ((flux_ds['time.year'] == year) & flux_ds['time.month'].isin([1, 2]))
            ))
    elif season == 'ALL':
        sel = flux_ds.sel(time=(flux_ds['time.year'] == year))
    else:
        sel = flux_ds.sel(time=(flux_ds['time.year'] == year) & flux_ds['time.month'].isin(months))
    return sel.mean('time') if sel.time.size > 0 else xr.full_like(flux_ds.isel(time=0), np.nan)

# ---------------------------
# MSE helper functions
# ---------------------------

def compute_mse_ts(ds):
    """Return MSE (kJ/kg) DataArray with dims (time, latitude, longitude)."""
    Z = ds['z'] / g
    T = ds['t']
    q = ds['q']
    mse = (Cp * T + g * Z + Lv * q) / 1000.0  # kJ/kg
    return mse

def ensure_latlon(ds):
    """Ensure standard lat/lon coordinates exist"""
    rename_map = {}
    if 'lat' not in ds.coords:
        if 'latitude' in ds.coords:
            rename_map['latitude'] = 'lat'
        elif 'Latitude' in ds.coords:
            rename_map['Latitude'] = 'lat'
    if 'lon' not in ds.coords:
        if 'longitude' in ds.coords:
            rename_map['longitude'] = 'lon'
        elif 'long' in ds.coords:
            rename_map['long'] = 'lon'
        elif 'Longitude' in ds.coords:
            rename_map['Longitude'] = 'lon'
    return ds.rename(rename_map) if rename_map else ds

def ensure_time_coord(ds):
    """Ensure standard time coordinate exists"""
    rename_map = {}
    if 'time' not in ds.coords:
        if 'valid_time' in ds.coords:
            rename_map['valid_time'] = 'time'
        elif 'forecast_time' in ds.coords:
            rename_map['forecast_time'] = 'time'
        elif 'Time' in ds.coords:
            rename_map['Time'] = 'time'
    return ds.rename(rename_map) if rename_map else ds

def load_mse_time_series(levels):
    """Load MSE data for multiple pressure levels from ERA5 monthly files - same method as freedom.py"""
    print("Loading MSE from ERA5 monthly files (NetCDF + GRIB file-by-file approach)...")
    
    # Initialize level containers
    level_data = {level: [] for level in levels}
    
    # Variables needed to compute MSE
    var_folders = {
        'z': 'geopotential',
        't': 'temperature',
        'q': 'specific_humidity'
    }
    
    files_processed = 0
    files_failed = 0
    
    for year in years:
        for month in range(1, 13):
            try:
                # Determine if we should use .grb files (from July 2023 onwards) or .nc files
                use_grb_files = (year > 2023) or (year == 2023 and month >= 7)
                
                if use_grb_files:
                    # Build file paths for GRIB files (from 07/23 onwards)
                    file_extension = 'grb'
                    subdir = 'grb'
                    paths = {
                        'z': os.path.join(mse_base_dir, var_folders['z'], subdir, f"era5_hourly_geopotential_{year}{month:02d}.grb"),
                        't': os.path.join(mse_base_dir, var_folders['t'], subdir, f"era5_hourly_temperature_{year}{month:02d}.grb"),
                        'q': os.path.join(mse_base_dir, var_folders['q'], subdir, f"era5_hourly_specific_humidity_{year}{month:02d}.grb"),
                    }
                else:
                    # Build file paths for NetCDF files (before 07/23)
                    file_extension = 'nc'
                    subdir = 'nc'
                    paths = {
                        'z': os.path.join(mse_base_dir, var_folders['z'], subdir, f"era5_hourly_geopotential_{year}_{month:02d}.nc"),
                        't': os.path.join(mse_base_dir, var_folders['t'], subdir, f"era5_hourly_temperature_{year}_{month:02d}.nc"),
                        'q': os.path.join(mse_base_dir, var_folders['q'], subdir, f"era5_hourly_specific_humidity_{year}_{month:02d}.nc"),
                    }
                
                # Check if all files exist
                if not all(os.path.exists(p) for p in paths.values()):
                    missing = [k for k, p in paths.items() if not os.path.exists(p)]
                    if files_failed % 50 == 0:  # Print every 50th missing file to reduce noise
                        print(f"  Missing {missing} for {year}-{month:02d} ({file_extension} files)")
                    files_failed += 1
                    continue
                
                print(f"  Processing {year}-{month:02d} ({file_extension} files)...")
                
                # Open all three variable files and normalize coordinates
                if use_grb_files:
                    # Open GRIB files with cfgrib engine and harmonize coordinates
                    ds_z = ensure_time_coord(ensure_latlon(harmonise(xr.open_dataset(paths['z'], engine='cfgrib', backend_kwargs={'indexpath': ''}))))
                    ds_t = ensure_time_coord(ensure_latlon(harmonise(xr.open_dataset(paths['t'], engine='cfgrib', backend_kwargs={'indexpath': ''}))))
                    ds_q = ensure_time_coord(ensure_latlon(harmonise(xr.open_dataset(paths['q'], engine='cfgrib', backend_kwargs={'indexpath': ''}))))
                else:
                    # Open NetCDF files normally
                    ds_z = ensure_time_coord(ensure_latlon(xr.open_dataset(paths['z'])))
                    ds_t = ensure_time_coord(ensure_latlon(xr.open_dataset(paths['t'])))
                    ds_q = ensure_time_coord(ensure_latlon(xr.open_dataset(paths['q'])))
                
                # Spatial subset first (before level selection for efficiency)
                ds_z = subset_latlon(ds_z, lat_min, lat_max, lon_min, lon_max)
                ds_t = subset_latlon(ds_t, lat_min, lat_max, lon_min, lon_max)
                ds_q = subset_latlon(ds_q, lat_min, lat_max, lon_min, lon_max)
                
                # Noon extraction 
                ds_z = ds_z.sel(time=ds_z['time'].dt.hour == 12)
                ds_t = ds_t.sel(time=ds_t['time'].dt.hour == 12)
                ds_q = ds_q.sel(time=ds_q['time'].dt.hour == 12)
                
                if ds_z.time.size == 0 or ds_t.time.size == 0 or ds_q.time.size == 0:
                    print(f"    No noon data for {year}-{month:02d}")
                    continue
                
                # Process each requested pressure level for this file
                for level in levels:
                    try:
                        # Select pressure level from each dataset
                        def sel_level(ds, target_level):
                            if 'level' in ds.dims:
                                if target_level in ds.level.values:
                                    return ds.sel(level=target_level)
                                else:
                                    return None
                            elif 'plev' in ds.dims:
                                target_plev = target_level * 100  # Convert hPa to Pa
                                if target_plev in ds.plev.values:
                                    return ds.sel(plev=target_plev)
                                else:
                                    return None
                            else:
                                return None
                        
                        ds_zl = sel_level(ds_z, level)
                        ds_tl = sel_level(ds_t, level)
                        ds_ql = sel_level(ds_q, level)
                        
                        if ds_zl is None or ds_tl is None or ds_ql is None:
                            continue  # Level not available in this file
                        
                        # Get variable names (first data variable in each dataset)
                        z_name = list(ds_zl.data_vars)[0]
                        t_name = list(ds_tl.data_vars)[0]
                        q_name = list(ds_ql.data_vars)[0]
                        
                        # Build combined dataset for MSE calculation
                        combo = xr.Dataset({
                            'z': ds_zl[z_name],
                            't': ds_tl[t_name], 
                            'q': ds_ql[q_name]
                        })
                        
                        # Compute MSE for this month and level
                        mse_month = compute_mse_ts(combo)
                        level_data[level].append(mse_month)
                        
                    except Exception as e:
                        print(f"    Error processing level {level} for {year}-{month:02d}: {e}")
                        continue
                
                files_processed += 1
                if files_processed % 50 == 0:
                    print(f"    Processed {files_processed} file sets so far...")
                    
            except Exception as e:
                print(f"  Error processing {year}-{month:02d}: {e}")
                files_failed += 1
                continue
    
    print(f"File processing complete: {files_processed} successful, {files_failed} failed")
    
    # Concatenate data for each level
    out = {}
    for level in levels:
        if level_data[level]:
            try:
                print(f"Concatenating {len(level_data[level])} time slices for {level} hPa...")
                out[level] = xr.concat(level_data[level], dim='time').sortby('time')
                print(f"  ✓ {level} hPa: {out[level].sizes.get('time', 0)} total time steps")
            except Exception as e:
                print(f"  Error concatenating {level} hPa: {e}")
        else:
            print(f"  ✗ No data found for {level} hPa")
    
    return out

def build_yearly_seasonal_means_for_scree(mse_ts, months):
    """Build yearly seasonal means for MSE scree analysis."""
    yearly_data = []
    
    for year in years:
        year_mse_stack = []
        valid_levels = []
        
        for level in sorted(mse_ts.keys()):
            if level not in mse_ts:
                continue
                
            da = mse_ts[level]
            # Filter by season months
            da_season = da.where(da['time'].dt.month.isin(months), drop=True)
            if da_season.time.size == 0:
                continue
                
            # Filter by year
            da_year = da_season.where(da_season['time'].dt.year == year, drop=True)
            if da_year.time.size == 0:
                continue
                
            # Calculate seasonal mean for this year and level
            seasonal_mean = da_year.mean("time", skipna=True)
            year_mse_stack.append(seasonal_mean.values.flatten())
            valid_levels.append(level)
        
        if len(year_mse_stack) > 0:
            # Stack levels for this year
            year_data = np.array(year_mse_stack).T  # [spatial_points, levels]
            yearly_data.append(year_data)
    
    if len(yearly_data) == 0:
        return None
    
    # Combine all years
    all_data = np.vstack(yearly_data)  # [all_spatial_points_all_years, levels]
    
    # Remove rows with any NaN values
    valid_rows = ~np.any(np.isnan(all_data), axis=1)
    clean_data = all_data[valid_rows]
    
    if len(clean_data) == 0:
        return None
    
    return clean_data

def prepare_mse_seasonal_data_for_scree(season):
    """Prepare MSE data for seasonal scree plot analysis."""
    print(f"Preparing MSE data for {season}...")
    months = seasons[season]
    
    # Access the global MSE time series
    global mse_ts
    if mse_ts is None or len(mse_ts) == 0:
        print("  MSE time series not available")
        return None
    
    return build_yearly_seasonal_means_for_scree(mse_ts, months)

# ---------------------------
# CAPE helper functions
# ---------------------------

def load_cape_time_series():
    """Load CAPE data from NetCDF files in the CAPE directory - same method as freedom.py"""
    print("Loading CAPE from NetCDF files in CAPE directory...")
    cape_data = []
    
    # Get all NetCDF files from CAPE directory
    cape_files = glob(os.path.join(cape_dir, "*.nc"))
    
    if not cape_files:
        print(f"✗ No NetCDF files found in {cape_dir}")
        return None
    
    print(f"Found {len(cape_files)} NetCDF files to process")
    
    for cape_file in sorted(cape_files):
        try:
            print(f"  Processing: {os.path.basename(cape_file)}")
            
            # Open dataset and handle coordinate naming
            ds = xr.open_dataset(cape_file)
            ds = ensure_time_coord(ensure_latlon(ds))
            
            # Get CAPE variable (try different names)
            cape_var = None
            for var_name in ['cape', 'CAPE', 'convective_available_potential_energy']:
                if var_name in ds.data_vars:
                    cape_var = ds[var_name]
                    break
            
            if cape_var is None:
                print(f"    Warning: No CAPE variable found in {os.path.basename(cape_file)}")
                continue
            
            # Spatial subset
            cape_subset = subset_latlon(cape_var, lat_min, lat_max, lon_min, lon_max)
            
            # Noon extraction (12 UTC)
            cape_noon = cape_subset.sel(time=cape_subset['time'].dt.hour == 12)
            
            if cape_noon.time.size == 0:
                print(f"    Warning: No noon data in {os.path.basename(cape_file)}, using all times")
                cape_noon = cape_subset
            
            if cape_noon.time.size > 0:
                cape_data.append(cape_noon)
                print(f"    ✓ Loaded {cape_noon.time.size} time steps")
            else:
                print(f"    ✗ No valid data in {os.path.basename(cape_file)}")
                
        except Exception as e:
            print(f"    ✗ Error processing {os.path.basename(cape_file)}: {e}")
            continue
    
    if len(cape_data) == 0:
        print("✗ No CAPE data could be loaded")
        return None
    
    # Concatenate all CAPE data
    try:
        cape_combined = xr.concat(cape_data, dim='time').sortby('time')
        print(f"✓ Successfully loaded CAPE data: {cape_combined.sizes}")
        return cape_combined
    except Exception as e:
        print(f"✗ Error concatenating CAPE data: {e}")
        return None

def build_yearly_seasonal_means_for_cape_scree(cape_ts, months):
    """Build yearly seasonal means for CAPE scree analysis."""
    yearly_data = []
    
    for year in years:
        # Filter by season months
        cape_season = cape_ts.where(cape_ts['time'].dt.month.isin(months), drop=True)
        if cape_season.time.size == 0:
            continue
            
        # Filter by year
        cape_year = cape_season.where(cape_season['time'].dt.year == year, drop=True)
        if cape_year.time.size == 0:
            continue
            
        # Calculate seasonal mean for this year
        seasonal_mean = cape_year.mean("time", skipna=True)
        year_data = seasonal_mean.values.flatten()
        
        # Remove NaN values for this year
        valid_data = year_data[~np.isnan(year_data)]
        if len(valid_data) > 0:
            yearly_data.extend(valid_data)
    
    if len(yearly_data) == 0:
        return None
    
    # Return as column vector for clustering
    return np.array(yearly_data).reshape(-1, 1)

def prepare_cape_seasonal_data_for_scree(season):
    """Prepare CAPE data for seasonal scree plot analysis."""
    print(f"Preparing CAPE data for {season}...")
    months = seasons[season]
    
    # Access the global CAPE time series
    global cape_ts
    if cape_ts is None:
        print("  CAPE time series not available")
        return None
    
    return build_yearly_seasonal_means_for_cape_scree(cape_ts, months)

# ---------------------------
# Flux data preparation functions
# ---------------------------

def load_flux_time_series(var_name):
    """Load flux data from ERA5 NetCDF files - same method as freedom.py"""
    print(f"Loading {var_name} from ERA5 NetCDF files...")
    
    # Variable configuration
    var_config = {
        'LHFLX': {'dir': 'surface_latent_heat_flux', 'var': 'slhf'},
        'SHFLX': {'dir': 'surface_sensible_heat_flux', 'var': 'sshf'}
    }
    
    if var_name not in var_config:
        print(f"Unknown variable: {var_name}")
        return None
    
    config = var_config[var_name]
    var_dir = config['dir']
    var_key = config['var']
    
    flux_base_dir = "/soge-home/data/analysis/era5/0.28125x0.28125/hourly/"
    
    flux_data = []
    files_processed = 0
    files_failed = 0
    
    for year in years:
        # Strategy for flux data loading (yearly files ≤2019, monthly files >2019)
        if year <= 2019:
            # Try yearly file first
            flux_file = os.path.join(flux_base_dir, var_dir, 'nc', f"era5_hourly_{var_dir}_{year}.nc")
            if os.path.exists(flux_file):
                try:
                    print(f"  Loading yearly file: {os.path.basename(flux_file)}")
                    ds = xr.open_dataset(flux_file)
                    ds = ensure_time_coord(ensure_latlon(ds))
                    
                    # Extract flux variable
                    if var_key in ds.data_vars:
                        flux = ds[var_key]
                    else:
                        print(f"    Variable {var_key} not found in {flux_file}")
                        continue
                    
                    # Spatial subset and noon extraction
                    flux_subset = subset_latlon(flux, lat_min, lat_max, lon_min, lon_max)
                    flux_noon = flux_subset.sel(time=flux_subset['time'].dt.hour == 12)
                    
                    if flux_noon.time.size > 0:
                        flux_data.append(flux_noon)
                        files_processed += 1
                        print(f"    ✓ Loaded {flux_noon.time.size} time steps")
                    else:
                        print(f"    ✗ No noon data in {flux_file}")
                        
                except Exception as e:
                    print(f"    ✗ Error loading {flux_file}: {e}")
                    files_failed += 1
            else:
                print(f"  Missing yearly file for {year}, trying monthly files...")
                # Fall back to monthly files
                for month in range(1, 13):
                    flux_file = os.path.join(flux_base_dir, var_dir, 'nc', f"era5_hourly_{var_dir}_{year}{month:02d}.nc")
                    if os.path.exists(flux_file):
                        try:
                            ds = xr.open_dataset(flux_file)
                            ds = ensure_time_coord(ensure_latlon(ds))
                            
                            if var_key in ds.data_vars:
                                flux = ds[var_key]
                                flux_subset = subset_latlon(flux, lat_min, lat_max, lon_min, lon_max)
                                flux_noon = flux_subset.sel(time=flux_subset['time'].dt.hour == 12)
                                
                                if flux_noon.time.size > 0:
                                    flux_data.append(flux_noon)
                                    
                        except Exception as e:
                            print(f"    ✗ Error loading monthly file {flux_file}: {e}")
                            files_failed += 1
                            continue
        else:
            # Use monthly files for years > 2019
            for month in range(1, 13):
                flux_file = os.path.join(flux_base_dir, var_dir, 'nc', f"era5_hourly_{var_dir}_{year}{month:02d}.nc")
                if os.path.exists(flux_file):
                    try:
                        ds = xr.open_dataset(flux_file)
                        ds = ensure_time_coord(ensure_latlon(ds))
                        
                        if var_key in ds.data_vars:
                            flux = ds[var_key]
                            flux_subset = subset_latlon(flux, lat_min, lat_max, lon_min, lon_max)
                            flux_noon = flux_subset.sel(time=flux_subset['time'].dt.hour == 12)
                            
                            if flux_noon.time.size > 0:
                                flux_data.append(flux_noon)
                                
                    except Exception as e:
                        print(f"    ✗ Error loading {flux_file}: {e}")
                        files_failed += 1
                        continue
                else:
                    files_failed += 1
    
    print(f"File processing complete: {files_processed} successful, {files_failed} failed")
    
    if len(flux_data) == 0:
        print(f"✗ No {var_name} data could be loaded")
        return None
    
    # Concatenate all flux data
    try:
        flux_combined = xr.concat(flux_data, dim='time').sortby('time')
        print(f"✓ Successfully loaded {var_name} data: {flux_combined.sizes}")
        return flux_combined
    except Exception as e:
        print(f"✗ Error concatenating {var_name} data: {e}")
        return None

def prepare_flux_data_for_scree(flux_ds, season):
    """Prepare flux data for scree plot analysis."""
    print(f"Preparing flux data for {season}...")
    nlat = len(flux_ds['lat'])
    nlon = len(flux_ds['lon'])
    nyears = len(years)

    # Build seasonal mean data array [year, lat, lon]
    data_array = np.full((nyears, nlat, nlon), np.nan)
    for i, y in enumerate(years):
        season_mean = seasonal_mean_field(flux_ds, y, season)
        data_array[i] = season_mean.values

    # Reshape for clustering and remove NaN values
    data_flat = data_array.flatten()
    valid_data = data_flat[~np.isnan(data_flat)]
    
    if len(valid_data) == 0:
        return None
    
    return valid_data.reshape(-1, 1)

# ---------------------------
# Scree plot generation functions
# ---------------------------

def calculate_clustering_metrics(data, cluster_range):
    """Calculate inertia (within-cluster sum of squares) and silhouette scores for different cluster numbers."""
    inertias = []
    silhouette_scores = []
    
    for n_clusters in cluster_range:
        print(f"  Testing {n_clusters} clusters...")
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(data)
        
        # Calculate inertia (within-cluster sum of squares)
        inertias.append(kmeans.inertia_)
        
        # Calculate silhouette score (only if we have more than 1 cluster and less than n_samples)
        if n_clusters > 1 and n_clusters < len(data):
            try:
                sil_score = silhouette_score(data, labels)
                silhouette_scores.append(sil_score)
            except:
                silhouette_scores.append(np.nan)
        else:
            silhouette_scores.append(np.nan)
    
    return inertias, silhouette_scores

def find_elbow_point(inertias, cluster_range):
    """Find the elbow point using the rate of change method."""
    # Calculate the rate of change (differences between consecutive points)
    diffs = np.diff(inertias)
    # Calculate the second derivative (rate of change of the rate of change)
    second_diffs = np.diff(diffs)
    
    # Find the point where the second derivative is maximum (sharpest change)
    if len(second_diffs) > 0:
        elbow_idx = np.argmax(second_diffs) + 2  # +2 because we lost 2 points in double diff
        elbow_clusters = cluster_range[elbow_idx] if elbow_idx < len(cluster_range) else cluster_range[-1]
    else:
        elbow_clusters = cluster_range[len(cluster_range)//2]  # Default to middle if calculation fails
    
    return elbow_clusters

def plot_scree_analysis(variable_name, season, cluster_range, inertias, silhouette_scores, elbow_point):
    """Create a comprehensive scree plot with both inertia and silhouette analysis."""
    
    # Define variable-specific colors
    var_colors = {
        'MSE': 'blue',
        'CAPE': 'orange', 
        'LHFLX': 'red',
        'SHFLX': 'purple'
    }
    
    var_color = var_colors.get(variable_name, 'black')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Inertia (Elbow Plot)
    ax1.plot(cluster_range, inertias, 'o-', color=var_color, linewidth=2, markersize=8)
    ax1.axvline(x=elbow_point, color='red', linestyle='--', linewidth=2, 
                label=f'Elbow at {elbow_point} clusters')
    ax1.set_xlabel('Number of Clusters', fontsize=12)
    ax1.set_ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=12)
    ax1.set_title(f'{variable_name} - {season} Season\nElbow Method for Optimal Clusters', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add percentage of variance explained
    total_variance = inertias[0]
    variance_explained = [(total_variance - inertia) / total_variance * 100 for inertia in inertias]
    ax1_twin = ax1.twinx()
    ax1_twin.plot(cluster_range, variance_explained, '^-', color='green', alpha=0.7, label='Variance Explained %')
    ax1_twin.set_ylabel('Variance Explained (%)', fontsize=12, color='green')
    ax1_twin.tick_params(axis='y', labelcolor='green')
    
    # Plot 2: Silhouette Scores
    valid_sil_scores = [score for score in silhouette_scores if not np.isnan(score)]
    valid_cluster_range = cluster_range[:len(valid_sil_scores)]
    
    if len(valid_sil_scores) > 0:
        ax2.plot(valid_cluster_range, valid_sil_scores, 'o-', color=var_color, linewidth=2, markersize=8)
        
        # Find the best silhouette score
        best_sil_idx = np.argmax(valid_sil_scores)
        best_sil_clusters = valid_cluster_range[best_sil_idx]
        ax2.axvline(x=best_sil_clusters, color='green', linestyle='--', linewidth=2,
                   label=f'Best Silhouette at {best_sil_clusters} clusters')
        
        ax2.set_xlabel('Number of Clusters', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title(f'{variable_name} - {season} Season\nSilhouette Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(bottom=0)
    else:
        ax2.text(0.5, 0.5, 'Silhouette scores\nnot available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title(f'{variable_name} - {season} Season\nSilhouette Analysis', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"{variable_name}_{season}_scree_analysis.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved scree plot: {save_path}")
    
    return elbow_point, best_sil_clusters if len(valid_sil_scores) > 0 else None

# ---------------------------
# Main analysis function
# ---------------------------

def analyze_variable_clustering(variable_name, data_prep_func, seasons_to_analyze=None):
    """Analyze clustering for a specific variable across seasons."""
    
    if seasons_to_analyze is None:
        seasons_to_analyze = ['DJF', 'MAM', 'JJA', 'SON', 'ALL']
    
    print(f"\n{'='*60}")
    print(f"CLUSTERING ANALYSIS FOR {variable_name.upper()}")
    print(f"{'='*60}")
    
    results = {}
    
    for season in seasons_to_analyze:
        print(f"\nAnalyzing {variable_name} for {season} season...")
        
        try:
            # Prepare data using the provided function
            data = data_prep_func(season)
            
            if data is None:
                print(f"  No valid data for {variable_name} in {season} season")
                continue
            
            print(f"  Data shape: {data.shape}")
            print(f"  Valid data points: {len(data)}")
            
            # Calculate clustering metrics
            print(f"  Calculating clustering metrics...")
            inertias, silhouette_scores = calculate_clustering_metrics(data, cluster_range)
            
            # Find elbow point
            elbow_point = find_elbow_point(inertias, cluster_range)
            
            # Create and save scree plot
            elbow_clusters, best_sil_clusters = plot_scree_analysis(
                variable_name, season, cluster_range, inertias, silhouette_scores, elbow_point
            )
            
            # Store results
            results[season] = {
                'elbow_point': elbow_clusters,
                'best_silhouette': best_sil_clusters,
                'inertias': inertias,
                'silhouette_scores': silhouette_scores,
                'variance_explained_at_elbow': ((inertias[0] - inertias[elbow_clusters-2]) / inertias[0] * 100) if elbow_clusters > 2 else 0
            }
            
            print(f"  Elbow method suggests: {elbow_clusters} clusters")
            if best_sil_clusters:
                print(f"  Best silhouette score at: {best_sil_clusters} clusters")
            print(f"  Variance explained at elbow: {results[season]['variance_explained_at_elbow']:.1f}%")
            
        except Exception as e:
            print(f"  Error analyzing {variable_name} for {season}: {e}")
            continue
    
    return results

# ---------------------------
# Summary functions
# ---------------------------

def create_4x4_scree_grid(all_results):
    """Create a 5x4 grid showing scree plots for each variable across the 4 main seasons."""
    
    variables = ['MSE', 'CAPE', 'LHFLX', 'SHFLX']  # 4 variables with CAPE as second
    seasons = ['DJF', 'MAM', 'JJA', 'SON']  # 4 seasons
    
    # Define consistent color schemes for each variable
    var_colors = {
        'MSE': 'blue',      # Blue tones for MSE
        'CAPE': 'orange',   # Orange tones for CAPE (distinct from MSE)
        'LHFLX': 'red',     # Red tones for latent heat flux
        'SHFLX': 'purple'   # Purple tones for sensible heat flux
    }
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))  # 4x4 grid for 4 variables
    fig.suptitle('Scree Plot Analysis: 4 Variables × 4 Seasons\n(MSE, CAPE, LHFLX, SHFLX)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Add column headers (seasons)
    for j, season in enumerate(seasons):
        axes[0, j].set_title(f'{season}', fontsize=14, fontweight='bold', pad=20)
    
    # Add row headers (variables)
    for i, var in enumerate(variables):
        axes[i, 0].set_ylabel(f'{var}\n\nInertia', fontsize=12, fontweight='bold')
    
    for i, var in enumerate(variables):
        for j, season in enumerate(seasons):
            ax = axes[i, j]
            
            # Get variable-specific color
            var_color = var_colors.get(var, 'black')
            
            # Check if we have data for this variable and season
            if (var in all_results and season in all_results[var] and 
                all_results[var][season] is not None):
                
                result = all_results[var][season]
                inertias = result['inertias']
                elbow_point = result['elbow_point']
                
                # Plot the scree curve with variable-specific color
                ax.plot(cluster_range, inertias, 'o-', color=var_color, 
                       linewidth=2, markersize=6, alpha=0.8)
                ax.axvline(x=elbow_point, color='red', linestyle='--', linewidth=2, alpha=0.8)
                
                # Add elbow point annotation
                ax.text(0.02, 0.98, f'Elbow: {elbow_point}', transform=ax.transAxes, 
                       fontsize=10, fontweight='bold', verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                
                # Add variance explained at elbow
                variance_explained = result['variance_explained_at_elbow']
                ax.text(0.02, 0.85, f'Var Exp: {variance_explained:.1f}%', 
                       transform=ax.transAxes, fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
                
            else:
                # No data available
                ax.text(0.5, 0.5, f'No {var}\ndata for\n{season}', 
                       transform=ax.transAxes, ha='center', va='center', 
                       fontsize=10, style='italic', color='gray')
                ax.set_xlim(cluster_range[0], cluster_range[-1])
            
            # Formatting
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Clusters' if i == 3 else '')  # Only bottom row gets x-label (4th row)
            
            # Set consistent axis limits
            if var in all_results and any(season in all_results[var] for season in seasons):
                # Find max inertia across all seasons for this variable to set consistent y-axis
                max_inertia = 0
                for s in seasons:
                    if s in all_results[var] and all_results[var][s]:
                        max_inertia = max(max_inertia, max(all_results[var][s]['inertias']))
                if max_inertia > 0:
                    ax.set_ylim(0, max_inertia * 1.05)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # Save 5x4 grid plot
    save_path = os.path.join(output_dir, "scree_plots_5x4_grid_with_CAPE.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved 5x4 scree grid: {save_path}")

def create_summary_plot(all_results):
    """Create a summary plot showing optimal cluster numbers for all variables and seasons."""
    
    variables = ['MSE', 'CAPE', 'LHFLX', 'SHFLX']  # 4 variables including CAPE
    seasons = ['DJF', 'MAM', 'JJA', 'SON', 'ALL']
    
    # Define consistent colors for each variable
    var_colors = {
        'MSE': 'blue',
        'CAPE': 'orange',
        'LHFLX': 'red',
        'SHFLX': 'purple'
    }
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Elbow method results
    x_pos = np.arange(len(seasons))
    width = 0.18  # Adjusted bars for 4 variables
    
    for i, var in enumerate(variables):
        elbow_points = []
        for season in seasons:
            if season in all_results.get(var, {}) and all_results[var][season]:
                elbow_points.append(all_results[var][season]['elbow_point'])
            else:
                elbow_points.append(0)  # No data
        
        color = var_colors.get(var, 'gray')
        ax1.bar(x_pos + i * width, elbow_points, width, label=var, alpha=0.8, color=color)
    
    ax1.set_xlabel('Season', fontsize=12)
    ax1.set_ylabel('Optimal Number of Clusters (Elbow Method)', fontsize=12)
    ax1.set_title('Optimal Cluster Numbers by Variable and Season\n(Elbow Method)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos + width * (len(variables) - 1) / 2)
    ax1.set_xticklabels(seasons)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(cluster_range) + 1)
    
    # Plot 2: Silhouette method results
    for i, var in enumerate(variables):
        sil_points = []
        for season in seasons:
            if (season in all_results.get(var, {}) and all_results[var][season] and 
                all_results[var][season]['best_silhouette'] is not None):
                sil_points.append(all_results[var][season]['best_silhouette'])
            else:
                sil_points.append(0)  # No data
        
        color = var_colors.get(var, 'gray')
        ax2.bar(x_pos + i * width, sil_points, width, label=var, alpha=0.8, color=color)
    
    ax2.set_xlabel('Season', fontsize=12)
    ax2.set_ylabel('Optimal Number of Clusters (Silhouette Method)', fontsize=12)
    ax2.set_title('Optimal Cluster Numbers by Variable and Season\n(Silhouette Method)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos + width * (len(variables) - 1) / 2)
    ax2.set_xticklabels(seasons)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(cluster_range) + 1)
    
    plt.tight_layout()
    
    # Save summary plot
    save_path = os.path.join(output_dir, "clustering_summary_all_variables_with_CAPE.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved summary plot: {save_path}")

def save_results_to_csv(all_results):
    """Save clustering analysis results to CSV file."""
    
    print(f"\nSaving clustering results to CSV...")
    
    # Create list to store all results
    csv_data = []
    
    # Define seasons and variables
    seasons = ['DJF', 'MAM', 'JJA', 'SON', 'ALL']
    
    for variable in all_results.keys():
        for season in seasons:
            if season in all_results[variable] and all_results[variable][season]:
                result = all_results[variable][season]
                
                # Elbow method result
                csv_data.append({
                    'Variable': variable,
                    'Season': season,
                    'Method': 'Elbow',
                    'Optimal_Clusters': result['elbow_point'],
                    'Variance_Explained_Percent': round(result['variance_explained_at_elbow'], 2),
                    'Final_Inertia': round(result['inertias'][result['elbow_point']-2], 2) if result['elbow_point'] > 2 else 'N/A',
                    'Recommended': 'Yes' if result['elbow_point'] <= 8 else 'Check'  # Flag if too many clusters
                })
                
                # Silhouette method result (if available)
                if result['best_silhouette'] is not None:
                    # Find the silhouette score for the best number of clusters
                    best_sil_score = 'N/A'
                    if len(result['silhouette_scores']) > 0:
                        valid_scores = [score for score in result['silhouette_scores'] if not np.isnan(score)]
                        if len(valid_scores) > 0:
                            best_sil_score = round(max(valid_scores), 3)
                    
                    csv_data.append({
                        'Variable': variable,
                        'Season': season,
                        'Method': 'Silhouette',
                        'Optimal_Clusters': result['best_silhouette'],
                        'Variance_Explained_Percent': 'N/A',
                        'Final_Inertia': 'N/A',
                        'Recommended': 'Yes' if result['best_silhouette'] <= 8 else 'Check'
                    })
                
                # Add recommended clusters (consensus between methods)
                if result['best_silhouette'] is not None:
                    # If both methods agree or are close (within 1-2 clusters)
                    elbow_clusters = result['elbow_point']
                    sil_clusters = result['best_silhouette']
                    
                    if abs(elbow_clusters - sil_clusters) <= 1:
                        recommended_clusters = elbow_clusters  # Use elbow as primary
                        consensus = 'Strong'
                    elif abs(elbow_clusters - sil_clusters) <= 2:
                        recommended_clusters = elbow_clusters  # Use elbow as primary
                        consensus = 'Moderate'
                    else:
                        recommended_clusters = elbow_clusters  # Default to elbow
                        consensus = 'Weak'
                    
                    csv_data.append({
                        'Variable': variable,
                        'Season': season,
                        'Method': 'Consensus',
                        'Optimal_Clusters': recommended_clusters,
                        'Variance_Explained_Percent': round(result['variance_explained_at_elbow'], 2),
                        'Final_Inertia': 'N/A',
                        'Recommended': consensus
                    })
                else:
                    # Only elbow method available
                    csv_data.append({
                        'Variable': variable,
                        'Season': season,
                        'Method': 'Consensus',
                        'Optimal_Clusters': result['elbow_point'],
                        'Variance_Explained_Percent': round(result['variance_explained_at_elbow'], 2),
                        'Final_Inertia': 'N/A',
                        'Recommended': 'Elbow_Only'
                    })
            else:
                # No data available for this variable-season combination
                csv_data.append({
                    'Variable': variable,
                    'Season': season,
                    'Method': 'No_Data',
                    'Optimal_Clusters': 'N/A',
                    'Variance_Explained_Percent': 'N/A',
                    'Final_Inertia': 'N/A',
                    'Recommended': 'N/A'
                })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(csv_data)
    
    # Sort by Variable, Season, Method for better organization
    method_order = ['Elbow', 'Silhouette', 'Consensus', 'No_Data']
    df['Method'] = pd.Categorical(df['Method'], categories=method_order, ordered=True)
    df = df.sort_values(['Variable', 'Season', 'Method']).reset_index(drop=True)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "clustering_analysis_results.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"✓ Results saved to: {csv_path}")
    
    # Also create a simplified "recommended clusters" CSV
    consensus_data = df[df['Method'] == 'Consensus'].copy()
    consensus_data = consensus_data[['Variable', 'Season', 'Optimal_Clusters', 'Variance_Explained_Percent', 'Recommended']]
    consensus_data.columns = ['Variable', 'Season', 'Recommended_Clusters', 'Variance_Explained_Percent', 'Confidence']
    
    simplified_path = os.path.join(output_dir, "recommended_clusters_summary.csv")
    consensus_data.to_csv(simplified_path, index=False)
    
    print(f"✓ Simplified recommendations saved to: {simplified_path}")
    
    return csv_path, simplified_path

def print_summary_table(all_results):
    """Print a summary table of optimal cluster numbers."""
    
    print(f"\n{'='*80}")
    print("SUMMARY: OPTIMAL CLUSTER NUMBERS")
    print(f"{'='*80}")
    
    # Create summary table
    seasons = ['DJF', 'MAM', 'JJA', 'SON', 'ALL']
    variables = list(all_results.keys())
    
    print(f"{'Variable':<10} {'Method':<12} {'DJF':<5} {'MAM':<5} {'JJA':<5} {'SON':<5} {'ALL':<5}")
    print("-" * 60)
    
    for var in variables:
        # Elbow method row
        elbow_row = f"{var:<10} {'Elbow':<12}"
        for season in seasons:
            if season in all_results[var] and all_results[var][season]:
                elbow_row += f" {all_results[var][season]['elbow_point']:<4}"
            else:
                elbow_row += f" {'N/A':<4}"
        print(elbow_row)
        
        # Silhouette method row
        sil_row = f"{'':10} {'Silhouette':<12}"
        for season in seasons:
            if (season in all_results[var] and all_results[var][season] and 
                all_results[var][season]['best_silhouette'] is not None):
                sil_row += f" {all_results[var][season]['best_silhouette']:<4}"
            else:
                sil_row += f" {'N/A':<4}"
        print(sil_row)
        print()

# ---------------------------
# Main execution
# ---------------------------

if __name__ == "__main__":
    print("CLUSTERING SCREE PLOT ANALYSIS")
    print("=" * 50)
    print(f"Testing cluster range: {min(cluster_range)} to {max(cluster_range)}")
    print(f"Years analyzed: {years[0]} to {years[-1]} ({len(years)} years)")
    
    all_results = {}
    
    # Global variables to store time series for seasonal analysis
    global mse_ts, cape_ts
    
    # Load data first
    print("\nLoading base datasets...")
    
    # Load flux data using freedom.py method
    try:
        lhflx = load_flux_time_series('LHFLX')
        shflx = load_flux_time_series('SHFLX')
        
        if lhflx is not None and shflx is not None:
            print("✓ Flux data loaded successfully from ERA5 NetCDF files")
        else:
            print("✗ Some flux data could not be loaded")
    except Exception as e:
        print(f"✗ Error loading flux data: {e}")
        lhflx = shflx = None
    
    # Load MSE data
    try:
        mse_ts = load_mse_time_series(target_levels)
        if mse_ts:
            print("✓ MSE data loaded successfully")
        else:
            mse_ts = None
    except Exception as e:
        print(f"✗ Error loading MSE data: {e}")
        mse_ts = None
    
    # Load CAPE data
    try:
        cape_ts = load_cape_time_series()
        if cape_ts is not None:
            print("✓ CAPE data loaded successfully")
        else:
            cape_ts = None
    except Exception as e:
        print(f"✗ Error loading CAPE data: {e}")
        cape_ts = None
    
    # Analyze each variable
    
    # 1. MSE Analysis
    if mse_ts is not None:
        all_results['MSE'] = analyze_variable_clustering(
            'MSE', 
            prepare_mse_seasonal_data_for_scree,
            ['DJF', 'MAM', 'JJA', 'SON']  # Analyze MSE for each season
        )
    
    # 2. CAPE Analysis (second position as requested)
    if cape_ts is not None:
        all_results['CAPE'] = analyze_variable_clustering(
            'CAPE',
            prepare_cape_seasonal_data_for_scree,
            ['DJF', 'MAM', 'JJA', 'SON']  # Analyze CAPE for each season
        )
    
    # 3. LHFLX Analysis
    if lhflx is not None:
        all_results['LHFLX'] = analyze_variable_clustering(
            'LHFLX',
            lambda season: prepare_flux_data_for_scree(lhflx, season)
        )
    
    # 4. SHFLX Analysis
    if shflx is not None:
        all_results['SHFLX'] = analyze_variable_clustering(
            'SHFLX',
            lambda season: prepare_flux_data_for_scree(shflx, season)
        )
    
    # Create summary visualizations and tables
    if all_results:
        # Create 4x4 grid of scree plots (including CAPE)
        create_4x4_scree_grid(all_results)
        
        # Create summary bar charts
        create_summary_plot(all_results)
        
        # Save results to CSV files
        csv_path, simplified_path = save_results_to_csv(all_results)
        
        # Print summary table
        print_summary_table(all_results)
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"All plots saved to: {output_dir}")
        print(f"Full results CSV: {csv_path}")
        print(f"Simplified recommendations CSV: {simplified_path}")
        print(f"{'='*80}")
        
        # Recommendations
        print("\nRECOMMENDations:")
        print("1. Use the elbow method as the primary indicator for optimal cluster numbers")
        print("2. Cross-validate with silhouette scores where available")
        print("3. Consider domain knowledge - 5 clusters often provides good interpretability")
        print("4. If methods disagree, test both numbers and choose based on interpretability")
        
    else:
        print("No successful analyses completed. Check data availability and paths.")
