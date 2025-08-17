import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.cluster import KMeans
import pandas as pd
from glob import glob

# ---------------------------
# Constants & region bounds
# ---------------------------
Cp = 1004
g = 9.81
Lv = 2.5e6

lat_min, lat_max = -10, 11
lon_min, lon_max = 10, 30
target_levels = list(range(400, 1001, 50))

data_dir = "./"
ndvi_data_dir = "/Volumes/LaCie/Python/NDVI"
output_dir = "cluster_animations"
os.makedirs(output_dir, exist_ok=True)

seasons = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
    "ALL": list(range(1, 13))
}

years = np.arange(1983, 2025)
N_CLUSTERS = 5  # Number of clusters for all variables

# ---------------------------
# Helper functions
# ---------------------------

def open_grib(fname):
    ds = xr.open_dataset(fname, engine='cfgrib', backend_kwargs={'indexpath': ''})
    return harmonise(ds)

def harmonise(ds):
    # Rename latitude/longitude coords to lat/lon consistently
    rename_map = {}
    for old_name, new_name in [('latitude', 'lat'), ('longitude', 'lon')]:
        if old_name in ds.coords:
            rename_map[old_name] = new_name
    if rename_map:
        ds = ds.rename(rename_map)
    # Remove unwanted dims if present
    for dim in ['number', 'surface', 'step']:
        if dim in ds.dims:
            ds = ds.isel({dim: 0})
    return ds

def subset_latlon(ds, lat_min, lat_max, lon_min, lon_max):
    # Work with lat/lon coords renamed already
    if 'lat' not in ds.coords or 'lon' not in ds.coords:
        raise KeyError(f"Dataset missing lat or lon coords")
    lats = ds['lat']
    # Correct slice depending on lat ordering
    if lats[0] > lats[-1]:
        return ds.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    else:
        return ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

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
        print(f"Warning: No valid data for {var_name}")
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

def prepare_flux_features(flux_ds, season):
    nlat, nlon, nyears = len(flux_ds['lat']), len(flux_ds['lon']), len(years)
    flux_stack = np.full((nyears, nlat, nlon), np.nan)
    for i, y in enumerate(years):
        flux_stack[i] = seasonal_mean_field(flux_ds, y, season).values
    flux_flat = flux_stack.reshape(nyears * nlat * nlon, 1)
    valid = ~np.isnan(flux_flat).flatten()
    return flux_flat[valid], valid, (nyears, nlat, nlon), flux_ds['lat'].values, flux_ds['lon'].values

def assign_labels(labels, valid_mask, shape):
    out = np.full(valid_mask.shape, np.nan)
    out[valid_mask] = labels
    return out.reshape(shape)

def get_description(scaled_num):
    descriptions = [
        "Very low MSE (cold/dry surface)",
        "Low MSE",
        "Moderate-low MSE",
        "Moderate MSE",
        "Moderate-high MSE",
        "High MSE",
        "Very high MSE (warm/moist surface)"
    ]
    idx = int((scaled_num / (N_CLUSTERS - 1)) * (len(descriptions) - 1))
    return descriptions[idx]

# ---------------------------
# NDVI helper functions
# ---------------------------

def load_year_data(year):
    folder = os.path.join(ndvi_data_dir, str(year))
    if not os.path.exists(folder):
        print(f"  NDVI folder {folder} does not exist")
        return None
    
    files = sorted(glob(os.path.join(folder, "*.nc")))
    print(f"  Loading NDVI year {year}: {len(files)} files")
    
    if len(files) == 0:
        return None
        
    datasets = []
    for i, f in enumerate(files):
        ds = xr.open_dataset(f)
        
        # Check coordinate order and apply appropriate slicing
        if i == 0:  # Debug info for first file only
            print(f"    Lat range: {ds.latitude.min().values:.2f} to {ds.latitude.max().values:.2f}")
            print(f"    Lat order: {'descending' if ds.latitude[0] > ds.latitude[-1] else 'ascending'}")
        
        # Handle coordinate ordering properly
        if ds.latitude[0] > ds.latitude[-1]:  # Descending coordinates (90 to -90)
            ndvi = ds['NDVI'].sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
        else:  # Ascending coordinates (-90 to 90)
            ndvi = ds['NDVI'].sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
        
        if i == 0:  # Check result of slicing
            print(f"    After slicing - shape: {ndvi.shape}, valid values: {np.isfinite(ndvi.values).sum()}")
        
        datasets.append(ndvi)
    
    year_data = xr.concat(datasets, dim='time')
    if not np.issubdtype(year_data['time'].dtype, np.datetime64):
        year_data['time'] = pd.to_datetime(year_data['time'].values)
    return year_data

def get_season_data(ndvi, months):
    season_data = ndvi.sel(time=ndvi.time.dt.month.isin(months))
    return season_data.mean(dim='time')

def cluster_ndvi(ndvi_2d, n_clusters=5):
    """Cluster NDVI data using the same approach as the working code"""
    flat_data = ndvi_2d.values.reshape(-1, 1)
    valid_mask = ~np.isnan(flat_data).flatten()
    valid_data = flat_data[valid_mask].reshape(-1, 1)
    
    if valid_data.shape[0] == 0:
        print("No valid NDVI data for clustering")
        return np.full(ndvi_2d.shape, np.nan)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(valid_data)
    labels = np.full(flat_data.shape[0], np.nan)
    labels[valid_mask] = kmeans.labels_
    return labels.reshape(ndvi_2d.shape)

def cluster_ndvi_temporal(season, lat_vals, lon_vals):
    """Cluster NDVI using temporal method similar to flux clustering but with working NDVI approach"""
    print(f"Clustering NDVI for {season}...")
    months = seasons[season]
    
    # Load all years data for this season using the working approach
    seasonal_data = []
    sample_lat = None
    sample_lon = None
    
    for year in years:
        try:
            ndvi_year = load_year_data(year)
            if ndvi_year is None:
                print(f"    Year {year}: No NDVI data available")
                seasonal_data.append(None)
                continue
                
            print(f"    Year {year}: NDVI data loaded successfully")
            
            # Get seasonal data using the working method
            if season == 'DJF':
                if year == years[0]:
                    season_sel = ndvi_year.sel(time=ndvi_year.time.dt.month.isin([1, 2]))
                else:
                    season_sel = ndvi_year.sel(time=ndvi_year.time.dt.month.isin([1, 2]))
            else:
                season_sel = ndvi_year.sel(time=ndvi_year.time.dt.month.isin(months))
            
            if season_sel.time.size > 0:
                season_mean = get_season_data(ndvi_year, months)
                print(f"    Season mean shape: {season_mean.shape}, valid values: {np.isfinite(season_mean.values).sum()}")
                
                # Check if we have valid spatial data
                if season_mean.size > 0 and np.isfinite(season_mean.values).sum() > 0:
                    # Cluster this year's seasonal data
                    cluster_map = cluster_ndvi(season_mean, n_clusters=N_CLUSTERS)
                    seasonal_data.append(cluster_map)
                    
                    # Store coordinate info from first valid dataset
                    if sample_lat is None:
                        sample_lat = season_mean.latitude.values
                        sample_lon = season_mean.longitude.values
                else:
                    print(f"    Year {year}: No valid seasonal data")
                    seasonal_data.append(None)
            else:
                seasonal_data.append(None)
        except Exception as e:
            print(f"Error processing NDVI for year {year}: {e}")
            seasonal_data.append(None)
    
    # Use provided coordinates if no NDVI data found
    if sample_lat is None:
        sample_lat = lat_vals
        sample_lon = lon_vals
        print(f"No NDVI data found for {season}, using default coordinates")
        return np.full((len(years), len(sample_lat), len(sample_lon)), np.nan)
    
    # Convert to numpy array with consistent shape
    nlat, nlon = len(sample_lat), len(sample_lon)
    full_labels = np.full((len(years), nlat, nlon), np.nan)
    
    for i, data in enumerate(seasonal_data):
        if data is not None and data.shape == (nlat, nlon):
            full_labels[i] = data
    
    print(f"NDVI clustering completed for {season}")
    return full_labels

# ---------------------------
# Load flux data
# ---------------------------

print("Loading and preparing flux data...")

# Load combined flux file
fluxes_ds = open_grib(os.path.join(data_dir, 'Fluxes.grib'))

# Extract latent and sensible heat fluxes from combined file
lhflx = subset_latlon(pick_var(fluxes_ds, ['slhf', 'lhflx', 'latent']), lat_min, lat_max, lon_min, lon_max)
shflx = subset_latlon(pick_var(fluxes_ds, ['sshf', 'shflx', 'sensible']), lat_min, lat_max, lon_min, lon_max)

lhflx = fix_flux_sign_if_needed(lhflx, "LHFLX")
shflx = fix_flux_sign_if_needed(shflx, "SHFLX")

# ---------------------------
# Load and process CAPE data
# ---------------------------

print("Loading and preparing CAPE data...")

try:
    cape_ds = open_grib(os.path.join(data_dir, 'CAPE.grib'))
    cape = subset_latlon(pick_var(cape_ds, ['cape', 'CAPE']), lat_min, lat_max, lon_min, lon_max)
    print("CAPE data loaded successfully")
except Exception as e:
    print(f"Error loading CAPE data: {e}")
    cape = None

def cluster_cape_time_series(cape_ds, season):
    """Cluster CAPE using the same methodology as flux clustering"""
    if cape_ds is None:
        print(f"No CAPE data available for {season}")
        # Return dummy data with same shape as other variables
        return np.full((len(years), len(lat_vals), len(lon_vals)), np.nan)
    
    nlat = len(cape_ds['lat'])
    nlon = len(cape_ds['lon'])
    nyears = len(years)

    # Build seasonal mean data array [year, lat, lon]
    data_array = np.full((nyears, nlat, nlon), np.nan)
    for i, y in enumerate(years):
        try:
            season_mean = seasonal_mean_field(cape_ds, y, season)
            
            # Check if season_mean has valid data and correct shape
            if season_mean is not None and hasattr(season_mean, 'values'):
                values = season_mean.values
                if values.shape == (nlat, nlon):
                    data_array[i] = values
                else:
                    print(f"Warning: CAPE shape mismatch for year {y}, season {season}: expected {(nlat, nlon)}, got {values.shape}")
                    continue
            else:
                print(f"Warning: No CAPE seasonal mean data for year {y}, season {season}")
                continue
                
        except Exception as e:
            print(f"Warning: Could not process CAPE for year {y}, season {season}: {e}")
            # Add more detailed error information for debugging
            import traceback
            print(f"  Detailed error: {traceback.format_exc()}")
            continue

    # Reshape for clustering
    data_2d = data_array.reshape(nyears * nlat * nlon, 1)
    valid = ~np.isnan(data_2d).flatten()
    data_valid = data_2d[valid]

    if data_valid.shape[0] == 0:
        print(f"No valid CAPE data for clustering in season {season}")
        return np.full((nyears, nlat, nlon), np.nan)

    # Fit KMeans cluster on all valid data points
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10)
    kmeans.fit(data_valid)

    # Assign labels back to original shape
    labels = np.full(data_2d.shape[0], np.nan)
    labels[valid] = kmeans.labels_
    labels_3d = labels.reshape(nyears, nlat, nlon)
    return labels_3d

# ---------------------------
# Load and process MSE data using time-varying clustering method
# ---------------------------

print("Processing MSE data for clustering...")

def compute_mse_ts(ds):
    """Return MSE (kJ/kg) DataArray with dims (time, latitude, longitude)."""
    Z = ds['z'] / g
    T = ds['t']
    q = ds['q']
    mse = (Cp * T + g * Z + Lv * q) / 1000.0  # kJ/kg
    return mse

def load_mse_time_series(levels):
    """
    Load per-level time series of MSE over the box, at 12 UTC only.
    Returns dict[level] = DataArray(time, lat, lon).
    """
    out = {}
    for level in levels:
        f = os.path.join(data_dir, f"{level}.grib")
        if not os.path.exists(f):
            print(f"Missing {f}, skipping.")
            continue
        try:
            ds = xr.open_dataset(f, engine="cfgrib")
            ds = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
            ds = ds.sel(time=ds['time'].dt.hour == 12)
            if ds.time.size == 0 or not {'z', 't', 'q'}.issubset(ds.data_vars):
                print(f"{level} hPa: no valid variables/time, skipping.")
                continue
            mse = compute_mse_ts(ds)
            out[level] = mse
        except Exception as e:
            print(f"Error reading {f}: {e}")
    return out

def build_climatology_for_clustering(mse_ts):
    """
    Build a single ALL-time mean per level to do the global clustering.
    Returns X_valid (n_pixels x n_levels_valid), valid_mask, valid_levels (sorted).
    """
    X_stack = []
    valid_levels = []
    for level in sorted(mse_ts.keys()):
        mse_mean = mse_ts[level].mean("time", skipna=True)
        X_stack.append(mse_mean.values.flatten())
        valid_levels.append(level)
    if not X_stack:
        raise RuntimeError("No valid levels found to cluster on.")
    X_all = np.array(X_stack).T
    valid_rows = ~np.any(np.isnan(X_all), axis=1)
    return X_all[valid_rows], valid_rows, valid_levels

def fit_kmeans_mse(X_valid, n_clusters, valid_levels):
    """Fit KMeans and reorder clusters by MSE at 1000 hPa ascending."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    kmeans.fit(X_valid)

    # Find index of 1000 hPa inside valid_levels (should exist)
    try:
        idx_1000 = valid_levels.index(1000)
    except ValueError:
        raise ValueError("1000 hPa not found in valid_levels – required to order by surface MSE.")

    cluster_1000 = []
    for c in range(n_clusters):
        mean_1000 = np.nanmean(X_valid[kmeans.labels_ == c, idx_1000])
        cluster_1000.append((c, mean_1000))
    cluster_1000.sort(key=lambda x: x[1])  # low -> high

    remap = {orig: new for new, (orig, _) in enumerate(cluster_1000)}
    # remap training labels
    kmeans.labels_ = np.array([remap[l] for l in kmeans.labels_])
    return kmeans, remap

def build_yearly_seasonal_means(mse_ts, months):
    """
    For a given set of months, return:
        { year -> { level : (lat, lon) seasonal-mean mse } }
    """
    years = None
    for level, da in mse_ts.items():
        mask = da['time'].dt.month.isin(months)
        if mask.sum() == 0:
            continue
        years = np.unique(da['time'].dt.year.where(mask, drop=True).values)
        break
    if years is None:
        return {}

    out = {int(y): {} for y in years}
    for level, da in mse_ts.items():
        da_season = da.where(da['time'].dt.month.isin(months), drop=True)
        if da_season.time.size == 0:
            continue
        for y in years:
            da_year = da_season.where(da_season['time'].dt.year == y, drop=True)
            if da_year.time.size == 0:
                continue
            out[int(y)][level] = da_year.mean("time", skipna=True)
    return out

def predict_cluster_map(kmeans, remap, X_levels, valid_mask, shape, cmap_levels_ordered):
    """
    Predict cluster labels for a stack of levels X_levels (dict[level] -> 2D DataArray),
    flatten & mask, and return 2D label map.
    """
    data_stack = [X_levels[lvl].values.flatten() for lvl in cmap_levels_ordered]
    X = np.array(data_stack).T
    labels = np.full(X.shape[0], np.nan)
    raw = kmeans.predict(X[valid_mask])
    labels[valid_mask] = [remap[c] for c in raw]
    return labels.reshape(shape)

# Load MSE time series
print("Loading MSE time series...")
mse_ts = load_mse_time_series(target_levels)

print("Building climatology for clustering...")
X_valid, valid_mask, valid_levels = build_climatology_for_clustering(mse_ts)

print("Fitting KMeans...")
kmeans_mse, remap = fit_kmeans_mse(X_valid, N_CLUSTERS, valid_levels)

print("Cluster ordering by MSE at 1000 hPa:")
for new_c in range(N_CLUSTERS):
    desc = get_description(new_c)
    print(f"  Cluster {new_c}: {desc}")

# Get coordinate values for later use
sample_level = next(iter(mse_ts))
lat_vals = mse_ts[sample_level]['latitude'].values
lon_vals = mse_ts[sample_level]['longitude'].values

# ---------------------------
# Cluster LHFLX and SHFLX independently but with same number of clusters
# ---------------------------

# Ensure lat_vals and lon_vals are defined
if lat_vals is None or lon_vals is None:
    # Get coordinates from flux data as fallback
    lat_vals = lhflx['lat'].values
    lon_vals = lhflx['lon'].values
    print("Using flux data coordinates for clustering")

def cluster_flux_time_series(flux_ds, season):
    nlat = len(flux_ds['lat'])
    nlon = len(flux_ds['lon'])
    nyears = len(years)

    # Build seasonal mean data array [year, lat, lon]
    data_array = np.full((nyears, nlat, nlon), np.nan)
    for i, y in enumerate(years):
        season_mean = seasonal_mean_field(flux_ds, y, season)
        data_array[i] = season_mean.values

    # Reshape for clustering
    data_2d = data_array.reshape(nyears * nlat * nlon, 1)
    valid = ~np.isnan(data_2d).flatten()
    data_valid = data_2d[valid]

    # Fit KMeans cluster on all valid data points
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10)
    kmeans.fit(data_valid)

    # Assign labels back to original shape
    labels = np.full(data_2d.shape[0], np.nan)
    labels[valid] = kmeans.labels_
    labels_3d = labels.reshape(nyears, nlat, nlon)
    return labels_3d

# Cluster for all seasons
seasonal_clusters = {}
for season in seasons.keys():
    print(f"Clustering LHFLX for {season}...")
    seasonal_clusters[f'lh_{season}'] = cluster_flux_time_series(lhflx, season)
    
    print(f"Clustering CAPE for {season}...")
    seasonal_clusters[f'cape_{season}'] = cluster_cape_time_series(cape, season)
    
    print(f"Clustering SHFLX for {season}...")
    seasonal_clusters[f'sh_{season}'] = cluster_flux_time_series(shflx, season)
    
    # Cluster NDVI for each season (pass coordinates)
    seasonal_clusters[f'ndvi_{season}'] = cluster_ndvi_temporal(season, lat_vals, lon_vals)

# Generate MSE cluster maps for each season using time-varying clustering method
seasonal_mse_clusters = {}
for season in seasons.keys():
    print(f"Processing MSE clustering for {season}...")
    months = seasons[season]
    
    # Build yearly seasonal means for this season
    by_year = build_yearly_seasonal_means(mse_ts, months)
    if not by_year:
        print(f"No MSE data for season {season}, skipping.")
        continue
    
    # Get sample shape from first year/level
    first_year = sorted(by_year.keys())[0]
    first_level = sorted(by_year[first_year].keys())[0]
    sample = by_year[first_year][first_level]
    
    # Build yearly cluster maps
    yearly_labels = []
    for year in years:
        if year in by_year:
            year_levels = by_year[year]
            # Only process if all clustering levels are present
            if all(lvl in year_levels for lvl in valid_levels):
                cluster_map = predict_cluster_map(
                    kmeans_mse, remap, year_levels, valid_mask,
                    shape=sample.shape,
                    cmap_levels_ordered=valid_levels
                )
                yearly_labels.append(cluster_map)
            else:
                # Missing levels - use NaN map
                yearly_labels.append(np.full(sample.shape, np.nan))
        else:
            # Missing year - use NaN map
            yearly_labels.append(np.full(sample.shape, np.nan))
    
    seasonal_mse_clusters[season] = np.array(yearly_labels)


# ---------------------------
# Animation: Single animation with all 4 seasons (5x4 grid)
# ---------------------------

def animate_all_seasons(seasonal_mse_clusters, seasonal_clusters, lat, lon):
    # Filter to only the 4 main seasons (exclude 'ALL')
    main_seasons = ['DJF', 'MAM', 'JJA', 'SON']
    
    fig = plt.figure(figsize=(25, 20))  # Increased width to accommodate 5 columns
    
    # Create a grid with explicit rows for plots and legends
    # 8 rows total: plot, legend, plot, legend, plot, legend, plot, legend
    # 5 columns total: MSE, CAPE, NDVI, LHFLX, SHFLX
    gs = fig.add_gridspec(8, 5, 
                         height_ratios=[4, 0.4, 4, 0.4, 4, 0.4, 4, 0.4],  # Smaller legend height
                         hspace=0.15, wspace=0.12,  # More space between rows
                         bottom=0.06, top=0.90, left=0.06, right=0.94)

    # Consistent color schemes for each variable
    mse_colors = ['#f2f0f7', '#cbc9e2', '#9e9ac8', '#756bb1', '#54278f']  # Purple
    mse_cmap = plt.matplotlib.colors.ListedColormap(mse_colors)
    
    # CAPE colors - Orange scheme as requested
    cape_colors = ['#feedde', '#fdd0a2', '#fdae6b', '#fd8d3c', '#d94701']  # Light to dark orange
    cape_cmap = plt.matplotlib.colors.ListedColormap(cape_colors)
    
    # NDVI colors - green scheme
    ndvi_colors = ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#41ab5d']
    ndvi_cmap = plt.matplotlib.colors.ListedColormap(ndvi_colors)
    
    # Consistent LHFLX colors (reds)
    lh_colors = ['#fee5d9', '#fcbba1', '#fc9272', '#fb6a4a', '#cb181d']
    lh_cmap = plt.matplotlib.colors.ListedColormap(lh_colors)
    
    # Consistent SHFLX colors (blues)
    sh_colors = ['#deebf7', '#9ecae1', '#6baed6', '#3182bd', '#08519c']
    sh_cmap = plt.matplotlib.colors.ListedColormap(sh_colors)

    # Create subplot axes for each season and variable (only on plot rows: 0, 2, 4, 6)
    axes = {}
    plot_rows = [0, 2, 4, 6]  # Rows dedicated to plots
    for i, season in enumerate(main_seasons):
        row = plot_rows[i]
        axes[season] = {}
        # MSE, CAPE, NDVI, LHFLX, SHFLX
        axes[season]['mse'] = fig.add_subplot(gs[row, 0], projection=ccrs.PlateCarree())
        axes[season]['cape'] = fig.add_subplot(gs[row, 1], projection=ccrs.PlateCarree())
        axes[season]['ndvi'] = fig.add_subplot(gs[row, 2], projection=ccrs.PlateCarree())
        axes[season]['lhflx'] = fig.add_subplot(gs[row, 3], projection=ccrs.PlateCarree())
        axes[season]['shflx'] = fig.add_subplot(gs[row, 4], projection=ccrs.PlateCarree())

    # Column headers removed as requested

    # Add row headers for seasons (adjusted for better spacing)
    season_y_positions = [0.80, 0.60, 0.40, 0.20]  # More evenly distributed
    for i, season in enumerate(main_seasons):
        fig.text(0.02, season_y_positions[i], season, fontsize=16, fontweight='bold', 
                ha='center', va='center', rotation=90)

    # Create meshgrid for plotting using the coordinate arrays from the data
    # Ensure we use the coordinate arrays that match our actual data
    print(f"Using coordinates - lat: {len(lat)} points, lon: {len(lon)} points")
    lon2d, lat2d = np.meshgrid(lon, lat)

    def draw_map(ax, data, title, cmap, season, var_name):
        ax.clear()
        ax.set_extent([lon_min, lon_max, lat_min, lat_max])
        ax.coastlines(resolution='50m', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        
        # Handle NDVI data separately due to different grid resolution
        if var_name == "NDVI":
            # For NDVI, use its own coordinate system
            if data.shape[0] > 0 and data.shape[1] > 0 and np.isfinite(data).sum() > 0:
                mesh = ax.imshow(data, origin='upper', cmap=cmap, 
                               extent=[lon_min, lon_max, lat_min, lat_max],
                               vmin=0, vmax=N_CLUSTERS-1, transform=ccrs.PlateCarree())
            else:
                # Create a dummy mesh for NDVI with no data
                dummy_data = np.full((100, 100), np.nan)  # Use standard dummy size
                mesh = ax.imshow(dummy_data, origin='upper', cmap=cmap,
                               extent=[lon_min, lon_max, lat_min, lat_max],
                               vmin=0, vmax=N_CLUSTERS-1, transform=ccrs.PlateCarree())
                ax.text(0.5, 0.5, 'No NDVI Data', transform=ax.transAxes, ha='center', va='center', 
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            # For MSE, CAPE, LHFLX, SHFLX - use the shared coordinate grid
            if data.shape == lon2d.shape and np.isfinite(data).sum() > 0:
                mesh = ax.imshow(data, origin='upper', cmap=cmap, 
                               extent=[lon2d.min(), lon2d.max(), lat2d.min(), lat2d.max()],
                               vmin=0, vmax=N_CLUSTERS-1, transform=ccrs.PlateCarree())
            else:
                # Handle mismatched dimensions or no data
                if data.shape != lon2d.shape:
                    print(f"Warning: Data shape {data.shape} doesn't match coordinate grid {lon2d.shape} for {var_name}")
                dummy_data = np.full(lon2d.shape, np.nan)
                mesh = ax.imshow(dummy_data, origin='upper', cmap=cmap,
                               extent=[lon2d.min(), lon2d.max(), lat2d.min(), lat2d.max()],
                               vmin=0, vmax=N_CLUSTERS-1, transform=ccrs.PlateCarree())
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center', 
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        # Remove lat/lon labels to save space
        ax.set_xticks([])
        ax.set_yticks([])
        return mesh

    def update(frame):
        year = years[frame]
        
        # Add overall title at the top of the figure
        fig.suptitle(f"Multi-Variable Clustering Analysis - {year}", fontsize=18, fontweight='bold', y=0.94)
        
        # Clear previous images
        images = {}
        
        for season in main_seasons:
            # Draw maps for each variable in this season - with safety checks
            try:
                mse_data = seasonal_mse_clusters[season][frame]
                if mse_data.size == 0:
                    mse_data = np.full(lon2d.shape, np.nan)
            except (KeyError, IndexError):
                mse_data = np.full(lon2d.shape, np.nan)
                
            try:
                cape_data = seasonal_clusters[f'cape_{season}'][frame]
                if cape_data.size == 0:
                    cape_data = np.full(lon2d.shape, np.nan)
            except (KeyError, IndexError):
                cape_data = np.full(lon2d.shape, np.nan)
                
            try:
                ndvi_data = seasonal_clusters[f'ndvi_{season}'][frame]
                if ndvi_data.size == 0:
                    ndvi_data = np.full(lon2d.shape, np.nan)
            except (KeyError, IndexError):
                ndvi_data = np.full(lon2d.shape, np.nan)
                
            try:
                lhflx_data = seasonal_clusters[f'lh_{season}'][frame]
                if lhflx_data.size == 0:
                    lhflx_data = np.full(lon2d.shape, np.nan)
            except (KeyError, IndexError):
                lhflx_data = np.full(lon2d.shape, np.nan)
                
            try:
                shflx_data = seasonal_clusters[f'sh_{season}'][frame]
                if shflx_data.size == 0:
                    shflx_data = np.full(lon2d.shape, np.nan)
            except (KeyError, IndexError):
                shflx_data = np.full(lon2d.shape, np.nan)
            
            images[f'{season}_mse'] = draw_map(
                axes[season]['mse'], 
                mse_data, 
                f"MSE Profile Clusters", 
                mse_cmap, season, "MSE"
            )
            
            images[f'{season}_cape'] = draw_map(
                axes[season]['cape'], 
                cape_data, 
                f"CAPE Clusters", 
                cape_cmap, season, "CAPE"
            )
            
            images[f'{season}_ndvi'] = draw_map(
                axes[season]['ndvi'], 
                ndvi_data, 
                f"NDVI Clusters", 
                ndvi_cmap, season, "NDVI"
            )
            
            images[f'{season}_lhflx'] = draw_map(
                axes[season]['lhflx'], 
                lhflx_data, 
                f"LHFLX Clusters", 
                lh_cmap, season, "LHFLX"
            )
            
            images[f'{season}_shflx'] = draw_map(
                axes[season]['shflx'], 
                shflx_data, 
                f"SHFLX Clusters", 
                sh_cmap, season, "SHFLX"
            )

        # Remove any previous colorbars (start from axis 20 since we have 20 plot axes)
        while len(fig.axes) > 20:
            fig.delaxes(fig.axes[-1])

        # Add colorbars in the dedicated legend rows (1, 3, 5, 7)
        legend_rows = [1, 3, 5, 7]
        
        for i, season in enumerate(main_seasons):
            legend_row = legend_rows[i]
            
            # Create colorbar axes in the legend row, positioned under each plot
            cbar_mse_ax = fig.add_subplot(gs[legend_row, 0])
            cbar_cape_ax = fig.add_subplot(gs[legend_row, 1])
            cbar_ndvi_ax = fig.add_subplot(gs[legend_row, 2])
            cbar_lh_ax = fig.add_subplot(gs[legend_row, 3])
            cbar_sh_ax = fig.add_subplot(gs[legend_row, 4])
            
            # Clear the axes and use them for colorbars
            cbar_mse_ax.clear()
            cbar_cape_ax.clear()
            cbar_ndvi_ax.clear()
            cbar_lh_ax.clear()
            cbar_sh_ax.clear()
            
            # Turn off axis frames
            for ax in [cbar_mse_ax, cbar_cape_ax, cbar_ndvi_ax, cbar_lh_ax, cbar_sh_ax]:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
            
            # Create colorbars positioned directly below plots with no borders
            cbar_mse = fig.colorbar(images[f'{season}_mse'], ax=cbar_mse_ax, orientation='horizontal', 
                                   ticks=[0, N_CLUSTERS-1])
            cbar_mse.ax.set_xticklabels(['Low', 'High'], fontsize=12)
            cbar_mse.ax.tick_params(labelsize=12)
            cbar_mse.outline.set_visible(False)  # Remove colorbar border
            
            cbar_cape = fig.colorbar(images[f'{season}_cape'], ax=cbar_cape_ax, orientation='horizontal', 
                                    ticks=[0, N_CLUSTERS-1])
            cbar_cape.ax.set_xticklabels(['Low', 'High'], fontsize=12)
            cbar_cape.ax.tick_params(labelsize=12)
            cbar_cape.outline.set_visible(False)  # Remove colorbar border
            
            cbar_ndvi = fig.colorbar(images[f'{season}_ndvi'], ax=cbar_ndvi_ax, orientation='horizontal', 
                                    ticks=[0, N_CLUSTERS-1])
            cbar_ndvi.ax.set_xticklabels(['Low', 'High'], fontsize=12)
            cbar_ndvi.ax.tick_params(labelsize=12)
            cbar_ndvi.outline.set_visible(False)  # Remove colorbar border
            
            cbar_lh = fig.colorbar(images[f'{season}_lhflx'], ax=cbar_lh_ax, orientation='horizontal', 
                                  ticks=[0, N_CLUSTERS-1])
            cbar_lh.ax.set_xticklabels(['Low', 'High'], fontsize=12)
            cbar_lh.ax.tick_params(labelsize=12)
            cbar_lh.outline.set_visible(False)  # Remove colorbar border
            
            cbar_sh = fig.colorbar(images[f'{season}_shflx'], ax=cbar_sh_ax, orientation='horizontal', 
                                  ticks=[0, N_CLUSTERS-1])
            cbar_sh.ax.set_xticklabels(['Low', 'High'], fontsize=12)
            cbar_sh.ax.tick_params(labelsize=12)
            cbar_sh.outline.set_visible(False)  # Remove colorbar border

        return list(images.values())

    ani = animation.FuncAnimation(fig, update, frames=len(years), blit=False)
    save_path = os.path.join(output_dir, f"all_seasons_5x4_clusters_animation.mp4")
    print(f"Saving combined animation to {save_path} ...")
    ani.save(save_path, writer='ffmpeg', fps=1, dpi=120)  # 1 fps = 1 second per frame
    plt.close(fig)
    print("Combined animation saved.")

# Create separate 5-panel animations for individual seasons
def animate_five_panel(mse_labels, cape_labels, ndvi_labels, lh_labels, sh_labels, lat, lon, season):
    fig = plt.figure(figsize=(20, 12))  # Made wider to accommodate 5 variables
    
    # Create grid with explicit rows for plots and legends
    # 4 rows: plot row 1, legend row 1, plot row 2, legend row 2
    gs = fig.add_gridspec(4, 5, 
                         height_ratios=[4, 0.5, 4, 0.5],  # Smaller legend height
                         hspace=0.20, wspace=0.15,  # More space between rows
                         bottom=0.12, top=0.88)

    # Create plot axes in rows 0 and 2
    ax_mse = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_cape = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax_ndvi = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
    ax_lh = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
    ax_sh = fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree())

    # Consistent color schemes (same as combined animation)
    mse_colors = ['#f2f0f7', '#cbc9e2', '#9e9ac8', '#756bb1', '#54278f']  # Purple
    mse_cmap = plt.matplotlib.colors.ListedColormap(mse_colors)
    
    # CAPE colors - Orange scheme
    cape_colors = ['#feedde', '#fdd0a2', '#fdae6b', '#fd8d3c', '#d94701']
    cape_cmap = plt.matplotlib.colors.ListedColormap(cape_colors)
    
    # NDVI colors - green scheme
    ndvi_colors = ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#41ab5d']
    ndvi_cmap = plt.matplotlib.colors.ListedColormap(ndvi_colors)
    
    # Consistent LHFLX colors (reds)
    lh_colors = ['#fee5d9', '#fcbba1', '#fc9272', '#fb6a4a', '#cb181d']
    lh_cmap = plt.matplotlib.colors.ListedColormap(lh_colors)
    
    # Consistent SHFLX colors (blues)
    sh_colors = ['#deebf7', '#9ecae1', '#6baed6', '#3182bd', '#08519c']
    sh_cmap = plt.matplotlib.colors.ListedColormap(sh_colors)

    print(f"Using coordinates - lat: {len(lat)} points, lon: {len(lon)} points")
    lon2d, lat2d = np.meshgrid(lon, lat)

    def draw_map(ax, data, title, cmap, var_name):
        ax.clear()
        ax.set_extent([lon_min, lon_max, lat_min, lat_max])
        ax.coastlines(resolution='50m', linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        
        # Handle NDVI data separately due to different grid resolution
        if var_name == "NDVI":
            # For NDVI, use its own coordinate system
            if data.shape[0] > 0 and data.shape[1] > 0 and np.isfinite(data).sum() > 0:
                mesh = ax.imshow(data, origin='upper', cmap=cmap,
                               extent=[lon_min, lon_max, lat_min, lat_max],
                               vmin=0, vmax=N_CLUSTERS-1, transform=ccrs.PlateCarree())
            else:
                # Create a dummy mesh for NDVI with no data
                dummy_data = np.full((100, 100), np.nan)  # Use standard dummy size
                mesh = ax.imshow(dummy_data, origin='upper', cmap=cmap,
                               extent=[lon_min, lon_max, lat_min, lat_max],
                               vmin=0, vmax=N_CLUSTERS-1, transform=ccrs.PlateCarree())
                ax.text(0.5, 0.5, 'No NDVI Data', transform=ax.transAxes, ha='center', va='center', 
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            # For MSE, LHFLX, SHFLX - use the shared coordinate grid
            if data.shape == lon2d.shape and np.isfinite(data).sum() > 0:
                mesh = ax.imshow(data, origin='upper', cmap=cmap,
                               extent=[lon2d.min(), lon2d.max(), lat2d.min(), lat2d.max()],
                               vmin=0, vmax=N_CLUSTERS-1, transform=ccrs.PlateCarree())
            else:
                # Handle mismatched dimensions or no data
                if data.shape != lon2d.shape:
                    print(f"Warning: Data shape {data.shape} doesn't match coordinate grid {lon2d.shape} for {var_name}")
                dummy_data = np.full(lon2d.shape, np.nan)
                mesh = ax.imshow(dummy_data, origin='upper', cmap=cmap,
                               extent=[lon2d.min(), lon2d.max(), lat2d.min(), lat2d.max()],
                               vmin=0, vmax=N_CLUSTERS-1, transform=ccrs.PlateCarree())
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center', 
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(title, fontweight='bold')
        return mesh

    def update(frame):
        year = years[frame]
        
        # Add overall season title at the top of the figure
        fig.suptitle(f"{season} Season Clustering Analysis - {year}", fontsize=16, fontweight='bold', y=0.92)
        
        im_mse = draw_map(ax_mse, mse_labels[frame], f"MSE Profile Clusters", mse_cmap, "MSE")
        im_cape = draw_map(ax_cape, cape_labels[frame], f"CAPE Clusters", cape_cmap, "CAPE")
        im_ndvi = draw_map(ax_ndvi, ndvi_labels[frame], f"NDVI Clusters", ndvi_cmap, "NDVI")
        im_lh = draw_map(ax_lh, lh_labels[frame], f"LHFLX Clusters", lh_cmap, "LHFLX")
        im_sh = draw_map(ax_sh, sh_labels[frame], f"SHFLX Clusters", sh_cmap, "SHFLX")

        # Remove previous colorbars (keep only the 5 plot axes)
        while len(fig.axes) > 5:
            fig.delaxes(fig.axes[-1])

        # Create colorbar axes in the dedicated legend rows (1 and 3)
        cbar_mse_ax = fig.add_subplot(gs[1, 0])  # Below MSE plot
        cbar_cape_ax = fig.add_subplot(gs[1, 1])  # Below CAPE plot  
        cbar_ndvi_ax = fig.add_subplot(gs[1, 2])  # Below NDVI plot
        cbar_lh_ax = fig.add_subplot(gs[3, 0])   # Below LHFLX plot
        cbar_sh_ax = fig.add_subplot(gs[3, 1])   # Below SHFLX plot
        
        # Clear the axes and use them for colorbars
        for ax in [cbar_mse_ax, cbar_cape_ax, cbar_ndvi_ax, cbar_lh_ax, cbar_sh_ax]:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        # Create colorbars positioned directly below plots with no borders
        cbar_mse = fig.colorbar(im_mse, ax=cbar_mse_ax, orientation='horizontal', 
                               ticks=[0, N_CLUSTERS-1], shrink=0.8, pad=0.1)
        cbar_mse.ax.set_xticklabels(['Low Surface MSE\n(Cold/Dry)', 'High Surface MSE\n(Warm/Moist)'], fontsize=12)
        cbar_mse.outline.set_visible(False)  # Remove colorbar border
        
        cbar_cape = fig.colorbar(im_cape, ax=cbar_cape_ax, orientation='horizontal', 
                                ticks=[0, N_CLUSTERS-1], shrink=0.8, pad=0.1)
        cbar_cape.ax.set_xticklabels(['Low CAPE\n(Stable)', 'High CAPE\n(Unstable)'], fontsize=12)
        cbar_cape.outline.set_visible(False)  # Remove colorbar border
        
        cbar_ndvi = fig.colorbar(im_ndvi, ax=cbar_ndvi_ax, orientation='horizontal', 
                                ticks=[0, N_CLUSTERS-1], shrink=0.8, pad=0.1)
        cbar_ndvi.ax.set_xticklabels(['Low NDVI\n(Sparse vegetation)', 'High NDVI\n(Dense vegetation)'], fontsize=12)
        cbar_ndvi.outline.set_visible(False)  # Remove colorbar border
        
        cbar_lh = fig.colorbar(im_lh, ax=cbar_lh_ax, orientation='horizontal', 
                              ticks=[0, N_CLUSTERS-1], shrink=0.8, pad=0.1)
        cbar_lh.ax.set_xticklabels(['Low', 'High'], fontsize=12)
        cbar_lh.outline.set_visible(False)  # Remove colorbar border
        
        cbar_sh = fig.colorbar(im_sh, ax=cbar_sh_ax, orientation='horizontal', 
                              ticks=[0, N_CLUSTERS-1], shrink=0.8, pad=0.1)
        cbar_sh.ax.set_xticklabels(['Low', 'High'], fontsize=12)
        cbar_sh.outline.set_visible(False)  # Remove colorbar border

        return im_mse, im_cape, im_ndvi, im_lh, im_sh
        cbar_mse_ax = fig.add_subplot(gs[1, 0])  # Below MSE plot
        cbar_ndvi_ax = fig.add_subplot(gs[1, 1])  # Below NDVI plot
        cbar_lh_ax = fig.add_subplot(gs[3, 0])   # Below LHFLX plot
        cbar_sh_ax = fig.add_subplot(gs[3, 1])   # Below SHFLX plot
        
        # Clear the axes and use them for colorbars
        for ax in [cbar_mse_ax, cbar_ndvi_ax, cbar_lh_ax, cbar_sh_ax]:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        # Create colorbars positioned directly below plots with no borders
        cbar_mse = fig.colorbar(im_mse, ax=cbar_mse_ax, orientation='horizontal', 
                               ticks=[0, N_CLUSTERS-1], shrink=0.8, pad=0.1)
        cbar_mse.ax.set_xticklabels(['Low Surface MSE\n(Cold/Dry)', 'High Surface MSE\n(Warm/Moist)'], fontsize=12)
        cbar_mse.outline.set_visible(False)  # Remove colorbar border
        
        cbar_ndvi = fig.colorbar(im_ndvi, ax=cbar_ndvi_ax, orientation='horizontal', 
                                ticks=[0, N_CLUSTERS-1], shrink=0.8, pad=0.1)
        cbar_ndvi.ax.set_xticklabels(['Low NDVI\n(Sparse vegetation)', 'High NDVI\n(Dense vegetation)'], fontsize=12)
        cbar_ndvi.outline.set_visible(False)  # Remove colorbar border
        
        cbar_lh = fig.colorbar(im_lh, ax=cbar_lh_ax, orientation='horizontal', 
                              ticks=[0, N_CLUSTERS-1], shrink=0.8, pad=0.1)
        cbar_lh.ax.set_xticklabels(['Low', 'High'], fontsize=12)
        cbar_lh.outline.set_visible(False)  # Remove colorbar border
        
        cbar_sh = fig.colorbar(im_sh, ax=cbar_sh_ax, orientation='horizontal', 
                              ticks=[0, N_CLUSTERS-1], shrink=0.8, pad=0.1)
        cbar_sh.ax.set_xticklabels(['Low', 'High'], fontsize=12)
        cbar_sh.outline.set_visible(False)  # Remove colorbar border

        return im_mse, im_ndvi, im_lh, im_sh

    ani = animation.FuncAnimation(fig, update, frames=len(years), blit=False)
    save_path = os.path.join(output_dir, f"{season}_5panel_clusters_animation.mp4")
    print(f"Saving animation to {save_path} ...")
    ani.save(save_path, writer='ffmpeg', fps=1, dpi=200)
    plt.close(fig)
    print("Animation saved.")

# ---------------------------
# Run animations
# ---------------------------

# Create the combined 5x4 animation showing all 4 seasons together
print("Creating combined 5x4 animation with all seasons...")
animate_all_seasons(seasonal_mse_clusters, seasonal_clusters, lat_vals, lon_vals)

# Create separate animations for each individual season
main_seasons = ['DJF', 'MAM', 'JJA', 'SON']
for season in main_seasons:
    print(f"Creating individual animation for {season}...")
    animate_five_panel(
        seasonal_mse_clusters[season], 
        seasonal_clusters[f'cape_{season}'], 
        seasonal_clusters[f'ndvi_{season}'], 
        seasonal_clusters[f'lh_{season}'], 
        seasonal_clusters[f'sh_{season}'], 
        lat_vals, lon_vals, season
    )

print("All done.")
