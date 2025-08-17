import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.cluster import KMeans

# For animations
import imageio
try:
    import imageio.v2 as imageio_v2  # newer API
    imsave = imageio_v2.mimsave
    ioread = imageio_v2.imread
except ImportError:
    imsave = imageio.mimsave
    ioread = imageio.imread

try:
    import imageio_ffmpeg  # optional, for mp4 via imageio
    HAS_FFMPEG = True
except ImportError:
    HAS_FFMPEG = False

# -------------------------
# Constants
# -------------------------
Cp = 1004.0
g = 9.81
Lv = 2.5e6

# Region bounds
lat_min, lat_max = -10, 5
lon_min, lon_max = 10, 30

# Config
data_dir   = "./"
output_dir = "mse_cluster_animations"
os.makedirs(output_dir, exist_ok=True)

# Fixed as requested
n_clusters    = 5
target_levels = list(range(600, 1001, 50))  # 600–1000 by 50

# Seasons to animate (ALL five seasons)
seasons_to_animate = {
    "ALL": list(range(1, 13)),
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

# -------------------------
# Helpers
# -------------------------
def compute_mse(ds):
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
            mse = compute_mse(ds)
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

def fit_kmeans(X_valid, n_clusters, valid_levels):
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

def make_season_animation(season_name, months, mse_ts, kmeans, remap, valid_mask, valid_levels):
    """
    Build a GIF (and MP4 if possible) showing one frame per year of the season.
    Each frame is titled with '<SEASON> Cluster Map – <YEAR> (n=11)'.
    """
    by_year = build_yearly_seasonal_means(mse_ts, months)
    if not by_year:
        print(f"No data for season {season_name}, skipping animation.")
        return

    # Grab a sample (first year, first level) to get lat/lon & shape
    first_year = sorted(by_year.keys())[0]
    first_level = sorted(by_year[first_year].keys())[0]
    sample = by_year[first_year][first_level]
    lat = sample['latitude'].values
    lon = sample['longitude'].values
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Build frames
    frames = []
    cmap = plt.get_cmap('tab20', kmeans.n_clusters)

    for year in sorted(by_year.keys()):
        year_levels = by_year[year]
        # Only keep frames with all clustering levels present
        if not all(lvl in year_levels for lvl in valid_levels):
            print(f"{season_name} {year}: missing some levels, skipping this frame.")
            continue

        cluster_map = predict_cluster_map(
            kmeans, remap, year_levels, valid_mask,
            shape=sample.shape,
            cmap_levels_ordered=valid_levels
        )

        # plot frame
        fig = plt.figure(figsize=(7, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        im = ax.pcolormesh(lon2d, lat2d, cluster_map, cmap=cmap, shading='auto')
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.set_title(f"{season_name} Cluster Map – {year}  (n={kmeans.n_clusters})", fontsize=12, fontweight='bold')

        # 2-tick colorbar (low/high surface MSE)
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.07, fraction=0.05)
        cbar.set_ticks([0, kmeans.n_clusters - 1])
        cbar.ax.set_xticklabels(["Low Surface MSE (dry)", "High Surface MSE (humid)"])

        # save temp frame
        tmp_path = os.path.join(output_dir, f"__tmp_{season_name}_{year}.png")
        fig.savefig(tmp_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        frames.append(tmp_path)

    if not frames:
        print(f"No frames were produced for {season_name}.")
        return

    # Write GIF
    gif_path = os.path.join(output_dir, f"anim_{season_name}_600-1000_n{n_clusters}.gif")
    imgs = [ioread(f) for f in frames]
    imsave(gif_path, imgs, fps=2)
    print(f"Saved GIF: {gif_path}")

    # Write MP4 if possible
    if HAS_FFMPEG:
        mp4_path = os.path.join(output_dir, f"anim_{season_name}_600-1000_n{n_clusters}.mp4")
        imageio.mimsave(mp4_path, imgs, fps=2, codec='libx264')
        print(f"Saved MP4: {mp4_path}")

    # cleanup
    for f in frames:
        try:
            os.remove(f)
        except OSError:
            pass

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    print("Loading MSE time series ...")
    mse_ts = load_mse_time_series(target_levels)

    print("Building climatology for clustering ...")
    X_valid, valid_mask, valid_levels = build_climatology_for_clustering(mse_ts)

    print("Fitting KMeans ...")
    kmeans, remap = fit_kmeans(X_valid, n_clusters, valid_levels)
    print("Done. Making animations for ALL, DJF, MAM, JJA, SON ...")

    for season_name, months in seasons_to_animate.items():
        make_season_animation(season_name, months, mse_ts, kmeans, remap, valid_mask, valid_levels)

    print("✅ Finished.")
