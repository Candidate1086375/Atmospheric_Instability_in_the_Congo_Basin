import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from scipy import ndimage

# Set font
mpl.rcParams['font.family'] = 'Times New Roman'

def create_fast_elevation_map_3d():
    """Create 3D elevation map using ETOPO1 global data - much faster than tile merging"""
    
    print("=" * 60)
    print("FAST AFRICAN 3D ELEVATION MAP GENERATOR")
    print("Using ETOPO1 Global Relief Data (1 arc-minute resolution)")
    print("=" * 60)
    
    # ETOPO1 global relief data URL
    url = "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO1_Ice_g_gmt4.nc"
    
    # Define our region of interest - extended buffer for better coverage
    # Focus area: Central Africa (7°E to 29°E, -15°N to 10°N)
    # Extended buffer: 2°E to 34°E, -25°N to 15°N (larger southern coverage)
    focus_lon_min, focus_lon_max = 7, 29   # Central Africa longitude
    focus_lat_min, focus_lat_max = -15, 10 # Central Africa latitude
    
    buffer = 5
    west_total = focus_lon_min - buffer  # 2°E
    east_total = focus_lon_max + buffer  # 34°E
    south_total = focus_lat_min - 15     # -30°N (extended further south)
    north_total = focus_lat_max + buffer  # 15°N    print(f"\n1. DOWNLOADING ETOPO1 DATA FOR REGION...")
    print(f"   Region: {west_total}°E to {east_total}°E, {south_total}°N to {north_total}°N")
    
    try:
        # Open the dataset directly from THREDDS server
        print("   Accessing ETOPO1 dataset...")
        ds = xr.open_dataset(url)
        
        # Subset to our region
        print("   Subsetting to Africa region...")
        elevation_data = ds.sel(
            lon=slice(west_total, east_total),
            lat=slice(south_total, north_total)
        )
        
        print("   ✓ Data downloaded and subset successfully")
        
    except Exception as e:
        print(f"   ✗ Error accessing ETOPO1 data: {e}")
        print("   This might be due to internet connectivity issues.")
        return False
    
    print(f"\n2. PROCESSING ELEVATION DATA...")
    
    # Get the elevation data
    elev_values = elevation_data['z'].values
    lons = elevation_data['lon'].values  
    lats = elevation_data['lat'].values
    
    print(f"   Data shape: {elev_values.shape}")
    print(f"   Longitude range: {lons.min():.1f}° to {lons.max():.1f}°")
    print(f"   Latitude range: {lats.min():.1f}° to {lats.max():.1f}°")
    print(f"   Elevation range: {np.nanmin(elev_values):.1f}m to {np.nanmax(elev_values):.1f}m")
    
    # Subsample data moderately and apply smoothing
    subsample = 2  # Moderate subsampling for performance but smoother than squares
    elev_sub = elev_values[::subsample, ::subsample]
    lons_sub = lons[::subsample]
    lats_sub = lats[::subsample]
    
    # Apply Gaussian smoothing to remove square appearance
    from scipy import ndimage
    elev_sub = ndimage.gaussian_filter(elev_sub, sigma=1.0)
    
    # Force straight edges at all plot boundaries for clean appearance
    # Set all edges to have consistent elevations for straight boundaries
    
    # Bottom edge (southern boundary) - set to cut-off elevation
    bottom_row_avg = np.mean(elev_sub[-1, :])
    elev_sub[-1, :] = bottom_row_avg
    
    # Top edge (northern boundary)
    top_row_avg = np.mean(elev_sub[0, :])
    elev_sub[0, :] = top_row_avg
    
    # Left edge (western boundary)
    left_col_avg = np.mean(elev_sub[:, 0])
    elev_sub[:, 0] = left_col_avg
    
    # Right edge (eastern boundary)
    right_col_avg = np.mean(elev_sub[:, -1])
    elev_sub[:, -1] = right_col_avg
    
    # Smooth transitions from edges inward
    for i in range(1, 4):  # Smooth 3 pixels from each edge
        if len(elev_sub) - 1 - i >= 0:
            blend_factor = i / 4.0
            # Bottom blend
            elev_sub[-1-i, :] = (blend_factor * elev_sub[-1-i, :] + 
                                (1-blend_factor) * bottom_row_avg)
            # Top blend
            if i < len(elev_sub):
                elev_sub[i, :] = (blend_factor * elev_sub[i, :] + 
                                 (1-blend_factor) * top_row_avg)
        
        if elev_sub.shape[1] - 1 - i >= 0:
            # Left blend
            elev_sub[:, i] = (blend_factor * elev_sub[:, i] + 
                             (1-blend_factor) * left_col_avg)
            # Right blend
            if i < elev_sub.shape[1]:
                elev_sub[:, -1-i] = (blend_factor * elev_sub[:, -1-i] + 
                                    (1-blend_factor) * right_col_avg)
    
    # Use Cartopy Natural Earth features to get accurate water bodies
    print("   Loading water bodies from Natural Earth dataset...")
    
    # Get Natural Earth water features
    rivers_50m = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '50m')
    lakes_50m = cfeature.NaturalEarthFeature('physical', 'lakes', '50m')
    
    # Create a temporary 2D plot to extract water feature geometries
    temp_fig = plt.figure(figsize=(1, 1))
    temp_ax = temp_fig.add_subplot(111, projection=ccrs.PlateCarree())
    temp_ax.set_extent([west_total, east_total, south_total, north_total], crs=ccrs.PlateCarree())
    
    # Add features to extract their geometries
    temp_ax.add_feature(rivers_50m, facecolor='none', edgecolor='blue')
    temp_ax.add_feature(lakes_50m, facecolor='blue', edgecolor='blue')
    
    # Extract river and lake coordinates from Natural Earth features
    water_features_2d = []
    
    try:
        # Get river geometries
        for geometry in rivers_50m.geometries():
            if hasattr(geometry, 'coords'):
                coords = list(geometry.coords)
                if len(coords) > 1:
                    lons = [coord[0] for coord in coords]
                    lats = [coord[1] for coord in coords]
                    # Filter to our region
                    if (any(west_total <= lon <= east_total for lon in lons) and 
                        any(south_total <= lat <= north_total for lat in lats)):
                        water_features_2d.append({
                            'type': 'river',
                            'lons': lons,
                            'lats': lats
                        })
        
        # Get lake geometries
        for geometry in lakes_50m.geometries():
            if hasattr(geometry, 'exterior'):
                coords = list(geometry.exterior.coords)
                lons = [coord[0] for coord in coords]
                lats = [coord[1] for coord in coords]
                # Filter to our region
                if (any(west_total <= lon <= east_total for lon in lons) and 
                    any(south_total <= lat <= north_total for lat in lats)):
                    water_features_2d.append({
                        'type': 'lake',
                        'lons': lons,
                        'lats': lats
                    })
    except Exception as e:
        print(f"   Warning: Could not extract all water features: {e}")
        # Fallback to major lakes only
        water_features_2d = []
    
    plt.close(temp_fig)  # Clean up temporary figure
    
    print(f"   Subsampled shape: {elev_sub.shape} (for better 3D performance)")
    print(f"   Loaded {len(water_features_2d)} water features from Natural Earth dataset")
    
    # Create coordinate meshgrid
    lon2d, lat2d = np.meshgrid(lons_sub, lats_sub)
    
    # Handle water bodies and create color mapping
    water_mask = elev_sub < 0
    print(f"   Water pixels: {np.sum(water_mask)} out of {elev_sub.size} total pixels")
    
    print(f"\n3. CREATING 3D ELEVATION MAP...")
    
    min_elev = np.nanmin(elev_sub)
    max_elev = np.nanmax(elev_sub)
    
    # Create the 3D plot with clean formatting
    fig = plt.figure(figsize=(18, 12), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    # Clean background
    fig.patch.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    # Remove z-axis (elevation axis)
    ax.zaxis.set_visible(False)
    
    # Create custom colormap for dramatic terrain visualization like the US map
    if min_elev < 0:
        print("   Creating dramatic terrain color scheme...")
        
        # Create simplified ocean to highland color scheme with consistent blue
        water_colors = [
            '#004499',  # Consistent ocean blue for all water
            '#004499',  # Same blue
            '#004499',  # Same blue
            '#004499'   # Same blue
        ]
        
        # Land colors from lowlands to peaks - dramatic terrain colors
        land_colors = [
            '#1a3d1a',  # Very dark green (lowest land)
            '#2d5016',  # Dark forest green
            '#4d6b2d',  # Forest green
            '#6b8f47',  # Medium green
            '#8fbc8f',  # Light green
            '#a0a040',  # Olive
            '#b8b860',  # Light olive
            '#d2b48c',  # Tan
            '#daa520',  # Goldenrod
            '#cd853f',  # Peru brown
            '#a0522d',  # Sienna
            '#8b4513',  # Saddle brown
            '#654321',  # Dark brown
            '#8b7355',  # Light brown (high peaks)
            '#a0826d',  # Rosy brown
            '#bc9a6a'   # Light grayish brown (highest peaks)
        ]
        
        # More colors for smoother transitions
        from matplotlib.colors import LinearSegmentedColormap
        n_water = len(water_colors)
        n_land = 300  # More land colors for ultra-smooth transitions
        
        # Create land colormap
        land_cmap = LinearSegmentedColormap.from_list("dramatic_land", land_colors, N=n_land)
        land_color_list = [land_cmap(i/n_land) for i in range(n_land)]
        
        # Combine all colors
        all_colors = water_colors + land_color_list
        
        # Create elevation boundaries - exclude water from colorbar
        land_max = max_elev
        
        # Water boundaries (will not show in colorbar)
        water_levels = np.linspace(min_elev, -1, n_water)
        land_levels = np.linspace(0, land_max, n_land)
        
        # Combine boundaries
        boundaries = list(water_levels) + [0] + list(land_levels)
        
        # Create colormap and normalization
        cmap = ListedColormap(all_colors)
        norm = BoundaryNorm(boundaries, len(all_colors))
        
        # Create land-only colormap for the colorbar (no blue water colors)
        land_cmap_only = LinearSegmentedColormap.from_list("land_only", land_colors, N=n_land)
        land_norm_only = plt.Normalize(vmin=0, vmax=land_max)
        
        # No vertical exaggeration - use actual elevation data
        vertical_exaggeration = 1  # No exaggeration for realistic terrain
        elev_3d = elev_sub * vertical_exaggeration
        
        # Make all ocean consistently blue at the same level
        elev_3d = np.where(elev_sub < 0, -10, elev_3d)  # Set all ocean to consistent depth level
        
        # Plot the 3D surface with smooth interpolation
        surf = ax.plot_surface(lon2d, lat2d, elev_3d, 
                              facecolors=cmap(norm(elev_sub)),
                              alpha=1.0, linewidth=0, antialiased=True,
                              rasterized=False, shade=True,
                              rstride=1, cstride=1)  # Higher resolution surface
        
    else:
        print("   No water bodies detected, using terrain colormap...")
        vertical_exaggeration = 1  # No exaggeration to match water case
        elev_3d = elev_sub * vertical_exaggeration
        surf = ax.plot_surface(lon2d, lat2d, elev_3d, 
                              cmap='terrain', alpha=1.0,
                              linewidth=0, antialiased=True,
                              rasterized=False, shade=True,
                              rstride=1, cstride=1)  # Higher resolution surface
        
        # Create land colormap for areas without water bodies
        land_colors = [
            '#1a3d1a',  # Very dark green (lowest land)
            '#2d5016',  # Dark forest green
            '#4d6b2d',  # Forest green
            '#6b8f47',  # Medium green
            '#8fbc8f',  # Light green
            '#a0a040',  # Olive
            '#b8b860',  # Light olive
            '#d2b48c',  # Tan
            '#daa520',  # Goldenrod
            '#cd853f',  # Peru brown
            '#a0522d',  # Sienna
            '#8b4513',  # Saddle brown
            '#654321',  # Dark brown
            '#8b7355',  # Light brown (high peaks)
            '#a0826d',  # Rosy brown
            '#bc9a6a'   # Light grayish brown (highest peaks)
        ]
        land_cmap_only = LinearSegmentedColormap.from_list("land_only", land_colors, N=300)
        land_norm_only = plt.Normalize(vmin=0, vmax=max_elev)
    
    print("   Adding clear focus area boundary...")
    # Add focus area boundary as thick outlined box on the surface
    # Get average elevation in focus area for boundary placement
    focus_mask = ((lon2d >= focus_lon_min) & (lon2d <= focus_lon_max) & 
                  (lat2d >= focus_lat_min) & (lat2d <= focus_lat_max))
    if np.any(focus_mask):
        avg_elev = np.mean(elev_3d[focus_mask]) + 500  # Slightly above average
    else:
        avg_elev = max_elev * vertical_exaggeration * 0.1
    
    # Create thick boundary lines
    boundary_x = [focus_lon_min, focus_lon_max, focus_lon_max, focus_lon_min, focus_lon_min]
    boundary_y = [focus_lat_max, focus_lat_max, focus_lat_min, focus_lat_min, focus_lat_max]
    boundary_z = [avg_elev] * len(boundary_x)
    
    # Plot thick red boundary (thinner line)
    ax.plot(boundary_x, boundary_y, boundary_z, color='red', linewidth=3, 
            label='Focus Area', alpha=1.0, zorder=1000)
    
    # Add corner markers for better visibility (smaller markers)
    for x, y in [(focus_lon_min, focus_lat_max), (focus_lon_max, focus_lat_max),
                 (focus_lon_max, focus_lat_min), (focus_lon_min, focus_lat_min)]:
        ax.scatter([x], [y], [avg_elev], color='red', s=50, alpha=1.0, zorder=1001)
    
    print("   Setting optimal view and formatting...")
    # Set straight top-down view rotated 270 degrees with axis labels adjusted
    ax.view_init(elev=90, azim=270)  # Straight top-down view, rotated 270 degrees
    
    # Set axis limits to perfectly fit the data with north at top
    ax.set_xlim(west_total, east_total)
    ax.set_ylim(south_total, north_total)
    ax.set_zlim(elev_3d.min(), elev_3d.max())
    
    # Clean up axis appearance
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    
    # Force the plot to fill the entire axis area without exceeding boundaries
    ax.margins(0)  # Remove any margins
    ax.set_box_aspect([1,1,0.1])  # Flatten z-dimension to keep within axis box
    
    # Remove all axis visibility except labels - hide all 3D axis elements
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    # Completely hide the z-axis and all its components
    ax.zaxis.set_visible(False)
    
    # Remove all axis labels - no longitude or latitude labels
    ax.set_xlabel('')  # No label for longitude axis
    ax.set_ylabel('')  # No label for latitude axis
    
    # Ensure no z-axis labels or ticks are shown
    ax.set_zticks([])
    ax.zaxis.line.set_visible(False)
    
    # Add Natural Earth water features as 2D blue lines on top of 3D surface
    print("   Adding water features from Natural Earth as blue lines...")
    
    for water_feature in water_features_2d:
        feature_lons = water_feature['lons']
        feature_lats = water_feature['lats']
        feature_type = water_feature['type']
        
        # Filter coordinates to our region and remove any out-of-bounds points
        filtered_lons = []
        filtered_lats = []
        
        for lon_w, lat_w in zip(feature_lons, feature_lats):
            if (west_total <= lon_w <= east_total and 
                south_total <= lat_w <= north_total):
                filtered_lons.append(lon_w)
                filtered_lats.append(lat_w)
        
        if len(filtered_lons) < 2:  # Skip if not enough points
            continue
            
        # Get surface elevation for each water feature point for proper positioning
        water_elevs = []
        for lon_w, lat_w in zip(filtered_lons, filtered_lats):
            # Find nearest grid point
            lon_idx = np.argmin(np.abs(lons_sub - lon_w))
            lat_idx = np.argmin(np.abs(lats_sub - lat_w))
            
            # Ensure indices are within bounds
            lat_idx = min(max(lat_idx, 0), len(elev_3d) - 1)
            lon_idx = min(max(lon_idx, 0), len(elev_3d[0]) - 1)
            
            # Position water feature line slightly above surface
            surface_elev = elev_3d[lat_idx, lon_idx]
            water_elevs.append(surface_elev + 150)  # 150m above surface
        
        # Plot water feature as blue line with proper scaling
        if feature_type == 'river':
            ax.plot(filtered_lons, filtered_lats, water_elevs, 
                   color='#004499', linewidth=1.2, alpha=0.8, zorder=1002)
        else:  # lake
            ax.plot(filtered_lons, filtered_lats, water_elevs, 
                   color='#004499', linewidth=1.5, alpha=0.9, zorder=1002)
    
    # Create custom grid with clean labels and degree symbols
    ax.grid(False)  # Turn off default grid
    
    # Add custom coordinate grid lines - remove dashes, keep only labels
    x_ticks = np.arange(west_total, east_total + 1, 5)
    y_ticks = np.arange(south_total, north_total + 1, 5)
    
    # Set custom ticks
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    # Create custom tick labels with degree symbols
    x_labels = [f'{int(x)}°E' for x in x_ticks]
    y_labels = [f'{int(y)}°N' if y >= 0 else f'{int(abs(y))}°S' for y in y_ticks]
    
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    
    # Style the tick labels - position them on the left side
    ax.tick_params(axis='x', labelsize=12, pad=5, labelbottom=True, labeltop=False)
    ax.tick_params(axis='y', labelsize=12, pad=5, labelleft=True, labelright=False)
    
    # Add colorbar and legend positioned closer to plot
    print("   Adding color key and legend...")
    
    # Position colorbar horizontally below the plot, same width as longitude axis
    sm = plt.cm.ScalarMappable(cmap=land_cmap_only, norm=land_norm_only)
    sm.set_array([])
    
    # Create horizontal colorbar below the plot, exactly matching x-axis (longitude) length
    cb = plt.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.8, aspect=40, 
                     pad=0.15, fraction=0.04)
    cb.set_label('Elevation (meters above sea level)', fontsize=12, fontweight='bold')
    
    # Set clean elevation values for the colorbar every 500m
    elevation_ticks = list(range(0, int(max_elev) + 500, 500))
    # Filter ticks to only show values within our data range
    valid_ticks = [tick for tick in elevation_ticks if 0 <= tick <= max_elev]
    
    cb.set_ticks(valid_ticks)
    cb.set_ticklabels([f'{tick:,}' for tick in valid_ticks])  # Add comma separators
    cb.ax.tick_params(labelsize=10, length=0)  # Remove tick marks
    
    # Add focus area legend positioned on top left of the plot - extremely small
    legend = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), 
                      frameon=True, fancybox=False, shadow=False,
                      fontsize=12, edgecolor='red', facecolor='white', 
                      framealpha=0.9, handlelength=0.6, handletextpad=0.2,
                      borderpad=0.15, columnspacing=0.2, markerscale=0.6)
    if legend.get_frame():
        legend.get_frame().set_linewidth(2)
    
    print(f"\n4. SAVING BIRD'S EYE 3D TERRAIN MAP...")
    plt.savefig('africa_elevation_fast_etopo1_3d.png', dpi=1500, 
               bbox_inches='tight', facecolor='white', edgecolor='none',
               pad_inches=0.05)  # Minimal padding to prevent clipping
    
    print("   ✓ Bird's Eye 3D Map saved as 'africa_elevation_fast_etopo1_3d.png' at 1500 DPI")
    
    # Create separate axis image with red border labels
    print(f"\n5. CREATING SEPARATE AXIS IMAGE WITH RED BORDER LABELS...")
    
    # Create a clean axis-only figure
    axis_fig, axis_ax = plt.subplots(figsize=(12, 8), dpi=300)
    axis_ax.set_xlim(west_total, east_total)
    axis_ax.set_ylim(south_total, north_total)
    
    # Remove all content, keep only axis structure
    axis_ax.set_facecolor('white')
    
    # Add grid
    x_ticks = np.arange(west_total, east_total + 1, 5)
    y_ticks = np.arange(south_total, north_total + 1, 5)
    
    axis_ax.set_xticks(x_ticks)
    axis_ax.set_yticks(y_ticks)
    axis_ax.grid(True, alpha=0.3, linestyle='-', color='gray')
    
    # Create custom tick labels with degree symbols
    x_labels = [f'{int(x)}°E' for x in x_ticks]
    y_labels = [f'{int(y)}°N' if y >= 0 else f'{int(abs(y))}°S' for y in y_ticks]
    
    axis_ax.set_xticklabels(x_labels, fontsize=12)
    axis_ax.set_yticklabels(y_labels, fontsize=12)
    
    # Add focus area border with RED LABELS
    focus_border = plt.Rectangle((focus_lon_min, focus_lat_min), 
                                focus_lon_max - focus_lon_min, 
                                focus_lat_max - focus_lat_min,
                                fill=False, edgecolor='red', linewidth=3)
    axis_ax.add_patch(focus_border)
    
    # Add RED LABELS at the border corners and midpoints
    # Corner labels
    axis_ax.text(focus_lon_min, focus_lat_max, f'{focus_lon_min}°E,{focus_lat_max}°N', 
                ha='right', va='bottom', fontsize=11, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    axis_ax.text(focus_lon_max, focus_lat_max, f'{focus_lon_max}°E,{focus_lat_max}°N', 
                ha='left', va='bottom', fontsize=11, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    axis_ax.text(focus_lon_min, focus_lat_min, f'{focus_lon_min}°E,{focus_lat_min}°N', 
                ha='right', va='top', fontsize=11, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    axis_ax.text(focus_lon_max, focus_lat_min, f'{focus_lon_max}°E,{focus_lat_min}°N', 
                ha='left', va='top', fontsize=11, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add axis labels
    axis_ax.set_xlabel('Longitude', fontsize=14, fontweight='bold')
    axis_ax.set_ylabel('Latitude', fontsize=14, fontweight='bold')
    axis_ax.set_title('Study Area Coordinates with Focus Region', fontsize=16, fontweight='bold')
    
    # Save the separate axis image
    plt.savefig('africa_axis_with_borders.png', dpi=300, 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    
    print("   ✓ Separate axis image saved as 'africa_axis_with_borders.png'")
    
    plt.close(axis_fig)  # Close the axis figure
    
    plt.show()  # Show the main 3D plot
    
    print("\n" + "=" * 60)
    print("✓ CLEAN 3D TERRAIN MAP COMPLETED SUCCESSFULLY!")
    print("✓ View: North-facing bird's eye perspective")
    print("✓ Formatting: Clean with no elevation axis, smooth blue ocean")
    print("✓ Longitude along X-axis, Latitude along Y-axis")
    print("✓ This method is much faster than merging individual tiles")
    print("✓ Resolution: ~1.8km at the equator (1 arc-minute)")
    print(f"✓ Vertical exaggeration: {vertical_exaggeration}x (no exaggeration for realistic terrain)")
    print("✓ Includes: Clear focus area boundary, close color key and legend")
    print("✓ Separate axis image with red border labels created")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    create_fast_elevation_map_3d()
