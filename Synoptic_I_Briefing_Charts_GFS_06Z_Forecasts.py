from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.units import units
import numpy as np
from siphon.ncss import NCSS
import cartopy.util as cutil
import matplotlib.gridspec as gridspec
from metpy.calc import find_peaks
from metpy.plots import scattertext
from metpy.plots import StationPlot
from siphon.simplewebservice.wyoming import WyomingUpperAir
import pandas as pd
import urllib.error
import matplotlib.patheffects as PathEffects
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter
import os


from pathlib import Path

output_dir = Path("/mnt/c/Users/jakewiley/Documents")

for png_file in output_dir.glob("*.png"):
    png_file.unlink()

print("Existing PNG files removed.")

os.chdir(r"/mnt/c/Users/jakewiley/Documents")

from datetime import datetime, timezone

now_utc = datetime.now(timezone.utc)

# 00Z analysis date
master_date_00Z = now_utc.strftime("%Y%m%d_0000")
year_00Z = now_utc.year
month_00Z = now_utc.month
month_name_00Z = now_utc.strftime("%B")
day_00Z = now_utc.day
hour_00Z = "00"

# 12Z analysis date
master_date_12Z = now_utc.strftime("%Y%m%d_1200")
year_12Z = now_utc.year
month_12Z = now_utc.month
month_name_12Z = now_utc.strftime("%B")
day_12Z = now_utc.day
hour_12Z = "12"

# Forecast charts use the 06Z GFS cycle so they are available earlier,
# while retaining the same valid times previously produced from the
# 12Z cycle at +12 h and +24 h. Therefore, the 06Z forecasts use
# +18 h and +30 h lead times, respectively.

date_12z_for_forecast = datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z))
date_06z = date_12z_for_forecast - timedelta(hours=6)

master_date_06Z = date_06z.strftime("%Y%m%d_%H%M")
year_06Z = date_06z.year
month_06Z = date_06z.month
month_name_06Z = date_06z.strftime("%B")
day_06Z = date_06z.day
hour_06Z = date_06z.strftime("%H")

fcst_18hr = date_06z + timedelta(hours=18)
fcst_30hr = date_06z + timedelta(hours=30)

fcst_18hr_year = fcst_18hr.year
fcst_18hr_month = fcst_18hr.month
fcst_18hr_month_name = fcst_18hr.strftime("%B")
fcst_18hr_day = fcst_18hr.day
fcst_18hr_hour = fcst_18hr.strftime("%H")

fcst_30hr_year = fcst_30hr.year
fcst_30hr_month = fcst_30hr.month
fcst_30hr_month_name = fcst_30hr.strftime("%B")
fcst_30hr_day = fcst_30hr.day
fcst_30hr_hour = fcst_30hr.strftime("%H")


################################################################################################################
# ANALYSIS CHARTS (00Z)
################################################################################################################

# CHART 1: 500-mb Hemispheric Chart (Analysis)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_00Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=360, west=0)
query.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'u-component_of_wind_isobaric',
                'v-component_of_wind_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
u_wind_var = data_query.variables['u-component_of_wind_isobaric']
v_wind_var = data_query.variables['v-component_of_wind_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]

# Smooth the 500-hPa heights using a gaussian filter from scipy.ndimage
hgt_500, lon = cutil.add_cyclic_point(data_query.variables['Geopotential_height_isobaric'][:],
                                      coord=lon)
Z_500 = mpcalc.smooth_n_point(hgt_500[0, 0, :, :], 9, 50)

height_dam = Z_500 / 10

u500 = (units(data_query.variables['u-component_of_wind_isobaric'].units) *
        data_query.variables['u-component_of_wind_isobaric'][0, 0, :, :])
v500 = (units(data_query.variables['v-component_of_wind_isobaric'].units) *
        data_query.variables['v-component_of_wind_isobaric'][0, 0, :, :])
u500 = u500.units * cutil.add_cyclic_point(u500)
v500 = v500.units * cutil.add_cyclic_point(v500)
wspd500 = mpcalc.wind_speed(u500, v500).to('knots')

datacrs = ccrs.PlateCarree()
plotcrs = ccrs.NorthPolarStereo(central_longitude=-100.0)

# Make a grid of lat/lon values to use for plotting with Basemap.
lons, lats = np.meshgrid(lon, lat)

fig = plt.figure(1, figsize=(12., 13.))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, .02],
                       bottom=.07, top=.99, hspace=0.01, wspace=0.01)

ax = plt.subplot(gs[0], projection=plotcrs)
ax.set_title('GFS 500-mb Geopotential Heights (dam) and Wind Barbs (kt)', loc='left')
ax.set_title('VALID: ' + str(day_00Z) + ' ' + str(month_name_00Z) + ' ' + str(year_00Z) + ' ' + str(hour_00Z) + '00 UTC', loc='right')

#   ax.set_extent([west long, east long, south lat, north lat])
ax.set_extent([-180, 180, 20, 90], ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS, edgecolor='brown', linewidth=1.5)
ax.add_feature(cfeature.COASTLINE, edgecolor='brown', linewidth=1.5)
ax.add_feature(cfeature.STATES, edgecolor='brown', linewidth=1)
#provinces = cartopy.feature.NaturalEarthFeature(category='cultural', 
#    name='admin_1_states_provinces_lines', scale='50m', facecolor='none', edgecolor='brown')
#ax.add_feature(provinces, linewidth=1, edgecolor="brown", zorder=10)

clev500 = np.arange(0, 800, 6)
cs = ax.contour(lons, lats, height_dam, clev500, colors='k',
                linewidths=1.5, linestyles='solid', transform=datacrs)
plt.clabel(cs, fontsize=10, inline=1, inline_spacing=10, fmt='%i',
           rightside_up=True, use_clabeltext=True)

clev564 = np.arange(564, 570, 6)
cs2 = ax.contour(lons, lats, height_dam, clev564, colors='k',
                linewidths=3, linestyles='solid', transform=datacrs)
plt.clabel(cs2, fontsize=10, inline=1, inline_spacing=10, fmt='%i',
           rightside_up=True, use_clabeltext=True)

wind = ax.barbs(lons, lats,
                 u500.to('kt').m, v500.to('kt').m,
                 color='darkblue', regrid_shape=20, length=6, zorder=2, transform=ccrs.PlateCarree())

h_y, h_x = find_peaks(height_dam)
l_y, l_x = find_peaks(height_dam, maxima=False)

# Using scattertext() plot the high centers using a red 'H' and put the height value
# below the 'H' using a smaller font.
scattertext(ax, lon[h_x], lat[h_y], 'H', size=20, color='blue',
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
scattertext(ax, lon[h_x], lat[h_y], height_dam[h_y, h_x], formatter='.0f',
            size=12, color='blue', loc=(0, -15), fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())

# Now do the same for the lows using a blue 'L'
scattertext(ax, lon[l_x], lat[l_y], 'L', size=20, color='red',
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
scattertext(ax, lon[l_x], lat[l_y], height_dam[l_y, l_x], formatter='.0f',
            size=12, color='red', loc=(0, -15), fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())

plt.savefig('500mb_Hemispheric_Chart_GFS_00Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 1 complete')


################################################################################################################

# CHART 2: 250-mb Jet Streak Chart (Analysis)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_00Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'u-component_of_wind_isobaric',
                'v-component_of_wind_isobaric')
query.vertical_level(25000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
u_wind_var = data_query.variables['u-component_of_wind_isobaric']
v_wind_var = data_query.variables['v-component_of_wind_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

# Get actual data values and remove any size 1 dimensions
height = height_var[0, 0, :, :].squeeze()
u_wind = u_wind_var[0, 0, :, :].squeeze() * units('m/s')
v_wind = v_wind_var[0, 0, :, :].squeeze() * units('m/s')
u_var = u_wind_var[0, 0, :, :].squeeze().data * units('m/s')
v_var = v_wind_var[0, 0, :, :].squeeze().data * units('m/s')
u_knots = u_var.to('knots')
v_knots = v_var.to('knots')

# Smooth height data
height = mpcalc.smooth_n_point(height, 9, 5)

# Calculate the 250-mb wind speed
u250 = (units(data_query.variables['u-component_of_wind_isobaric'].units) *
        data_query.variables['u-component_of_wind_isobaric'][0, 0, :, :])
v250 = (units(data_query.variables['v-component_of_wind_isobaric'].units) *
        data_query.variables['v-component_of_wind_isobaric'][0, 0, :, :])
wspd250 = mpcalc.wind_speed(u250, v250).to('knots')
wspd250 = mpcalc.smooth_n_point(wspd250, 9, 5)

# Create new figure
fig = plt.figure(figsize=(14, 12))

# Add the map and set the extent
ax = plt.axes(projection=ccrs.LambertConformal(central_latitude=35., central_longitude=-100., standard_parallels=(30, 60)))
ax.set_extent([-128.5, -73.2, 18.5, 58])
#ax.background_patch.set_fill(False)

# Add state boundaries to plot
ax.add_feature(cfeature.BORDERS, edgecolor='brown', linewidth=1)
ax.add_feature(cfeature.COASTLINE, edgecolor='brown', linewidth=1)
ax.add_feature(cfeature.STATES, edgecolor='brown', linewidth=1)

# Contour the heights
height_dam = height / 10
contours = np.arange(0, 1200, 12)
# Because we have a very local graphics area, the contours have joints
# to smooth those out we can use `ndimage.zoom`
#zoom_500 = ndimage.zoom(height, 5)
#zlon = ndimage.zoom(lon, 5)
#zlat = ndimage.zoom(lat, 5)
c = ax.contour(lons, lats, height_dam, levels=contours,
               colors='black', linewidths=2, transform=ccrs.PlateCarree())
labels = ax.clabel(c, fontsize=10, inline=1, inline_spacing=3, fmt='%i')
for label in labels:
    label.set_bbox(dict(facecolor='white', edgecolor='black', boxstyle='square, pad=0.2'))

# Set up parameters for quiver plot. The slices below are used to subset the data (here
# taking every 4th point in x and y). The quiver_kwargs are parameters to control the
# appearance of the quiver so that they stay consistent between the calls.
barbs_slices = (slice(None, None, 5), slice(None, None, 5))

# Plot the 250-mb wind speed as contour fill
# Create colorbar and set properties
colors = [
"#00eeee",
"#1e90ff",
"#912cee",
"#ff00ff",
"#ff0000",
"#ffff00"
]
colormap = matplotlib.colors.ListedColormap(colors)
levels = np.array([70, 90, 110, 130, 150, 200, 250])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=6)
cf = ax.contourf(lons, lats, wspd250, levels, norm=norm, cmap=colormap, transform=ccrs.PlateCarree())
cbar = plt.colorbar(cf, ticks=levels, orientation='horizontal', pad=0.02, shrink=0.5)

# Plot the wind vectors
wind = ax.barbs(lons[barbs_slices], lats[barbs_slices],
                 u_knots[barbs_slices], v_knots[barbs_slices],
                 color='darkblue', length=6, zorder=2, transform=ccrs.PlateCarree())

# Add a title to the plot
plt.title('GFS 250-mb Geopotential Heights (dam) and Wind Speed (knots)\n'
          'VALID: ' + str(day_00Z) + ' ' + str(month_name_00Z) + ' ' + str(year_00Z) + ' ' + str(hour_00Z) + '00 UTC',
          color='black', size=14)

plt.savefig('250mb_Jet_Streak_Chart_GFS_00Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 2 complete')


################################################################################################################

# CHART 3: 200-mb Jet Streak Chart (Analysis)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_00Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'u-component_of_wind_isobaric',
                'v-component_of_wind_isobaric')
query.vertical_level(20000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
u_wind_var = data_query.variables['u-component_of_wind_isobaric']
v_wind_var = data_query.variables['v-component_of_wind_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

# Get actual data values and remove any size 1 dimensions
height = height_var[0, 0, :, :].squeeze()
u_wind = u_wind_var[0, 0, :, :].squeeze() * units('m/s')
v_wind = v_wind_var[0, 0, :, :].squeeze() * units('m/s')
u_var = u_wind_var[0, 0, :, :].squeeze().data * units('m/s')
v_var = v_wind_var[0, 0, :, :].squeeze().data * units('m/s')
u_knots = u_var.to('knots')
v_knots = v_var.to('knots')

# Smooth height data
height = mpcalc.smooth_n_point(height, 9, 5)

# Calculate the 200-mb wind speed
u200 = (units(data_query.variables['u-component_of_wind_isobaric'].units) *
        data_query.variables['u-component_of_wind_isobaric'][0, 0, :, :])
v200 = (units(data_query.variables['v-component_of_wind_isobaric'].units) *
        data_query.variables['v-component_of_wind_isobaric'][0, 0, :, :])
wspd200 = mpcalc.wind_speed(u200, v200).to('knots')
wspd200 = mpcalc.smooth_n_point(wspd200, 9, 5)

# Create new figure
fig = plt.figure(figsize=(14, 12))

# Add the map and set the extent
ax = plt.axes(projection=ccrs.LambertConformal(central_latitude=35., central_longitude=-100., standard_parallels=(30, 60)))
ax.set_extent([-128.5, -73.2, 18.5, 58])
#ax.background_patch.set_fill(False)

# Add state boundaries to plot
ax.add_feature(cfeature.BORDERS, edgecolor='brown', linewidth=1)
ax.add_feature(cfeature.COASTLINE, edgecolor='brown', linewidth=1)
ax.add_feature(cfeature.STATES, edgecolor='brown', linewidth=1)

# Contour the heights
height_dam = height / 10
contours = np.arange(0, 1300, 12)
# Because we have a very local graphics area, the contours have joints
# to smooth those out we can use `ndimage.zoom`
#zoom_500 = ndimage.zoom(height, 5)
#zlon = ndimage.zoom(lon, 5)
#zlat = ndimage.zoom(lat, 5)
c = ax.contour(lons, lats, height_dam, levels=contours,
               colors='black', linewidths=2, transform=ccrs.PlateCarree())
labels = ax.clabel(c, fontsize=10, inline=1, inline_spacing=3, fmt='%i')
for label in labels:
    label.set_bbox(dict(facecolor='white', edgecolor='black', boxstyle='square, pad=0.2'))

# Set up parameters for quiver plot. The slices below are used to subset the data (here
# taking every 4th point in x and y). The quiver_kwargs are parameters to control the
# appearance of the quiver so that they stay consistent between the calls.
barbs_slices = (slice(None, None, 5), slice(None, None, 5))

# Plot the 250-mb wind speed as contour fill
# Create colorbar and set properties
colors = [
"#00eeee",
"#1e90ff",
"#912cee",
"#ff00ff",
"#ff0000",
"#ffff00"
]
colormap = matplotlib.colors.ListedColormap(colors)
levels = np.array([70, 90, 110, 130, 150, 200, 250])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=6)
cf = ax.contourf(lons, lats, wspd200, levels, norm=norm, cmap=colormap, transform=ccrs.PlateCarree())
cbar = plt.colorbar(cf, ticks=levels, orientation='horizontal', pad=0.02, shrink=0.5)

# Plot the wind vectors
wind = ax.barbs(lons[barbs_slices], lats[barbs_slices],
                 u_knots[barbs_slices], v_knots[barbs_slices],
                 color='darkblue', length=6, zorder=2, transform=ccrs.PlateCarree())

# Add a title to the plot
plt.title('GFS 200-mb Geopotential Heights (dam) and Wind Speed (knots)\n'
          'VALID: ' + str(day_00Z) + ' ' + str(month_name_00Z) + ' ' + str(year_00Z) + ' ' + str(hour_00Z) + '00 UTC',
          color='black', size=14)

plt.savefig('200mb_Jet_Streak_Chart_GFS_00Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 3 complete')


################################################################################################################

# CHARTS 4–6: DIFAX Replicas at the 500-mb, 700-mb, and 850-mb Levels (Analysis)

# Plotting High/Low Symbols
# -------------------------
#
# A helper function to plot a text symbol (e.g., H, L) for relative
# maximum/minimum for a given field (e.g., geopotential height).
#


def plot_maxmin_points(ax, lon, lat, data, extrema, nsize, symbol,
                       color='k', plot_value=True,
                       transform=ccrs.PlateCarree(),
                       edge_buffer=0.02,
                       min_dist=10e5):
    """
    Plot local maxima or minima on a map using a given symbol, with optional buffer
    to avoid plotting right at map edges and thinning of clustered points.

    Parameters
    ----------
    min_dist : float
        Minimum distance (in map projection units) to separate points when thinning.
    """
    import numpy as np
    from scipy.ndimage import maximum_filter, minimum_filter

    # Find local maxima/minima
    if extrema == 'max':
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif extrema == 'min':
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError("extrema must be 'max' or 'min'")

    # Ensure lon/lat are 2D
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    # Indices of extrema points
    mxx, mxy = np.where(data_ext == data)

    # Stack lon/lat and transform to map projection
    points = np.stack([lon[mxx, mxy], lat[mxx, mxy]], axis=1)
    proj_points = ax.projection.transform_points(transform,
                                                 points[:, 0], points[:, 1])

    # Get map extent in projected coordinates
    x0, x1, y0, y1 = ax.get_extent()

    # Apply edge buffer
    dx = (x1 - x0) * edge_buffer
    dy = (y1 - y0) * edge_buffer
    x0_buff, x1_buff = x0 + dx, x1 - dx
    y0_buff, y1_buff = y0 + dy, y1 - dy

    # Filter points to only those inside buffered map
    inside = (proj_points[:, 0] >= x0_buff) & (proj_points[:, 0] <= x1_buff) & \
             (proj_points[:, 1] >= y0_buff) & (proj_points[:, 1] <= y1_buff)

    lon_keep = points[inside, 0]
    lat_keep = points[inside, 1]
    data_keep = data[mxx, mxy][inside]
    proj_keep = proj_points[inside, :2]

    # Thinning: keep points separated by at least min_dist
    selected_idx = []
    for i, p in enumerate(proj_keep):
        if len(selected_idx) == 0:
            selected_idx.append(i)
            continue
        dists = np.linalg.norm(proj_keep[selected_idx] - p, axis=1)
        if np.all(dists >= min_dist):
            selected_idx.append(i)

    # Keep only thinned points
    lon_keep = lon_keep[selected_idx]
    lat_keep = lat_keep[selected_idx]
    data_keep = data_keep[selected_idx]

    # Plot symbols and optional values
    for x, y, val in zip(lon_keep, lat_keep, data_keep):
        ax.text(
            x, y, symbol,
            size=36,
            color=color,
            ha='center', va='center',
            transform=transform,
            clip_on=True,
            zorder=10
        )
        if plot_value:
            ax.text(
                x, y - 1,
                f"{int(val)}",
                size=12,
                fontweight='bold',
                color=color,
                ha='center', va='top',
                transform=transform,
                clip_on=True,
                zorder=10
            )

# Observation Data
# ----------------
#
# Set a date and time for upper-air observations (should only be 00 or 12
# UTC for the hour).
#
# Request all data from Iowa State using the Siphon package. The result is
# a pandas DataFrame containing all of the sounding data from all
# available stations.
#

# Set date for desired UPA data
# today = datetime.utcnow()
date = datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z))

# Previous is 12 hours before that time
prev = date - timedelta(hours=12)

# Request data using Siphon request for data from Iowa State Archive
#data = IAStateUpperAir.request_all_data(date)
#prev_data = IAStateUpperAir.request_all_data(prev)

data_ua = pd.DataFrame()
prev_data_ua = pd.DataFrame()
upper_air_stations = ['BMX','FFC','EYW','MFL','TBW','JAX','CHS','MHX','GSO','RNK','IAD','WAL','OKX','CAR','ALB','BUF','PIT','ILN','DTX','APX','BNA','JAN','LIX','BRO','FWD','MAF','OUN','DDC','TOP','LZK','SGF','ILX','DVN','GRB','MPX','INL','ABR','EPZ','RIW','BOI','SLC','VEF','LKN','REV','ANT','YZT','YYE','ZXS','WVK','WSE','YQD','YYQ','YPL','WMW','YAH','YVP','YZV','YYR','YQI','YJT','GGW','VBG','SLE','DRT','MCU','MLP','GYM','YSM']
for station in range(0,len(upper_air_stations)):
    try:
        data_0 = WyomingUpperAir.request_data(date, upper_air_stations[station])
        prev_data_0 = WyomingUpperAir.request_data(prev, upper_air_stations[station])
        data_ua = pd.concat([data_ua, data_0], ignore_index=True)
        prev_data_ua = pd.concat([prev_data_ua, prev_data_0], ignore_index=True)
        print(station)
    except urllib.error.HTTPError:
        print(upper_air_stations[station])
        pass
    except ValueError:
        print(upper_air_stations[station])
        pass

data_ua_00Z = pd.DataFrame()
upper_air_stations_00Z = ['GYX','SHV','LCH','CRP','AMA','BIS','UNR','LBF','OAX','ABQ','GJT','FGZ','OTX','UIL','MFR','OAK','TUS']
for station_00Z in range(0,len(upper_air_stations_00Z)):
    try:
        data_0_00Z = WyomingUpperAir.request_data(date, upper_air_stations_00Z[station_00Z])
        data_ua_00Z = pd.concat([data_ua_00Z, data_0_00Z], ignore_index=True)
        print(station_00Z)
    except urllib.error.HTTPError:
        print(upper_air_stations_00Z[station_00Z])
        pass
    except ValueError:
        print(upper_air_stations_00Z[station_00Z])
        pass

# Subset Observational Data
# -------------------------
#
# From the request above will give all levels from all radiosonde sites
# available through the service. For plotting a pressure surface map there
# is only need to have the data from that level. Below the data is subset
# and a few parameters set based on the level chosen. Additionally, the
# station information is obtained and latitude and longitude data is added
# to the DataFrame.
#

level = 500

if (level == 925) | (level == 850) | (level == 700):
    cint = 30
    def hght_format(v): return format(v, '.0f')[1:]
elif level == 500:
    cint = 60
    def hght_format(v): return format(v, '.0f')[:3]
elif level == 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[:3]
elif level < 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[1:4]

# Create subset of all data for a given level
data_subset = data_ua.pressure == level
df = data_ua[data_subset]

data_subset_00Z = data_ua_00Z.pressure == level
df_00Z = data_ua_00Z[data_subset_00Z]

# Given our data, add station information and drop any stations that are missing lat/lon. Then set
# 'station' as the index column so that operations between datasets will align on that column
#data = add_station_lat_lon(data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')
#prev_data = add_station_lat_lon(prev_data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')

# Create subset of all data for a given level
df = data_ua[data_ua.pressure == level].copy()
prev_df = prev_data_ua[prev_data_ua.pressure == level]

# Keep only the rows in prev_df where the 'station' value is also in df['station']
prev_df = prev_df[prev_df['station'].isin(df['station'])]

# Keep only the rows in df where the 'station' value is also in prev_df['station']
df = df[df['station'].isin(prev_df['station'])]

# Calculate the change on the aligned data frames--needs to be
# added to original dataframe to maintain consistent ordering
# with later use
df['height_change'] = df['height'].values - prev_df['height'].values

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_00Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'Temperature_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
temp_var = data_query.variables['Temperature_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
temp = temp_var[0, 0, :, :].squeeze()
smooth_tmpk = mpcalc.smooth_n_point(temp, 9, 5)
smooth_tmpc = smooth_tmpk - 273.15


# Create DIFAX Replication
# ------------------------
#
# Plot the observational data and contours on a Lambert Conformal map and
# add features that resemble the historic DIFAX maps.
#

# Set up map coordinate reference system
mapcrs = ccrs.LambertConformal(
    central_latitude=45, central_longitude=-100, standard_parallels=(30, 60))

# Start figure and set graphics extent
fig = plt.figure(1, figsize=(17, 15))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58])

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', zorder=1)
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white', edgecolor='grey', zorder=1)
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey', zorder=1)

# Plot plus signs every degree lat/lon
plus_lat = []
plus_lon = []
other_lat = []
other_lon = []

for x in range(0,360,1):
    for y in range(0,90,1):
        if (round(x) % 5 == 0) | (round(y) % 5 == 0):
            plus_lon.append(x)
            plus_lat.append(y)
        else:
            other_lon.append(x)
            other_lat.append(y)
ax.scatter(other_lon, other_lat, s=5, marker='o',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)
ax.scatter(plus_lon, plus_lat, s=30, marker='+',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)

# Add gridlines for every 5 degree lat/lon
gl = ax.gridlines(linestyle='solid', ylocs=range(15, 71, 5), xlocs=range(-160, -49, 5))
gl.zorder = -1

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 10 pt.
stationplot = StationPlot(ax, df['longitude'].values, df['latitude'].values, clip_on=True,
                          transform=ccrs.PlateCarree(), fontsize=10)

# Plot the temperature and dew point depression to the upper and lower left, respectively, of
# the center point.
stationplot.plot_parameter('NW', df['temperature'], color='black')
stationplot.plot_parameter('SW', df['temperature'] - df['dewpoint'], color='black')

# A more complex example uses a custom formatter to control how the geopotential height
# values are plotted. This is set in an earlier if-statement to work appropriate for
# different levels.
stationplot.plot_parameter('NE', df['height'], formatter=hght_format, color='black')

# Add wind barbs
MS_TO_KT = 1.94384

stationplot.plot_barb(df['u_wind'] * MS_TO_KT, df['v_wind'] * MS_TO_KT, length=7, pivot='tip')

# Add height falls. This converts the change to decameters with /10, then
# formats with a +/- and 0 padding
def height_fall_formatter(v):
    return f'{int(v / 10):+03d}'

# Plot the parameter with an italic font
stationplot.plot_parameter('SE', df['height_change'], formatter=height_fall_formatter,
                           fontstyle='italic', color='black')

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 10 pt.
stationplot_00Z = StationPlot(ax, df_00Z['longitude'].values, df_00Z['latitude'].values, clip_on=True,
                          transform=ccrs.PlateCarree(), fontsize=10)

# Plot the temperature and dew point depression to the upper and lower left, respectively, of
# the center point.
stationplot_00Z.plot_parameter('NW', df_00Z['temperature'], color='black')
stationplot_00Z.plot_parameter('SW', df_00Z['temperature'] - df_00Z['dewpoint'], color='black')

# A more complex example uses a custom formatter to control how the geopotential height
# values are plotted. This is set in an earlier if-statement to work appropriate for
# different levels.
stationplot_00Z.plot_parameter('NE', df_00Z['height'], formatter=hght_format, color='black')

# Add wind barbs
stationplot_00Z.plot_barb(df_00Z['u_wind'] * MS_TO_KT, df_00Z['v_wind'] * MS_TO_KT, length=7, pivot='tip')

# Plot Solid Contours of Geopotential Height
cs = ax.contour(lons, lats, smooth_height,
                range(0, 20000, cint), colors='black', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot Dashed Contours of Temperature
cs2 = ax.contour(lons, lats, smooth_tmpc, range(-60, 51, 3),
                 colors='#c9514b', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs2, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Set longer dashes than default
try:
    for c in cs2.collections:
        c.set_dashes([(0, (5.0, 3.0))])
except AttributeError:
    for c in cs2.get_children():
        c.set_dashes([(0, (5.0, 3.0))])

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot filled circles for Radiosonde Obs
ax.scatter(df['longitude'].values, df['latitude'].values, s=12,
           marker='o', color='black', transform=ccrs.PlateCarree())
ax.scatter(df_00Z['longitude'].values, df_00Z['latitude'].values, s=12,
           marker='o', color='black', transform=ccrs.PlateCarree())

# Use definition to plot H/L symbols
plot_maxmin_points(ax, lons, lats, smooth_height, 'max', 50,
                   symbol='H', color='black', transform=ccrs.PlateCarree())
plot_maxmin_points(ax, lons, lats, smooth_height, 'min', 25,
                   symbol='L', color='black', transform=ccrs.PlateCarree())

# Add titles
plt.title('GFS {}-mb Analysis of Geopotential Heights (m) and Temperature (°C)'.format(level),
          loc='left')
plt.title('VALID: ' + str(day_00Z) + ' ' + str(month_name_00Z) + ' ' + str(year_00Z) + ' ' + str(hour_00Z) + '00 UTC', loc='right')

plt.savefig('500mb_DIFAX_Chart_GFS_00Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 4 complete')


# Subset Observational Data
# -------------------------
#
# From the request above will give all levels from all radiosonde sites
# available through the service. For plotting a pressure surface map there
# is only need to have the data from that level. Below the data is subset
# and a few parameters set based on the level chosen. Additionally, the
# station information is obtained and latitude and longitude data is added
# to the DataFrame.
#

level = 700

if (level == 925) | (level == 850) | (level == 700):
    cint = 30
    def hght_format(v): return format(v, '.0f')[1:]
elif level == 500:
    cint = 60
    def hght_format(v): return format(v, '.0f')[:3]
elif level == 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[:3]
elif level < 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[1:4]

# Create subset of all data for a given level
data_subset = data_ua.pressure == level
df = data_ua[data_subset]

data_subset_00Z = data_ua_00Z.pressure == level
df_00Z = data_ua_00Z[data_subset_00Z]

# Given our data, add station information and drop any stations that are missing lat/lon. Then set
# 'station' as the index column so that operations between datasets will align on that column
#data = add_station_lat_lon(data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')
#prev_data = add_station_lat_lon(prev_data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')

# Create subset of all data for a given level
df = data_ua[data_ua.pressure == level].copy()
prev_df = prev_data_ua[prev_data_ua.pressure == level]

# Keep only the rows in prev_df where the 'station' value is also in df['station']
prev_df = prev_df[prev_df['station'].isin(df['station'])]

# Keep only the rows in df where the 'station' value is also in prev_df['station']
df = df[df['station'].isin(prev_df['station'])]

# Calculate the change on the aligned data frames--needs to be
# added to original dataframe to maintain consistent ordering
# with later use
df['height_change'] = df['height'].values - prev_df['height'].values

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_00Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'Temperature_isobaric')
query.vertical_level(70000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
temp_var = data_query.variables['Temperature_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
temp = temp_var[0, 0, :, :].squeeze()
smooth_tmpk = mpcalc.smooth_n_point(temp, 9, 5)
smooth_tmpc = smooth_tmpk - 273.15

# Create DIFAX Replication
# ------------------------
#
# Plot the observational data and contours on a Lambert Conformal map and
# add features that resemble the historic DIFAX maps.
#

# Set up map coordinate reference system
mapcrs = ccrs.LambertConformal(
    central_latitude=45, central_longitude=-100, standard_parallels=(30, 60))

# Start figure and set graphics extent
fig = plt.figure(1, figsize=(17, 15))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58])

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', zorder=1)
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white', edgecolor='grey', zorder=1)
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey', zorder=1)

# Plot plus signs every degree lat/lon
plus_lat = []
plus_lon = []
other_lat = []
other_lon = []

for x in range(0,360,1):
    for y in range(0,90,1):
        if (round(x) % 5 == 0) | (round(y) % 5 == 0):
            plus_lon.append(x)
            plus_lat.append(y)
        else:
            other_lon.append(x)
            other_lat.append(y)
ax.scatter(other_lon, other_lat, s=5, marker='o',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)
ax.scatter(plus_lon, plus_lat, s=30, marker='+',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)

# Add gridlines for every 5 degree lat/lon
gl = ax.gridlines(linestyle='solid', ylocs=range(15, 71, 5), xlocs=range(-160, -49, 5))
gl.zorder = -1

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 10 pt.
stationplot = StationPlot(ax, df['longitude'].values, df['latitude'].values, clip_on=True,
                          transform=ccrs.PlateCarree(), fontsize=10)

# Plot the temperature and dew point depression to the upper and lower left, respectively, of
# the center point.
stationplot.plot_parameter('NW', df['temperature'], color='black')
stationplot.plot_parameter('SW', df['temperature'] - df['dewpoint'], color='black')

# A more complex example uses a custom formatter to control how the geopotential height
# values are plotted. This is set in an earlier if-statement to work appropriate for
# different levels.
stationplot.plot_parameter('NE', df['height'], formatter=hght_format, color='black')

# Add wind barbs
stationplot.plot_barb(df['u_wind'] * MS_TO_KT, df['v_wind'] * MS_TO_KT, length=7, pivot='tip')

# Add height falls. This converts the change to decameters with /10, then
# formats with a +/- and 0 padding
def height_fall_formatter(v):
    return f'{int(v / 10):+03d}'

# Plot the parameter with an italic font
stationplot.plot_parameter('SE', df['height_change'], formatter=height_fall_formatter,
                           fontstyle='italic', color='black')

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 10 pt.
stationplot_00Z = StationPlot(ax, df_00Z['longitude'].values, df_00Z['latitude'].values, clip_on=True,
                          transform=ccrs.PlateCarree(), fontsize=10)

# Plot the temperature and dew point depression to the upper and lower left, respectively, of
# the center point.
stationplot_00Z.plot_parameter('NW', df_00Z['temperature'], color='black')
stationplot_00Z.plot_parameter('SW', df_00Z['temperature'] - df_00Z['dewpoint'], color='black')

# A more complex example uses a custom formatter to control how the geopotential height
# values are plotted. This is set in an earlier if-statement to work appropriate for
# different levels.
stationplot_00Z.plot_parameter('NE', df_00Z['height'], formatter=hght_format, color='black')

# Add wind barbs
stationplot_00Z.plot_barb(df_00Z['u_wind'] * MS_TO_KT, df_00Z['v_wind'] * MS_TO_KT, length=7, pivot='tip')

# Plot Solid Contours of Geopotential Height
cs = ax.contour(lons, lats, smooth_height,
                range(0, 20000, cint), colors='black', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot Dashed Contours of Temperature
cs2 = ax.contour(lons, lats, smooth_tmpc, range(-60, 51, 3),
                 colors='#c9514b', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs2, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Set longer dashes than default
try:
    for c in cs2.collections:
        c.set_dashes([(0, (5.0, 3.0))])
except AttributeError:
    for c in cs2.get_children():
        c.set_dashes([(0, (5.0, 3.0))])

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot filled circles for Radiosonde Obs
ax.scatter(df['longitude'].values, df['latitude'].values, s=12,
           marker='o', color='black', transform=ccrs.PlateCarree())
ax.scatter(df_00Z['longitude'].values, df_00Z['latitude'].values, s=12,
           marker='o', color='black', transform=ccrs.PlateCarree())

# Use definition to plot H/L symbols
plot_maxmin_points(ax, lons, lats, smooth_height, 'max', 50,
                   symbol='H', color='black', transform=ccrs.PlateCarree())
plot_maxmin_points(ax, lons, lats, smooth_height, 'min', 25,
                   symbol='L', color='black', transform=ccrs.PlateCarree())

# Add titles
plt.title('GFS {}-mb Analysis of Geopotential Heights (m) and Temperature (°C)'.format(level),
          loc='left')
plt.title('VALID: ' + str(day_00Z) + ' ' + str(month_name_00Z) + ' ' + str(year_00Z) + ' ' + str(hour_00Z) + '00 UTC', loc='right')

plt.savefig('700mb_DIFAX_Chart_GFS_00Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 5 complete')


# Subset Observational Data
# -------------------------
#
# From the request above will give all levels from all radiosonde sites
# available through the service. For plotting a pressure surface map there
# is only need to have the data from that level. Below the data is subset
# and a few parameters set based on the level chosen. Additionally, the
# station information is obtained and latitude and longitude data is added
# to the DataFrame.
#

level = 850

if (level == 925) | (level == 850) | (level == 700):
    cint = 30
    def hght_format(v): return format(v, '.0f')[1:]
elif level == 500:
    cint = 60
    def hght_format(v): return format(v, '.0f')[:3]
elif level == 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[:3]
elif level < 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[1:4]

# Create subset of all data for a given level
data_subset = data_ua.pressure == level
df = data_ua[data_subset]

data_subset_00Z = data_ua_00Z.pressure == level
df_00Z = data_ua_00Z[data_subset_00Z]

# Given our data, add station information and drop any stations that are missing lat/lon. Then set
# 'station' as the index column so that operations between datasets will align on that column
#data = add_station_lat_lon(data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')
#prev_data = add_station_lat_lon(prev_data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')

# Create subset of all data for a given level
df = data_ua[data_ua.pressure == level].copy()
prev_df = prev_data_ua[prev_data_ua.pressure == level]

# Keep only the rows in prev_df where the 'station' value is also in df['station']
prev_df = prev_df[prev_df['station'].isin(df['station'])]

# Keep only the rows in df where the 'station' value is also in prev_df['station']
df = df[df['station'].isin(prev_df['station'])]

# Calculate the change on the aligned data frames--needs to be
# added to original dataframe to maintain consistent ordering
# with later use
df['height_change'] = df['height'].values - prev_df['height'].values

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_00Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'Temperature_isobaric')
query.vertical_level(85000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
temp_var = data_query.variables['Temperature_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
temp = temp_var[0, 0, :, :].squeeze()
smooth_tmpk = mpcalc.smooth_n_point(temp, 9, 5)
smooth_tmpc = smooth_tmpk - 273.15

# Create DIFAX Replication
# ------------------------
#
# Plot the observational data and contours on a Lambert Conformal map and
# add features that resemble the historic DIFAX maps.
#

# Set up map coordinate reference system
mapcrs = ccrs.LambertConformal(
    central_latitude=45, central_longitude=-100, standard_parallels=(30, 60))

# Start figure and set graphics extent
fig = plt.figure(1, figsize=(17, 15))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58])

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', zorder=1)
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white', edgecolor='grey', zorder=1)
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey', zorder=1)

# Plot plus signs every degree lat/lon
plus_lat = []
plus_lon = []
other_lat = []
other_lon = []

for x in range(0,360,1):
    for y in range(0,90,1):
        if (round(x) % 5 == 0) | (round(y) % 5 == 0):
            plus_lon.append(x)
            plus_lat.append(y)
        else:
            other_lon.append(x)
            other_lat.append(y)
ax.scatter(other_lon, other_lat, s=5, marker='o',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)
ax.scatter(plus_lon, plus_lat, s=30, marker='+',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)

# Add gridlines for every 5 degree lat/lon
gl = ax.gridlines(linestyle='solid', ylocs=range(15, 71, 5), xlocs=range(-160, -49, 5))
gl.zorder = -1

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 10 pt.
stationplot = StationPlot(ax, df['longitude'].values, df['latitude'].values, clip_on=True,
                          transform=ccrs.PlateCarree(), fontsize=10)

# Plot the temperature and dew point depression to the upper and lower left, respectively, of
# the center point.
stationplot.plot_parameter('NW', df['temperature'], color='black')
stationplot.plot_parameter('SW', df['temperature'] - df['dewpoint'], color='black')

# A more complex example uses a custom formatter to control how the geopotential height
# values are plotted. This is set in an earlier if-statement to work appropriate for
# different levels.
stationplot.plot_parameter('NE', df['height'], formatter=hght_format, color='black')

# Add wind barbs
stationplot.plot_barb(df['u_wind'] * MS_TO_KT, df['v_wind'] * MS_TO_KT, length=7, pivot='tip')

# Add height falls. This converts the change to decameters with /10, then
# formats with a +/- and 0 padding
def height_fall_formatter(v):
    return f'{int(v / 10):+03d}'

# Plot the parameter with an italic font
stationplot.plot_parameter('SE', df['height_change'], formatter=height_fall_formatter,
                           fontstyle='italic', color='black')

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 10 pt.
stationplot_00Z = StationPlot(ax, df_00Z['longitude'].values, df_00Z['latitude'].values, clip_on=True,
                          transform=ccrs.PlateCarree(), fontsize=10)

# Plot the temperature and dew point depression to the upper and lower left, respectively, of
# the center point.
stationplot_00Z.plot_parameter('NW', df_00Z['temperature'], color='black')
stationplot_00Z.plot_parameter('SW', df_00Z['temperature'] - df_00Z['dewpoint'], color='black')

# A more complex example uses a custom formatter to control how the geopotential height
# values are plotted. This is set in an earlier if-statement to work appropriate for
# different levels.
stationplot_00Z.plot_parameter('NE', df_00Z['height'], formatter=hght_format, color='black')

# Add wind barbs
stationplot_00Z.plot_barb(df_00Z['u_wind'] * MS_TO_KT, df_00Z['v_wind'] * MS_TO_KT, length=7, pivot='tip')

# Plot Solid Contours of Geopotential Height
cs = ax.contour(lons, lats, smooth_height,
                range(0, 20000, cint), colors='black', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot Dashed Contours of Temperature
cs2 = ax.contour(lons, lats, smooth_tmpc, range(-60, 51, 3),
                 colors='#c9514b', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs2, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Set longer dashes than default
try:
    for c in cs2.collections:
        c.set_dashes([(0, (5.0, 3.0))])
except AttributeError:
    for c in cs2.get_children():
        c.set_dashes([(0, (5.0, 3.0))])

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot filled circles for Radiosonde Obs
ax.scatter(df['longitude'].values, df['latitude'].values, s=12,
           marker='o', color='black', transform=ccrs.PlateCarree())
ax.scatter(df_00Z['longitude'].values, df_00Z['latitude'].values, s=12,
           marker='o', color='black', transform=ccrs.PlateCarree())

# Use definition to plot H/L symbols
plot_maxmin_points(ax, lons, lats, smooth_height, 'max', 50,
                   symbol='H', color='black', transform=ccrs.PlateCarree())
plot_maxmin_points(ax, lons, lats, smooth_height, 'min', 25,
                   symbol='L', color='black', transform=ccrs.PlateCarree())

# Add titles
plt.title('GFS {}-mb Analysis of Geopotential Heights (m) and Temperature (°C)'.format(level),
          loc='left')
plt.title('VALID: ' + str(day_00Z) + ' ' + str(month_name_00Z) + ' ' + str(year_00Z) + ' ' + str(hour_00Z) + '00 UTC', loc='right')

plt.savefig('850mb_DIFAX_Chart_GFS_00Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 6 complete')


################################################################################################################

# CHART 7: 500-mb Vorticity Chart (Analysis)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_00Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'u-component_of_wind_isobaric',
                'v-component_of_wind_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
u_wind_var = data_query.variables['u-component_of_wind_isobaric']
v_wind_var = data_query.variables['v-component_of_wind_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
height = mpcalc.smooth_n_point(height, 9, 5)
u_wind = u_wind_var[0, 0, :, :].squeeze() * units('m/s')
v_wind = v_wind_var[0, 0, :, :].squeeze() * units('m/s')

# Compute north-relative wind components for plotting purposes
#uwnd_er, vwnd_er = earth_relative_wind_components(u_wind, v_wind)

# Smooth wind components as desired
uwnd_500 = mpcalc.smooth_n_point(u_wind, 9, 5) * units('m/s')
vwnd_500 = mpcalc.smooth_n_point(v_wind, 9, 5) * units('m/s')

# Calculate grid spacing that is sign aware to use in absolute vorticity calculation
dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

# Calculate absolute vorticity from MetPy function
lats = lats.filled(np.nan)  # Replace mask with NaNs
avor_500 = mpcalc.absolute_vorticity(uwnd_500, vwnd_500, dx, dy, lats * units.degrees)

# Set up the projection that will be used for plotting
mapcrs = ccrs.LambertConformal(central_longitude=-100, central_latitude=35,
                               standard_parallels=(30, 60))

# Set up the projection of the data; if lat/lon then PlateCarree is what you want
datacrs = ccrs.PlateCarree()

# Start the figure and create plot axes with proper projection
fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add geopolitical boundaries for map reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))

# Absolute Vorticity contour levels
clevs_500_avor = np.arange(-4, 34, 2)

#Colors
colors = [
"#fe01fe",
"#be00fd",
"#7e00fe",
"#0000fe",
"#0158fe",
"#028bfe",
"#02befe",
"#00fefd",
"#03e4ca",
"#01cb7d",
"#00b300",
"#7ecb05",
"#cbe506",
"#fdfd00",
"#fdcb00",
"#fe9800",
"#fd6503",
"#fd0100"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#cc00cc")
colormap.set_over("#cc0100")
levels = np.array([-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=18)

# Plot absolute vorticity values (multiplying by 10^5 to scale appropriately)
cf = ax.contourf(lons, lats, avor_500*1e5, clevs_500_avor, norm=norm, cmap=colormap, extend='both',
                 transform=datacrs)
cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_label('Abs. Vorticity ($x 10^{-5} s^{-1}$)')

# Plot 500-hPa Geopotential Heights in decameters
hght_500_dam = height / 10
clevs_500_hght = np.arange(0, 8000, 6)
cs = ax.contour(lons, lats, hght_500_dam, clevs_500_hght, colors='white', transform=datacrs)
plt.clabel(cs, fmt='%d')

hght_500_data = hght_500_dam.magnitude
h_y, h_x = find_peaks(hght_500_data)
l_y, l_x = find_peaks(hght_500_data, maxima=False)

# Extract pressure values at those points
high_vals = hght_500_data[h_y, h_x].ravel()
low_vals = hght_500_data[l_y, l_x].ravel()

# Format labels as strings for plotting
high_labels = [f'{v:.0f}' for v in high_vals]
low_labels = [f'{v:.0f}' for v in low_vals]

text1 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], 'H', size=20, color='white',
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text2 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], high_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text3 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], 'L', size=20, color='white',
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text4 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], low_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text1.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text2.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text3.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text4.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])

# Plot two titles, one on right and left side
plt.title('GFS 500-mb Geopotential Heights (dam)'
          ' and Abs. Vorticity ($x 10^{-5} s^{-1}$)', loc='left')
plt.title('VALID: ' + str(day_00Z) + ' ' + str(month_name_00Z) + ' ' + str(year_00Z) + ' ' + str(hour_00Z) + '00 UTC', loc='right')

plt.savefig('500mb_Vorticity_Chart_GFS_00Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 7 complete')


################################################################################################################

# CHART 8: 850-mb Temperature Chart (Analysis)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_00Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'Temperature_isobaric')
query.vertical_level(85000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
temp_var = data_query.variables['Temperature_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
temp = temp_var[0, 0, :, :].squeeze()
smooth_tmpk = mpcalc.smooth_n_point(temp, 9, 5)
smooth_tmpc = smooth_tmpk - 273.15

# Set up the projection that will be used for plotting
mapcrs = ccrs.LambertConformal(central_longitude=-100, central_latitude=35,
                               standard_parallels=(30, 60))

# Set up the projection of the data; if lat/lon then PlateCarree is what you want
datacrs = ccrs.PlateCarree()

# Start the figure and create plot axes with proper projection
fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add geopolitical boundaries for map reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))

# Temperature contour levels
clevs_850_tmp = np.arange(-14, 32, 2)

#Colors
colors = [
"#fe01fe",
"#be00fd",
"#7e00fe",
"#0000fe",
"#0059fe",
"#008cfe",
"#03bffe",
"#00fefe",
"#00e5ca",
"#00cc7f",
"#02b202",
"#7fcb00",
"#cce505",
"#fefe00",
"#fecb02",
"#fe9800",
"#fe6600",
"#fe0000",
"#cc0100",
"#980001",
"#660000",
"#660066"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#cc00cc")
colormap.set_over("#980098")
levels = np.array([-14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=22)

# Plot 850-mb temperature values
cf = ax.contourf(lons, lats, smooth_tmpc, clevs_850_tmp, norm=norm, cmap=colormap, extend='both',
                 transform=datacrs)
cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_ticks(levels[::1])
cb.set_label('Temperature (°C)')

# Plot 850-hPa Geopotential Heights in decameters
hght_850_dam = smooth_height / 10
clevs_850_hght = np.arange(0, 8000, 3)
cs = ax.contour(lons, lats, hght_850_dam, clevs_850_hght, colors='white', transform=datacrs)
plt.clabel(cs, fmt='%d')

hght_850_data = hght_850_dam.magnitude
h_y, h_x = find_peaks(hght_850_data)
l_y, l_x = find_peaks(hght_850_data, maxima=False)

# Extract pressure values at those points
high_vals = hght_850_data[h_y, h_x].ravel()
low_vals = hght_850_data[l_y, l_x].ravel()

# Format labels as strings for plotting
high_labels = [f'{v:.0f}' for v in high_vals]
low_labels = [f'{v:.0f}' for v in low_vals]

text1 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], 'H', size=20, color='white',
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text2 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], high_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text3 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], 'L', size=20, color='white',
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text4 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], low_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())

text1.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text2.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text3.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text4.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])

# Plot two titles, one on right and left side
plt.title('GFS 850-mb Geopotential Heights (dam)'
          ' and Temperature (°C)', loc='left')
plt.title('VALID: ' + str(day_00Z) + ' ' + str(month_name_00Z) + ' ' + str(year_00Z) + ' ' + str(hour_00Z) + '00 UTC', loc='right')

plt.savefig('850mb_Temperature_Chart_GFS_00Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 8 complete')


################################################################################################################

# CHART 9: MSLP/500-mb Height Stack Chart (Analysis)

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_00Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Create lat/lon box for location you want to get data for
query_mslp = ncss.query()
query_mslp.lonlat_box(north=90, south=0, east=-30, west=-150)
query_mslp.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for MSLP
query_mslp.variables('Pressure_reduced_to_MSL_msl')
data_query_mslp = ncss.get_data(query_mslp)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
mslp_var = data_query_mslp.variables['Pressure_reduced_to_MSL_msl']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
mslp = mslp_var[0, :, :].squeeze()
smooth_mslp_Pa = mpcalc.smooth_n_point(mslp, 9, 5)
smooth_mslp_hPa = smooth_mslp_Pa / 100

# Create the Chart
# ------------
#
# Set projection of map display
mapproj = ccrs.LambertConformal(central_latitude=45., central_longitude=-100.)

# Set projection of data
dataproj = ccrs.PlateCarree()

fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapproj)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='0.25')
ax.add_feature(cfeature.LAND.with_scale('50m'), edgecolor='0.25')
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='0.25')

#Colors
colors = [
"#be00fd",
"#7e00fe",
"#0000fe",
"#0158fe",
#"#028bfe",
"#02befe",
"#00fefd",
#"#03e4ca",
"#01cb7d",
"#00b300",
#"#7ecb05",
"#cbe506",
"#fdfd00",
"#fdcb00",
"#fe9800",
"#fd6503",
"#fd0100",
"#980001"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#fe01fe")
colormap.set_over("#660066")
levels = np.array([504, 510, 516, 522, 528, 534, 540, 546, 552, 558, 564, 570, 576, 582, 588, 594])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=15)

# Plot 500-mb Heights
hght_500_dam = smooth_height / 10
clevs_500_hght = np.arange(504, 600, 6)
cf = ax.contourf(lons, lats, hght_500_dam, clevs_500_hght, norm=norm, cmap=colormap, extend='both', transform=dataproj)

cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_ticks(levels[::1])

# Plot MSLP
clevmslp = np.arange(800., 1120., 4)
cs = ax.contour(lons, lats, smooth_mslp_hPa, clevmslp, colors='k',
                 linestyles='solid', transform=dataproj)
kw_clabels = {'fontsize': 11, 'inline': True, 'inline_spacing': 5, 'fmt': '%i',
              'rightside_up': True, 'use_clabeltext': True}
plt.clabel(cs, **kw_clabels, zorder=1)

# Use definition to plot H/L symbols
# Use smoothed MSLP directly (in hPa) to find local maxima/minima
# Remove units before processing
mslp_data = smooth_mslp_hPa.magnitude
h_y, h_x = find_peaks(mslp_data)
l_y, l_x = find_peaks(mslp_data, maxima=False)

# Extract pressure values at those points
high_vals = mslp_data[h_y, h_x].ravel()
low_vals = mslp_data[l_y, l_x].ravel()

# Format labels as strings for plotting
high_labels = [f'{v:.0f}' for v in high_vals]
low_labels = [f'{v:.0f}' for v in low_vals]

text1 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], 'H', size=20, color='white',
            fontweight='bold', clip_on=True, transform=ccrs.PlateCarree(), zorder=5)
text2 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], high_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, transform=ccrs.PlateCarree(), zorder=5)
text3 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], 'L', size=20, color='white',
            fontweight='bold', clip_on=True, transform=ccrs.PlateCarree(), zorder=5)
text4 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], low_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, transform=ccrs.PlateCarree(), zorder=5)

text1.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text2.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text3.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text4.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])

# Put on some titles
plt.title('GFS MSLP (mb) and 500-mb Geopotential Heights (dam)', loc='left')
plt.title('VALID: ' + str(day_00Z) + ' ' + str(month_name_00Z) + ' ' + str(year_00Z) + ' ' + str(hour_00Z) + '00 UTC', loc='right')

plt.savefig('Stack_Chart_GFS_00Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 9 complete')


################################################################################################################

# CHART 10: MSLP/1000–500 mb Thickness Chart (Analysis)

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_00Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query_500hPa = ncss.query()
query_500hPa.lonlat_box(north=90, south=0, east=-30, west=-150)
query_500hPa.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query_500hPa.variables('Geopotential_height_isobaric')
query_500hPa.vertical_level(50000)
data_query_500hPa = ncss.get_data(query_500hPa)

# Create lat/lon box for location you want to get data for
query_1000hPa = ncss.query()
query_1000hPa.lonlat_box(north=90, south=0, east=-30, west=-150)
query_1000hPa.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for geopotential height
query_1000hPa.variables('Geopotential_height_isobaric')
query_1000hPa.vertical_level(100000)
data_query_1000hPa = ncss.get_data(query_1000hPa)

# Create lat/lon box for location you want to get data for
query_mslp = ncss.query()
query_mslp.lonlat_box(north=90, south=0, east=-30, west=-150)
query_mslp.time(datetime(year_00Z, month_00Z, day_00Z, int(hour_00Z)))

# Request data for MSLP
query_mslp.variables('Pressure_reduced_to_MSL_msl')
data_query_mslp = ncss.get_data(query_mslp)

# Pull out variables you want to use
height_500hPa_var = data_query_500hPa.variables['Geopotential_height_isobaric']
height_1000hPa_var = data_query_1000hPa.variables['Geopotential_height_isobaric']
mslp_var = data_query_mslp.variables['Pressure_reduced_to_MSL_msl']

dlat = data_query_500hPa.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query_500hPa.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query_500hPa.variables[dlat][:]
lon = data_query_500hPa.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height500 = height_500hPa_var[0, 0, :, :].squeeze()
smooth_height500 = mpcalc.smooth_n_point(height500, 9, 5)
height1000 = height_1000hPa_var[0, 0, :, :].squeeze()
smooth_height1000 = mpcalc.smooth_n_point(height1000, 9, 5)
mslp = mslp_var[0, :, :].squeeze()
smooth_mslp_Pa = mpcalc.smooth_n_point(mslp, 9, 5)
smooth_mslp_hPa = smooth_mslp_Pa / 100

thickness_1000_500 = smooth_height500 - smooth_height1000

# Create the Chart
# ------------
#
# Set projection of map display
mapproj = ccrs.LambertConformal(central_latitude=45., central_longitude=-100.)

# Set projection of data
dataproj = ccrs.PlateCarree()

fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapproj)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', zorder=1)
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white', edgecolor='grey', zorder=1)
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey', zorder=1)

# Plot 1000-500 mb thickness with multiple colors
clevs = (np.arange(0, 5400, 60),
         np.array([5400]),
         np.arange(5460, 7000, 60))
colors = ('tab:blue', 'b', 'tab:red')
kw_clabels = {'fontsize': 11, 'inline': True, 'inline_spacing': 5, 'fmt': '%i',
              'rightside_up': True, 'use_clabeltext': True}
for clevthick, color in zip(clevs, colors):
    cs = ax.contour(lons, lats, thickness_1000_500, levels=clevthick, colors=color,
                    linewidths=1.25, linestyles='dashed', transform=dataproj)
    plt.clabel(cs, **kw_clabels)

# Plot MSLP
clevmslp = np.arange(800., 1120., 4)
cs2 = ax.contour(lons, lats, smooth_mslp_hPa, clevmslp, colors='k', linewidths=1.5,
                 linestyles='solid', transform=dataproj)
plt.clabel(cs2, **kw_clabels)

# Use definition to plot H/L symbols
# Use smoothed MSLP directly (in hPa) to find local maxima/minima
# Remove units before processing

# Plot MSLP high and low centers
mslp_data = smooth_mslp_hPa.magnitude

plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'max', 50,
    symbol='H',
    color='blue',
    transform=ccrs.PlateCarree()
)

plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'min', 25,
    symbol='L',
    color='red',
    transform=ccrs.PlateCarree()
)

# Put on some titles
plt.title('GFS MSLP (mb) and 1000–500 mb Thickness (m)', loc='left')
plt.title('VALID: ' + str(day_00Z) + ' ' + str(month_name_00Z) + ' ' + str(year_00Z) + ' ' + str(hour_00Z) + '00 UTC', loc='right')

plt.savefig('MSLP_1000-500mb_Thickness_Chart_GFS_00Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 10 complete')


################################################################################################################
# ANALYSIS CHARTS (12Z)
################################################################################################################

# CHART 11: 250-mb Jet Streak Chart (Analysis)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_12Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'u-component_of_wind_isobaric',
                'v-component_of_wind_isobaric')
query.vertical_level(25000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
u_wind_var = data_query.variables['u-component_of_wind_isobaric']
v_wind_var = data_query.variables['v-component_of_wind_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

# Get actual data values and remove any size 1 dimensions
height = height_var[0, 0, :, :].squeeze()
u_wind = u_wind_var[0, 0, :, :].squeeze() * units('m/s')
v_wind = v_wind_var[0, 0, :, :].squeeze() * units('m/s')
u_var = u_wind_var[0, 0, :, :].squeeze().data * units('m/s')
v_var = v_wind_var[0, 0, :, :].squeeze().data * units('m/s')
u_knots = u_var.to('knots')
v_knots = v_var.to('knots')

# Smooth height data
height = mpcalc.smooth_n_point(height, 9, 5)

# Calculate the 250-mb wind speed
u250 = (units(data_query.variables['u-component_of_wind_isobaric'].units) *
        data_query.variables['u-component_of_wind_isobaric'][0, 0, :, :])
v250 = (units(data_query.variables['v-component_of_wind_isobaric'].units) *
        data_query.variables['v-component_of_wind_isobaric'][0, 0, :, :])
wspd250 = mpcalc.wind_speed(u250, v250).to('knots')
wspd250 = mpcalc.smooth_n_point(wspd250, 9, 5)

# Create new figure
fig = plt.figure(figsize=(14, 12))

# Add the map and set the extent
ax = plt.axes(projection=ccrs.LambertConformal(central_latitude=35., central_longitude=-100., standard_parallels=(30, 60)))
ax.set_extent([-128.5, -73.2, 18.5, 58])
#ax.background_patch.set_fill(False)

# Add state boundaries to plot
ax.add_feature(cfeature.BORDERS, edgecolor='brown', linewidth=1)
ax.add_feature(cfeature.COASTLINE, edgecolor='brown', linewidth=1)
ax.add_feature(cfeature.STATES, edgecolor='brown', linewidth=1)

# Contour the heights
height_dam = height / 10
contours = np.arange(0, 1200, 12)
# Because we have a very local graphics area, the contours have joints
# to smooth those out we can use `ndimage.zoom`
#zoom_500 = ndimage.zoom(height, 5)
#zlon = ndimage.zoom(lon, 5)
#zlat = ndimage.zoom(lat, 5)
c = ax.contour(lons, lats, height_dam, levels=contours,
               colors='black', linewidths=2, transform=ccrs.PlateCarree())
labels = ax.clabel(c, fontsize=10, inline=1, inline_spacing=3, fmt='%i')
for label in labels:
    label.set_bbox(dict(facecolor='white', edgecolor='black', boxstyle='square, pad=0.2'))

# Set up parameters for quiver plot. The slices below are used to subset the data (here
# taking every 4th point in x and y). The quiver_kwargs are parameters to control the
# appearance of the quiver so that they stay consistent between the calls.
barbs_slices = (slice(None, None, 5), slice(None, None, 5))

# Plot the 250-mb wind speed as contour fill
# Create colorbar and set properties
colors = [
"#00eeee",
"#1e90ff",
"#912cee",
"#ff00ff",
"#ff0000",
"#ffff00"
]
colormap = matplotlib.colors.ListedColormap(colors)
levels = np.array([70, 90, 110, 130, 150, 200, 250])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=6)
cf = ax.contourf(lons, lats, wspd250, levels, norm=norm, cmap=colormap, transform=ccrs.PlateCarree())
cbar = plt.colorbar(cf, ticks=levels, orientation='horizontal', pad=0.02, shrink=0.5)

# Plot the wind vectors
wind = ax.barbs(lons[barbs_slices], lats[barbs_slices],
                 u_knots[barbs_slices], v_knots[barbs_slices],
                 color='darkblue', length=6, zorder=2, transform=ccrs.PlateCarree())

# Add a title to the plot
plt.title('GFS 250-mb Geopotential Heights (dam) and Wind Speed (knots)\n'
          'VALID: ' + str(day_12Z) + ' ' + str(month_name_12Z) + ' ' + str(year_12Z) + ' ' + str(hour_12Z) + '00 UTC',
          color='black', size=14)

plt.savefig('250mb_Jet_Streak_Chart_GFS_12Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 11 complete')


################################################################################################################

# CHART 12: 200-mb Jet Streak Chart (Analysis)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_12Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'u-component_of_wind_isobaric',
                'v-component_of_wind_isobaric')
query.vertical_level(20000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
u_wind_var = data_query.variables['u-component_of_wind_isobaric']
v_wind_var = data_query.variables['v-component_of_wind_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

# Get actual data values and remove any size 1 dimensions
height = height_var[0, 0, :, :].squeeze()
u_wind = u_wind_var[0, 0, :, :].squeeze() * units('m/s')
v_wind = v_wind_var[0, 0, :, :].squeeze() * units('m/s')
u_var = u_wind_var[0, 0, :, :].squeeze().data * units('m/s')
v_var = v_wind_var[0, 0, :, :].squeeze().data * units('m/s')
u_knots = u_var.to('knots')
v_knots = v_var.to('knots')

# Smooth height data
height = mpcalc.smooth_n_point(height, 9, 5)

# Calculate the 200-mb wind speed
u200 = (units(data_query.variables['u-component_of_wind_isobaric'].units) *
        data_query.variables['u-component_of_wind_isobaric'][0, 0, :, :])
v200 = (units(data_query.variables['v-component_of_wind_isobaric'].units) *
        data_query.variables['v-component_of_wind_isobaric'][0, 0, :, :])
wspd200 = mpcalc.wind_speed(u200, v200).to('knots')
wspd200 = mpcalc.smooth_n_point(wspd200, 9, 5)

# Create new figure
fig = plt.figure(figsize=(14, 12))

# Add the map and set the extent
ax = plt.axes(projection=ccrs.LambertConformal(central_latitude=35., central_longitude=-100., standard_parallels=(30, 60)))
ax.set_extent([-128.5, -73.2, 18.5, 58])
#ax.background_patch.set_fill(False)

# Add state boundaries to plot
ax.add_feature(cfeature.BORDERS, edgecolor='brown', linewidth=1)
ax.add_feature(cfeature.COASTLINE, edgecolor='brown', linewidth=1)
ax.add_feature(cfeature.STATES, edgecolor='brown', linewidth=1)

# Contour the heights
height_dam = height / 10
contours = np.arange(0, 1300, 12)
# Because we have a very local graphics area, the contours have joints
# to smooth those out we can use `ndimage.zoom`
#zoom_500 = ndimage.zoom(height, 5)
#zlon = ndimage.zoom(lon, 5)
#zlat = ndimage.zoom(lat, 5)
c = ax.contour(lons, lats, height_dam, levels=contours,
               colors='black', linewidths=2, transform=ccrs.PlateCarree())
labels = ax.clabel(c, fontsize=10, inline=1, inline_spacing=3, fmt='%i')
for label in labels:
    label.set_bbox(dict(facecolor='white', edgecolor='black', boxstyle='square, pad=0.2'))

# Set up parameters for quiver plot. The slices below are used to subset the data (here
# taking every 4th point in x and y). The quiver_kwargs are parameters to control the
# appearance of the quiver so that they stay consistent between the calls.
barbs_slices = (slice(None, None, 5), slice(None, None, 5))

# Plot the 250-mb wind speed as contour fill
# Create colorbar and set properties
colors = [
"#00eeee",
"#1e90ff",
"#912cee",
"#ff00ff",
"#ff0000",
"#ffff00"
]
colormap = matplotlib.colors.ListedColormap(colors)
levels = np.array([70, 90, 110, 130, 150, 200, 250])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=6)
cf = ax.contourf(lons, lats, wspd200, levels, norm=norm, cmap=colormap, transform=ccrs.PlateCarree())
cbar = plt.colorbar(cf, ticks=levels, orientation='horizontal', pad=0.02, shrink=0.5)

# Plot the wind vectors
wind = ax.barbs(lons[barbs_slices], lats[barbs_slices],
                 u_knots[barbs_slices], v_knots[barbs_slices],
                 color='darkblue', length=6, zorder=2, transform=ccrs.PlateCarree())

# Add a title to the plot
plt.title('GFS 200-mb Geopotential Heights (dam) and Wind Speed (knots)\n'
          'VALID: ' + str(day_12Z) + ' ' + str(month_name_12Z) + ' ' + str(year_12Z) + ' ' + str(hour_12Z) + '00 UTC',
          color='black', size=14)

plt.savefig('200mb_Jet_Streak_Chart_GFS_12Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 12 complete')


################################################################################################################

# CHARTS 13–15: DIFAX Replicas at the 500-mb, 700-mb, and 850-mb Levels (Analysis)

# Plotting High/Low Symbols
# -------------------------
#
# A helper function to plot a text symbol (e.g., H, L) for relative
# maximum/minimum for a given field (e.g., geopotential height).
#


def plot_maxmin_points(ax, lon, lat, data, extrema, nsize, symbol,
                       color='k', plot_value=True,
                       transform=ccrs.PlateCarree(),
                       edge_buffer=0.02,
                       min_dist=10e5):
    """
    Plot local maxima or minima on a map using a given symbol, with optional buffer
    to avoid plotting right at map edges and thinning of clustered points.

    Parameters
    ----------
    min_dist : float
        Minimum distance (in map projection units) to separate points when thinning.
    """
    import numpy as np
    from scipy.ndimage import maximum_filter, minimum_filter

    # Find local maxima/minima
    if extrema == 'max':
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif extrema == 'min':
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError("extrema must be 'max' or 'min'")

    # Ensure lon/lat are 2D
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    # Indices of extrema points
    mxx, mxy = np.where(data_ext == data)

    # Stack lon/lat and transform to map projection
    points = np.stack([lon[mxx, mxy], lat[mxx, mxy]], axis=1)
    proj_points = ax.projection.transform_points(transform,
                                                 points[:, 0], points[:, 1])

    # Get map extent in projected coordinates
    x0, x1, y0, y1 = ax.get_extent()

    # Apply edge buffer
    dx = (x1 - x0) * edge_buffer
    dy = (y1 - y0) * edge_buffer
    x0_buff, x1_buff = x0 + dx, x1 - dx
    y0_buff, y1_buff = y0 + dy, y1 - dy

    # Filter points to only those inside buffered map
    inside = (proj_points[:, 0] >= x0_buff) & (proj_points[:, 0] <= x1_buff) & \
             (proj_points[:, 1] >= y0_buff) & (proj_points[:, 1] <= y1_buff)

    lon_keep = points[inside, 0]
    lat_keep = points[inside, 1]
    data_keep = data[mxx, mxy][inside]
    proj_keep = proj_points[inside, :2]

    # Thinning: keep points separated by at least min_dist
    selected_idx = []
    for i, p in enumerate(proj_keep):
        if len(selected_idx) == 0:
            selected_idx.append(i)
            continue
        dists = np.linalg.norm(proj_keep[selected_idx] - p, axis=1)
        if np.all(dists >= min_dist):
            selected_idx.append(i)

    # Keep only thinned points
    lon_keep = lon_keep[selected_idx]
    lat_keep = lat_keep[selected_idx]
    data_keep = data_keep[selected_idx]

    # Plot symbols and optional values
    for x, y, val in zip(lon_keep, lat_keep, data_keep):
        ax.text(
            x, y, symbol,
            size=36,
            color=color,
            ha='center', va='center',
            transform=transform,
            clip_on=True,
            zorder=10
        )
        if plot_value:
            ax.text(
                x, y - 1,
                f"{int(val)}",
                size=12,
                fontweight='bold',
                color=color,
                ha='center', va='top',
                transform=transform,
                clip_on=True,
                zorder=10
            )

# Observation Data
# ----------------
#
# Set a date and time for upper-air observations (should only be 00 or 12
# UTC for the hour).
#
# Request all data from Iowa State using the Siphon package. The result is
# a pandas DataFrame containing all of the sounding data from all
# available stations.
#

# Set date for desired UPA data
# today = datetime.utcnow()
date = datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z))

# Previous is 12 hours before that time
prev = date - timedelta(hours=12)

# Request data using Siphon request for data from Iowa State Archive
#data = IAStateUpperAir.request_all_data(date)
#prev_data = IAStateUpperAir.request_all_data(prev)

data_ua = pd.DataFrame()
prev_data_ua = pd.DataFrame()
upper_air_stations = ['BMX','FFC','EYW','MFL','TBW','JAX','CHS','MHX','GSO','RNK','IAD','WAL','OKX','GYX','CAR','ALB','BUF','PIT','ILN','DTX','APX','BNA','JAN','LIX','SHV','LCH','BRO','CRP','FWD','MAF','AMA','OUN','DDC','TOP','LZK','SGF','ILX','DVN','GRB','MPX','INL','BIS','ABR','UNR','LBF','OAX','EPZ','ABQ','GJT','RIW','TFX','BOI','SLC','FGZ','VEF','LKN','REV','OTX','UIL','MFR','OAK','ANT','YZT','YYE','ZXS','WVK','WSE','YQD','YYQ','YPL','WMW','YAH','YVP','YZV','YYR','YQI','AWE','YJT','GGW','VBG','SLE','DRT','MCU','MLP','TUS','GYM','YSM']
for station in range(0,len(upper_air_stations)):
    try:
        data_0 = WyomingUpperAir.request_data(date, upper_air_stations[station])
        prev_data_0 = WyomingUpperAir.request_data(prev, upper_air_stations[station])
        data_ua = pd.concat([data_ua, data_0], ignore_index=True)
        prev_data_ua = pd.concat([prev_data_ua, prev_data_0], ignore_index=True)
        print(station)
    except urllib.error.HTTPError:
        print(upper_air_stations[station])
        pass
    except ValueError:
        print(upper_air_stations[station])
        pass

# Subset Observational Data
# -------------------------
#
# From the request above will give all levels from all radiosonde sites
# available through the service. For plotting a pressure surface map there
# is only need to have the data from that level. Below the data is subset
# and a few parameters set based on the level chosen. Additionally, the
# station information is obtained and latitude and longitude data is added
# to the DataFrame.
#

level = 500

if (level == 925) | (level == 850) | (level == 700):
    cint = 30
    def hght_format(v): return format(v, '.0f')[1:]
elif level == 500:
    cint = 60
    def hght_format(v): return format(v, '.0f')[:3]
elif level == 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[:3]
elif level < 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[1:4]

# Create subset of all data for a given level
data_subset = data_ua.pressure == level
df = data_ua[data_subset]

# Given our data, add station information and drop any stations that are missing lat/lon. Then set
# 'station' as the index column so that operations between datasets will align on that column
#data = add_station_lat_lon(data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')
#prev_data = add_station_lat_lon(prev_data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')

# Create subset of all data for a given level
df = data_ua[data_ua.pressure == level].copy()
prev_df = prev_data_ua[prev_data_ua.pressure == level]

# Keep only the rows in prev_df where the 'station' value is also in df['station']
prev_df = prev_df[prev_df['station'].isin(df['station'])]

# Keep only the rows in df where the 'station' value is also in prev_df['station']
df = df[df['station'].isin(prev_df['station'])]

# Calculate the change on the aligned data frames--needs to be
# added to original dataframe to maintain consistent ordering
# with later use
df['height_change'] = df['height'].values - prev_df['height'].values

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_12Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'Temperature_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
temp_var = data_query.variables['Temperature_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
temp = temp_var[0, 0, :, :].squeeze()
smooth_tmpk = mpcalc.smooth_n_point(temp, 9, 5)
smooth_tmpc = smooth_tmpk - 273.15

# Create DIFAX Replication
# ------------------------
#
# Plot the observational data and contours on a Lambert Conformal map and
# add features that resemble the historic DIFAX maps.
#

# Set up map coordinate reference system
mapcrs = ccrs.LambertConformal(
    central_latitude=45, central_longitude=-100, standard_parallels=(30, 60))

# Start figure and set graphics extent
fig = plt.figure(1, figsize=(17, 15))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58])

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', zorder=1)
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white', edgecolor='grey', zorder=1)
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey', zorder=1)

# Plot plus signs every degree lat/lon
plus_lat = []
plus_lon = []
other_lat = []
other_lon = []

for x in range(0,360,1):
    for y in range(0,90,1):
        if (round(x) % 5 == 0) | (round(y) % 5 == 0):
            plus_lon.append(x)
            plus_lat.append(y)
        else:
            other_lon.append(x)
            other_lat.append(y)
ax.scatter(other_lon, other_lat, s=5, marker='o',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)
ax.scatter(plus_lon, plus_lat, s=30, marker='+',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)

# Add gridlines for every 5 degree lat/lon
gl = ax.gridlines(linestyle='solid', ylocs=range(15, 71, 5), xlocs=range(-160, -49, 5))
gl.zorder = -1

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 10 pt.
stationplot = StationPlot(ax, df['longitude'].values, df['latitude'].values, clip_on=True,
                          transform=ccrs.PlateCarree(), fontsize=10)

# Plot the temperature and dew point depression to the upper and lower left, respectively, of
# the center point.
stationplot.plot_parameter('NW', df['temperature'], color='black')
stationplot.plot_parameter('SW', df['temperature'] - df['dewpoint'], color='black')

# A more complex example uses a custom formatter to control how the geopotential height
# values are plotted. This is set in an earlier if-statement to work appropriate for
# different levels.
stationplot.plot_parameter('NE', df['height'], formatter=hght_format, color='black')

# Add wind barbs
stationplot.plot_barb(df['u_wind'] * MS_TO_KT, df['v_wind'] * MS_TO_KT, length=7, pivot='tip')

# Add height falls. This converts the change to decameters with /10, then
# formats with a +/- and 0 padding
def height_fall_formatter(v):
    return f'{int(v / 10):+03d}'

# Plot the parameter with an italic font
stationplot.plot_parameter('SE', df['height_change'], formatter=height_fall_formatter,
                           fontstyle='italic', color='black')

# Plot Solid Contours of Geopotential Height
cs = ax.contour(lons, lats, smooth_height,
                range(0, 20000, cint), colors='black', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot Dashed Contours of Temperature
cs2 = ax.contour(lons, lats, smooth_tmpc, range(-60, 51, 3),
                 colors='#c9514b', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs2, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Set longer dashes than default
try:
    for c in cs2.collections:
        c.set_dashes([(0, (5.0, 3.0))])
except AttributeError:
    for c in cs2.get_children():
        c.set_dashes([(0, (5.0, 3.0))])

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot filled circles for Radiosonde Obs
ax.scatter(df['longitude'].values, df['latitude'].values, s=12,
           marker='o', color='black', transform=ccrs.PlateCarree())

# Use definition to plot H/L symbols
plot_maxmin_points(ax, lons, lats, smooth_height, 'max', 50,
                   symbol='H', color='black', transform=ccrs.PlateCarree())
plot_maxmin_points(ax, lons, lats, smooth_height, 'min', 25,
                   symbol='L', color='black', transform=ccrs.PlateCarree())

# Add titles
plt.title('GFS {}-mb Analysis of Geopotential Heights (m) and Temperature (°C)'.format(level),
          loc='left')
plt.title('VALID: ' + str(day_12Z) + ' ' + str(month_name_12Z) + ' ' + str(year_12Z) + ' ' + str(hour_12Z) + '00 UTC', loc='right')

plt.savefig('500mb_DIFAX_Chart_GFS_12Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 13 complete')


# Subset Observational Data
# -------------------------
#
# From the request above will give all levels from all radiosonde sites
# available through the service. For plotting a pressure surface map there
# is only need to have the data from that level. Below the data is subset
# and a few parameters set based on the level chosen. Additionally, the
# station information is obtained and latitude and longitude data is added
# to the DataFrame.
#

level = 700

if (level == 925) | (level == 850) | (level == 700):
    cint = 30
    def hght_format(v): return format(v, '.0f')[1:]
elif level == 500:
    cint = 60
    def hght_format(v): return format(v, '.0f')[:3]
elif level == 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[:3]
elif level < 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[1:4]

# Create subset of all data for a given level
data_subset = data_ua.pressure == level
df = data_ua[data_subset]

# Given our data, add station information and drop any stations that are missing lat/lon. Then set
# 'station' as the index column so that operations between datasets will align on that column
#data = add_station_lat_lon(data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')
#prev_data = add_station_lat_lon(prev_data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')

# Create subset of all data for a given level
df = data_ua[data_ua.pressure == level].copy()
prev_df = prev_data_ua[prev_data_ua.pressure == level]

# Keep only the rows in prev_df where the 'station' value is also in df['station']
prev_df = prev_df[prev_df['station'].isin(df['station'])]

# Keep only the rows in df where the 'station' value is also in prev_df['station']
df = df[df['station'].isin(prev_df['station'])]

# Calculate the change on the aligned data frames--needs to be
# added to original dataframe to maintain consistent ordering
# with later use
df['height_change'] = df['height'].values - prev_df['height'].values

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_12Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'Temperature_isobaric')
query.vertical_level(70000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
temp_var = data_query.variables['Temperature_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
temp = temp_var[0, 0, :, :].squeeze()
smooth_tmpk = mpcalc.smooth_n_point(temp, 9, 5)
smooth_tmpc = smooth_tmpk - 273.15

# Create DIFAX Replication
# ------------------------
#
# Plot the observational data and contours on a Lambert Conformal map and
# add features that resemble the historic DIFAX maps.
#

# Set up map coordinate reference system
mapcrs = ccrs.LambertConformal(
    central_latitude=45, central_longitude=-100, standard_parallels=(30, 60))

# Start figure and set graphics extent
fig = plt.figure(1, figsize=(17, 15))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58])

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', zorder=1)
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white', edgecolor='grey', zorder=1)
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey', zorder=1)

# Plot plus signs every degree lat/lon
plus_lat = []
plus_lon = []
other_lat = []
other_lon = []

for x in range(0,360,1):
    for y in range(0,90,1):
        if (round(x) % 5 == 0) | (round(y) % 5 == 0):
            plus_lon.append(x)
            plus_lat.append(y)
        else:
            other_lon.append(x)
            other_lat.append(y)
ax.scatter(other_lon, other_lat, s=5, marker='o',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)
ax.scatter(plus_lon, plus_lat, s=30, marker='+',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)

# Add gridlines for every 5 degree lat/lon
gl = ax.gridlines(linestyle='solid', ylocs=range(15, 71, 5), xlocs=range(-160, -49, 5))
gl.zorder = -1

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 10 pt.
stationplot = StationPlot(ax, df['longitude'].values, df['latitude'].values, clip_on=True,
                          transform=ccrs.PlateCarree(), fontsize=10)

# Plot the temperature and dew point depression to the upper and lower left, respectively, of
# the center point.
stationplot.plot_parameter('NW', df['temperature'], color='black')
stationplot.plot_parameter('SW', df['temperature'] - df['dewpoint'], color='black')

# A more complex example uses a custom formatter to control how the geopotential height
# values are plotted. This is set in an earlier if-statement to work appropriate for
# different levels.
stationplot.plot_parameter('NE', df['height'], formatter=hght_format, color='black')

# Add wind barbs
stationplot.plot_barb(df['u_wind'] * MS_TO_KT, df['v_wind'] * MS_TO_KT, length=7, pivot='tip')

# Add height falls. This converts the change to decameters with /10, then
# formats with a +/- and 0 padding
def height_fall_formatter(v):
    return f'{int(v / 10):+03d}'

# Plot the parameter with an italic font
stationplot.plot_parameter('SE', df['height_change'], formatter=height_fall_formatter,
                           fontstyle='italic', color='black')

# Plot Solid Contours of Geopotential Height
cs = ax.contour(lons, lats, smooth_height,
                range(0, 20000, cint), colors='black', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot Dashed Contours of Temperature
cs2 = ax.contour(lons, lats, smooth_tmpc, range(-60, 51, 3),
                 colors='#c9514b', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs2, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Set longer dashes than default
try:
    for c in cs2.collections:
        c.set_dashes([(0, (5.0, 3.0))])
except AttributeError:
    for c in cs2.get_children():
        c.set_dashes([(0, (5.0, 3.0))])

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot filled circles for Radiosonde Obs
ax.scatter(df['longitude'].values, df['latitude'].values, s=12,
           marker='o', color='black', transform=ccrs.PlateCarree())

# Use definition to plot H/L symbols
plot_maxmin_points(ax, lons, lats, smooth_height, 'max', 50,
                   symbol='H', color='black', transform=ccrs.PlateCarree())
plot_maxmin_points(ax, lons, lats, smooth_height, 'min', 25,
                   symbol='L', color='black', transform=ccrs.PlateCarree())

# Add titles
plt.title('GFS {}-mb Analysis of Geopotential Heights (m) and Temperature (°C)'.format(level),
          loc='left')
plt.title('VALID: ' + str(day_12Z) + ' ' + str(month_name_12Z) + ' ' + str(year_12Z) + ' ' + str(hour_12Z) + '00 UTC', loc='right')

plt.savefig('700mb_DIFAX_Chart_GFS_12Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 14 complete')


# Subset Observational Data
# -------------------------
#
# From the request above will give all levels from all radiosonde sites
# available through the service. For plotting a pressure surface map there
# is only need to have the data from that level. Below the data is subset
# and a few parameters set based on the level chosen. Additionally, the
# station information is obtained and latitude and longitude data is added
# to the DataFrame.
#

level = 850

if (level == 925) | (level == 850) | (level == 700):
    cint = 30
    def hght_format(v): return format(v, '.0f')[1:]
elif level == 500:
    cint = 60
    def hght_format(v): return format(v, '.0f')[:3]
elif level == 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[:3]
elif level < 300:
    cint = 120
    def hght_format(v): return format(v, '.0f')[1:4]

# Create subset of all data for a given level
data_subset = data_ua.pressure == level
df = data_ua[data_subset]

# Given our data, add station information and drop any stations that are missing lat/lon. Then set
# 'station' as the index column so that operations between datasets will align on that column
#data = add_station_lat_lon(data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')
#prev_data = add_station_lat_lon(prev_data).dropna(how='any', subset=('longitude', 'latitude')).set_index('station')

# Create subset of all data for a given level
df = data_ua[data_ua.pressure == level].copy()
prev_df = prev_data_ua[prev_data_ua.pressure == level]

# Keep only the rows in prev_df where the 'station' value is also in df['station']
prev_df = prev_df[prev_df['station'].isin(df['station'])]

# Keep only the rows in df where the 'station' value is also in prev_df['station']
df = df[df['station'].isin(prev_df['station'])]

# Calculate the change on the aligned data frames--needs to be
# added to original dataframe to maintain consistent ordering
# with later use
df['height_change'] = df['height'].values - prev_df['height'].values

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_12Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'Temperature_isobaric')
query.vertical_level(85000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
temp_var = data_query.variables['Temperature_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
temp = temp_var[0, 0, :, :].squeeze()
smooth_tmpk = mpcalc.smooth_n_point(temp, 9, 5)
smooth_tmpc = smooth_tmpk - 273.15

# Create DIFAX Replication
# ------------------------
#
# Plot the observational data and contours on a Lambert Conformal map and
# add features that resemble the historic DIFAX maps.
#

# Set up map coordinate reference system
mapcrs = ccrs.LambertConformal(
    central_latitude=45, central_longitude=-100, standard_parallels=(30, 60))

# Start figure and set graphics extent
fig = plt.figure(1, figsize=(17, 15))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58])

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', zorder=1)
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white', edgecolor='grey', zorder=1)
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey', zorder=1)

# Plot plus signs every degree lat/lon
plus_lat = []
plus_lon = []
other_lat = []
other_lon = []

for x in range(0,360,1):
    for y in range(0,90,1):
        if (round(x) % 5 == 0) | (round(y) % 5 == 0):
            plus_lon.append(x)
            plus_lat.append(y)
        else:
            other_lon.append(x)
            other_lat.append(y)
ax.scatter(other_lon, other_lat, s=5, marker='o',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)
ax.scatter(plus_lon, plus_lat, s=30, marker='+',
           transform=ccrs.PlateCarree(), color='lightgrey', zorder=-1)

# Add gridlines for every 5 degree lat/lon
gl = ax.gridlines(linestyle='solid', ylocs=range(15, 71, 5), xlocs=range(-160, -49, 5))
gl.zorder = -1

# Start the station plot by specifying the axes to draw on, as well as the
# lon/lat of the stations (with transform). We also the fontsize to 10 pt.
stationplot = StationPlot(ax, df['longitude'].values, df['latitude'].values, clip_on=True,
                          transform=ccrs.PlateCarree(), fontsize=10)

# Plot the temperature and dew point depression to the upper and lower left, respectively, of
# the center point.
stationplot.plot_parameter('NW', df['temperature'], color='black')
stationplot.plot_parameter('SW', df['temperature'] - df['dewpoint'], color='black')

# A more complex example uses a custom formatter to control how the geopotential height
# values are plotted. This is set in an earlier if-statement to work appropriate for
# different levels.
stationplot.plot_parameter('NE', df['height'], formatter=hght_format, color='black')

# Add wind barbs
stationplot.plot_barb(df['u_wind'] * MS_TO_KT, df['v_wind'] * MS_TO_KT, length=7, pivot='tip')

# Add height falls. This converts the change to decameters with /10, then
# formats with a +/- and 0 padding
def height_fall_formatter(v):
    return f'{int(v / 10):+03d}'

# Plot the parameter with an italic font
stationplot.plot_parameter('SE', df['height_change'], formatter=height_fall_formatter,
                           fontstyle='italic', color='black')

# Plot Solid Contours of Geopotential Height
cs = ax.contour(lons, lats, smooth_height,
                range(0, 20000, cint), colors='black', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot Dashed Contours of Temperature
cs2 = ax.contour(lons, lats, smooth_tmpc, range(-60, 51, 3),
                 colors='#c9514b', transform=ccrs.PlateCarree())
clabels = plt.clabel(cs2, fmt='%d', colors='white', fontsize=8, inline_spacing=5, use_clabeltext=True)

# Set longer dashes than default
try:
    for c in cs2.collections:
        c.set_dashes([(0, (5.0, 3.0))])
except AttributeError:
    for c in cs2.get_children():
        c.set_dashes([(0, (5.0, 3.0))])

# Contour labels with black boxes and white text
for t in clabels:
    t.set_bbox({'facecolor': 'black', 'pad': 4})
    t.set_fontweight('heavy')

# Plot filled circles for Radiosonde Obs
ax.scatter(df['longitude'].values, df['latitude'].values, s=12,
           marker='o', color='black', transform=ccrs.PlateCarree())

# Use definition to plot H/L symbols
plot_maxmin_points(ax, lons, lats, smooth_height, 'max', 50,
                   symbol='H', color='black', transform=ccrs.PlateCarree())
plot_maxmin_points(ax, lons, lats, smooth_height, 'min', 25,
                   symbol='L', color='black', transform=ccrs.PlateCarree())

# Add titles
plt.title('GFS {}-mb Analysis of Geopotential Heights (m) and Temperature (°C)'.format(level),
          loc='left')
plt.title('VALID: ' + str(day_12Z) + ' ' + str(month_name_12Z) + ' ' + str(year_12Z) + ' ' + str(hour_12Z) + '00 UTC', loc='right')

plt.savefig('850mb_DIFAX_Chart_GFS_12Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 15 complete')


################################################################################################################

# CHART 16: 500-mb Vorticity Chart (Analysis)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_12Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'u-component_of_wind_isobaric',
                'v-component_of_wind_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
u_wind_var = data_query.variables['u-component_of_wind_isobaric']
v_wind_var = data_query.variables['v-component_of_wind_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
height = mpcalc.smooth_n_point(height, 9, 5)
u_wind = u_wind_var[0, 0, :, :].squeeze() * units('m/s')
v_wind = v_wind_var[0, 0, :, :].squeeze() * units('m/s')

# Compute north-relative wind components for plotting purposes
#uwnd_er, vwnd_er = earth_relative_wind_components(u_wind, v_wind)

# Smooth wind components as desired
uwnd_500 = mpcalc.smooth_n_point(u_wind, 9, 5) * units('m/s')
vwnd_500 = mpcalc.smooth_n_point(v_wind, 9, 5) * units('m/s')

# Calculate grid spacing that is sign aware to use in absolute vorticity calculation
dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

# Calculate absolute vorticity from MetPy function
lats = lats.filled(np.nan)  # Replace mask with NaNs
avor_500 = mpcalc.absolute_vorticity(uwnd_500, vwnd_500, dx, dy, lats * units.degrees)

# Set up the projection that will be used for plotting
mapcrs = ccrs.LambertConformal(central_longitude=-100, central_latitude=35,
                               standard_parallels=(30, 60))

# Set up the projection of the data; if lat/lon then PlateCarree is what you want
datacrs = ccrs.PlateCarree()

# Start the figure and create plot axes with proper projection
fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add geopolitical boundaries for map reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))

# Absolute Vorticity contour levels
clevs_500_avor = np.arange(-4, 34, 2)

#Colors
colors = [
"#fe01fe",
"#be00fd",
"#7e00fe",
"#0000fe",
"#0158fe",
"#028bfe",
"#02befe",
"#00fefd",
"#03e4ca",
"#01cb7d",
"#00b300",
"#7ecb05",
"#cbe506",
"#fdfd00",
"#fdcb00",
"#fe9800",
"#fd6503",
"#fd0100"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#cc00cc")
colormap.set_over("#cc0100")
levels = np.array([-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=18)

# Plot absolute vorticity values (multiplying by 10^5 to scale appropriately)
cf = ax.contourf(lons, lats, avor_500*1e5, clevs_500_avor, norm=norm, cmap=colormap, extend='both',
                 transform=datacrs)
cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_label('Abs. Vorticity ($x 10^{-5} s^{-1}$)')

# Plot 500-hPa Geopotential Heights in decameters
hght_500_dam = height / 10
clevs_500_hght = np.arange(0, 8000, 6)
cs = ax.contour(lons, lats, hght_500_dam, clevs_500_hght, colors='white', transform=datacrs)
plt.clabel(cs, fmt='%d')

hght_500_data = hght_500_dam.magnitude
h_y, h_x = find_peaks(hght_500_data)
l_y, l_x = find_peaks(hght_500_data, maxima=False)

# Extract pressure values at those points
high_vals = hght_500_data[h_y, h_x].ravel()
low_vals = hght_500_data[l_y, l_x].ravel()

# Format labels as strings for plotting
high_labels = [f'{v:.0f}' for v in high_vals]
low_labels = [f'{v:.0f}' for v in low_vals]

text1 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], 'H', size=20, color='white',
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text2 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], high_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text3 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], 'L', size=20, color='white',
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text4 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], low_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text1.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text2.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text3.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text4.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])

# Plot two titles, one on right and left side
plt.title('GFS 500-mb Geopotential Heights (dam)'
          ' and Abs. Vorticity ($x 10^{-5} s^{-1}$)', loc='left')
plt.title('VALID: ' + str(day_12Z) + ' ' + str(month_name_12Z) + ' ' + str(year_12Z) + ' ' + str(hour_12Z) + '00 UTC', loc='right')

plt.savefig('500mb_Vorticity_Chart_GFS_12Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 16 complete')


################################################################################################################

# CHART 17: 850-mb Temperature Chart (Analysis)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_12Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'Temperature_isobaric')
query.vertical_level(85000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
temp_var = data_query.variables['Temperature_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
temp = temp_var[0, 0, :, :].squeeze()
smooth_tmpk = mpcalc.smooth_n_point(temp, 9, 5)
smooth_tmpc = smooth_tmpk - 273.15

# Set up the projection that will be used for plotting
mapcrs = ccrs.LambertConformal(central_longitude=-100, central_latitude=35,
                               standard_parallels=(30, 60))

# Set up the projection of the data; if lat/lon then PlateCarree is what you want
datacrs = ccrs.PlateCarree()

# Start the figure and create plot axes with proper projection
fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add geopolitical boundaries for map reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))

# Temperature contour levels
clevs_850_tmp = np.arange(-14, 32, 2)

#Colors
colors = [
"#fe01fe",
"#be00fd",
"#7e00fe",
"#0000fe",
"#0059fe",
"#008cfe",
"#03bffe",
"#00fefe",
"#00e5ca",
"#00cc7f",
"#02b202",
"#7fcb00",
"#cce505",
"#fefe00",
"#fecb02",
"#fe9800",
"#fe6600",
"#fe0000",
"#cc0100",
"#980001",
"#660000",
"#660066"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#cc00cc")
colormap.set_over("#980098")
levels = np.array([-14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=22)

# Plot 850-mb temperature values
cf = ax.contourf(lons, lats, smooth_tmpc, clevs_850_tmp, norm=norm, cmap=colormap, extend='both',
                 transform=datacrs)
cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_ticks(levels[::1])
cb.set_label('Temperature (°C)')

# Plot 850-hPa Geopotential Heights in decameters
hght_850_dam = smooth_height / 10
clevs_850_hght = np.arange(0, 8000, 3)
cs = ax.contour(lons, lats, hght_850_dam, clevs_850_hght, colors='white', linewidths=1.5, transform=datacrs)
plt.clabel(cs, fmt='%d')

hght_850_data = hght_850_dam.magnitude
h_y, h_x = find_peaks(hght_850_data)
l_y, l_x = find_peaks(hght_850_data, maxima=False)

# Extract pressure values at those points
high_vals = hght_850_data[h_y, h_x].ravel()
low_vals = hght_850_data[l_y, l_x].ravel()

# Format labels as strings for plotting
high_labels = [f'{v:.0f}' for v in high_vals]
low_labels = [f'{v:.0f}' for v in low_vals]

text1 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], 'H', size=20, color='white',
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text2 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], high_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text3 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], 'L', size=20, color='white',
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text4 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], low_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())

text1.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text2.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text3.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text4.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])

# Plot two titles, one on right and left side
plt.title('GFS 850-mb Geopotential Heights (dam)'
          ' and Temperature (°C)', loc='left')
plt.title('VALID: ' + str(day_12Z) + ' ' + str(month_name_12Z) + ' ' + str(year_12Z) + ' ' + str(hour_12Z) + '00 UTC', loc='right')

plt.savefig('850mb_Temperature_Chart_GFS_12Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 17 complete')


################################################################################################################

# CHART 18: MSLP/500-mb Height Stack Chart (Analysis)

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_12Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Create lat/lon box for location you want to get data for
query_mslp = ncss.query()
query_mslp.lonlat_box(north=90, south=0, east=-30, west=-150)
query_mslp.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for MSLP
query_mslp.variables('Pressure_reduced_to_MSL_msl')
data_query_mslp = ncss.get_data(query_mslp)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
mslp_var = data_query_mslp.variables['Pressure_reduced_to_MSL_msl']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
mslp = mslp_var[0, :, :].squeeze()
smooth_mslp_Pa = mpcalc.smooth_n_point(mslp, 9, 5)
smooth_mslp_hPa = smooth_mslp_Pa / 100

# Create the Chart
# ------------
#
# Set projection of map display
mapproj = ccrs.LambertConformal(central_latitude=45., central_longitude=-100.)

# Set projection of data
dataproj = ccrs.PlateCarree()

fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapproj)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='0.25')
ax.add_feature(cfeature.LAND.with_scale('50m'), edgecolor='0.25')
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='0.25')

#Colors
colors = [
"#be00fd",
"#7e00fe",
"#0000fe",
"#0158fe",
#"#028bfe",
"#02befe",
"#00fefd",
#"#03e4ca",
"#01cb7d",
"#00b300",
#"#7ecb05",
"#cbe506",
"#fdfd00",
"#fdcb00",
"#fe9800",
"#fd6503",
"#fd0100",
"#980001"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#fe01fe")
colormap.set_over("#660066")
levels = np.array([504, 510, 516, 522, 528, 534, 540, 546, 552, 558, 564, 570, 576, 582, 588, 594])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=15)

# Plot 500-mb Heights
hght_500_dam = smooth_height / 10
clevs_500_hght = np.arange(504, 600, 6)
cf = ax.contourf(lons, lats, hght_500_dam, clevs_500_hght, norm=norm, cmap=colormap, extend='both', transform=dataproj)

cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_ticks(levels[::1])

# Plot MSLP
clevmslp = np.arange(800., 1120., 4)
cs = ax.contour(lons, lats, smooth_mslp_hPa, clevmslp, colors='k', linewidths=1.5,
                 linestyles='solid', transform=dataproj)
kw_clabels = {'fontsize': 11, 'inline': True, 'inline_spacing': 5, 'fmt': '%i',
              'rightside_up': True, 'use_clabeltext': True}
plt.clabel(cs, **kw_clabels, zorder=1)

# Use definition to plot H/L symbols
# Use smoothed MSLP directly (in hPa) to find local maxima/minima
# Remove units before processing
mslp_data = smooth_mslp_hPa.magnitude
h_y, h_x = find_peaks(mslp_data)
l_y, l_x = find_peaks(mslp_data, maxima=False)

# Extract pressure values at those points
high_vals = mslp_data[h_y, h_x].ravel()
low_vals = mslp_data[l_y, l_x].ravel()

# Format labels as strings for plotting
high_labels = [f'{v:.0f}' for v in high_vals]
low_labels = [f'{v:.0f}' for v in low_vals]

text1 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], 'H', size=20, color='white',
            fontweight='bold', clip_on=True, transform=ccrs.PlateCarree(), zorder=5)
text2 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], high_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, transform=ccrs.PlateCarree(), zorder=5)
text3 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], 'L', size=20, color='white',
            fontweight='bold', clip_on=True, transform=ccrs.PlateCarree(), zorder=5)
text4 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], low_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, transform=ccrs.PlateCarree(), zorder=5)

text1.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text2.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text3.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text4.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])

# Put on some titles
plt.title('GFS MSLP (mb) and 500-mb Geopotential Heights (dam)', loc='left')
plt.title('VALID: ' + str(day_12Z) + ' ' + str(month_name_12Z) + ' ' + str(year_12Z) + ' ' + str(hour_12Z) + '00 UTC', loc='right')

plt.savefig('Stack_Chart_GFS_12Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 18 complete')


################################################################################################################

# CHART 19: MSLP/1000–500 mb Thickness Chart (Analysis)

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_' + str(master_date_12Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query_500hPa = ncss.query()
query_500hPa.lonlat_box(north=90, south=0, east=-30, west=-150)
query_500hPa.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for geopotential height
query_500hPa.variables('Geopotential_height_isobaric')
query_500hPa.vertical_level(50000)
data_query_500hPa = ncss.get_data(query_500hPa)

# Create lat/lon box for location you want to get data for
query_1000hPa = ncss.query()
query_1000hPa.lonlat_box(north=90, south=0, east=-30, west=-150)
query_1000hPa.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for geopotential height
query_1000hPa.variables('Geopotential_height_isobaric')
query_1000hPa.vertical_level(100000)
data_query_1000hPa = ncss.get_data(query_1000hPa)

# Create lat/lon box for location you want to get data for
query_mslp = ncss.query()
query_mslp.lonlat_box(north=90, south=0, east=-30, west=-150)
query_mslp.time(datetime(year_12Z, month_12Z, day_12Z, int(hour_12Z)))

# Request data for MSLP
query_mslp.variables('Pressure_reduced_to_MSL_msl')
data_query_mslp = ncss.get_data(query_mslp)

# Pull out variables you want to use
height_500hPa_var = data_query_500hPa.variables['Geopotential_height_isobaric']
height_1000hPa_var = data_query_1000hPa.variables['Geopotential_height_isobaric']
mslp_var = data_query_mslp.variables['Pressure_reduced_to_MSL_msl']

dlat = data_query_500hPa.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query_500hPa.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query_500hPa.variables[dlat][:]
lon = data_query_500hPa.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height500 = height_500hPa_var[0, 0, :, :].squeeze()
smooth_height500 = mpcalc.smooth_n_point(height500, 9, 5)
height1000 = height_1000hPa_var[0, 0, :, :].squeeze()
smooth_height1000 = mpcalc.smooth_n_point(height1000, 9, 5)
mslp = mslp_var[0, :, :].squeeze()
smooth_mslp_Pa = mpcalc.smooth_n_point(mslp, 9, 5)
smooth_mslp_hPa = smooth_mslp_Pa / 100

thickness_1000_500 = smooth_height500 - smooth_height1000

# Create the Chart
# ------------
#
# Set projection of map display
mapproj = ccrs.LambertConformal(central_latitude=45., central_longitude=-100.)

# Set projection of data
dataproj = ccrs.PlateCarree()

fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapproj)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', zorder=1)
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white', edgecolor='grey', zorder=1)
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey', zorder=1)

# Plot 1000-500 mb thickness with multiple colors
clevs = (np.arange(0, 5400, 60),
         np.array([5400]),
         np.arange(5460, 7000, 60))
colors = ('tab:blue', 'b', 'tab:red')
kw_clabels = {'fontsize': 11, 'inline': True, 'inline_spacing': 5, 'fmt': '%i',
              'rightside_up': True, 'use_clabeltext': True}
for clevthick, color in zip(clevs, colors):
    cs = ax.contour(lons, lats, thickness_1000_500, levels=clevthick, colors=color,
                    linewidths=1.5, linestyles='dashed', transform=dataproj)
    plt.clabel(cs, **kw_clabels)

# Plot MSLP
clevmslp = np.arange(800., 1120., 4)
cs2 = ax.contour(lons, lats, smooth_mslp_hPa, clevmslp, colors='k', linewidths=1.5,
                 linestyles='solid', transform=dataproj)
plt.clabel(cs2, **kw_clabels)

# Plot MSLP high and low centers using the helper function.
# This filters extrema to the displayed map extent and avoids invalid projected
# coordinates that can cause scattertext() to fail during savefig().
mslp_data = smooth_mslp_hPa.magnitude

plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'max', 50,
    symbol='H',
    color='blue',
    transform=ccrs.PlateCarree()
)

plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'min', 25,
    symbol='L',
    color='red',
    transform=ccrs.PlateCarree()
)

# Put on some titles
plt.title('GFS MSLP (mb) and 1000–500 mb Thickness (m)', loc='left')
plt.title('VALID: ' + str(day_12Z) + ' ' + str(month_name_12Z) + ' ' + str(year_12Z) + ' ' + str(hour_12Z) + '00 UTC', loc='right')

plt.savefig('MSLP_1000-500mb_Thickness_Chart_GFS_12Z_Analysis.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 19 complete')


################################################################################################################
# FORECAST CHARTS
################################################################################################################

# CHART 20: 500-mb Vorticity Chart (18-hr Forecast)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg/GFS_Global_0p5deg_' + str(master_date_06Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(fcst_18hr_year, fcst_18hr_month, fcst_18hr_day, int(fcst_18hr_hour)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'u-component_of_wind_isobaric',
                'v-component_of_wind_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
u_wind_var = data_query.variables['u-component_of_wind_isobaric']
v_wind_var = data_query.variables['v-component_of_wind_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
height = mpcalc.smooth_n_point(height, 9, 5)
u_wind = u_wind_var[0, 0, :, :].squeeze() * units('m/s')
v_wind = v_wind_var[0, 0, :, :].squeeze() * units('m/s')

# Compute north-relative wind components for plotting purposes
#uwnd_er, vwnd_er = earth_relative_wind_components(u_wind, v_wind)

# Smooth wind components as desired
uwnd_500 = mpcalc.smooth_n_point(u_wind, 9, 5) * units('m/s')
vwnd_500 = mpcalc.smooth_n_point(v_wind, 9, 5) * units('m/s')

# Calculate grid spacing that is sign aware to use in absolute vorticity calculation
dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

# Calculate absolute vorticity from MetPy function
lats = lats.filled(np.nan)  # Replace mask with NaNs
avor_500 = mpcalc.absolute_vorticity(uwnd_500, vwnd_500, dx, dy, lats * units.degrees)

# Set up the projection that will be used for plotting
mapcrs = ccrs.LambertConformal(central_longitude=-100, central_latitude=35,
                               standard_parallels=(30, 60))

# Set up the projection of the data; if lat/lon then PlateCarree is what you want
datacrs = ccrs.PlateCarree()

# Start the figure and create plot axes with proper projection
fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add geopolitical boundaries for map reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))

# Absolute Vorticity contour levels
clevs_500_avor = np.arange(-4, 34, 2)

#Colors
colors = [
"#fe01fe",
"#be00fd",
"#7e00fe",
"#0000fe",
"#0158fe",
"#028bfe",
"#02befe",
"#00fefd",
"#03e4ca",
"#01cb7d",
"#00b300",
"#7ecb05",
"#cbe506",
"#fdfd00",
"#fdcb00",
"#fe9800",
"#fd6503",
"#fd0100"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#cc00cc")
colormap.set_over("#cc0100")
levels = np.array([-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=18)

# Plot absolute vorticity values (multiplying by 10^5 to scale appropriately)
cf = ax.contourf(lons, lats, avor_500*1e5, clevs_500_avor, norm=norm, cmap=colormap, extend='both',
                 transform=datacrs)
cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_label('Abs. Vorticity ($x 10^{-5} s^{-1}$)')

# Plot 500-hPa Geopotential Heights in decameters
hght_500_dam = height / 10
clevs_500_hght = np.arange(0, 8000, 6)
cs = ax.contour(lons, lats, hght_500_dam, clevs_500_hght, colors='white', transform=datacrs)
plt.clabel(cs, fmt='%d')

hght_500_data = hght_500_dam.magnitude
h_y, h_x = find_peaks(hght_500_data)
l_y, l_x = find_peaks(hght_500_data, maxima=False)

# Extract pressure values at those points
high_vals = hght_500_data[h_y, h_x].ravel()
low_vals = hght_500_data[l_y, l_x].ravel()

# Format labels as strings for plotting
high_labels = [f'{v:.0f}' for v in high_vals]
low_labels = [f'{v:.0f}' for v in low_vals]

text1 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], 'H', size=20, color='white',
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text2 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], high_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text3 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], 'L', size=20, color='white',
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text4 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], low_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text1.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text2.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text3.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text4.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])

# Plot two titles, one on right and left side
plt.title('GFS 500-mb Geopotential Heights (dam)'
          ' and Abs. Vorticity ($x 10^{-5} s^{-1}$)', loc='left')
plt.title('VALID: ' + str(fcst_18hr_day) + ' ' + str(fcst_18hr_month_name) + ' ' + str(fcst_18hr_year) + ' ' + str(fcst_18hr_hour) + '00 UTC', loc='right')

plt.savefig('500mb_Vorticity_Chart_GFS_18hr_Forecast.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 20 complete')


################################################################################################################

# CHART 21: 500-mb Vorticity Chart (30-hr Forecast)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg/GFS_Global_0p5deg_' + str(master_date_06Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(fcst_30hr_year, fcst_30hr_month, fcst_30hr_day, int(fcst_30hr_hour)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'u-component_of_wind_isobaric',
                'v-component_of_wind_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
u_wind_var = data_query.variables['u-component_of_wind_isobaric']
v_wind_var = data_query.variables['v-component_of_wind_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
height = mpcalc.smooth_n_point(height, 9, 5)
u_wind = u_wind_var[0, 0, :, :].squeeze() * units('m/s')
v_wind = v_wind_var[0, 0, :, :].squeeze() * units('m/s')

# Compute north-relative wind components for plotting purposes
#uwnd_er, vwnd_er = earth_relative_wind_components(u_wind, v_wind)

# Smooth wind components as desired
uwnd_500 = mpcalc.smooth_n_point(u_wind, 9, 5) * units('m/s')
vwnd_500 = mpcalc.smooth_n_point(v_wind, 9, 5) * units('m/s')

# Calculate grid spacing that is sign aware to use in absolute vorticity calculation
dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

# Calculate absolute vorticity from MetPy function
lats = lats.filled(np.nan)  # Replace mask with NaNs
avor_500 = mpcalc.absolute_vorticity(uwnd_500, vwnd_500, dx, dy, lats * units.degrees)

# Set up the projection that will be used for plotting
mapcrs = ccrs.LambertConformal(central_longitude=-100, central_latitude=35,
                               standard_parallels=(30, 60))

# Set up the projection of the data; if lat/lon then PlateCarree is what you want
datacrs = ccrs.PlateCarree()

# Start the figure and create plot axes with proper projection
fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add geopolitical boundaries for map reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))

# Absolute Vorticity contour levels
clevs_500_avor = np.arange(-4, 34, 2)

#Colors
colors = [
"#fe01fe",
"#be00fd",
"#7e00fe",
"#0000fe",
"#0158fe",
"#028bfe",
"#02befe",
"#00fefd",
"#03e4ca",
"#01cb7d",
"#00b300",
"#7ecb05",
"#cbe506",
"#fdfd00",
"#fdcb00",
"#fe9800",
"#fd6503",
"#fd0100"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#cc00cc")
colormap.set_over("#cc0100")
levels = np.array([-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=18)

# Plot absolute vorticity values (multiplying by 10^5 to scale appropriately)
cf = ax.contourf(lons, lats, avor_500*1e5, clevs_500_avor, norm=norm, cmap=colormap, extend='both',
                 transform=datacrs)
cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_label('Abs. Vorticity ($x 10^{-5} s^{-1}$)')

# Plot 500-hPa Geopotential Heights in decameters
hght_500_dam = height / 10
clevs_500_hght = np.arange(0, 8000, 6)
cs = ax.contour(lons, lats, hght_500_dam, clevs_500_hght, colors='white', transform=datacrs)
plt.clabel(cs, fmt='%d')

hght_500_data = hght_500_dam.magnitude
h_y, h_x = find_peaks(hght_500_data)
l_y, l_x = find_peaks(hght_500_data, maxima=False)

# Extract pressure values at those points
high_vals = hght_500_data[h_y, h_x].ravel()
low_vals = hght_500_data[l_y, l_x].ravel()

# Format labels as strings for plotting
high_labels = [f'{v:.0f}' for v in high_vals]
low_labels = [f'{v:.0f}' for v in low_vals]

text1 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], 'H', size=20, color='white',
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text2 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], high_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text3 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], 'L', size=20, color='white',
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text4 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], low_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text1.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text2.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text3.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text4.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])

# Plot two titles, one on right and left side
plt.title('GFS 500-mb Geopotential Heights (dam)'
          ' and Abs. Vorticity ($x 10^{-5} s^{-1}$)', loc='left')
plt.title('VALID: ' + str(fcst_30hr_day) + ' ' + str(fcst_30hr_month_name) + ' ' + str(fcst_30hr_year) + ' ' + str(fcst_30hr_hour) + '00 UTC', loc='right')

plt.savefig('500mb_Vorticity_Chart_GFS_30hr_Forecast.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 21 complete')


################################################################################################################

# CHART 22: 850-mb Temperature Chart (18-hr Forecast)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg/GFS_Global_0p5deg_' + str(master_date_06Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(fcst_18hr_year, fcst_18hr_month, fcst_18hr_day, int(fcst_18hr_hour)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'Temperature_isobaric')
query.vertical_level(85000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
temp_var = data_query.variables['Temperature_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
temp = temp_var[0, 0, :, :].squeeze()
smooth_tmpk = mpcalc.smooth_n_point(temp, 9, 5)
smooth_tmpc = smooth_tmpk - 273.15

# Set up the projection that will be used for plotting
mapcrs = ccrs.LambertConformal(central_longitude=-100, central_latitude=35,
                               standard_parallels=(30, 60))

# Set up the projection of the data; if lat/lon then PlateCarree is what you want
datacrs = ccrs.PlateCarree()

# Start the figure and create plot axes with proper projection
fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add geopolitical boundaries for map reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))

# Temperature contour levels
clevs_850_tmp = np.arange(-14, 32, 2)

#Colors
colors = [
"#fe01fe",
"#be00fd",
"#7e00fe",
"#0000fe",
"#0059fe",
"#008cfe",
"#03bffe",
"#00fefe",
"#00e5ca",
"#00cc7f",
"#02b202",
"#7fcb00",
"#cce505",
"#fefe00",
"#fecb02",
"#fe9800",
"#fe6600",
"#fe0000",
"#cc0100",
"#980001",
"#660000",
"#660066"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#cc00cc")
colormap.set_over("#980098")
levels = np.array([-14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=22)

# Plot 850-mb temperature values
cf = ax.contourf(lons, lats, smooth_tmpc, clevs_850_tmp, norm=norm, cmap=colormap, extend='both',
                 transform=datacrs)
cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_ticks(levels[::1])
cb.set_label('Temperature (°C)')

# Plot 850-hPa Geopotential Heights in decameters
hght_850_dam = smooth_height / 10
clevs_850_hght = np.arange(0, 8000, 3)
cs = ax.contour(lons, lats, hght_850_dam, clevs_850_hght, colors='white', transform=datacrs)
plt.clabel(cs, fmt='%d')

hght_850_data = hght_850_dam.magnitude
h_y, h_x = find_peaks(hght_850_data)
l_y, l_x = find_peaks(hght_850_data, maxima=False)

# Extract pressure values at those points
high_vals = hght_850_data[h_y, h_x].ravel()
low_vals = hght_850_data[l_y, l_x].ravel()

# Format labels as strings for plotting
high_labels = [f'{v:.0f}' for v in high_vals]
low_labels = [f'{v:.0f}' for v in low_vals]

text1 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], 'H', size=20, color='white',
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text2 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], high_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text3 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], 'L', size=20, color='white',
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text4 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], low_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())

text1.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text2.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text3.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text4.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])

# Plot two titles, one on right and left side
plt.title('GFS 850-mb Geopotential Heights (dam)'
          ' and Temperature (°C)', loc='left')
plt.title('VALID: ' + str(fcst_18hr_day) + ' ' + str(fcst_18hr_month_name) + ' ' + str(fcst_18hr_year) + ' ' + str(fcst_18hr_hour) + '00 UTC', loc='right')

plt.savefig('850mb_Temperature_Chart_GFS_18hr_Forecast.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 22 complete')


################################################################################################################

# CHART 23: 850-mb Temperature Chart (30-hr Forecast)

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg/GFS_Global_0p5deg_' + str(master_date_06Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(fcst_30hr_year, fcst_30hr_month, fcst_30hr_day, int(fcst_30hr_hour)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric', 'Temperature_isobaric')
query.vertical_level(85000)
data_query = ncss.get_data(query)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
temp_var = data_query.variables['Temperature_isobaric']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
temp = temp_var[0, 0, :, :].squeeze()
smooth_tmpk = mpcalc.smooth_n_point(temp, 9, 5)
smooth_tmpc = smooth_tmpk - 273.15

# Set up the projection that will be used for plotting
mapcrs = ccrs.LambertConformal(central_longitude=-100, central_latitude=35,
                               standard_parallels=(30, 60))

# Set up the projection of the data; if lat/lon then PlateCarree is what you want
datacrs = ccrs.PlateCarree()

# Start the figure and create plot axes with proper projection
fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add geopolitical boundaries for map reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))

# Temperature contour levels
clevs_850_tmp = np.arange(-14, 32, 2)

#Colors
colors = [
"#fe01fe",
"#be00fd",
"#7e00fe",
"#0000fe",
"#0059fe",
"#008cfe",
"#03bffe",
"#00fefe",
"#00e5ca",
"#00cc7f",
"#02b202",
"#7fcb00",
"#cce505",
"#fefe00",
"#fecb02",
"#fe9800",
"#fe6600",
"#fe0000",
"#cc0100",
"#980001",
"#660000",
"#660066"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#cc00cc")
colormap.set_over("#980098")
levels = np.array([-14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=22)

# Plot 850-mb temperature values
cf = ax.contourf(lons, lats, smooth_tmpc, clevs_850_tmp, norm=norm, cmap=colormap, extend='both',
                 transform=datacrs)
cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_ticks(levels[::1])
cb.set_label('Temperature (°C)')

# Plot 850-hPa Geopotential Heights in decameters
hght_850_dam = smooth_height / 10
clevs_850_hght = np.arange(0, 8000, 3)
cs = ax.contour(lons, lats, hght_850_dam, clevs_850_hght, colors='white', transform=datacrs)
plt.clabel(cs, fmt='%d')

hght_850_data = hght_850_dam.magnitude
h_y, h_x = find_peaks(hght_850_data)
l_y, l_x = find_peaks(hght_850_data, maxima=False)

# Extract pressure values at those points
high_vals = hght_850_data[h_y, h_x].ravel()
low_vals = hght_850_data[l_y, l_x].ravel()

# Format labels as strings for plotting
high_labels = [f'{v:.0f}' for v in high_vals]
low_labels = [f'{v:.0f}' for v in low_vals]

text1 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], 'H', size=20, color='white',
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text2 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], high_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text3 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], 'L', size=20, color='white',
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())
text4 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], low_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, zorder=5, transform=ccrs.PlateCarree())

text1.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text2.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text3.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text4.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])

# Plot two titles, one on right and left side
plt.title('GFS 850-mb Geopotential Heights (dam)'
          ' and Temperature (°C)', loc='left')
plt.title('VALID: ' + str(fcst_30hr_day) + ' ' + str(fcst_30hr_month_name) + ' ' + str(fcst_30hr_year) + ' ' + str(fcst_30hr_hour) + '00 UTC', loc='right')

plt.savefig('850mb_Temperature_Chart_GFS_30hr_Forecast.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 23 complete')


################################################################################################################

# CHART 24: MSLP/500-mb Height Stack Chart (18-hr Forecast)

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg/GFS_Global_0p5deg_' + str(master_date_06Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(fcst_18hr_year, fcst_18hr_month, fcst_18hr_day, int(fcst_18hr_hour)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Create lat/lon box for location you want to get data for
query_mslp = ncss.query()
query_mslp.lonlat_box(north=90, south=0, east=-30, west=-150)
query_mslp.time(datetime(fcst_18hr_year, fcst_18hr_month, fcst_18hr_day, int(fcst_18hr_hour)))

# Request data for MSLP
query_mslp.variables('Pressure_reduced_to_MSL_msl')
data_query_mslp = ncss.get_data(query_mslp)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
mslp_var = data_query_mslp.variables['Pressure_reduced_to_MSL_msl']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
mslp = mslp_var[0, :, :].squeeze()
smooth_mslp_Pa = mpcalc.smooth_n_point(mslp, 9, 5)
smooth_mslp_hPa = smooth_mslp_Pa / 100

# Create the Chart
# ------------
#
# Set projection of map display
mapproj = ccrs.LambertConformal(central_latitude=45., central_longitude=-100.)

# Set projection of data
dataproj = ccrs.PlateCarree()

fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapproj)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='0.25')
ax.add_feature(cfeature.LAND.with_scale('50m'), edgecolor='0.25')
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='0.25')

#Colors
colors = [
"#be00fd",
"#7e00fe",
"#0000fe",
"#0158fe",
#"#028bfe",
"#02befe",
"#00fefd",
#"#03e4ca",
"#01cb7d",
"#00b300",
#"#7ecb05",
"#cbe506",
"#fdfd00",
"#fdcb00",
"#fe9800",
"#fd6503",
"#fd0100",
"#980001"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#fe01fe")
colormap.set_over("#660066")
levels = np.array([504, 510, 516, 522, 528, 534, 540, 546, 552, 558, 564, 570, 576, 582, 588, 594])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=15)

# Plot 500-mb Heights
hght_500_dam = smooth_height / 10
clevs_500_hght = np.arange(504, 600, 6)
cf = ax.contourf(lons, lats, hght_500_dam, clevs_500_hght, norm=norm, cmap=colormap, extend='both', transform=dataproj)

cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_ticks(levels[::1])

# Plot MSLP
clevmslp = np.arange(800., 1120., 4)
cs = ax.contour(lons, lats, smooth_mslp_hPa, clevmslp, colors='k', linewidths=1.5,
                 linestyles='solid', transform=dataproj)
kw_clabels = {'fontsize': 11, 'inline': True, 'inline_spacing': 5, 'fmt': '%i',
              'rightside_up': True, 'use_clabeltext': True}
plt.clabel(cs, **kw_clabels, zorder=1)

# Use definition to plot H/L symbols
# Use smoothed MSLP directly (in hPa) to find local maxima/minima
# Remove units before processing
mslp_data = smooth_mslp_hPa.magnitude
h_y, h_x = find_peaks(mslp_data)
l_y, l_x = find_peaks(mslp_data, maxima=False)

# Extract pressure values at those points
high_vals = mslp_data[h_y, h_x].ravel()
low_vals = mslp_data[l_y, l_x].ravel()

# Format labels as strings for plotting
high_labels = [f'{v:.0f}' for v in high_vals]
low_labels = [f'{v:.0f}' for v in low_vals]

text1 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], 'H', size=20, color='white',
            fontweight='bold', clip_on=True, transform=ccrs.PlateCarree(), zorder=5)
text2 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], high_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, transform=ccrs.PlateCarree(), zorder=5)
text3 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], 'L', size=20, color='white',
            fontweight='bold', clip_on=True, transform=ccrs.PlateCarree(), zorder=5)
text4 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], low_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, transform=ccrs.PlateCarree(), zorder=5)

text1.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text2.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text3.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text4.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])

# Put on some titles
plt.title('GFS MSLP (mb) and 500-mb Geopotential Heights (dam)', loc='left')
plt.title('VALID: ' + str(fcst_18hr_day) + ' ' + str(fcst_18hr_month_name) + ' ' + str(fcst_18hr_year) + ' ' + str(fcst_18hr_hour) + '00 UTC', loc='right')

plt.savefig('Stack_Chart_GFS_18hr_Forecast.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 24 complete')


################################################################################################################

# CHART 25: MSLP/500-mb Height Stack Chart (30-hr Forecast)

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg/GFS_Global_0p5deg_' + str(master_date_06Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query = ncss.query()
query.lonlat_box(north=90, south=0, east=-30, west=-150)
query.time(datetime(fcst_30hr_year, fcst_30hr_month, fcst_30hr_day, int(fcst_30hr_hour)))

# Request data for geopotential height
query.variables('Geopotential_height_isobaric')
query.vertical_level(50000)
data_query = ncss.get_data(query)

# Create lat/lon box for location you want to get data for
query_mslp = ncss.query()
query_mslp.lonlat_box(north=90, south=0, east=-30, west=-150)
query_mslp.time(datetime(fcst_30hr_year, fcst_30hr_month, fcst_30hr_day, int(fcst_30hr_hour)))

# Request data for MSLP
query_mslp.variables('Pressure_reduced_to_MSL_msl')
data_query_mslp = ncss.get_data(query_mslp)

# Pull out variables you want to use
height_var = data_query.variables['Geopotential_height_isobaric']
mslp_var = data_query_mslp.variables['Pressure_reduced_to_MSL_msl']

dlat = data_query.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query.variables[dlat][:]
lon = data_query.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height = height_var[0, 0, :, :].squeeze()
smooth_height = mpcalc.smooth_n_point(height, 9, 5)
mslp = mslp_var[0, :, :].squeeze()
smooth_mslp_Pa = mpcalc.smooth_n_point(mslp, 9, 5)
smooth_mslp_hPa = smooth_mslp_Pa / 100

# Create the Chart
# ------------
#
# Set projection of map display
mapproj = ccrs.LambertConformal(central_latitude=45., central_longitude=-100.)

# Set projection of data
dataproj = ccrs.PlateCarree()

fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapproj)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='0.25')
ax.add_feature(cfeature.LAND.with_scale('50m'), edgecolor='0.25')
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='0.25')

#Colors
colors = [
"#be00fd",
"#7e00fe",
"#0000fe",
"#0158fe",
#"#028bfe",
"#02befe",
"#00fefd",
#"#03e4ca",
"#01cb7d",
"#00b300",
#"#7ecb05",
"#cbe506",
"#fdfd00",
"#fdcb00",
"#fe9800",
"#fd6503",
"#fd0100",
"#980001"
]
colormap = matplotlib.colors.ListedColormap(colors)
colormap.set_under("#fe01fe")
colormap.set_over("#660066")
levels = np.array([504, 510, 516, 522, 528, 534, 540, 546, 552, 558, 564, 570, 576, 582, 588, 594])
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=15)

# Plot 500-mb Heights
hght_500_dam = smooth_height / 10
clevs_500_hght = np.arange(504, 600, 6)
cf = ax.contourf(lons, lats, hght_500_dam, clevs_500_hght, norm=norm, cmap=colormap, extend='both', transform=dataproj)

cb = plt.colorbar(cf, orientation='horizontal', pad=0.02, aspect=50)
cb.set_ticks(levels[::1])

# Plot MSLP
clevmslp = np.arange(800., 1120., 4)
cs = ax.contour(lons, lats, smooth_mslp_hPa, clevmslp, colors='k', linewidths=1.5,
                 linestyles='solid', transform=dataproj)
kw_clabels = {'fontsize': 11, 'inline': True, 'inline_spacing': 5, 'fmt': '%i',
              'rightside_up': True, 'use_clabeltext': True}
plt.clabel(cs, **kw_clabels, zorder=1)

# Use definition to plot H/L symbols
# Use smoothed MSLP directly (in hPa) to find local maxima/minima
# Remove units before processing
mslp_data = smooth_mslp_hPa.magnitude
h_y, h_x = find_peaks(mslp_data)
l_y, l_x = find_peaks(mslp_data, maxima=False)

# Extract pressure values at those points
high_vals = mslp_data[h_y, h_x].ravel()
low_vals = mslp_data[l_y, l_x].ravel()

# Format labels as strings for plotting
high_labels = [f'{v:.0f}' for v in high_vals]
low_labels = [f'{v:.0f}' for v in low_vals]

text1 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], 'H', size=20, color='white',
            fontweight='bold', clip_on=True, transform=ccrs.PlateCarree(), zorder=5)
text2 = scattertext(ax, lons[h_y, h_x], lats[h_y, h_x], high_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, transform=ccrs.PlateCarree(), zorder=5)
text3 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], 'L', size=20, color='white',
            fontweight='bold', clip_on=True, transform=ccrs.PlateCarree(), zorder=5)
text4 = scattertext(ax, lons[l_y, l_x], lats[l_y, l_x], low_labels,
            size=12, color='white', loc=(0, -15),
            fontweight='bold', clip_on=True, transform=ccrs.PlateCarree(), zorder=5)

text1.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text2.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text3.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])
text4.set_path_effects([
    PathEffects.withStroke(linewidth=2, foreground='black')  # edge color
])

# Put on some titles
plt.title('GFS MSLP (mb) and 500-mb Geopotential Heights (dam)', loc='left')
plt.title('VALID: ' + str(fcst_30hr_day) + ' ' + str(fcst_30hr_month_name) + ' ' + str(fcst_30hr_year) + ' ' + str(fcst_30hr_hour) + '00 UTC', loc='right')

plt.savefig('Stack_Chart_GFS_30hr_Forecast.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 25 complete')


################################################################################################################

# CHART 26: MSLP/1000–500 mb Thickness Chart (18-hr Forecast)

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg/GFS_Global_0p5deg_' + str(master_date_06Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query_500hPa = ncss.query()
query_500hPa.lonlat_box(north=90, south=0, east=-30, west=-150)
query_500hPa.time(datetime(fcst_18hr_year, fcst_18hr_month, fcst_18hr_day, int(fcst_18hr_hour)))

# Request data for geopotential height
query_500hPa.variables('Geopotential_height_isobaric')
query_500hPa.vertical_level(50000)
data_query_500hPa = ncss.get_data(query_500hPa)

# Create lat/lon box for location you want to get data for
query_1000hPa = ncss.query()
query_1000hPa.lonlat_box(north=90, south=0, east=-30, west=-150)
query_1000hPa.time(datetime(fcst_18hr_year, fcst_18hr_month, fcst_18hr_day, int(fcst_18hr_hour)))

# Request data for geopotential height
query_1000hPa.variables('Geopotential_height_isobaric')
query_1000hPa.vertical_level(100000)
data_query_1000hPa = ncss.get_data(query_1000hPa)

# Create lat/lon box for location you want to get data for
query_mslp = ncss.query()
query_mslp.lonlat_box(north=90, south=0, east=-30, west=-150)
query_mslp.time(datetime(fcst_18hr_year, fcst_18hr_month, fcst_18hr_day, int(fcst_18hr_hour)))

# Request data for MSLP
query_mslp.variables('Pressure_reduced_to_MSL_msl')
data_query_mslp = ncss.get_data(query_mslp)

# Create lat/lon box for location you want to get data for
query_precip = ncss.query()
query_precip.lonlat_box(north=90, south=0, east=-30, west=-150)
query_precip.time(datetime(fcst_18hr_year, fcst_18hr_month, fcst_18hr_day, int(fcst_18hr_hour)))

# Request data for MSLP
query_precip.variables('Total_precipitation_surface_Mixed_intervals_Accumulation')
data_query_precip = ncss.get_data(query_precip)

# Pull out variables you want to use
height_500hPa_var = data_query_500hPa.variables['Geopotential_height_isobaric']
height_1000hPa_var = data_query_1000hPa.variables['Geopotential_height_isobaric']
mslp_var = data_query_mslp.variables['Pressure_reduced_to_MSL_msl']
precip_var = data_query_precip.variables['Total_precipitation_surface_Mixed_intervals_Accumulation']

dlat = data_query_500hPa.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query_500hPa.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query_500hPa.variables[dlat][:]
lon = data_query_500hPa.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height500 = height_500hPa_var[0, 0, :, :].squeeze()
smooth_height500 = mpcalc.smooth_n_point(height500, 9, 5)
height1000 = height_1000hPa_var[0, 0, :, :].squeeze()
smooth_height1000 = mpcalc.smooth_n_point(height1000, 9, 5)
mslp = mslp_var[0, :, :].squeeze()
smooth_mslp_Pa = mpcalc.smooth_n_point(mslp, 9, 5)
smooth_mslp_hPa = smooth_mslp_Pa / 100
precip_mm = precip_var[0, :, :].squeeze()
precip_inches = precip_mm / 25.4

thickness_1000_500 = smooth_height500 - smooth_height1000

# Create the Chart
# ------------
#
# Set projection of map display
mapproj = ccrs.LambertConformal(central_latitude=45., central_longitude=-100.)

# Set projection of data
dataproj = ccrs.PlateCarree()

fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapproj)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', zorder=2)
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='none', edgecolor='grey', zorder=2)
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey', zorder=2)

# Plot 1000-500 mb thickness with multiple colors
clevs = (np.arange(0, 5400, 60),
         np.array([5400]),
         np.arange(5460, 7000, 60))
colors = ('tab:blue', 'b', 'tab:red')
kw_clabels = {'fontsize': 11, 'inline': True, 'inline_spacing': 5, 'fmt': '%i',
              'rightside_up': True, 'use_clabeltext': True}
for clevthick, color in zip(clevs, colors):
    cs = ax.contour(lons, lats, thickness_1000_500, levels=clevthick, colors=color,
                    linewidths=1.5, linestyles='dashed', transform=dataproj)
    plt.clabel(cs, **kw_clabels)

# Plot MSLP
clevmslp = np.arange(800., 1120., 4)
cs2 = ax.contour(lons, lats, smooth_mslp_hPa, clevmslp, colors='k', linewidths=1.5,
                 linestyles='solid', transform=dataproj)
plt.clabel(cs2, **kw_clabels)

# Use definition to plot H/L symbols
# Use smoothed MSLP directly (in hPa) to find local maxima/minima
# Remove units before processing
mslp_data = smooth_mslp_hPa.magnitude

plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'max', 50,
    symbol='H',
    color='blue',
    transform=ccrs.PlateCarree()
)

plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'min', 25,
    symbol='L',
    color='red',
    transform=ccrs.PlateCarree()
)

# Plot 3-hour precipitation
nws_precip_colors = [
#"#b2fffe",
#"#04e9e7",
#"#019ff4",
#"#0300f4",
"#02fd02",
"#01c501",
"#008e00",
"#fdf802",
"#e5bc00",
"#fd9500",
"#e84a00",
"#d40000",
"#a90000",
"#f800fd",
"#c82ae2",
"#9854c6"
]
precip_colormap = ListedColormap(nws_precip_colors)
levels = np.array([0.01, 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00])
norm = BoundaryNorm(boundaries=levels, ncolors=12)

# Plot with contour fill using NWS Reflectivity colormap
precip_contourfill = ax.contourf(lons, lats, precip_inches, levels, norm=norm, cmap=precip_colormap, transform=ccrs.PlateCarree())

# Colorbar
def format_tick(x, pos):
    return f"{x:.3f}".rstrip('0').rstrip('.') if '.' in f"{x:.3f}" else f"{x:.3f}"
cbar = plt.colorbar(precip_contourfill, ticks=levels, orientation='horizontal', pad=0.02, shrink=0.5, fraction=0.075)
cbar.ax.xaxis.set_major_formatter(FuncFormatter(format_tick))

# Put on some titles
#plt.title('NAM MSLP (mb) and 1000–500 mb Thickness (m)', loc='left')
plt.title('GFS MSLP (mb), 1000–500 mb Thickness (m), and 3-hr Precipitation (in)', loc='left')
plt.title('VALID: ' + str(fcst_18hr_day) + ' ' + str(fcst_18hr_month_name) + ' ' + str(fcst_18hr_year) + ' ' + str(fcst_18hr_hour) + '00 UTC', loc='right')

plt.savefig('MSLP_1000-500mb_Thickness_Chart_GFS_18hr_Forecast.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 26 complete')


################################################################################################################

# CHART 27: MSLP/1000–500 mb Thickness Chart (30-hr Forecast)

# Gridded Data
# ------------
#
# Obtain NAM gridded output for contour plotting. Specifically,
# geopotential height and temperature data for the given level and subset
# for over North America. Data are smoothed for aesthetic reasons.
#

# Create NCSS object to access the NetcdfSubset
url = 'https://thredds.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg/GFS_Global_0p5deg_' + str(master_date_06Z) + '.grib2'
ncss = NCSS(url)

# Create lat/lon box for location you want to get data for
query_500hPa = ncss.query()
query_500hPa.lonlat_box(north=90, south=0, east=-30, west=-150)
query_500hPa.time(datetime(fcst_30hr_year, fcst_30hr_month, fcst_30hr_day, int(fcst_30hr_hour)))

# Request data for geopotential height
query_500hPa.variables('Geopotential_height_isobaric')
query_500hPa.vertical_level(50000)
data_query_500hPa = ncss.get_data(query_500hPa)

# Create lat/lon box for location you want to get data for
query_1000hPa = ncss.query()
query_1000hPa.lonlat_box(north=90, south=0, east=-30, west=-150)
query_1000hPa.time(datetime(fcst_30hr_year, fcst_30hr_month, fcst_30hr_day, int(fcst_30hr_hour)))

# Request data for geopotential height
query_1000hPa.variables('Geopotential_height_isobaric')
query_1000hPa.vertical_level(100000)
data_query_1000hPa = ncss.get_data(query_1000hPa)

# Create lat/lon box for location you want to get data for
query_mslp = ncss.query()
query_mslp.lonlat_box(north=90, south=0, east=-30, west=-150)
query_mslp.time(datetime(fcst_30hr_year, fcst_30hr_month, fcst_30hr_day, int(fcst_30hr_hour)))

# Request data for MSLP
query_mslp.variables('Pressure_reduced_to_MSL_msl')
data_query_mslp = ncss.get_data(query_mslp)

# Create lat/lon box for location you want to get data for
query_precip = ncss.query()
query_precip.lonlat_box(north=90, south=0, east=-30, west=-150)
query_precip.time(datetime(fcst_30hr_year, fcst_30hr_month, fcst_30hr_day, int(fcst_30hr_hour)))

# Request data for MSLP
query_precip.variables('Total_precipitation_surface_Mixed_intervals_Accumulation')
data_query_precip = ncss.get_data(query_precip)

# Pull out variables you want to use
height_500hPa_var = data_query_500hPa.variables['Geopotential_height_isobaric']
height_1000hPa_var = data_query_1000hPa.variables['Geopotential_height_isobaric']
mslp_var = data_query_mslp.variables['Pressure_reduced_to_MSL_msl']
precip_var = data_query_precip.variables['Total_precipitation_surface_Mixed_intervals_Accumulation']

dlat = data_query_500hPa.variables['Geopotential_height_isobaric'].dimensions[2]
dlon = data_query_500hPa.variables['Geopotential_height_isobaric'].dimensions[3]
lat = data_query_500hPa.variables[dlat][:]
lon = data_query_500hPa.variables[dlon][:]
lons, lats = np.meshgrid(lon, lat)

height500 = height_500hPa_var[0, 0, :, :].squeeze()
smooth_height500 = mpcalc.smooth_n_point(height500, 9, 5)
height1000 = height_1000hPa_var[0, 0, :, :].squeeze()
smooth_height1000 = mpcalc.smooth_n_point(height1000, 9, 5)
mslp = mslp_var[0, :, :].squeeze()
smooth_mslp_Pa = mpcalc.smooth_n_point(mslp, 9, 5)
smooth_mslp_hPa = smooth_mslp_Pa / 100
precip_mm = precip_var[0, :, :].squeeze()
precip_inches = precip_mm / 25.4

thickness_1000_500 = smooth_height500 - smooth_height1000

# Create the Chart
# ------------
#
# Set projection of map display
mapproj = ccrs.LambertConformal(central_latitude=45., central_longitude=-100.)

# Set projection of data
dataproj = ccrs.PlateCarree()

fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapproj)
ax.set_extent([-128.5, -72.5, 20.5, 58], ccrs.PlateCarree())

# Add map features for geographic reference
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey', zorder=2)
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='none', edgecolor='grey', zorder=2)
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey', zorder=2)

# Plot 1000-500 mb thickness with multiple colors
clevs = (np.arange(0, 5400, 60),
         np.array([5400]),
         np.arange(5460, 7000, 60))
colors = ('tab:blue', 'b', 'tab:red')
kw_clabels = {'fontsize': 11, 'inline': True, 'inline_spacing': 5, 'fmt': '%i',
              'rightside_up': True, 'use_clabeltext': True}
for clevthick, color in zip(clevs, colors):
    cs = ax.contour(lons, lats, thickness_1000_500, levels=clevthick, colors=color,
                    linewidths=1.5, linestyles='dashed', transform=dataproj)
    plt.clabel(cs, **kw_clabels)

# Plot MSLP
clevmslp = np.arange(800., 1120., 4)
cs2 = ax.contour(lons, lats, smooth_mslp_hPa, clevmslp, colors='k', linewidths=1.5,
                 linestyles='solid', transform=dataproj)
plt.clabel(cs2, **kw_clabels)

# Use definition to plot H/L symbols
# Use smoothed MSLP directly (in hPa) to find local maxima/minima
# Remove units before processing
mslp_data = smooth_mslp_hPa.magnitude

plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'max', 50,
    symbol='H',
    color='blue',
    transform=ccrs.PlateCarree()
)

plot_maxmin_points(
    ax, lons, lats, mslp_data,
    'min', 25,
    symbol='L',
    color='red',
    transform=ccrs.PlateCarree()
)

# Plot 3-hour precipitation
nws_precip_colors = [
#"#b2fffe",
#"#04e9e7",
#"#019ff4",
#"#0300f4",
"#02fd02",
"#01c501",
"#008e00",
"#fdf802",
"#e5bc00",
"#fd9500",
"#e84a00",
"#d40000",
"#a90000",
"#f800fd",
"#c82ae2",
"#9854c6"
]
precip_colormap = ListedColormap(nws_precip_colors)
levels = np.array([0.01, 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00])
norm = BoundaryNorm(boundaries=levels, ncolors=12)

# Plot with contour fill using NWS Reflectivity colormap
precip_contourfill = ax.contourf(lons, lats, precip_inches, levels, norm=norm, cmap=precip_colormap, transform=ccrs.PlateCarree())

# Colorbar
def format_tick(x, pos):
    return f"{x:.3f}".rstrip('0').rstrip('.') if '.' in f"{x:.3f}" else f"{x:.3f}"
cbar = plt.colorbar(precip_contourfill, ticks=levels, orientation='horizontal', pad=0.02, shrink=0.5, fraction=0.075)
cbar.ax.xaxis.set_major_formatter(FuncFormatter(format_tick))

# Put on some titles
#plt.title('NAM MSLP (mb) and 1000–500 mb Thickness (m)', loc='left')
plt.title('GFS MSLP (mb), 1000–500 mb Thickness (m), and 3-hr Precipitation (in)', loc='left')
plt.title('VALID: ' + str(fcst_30hr_day) + ' ' + str(fcst_30hr_month_name) + ' ' + str(fcst_30hr_year) + ' ' + str(fcst_30hr_hour) + '00 UTC', loc='right')

plt.savefig('MSLP_1000-500mb_Thickness_Chart_GFS_30hr_Forecast.png', dpi=500, bbox_inches='tight')
plt.close()
print('Chart 27 complete')

import subprocess

print("All charts completed successfully.")
print("Uploading charts to GitHub...")

subprocess.run(
    ['/home/wiley/Synoptic-Charts/upload_synoptic_charts_revised.sh'],
    check=True
)

print("Charts successfully uploaded to GitHub.")
