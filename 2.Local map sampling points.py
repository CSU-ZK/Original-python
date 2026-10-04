import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import re
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 16

def parse_cleaned_coord(coord_str):
    if pd.isna(coord_str) or str(coord_str).strip() == '':
        return None
    s = str(coord_str).strip().upper()
    match = re.search(r'([-+]?\d*\.\d+|\d+)', s)
    if not match:
        return None
    val = float(match.group(1))
    if any(suffix in s for suffix in ['S', 'W']) and val > 0:
        val = -val
    return val

file_path = r'C:\Users\86136\Desktop\ML-ZK\raw-soil.csv'
df = pd.read_csv(file_path, encoding='gb18030')

coords = []
for _, row in df.iterrows():
    lat = parse_cleaned_coord(row['Latitude'])
    lon = parse_cleaned_coord(row['Longitude'])
    if lat is not None and lon is not None:
        coords.append((lat, lon))

unique_coords = list(set(coords))
lats = [c[0] for c in unique_coords]
lons = [c[1] for c in unique_coords]

fig = plt.figure(figsize=(12, 10), dpi=300)
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())

ax.stock_img()
ax.add_feature(cfeature.OCEAN, facecolor='#1B2631', alpha=0.55, zorder=1)

ax.add_feature(cfeature.LAKES, edgecolor='#222222', facecolor='#1B2631', alpha=0.4, zorder=2)
ax.add_feature(cfeature.RIVERS, linewidth=0.4, edgecolor='#1B2631', alpha=0.2, zorder=2)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='#222222', zorder=3)
ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.4, edgecolor='#555555', zorder=3)

ax.scatter(lons, lats, color='#E31A23', s=120, transform=ccrs.PlateCarree(), alpha=0.2, edgecolors='none', zorder=10)
ax.scatter(lons, lats, color='#E31A23', s=40, transform=ccrs.PlateCarree(), edgecolors='white', linewidth=1.2, zorder=11, label='Sampling Site')

gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.4, color='gray', alpha=0.3)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 18, 'weight': 'bold'}
gl.ylabel_style = {'size': 18, 'weight': 'bold'}
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

plt.legend(loc='upper right', frameon=True, fontsize=20, edgecolor='black', facecolor='white', framealpha=0.9, prop={'family': 'Times New Roman', 'weight': 'bold'})

plt.savefig('Local map sampling points.tiff', dpi=600, bbox_inches='tight', format='tiff')

plt.show()