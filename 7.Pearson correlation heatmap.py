import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['mathtext.default'] = 'regular'

file_path = r"C:\Users\86136\Desktop\ML-ZK\train.csv"
df = pd.read_csv(file_path, encoding='utf-8-sig')

targets = ['ΔAs', 'ΔCd']
features = [col for col in df.columns if col not in targets]
ordered_cols = features + targets
df_ordered = df[ordered_cols]

corr = df_ordered.corr(method='pearson')
n_vars = len(ordered_cols)

fig, ax = plt.subplots(figsize=(14, 14))

cmap = plt.get_cmap('bwr')
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

ax.set_xlim(-0.5, n_vars - 0.5)
ax.set_ylim(-0.5, n_vars - 0.5)
ax.invert_yaxis()

for i in range(n_vars):
    for j in range(n_vars):
        r = corr.iloc[i, j]

        rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=0.8, edgecolor='#D3D3D3', facecolor='none', zorder=1)
        ax.add_patch(rect)

        if i == j:
            color_1 = cmap(norm(1.0))
            circle = patches.Circle((j, i), radius=0.38, facecolor=color_1, edgecolor='none', zorder=3)
            ax.add_patch(circle)
        elif j > i:
            color = cmap(norm(r))
            ax.text(j, i, f"{r:.2f}", ha='center', va='center', fontsize=12, fontweight='bold', color=color, zorder=4)
        else:
            color = cmap(norm(r))
            abs_r = abs(r)
            width = 0.85
            height = 0.85 * (1 - abs_r**0.8)
            angle = -45 if r > 0 else 45

            ellipse = patches.Ellipse(xy=(j, i), width=width, height=height, angle=angle, facecolor=color, edgecolor='none', alpha=0.85, zorder=2)
            ax.add_patch(ellipse)
            ax.text(j, i, f"{r:.2f}", ha='center', va='center', fontsize=11.5, fontweight='bold', color='black', zorder=4)

ax.set_xticks(range(n_vars))
ax.set_yticks(range(n_vars))
ax.set_xticklabels(ordered_cols, rotation=45, ha='right', fontsize=13, fontweight='bold')
ax.set_yticklabels(ordered_cols, rotation=0, fontsize=13, fontweight='bold')

ax.tick_params(axis='both', which='both', length=0)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_color('black')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.25)

cbar = fig.colorbar(sm, cax=cax)
cbar.set_ticks(np.arange(-1.0, 1.1, 0.2))
cbar.ax.tick_params(labelsize=12, width=1.2)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
cbar.outline.set_linewidth(1.5)
cbar.outline.set_edgecolor('black')

plt.tight_layout()

output_path = r"C:\Users\86136\Desktop\ML-ZK\Pearson correlation heatmap.tiff"
plt.savefig(
    output_path,
    dpi=600,
    format='tiff',
    bbox_inches='tight',
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close()