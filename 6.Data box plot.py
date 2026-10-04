import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

df = pd.read_csv(r"C:\Users\86136\Desktop\ML-ZK\train data.csv")

cols_info = [
    ('SSA', 'SSA (m$^2$/g)', 1),
    ('C', 'C (%)', 1),
    ('H', 'H (%)', 1),
    ('O', 'O (%)', 1),
    ('N', 'N (%)', 1),
    ('O/C', 'O/C', 1),
    ('H/C', 'H/C', 1),
    ('(O+N)/C', '(O+N)/C', 1),
    ('BC-EC', 'BC-EC (mS/cm)', 1),
    ('Soil-pH', 'pH', 1),
    ('Soil-CEC', 'CEC (cmol(+)/kg)', 1),
    ('Soil-EC', 'EC (mS/cm)', 1),
    ('OM', r'OM ($\times 10^3$ mg/kg)', 1000),
    ('Sand', 'sand (%)', 1),
    ('Silt', 'silt (%)', 1),
    ('Clay', 'clay (%)', 1),
    ('TN', 'TN (g/kg)', 1),
    ('TP', 'TP (g/kg)', 1),
    ('T-As', r'T-As ($\times 10^3$ mg/kg)', 1000),
    ('T-Cd', 'T-Cd (mg/kg)', 1),
    ('BC-pH', 'BC-pH', 1),
    ('Ash', 'Ash (%)', 1),
    ('BC-CEC', 'BC-CEC (cmol(+)/kg)', 1),
    ('PD', 'PD (nm)', 1),
    ('TPV', 'TPV (cm$^3$/g)', 1),
    ('BC-Dosage', r'BC-Dosage ($\times 10^3$ kg/ha)', 1000),
    ('Time', 'Time (d)', 1),
    ('ΔAs', r'$\Delta$As (%)', 1),
    ('ΔCd', r'$\Delta$Cd (%)', 1)
]

color_palette = ['#56C8FF', '#FF9ebe', '#96C982', '#FFD700', '#FF6B6B', '#A78BFA']
output_dir = r"C:\Users\86136\Desktop\ML-ZK"

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

info_dict = {item[0]: (item[1], item[2]) for item in cols_info}
existing_cols = [col for col in df.columns if col in info_dict]
valid_cols = [(col, info_dict[col][0], info_dict[col][1]) for col in existing_cols]

n_cols = 5
n_rows = int(np.ceil(len(valid_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.6))
axes = axes.flatten()

for idx, (col_name, label_name, scale_factor) in enumerate(valid_cols):
    ax = axes[idx]
    data = df[col_name].dropna().values / scale_factor
    color = color_palette[idx % n_cols]

    bp = ax.boxplot(data, positions=[1], widths=0.4, patch_artist=True, showfliers=False)
    for box in bp['boxes']:
        box.set(facecolor=color, alpha=0.4, edgecolor=color, linewidth=2.0)
    for median in bp['medians']:
        median.set(color=color, linewidth=2.8)
    for whisker in bp['whiskers']:
        whisker.set(color=color, linewidth=2.0)
    for cap in bp['caps']:
        cap.set(color=color, linewidth=2.0)

    try:
        kde = gaussian_kde(data)
        y_vals = np.linspace(min(data), max(data), 200)
        x_vals = kde(y_vals)
        x_vals_scaled = 1.4 + (x_vals / max(x_vals)) * 0.4
        ax.plot(x_vals_scaled, y_vals, color=color, linewidth=2.0)

        jitter = np.random.uniform(-0.12, 0.12, size=len(data))
        x_jitter = 1.4 + (kde(data) / max(x_vals)) * 0.4 * 0.4 + 0.05 + jitter
        ax.scatter(x_jitter, data, color='white', edgecolor=color, alpha=0.8, s=16, linewidth=1.4)
    except:
        jitter = np.random.uniform(1.3, 1.7, size=len(data))
        ax.scatter(jitter, data, color='white', edgecolor=color, alpha=0.8, s=16, linewidth=1.4)

    ax.set_xlim(0.6, 2.0)
    ax.set_xticks([1.3])
    ax.set_xticklabels([col_name], fontsize=16, fontweight='bold')
    ax.set_ylabel(label_name, fontsize=13, fontweight='bold')

    ax.tick_params(axis='y', direction='in', left=True, right=False, labelsize=12, width=1.5, length=4)
    ax.tick_params(axis='x', bottom=False, top=False)

    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

for idx in range(len(valid_cols), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(hspace=0.48, wspace=0.38)

filepath = os.path.join(output_dir, "Box-Plots.tiff")
plt.savefig(filepath, dpi=600, format='tiff', bbox_inches='tight')
plt.close()