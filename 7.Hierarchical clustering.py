import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette

file_path = r'C:\Users\86136\Desktop\ML-ZK\train.csv'
df = pd.read_csv(file_path, encoding='utf-8-sig')

targets = ['ΔAs', 'ΔCd']
feature_cols = [col for col in df.columns if col not in targets]
df_features = df[feature_cols]

Z = linkage(df_features.T, method='ward')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.size'] = 28

set_link_color_palette(['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00'])

fig, ax = plt.subplots(figsize=(20, 14))

max_dist = max(Z[:, 2])
threshold = 0.7 * max_dist

print("Maximum linkage distance (root height):", max_dist)
print("Color threshold (0.7 * max):", threshold)

dendro = dendrogram(
    Z,
    labels=df_features.columns.tolist(),
    leaf_rotation=90,
    leaf_font_size=34,
    color_threshold=threshold,
    above_threshold_color='black',
    ax=ax
)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_color('black')

ax.set_ylabel('Distance', fontsize=38, fontweight='bold', labelpad=20)
ax.set_xlabel('Features', fontsize=38, fontweight='bold', labelpad=25)

plt.xticks(fontsize=30, fontweight='bold')
plt.yticks(fontsize=30, fontweight='bold')

plt.tight_layout()
output_name = r'C:\Users\86136\Desktop\ML-ZK\Hierarchical Clustering.tif'
plt.savefig(output_name, dpi=600, format='tif',
            pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')

set_link_color_palette(None)
plt.show()