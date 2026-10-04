import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 16

def plot_distributions_final(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return

    cols = df.columns
    num_cols = len(cols)
    cols_per_row = 5  # 改为五列
    num_rows = math.ceil(num_cols / cols_per_row)

    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(20, 3.8 * num_rows))
    axes_flat = axes.flatten()

    for i, col_name in enumerate(cols):
        ax = axes_flat[i]
        data = df[col_name].dropna()

        mean_val = data.mean()
        std_val = data.std()

        sns.histplot(data, ax=ax, bins=30, kde=False, color='skyblue', edgecolor='black', alpha=0.8)

        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.2)
        ax.axvline(mean_val + 3 * std_val, color='green', linestyle='--', linewidth=1.2)
        ax.axvline(mean_val - 3 * std_val, color='green', linestyle='--', linewidth=1.2)

        margin = 3.2 * std_val
        ax.set_xlim(mean_val - margin, mean_val + margin)

        ax.set_title(col_name, fontsize=20, fontweight='bold')
        ax.set_xlabel('Values', fontsize=18, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=18, fontweight='bold')

        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(14)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(14)

    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', lw=2, label='Mean'),
        Line2D([0], [0], color='green', linestyle='--', lw=2, label='Mean ± 3Std Dev')
    ]

    if num_cols < len(axes_flat):
        target_ax = axes_flat[num_cols]
        target_ax.axis('off')
        target_ax.legend(handles=legend_elements, loc='center', prop={'size': 18, 'weight': 'bold'}, frameon=True)
    else:
        axes_flat[-1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 18, 'weight': 'bold'})

    for j in range(num_cols + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()

    output_path = r'C:\Users\86136\Desktop\ML-ZK\Data distribution.tiff'
    try:
        plt.savefig(output_path, dpi=600, format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    except:
        plt.savefig(output_path, dpi=600, format='tiff')

    plt.show()

file_path = r'C:\Users\86136\Desktop\ML-ZK\train.csv'
plot_distributions_final(file_path)