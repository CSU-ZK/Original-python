import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

path_origin = r'C:\Users\86136\Desktop\ML-ZK\raw-all.csv'
path_knn = r'C:\Users\86136\Desktop\ML-ZK\raw-all-KNN.csv'
path_rf = r'C:\Users\86136\Desktop\ML-ZK\raw-all-RF.csv'
tiff_output = r'C:\Users\86136\Desktop\Comparison before and after supplementary data.tiff'

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2.0
plt.rcParams["font.size"] = 16

plot_cols = ['BC-pH', 'Ash', 'BC-CEC', 'PD', 'TPV', 'SSA', 'C', 'H', 'O', 'N', 'BC-EC', 'TP']

df_origin = pd.read_csv(path_origin, encoding='gb18030')
df_knn = pd.read_csv(path_knn, encoding='utf-8-sig')
df_rf = pd.read_csv(path_rf, encoding='utf-8-sig')

fig, axes = plt.subplots(4, 3, figsize=(18, 24))
axes = axes.flatten()

colors = ["#7F8C8D", "#3498DB", "#E74C3C"]

for i, col in enumerate(plot_cols):
    if col in df_origin.columns:
        clean_title = col.split('(')[0].strip()
        orig_data = df_origin[col].dropna()

        plot_df = pd.DataFrame({
            'Value': np.concatenate([orig_data.values, df_knn[col].values, df_rf[col].values]),
            'Method': (['Original'] * len(orig_data) + ['KNN'] * len(df_knn) + ['RF'] * len(df_rf))
        })

        sns.boxplot(x='Method', y='Value', data=plot_df, ax=axes[i],
                    hue='Method', palette=colors, legend=False,
                    width=0.5, linewidth=2.5)

        axes[i].set_title(clean_title, fontsize=24, fontweight='bold')
        axes[i].set_xlabel('', fontsize=18, fontweight='bold')
        axes[i].set_ylabel('Values', fontsize=20, fontweight='bold')

        for label in axes[i].get_xticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(18)
        for label in axes[i].get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(16)

plt.tight_layout()
plt.savefig(tiff_output, dpi=600, format='tiff', bbox_inches='tight')
plt.show()