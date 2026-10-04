import sys
import os
import numpy as np

if not hasattr(np, '_ARRAY_API'):
    sys.modules['numpy._ARRAY_API'] = np

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

DATA_CONFIG = {
    'train_path': r'C:\Users\86136\Desktop\ML-ZK\train data.csv',
    'target': "ΔAs",
    'exclude': "ΔCd"
}

FEATURES = ['O', 'TPV', 'Clay', 'C', 'TN', 'BC-Dosage', 'OM', 'N', 'TP', 'Time', 'BC-EC', 'BC-pH', 'Soil-CEC']

FEATURE_UNITS = {
    'O': '%', 'TPV': 'cm³/g', 'Clay': '%', 'C': '%', 'TN': 'g/kg',
    'BC-Dosage': 'kg/ha', 'OM': 'mg/kg', 'N': '%', 'T-As': 'mg/kg',
    'TP': 'g/kg', 'Time': 'd', 'BC-EC': 'mS/cm', 'BC-pH': '-', 'Soil-CEC': 'cmol(+)/kg'
}

def plot_2d_pdp(pair, X_raw, scaler, model, output_path):
    feat_x, feat_y = pair
    x_vals = np.linspace(X_raw[feat_x].min(), X_raw[feat_x].max(), 50)
    y_vals = np.linspace(X_raw[feat_y].min(), X_raw[feat_y].max(), 50)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

    Z = np.zeros_like(X_grid)
    X_temp = X_raw.copy()
    for i in range(50):
        for j in range(50):
            X_temp[feat_x] = X_grid[i, j]
            X_temp[feat_y] = Y_grid[i, j]
            X_temp_scaled = scaler.transform(X_temp)
            preds = model.predict(X_temp_scaled)
            Z[i, j] = np.mean(preds)

    colors = ["#E0F7FA", "#B2EBF2", "#4DD0E1", "#29B6F6", "#1E88E5", "#3949AB"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_blue", colors, N=100)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)

    contour = ax.contourf(X_grid, Y_grid, Z, levels=10, cmap=custom_cmap)
    lines = ax.contour(X_grid, Y_grid, Z, levels=10, colors='white', linewidths=0.5, alpha=0.6)
    ax.clabel(lines, inline=True, fontsize=12, fmt='%.1f', colors='black')

    x_sample = X_raw[feat_x].dropna()
    if len(x_sample) > 50:
        x_sample = x_sample.sample(50, random_state=42)
    y_min, y_max = ax.get_ylim()
    rug_height_y = (y_max - y_min) * 0.04
    ax.vlines(x_sample, ymin=y_min, ymax=y_min + rug_height_y, color='black', linewidth=0.6, alpha=0.7)

    y_sample = X_raw[feat_y].dropna()
    if len(y_sample) > 50:
        y_sample = y_sample.sample(50, random_state=42)
    x_min, x_max = ax.get_xlim()
    rug_height_x = (x_max - x_min) * 0.04
    ax.hlines(y_sample, xmin=x_min, xmax=x_min + rug_height_x, color='black', linewidth=0.6, alpha=0.7)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    x_unit = FEATURE_UNITS.get(feat_x, '-')
    ax.set_xlabel(f"{feat_x} ({x_unit})" if x_unit != '-' else feat_x, fontsize=18, fontname='Times New Roman', fontweight='bold')
    y_unit = FEATURE_UNITS.get(feat_y, '-')
    ax.set_ylabel(f"{feat_y} ({y_unit})" if y_unit != '-' else feat_y, fontsize=18, fontname='Times New Roman', fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=15)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontweight('bold')

    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)
        spine.set_visible(True)

    plt.tight_layout()
    plt.savefig(output_path, format='tiff', dpi=600, bbox_inches='tight')
    plt.close()

def main():
    if not os.path.exists(DATA_CONFIG['train_path']):
        raise FileNotFoundError(f"Missing file: {DATA_CONFIG['train_path']}")

    df = pd.read_csv(DATA_CONFIG['train_path'])
    all_needed = FEATURES + ['T-As']
    available_features = [f for f in FEATURES if f in df.columns]

    if 'T-As' not in df.columns:
        raise ValueError("Dataset must contain 'T-As'.")

    X_raw = df[all_needed]
    y = df[DATA_CONFIG['target']]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=all_needed)

    best_params_ann = {
        'activation': 'relu',
        'alpha': 0.01330112814447008,
        'learning_rate_init': 0.053698212690683665,
        'hidden_layer_sizes': (80, 64, 96),
        'solver': 'adam',
        'max_iter': 1000,
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.1
    }

    model = MLPRegressor(**best_params_ann)
    model.fit(X_scaled, y)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'

    pair1 = ('BC-Dosage', 'N')
    pair2 = ('Time', 'BC-EC')

    plot_2d_pdp(pair1, X_raw, scaler, model, r'C:\Users\86136\Desktop\ML-ZK\As-ANN-2DPDP-BC-Dosage-N.tiff')
    plot_2d_pdp(pair2, X_raw, scaler, model, r'C:\Users\86136\Desktop\ML-ZK\As-ANN-2DPDP-Time-BC-EC.tiff')

if __name__ == '__main__':
    main()