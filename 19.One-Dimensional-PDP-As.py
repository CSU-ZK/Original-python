import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

if not hasattr(np, '_ARRAY_API'):
    sys.modules['numpy._ARRAY_API'] = np

DATA_CONFIG = {
    'train_path': r'C:\Users\86136\Desktop\ML-ZK\train data.csv',
    'target': "ΔAs",
    'exclude': "ΔCd"
}

FEATURES = ['O', 'TPV', 'Clay', 'C', 'TN', 'BC-Dosage', 'OM', 'N', 'T-As', 'TP', 'Time', 'BC-EC', 'BC-pH', 'Soil-CEC']

FEATURE_UNITS = {
    'O': '%', 'TPV': 'cm³/g', 'Clay': '%', 'C': '%', 'TN': 'g/kg',
    'BC-Dosage': 'kg/ha', 'OM': 'mg/kg', 'N': '%', 'T-As': 'mg/kg',
    'TP': 'g/kg', 'Time': 'd', 'BC-EC': 'mS/cm', 'BC-pH': '-', 'Soil-CEC': 'cmol(+)/kg'
}

def main():
    if not os.path.exists(DATA_CONFIG['train_path']):
        raise FileNotFoundError(f"Missing file: {DATA_CONFIG['train_path']}")

    df = pd.read_csv(DATA_CONFIG['train_path'])
    available_features = [f for f in FEATURES if f in df.columns]

    X_raw = df[available_features]
    y = df[DATA_CONFIG['target']]

    scaler = StandardScaler()
    X_scaled_base = pd.DataFrame(scaler.fit_transform(X_raw), columns=available_features)

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
    model.fit(X_scaled_base, y)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'

    fig = plt.figure(figsize=(16, 14), dpi=600)
    gs = GridSpec(4, 4, figure=fig)

    axes_list = []
    for r in range(3):
        for c in range(4):
            axes_list.append(fig.add_subplot(gs[r, c]))

    axes_list.append(fig.add_subplot(gs[3, 1]))
    axes_list.append(fig.add_subplot(gs[3, 2]))

    for i, feature in enumerate(available_features):
        ax = axes_list[i]

        grid_values_raw = np.linspace(X_raw[feature].min(), X_raw[feature].max(), 100)

        feat_idx = available_features.index(feature)
        grid_values_scaled = (grid_values_raw - scaler.mean_[feat_idx]) / np.sqrt(scaler.var_[feat_idx])

        average_preds = []
        X_temp_scaled = X_scaled_base.copy()

        for val_scaled in grid_values_scaled:
            X_temp_scaled[feature] = val_scaled
            preds = model.predict(X_temp_scaled)
            average_preds.append(np.mean(preds))

        ax.plot(grid_values_raw, average_preds, color='#1f77b4', linewidth=1.5)

        sample_data = X_raw[feature].dropna()
        if len(sample_data) > 100:
            sample_data = sample_data.sample(100, random_state=42)

        y_min, y_max = ax.get_ylim()
        rug_height = (y_max - y_min) * 0.04
        ax.vlines(sample_data, ymin=y_min, ymax=y_min + rug_height, color='black', linewidth=0.6, alpha=0.7)
        ax.set_ylim(y_min, y_max)

        unit = FEATURE_UNITS.get(feature, '-')
        xlabel_str = f"{feature}" if unit == '-' else f"{feature} ({unit})"
        ax.set_xlabel(xlabel_str, fontsize=12, fontname='Times New Roman', fontweight='bold')

        if i in [0, 4, 8] or i == 12:
            ax.set_ylabel('Predicted ΔAs (%)', fontsize=12, fontname='Times New Roman', fontweight='bold')

        ax.tick_params(axis='both', which='major', labelsize=11)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontweight('bold')

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.8)

    for j in range(len(available_features), len(axes_list)):
        fig.delaxes(axes_list[j])

    plt.tight_layout()
    output_path = r'C:\Users\86136\Desktop\ML-ZK\As-ANN-One Dimensional-PDP.tiff'
    plt.savefig(output_path, format='tiff', dpi=600, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()