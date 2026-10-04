import sys
import numpy as np

if not hasattr(np, '_ARRAY_API'):
    sys.modules['numpy._ARRAY_API'] = np

import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import shap
from matplotlib.patches import Patch

DATA_CONFIG = {
    'train_path': r'C:\Users\86136\Desktop\ML-ZK\train.csv',
    'target': "ΔAs",
    'exclude': "ΔCd"
}

FEATURES = ['O', 'TPV', 'Clay', 'C', 'TN', 'BC-Dosage', 'OM', 'N', 'T-As', 'TP', 'Time', 'BC-EC', 'BC-pH', 'Soil-CEC']

FEATURE_GROUPS = {
    'Biochar properties': [
        'BC-pH', 'Ash', 'BC-CEC', 'PD', 'TPV', 'SSA',
        'C', 'H', 'O', 'N', 'O/C', 'H/C', '(O+N)/C', 'BC-EC'
    ],
    'Soil properties': [
        'Soil-pH', 'Soil-CEC', 'Soil-EC', 'OM', 'Sand',
        'Silt', 'Clay', 'TN', 'TP'
    ],
    'Pollution situation': [
        'T-As', 'T-Cd'
    ],
    'Experiments conditions': [
        'BC-Dosage', 'Time'
    ]
}

COLOR_PALETTE = {
    'Biochar properties': '#E64B35',
    'Soil properties': '#4DBBD5',
    'Pollution situation': '#00A087',
    'Experiments conditions': '#3C5488'
}

def main():
    random.seed(42)
    np.random.seed(42)

    if not os.path.exists(DATA_CONFIG['train_path']):
        raise FileNotFoundError(f"Missing file: {DATA_CONFIG['train_path']}")

    df = pd.read_csv(DATA_CONFIG['train_path'])

    available_features = [f for f in FEATURES if f in df.columns]

    X = df[available_features]
    y = df[DATA_CONFIG['target']]

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
    model.fit(X, y)

    background = shap.kmeans(X, 10)
    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(X)

    mean_shap = np.abs(shap_values).mean(axis=0)

    total_shap = np.sum(mean_shap)
    if total_shap > 0:
        mean_shap = mean_shap / total_shap

    shap_df = pd.DataFrame({
        'Feature': available_features,
        'SHAP_Value': mean_shap
    })

    feature_to_group = {}
    for group, feats in FEATURE_GROUPS.items():
        for f in feats:
            feature_to_group[f] = group

    shap_df['Group'] = shap_df['Feature'].map(feature_to_group)
    shap_df['Color'] = shap_df['Group'].map(COLOR_PALETTE)

    group_shap = shap_df.groupby('Group')['SHAP_Value'].sum().reset_index()
    group_shap['Color'] = group_shap['Group'].map(COLOR_PALETTE)
    group_shap = group_shap.sort_values(by='SHAP_Value', ascending=False)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'

    shap_df_sorted = shap_df.sort_values(by='SHAP_Value', ascending=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=600)
    bars = ax.barh(shap_df_sorted['Feature'], shap_df_sorted['SHAP_Value'], color=shap_df_sorted['Color'], edgecolor='black', linewidth=0.6, height=0.7)
    ax.set_xlabel('Mean(|SHAP value|)', fontsize=11, fontname='Times New Roman', fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontweight('bold')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)
    max_val = shap_df_sorted['SHAP_Value'].max()
    ax.set_xlim(0, max_val * 1.15)
    legend_elements = [
        Patch(facecolor=COLOR_PALETTE[group], edgecolor='black', linewidth=0.6, label=group)
        for group in FEATURE_GROUPS.keys() if group in shap_df_sorted['Group'].values
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=False, prop={'family': 'Times New Roman', 'size': 10, 'weight': 'bold'})
    plt.tight_layout()
    plt.savefig(r'C:\Users\86136\Desktop\ML-ZK\As-ANN-SHAP-OPTIMIZATION-1.tiff', format='tiff', dpi=600, bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(6, 6), dpi=600)
    ax_pie = fig.add_subplot(111, projection='3d')
    ax_pie.set_box_aspect((1, 1, 0.15))

    angles = np.zeros(len(group_shap) + 1)
    angles[0] = 140
    for i, val in enumerate(group_shap['SHAP_Value']):
        angles[i+1] = angles[i] + val * 360

    num_groups = len(group_shap)
    explode_distances = np.linspace(0.12, 0.04, num_groups)

    for i in range(num_groups):
        theta1 = np.radians(angles[i])
        theta2 = np.radians(angles[i+1])
        mid_theta = (theta1 + theta2) / 2

        offset_r = explode_distances[i]
        xo = offset_r * np.cos(mid_theta)
        yo = offset_r * np.sin(mid_theta)

        t = np.linspace(theta1, theta2, 100)
        v = np.linspace(0, 1, 2)
        T, V = np.meshgrid(t, v)

        X_mesh = xo + V * np.cos(T)
        Y_mesh = yo + V * np.sin(T)
        Z_mesh = np.zeros_like(X_mesh)

        ax_pie.plot_surface(X_mesh, Y_mesh, Z_mesh, color=group_shap['Color'].iloc[i], antialiased=True, shade=False)
        ax_pie.plot_surface(X_mesh, Y_mesh, Z_mesh - 0.18, color=group_shap['Color'].iloc[i], antialiased=True, shade=True)

        X_side1 = xo + np.array([[0, 0], [np.cos(theta1), np.cos(theta1)]])
        Y_side1 = yo + np.array([[0, 0], [np.sin(theta1), np.sin(theta1)]])
        Z_side1 = np.array([[0, -0.18], [0, -0.18]])
        ax_pie.plot_surface(X_side1, Y_side1, Z_side1, color=group_shap['Color'].iloc[i], antialiased=True, shade=True)

        X_side2 = xo + np.array([[0, 0], [np.cos(theta2), np.cos(theta2)]])
        Y_side2 = yo + np.array([[0, 0], [np.sin(theta2), np.sin(theta2)]])
        Z_side2 = np.array([[0, -0.18], [0, -0.18]])
        ax_pie.plot_surface(X_side2, Y_side2, Z_side2, color=group_shap['Color'].iloc[i], antialiased=True, shade=True)

        t_edge = np.linspace(theta1, theta2, 100)
        X_edge = xo + np.array([np.cos(t_edge), np.cos(t_edge)])
        Y_edge = yo + np.array([np.sin(t_edge), np.sin(t_edge)])
        Z_edge = np.array([np.zeros_like(t_edge), -0.18 * np.ones_like(t_edge)])
        ax_pie.plot_surface(X_edge, Y_edge, Z_edge, color=group_shap['Color'].iloc[i], antialiased=True, shade=True)

        val_pct = group_shap['SHAP_Value'].iloc[i] * 100
        pct_str = f"{val_pct:.2f}%"

        if val_pct < 1.0:
            r_text = 1.25
        else:
            r_text = 0.65

        text_x = xo + r_text * np.cos(mid_theta)
        text_y = yo + r_text * np.sin(mid_theta)
        ax_pie.text(text_x, text_y, 0.05, pct_str, ha='center', va='center', family='Times New Roman', weight='bold', size=10, color='black', zorder=100)

    ax_pie.view_init(elev=45, azim=-60)
    ax_pie.set_axis_off()

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(r'C:\Users\86136\Desktop\ML-ZK\As-ANN-SHAP-PIE-1.tiff', format='tiff', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == '__main__':
    main()