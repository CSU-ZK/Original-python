import sys
import numpy as np

if not hasattr(np, '_ARRAY_API'):
    sys.modules['numpy._ARRAY_API'] = np

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import shap
from matplotlib.patches import Patch

DATA_CONFIG = {
    'train_path': r'C:\Users\86136\Desktop\ML-ZK\train.csv',
    'target': "ΔCd",
    'exclude': "ΔAs"
}

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
    if not os.path.exists(DATA_CONFIG['train_path']):
        raise FileNotFoundError(f"Missing file: {DATA_CONFIG['train_path']}")

    df = pd.read_csv(DATA_CONFIG['train_path'])

    all_features = []
    for group, feats in FEATURE_GROUPS.items():
        all_features.extend(feats)

    available_features = [f for f in all_features if f in df.columns]

    X = df[available_features]
    y = df[DATA_CONFIG['target']]

    best_params_svr = {
        'C': 91.76333540598984,
        'gamma': 0.04837371360578683,
        'epsilon': 0.9933172124085634
    }

    model = SVR(**best_params_svr)
    model.fit(X, y)

    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)

    mean_shap = np.abs(shap_values.values).mean(axis=0)

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

    shap_df = shap_df.sort_values(by='SHAP_Value', ascending=True)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['font.size'] = 12

    fig, ax = plt.subplots(figsize=(8, 7), dpi=600)

    bars = ax.barh(shap_df['Feature'], shap_df['SHAP_Value'], color=shap_df['Color'], edgecolor='black', linewidth=0.6, height=0.7)

    ax.set_xlabel('Mean(|SHAP value|)', fontsize=14, fontname='Times New Roman', fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)

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

    max_val = shap_df['SHAP_Value'].max()
    ax.set_xlim(0, 0.16)

    legend_elements = [
        Patch(facecolor=COLOR_PALETTE[group], edgecolor='black', linewidth=0.6, label=group)
        for group in FEATURE_GROUPS.keys() if group in shap_df['Group'].values
    ]

    leg = ax.legend(handles=legend_elements, loc='lower right', frameon=False, prop={'family': 'Times New Roman', 'size': 14, 'weight': 'bold'})

    plt.tight_layout()
    output_filename = r'C:\Users\86136\Desktop\ML-ZK\Cd-SVR-SHAP.tiff'
    plt.savefig(output_filename, format='tiff', dpi=600, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()