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

DATA_CONFIG = {
    'train_path': r'C:\Users\86136\Desktop\ML-ZK\train.csv',
    'target': "ΔAs",
    'exclude': "ΔCd"
}

FEATURES = ['O', 'TPV', 'Clay', 'C', 'TN', 'BC-Dosage', 'OM', 'N', 'T-As', 'TP', 'Time', 'BC-EC', 'BC-pH', 'Soil-CEC']

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

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'

    plt.figure(figsize=(7.5, 6.5), dpi=600)

    shap.summary_plot(shap_values, X, plot_type="dot", show=False)

    ax = plt.gca()
    ax.set_xlabel('SHAP value', fontsize=13, fontname='Times New Roman', fontweight='bold')
    ax.set_xlim(-100, 100)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontweight('bold')
        label.set_fontsize(12)

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)

    cf = plt.gcf()
    for axis in cf.get_axes():
        if axis != ax:
            axis.set_ylabel('Feature value', fontsize=13, fontname='Times New Roman', fontweight='bold')
            for label in axis.get_yticklabels():
                label.set_fontname('Times New Roman')
                label.set_fontweight('bold')
                label.set_fontsize(12)

    plt.tight_layout()
    plt.savefig(r'C:\Users\86136\Desktop\ML-ZK\ANN-As-Optimization-SHAP-Scatter plot.tiff', format='tiff', dpi=600, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()