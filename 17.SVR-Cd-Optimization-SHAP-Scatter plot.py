import sys
import numpy as np

if not hasattr(np, '_ARRAY_API'):
    sys.modules['numpy._ARRAY_API'] = np

import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import shap

DATA_CONFIG = {
    'train_path': r'C:\Users\86136\Desktop\ML-ZK\train.csv',
    'target': "ΔCd",
    'exclude': "ΔAs"
}

FEATURES = ['O', 'TPV', 'Ash', 'Silt', 'SSA', 'TN', 'BC-Dosage', 'OM', 'N', 'T-Cd', 'TP', 'Time', 'BC-EC', 'BC-pH', 'Soil-pH']

def main():
    random.seed(42)
    np.random.seed(42)

    if not os.path.exists(DATA_CONFIG['train_path']):
        raise FileNotFoundError(f"Missing file: {DATA_CONFIG['train_path']}")

    df = pd.read_csv(DATA_CONFIG['train_path'])

    available_features = [f for f in FEATURES if f in df.columns]

    X = df[available_features]
    y = df[DATA_CONFIG['target']]

    best_params_svr = {
        'kernel': 'rbf',
        'C': 51.2765147497251763,
        'gamma': 0.08641204502744947,
        'epsilon': 0.6941158123710329
    }

    model = SVR(**best_params_svr)
    model.fit(X, y)

    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)

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
    plt.savefig(r'C:\Users\86136\Desktop\ML-ZK\SVR-Cd-Optimization-SHAP-Pie chart.tiff', format='tiff', dpi=600, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()