import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import shap

if not hasattr(np, '_ARRAY_API'):
    sys.modules['numpy._ARRAY_API'] = np

DATA_PATH_NORM = r'C:\Users\86136\Desktop\ML-ZK\train.csv'
DATA_PATH_RAW = r'C:\Users\86136\Desktop\ML-ZK\train data.csv'
OUT_DIR = r'C:\Users\86136\Desktop\ML-ZK'

TARGET_FEATURES = ['N', 'Time', 'BC-Dosage', 'BC-EC', 'OM', 'TN']

FEATURES_AS = ['O', 'TPV', 'Clay', 'C', 'TN', 'BC-Dosage', 'OM', 'N', 'T-As', 'TP', 'Time', 'BC-EC', 'BC-pH', 'Soil-CEC']
FEATURES_CD = ['O', 'TPV', 'Ash', 'Silt', 'SSA', 'TN', 'BC-Dosage', 'OM', 'N', 'T-Cd', 'TP', 'Time', 'BC-EC', 'BC-pH', 'Soil-pH']

color_q1 = '#FCE4EC'
color_q2 = '#FFFDE7'
color_q3 = '#E1F5FE'
color_q4 = '#E8F5E9'

colors_colorbar = [
    '#0080ca', '#009fe3', '#40b4e5', '#80ccf0', '#bce2f7',
    '#f9c9db', '#f593b8', '#ee5c93', '#e60067', '#d0005a'
]
custom_cmap = LinearSegmentedColormap.from_list('custom_pink_blue', colors_colorbar, N=256)

def main():
    if not os.path.exists(DATA_PATH_NORM):
        raise FileNotFoundError(f"Missing normalized file: {DATA_PATH_NORM}")
    if not os.path.exists(DATA_PATH_RAW):
        raise FileNotFoundError(f"Missing raw file: {DATA_PATH_RAW}")

    df_norm = pd.read_csv(DATA_PATH_NORM)
    df_raw = pd.read_csv(DATA_PATH_RAW)

    avail_as = [f for f in FEATURES_AS if f in df_norm.columns]
    X_as = df_norm[avail_as]
    y_as = df_norm['ΔAs']

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
    model_as = MLPRegressor(**best_params_ann)
    model_as.fit(X_as, y_as)

    background = shap.kmeans(X_as, 10)
    explainer_as = shap.KernelExplainer(model_as.predict, background)
    shap_vals_as_raw = explainer_as.shap_values(X_as)

    avail_cd = [f for f in FEATURES_CD if f in df_norm.columns]
    X_cd = df_norm[avail_cd]
    y_cd = df_norm['ΔCd']

    best_params_svr = {
        'kernel': 'rbf',
        'C': 51.2765147497251763,
        'gamma': 0.08641204502744947,
        'epsilon': 0.6941158123710329
    }
    model_cd = SVR(**best_params_svr)
    model_cd.fit(X_cd, y_cd)

    explainer_cd = shap.Explainer(model_cd.predict, X_cd)
    shap_obj_cd = explainer_cd(X_cd)
    shap_vals_cd_raw = shap_obj_cd.values if hasattr(shap_obj_cd, 'values') else shap_obj_cd

    print("\n=================================================================")
    print("      SUMMARY OF FEATURE PHYSICAL RANGES FOR PAPER WRITING       ")
    print("=================================================================\n")

    for idx, feat in enumerate(TARGET_FEATURES):
        if feat not in X_as.columns or feat not in X_cd.columns:
            continue

        idx_as = list(X_as.columns).index(feat)
        idx_cd = list(X_cd.columns).index(feat)

        x_vals = shap_vals_as_raw[:, idx_as]
        y_vals = shap_vals_cd_raw[:, idx_cd]

        raw_vals = df_raw[feat].values

        m_q1 = (x_vals >= 0) & (y_vals >= 0)
        m_q2 = (x_vals < 0)  & (y_vals >= 0)
        m_q3 = (x_vals < 0)  & (y_vals < 0)
        m_q4 = (x_vals >= 0) & (y_vals < 0)

        print(f"Feature: {feat}")

        if np.sum(m_q3) > 0:
            q3_min, q3_max = raw_vals[m_q3].min(), raw_vals[m_q3].max()
            print(f"  [Q3 - Dual Reduction (-As, -Cd)]: Range is {q3_min:.4f}–{q3_max:.4f}")
        else:
            print("  [Q3 - Dual Reduction (-As, -Cd)]: No points found")

        if np.sum(m_q1) > 0:
            q1_min, q1_max = raw_vals[m_q1].min(), raw_vals[m_q1].max()
            print(f"  [Q1 - Dual Increase  (+As, +Cd)]: Range is {q1_min:.4f}–{q1_max:.4f}")

        if np.sum(m_q2) > 0:
            q2_min, q2_max = raw_vals[m_q2].min(), raw_vals[m_q2].max()
            print(f"  [Q2 - Reduce As, Increase Cd  ]: Range is {q2_min:.4f}–{q2_max:.4f}")

        if np.sum(m_q4) > 0:
            q4_min, q4_max = raw_vals[m_q4].min(), raw_vals[m_q4].max()
            print(f"  [Q4 - Increase As, Reduce Cd  ]: Range is {q4_min:.4f}–{q4_max:.4f}")

        print("-" * 65)

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']

        fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=600)

        x_p1, x_p99 = np.percentile(x_vals, [1, 99])
        y_p1, y_p99 = np.percentile(y_vals, [1, 99])

        x_min_v = min(x_p1, -abs(x_p99) * 0.08 if x_p99 != 0 else -0.1)
        x_max_v = max(x_p99, abs(x_p1) * 0.08 if x_p1 != 0 else 0.1)
        y_min_v = min(y_p1, -abs(y_p99) * 0.08 if y_p99 != 0 else -0.1)
        y_max_v = max(y_p99, abs(y_p1) * 0.08 if y_p1 != 0 else 0.1)

        x_pad = (x_max_v - x_min_v) * 0.12 if x_max_v != x_min_v else 0.1
        y_pad = (y_max_v - y_min_v) * 0.12 if y_max_v != y_min_v else 0.1

        xlim = (x_min_v - x_pad, x_max_v + x_pad)
        ylim = (y_min_v - y_pad, y_max_v + y_pad)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.fill_between([xlim[0], 0], ylim[0], 0, color=color_q3, zorder=0)
        ax.fill_between([xlim[0], 0], 0, ylim[1], color=color_q2, zorder=0)
        ax.fill_between([0, xlim[1]], ylim[0], 0, color=color_q4, zorder=0)
        ax.fill_between([0, xlim[1]], 0, ylim[1], color=color_q1, zorder=0)

        ax.axhline(0, color='#666666', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)
        ax.axvline(0, color='#666666', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)

        c_min = raw_vals.min()
        c_max = raw_vals.max()
        norm = Normalize(vmin=c_min, vmax=c_max)

        sc = ax.scatter(
            x_vals, y_vals,
            c=raw_vals,
            cmap=custom_cmap,
            norm=norm,
            s=38,
            edgecolors='white',
            linewidths=0.35,
            alpha=0.88,
            zorder=2
        )

        ax.set_xlabel(r'SHAP value of $\Delta$As', fontsize=14, fontweight='bold', fontfamily='Times New Roman', labelpad=6)
        ax.set_ylabel(r'SHAP value of $\Delta$Cd', fontsize=14, fontweight='bold', fontfamily='Times New Roman', labelpad=6)

        ax.tick_params(axis='both', which='major', labelsize=12, direction='in', length=5, width=1.0)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontname('Times New Roman')
            tick.set_fontweight('bold')

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')

        cbar = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.046)
        cbar.set_ticks([c_min, c_max])
        cbar.set_ticklabels(['Low', 'High'])
        cbar.ax.tick_params(labelsize=12, length=0)

        for label in cbar.ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontweight('bold')

        cbar.set_label(feat, fontsize=13, fontweight='bold', fontfamily='Times New Roman', labelpad=-20)
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        save_path = os.path.join(OUT_DIR, f'Interaction quadrant diagrams {feat}.tiff')
        plt.savefig(save_path, format='tiff', dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()

if __name__ == '__main__':
    main()