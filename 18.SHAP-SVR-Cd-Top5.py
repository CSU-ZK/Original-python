import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

if not hasattr(np, '_ARRAY_API'):
    sys.modules['numpy._ARRAY_API'] = np

try:
    import shap
except AttributeError:
    import importlib
    import cv2
    if not hasattr(cv2, '_ARRAY_API') and hasattr(np, '_ARRAY_API'):
        cv2._ARRAY_API = np._ARRAY_API
    importlib.reload(shap)

DATA_CONFIG = {
    'train_path': r'C:\Users\86136\Desktop\ML-ZK\train data.csv',
    'target': "ΔCd",
    'exclude': "ΔAs"
}

FEATURES = ['O', 'TPV', 'Ash', 'Silt', 'SSA', 'TN', 'BC-Dosage', 'OM', 'N', 'T-Cd', 'TP', 'Time', 'BC-EC', 'BC-pH', 'Soil-pH']

def format_coef(val, is_first=False):
    if abs(val) < 1e-7:
        return ""
    if is_first:
        sign = "" if val >= 0 else "-"
    else:
        sign = " + " if val >= 0 else " - "
    abs_val = abs(val)
    if abs_val < 0.001:
        exponent = int(np.floor(np.log10(abs_val)))
        base = abs_val / (10 ** exponent)
        return f"{sign}{base:.2f}×10^{{{exponent}}}"
    else:
        return f"{sign}{abs_val:.3f}"

def get_best_position(x_data, y_data):
    x_mid = (x_data.min() + x_data.max()) / 2
    y_mid = (y_data.min() + y_data.max()) / 2
    q1 = np.sum((x_data >= x_mid) & (y_data >= y_mid))
    q2 = np.sum((x_data < x_mid) & (y_data >= y_mid))
    q3 = np.sum((x_data < x_mid) & (y_data < y_mid))
    q4 = np.sum((x_data >= x_mid) & (y_data < y_mid))
    counts = {'q2': q2, 'q1': q1, 'q3': q3, 'q4': q4}
    min_quad = min(counts, key=counts.get)
    if min_quad == 'q2':
        return 0.06, 0.94, 'top', 'left'
    elif min_quad == 'q1':
        return 0.94, 0.94, 'top', 'right'
    elif min_quad == 'q3':
        return 0.06, 0.12, 'bottom', 'left'
    else:
        return 0.94, 0.12, 'bottom', 'right'

def main():
    if not os.path.exists(DATA_CONFIG['train_path']):
        raise FileNotFoundError(f"Missing file: {DATA_CONFIG['train_path']}")

    df = pd.read_csv(DATA_CONFIG['train_path'])
    available_features = [f for f in FEATURES if f in df.columns]

    X_raw = df[available_features]
    y = df[DATA_CONFIG['target']]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=available_features)

    best_params_svr = {
        'kernel': 'rbf',
        'C': 51.27651497251763,
        'gamma': 0.08641204502744947,
        'epsilon': 0.6941158123710329
    }

    model = SVR(**best_params_svr)
    model.fit(X_scaled, y)

    background = shap.kmeans(X_scaled, 10)
    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(X_scaled)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'

    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman:bold'
    plt.rcParams['mathtext.it'] = 'Times New Roman:bold:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    target_features = ['BC-Dosage', 'TN', 'BC-EC', 'N', 'OM']
    plot_features = [f for f in target_features if f in X_raw.columns]

    colors = ["#00a0e9", "#7ecef4", "#f7b5d2", "#e60067"]
    custom_cmap = LinearSegmentedColormap.from_list("shap_cmap", colors)

    for feat in plot_features:
        fig, ax = plt.subplots(figsize=(4.5, 4.2), dpi=600)

        x_data = X_raw[feat].values
        feat_idx = list(X_raw.columns).index(feat)
        y_data = shap_values[:, feat_idx]

        x_min, x_max = x_data.min(), x_data.max()
        if x_max != x_min:
            norm_color = (x_data - x_min) / (x_max - x_min)
        else:
            norm_color = np.zeros_like(x_data)

        colors_mapped = custom_cmap(norm_color)

        fit_order = 1
        if feat in ['BC-Dosage', 'TN']:
            fit_order = 3

        sns.regplot(
            x=x_data, y=y_data, ax=ax, order=fit_order,
            scatter=False,
            line_kws={"color": "#c6346a", "linewidth": 2.2, "zorder": 4},
            color="#f3beca"
        )

        pe = [
            path_effects.withSimplePatchShadow(offset=(1.0, -1.0), shadow_rgbFace='#000000', alpha=0.35),
            path_effects.withStroke(linewidth=1.8, foreground='#222222', alpha=0.15)
        ]

        ax.scatter(
            x_data, y_data,
            c=colors_mapped,
            s=55,
            edgecolors='#ffffff',
            linewidths=0.6,
            alpha=0.98,
            zorder=3,
            path_effects=pe
        )

        try:
            p = np.polyfit(x_data, y_data, fit_order)
            y_pred = np.polyval(p, x_data)
            y_bar = np.mean(y_data)
            ss_tot = np.sum((y_data - y_bar)**2)
            ss_res = np.sum((y_data - y_pred)**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            if fit_order == 1:
                item1 = f"{format_coef(p[0], is_first=True)}x" if abs(p[0]) >= 1e-7 else ""
                item2 = format_coef(p[1])
                formula_str = f"y = {item1}{item2}"
            elif fit_order == 2:
                item1 = f"{format_coef(p[0], is_first=True)}x^2" if abs(p[0]) >= 1e-7 else ""
                item2 = f"{format_coef(p[1])}x" if abs(p[1]) >= 1e-7 else ""
                item3 = format_coef(p[2])
                formula_str = f"y = {item1}{item2}{item3}"
            else:
                item1 = f"{format_coef(p[0], is_first=True)}x^3" if abs(p[0]) >= 1e-7 else ""
                item2 = f"{format_coef(p[1])}x^2" if abs(p[1]) >= 1e-7 else ""
                item3 = f"{format_coef(p[2])}x" if abs(p[2]) >= 1e-7 else ""
                item4 = format_coef(p[3])
                formula_str = f"y = {item1}{item2}{item3}{item4}"

            tx, ty, va_dir, ha_dir = get_best_position(x_data, y_data)
            final_text = f"$\\mathbf{{{formula_str}}}$\n$\\mathbf{{R^2 = {r2:.3f}}}$"

            ax.text(tx, ty, final_text,
                    transform=ax.transAxes, fontsize=11,
                    fontname='Times New Roman', fontweight='bold', va=va_dir, ha=ha_dir, zorder=5)
        except:
            pass

        ax.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.8, zorder=1)
        ax.set_xlabel(feat, fontsize=12, fontname='Times New Roman', fontweight='bold')
        ax.set_ylabel('SHAP value', fontsize=12, fontname='Times New Roman', fontweight='bold')

        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', which='major', labelsize=10.5, colors='black', direction='in', width=1.2)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontweight('bold')

        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("right", size="4%", pad=0.15)

        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap), cax=cbar_ax, ticks=[0, 1])
        cb.ax.set_yticklabels(['Low', 'High'], fontname='Times New Roman', fontweight='bold', fontsize=10.5)
        cb.outline.set_linewidth(1.2)
        cb.ax.tick_params(size=0)

        output_path = f"C:\\Users\\86136\\Desktop\\ML-ZK\\Cd-SHAP-Dependence-{feat}.tiff"
        plt.savefig(output_path, format='tiff', dpi=600, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    main()