import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib import rcParams

config = {
    "font.family": 'serif',
    "font.serif": ['Times New Roman'],
    "font.size": 12,
    "axes.unicode_minus": False,
    "xtick.direction": 'in',
    "ytick.direction": 'in',
    "savefig.dpi": 600
}
rcParams.update(config)

best_params_svr = {
    'kernel': 'rbf',
    'C': 51.27651497251763,
    'gamma': 0.08641204502744947,
    'epsilon': 0.6941158123710329
}

DATA_CONFIG = {
    'train_path': r'C:\Users\86136\Desktop\ML-ZK\train.csv',
    'test_path': r'C:\Users\86136\Desktop\ML-ZK\test.csv',
    'target': "ΔCd"
}

FEATURES = ['O', 'TPV', 'Ash', 'Silt', 'SSA', 'TN', 'BC-Dosage', 'OM', 'N', 'T-Cd', 'TP', 'Time', 'BC-EC', 'BC-pH', 'Soil-pH']

def load_data():
    train_df = pd.read_csv(DATA_CONFIG['train_path'], encoding='utf-8-sig')
    test_df = pd.read_csv(DATA_CONFIG['test_path'], encoding='utf-8-sig')
    return (train_df[FEATURES], train_df[DATA_CONFIG['target']],
            test_df[FEATURES], test_df[DATA_CONFIG['target']])

X_train, y_train, X_test, y_test = load_data()

model = SVR(**best_params_svr)
model.fit(X_train, y_train)
train_preds, test_preds = model.predict(X_train), model.predict(X_test)

def calc_prediction_intervals_scaled(y_train, y_train_pred, y_test_pred, scale_factor=1.2):
    residuals = y_train - y_train_pred
    sigma_residual = np.std(residuals, ddof=1) * scale_factor
    z_score = 1.645

    train_lower = y_train_pred - z_score * sigma_residual
    train_upper = y_train_pred + z_score * sigma_residual
    test_lower = y_test_pred - z_score * sigma_residual
    test_upper = y_test_pred + z_score * sigma_residual

    return train_lower, train_upper, test_lower, test_upper

def calc_picp(y_true, y_lower, y_upper):
    n = len(y_true)
    c = np.where((y_lower < y_true) & (y_true < y_upper), 1, 0)
    return np.sum(c) / n

train_lower, train_upper, test_lower, test_upper = calc_prediction_intervals_scaled(
    y_train, train_preds, test_preds, scale_factor=1.2
)

train_picp = calc_picp(y_train, train_lower, train_upper)
test_picp = calc_picp(y_test, test_lower, test_upper)

sns.set_style("ticks")
grid = sns.JointGrid(x=np.concatenate([y_train, y_test]),
                     y=np.concatenate([train_preds, test_preds]),
                     space=0, ratio=5)

grid.ax_joint.scatter(y_train, train_preds, color="#E84A4A", marker="o", s=45, alpha=0.8, edgecolors='k', linewidths=0.5)
grid.ax_joint.scatter(y_test, test_preds, color="#364AF1", marker="^", s=55, alpha=0.8, edgecolors='k', linewidths=0.5)

sns.kdeplot(x=y_train, ax=grid.ax_marg_x, color="#E84A4A", fill=True, alpha=0.2, linewidth=1.5)
sns.kdeplot(x=y_test, ax=grid.ax_marg_x, color="#364AF1", fill=True, alpha=0.2, linewidth=1.5)
sns.kdeplot(y=train_preds, ax=grid.ax_marg_y, color="#E84A4A", fill=True, alpha=0.2, linewidth=1.5)
sns.kdeplot(y=test_preds, ax=grid.ax_marg_y, color="#364AF1", fill=True, alpha=0.2, linewidth=1.5)

all_val = np.concatenate([y_train, y_test, train_preds, test_preds])
data_min, data_max = all_val.min(), all_val.max()
padding = (data_max - data_min) * 0.15
unified_lims = [data_min - padding, data_max + padding]

grid.ax_joint.set_xlim(unified_lims)
grid.ax_joint.set_ylim(unified_lims)

grid.ax_joint.plot(unified_lims, unified_lims, 'k--', alpha=0.7, lw=1.5, zorder=0)

for spine_name, spine in grid.ax_joint.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_edgecolor('black')

grid.ax_joint.xaxis.set_ticks_position('bottom')
grid.ax_joint.yaxis.set_ticks_position('left')

for ax in [grid.ax_marg_x, grid.ax_marg_y]:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

grid.ax_joint.set_xlabel('Measured ΔCd', fontsize=14, fontweight='bold', fontname='Times New Roman')
grid.ax_joint.set_ylabel('Predicted ΔCd', fontsize=14, fontweight='bold', fontname='Times New Roman')

for label in (grid.ax_joint.get_xticklabels() + grid.ax_joint.get_yticklabels()):
    label.set_fontname('Times New Roman')
    label.set_fontweight('bold')
    label.set_fontsize(11)

font_box = {'family': 'Times New Roman', 'weight': 'bold', 'size': 11}

t_info = (f"● Train Set\n"
          f"R$^2$: {r2_score(y_train, train_preds):.3f}\n"
          f"RMSE: {np.sqrt(mean_squared_error(y_train, train_preds)):.3f}\n"
          f"MAE: {mean_absolute_error(y_train, train_preds):.3f}\n"
          f"PICP: {train_picp:.3f}")

v_info = (f"▲ Test Set\n"
          f"R$^2$: {r2_score(y_test, test_preds):.3f}\n"
          f"RMSE: {np.sqrt(mean_squared_error(y_test, test_preds)):.3f}\n"
          f"MAE: {mean_absolute_error(y_test, test_preds):.3f}\n"
          f"PICP: {test_picp:.3f}")

grid.ax_joint.text(0.05, 0.95, t_info, transform=grid.ax_joint.transAxes,
                   va='top', ha='left', linespacing=1.5, color='#E84A4A', fontdict=font_box,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9,
                             edgecolor='#E84A4A', linestyle='--', linewidth=1))

grid.ax_joint.text(0.95, 0.05, v_info, transform=grid.ax_joint.transAxes,
                   va='bottom', ha='right', linespacing=1.5, color='#364AF1', fontdict=font_box,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9,
                             edgecolor='#364AF1', linestyle='--', linewidth=1))

output_path = r"C:\Users\86136\Desktop\ML-ZK\SVR-Cd-Prediction performance Optimization.tiff"
plt.savefig(output_path, dpi=600, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')
plt.show()