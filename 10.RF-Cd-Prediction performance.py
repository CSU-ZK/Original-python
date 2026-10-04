import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

best_params_rf = {
    'n_estimators': 78,
    'max_depth': 11,
    'max_features': 5,
    'min_samples_leaf': 1,
    'random_state': 42,
    'n_jobs': -1
}

DATA_CONFIG = {
    'train_path': r'C:\Users\86136\Desktop\ML-ZK\train.csv',
    'test_path': r'C:\Users\86136\Desktop\ML-ZK\test.csv',
    'target': "ΔCd",
    'exclude': "ΔAs"
}

def load_data():
    train_df = pd.read_csv(DATA_CONFIG['train_path'], encoding='utf-8-sig')
    test_df = pd.read_csv(DATA_CONFIG['test_path'], encoding='utf-8-sig')
    features = [c for c in train_df.columns if c not in [DATA_CONFIG['target'], DATA_CONFIG['exclude']]]
    return (train_df[features], train_df[DATA_CONFIG['target']],
            test_df[features], test_df[DATA_CONFIG['target']])

X_train, y_train, X_test, y_test = load_data()

model = RandomForestRegressor(**best_params_rf)
model.fit(X_train, y_train)
train_preds, test_preds = model.predict(X_train), model.predict(X_test)

def forest_quantiles(X, model, alpha_lower=0.05, alpha_upper=0.95):
    all_preds = np.array([tree.predict(X) for tree in model.estimators_])
    lower = np.quantile(all_preds, alpha_lower, axis=0)
    upper = np.quantile(all_preds, alpha_upper, axis=0)
    return lower, upper

train_lower, train_upper = forest_quantiles(X_train, model)
test_lower, test_upper = forest_quantiles(X_test, model)

def calc_picp(y_true, y_lower, y_upper):
    n = len(y_true)
    c = np.where((y_lower < y_true) & (y_true < y_upper), 1, 0)
    return np.sum(c) / n

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

output_path = r"C:\Users\86136\Desktop\ML-ZK\RF-Cd-Prediction performance.tiff"
plt.savefig(output_path, dpi=600, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')
plt.show()