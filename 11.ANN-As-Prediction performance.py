import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib import rcParams
from sklearn.utils import resample

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

best_params_ann = {
    'activation': 'relu',
    'alpha': 3.01323046912864e-05,
    'learning_rate_init': 0.04947291290920201,
    'hidden_layer_sizes': (128, 32, 32),
    'solver': 'adam',
    'max_iter': 1000,
    'random_state': 42,
    'early_stopping': True,
    'validation_fraction': 0.1
}

DATA_CONFIG = {
    'train_path': r'C:\Users\86136\Desktop\ML-ZK\train.csv',
    'test_path': r'C:\Users\86136\Desktop\ML-ZK\test.csv',
    'target': "ΔAs",
    'exclude': "ΔCd"
}

def load_data():
    train_df = pd.read_csv(DATA_CONFIG['train_path'], encoding='utf-8-sig')
    test_df = pd.read_csv(DATA_CONFIG['test_path'], encoding='utf-8-sig')
    features = [c for c in train_df.columns if c not in [DATA_CONFIG['target'], DATA_CONFIG['exclude']]]
    return (train_df[features], train_df[DATA_CONFIG['target']],
            test_df[features], test_df[DATA_CONFIG['target']])

X_train, y_train, X_test, y_test = load_data()

model = MLPRegressor(**best_params_ann)
model.fit(X_train, y_train)
train_preds, test_preds = model.predict(X_train), model.predict(X_test)

def bootstrap_std_intervals(X_train, y_train, X_test, base_params, n_bootstrap=100, z_score=1.645):
    all_train_preds = []
    all_test_preds = []
    for i in range(n_bootstrap):
        X_resampled, y_resampled = resample(X_train, y_train, replace=True, random_state=i+100)
        params = base_params.copy()
        params['random_state'] = i+200
        model_boot = MLPRegressor(**params)
        model_boot.fit(X_resampled, y_resampled)
        all_train_preds.append(model_boot.predict(X_train))
        all_test_preds.append(model_boot.predict(X_test))
    train_preds_stack = np.array(all_train_preds)
    test_preds_stack = np.array(all_test_preds)
    train_mean = np.mean(train_preds_stack, axis=0)
    train_std = np.std(train_preds_stack, axis=0, ddof=1)
    test_mean = np.mean(test_preds_stack, axis=0)
    test_std = np.std(test_preds_stack, axis=0, ddof=1)
    train_lower = train_mean - z_score * train_std
    train_upper = train_mean + z_score * train_std
    test_lower = test_mean - z_score * test_std
    test_upper = test_mean + z_score * test_std
    return train_lower, train_upper, test_lower, test_upper

train_lower, train_upper, test_lower, test_upper = bootstrap_std_intervals(X_train, y_train, X_test, best_params_ann, n_bootstrap=100)

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

grid.ax_joint.set_xlabel('Measured ΔAs', fontsize=14, fontweight='bold', fontname='Times New Roman')
grid.ax_joint.set_ylabel('Predicted ΔAs', fontsize=14, fontweight='bold', fontname='Times New Roman')

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

output_path = r"C:\Users\86136\Desktop\ML-ZK\ANN-As-Prediction performance.tiff"
plt.savefig(output_path, dpi=600, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')
plt.show()