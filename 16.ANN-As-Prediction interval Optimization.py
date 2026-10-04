import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.utils import resample
from matplotlib import rcParams

config = {
    "font.family": 'serif',
    "font.serif": ['Times New Roman'],
    "font.weight": 'bold',
    "font.size": 12,
    "xtick.direction": 'in',
    "ytick.direction": 'in',
    "axes.unicode_minus": False,
    "savefig.dpi": 600
}
rcParams.update(config)

DATA_CONFIG = {
    'train_path': r'C:\Users\86136\Desktop\ML-ZK\train.csv',
    'test_path': r'C:\Users\86136\Desktop\ML-ZK\test.csv',
    'target': "ΔAs"
}

FEATURES = ['O', 'TPV', 'Clay', 'C', 'TN', 'BC-Dosage', 'OM', 'N', 'T-As', 'TP', 'Time', 'BC-EC', 'BC-pH', 'Soil-CEC']

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

def load_data():
    train_df = pd.read_csv(DATA_CONFIG['train_path'], encoding='utf-8-sig')
    test_df = pd.read_csv(DATA_CONFIG['test_path'], encoding='utf-8-sig')

    train_df = train_df[train_df[DATA_CONFIG['target']] > -500]
    test_df = test_df[test_df[DATA_CONFIG['target']] > -500]

    return (train_df[FEATURES], train_df[DATA_CONFIG['target']],
            test_df[FEATURES], test_df[DATA_CONFIG['target']])

X_train, y_train, X_test, y_test = load_data()

model = MLPRegressor(**best_params_ann)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

def bootstrap_std_intervals(X_train, y_train, X_test, base_params, n_bootstrap=100, z_score=1.645):
    all_test_preds = []
    for i in range(n_bootstrap):
        X_resampled, y_resampled = resample(X_train, y_train, replace=True, random_state=i+100)
        params = base_params.copy()
        params['random_state'] = i+200
        model_boot = MLPRegressor(**params)
        model_boot.fit(X_resampled, y_resampled)
        all_test_preds.append(model_boot.predict(X_test))
    test_preds_stack = np.array(all_test_preds)
    test_mean = np.mean(test_preds_stack, axis=0)
    test_std = np.std(test_preds_stack, axis=0, ddof=1)
    test_lower = test_mean - z_score * test_std
    test_upper = test_mean + z_score * test_std
    return test_lower, test_upper

lower_bound, upper_bound = bootstrap_std_intervals(X_train, y_train, X_test, best_params_ann, n_bootstrap=100)

sample_idx = np.arange(len(y_test))

fig, ax = plt.subplots(figsize=(10, 6.5))

color_pi = '#87CEFA'
color_obs = '#d73027'
color_pred = '#084594'

ax.fill_between(sample_idx, lower_bound, upper_bound,
                color=color_pi, alpha=0.4, zorder=1, label='90% Prediction Interval')

ax.plot(sample_idx, lower_bound, color=color_pi, lw=2.0, linestyle='--', alpha=0.8, zorder=1)
ax.plot(sample_idx, upper_bound, color=color_pi, lw=2.0, linestyle='--', alpha=0.8, zorder=1)

ax.plot(sample_idx, y_pred, color=color_pred, linewidth=2.0, alpha=0.6, zorder=2)
ax.scatter(sample_idx, y_pred, color=color_pred, s=55, marker='o',
           edgecolors='white', linewidths=0.5, zorder=4, label='Predicted Value')

ax.scatter(sample_idx, y_test, facecolors='none', edgecolors=color_obs,
           s=130, marker='*', linewidths=1.5, zorder=3, label='Observed Value')

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

ax.set_xlabel('Predicted sample', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('Predicted ΔAs', fontsize=13, fontweight='bold', labelpad=10)

y_all_min = min(lower_bound.min(), y_test.min())
y_all_max = max(upper_bound.max(), y_test.max())
margin = (y_all_max - y_all_min) * 0.15
ax.set_ylim(y_all_min - margin, y_all_max + margin)

ax.legend(loc='lower left', frameon=True, fontsize=10,
          facecolor='white', framealpha=0.9,
          prop={'family': 'Times New Roman', 'weight': 'bold'})

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontproperties({'family': 'Times New Roman', 'size': 11, 'weight': 'bold'})

ax.grid(False)
plt.tight_layout()

save_path = r'C:\Users\86136\Desktop\ML-ZK\ANN-As-Prediction interval Optimization.tiff'
plt.savefig(save_path, dpi=600, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')
plt.show()