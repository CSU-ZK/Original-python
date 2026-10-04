import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
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
    'target': "ΔCd",
    'exclude': "ΔAs"
}

best_params_lgbm = {
    'n_estimators': 192,
    'max_depth': 6,
    'num_leaves': 7,
    'learning_rate': 0.075,
    'subsample': 0.65,
    'colsample_bytree': 0.75,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

def load_data():
    train_df = pd.read_csv(DATA_CONFIG['train_path'], encoding='utf-8-sig')
    test_df = pd.read_csv(DATA_CONFIG['test_path'], encoding='utf-8-sig')
    features = [c for c in train_df.columns if c not in [DATA_CONFIG['target'], DATA_CONFIG['exclude']]]
    return (train_df[features], train_df[DATA_CONFIG['target']],
            test_df[features], test_df[DATA_CONFIG['target']])

X_train, y_train, X_test, y_test = load_data()

model = LGBMRegressor(**best_params_lgbm)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

def train_quantile_model(alpha):
    q_params = best_params_lgbm.copy()
    q_params.update({
        'objective': 'quantile',
        'alpha': alpha
    })
    model_quant = LGBMRegressor(**q_params)
    model_quant.fit(X_train, y_train)
    return model_quant

lower_model = train_quantile_model(0.05)
upper_model = train_quantile_model(0.95)

lower_bound = lower_model.predict(X_test)
upper_bound = upper_model.predict(X_test)

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
ax.set_ylabel('Predicted ΔCd', fontsize=13, fontweight='bold', labelpad=10)

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

save_path = r'C:\Users\86136\Desktop\ML-ZK\LGBM-Cd-Prediction interval.tiff'
plt.savefig(save_path, dpi=600, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')
plt.show()