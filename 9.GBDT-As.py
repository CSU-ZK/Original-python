import optuna
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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

def optuna_objective(trial):
    actual_params = {
        'max_depth': trial.suggest_int('max_depth', 3, 6, step=1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200, step=1),
        'max_features': trial.suggest_int('max_features', 1, 3, step=1),
        'learning_rate': trial.suggest_int('learning_rate', 10, 200, step=1) / 1000,
        'subsample': trial.suggest_int('subsample', 500, 800, step=1) / 1000,
        'random_state': 42
    }

    model = GradientBoostingRegressor(**actual_params)

    cv = KFold(n_splits=9, shuffle=True, random_state=42)
    scores = cross_validate(
        model, X_train, y_train,
        scoring='neg_root_mean_squared_error',
        cv=cv
    )

    return np.mean(scores['test_score'])

def run_optimization(trials=500):
    sampler = optuna.samplers.TPESampler(n_startup_trials=30, seed=42)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(optuna_objective, n_trials=trials, show_progress_bar=True)
    return study.best_params

def evaluate_and_report(best_raw_params):
    final_params = best_raw_params.copy()
    if 'learning_rate' in final_params and final_params['learning_rate'] > 1:
        final_params['learning_rate'] /= 1000
    if 'subsample' in final_params and final_params['subsample'] > 1:
        final_params['subsample'] /= 1000

    for key, value in final_params.items():
        if key not in ['random_state']:
            print(f"  {key:<20} : {value}")

    model = GradientBoostingRegressor(**final_params, random_state=42)
    model.fit(X_train, y_train)

    for name, X, y in [("Train Set", X_train, y_train), ("Test Set", X_test, y_test)]:
        preds = model.predict(X)
        r2 = r2_score(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        mae = mean_absolute_error(y, preds)

        print(f" [{name}]")
        print(f"   R2   : {r2:.4f}")
        print(f"   RMSE : {rmse:.4f}")
        print(f"   MAE  : {mae:.4f}")

if __name__ == "__main__":
    best_found = run_optimization(500)
    evaluate_and_report(best_found)