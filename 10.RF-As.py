import optuna
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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
        'n_estimators': trial.suggest_int('n_estimators', 10, 300, step=1),
        'max_depth': trial.suggest_int('max_depth', 3, 15, step=1),
        'max_features': trial.suggest_int('max_features', 1, 6, step=1),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10, step=1),
        'random_state': 42,
        'n_jobs': -1
    }

    model = RandomForestRegressor(**actual_params)

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

    for key, value in final_params.items():
        if key not in ['random_state', 'n_jobs']:
            print(f"  {key:<20} : {value}")

    model = RandomForestRegressor(**final_params, random_state=42, n_jobs=-1)
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