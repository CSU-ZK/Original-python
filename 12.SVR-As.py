import optuna
import pandas as pd
import numpy as np
from sklearn.svm import SVR
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
        'kernel': 'rbf',
         'C': trial.suggest_float('C', 1e-1, 1e4, log=True),
        'gamma': trial.suggest_float('gamma', 1e-4, 1e0, log=True),
        'epsilon': trial.suggest_float('epsilon', 1e-4, 1e0, log=True)
    }

    model = SVR(**actual_params)

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
        print(f"  {key:<20} : {value}")

    model = SVR(**final_params, kernel='rbf')
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