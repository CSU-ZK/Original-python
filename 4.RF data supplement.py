import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

input_path = r'C:\Users\86136\Desktop\ML-ZK\raw-all.csv'
output_csv = r'C:\Users\86136\Desktop\ML-ZK\raw-all-RF.csv'

target_cols = [
    'BC-pH', 'Ash', 'BC-CEC', 'PD', 'TPV', 'SSA', 'C', 'H', 'O', 'N', 'BC-EC',
    'Soil-pH', 'Soil-CEC', 'Soil-EC', 'OM', 'Sand', 'Silt', 'Clay', 'TN', 'TP',
    'T-As', 'T-Cd', 'BC-Dosage', 'Time(d)'
]
preserved_cols = ['ΔAs', 'ΔCd']

all_needed_cols = target_cols + preserved_cols

df_raw = pd.read_csv(input_path, encoding='gb18030')
df_raw = df_raw[[c for c in all_needed_cols if c in df_raw.columns]].copy()

df_preserved = df_raw[[c for c in preserved_cols if c in df_raw.columns]].copy()
df = df_raw.drop(columns=[c for c in preserved_cols if c in df_raw.columns]).copy()

def text_matched_imputation(df_input, estimator):
    df_imputed = df_input.copy()

    for col in df_input.columns:
        if df_input[col].isnull().all():
            continue

        if df_input[col].isnull().any():
            is_missing = df_input[col].isnull()

            y_train = df_input.loc[~is_missing, col]
            X = df_input.drop(columns=[col])

            X_train = X.loc[~is_missing]
            X_test = X.loc[is_missing]

            if X_train.shape[0] == 0:
                continue

            valid_features = X_train.columns[~X_train.isnull().all()]
            X_train = X_train[valid_features]
            X_test = X_test[valid_features] if not X_test.empty else X_test

            sim_imp = SimpleImputer(strategy='mean')
            X_train_imp = sim_imp.fit_transform(X_train)
            X_test_imp = sim_imp.transform(X_test) if not X_test.empty else X_test

            if not X_test.empty:
                model = estimator
                model.fit(X_train_imp, y_train)
                y_pred = model.predict(X_test_imp)
                df_imputed.loc[is_missing, col] = np.round(y_pred, 3)

    return df_imputed

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
df_rf = text_matched_imputation(df, rf_regressor)

for col in df_preserved.columns:
    df_rf[col] = df_raw[col].values

df_final = df_raw.copy()
for col in df_rf.columns:
    df_final[col] = df_raw[col].fillna(df_rf[col])

df_final.to_csv(output_csv, index=False, encoding='utf-8-sig')