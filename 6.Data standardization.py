import pandas as pd
from sklearn.preprocessing import StandardScaler

path_train_in = r'C:\Users\86136\Desktop\ML-ZK\train data.csv'
path_test_in = r'C:\Users\86136\Desktop\ML-ZK\test data.csv'

path_train_out = r'C:\Users\86136\Desktop\ML-ZK\train.csv'
path_test_out = r'C:\Users\86136\Desktop\ML-ZK\test.csv'

df_train = pd.read_csv(path_train_in, encoding='utf-8-sig')
df_test = pd.read_csv(path_test_in, encoding='utf-8-sig')

preserved_cols = ['ΔAs', 'ΔCd']
feature_cols = [col for col in df_train.columns if col not in preserved_cols]

scaler = StandardScaler()

df_train_scaled = df_train.copy()
df_train_scaled[feature_cols] = scaler.fit_transform(df_train[feature_cols])

df_test_scaled = df_test.copy()
df_test_scaled[feature_cols] = scaler.transform(df_test[feature_cols])

df_train_scaled.to_csv(path_train_out, index=False, encoding='utf-8-sig')
df_test_scaled.to_csv(path_test_out, index=False, encoding='utf-8-sig')