import pandas as pd
from sklearn.model_selection import train_test_split
import os

input_path = r'C:\Users\86136\Desktop\ML-ZK\raw data.csv'
output_dir = r'C:\Users\86136\Desktop\ML-ZK'

for enc in ['utf-8-sig', 'gb18030', 'utf-8', 'gbk']:
    try:
        df = pd.read_csv(input_path, encoding=enc)
        break
    except Exception:
        continue

train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, shuffle=True)
train_df.to_csv(os.path.join(output_dir, 'train data.csv'), index=False, encoding='utf-8-sig')
test_df.to_csv(os.path.join(output_dir, 'test data.csv'), index=False, encoding='utf-8-sig')