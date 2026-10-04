import pandas as pd

def fmt_num(x):
    s = f"{x:.3f}"
    s = s.rstrip('0')
    s = s.rstrip('.')
    return s

df = pd.read_csv(r"C:\Users\86136\Desktop\ML-ZK\raw data.csv", encoding="gbk")

results = []

for column in df.select_dtypes(include=['number']).columns:
    count = df[column].count()
    min_val = df[column].min()
    max_val = df[column].max()
    mean_val = df[column].mean()
    sd_val = df[column].std()

    results.append({
        "Variable": column,
        "Count": count,
        "Data range": f"{fmt_num(min_val)}-{fmt_num(max_val)}",
        "Mean": fmt_num(mean_val),
        "SD": fmt_num(sd_val)
    })

summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))