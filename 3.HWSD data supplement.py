import pandas as pd
import numpy as np
import rasterio
from simpledbf import Dbf5

IN_CSV = r'C:\Users\86136\Desktop\ML-ZK\raw-soil.csv'
IN_TIF = r'C:\Users\86136\Desktop\【立方数据学社】2023年全球范围的土壤数据集HWSD2.0\HWSD2_DATA_D1_0-20cm.tif'
IN_DBF = r'C:\Users\86136\Desktop\【立方数据学社】2023年全球范围的土壤数据集HWSD2.0\HWSD2_DATA_D1_0-20cm.tif.vat.dbf'
OUT_CSV = r'C:\Users\86136\Desktop\ML-ZK\raw-soil-HWSD.csv'

def main():
    attr_df = Dbf5(IN_DBF).to_dataframe()
    valid_ids = set(attr_df.loc[attr_df['1_PH_WATER'] > 0, 'Value'])

    df_orig = pd.read_csv(IN_CSV, dtype=str)
    df_proc = df_orig.copy()

    df_proc['Latitude'] = pd.to_numeric(df_proc['Latitude'], errors='coerce')
    df_proc['Longitude'] = pd.to_numeric(df_proc['Longitude'], errors='coerce')

    target_values = []
    with rasterio.open(IN_TIF) as src:
        nodata = src.nodata or -9999
        for _, row in df_proc.iterrows():
            px_r, px_c = src.index(row['Longitude'], row['Latitude'])
            try:
                val = src.read(1, window=rasterio.windows.Window(px_c, px_r, 1, 1))[0, 0]
            except:
                val = nodata

            if val == nodata or val not in valid_ids:
                for radius in [10, 30, 60, 100]:
                    win = rasterio.windows.Window(px_c - radius, px_r - radius, 2 * radius + 1, 2 * radius + 1)
                    block = src.read(1, window=win)
                    mask = np.isin(block, list(valid_ids))
                    if np.any(mask):
                        val = block[mask][0]
                        break
                else:
                    val = np.nan
            target_values.append(val)

    df_proc['Value'] = target_values
    res = df_proc.merge(attr_df, on='Value', how='left')

    fields = {
        'Soil-pH': '1_PH_WATER',
        'Sand': '1_SAND',
        'Silt': '1_SILT',
        'Clay': '1_CLAY',
        'Soil-CEC': '1_CEC_SOIL',
        'Soil-EC': '1_ELEC_CON',
        'TN': '1_TOTAL_N'
    }

    for target, hwsd in fields.items():
        if target not in res.columns:
            res[target] = np.nan

        hwsd_data = pd.to_numeric(res[hwsd], errors='coerce').replace([-9, -99, -999, -9999], np.nan)
        orig_data = pd.to_numeric(res[target], errors='coerce')

        res[target] = orig_data.fillna(hwsd_data)
        res[target] = res[target].fillna(res[target].mean()).round(3)

    if 'OM' not in res.columns:
        res['OM'] = np.nan

    if '1_ORG_CARB' in res.columns:
        oc = pd.to_numeric(res['1_ORG_CARB'], errors='coerce').replace([-9, -99, -999, -9999], np.nan)
        orig_om = pd.to_numeric(res['OM'], errors='coerce')
        res['OM'] = orig_om.fillna(oc * 1724).round(3)
        res['OM'] = res['OM'].fillna(res['OM'].mean()).round(3)

    property_cols = ['Soil-pH', 'Soil-CEC', 'Soil-EC', 'OM', 'Sand', 'Silt', 'Clay', 'TN']

    out_df = pd.DataFrame()

    for col in property_cols:
        if col in res.columns:
            out_df[col] = pd.to_numeric(res[col], errors='coerce')
            out_df.loc[out_df[col] < 0, col] = np.nan

    out_df['Latitude'] = df_orig['Latitude']
    out_df['Longitude'] = df_orig['Longitude']

    out_df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()