import numpy as np 
import pandas as pd 
from metrics import smape, mase, rmse, rmsle 

DATA_PATH = "Model/dataset/sales_data.csv" 
RESULT_DIR = "Model/result" 
TARGET = "Units Sold" 
TRAIN_END = "2023-06-30" 
VAL_END = "2023-10-31" 
HORIZONS = [7, 14, 28] 
LOOKBACK = 30
LAG = 7 

df = pd.read_csv(DATA_PATH) 
df["Date"] = pd.to_datetime(df["Date"]) 
df = df.sort_values(["Store ID", "Product ID", "Date"]) 
series_dict = {} 

for (store, product), grp in df.groupby(["Store ID", "Product ID"]): 
     series_dict[f"{store}_{product}"] = grp.set_index("Date")[TARGET] 
     series_ids = sorted(series_dict.keys()) 
     n_series = len(series_ids) 
     print(f"Total series: {n_series}, target: {TARGET}")


def rolling_eval(series, split_end, horizon, lookback, model_fn): 
     eval_start = pd.Timestamp(split_end) + pd.Timedelta(days=1) 
     eval_end = series.index.max() 
     all_fc, all_ac, last_train = [], [], None 
     t = eval_start 
     while t + pd.Timedelta(days=horizon - 1) <= eval_end: 
          train = series[t - pd.Timedelta(days=lookback): t - pd.Timedelta(days=1)].values.astype(float) 
          actual = series[t: t + pd.Timedelta(days=horizon - 1)].values.astype(float) 
          if len(train) < 2 or len(actual) < horizon: 
               t += pd.Timedelta(days=horizon); continue 
          all_fc.append(model_fn(train, horizon)) 
          all_ac.append(actual) 
          last_train = train 
          t += pd.Timedelta(days=horizon) 
          if not all_fc: 
               return {k: np.nan for k in ["smape", "mase", "rmse", "rmsle"]} 
          fc_arr, ac_arr = np.array(all_fc), np.array(all_ac) 
          lag = min(LAG, len(last_train) - 1) 
          denom = np.mean(np.abs(last_train[lag:] - last_train[:-lag])) or 1.0 
          return { 
               "smape": (2 * np.abs(fc_arr - ac_arr) / (np.abs(fc_arr) + np.abs(ac_arr) + 1e-8)).mean() * 100, 
               "mase" : np.mean(np.abs(fc_arr - ac_arr)) / denom, 
               "rmse" : np.sqrt(np.mean((fc_arr - ac_arr) ** 2)), 
               "rmsle": np.sqrt(np.mean((np.log1p(np.clip(fc_arr, 0, None)) - np.log1p(np.clip(ac_arr, 0, None))) ** 2)), 
               }