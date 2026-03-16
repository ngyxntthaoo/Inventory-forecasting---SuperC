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
SEASONAL_LAG = 7 
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

def snaive_fn(train, h): 
     fc = [train[len(train) - SEASONAL_LAG + (i % SEASONAL_LAG)] 
           if len(train) - SEASONAL_LAG + (i % SEASONAL_LAG) >= 0 else train[-1] 
           for i in range(h)] 
     return np.clip(fc, 0, None) 


print(f"Lookback={LOOKBACK}d") 
print("\n=== Evaluating on TEST ===") 
results = [] 

for h in HORIZONS: 
     lb = LOOKBACK 
     scores = {k: [] for k in ["smape", "mase", "rmse", "rmsle"]} 
     for sid in series_ids: 
          r = rolling_eval(series_dict[sid], VAL_END, h, LOOKBACK, snaive_fn) 
          for k in scores: scores[k].append(r[k]) 
     row = {"model": "SNaive", "dataset": "retail_inventory_daily", "target": TARGET, 
            "horizon": h, "lookback": lb, 
            "mean_smape": round(float(np.nanmean(scores["smape"])), 4), 
            "median_smape": round(float(np.nanmedian(scores["smape"])), 4), 
            "mean_mase": round(float(np.nanmean(scores["mase"])), 4), 
            "median_mase": round(float(np.nanmedian(scores["mase"])), 4), 
            "mean_rmse": round(float(np.nanmean(scores["rmse"])), 4),
              "median_rmse": round(float(np.nanmedian(scores["rmse"])), 4), 
              "mean_rmsle": round(float(np.nanmean(scores["rmsle"])), 4), 
              "median_rmsle": round(float(np.nanmedian(scores["rmsle"])), 4)} 
     results.append(row) 
     print(f" H={h:2d} lb={lb}d | sMAPE={row['mean_smape']:.2f}% MASE={row['mean_mase']:.4f} RMSE={row['mean_rmse']:.2f} RMSLE={row['mean_rmsle']:.4f}") 
     

pd.DataFrame(results).to_csv(f"{RESULT_DIR}/snaive_daily_summary.csv", index=False) 
print(f"\nSaved to {RESULT_DIR}/snaive_daily_summary.csv")