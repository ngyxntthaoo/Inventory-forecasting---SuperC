import warnings
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.preprocessing import LabelEncoder
from metrics import smape, mase, rmse, rmsle

print("Starting SARIMAX script...")
warnings.filterwarnings("ignore")

DATA_PATH  = r"c:\Doris\Git Repo\Inventory-forecasting---SuperC\Model\dataset\sales_data.csv"
RESULT_DIR = r"c:\Doris\Git Repo\Inventory-forecasting---SuperC\Model\result"
TARGET     = "Units Sold"
TRAIN_END  = "2023-06-30"
VAL_END    = "2023-10-31"
HORIZONS   = [7, 14, 28]
LOOKBACK   = 30
LAG        = 7

# ── Load daily series ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Store ID", "Product ID", "Date"])

# Create exogenous variable: Weather Condition (encoded)
print("Creating exogenous variables...")
weather_encoder = LabelEncoder()
weather_encoder.fit(df["Weather Condition"].unique())

# Create mapping from date to encoded weather condition
exog_dict = {}
for date in df["Date"].unique():
    weather_val = df[df["Date"] == date]["Weather Condition"].iloc[0]
    exog_dict[pd.Timestamp(date)] = weather_encoder.transform([weather_val])[0]
print(f"Weather conditions: {list(weather_encoder.classes_)}")

series_dict = {}
for (store, product), grp in df.groupby(["Store ID", "Product ID"]):
    series_dict[f"{store}_{product}"] = grp.set_index("Date")[TARGET]

series_ids = sorted(series_dict.keys())
n_series   = len(series_ids)
print(f"Total series: {n_series}, target: {TARGET}")

def rolling_eval(series, split_end, horizon, lookback, model_fn):
    """
    Rolling evaluation for SARIMAX using exogenous variables (Weather Condition)
    Only uses training data - no validation/test data leakage
    """
    eval_start = pd.Timestamp(split_end) + pd.Timedelta(days=1)
    eval_end   = series.index.max()
    all_fc, all_ac, last_train = [], [], None
    t = eval_start
    
    while t + pd.Timedelta(days=horizon - 1) <= eval_end:
        # Training data: lookback window only
        train_idx  = (series.index >= t - pd.Timedelta(days=lookback)) & \
                     (series.index < t)
        train_vals = series[train_idx].values.astype(float)
        train_dates = series[train_idx].index
        
        # Actual data: forecast horizon
        actual_idx = (series.index >= t) & \
                     (series.index < t + pd.Timedelta(days=horizon))
        actual_vals = series[actual_idx].values.astype(float)
        actual_dates = series[actual_idx].index
        
        if len(train_vals) < 2 or len(actual_vals) < horizon:
            t += pd.Timedelta(days=horizon)
            continue
        
        # Extract exogenous variables
        X_train = np.array([[exog_dict[d]] for d in train_dates]).astype(float)
        X_test  = np.array([[exog_dict[d]] for d in actual_dates]).astype(float)
        
        all_fc.append(model_fn(train_vals, X_train, X_test, horizon))
        all_ac.append(actual_vals)
        last_train = train_vals
        t += pd.Timedelta(days=horizon)
    
    if not all_fc:
        return {k: np.nan for k in ["smape", "mase", "rmse", "rmsle"]}
    
    fc_arr, ac_arr = np.array(all_fc), np.array(all_ac)
    lag   = min(LAG, len(last_train) - 1) if last_train is not None else 1
    denom = np.mean(np.abs(last_train[lag:] - last_train[:-lag])) or 1.0
    
    return {
        "smape": (2 * np.abs(fc_arr - ac_arr) / (np.abs(fc_arr) + np.abs(ac_arr) + 1e-8)).mean() * 100,
        "mase" : np.mean(np.abs(fc_arr - ac_arr)) / denom,
        "rmse" : np.sqrt(np.mean((fc_arr - ac_arr) ** 2)),
        "rmsle": np.sqrt(np.mean((np.log1p(np.clip(fc_arr, 0, None)) - np.log1p(np.clip(ac_arr, 0, None))) ** 2)),
    }

def sarimax_fn(y_train, X_train, X_test, h):
    """
    Seasonal ARIMA with eXogenous variables (SARIMAX)
    - y_train: target values from training data only
    - X_train: exogenous variables for training data only
    - X_test: exogenous variables for forecast period
    - h: forecast horizon
    """
    try:
        # SARIMAX with weekly seasonality (m=7)
        m = auto_arima(y_train, exogenous=X_train, seasonal=True, m=7,
                       max_p=1, max_q=1, max_d=1, max_P=1, max_Q=1, max_D=1,
                       information_criterion="aic", stepwise=True, trace=False,
                       error_action="ignore", suppress_warnings=True,
                       maxiter=20)  # Reduced maxiter
        result = m.predict(n_periods=h, exogenous=X_test)
        return np.clip(result, 0, None)
    except Exception as e:
        try:
            # Fallback to non-seasonal ARIMAX
            m = auto_arima(y_train, exogenous=X_train, seasonal=False,
                           max_p=1, max_q=1, max_d=1, max_order=2,
                           information_criterion="aic", stepwise=True, trace=False,
                           error_action="ignore", suppress_warnings=True,
                           maxiter=20)
            result = m.predict(n_periods=h, exogenous=X_test)
            return np.clip(result, 0, None)
        except Exception:
            # Final fallback: use last value
            return np.clip([y_train[-1]] * h, 0, None)

print(f"Lookback={LOOKBACK}d, Exogenous variables: Weather Condition")

print("\n=== Evaluating on TEST ===")
results, details = [], []
for h in HORIZONS:
    lb = LOOKBACK
    scores = {k: [] for k in ["smape", "mase", "rmse", "rmsle"]}
    
    for i, sid in enumerate(series_ids):
        store, product = sid.split("_", 1)
        r = rolling_eval(series_dict[sid], VAL_END, h, LOOKBACK, sarimax_fn)
        for k in scores: scores[k].append(r[k])
        details.append({"model": "SARIMAX", "store": store, "product": product,
                        "horizon": h, **{k: round(float(r[k]), 4) for k in r}})
        if (i + 1) % 25 == 0: print(f"    {i+1}/{n_series} done")
    
    row = {"model": "SARIMAX", "dataset": "retail_inventory_daily", "target": TARGET,
           "horizon": h, "lookback": lb,
           "mean_smape":   round(float(np.nanmean(scores["smape"])), 4),
           "median_smape": round(float(np.nanmedian(scores["smape"])), 4),
           "mean_mase":    round(float(np.nanmean(scores["mase"])), 4),
           "median_mase":  round(float(np.nanmedian(scores["mase"])), 4),
           "mean_rmse":    round(float(np.nanmean(scores["rmse"])), 4),
           "median_rmse":  round(float(np.nanmedian(scores["rmse"])), 4),
           "mean_rmsle":   round(float(np.nanmean(scores["rmsle"])), 4),
           "median_rmsle": round(float(np.nanmedian(scores["rmsle"])), 4)}
    results.append(row)
    print(f"  H={h:2d} lb={lb}d | sMAPE={row['mean_smape']:.2f}% MASE={row['mean_mase']:.4f} RMSE={row['mean_rmse']:.2f} RMSLE={row['mean_rmsle']:.4f}")

pd.DataFrame(results).to_csv(f"{RESULT_DIR}/sarimax_daily_summary.csv", index=False)
pd.DataFrame(details).to_csv(f"{RESULT_DIR}/sarimax_daily_details.csv", index=False)
print(f"\nSaved summary to {RESULT_DIR}/sarimax_daily_summary.csv")
print(f"Saved details to {RESULT_DIR}/sarimax_daily_details.csv")