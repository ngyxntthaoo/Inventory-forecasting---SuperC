import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from metrics import smape, mase, rmse, rmsle

warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)

DATA_PATH  = r"c:\Doris\Git Repo\Inventory-forecasting---SuperC\Model\dataset\sales_data.csv"
RESULT_DIR = r"c:\Doris\Git Repo\Inventory-forecasting---SuperC\Model\result"
TARGET     = "Units Sold"
TRAIN_END  = "2023-06-30"
VAL_END    = "2023-10-31"
HORIZONS   = [7, 14, 28]
LOOKBACK   = 60  # Increased for LSTM training
LAG        = 7

# ── Load daily series ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Store ID", "Product ID", "Date"])

series_dict = {}
for (store, product), grp in df.groupby(["Store ID", "Product ID"]):
    series_dict[f"{store}_{product}"] = grp.set_index("Date")[TARGET]

series_ids = sorted(series_dict.keys())
n_series   = len(series_ids)
print(f"Total series: {n_series}, target: {TARGET}")

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=28, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take last timestep
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

def build_lstm_model(output_size):
    """Build PyTorch LSTM model"""
    return LSTMModel(output_size=output_size)

def rolling_eval(series, split_end, horizon, lookback, horizon_idx):
    """
    Rolling evaluation with PyTorch LSTM deep learning model
    """
    eval_start = pd.Timestamp(split_end) + pd.Timedelta(days=1)
    eval_end   = series.index.max()
    all_fc, all_ac, last_train = [], [], None
    t = eval_start

    # Build model once per series
    model = build_lstm_model(output_size=horizon)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    while t + pd.Timedelta(days=horizon - 1) <= eval_end:
        # Use all available historical data up to current point for training
        train_idx = (series.index < t)
        train_vals = series[train_idx].values.astype(float)
        
        actual_idx = (series.index >= t) & \
                     (series.index < t + pd.Timedelta(days=horizon))
        actual_vals = series[actual_idx].values.astype(float)
        
        if len(train_vals) < lookback + horizon + 1 or len(actual_vals) < horizon:
            # Fallback to moving average if insufficient data
            ma_window = min(7, len(train_vals)) if len(train_vals) > 0 else 1
            pred_val = np.mean(train_vals[-ma_window:]) if len(train_vals) > 0 else 0
            all_fc.append([pred_val] * horizon)
            all_ac.append(actual_vals[:horizon])
            last_train = train_vals
            t += pd.Timedelta(days=horizon)
            continue
        
        # Normalize training data
        train_mean, train_std = train_vals.mean(), train_vals.std() + 1e-8
        train_norm = (train_vals - train_mean) / train_std
        
        # Create training sequences for multi-step forecasting
        X_train, y_train = [], []
        for i in range(len(train_norm) - lookback - horizon + 1):
            # Input: lookback days
            X_train.append(train_norm[i:i+lookback])
            # Target: next horizon days
            y_train.append(train_norm[i+lookback:i+lookback+horizon])
        
        if len(X_train) < 2:
            # Fallback to moving average if insufficient training data
            ma_window = min(7, len(train_vals))
            pred_val = np.mean(train_vals[-ma_window:])
            all_fc.append([pred_val] * horizon)
            all_ac.append(actual_vals[:horizon])
            last_train = train_vals
            t += pd.Timedelta(days=horizon)
            continue
        
        X_train = torch.FloatTensor(np.array(X_train)).unsqueeze(-1)  # (batch, lookback, 1)
        y_train = torch.FloatTensor(np.array(y_train))  # (batch, horizon)
        
        # Train model for this rolling window
        model.train()
        for epoch in range(50):  # More training epochs
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if loss.item() < 0.01:  # Early stopping
                break
        
        # Predict
        model.eval()
        with torch.no_grad():
            # Use last lookback window for prediction
            X_test = torch.FloatTensor(train_norm[-lookback:]).unsqueeze(0).unsqueeze(-1)
            pred_norm = model(X_test).squeeze(0).numpy()
            
            # Denormalize
            pred = pred_norm * train_std + train_mean
            pred = np.clip(pred, 0, None)
            
            all_fc.append(pred)
            all_ac.append(actual_vals[:horizon])
            last_train = train_vals
        
        t += pd.Timedelta(days=horizon)
    
    if not all_fc:
        return {k: np.nan for k in ["smape", "mase", "rmse", "rmsle"]}
    
    fc_arr, ac_arr = np.array(all_fc), np.array(all_ac)
    
    # Calculate MASE denominator safely
    if last_train is not None and len(last_train) > LAG:
        lag = LAG
        denom = np.mean(np.abs(last_train[lag:] - last_train[:-lag])) or 1.0
    else:
        denom = 1.0  # Fallback denominator
    
    return {
        "smape": (2 * np.abs(fc_arr - ac_arr) / (np.abs(fc_arr) + np.abs(ac_arr) + 1e-8)).mean() * 100,
        "mase" : np.mean(np.abs(fc_arr - ac_arr)) / denom,
        "rmse" : np.sqrt(np.mean((fc_arr - ac_arr) ** 2)),
        "rmsle": np.sqrt(np.mean((np.log1p(np.clip(fc_arr, 0, None)) - np.log1p(np.clip(ac_arr, 0, None))) ** 2)),
    }

print("\n=== Evaluating on TEST ===")
results, details = [], []
for h_idx, h in enumerate(HORIZONS):
    lb = LOOKBACK
    scores = {k: [] for k in ["smape", "mase", "rmse", "rmsle"]}
    for i, sid in enumerate(series_ids):  # All series
        store, product = sid.split("_", 1)
        r = rolling_eval(series_dict[sid], VAL_END, h, LOOKBACK, h_idx)
        for k in scores: scores[k].append(r[k])
        details.append({"model": "LSTM-Univariate", "store": store, "product": product,
                        "horizon": h, **{k: round(float(r[k]), 4) for k in r}})
        if (i + 1) % 25 == 0: print(f"    {i+1}/{n_series} done")
    row = {"model": "LSTM-Univariate", "dataset": "retail_inventory_daily", "target": TARGET,
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

pd.DataFrame(results).to_csv(f"{RESULT_DIR}/lstm_uni_daily_summary.csv", index=False)
pd.DataFrame(details).to_csv(f"{RESULT_DIR}/lstm_uni_daily_details.csv", index=False)
print(f"\nSaved summary to {RESULT_DIR}/lstm_uni_daily_summary.csv")
print(f"Saved details to {RESULT_DIR}/lstm_uni_daily_details.csv")
