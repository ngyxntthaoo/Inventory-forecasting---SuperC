import warnings
import numpy as np
import pandas as pd
from pmdarima import auto_arima

warnings.filterwarnings("ignore")

DATA_PATH  = "../dataset/sales_data.csv"
RESULT_DIR = "../result"
TARGET     = "Units Sold"
TRAIN_END  = "2023-06-30"
VAL_END    = "2023-10-31"
HORIZONS   = [7, 14, 28]
LAG        = 7

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(
    ["Store ID", "Product ID", "Date"]
)


series_dict = {}
for (store, product), grp in df.groupby(
    ["Store ID", "Product ID"]
):
    s = grp.set_index("Date")[TARGET]
    # đảm bảo daily
    s = s.asfreq("D").fillna(0)

    series_dict[f"{store}_{product}"] = s


series_ids = sorted(series_dict.keys())

n_series = len(series_ids)

print(f"Total series: {n_series}, target: {TARGET}")


# =========================
# METRICS
# =========================

def smape(y, p):

    return np.mean(
        2 * np.abs(y - p) /
        (np.abs(y) + np.abs(p) + 1e-8)
    ) * 100


def mase(y, p, train):

    lag = min(LAG, len(train) - 1)

    denom = np.mean(
        np.abs(train[lag:] - train[:-lag])
    )

    if denom == 0:
        return np.nan

    return np.mean(np.abs(y - p)) / denom


# =========================
# DAILY ROLLING EVAL
# =========================

def rolling_eval_daily(series, split_end, horizon):

    split_end = pd.Timestamp(split_end)

    train_full = series[:split_end]

    test_series = series[
        split_end + pd.Timedelta(days=1):
    ]

    if len(train_full) < 20:
        return {"smape": np.nan, "mase": np.nan}

    # FIT ONCE
    try:

        model = auto_arima(
            train_full,
            seasonal=True,
            m=7,
            max_p=2,
            max_q=2,
            max_d=1,
            max_P=1,
            max_Q=1,
            max_D=1,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )

    except Exception:

        return {"smape": np.nan, "mase": np.nan}

    all_fc = []
    all_ac = []

    t = 0

    while t + horizon <= len(test_series):

        try:

            fc = model.predict(
                n_periods=horizon
            )

        except Exception:

            fc = [train_full.iloc[-1]] * horizon

        ac = test_series.iloc[
            t:t + horizon
        ].values

        all_fc.append(fc)
        all_ac.append(ac)

        t += horizon

    if not all_fc:

        return {"smape": np.nan, "mase": np.nan}

    fc_arr = np.array(all_fc)
    ac_arr = np.array(all_ac)

    return {

        "smape": smape(
            ac_arr,
            fc_arr
        ),

        "mase": mase(
            ac_arr,
            fc_arr,
            train_full.values
        ),

    }


print(
    "Skipping val tuning — full history context, seasonal ARIMA m=7"
)

print("\n=== Evaluating on TEST ===")


results = []
details = []


# =========================
# MAIN LOOP
# =========================

for h in HORIZONS:

    scores = {
        "smape": [],
        "mase": []
    }

    print(f"\n--- Horizon = {h} ---")

    for i, sid in enumerate(series_ids):

        store, product = sid.split("_", 1)

        r = rolling_eval_daily(
            series_dict[sid],
            VAL_END,
            h
        )

        for k in scores:
            scores[k].append(r[k])

        details.append({

            "model": "ARIMA_DAILY",

            "store": store,

            "product": product,

            "horizon": h,

            "smape": round(float(r["smape"]), 4),

            "mase": round(float(r["mase"]), 4),

        })

        if (i + 1) % 25 == 0:

            print(
                f"    {i+1}/{n_series} done"
            )

    row = {

        "model": "ARIMA_DAILY",

        "dataset":
        "retail_inventory_daily",

        "target": TARGET,

        "horizon": h,

        "mean_smape":
        round(
            float(
                np.nanmean(
                    scores["smape"]
                )
            ), 4
        ),

        "median_smape":
        round(
            float(
                np.nanmedian(
                    scores["smape"]
                )
            ), 4
        ),

        "mean_mase":
        round(
            float(
                np.nanmean(
                    scores["mase"]
                )
            ), 4
        ),

        "median_mase":
        round(
            float(
                np.nanmedian(
                    scores["mase"]
                )
            ), 4
        ),

    }

    results.append(row)

    print(
        f"  H={h:2d} | "
        f"sMAPE={row['mean_smape']:.2f}% "
        f"(med {row['median_smape']:.2f}) "
        f"MASE={row['mean_mase']:.4f} "
        f"(med {row['median_mase']:.4f})"
    )


# =========================
# SAVE
# =========================

pd.DataFrame(results).to_csv(
    f"{RESULT_DIR}/arima_daily_summary.csv",
    index=False
)

pd.DataFrame(details).to_csv(
    f"{RESULT_DIR}/arima_daily_details.csv",
    index=False
)


print(
    "\nSaved summary to",
    f"{RESULT_DIR}/arima_daily_summary.csv"
)

print(
    "Saved details to",
    f"{RESULT_DIR}/arima_daily_details.csv"
)