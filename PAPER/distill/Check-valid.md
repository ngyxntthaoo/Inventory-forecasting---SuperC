Dưới đây là đánh giá từng model theo các tiêu chí của Lec06:

## ✅ Những gì đúng chuẩn (tất cả models)
- Split theo thời gian: train → val → test đúng, không shuffle. eval_start = VAL_END + 1 day, eval_end = series.index.max() ✅
- Metrics đầy đủ: smape, mase, rmse, rmsle — cả mean lẫn median ✅
- Rolling evaluation: mỗi window dùng actual data làm input, không dùng predicted output ✅
- Scaler fit trên train only (LSTM2): scaler.fit_transform(train_raw) — fit trên window trước t, không leak ✅
- Save details + summary tách biệt ✅

## ⚠️ Vấn đề cần lưu ý
### 1. LOOKBACK = 30 ngày — quá ngắn cho tất cả models
Lec06 §6: features nên bao gồm lag, rolling, seasonal indicators. Với daily data có weekly seasonality, lookback 30 ngày chỉ bắt được ~4 tuần. Không đủ để ARIMA/Prophet học yearly pattern, và LSTM2 không thấy đủ chu kỳ.

Gợi ý: tăng lên 60–90 ngày, hoặc ít nhất 2 × max(HORIZONS) = 56.

### 2. Naive — dùng train[-1] từ sliding window, không phải last known value
```run_naive.py

train = series[t - lookback : t - 1]  # chỉ lấy 30 ngày gần nhất
naive_fn → train[-1]                   # = giá trị ngày t-1 ✅ đúng về logic
```

Logic đúng, nhưng vì dùng lookback=30 nên train[-1] luôn là series[t-1] — không có vấn đề. ✅

### 3. SNaive — seasonal lag hardcode = 7, nhưng index có thể không liên tục
```
train[len(train) - SEASONAL_LAG + (i % SEASONAL_LAG)]
```

Nếu series có missing dates, train[-7] không đảm bảo là "cùng ngày tuần trước". Nên lookup theo date index thay vì positional index.

### 4. ARIMA — dùng lookback=30 làm context, bỏ qua toàn bộ train history
```
train = series[t - 30d : t - 1d]  # chỉ 30 ngày
auto_arima(train, seasonal=False)  # không có seasonal component
```

Hai vấn đề:

- seasonal=False → không bắt được weekly seasonality (nên dùng seasonal=True, m=7)

- Chỉ 30 ngày context → ARIMA không đủ data để ước lượng tham số tốt, đặc biệt với max_p=2

### 5. Prophet — yearly_seasonality=False và lookback chỉ 30 ngày
```
Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
window = series[t - 30d : t - 1d]  # chỉ 30 ngày
```

Prophet được thiết kế để dùng toàn bộ history (Lec06 §5.1: "trend + seasonality + holidays"). Với chỉ 30 ngày, Prophet không khác gì linear regression ngắn hạn — mất đi toàn bộ lợi thế. Nên truyền series[:t-1] thay vì sliding window 30 ngày.

### 6. LSTM2 — recursive multi-step forecast, error accumulation
```
# mỗi bước dùng predicted output làm input bước tiếp theo
window.append(p)  # p là predicted, không phải actual
```

Lec06 §9: "Multi-step: dùng predicted output làm input → cẩn thận error accumulation; ưu tiên direct multi-horizon". Với H=28, lỗi tích lũy qua 28 bước có thể lớn. Đây là lý do LSTM2 có thể kém hơn SNaive ở H=28.

### 7. TRAIN_END khai báo nhưng không dùng
Tất cả models đều bắt đầu rolling từ VAL_END, không từ TRAIN_END. TRAIN_END chỉ là documentation, không có tác dụng trong code. Không phải lỗi, nhưng dễ gây nhầm lẫn.


Tóm tắt

| Model |	Split ✅	|Rolling eval ✅ |	Vấn đề chính |
|-------|----------|--------------|----------|
|Naive	|✅|	✅|	Lookback 30d (không ảnh hưởng logic)|
|SNaive|	✅	|✅	|Positional index có thể sai nếu missing |dates|
|ARIMA	|✅|	✅|	seasonal=False, lookback quá ngắn|
|Prophet|	✅|	✅|	Dùng sliding window 30d thay vì full history|
|LSTM2|	✅|	✅|	Recursive forecast → error accumulation ở H=14,28|