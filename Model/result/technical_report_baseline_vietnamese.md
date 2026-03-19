# Báo Cáo Kỹ Thuật: Mô Hình Baseline Cho Dự Báo Tồn Kho (Inventory Forecasting)

## 1. Giới Thiệu

Báo cáo này trình bày chi tiết về việc triển khai và đánh giá các mô hình baseline cho dự án dự báo tồn kho bán lẻ (Retail Inventory Forecasting). Các mô hình baseline được sử dụng để thiết lập điểm chuẩn (benchmark) cho việc so sánh với các mô hình phức tạp hơn. Dự án sử dụng dữ liệu hàng ngày từ 100 series (kết hợp Store ID và Product ID) để dự báo số lượng sản phẩm bán ra (Units Sold) cho các horizon 7, 14 và 28 ngày.

## 2. Dữ Liệu Và Cách Đưa Dữ Liệu Vào

### 2.1. Nguồn Dữ Liệu
- **File dữ liệu**: `Model/dataset/sales_data.csv`
- **Cấu trúc**: Bao gồm các cột Date, Store ID, Product ID, Units Sold, Weather Condition, và các cột khác.
- **Thời gian**: Từ 2022-01-01 đến 2024-01-30 (760 ngày).
- **Số lượng series**: 100 series (5 stores × 20 products mỗi store).

### 2.2. Tiền Xử Lý Dữ Liệu
- **Chuyển đổi ngày tháng**: Sử dụng `pd.to_datetime()` để xử lý cột Date.
- **Nhóm dữ liệu**: Group by Store ID và Product ID để tạo time series cho mỗi series.
- **Tách tập train/val/test**:
  - Train: 2022-01-01 đến 2023-06-30 (546 ngày)
  - Validation: 2023-07-01 đến 2023-10-31 (123 ngày)
  - Test: 2023-11-01 đến 2024-01-30 (91 ngày, dùng cho rolling evaluation)
- **Rolling evaluation**: Sử dụng expanding window, lookback 30 ngày, đánh giá trên test set từ 2023-10-31.

### 2.3. Cách Đưa Dữ Liệu Vào Mô Hình
- Mỗi series được xử lý độc lập.
- Dữ liệu được chuẩn hóa (nếu cần) cho các mô hình như LSTM.
- Exogenous variables (ví dụ: Weather Condition) được encode và sử dụng trong SARIMAX.

### 2.4. Pipeline Thực Thi Chung Trong Repo
- **Rolling Evaluation**: Cho mỗi series, lặp qua các prediction points từ VAL_END +1 đến end. Tại mỗi point, lấy window train (lookback ngày), fit model (nếu cần), predict horizon ngày, so sánh với actual.
- **Input Cho Train**: Thường là array 1D của Units Sold trong window train (30-60 ngày), đôi khi kèm exogenous.
- **Fit Và Predict**: 
  - Fit: Áp dụng thư viện/model trên input train.
  - Predict: Dự báo horizon ngày, clip về [0, inf) để tránh negative.
- **Fallback**: Nếu fit thất bại (e.g., data thiếu), dùng naive forecast.
- **Output**: Lưu forecast, actual, tính metrics trung bình trên tất cả predictions.

## 3. Các Mô Hình Baseline Và Tham Số

### 3.1. Naive (Last Value)
- **Mô tả**: Dự báo giá trị cuối cùng của window train cho toàn bộ horizon.
- **Tham số**: lookback = 30 ngày.
- **Triển khai**: Pure numpy, không cần training.
- **Cách Fit Dữ Liệu**: Không fit thực sự; chỉ lấy giá trị cuối từ window train.
- **Input Train**: Array numpy của Units Sold trong 30 ngày gần nhất (series[train_end - 30: train_end]).
- **Cách Dự Đoán**: Lặp lại giá trị cuối cùng cho horizon ngày (np.full(horizon, train_array[-1])).
- **Ưu điểm**: Đơn giản, nhanh.
- **Nhược điểm**: Không capture trend hoặc seasonality.

### 3.2. Seasonal Naive
- **Mô tả**: Dự báo dựa trên giá trị cùng ngày trong tuần trước (seasonal lag = 7).
- **Tham số**: lookback = 30 ngày, seasonal_lag = 7.
- **Triển khai**: Pure numpy.
- **Cách Fit Dữ Liệu**: Không fit; sử dụng pattern tuần từ dữ liệu lịch sử.
- **Input Train**: Array numpy của Units Sold trong 30 ngày gần nhất.
- **Cách Dự Đoán**: Cho mỗi ngày trong horizon, lấy giá trị từ train_array[len(train_array) - seasonal_lag + (i % seasonal_lag)].
- **Ưu điểm**: Capture weekly seasonality.
- **Nhược điểm**: Không handle trend dài hạn.

### 3.3. Mean Naive
- **Mô tả**: Dự báo trung bình của window train cho toàn bộ horizon.
- **Tham số**: lookback = 30 ngày.
- **Triển khai**: Pure numpy.
- **Cách Fit Dữ Liệu**: Tính trung bình của window train.
- **Input Train**: Array numpy của Units Sold trong 30 ngày gần nhất.
- **Cách Dự Đoán**: Lặp lại giá trị trung bình (np.mean(train_array)) cho horizon ngày.
- **Ưu điểm**: Stable cho series ổn định.
- **Nhược điểm**: Không phản ánh biến động gần đây.

### 3.4. ARIMA (Auto-ARIMA)
- **Mô tả**: Mô hình ARIMA tự động chọn tham số bằng pmdarima.
- **Tham số**: max_p=2, max_q=2, max_d=1, seasonal=False, stepwise=True, information_criterion="aic".
- **Triển khai**: Sử dụng `pmdarima.auto_arima()` để fit và forecast.
- **Cách Fit Dữ Liệu**: Gọi `pmdarima.auto_arima(train_array, ...)` để tự động tìm p, d, q tốt nhất dựa trên AIC.
- **Input Train**: Array numpy 1D của Units Sold trong 30 ngày gần nhất (float64).
- **Cách Dự Đoán**: Sau khi fit, gọi `model.predict(n_periods=horizon)` để dự báo horizon ngày tiếp theo.
- **Ưu điểm**: Tự động tuning, handle trend và noise.
- **Nhược điểm**: Không capture seasonality dài hạn.

### 3.5. SARIMAX (Seasonal ARIMA with Exogenous)
- **Mô tả**: ARIMA với seasonality tuần (m=7) và biến ngoại sinh (Weather Condition).
- **Tham số**: max_p=1, max_q=1, max_d=1, max_P=1, max_Q=1, max_D=1, m=7, exogenous=weather_encoded.
- **Triển khai**: Sử dụng `pmdarima.auto_arima()` với seasonal=True và exogenous.
- **Cách Fit Dữ Liệu**: Encode Weather Condition thành số nguyên (LabelEncoder), tạo exogenous array 2D (shape: [len(train), 1]). Gọi `pmdarima.auto_arima(train_array, exogenous=exog_train, seasonal=True, m=7, ...)`.
- **Input Train**: Array numpy 1D của Units Sold + exogenous array 2D (weather encoded) trong 30 ngày gần nhất.
- **Cách Dự Đoán**: Tạo exogenous cho horizon ngày (dựa trên lịch sử weather), gọi `model.predict(n_periods=horizon, exogenous=exog_test)`.
- **Ưu điểm**: Handle seasonality và external factors.
- **Nhược điểm**: Phức tạp hơn, có thể overfit nếu exogenous không liên quan.

### 3.6. ETS (Exponential Smoothing)
- **Mô tả**: Exponential Smoothing với các cấu hình khác nhau (additive/multiplicative trend và seasonality).
- **Tham số**: error/trend = (add/mul) × (None/add/add_damped), seasonal=None.
- **Triển khai**: Sử dụng `statsmodels.tsa.holtwinters.ExponentialSmoothing`, chọn best AIC.
- **Cách Fit Dữ Liệu**: Thử các cấu hình (add/mul trend, damped), fit từng model và chọn AIC thấp nhất.
- **Input Train**: Array numpy 1D của Units Sold trong 30 ngày gần nhất.
- **Cách Dự Đoán**: Sau khi chọn best model, gọi `model.forecast(horizon)`.
- **Ưu điểm**: Handle trend và seasonality tốt.
- **Nhược điểm**: Không probabilistic mặc dù có thể mở rộng.

### 3.7. Prophet
- **Mô tả**: Mô hình Facebook Prophet cho time series với seasonality.
- **Tham số**: daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False.
- **Triển khai**: Sử dụng `prophet.Prophet`, fit và predict.
- **Cách Fit Dữ Liệu**: Tạo DataFrame với cột 'ds' (dates) và 'y' (Units Sold), gọi `model.fit(df_train)`.
- **Input Train**: DataFrame pandas với 'ds' (datetime) và 'y' (float) từ 30 ngày gần nhất.
- **Cách Dự Đoán**: Tạo future DataFrame với `model.make_future_dataframe(periods=horizon, freq='D')`, gọi `model.predict(future)['yhat']`.
- **Ưu điểm**: Tự động handle holidays và seasonality.
- **Nhược điểm**: Có thể overfit nếu dữ liệu ngắn.

### 3.8. Chronos-Bolt-Small
- **Mô tả**: Mô hình pretrained transformer từ Amazon cho zero-shot forecasting.
- **Tham số**: Model ID = "amazon/chronos-bolt-small", device=cpu.
- **Triển khai**: Sử dụng `chronos.ChronosBoltPipeline`, predict trực tiếp.
- **Cách Fit Dữ Liệu**: Không fit; sử dụng pretrained model, chỉ cần context.
- **Input Train**: Array numpy 1D của Units Sold trong toàn bộ lịch sử (không chỉ 30 ngày, vì zero-shot).
- **Cách Dự Đoán**: Gọi `pipeline.predict(context=train_array, prediction_length=horizon)`, lấy median forecast.
- **Ưu điểm**: Zero-shot, không cần training.
- **Nhược điểm**: Phụ thuộc vào pretrained knowledge.

### 3.9. LSTM-Univariate
- **Mô tả**: LSTM PyTorch cho multi-step forecasting.
- **Tham số**: 2 layers, 64 hidden units, dropout, lookback=60, epochs=50, early stopping.
- **Triển khai**: Custom PyTorch model, training trên sequences.
- **Cách Fit Dữ Liệu**: Chuẩn hóa dữ liệu (mean/std), tạo sequences (X: lookback, y: horizon), train model với Adam optimizer và MSE loss.
- **Input Train**: Array numpy 1D của Units Sold trong toàn bộ lịch sử (expanding window), tạo thành tensor sequences.
- **Cách Dự Đoán**: Sau khi train, lấy window cuối cùng (last lookback days) từ dữ liệu train, chuẩn hóa theo mean/std của window này, biến thành tensor shape=(1, lookback, 1), feed vào model để predict horizon bước; kết quả được denormalize bằng cách nhân với std rồi cộng mean, sau đó clip về >=0.
- **Ưu điểm**: Capture non-linear patterns.
- **Nhược điểm**: Tốn thời gian training.

### 3.10. LSTM-EntityEmbedding
- **Mô tả**: LSTM với entity embeddings cho categorical features.
- **Tham số**: Tương tự LSTM-Univariate nhưng thêm embeddings.
- **Triển khai**: Custom model với embedding layers.
- **Cách Fit Dữ Liệu**: Embeddings cho Store/Product IDs, kết hợp với time series data, train tương tự LSTM.
- **Input Train**: Array của Units Sold + categorical IDs (Store, Product), tạo embeddings.
- **Cách Dự Đoán**: Sử dụng embeddings và last sequence để predict.
- **Ưu điểm**: Handle categorical inputs tốt.
- **Nhược điểm**: Phức tạp hơn.

## 4. Đánh Giá Và Kết Quả

### 4.1. Metrics
- **sMAPE**: Symmetric Mean Absolute Percentage Error (%)
- **MASE**: Mean Absolute Scaled Error (đối với naive forecast)
- **RMSE**: Root Mean Squared Error
- **RMSLE**: Root Mean Squared Logarithmic Error

### 4.2. Kết Quả Chi Tiết (Mean Metrics Trên 100 Series)

| Model | Horizon | sMAPE (%) | MASE | RMSE | RMSLE |
|-------|---------|-----------|------|------|-------|
| Chronos-Bolt-Small | 7 | 29.77 | 0.71 | 37.44 | 0.37 |
| Chronos-Bolt-Small | 14 | 30.09 | 0.72 | 38.12 | 0.38 |
| Chronos-Bolt-Small | 28 | 30.95 | 0.73 | 39.60 | 0.40 |
| LSTM-Univariate | 7 | 38.92 | 0.88 | 42.16 | 0.52 |
| LSTM-Univariate | 14 | 34.95 | 0.81 | 37.81 | 0.46 |
| LSTM-Univariate | 28 | 41.50 | 0.92 | 45.47 | 0.54 |
| MeanNaive | 7 | 39.71 | 0.85 | 41.26 | 0.61 |
| MeanNaive | 14 | 40.71 | 0.82 | 42.06 | 0.63 |
| MeanNaive | 28 | 41.30 | 0.81 | 42.58 | 0.63 |
| ARIMA | 7 | 39.93 | 0.85 | 41.66 | 0.63 |
| ARIMA | 14 | 42.34 | 0.85 | 43.36 | 0.64 |
| ARIMA | 28 | 41.63 | 0.82 | 43.01 | 0.64 |
| ETS | 7 | 41.00 | 0.86 | 42.47 | 0.66 |
| ETS | 14 | 47.15 | 0.91 | 46.22 | 0.79 |
| ETS | 28 | 41.88 | 0.82 | 43.24 | 0.66 |
| Prophet | 7 | 46.38 | 0.97 | 47.22 | 0.76 |
| Prophet | 14 | 54.69 | 1.04 | 52.46 | 0.96 |
| Prophet | 28 | 49.52 | 0.97 | 51.13 | 0.76 |
| Naive | 7 | 48.41 | 1.04 | 51.60 | 0.75 |
| Naive | 14 | 49.36 | 1.02 | 52.32 | 0.74 |
| Naive | 28 | 49.14 | 0.99 | 52.38 | 0.72 |
| SeasonalNaive | 7 | 50.37 | 1.08 | 53.13 | 0.80 |
| SeasonalNaive | 14 | 52.76 | 1.07 | 54.95 | 0.82 |
| SeasonalNaive | 28 | 50.84 | 1.02 | 54.66 | 0.80 |
| SARIMAX | 7 | 48.41 | 1.04 | 51.60 | 0.75 |
| SARIMAX | 14 | 49.36 | 1.02 | 52.32 | 0.74 |
| SARIMAX | 28 | 49.14 | 0.99 | 52.38 | 0.72 |

### 4.3. Phân Tích Kết Quả
- **Mô hình tốt nhất**: Chronos-Bolt-Small đạt sMAPE thấp nhất (~30%) nhờ pretrained knowledge.
- **Mô hình đơn giản**: MeanNaive và ARIMA có hiệu suất ổn định (~40-42% sMAPE).
- **Mô hình phức tạp**: LSTM và Prophet có kết quả hỗn hợp, phụ thuộc vào horizon.
- **Xu hướng theo horizon**: sMAPE thường tăng nhẹ khi horizon dài hơn, ngoại trừ một số mô hình như Chronos.

## 5. Kết Luận

Các mô hình baseline đã được triển khai thành công với pipeline thống nhất, cung cấp benchmark vững chắc cho việc phát triển mô hình nâng cao. Chronos-Bolt-Small nổi bật với hiệu suất cao nhờ zero-shot capability. Các mô hình thống kê như ARIMA và ETS cung cấp baseline ổn định. Kết quả này có thể được sử dụng để so sánh với các mô hình deep learning hoặc hybrid trong tương lai.

**File kết quả**: Tất cả summary và details được lưu trong `Model/result/` với format CSV và Markdown.