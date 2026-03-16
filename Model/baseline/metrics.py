import numpy as np 

def smape(forecasts, actuals): 
     s = 2 * np.abs(forecasts - actuals) / (np.abs(forecasts) + np.abs(actuals) + 1e-8) 
     return s.mean(axis=1) * 100 

def mase(forecasts, actuals, training_set, lag=52): 
     result = [] 
     for i, series in enumerate(training_set): 
          arr = np.array(series) 
          denom = np.mean(np.abs(arr[lag:] - arr[:-lag])) if len(arr) > lag else np.mean(np.abs(np.diff(arr))) 
          if denom == 0 or np.isnan(denom): 
               denom = np.mean(np.abs(np.diff(arr))) or 1.0 
               result.append(np.mean(np.abs(forecasts[i] - actuals[i])) / denom) 
               return np.array(result) 
          
def rmse(forecasts, actuals): 
     return np.sqrt(np.mean((forecasts - actuals) ** 2, axis=1)) 

def rmsle(forecasts, actuals): 
     fc = np.clip(forecasts, 0, None) 
     ac = np.clip(actuals, 0, None) 
     return np.sqrt(np.mean((np.log1p(fc) - np.log1p(ac)) ** 2, axis=1))