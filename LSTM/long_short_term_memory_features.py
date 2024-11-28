import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_percentage_error as mape

import numpy as np
import pandas as pd

import os
import random
import time

import gc
import multiprocessing

from matplotlib import pyplot as plt

import warnings
from warnings import simplefilter

from metrics_lstm import rmse, pbe, pocid, mase
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler

import pycatch22
from cesium import featurize
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features
from tsfresh import select_features
import tsfeatures

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
simplefilter(action='ignore', category=FutureWarning)

#Reproducibilty
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'

features_to_use_all = [
        "amplitude",
        "anderson_darling",
        "flux_percentile_ratio_mid20",
        "flux_percentile_ratio_mid35",
        "flux_percentile_ratio_mid50",
        "flux_percentile_ratio_mid65",
        "flux_percentile_ratio_mid80",
        "percent_beyond_1_std",
        "percent_beyond_1_std",
        "period_fast",
        "qso_log_chi2_qsonu",
        "qso_log_chi2nuNULL_chi2nu",
        "shapiro_wilk",
        "stetson_j",
        "stetson_k",
        "weighted_average",
        "maximum",
        "max_slope",
        "median",
        "median_absolute_deviation",
        "percent_close_to_median",
        "minimum",
        "skew",
        "std",
        "weighted_average",
        "all_times_nhist_numpeaks",
        "all_times_nhist_peak1_bin",
        "all_times_nhist_peak2_bin",
        "all_times_nhist_peak3_bin",
        "all_times_nhist_peak4_bin",
        "all_times_nhist_peak_1_to_2",
        "all_times_nhist_peak_1_to_3",
        "all_times_nhist_peak_1_to_4",
        "all_times_nhist_peak_2_to_3",
        "all_times_nhist_peak_2_to_4",
        "all_times_nhist_peak_3_to_4",
        "all_times_nhist_peak_val",
        "avg_double_to_single_step",
        "avg_err",
        "avgt",
        "cad_probs_1",
        "cad_probs_10",
        "cad_probs_20",
        "cad_probs_30",
        "cad_probs_40",
        "cad_probs_50",
        "cad_probs_100",
        "cad_probs_500",
        "cads_avg",
        "cads_kurtosis",
        "cads_med",
        "cads_skew",
        "cads_std",
        "mean",
        "med_err",
        "n_epochs",
        "std_double_to_single_step",
        "std_err",
        "total_time",
        "fold2P_slope_10percentile",
        "fold2P_slope_90percentile",
        "freq1_amplitude1",
        "freq1_amplitude2",
        "freq1_amplitude3",
        "freq1_amplitude4",
        "freq1_freq",
        "freq1_lambda",
        "freq1_rel_phase2",
        "freq1_rel_phase3",
        "freq1_rel_phase4",
        "freq1_signif",
        "freq2_amplitude1",
        "freq2_amplitude2",
        "freq2_amplitude3",
        "freq2_amplitude4",
        "freq2_freq",
        "freq2_rel_phase2",
        "freq2_rel_phase3",
        "freq2_rel_phase4",
        "freq3_amplitude1",
        "freq3_amplitude2",
        "freq3_amplitude3",
        "freq3_amplitude4",
        "freq3_freq",
        "freq3_rel_phase2",
        "freq3_rel_phase3",
        "freq3_rel_phase4",
        "freq_amplitude_ratio_21",
        "freq_amplitude_ratio_31",
        "freq_frequency_ratio_21",
        "freq_frequency_ratio_31",
        "freq_model_max_delta_mags",
        "freq_model_min_delta_mags",
        "freq_model_phi1_phi2",
        "freq_n_alias",
        "freq_signif_ratio_21",
        "freq_signif_ratio_31",
        "freq_varrat",
        "freq_y_offset",
        "linear_trend",
        "medperc90_2p_p",
        "p2p_scatter_2praw",
        "p2p_scatter_over_mad",
        "p2p_scatter_pfold_over_mad",
        "p2p_ssqr_diff_over_var",
        "scatter_res_raw",
    ]

def convert_date(date_string):
    """
    Convert a date string into a pandas Timestamp.

    Parameters:
    - date_string: str, date in 'YYYYMM' format

    Returns:
    - pd.Timestamp object representing the date
    """
    year_month = date_string.strip()
    year = int(year_month[:4])
    month = int(year_month[4:])
    return pd.Timestamp(year=year, month=month, day=1)

def znorm(x):
  epslon = 0.00005
  if np.std(x)!=0:
    x_znorm = (x - np.mean(x)) / np.std(x)
  else:
    x_znorm = (x - np.mean(x)) / (np.std(x) + epslon) 
   
  return x_znorm

def znorm_reverse(x, mean_x, std_x):
  x_denormalized = (np.array(x) * std_x) + mean_x
  return x_denormalized

def get_stats_norm(series, horizon, window):
  last_subsequence = series[-(horizon+window):-horizon].values
  last_mean = np.mean(last_subsequence)
  last_std = np.std(last_subsequence)
  return last_mean, last_std

# ============== Rolling Window Features ==============

def rolling_window(series, window, type_predictions):
    """
    Generate rolling window data for time series analysis.

    Parameters:
    - series: array-like, time series data
    - window: int, size of the rolling window

    Returns:
    - df: pandas DataFrame, containing the rolling window data
    - scaler: MinMaxScaler object, used for normalization
    """
    data = []

    if type_predictions == "recursive":
        for i in range(len(series) - window):
            example = znorm(np.array(series[i:i+window+1]))
            data.append(example)
    else:
        for i in range(len(series) - window):
            example = znorm(np.array(series[i:i+window+12]))
            data.append(example)

    df = pd.DataFrame(data)
    return df

def rolling_window_real(series, window):
  data = []
  for i in range(len(series)-window):
    # example = znorm(np.array(series[i:i+window+1]))
    example = np.array(series[i:i+window+1])
    data.append(example)
  df = pd.DataFrame(data)
  return df

def rolling_window_featureCatch22N(series, window):
  data = []
  for i in range(len(series)-window):
      example = np.array(series[i:i+window])
      new_elements = pycatch22.catch22_all(example)
      data_feature = znorm(new_elements['values'])
      data.append(data_feature)
  df = pd.DataFrame(data)
  return df

def rolling_window_featureTsCesiumN(series, window):
    data_out3 = []

    for i in range(len(series)-window):
        example = ((series[i:i+window]))
        fset_cesium = featurize.featurize_time_series(times=example["timestamp"],values=example["m3"].values, errors=None,features_to_use=features_to_use_all)
        new_elements_values_reshaped = np.squeeze(fset_cesium.values) 
        data_featuredf = pd.DataFrame(new_elements_values_reshaped)
        data_featuredf2 = data_featuredf#.apply(znorm, axis=1)
        data_out3.append(data_featuredf2.T.values)
    data_out4 = np.squeeze(data_out3) 
    df2 = pd.DataFrame((data_out4))
    return df2

def rolling_window_TsFreshN(series, window):
    concatenated_results = None  # Initialize as None
    for i in range(len(series) - window):  # Adjusted range to include the last window
        target_feat = series[i:i+window]
        target_feat2 = pd.DataFrame(target_feat)

        target_feat2.insert(0, 'id', 'D1')  # Example: inserting 'D1' as id

        result = extract_features(target_feat2, column_id="id")
        result = result.apply(znorm, axis=1)
        # result = result.fillna(0, inplace=True)
        
        result_values = result.values
        # result_values = result
        
        if concatenated_results is None:
            concatenated_results = result_values
        else:
            concatenated_results = np.concatenate((concatenated_results, result_values), axis=0)
    
    return concatenated_results

def rolling_window_featureTsFeaturesN(out_window, window, freqF):
  results=[]
  for i in range(len(out_window)-window):
    target_feat = out_window[i:window+i]
    target_feat.insert(0, 'unique_id', 'D1')
    result = tsfeatures.tsfeatures(target_feat, freq=freqF)
    result = result.drop(columns=['unique_id'])
    result = result.apply(znorm, axis=1)
    results.append((result.values))
  return results

# ============== Forecasting Features ==============

def recursive_multistep_forecasting_Catch22N(series, model, horizon, window, type_predictions):
    serieatu=series[:-horizon]
    seriec = serieatu.tail(window)
    mean_norm, std_norm = get_stats_norm(series, horizon, window)
    predsreal = []
    if type_predictions == 'recursive':
        for i in range(horizon):
            example = np.array(seriec)
            new_elements = pycatch22.catch22_all(example)
            exemple_feature = znorm(new_elements['values'])
            exemple_feature_df = pd.DataFrame(exemple_feature)
            exemple_feature_df.fillna(0, inplace=True)
            predn = model.predict(exemple_feature_df.T)[0]
            pred = znorm_reverse(predn, mean_norm, std_norm)
            predsreal.append(pred)
            series2 = seriec[1:]
            seriec = np.append(series2, pred)
    else:
        example = np.array(seriec)
        new_elements = pycatch22.catch22_all(example)
        exemple_feature = znorm(new_elements['values'])
        exemple_feature_df = pd.DataFrame(exemple_feature)
        exemple_feature_df.fillna(0, inplace=True)
        predn = model.predict(exemple_feature_df.T)[0]
        pred = znorm_reverse(predn, mean_norm, std_norm)
        predsreal.append(pred)
        series2 = seriec[1:]
        seriec = np.append(series2, pred)
    
    return predsreal

def recursive_multistep_forecasting_TsCesiumN(series, DataCesium, model, horizon, window):

    seriec = DataCesium
    preds = []
    mean_norm, std_norm = get_stats_norm(series, horizon, window)
    for i in range(horizon):
        example = seriec
        fset_cesium = featurize.featurize_time_series(times=example["timestamp"],values=example["y"].values, errors=None,features_to_use=features_to_use_all)
        new_elements_values_reshaped = np.squeeze(fset_cesium.values) 
        data_featuredf = pd.DataFrame(new_elements_values_reshaped)
        data_featuredf.apply(znorm)
        data_featuredf.fillna(0, inplace=True)
        data_featuredf.replace([float('inf'), -float('inf')], 0, inplace=True)
        predn = model.predict(data_featuredf.T)[0]
        pred = znorm_reverse(predn, mean_norm, std_norm)
        preds.append(pred)
        series2 = seriec.drop(seriec.index[0])
        last_timestamp = series2['timestamp'].iloc[-1]
        year = int(str(last_timestamp)[:4])
        month = int(str(last_timestamp)[4:])
        if month == 12:
            next_year = year + 1
            next_month = 1
        else:
            next_year = year
            next_month = month + 1
        next_timestamp = next_year * 100 + next_month
        new_row = pd.DataFrame({'timestamp': [next_timestamp], 'y': [pred]})
        seriec = pd.concat([series2, new_row], ignore_index=True)

    return preds

def recursive_multistep_forecasting_TsFreshN(series, model, horizon, window):
    serieatu=series[:-horizon]
    seriec = serieatu.tail(window).reset_index(drop=True)
    mean_norm, std_norm = get_stats_norm(series, horizon, window)
    predsreal = []  

    for i in range(horizon):
        target_feat = pd.DataFrame(seriec)
        target_feat.insert(0, 'id', 'D1')  # Example: inserting 'D1' as id
        result = extract_features(target_feat, column_id="id")
        result = result.apply(znorm, axis=1)
        result.fillna(0, inplace=True)
        # result_values = result.values
        result.columns = range(0, len(result.columns))
        predN = model.predict(result)
        pred = znorm_reverse(predN, mean_norm, std_norm)    
        predsreal.append(pred)
        series2 = seriec[1:]
        seriec = np.append(series2, pred)
        preds2 = [val[0] for val in predsreal]

    return preds2

def recursive_multistep_forecasting_TsFeaturesN(series, model, horizon, freqF, window):
    serieatu=series[:-horizon]
    seriec = serieatu.tail(2*window).reset_index(drop=True)
    mean_norm, std_norm = get_stats_norm(series, horizon, window)
    predsreal = []  
    for i in range(horizon):
        out_window2 = rolling_window_real(seriec,window)
        out_window2.rename(columns={window: 'y'}, inplace=True)
        example = out_window2
        example2 = pd.DataFrame(example)
        example2.insert(0, 'unique_id', 'D1')
        result = tsfeatures.tsfeatures(example2, freq=freqF)
        result = result.drop(columns=['unique_id'])
        result.apply(znorm)
        result.fillna(0, inplace=True)
        result.columns = range(result.shape[1])
        predN = model.predict(result)
        pred = znorm_reverse(predN, mean_norm, std_norm)    
        predsreal.append(pred)
        series2 = seriec[1:]
        seriec = np.append(series2, pred)
        preds2 = [val[0] for val in predsreal]

    return preds2

# ==============================================================
def train_test_split_cisia(data, horizon, type_predictions):
    if type_predictions == "recursive":
        X = data.iloc[:,:-1] # features
        y = data.iloc[:,-1] # target

        X_train = X[:-horizon] # features train
        X_test =  X[-horizon:] # features test

        y_train = y[:-horizon] # target train
        y_test = y[-horizon:] # target test

    else: 
        X = data.iloc[:, :22]  # primeiras 22 colunas como features
        y = data.iloc[:, 22:]  # últimas 12 colunas como target

        # Definir os conjuntos de treino e teste
        X_train = X[:-12]  # features train
        X_test = X[-1:]   # features test

        y_train = y[:-12]  # target train
        y_test = y[-1:]   # target test
        
    return X_train, X_test, y_train, y_test

def recursive_multistep_forecasting(X_test, model, horizon):
  # example é composto pelas últimas observações vistas
  # na prática, é o primeiro exemplo do conjunto de teste
  example = X_test[0].reshape(1,-1)

  preds = []
  for i in range(horizon):
    pred = model.predict(example)[0]
    preds.append(pred)

    # Descartar o valor da primeira posição do vetor de características
    example = example[:,1:]

    # Adicionar o valor predito na última posição do vetor de características
    example = np.append(example, pred)
    example = example.reshape(1,-1)
  return preds

def train_test_stats(data, horizon):
  train, test = data[:-horizon], data[-horizon:]
  return train, test

def set_seed(seed_value=42):
    # Fixar semente para o Python
    random.seed(seed_value)

    # Fixar semente para o NumPy
    np.random.seed(seed_value)

    # Fixar semente para o PyTorch (CPU)
    torch.manual_seed(seed_value)

    # Fixar semente para o PyTorch (GPU) se estiver usando
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # Para múltiplas GPUs
        
    # Assegurar que alguns comportamentos indeterminísticos sejam evitados
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_lstm_model(forecast_steps, time_steps, data, epochs, state, product, batch_size, type_predictions='recursive', type_feature='Catch22', show_plot=None):
    """
    Create and train an LSTM model for time series forecasting.

    Parameters:
        - forecast_steps (int): Number of steps to forecast into the future.
        - time_steps (int): Length of the input sequence (window size) used for training the LSTM model.
        - data (pd.DataFrame): The input dataset containing time series data.
        - epochs (int): Number of epochs for training the LSTM model.
        - state (str): Name of the state for which the model is being trained.
        - product (str): Name of the product for which the model is being trained.
        - batch_size (int): Batch size used for training.
        - type_predictions (str, optional): Type of prediction method ('recursive' or 'direct_dense12'). Default is 'recursive'.
        - type_feature (str, optional): Feature extraction method used ('Catch22', 'TsCesium', 'TsFresh', or 'TsFeatures'). Default is 'Catch22'.
        - show_plot (bool, optional): Whether to display a plot of the model's predictions. Default is None.

    Returns:
        None
    """

    y_preds_5_years = []

    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

    end_date = data['timestamp'].max()

    for years in range(5, 0, -1):
        start_date = end_date - pd.DateOffset(years=years-1)
        
        df = data[data['timestamp'] <= start_date]
        print(f'\nData filtered for {start_date.date()}\n')
        df = df['m3']

        # Feature-specific preprocessing
        if type_feature == 'Catch22':
            # ok
            data_feature = rolling_window_featureCatch22N(df, time_steps)

        elif type_feature == 'TsCesium':
            data_feature = rolling_window_featureTsCesiumN(df, time_steps)

        elif type_feature == 'TsFresh':
            # ok
            data_feature = rolling_window_TsFreshN(df, time_steps)

        elif type_feature == 'TsFeatures':
            freqF = 12
            out_window = rolling_window_real(df, time_steps)
            out_window.rename(columns={time_steps: 'y'}, inplace=True)
            data_feature = rolling_window_featureTsFeaturesN(out_window, time_steps, freqF)
        
        else:
            raise ValueError(f"Feature {type_feature} not recognized or implemented.")

        # Fill and replace NaNs and infinities
        data_feature = data_feature.fillna(0)
        data_feature.replace([float('inf'), -float('inf')], 0, inplace=True)

        # Rolling window for y normalization based on prediction type
        if type_predictions == 'recursive':
            int_dense = 1
            y_norm = rolling_window(df, time_steps, "recursive")
            data_feature['y'] = y_norm.iloc[:, -1] # catch22
            # df_feat2['y'] = y_norm[window] # Tsfresh
            # outTsCelsium_norm['y'] = y_temp2[window] # TsCesium
            X_train, X_test, y_train, y_test = train_test_split_cisia(data, forecast_steps, "recursive")

        elif type_predictions == 'direct_dense12':
            int_dense = 12
            y_norm = rolling_window(df, time_steps, "direct")
            data_feature = pd.concat([data_feature, y_norm.iloc[:, -12:]], axis=1)
            data_feature.dropna(inplace=True)
            X_train, X_test, y_train, y_test = train_test_split_cisia(data, forecast_steps, "direct")

        else:
            raise ValueError(f"Prediction type {type_predictions} not recognized.")
        
    
    # ============ MinMaxScaler ============
    
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)
    y_train = np.hstack([y_train, np.zeros((y_train.shape[0], time_steps - 1))])
    y_test = np.hstack([y_test, np.zeros((y_test.shape[0], time_steps - 1))])

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train)
    y_train = scaler.transform(y_train)[:, 0].reshape(-1, 1)
    X_test = scaler.transform(X_test)
    y_test = scaler.transform(y_test)[:, 0].reshape(-1, 1)

    # ============ Validation ============
    X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42, shuffle=False)

    # ============ Convert data to PyTorch tensors ============
    X_train_val = torch.tensor(X_train_val, dtype=torch.float32).unsqueeze(-1)
    y_train_val = torch.tensor(y_train_val, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)

     #  ============ Define LSTM model ============
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.linear = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.linear(out[:, -1, :])
            return out
        
        # def forward(self, x):
        #     out, _ = self.lstm(x)
        #     out = F.relu(out[:, -1, :])  
        #     out = self.linear(out)
        #     return out
        
        # def forward(self, x):
        #     out, _ = self.lstm(x)
        #     out = torch.tanh(out[:, -1, :])  
        #     out = self.linear(out)
        #     return out

    # Set device and initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)
    input_size = 1
    num_layers = 3
    hidden_size = 128
    output_size = int_dense
    dropout = 0.15

    model = LSTMModel(input_size, hidden_size, num_layers, dropout).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # DataLoader setup
    train_loader = DataLoader(TensorDataset(X_train_val, y_train_val), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        total_test_loss = sum(loss_fn(model(X.to(device)), y.to(device)).item() for X, y in test_loader) / len(test_loader)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}] - Training Loss: {total_loss / len(train_loader):.4f}, Test Loss: {total_test_loss:.4f}')

        # Forecasting
        if type_feature == 'Catch22':
            if type_predictions == "recursive":
                y_pred = recursive_multistep_forecasting_Catch22N(df, model, forecast_steps, time_steps, 'recursive')
            elif type_predictions == "direct_dense12":
                y_pred = np.array(recursive_multistep_forecasting_Catch22N(df, model, forecast_steps, time_steps, 'direct')).flatten()

        elif type_feature == 'TsCesium':
            y_pred = recursive_multistep_forecasting_TsCesiumN(df, 12, model, forecast_steps, time_steps)

        elif type_feature == 'TsFresh':
            y_pred = recursive_multistep_forecasting_TsFreshN(df, model, forecast_steps, time_steps)

        elif type_feature == 'TsFeatures':
            y_pred = recursive_multistep_forecasting_TsFeaturesN(df, model, forecast_steps, freqF, time_steps)
        
        y_test = df[-forecast_steps:].values
        y_baseline = df[-forecast_steps * 2:-forecast_steps].values

        # Evaluation metrics
        y_baseline = df[-forecast_steps*2:-forecast_steps].values
        rmse_result_lstm = rmse(y_test, y_pred)
        mape_result_lstm = mape(y_test, y_pred)
        pbe_result_lstm = pbe(y_test, y_pred)
        pocid_result_lstm = pocid(y_test, y_pred)
        mase_result_lstm = np.mean(np.abs(y_test - y_pred)) / np.mean(np.abs(y_test - y_baseline))

        print("\nResultados LSTM: \n")
        print(f'RMSE: {rmse_result_lstm}')
        print(f'MAPE: {mape_result_lstm}')
        print(f'PBE: {pbe_result_lstm}')
        print(f'POCID: {pocid_result_lstm}')
        print(f'MASE: {mase_result_lstm}')
    
        # return rmse_result_lstm, mape_result_lstm, pbe_result_lstm, pocid_result_lstm, mase_result_lstm, y_pred, batch_size
        y_preds_5_years.append(y_pred)
    
    return np.concatenate(y_preds_5_years).tolist()
    
                   
def run_lstm(state, product, forecast_steps, time_steps, data_filtered, epochs, bool_save, log_lock, batch_size, type_predictions='recursive', type_feature='Catch22'):
    """
    Execute LSTM model training and save the results to an Excel file.

    Parameters:
        - state (str): State for which the LSTM model is trained.
        - product (str): Product for which the LSTM model is trained.
        - forecast_steps (int): Number of steps to forecast in the future.
        - time_steps (int): Length of time steps (window size) for input data generation.
        - data_filtered (pd.DataFrame): Filtered dataset containing data for the specific state and product.
        - epochs (int): Number of training epochs for the LSTM model.
        - bool_save (bool): Whether to save the results to an Excel file.
        - log_lock (multiprocessing.Lock): Lock for synchronized logging.
        - batch_size (int): Batch size for model training.
        - type_predictions (str, optional): Type of prediction method ('recursive' or 'direct_dense12'). Default is 'recursive'.
        - type_features (str, optional): Type of feature extraction method ('Catch22', 'TsCesium', 'TsFresh', or 'TsFeatures'). Default is None.

    Returns:
        None
    """

    # Record the start time of execution
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nExecution started at: {start_timestamp}")
    start_time = time.time()

    try:
        # Run LSTM model training and capture performance metrics
        y_pred = \
        create_lstm_model(forecast_steps=forecast_steps, time_steps=time_steps, epochs=epochs, data=data_filtered, state=state, product=product, batch_size=batch_size, type_predictions=type_predictions, type_feature=type_feature, show_plot=True)
        
        # Create a DataFrame to store the results
        results_df = pd.DataFrame([{'FORECAST_STEPS': forecast_steps,
                                    'TIME_FORECAST': time_steps,
                                    'TYPE_PREDICTIONS': type_feature + '_LSTM_PYTORCH_' + type_predictions,
                                    'STATE': state,
                                    'PRODUCT': product,
                                    'RMSE': np.nan,
                                    'RRMSE': np.nan,
                                    'MAPE': np.nan,
                                    'PBE': np.nan,
                                    'POCID': np.nan,
                                    'MASE': np.nan,
                                    'PREDICTIONS': y_pred,
                                    'BATCH_SIZE': batch_size,
                                    'ERROR': np.nan}])
    except Exception as e:
        # Handle exceptions during model training
        print(f"An error occurred for product '{product}' in state '{state}': {e}")
        
        results_df = pd.DataFrame([{'FORECAST_STEPS': np.nan,
                                    'TIME_FORECAST': np.nan,
                                    'TYPE_PREDICTIONS': type_feature + '_LSTM_PYTORCH' + type_predictions,
                                    'STATE': state,
                                    'PRODUCT': product,
                                    'RMSE': np.nan,
                                    'RRMSE': np.nan,
                                    'MAPE': np.nan,
                                    'PBE': np.nan,
                                    'POCID': np.nan,
                                    'MASE': np.nan,
                                    'PREDICTIONS': np.nan,
                                    'BATCH_SIZE': np.nan,
                                    'ERROR': f"An error occurred for product '{product}' in state '{state}': {e}"}])
            
    # Save the results to an Excel file if requested
    if bool_save:
        output_dir = 'result'
        os.makedirs(output_dir, exist_ok=True)

        file_path = os.path.join(output_dir, 'lstm_results_pytorch_features.xlsx')

        # Load existing results if the file exists
        if os.path.exists(file_path):
            existing_df = pd.read_excel(file_path)
        else:
            existing_df = pd.DataFrame()

        # Combine new results with existing data and save
        combined_df = pd.concat([existing_df, results_df], ignore_index=True)
        combined_df.to_excel(file_path, index=False)

        print(f"Results saved to {file_path}")

    ## Calculate and display the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Function execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Log execution details
    log_entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')}, {state}, {product}, {forecast_steps}, {time_steps}, {type_predictions}, {execution_time:.2f}\n"

    # Write the log entry with a lock to prevent race conditions
    with log_lock:
        with open("training_log.csv", "a") as log_file:
            log_file.write(log_entry)

def run_lstm_in_thread(forecast_steps, time_steps, epochs, bool_save, batch_size, save_model=None, type_predictions='recursive', type_feature='Catch22'):
    """
     Execute LSTM model training in separate processes for different state and product combinations.

    Parameters:
        - forecast_steps (int): Number of steps to forecast in the future.
        - time_steps (int): Length of time steps (window size) for input data generation.
        - epochs (int): Number of training epochs for the LSTM model.
        - bool_save (bool): Whether to save the trained models (True/False).
        - batch_size (int): Batch size for model training.
        - save_model (bool, optional): If True, the trained model will be saved. Default is None.
        - type_predictions (str, optional): Type of prediction method ('recursive' or 'direct_dense12'). Default is 'recursive'.
        - type_features (str, optional): Type of feature extraction method ('Catch22', 'TsCesium', 'TsFresh', or 'TsFeatures'). Default is None.

    Returns:
        None
    """
    # Set the multiprocessing start method
    multiprocessing.set_start_method("spawn")

     # Lock for logging to prevent concurrent access
    log_lock = multiprocessing.Lock()

     # Load combined dataset
    data_path = '../database/combined_data.csv'
    try:
        all_data = pd.read_csv(data_path, sep=";")
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return

    # Group products by state for efficient processing
    state_product_dict = {
        state: list(all_data[all_data['state'] == state]['product'].unique())
        for state in all_data['state'].unique()
    }

    # Iterate through each state and associated products
    for state, products in state_product_dict.items():
        for product in products:
            print(f"========== Processing State: {state}, Product: {product} ==========")

            # Filter data for the current state and product
            data_filtered = all_data[(all_data['state'] == state) & (all_data['product'] == product)]

            # Create a separate process for running the LSTM model
            process = multiprocessing.Process(
                target=run_lstm,
                args=(
                    state, product, forecast_steps, time_steps,
                    data_filtered, epochs, bool_save, save_model,
                    log_lock, batch_size, type_predictions, type_feature
                )
            )

            # Start and wait for the process to complete
            process.start()
            process.join()

    print("All processes completed successfully.")
    
def product_and_single_thread_testing():    
    """
    Perform a simple training thread using LSTM model for time series forecasting.

    This function initializes random seeds, loads a database, executes an LSTM model,
    evaluates its performance, and prints results.

    Parameters:
        None

    Returns:
        None
    """

    state = "sp"
    product = "gasolinac"
    
    # Loading and preparing data
    data_filtered_test = pd.read_csv(f"../database/venda_process/mensal/uf/{product}/mensal_{state}_{product}.csv", sep=";",  parse_dates=['timestamp'], date_parser=convert_date)

    print(f" ========== Starting univariate test for the state of {state} - {product} ==========")

    # Recording start time of execution
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nExecution started at: {start_timestamp}")
    start_time = time.time()

    # Running the LSTM model
    rmse_result, mape_result, pbe_result, pocid_result, mase_result, y_pred, batch_size = \
    create_lstm_model(forecast_steps=12, time_steps=12, data=data_filtered_test, epochs=100, state=state, product=product,
                      batch_size=16, type_predictions='recursive', type_feature='Catch22', show_plot=True)

    # Recording end time and calculating execution duration
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nFunction execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

product_and_single_thread_testing()