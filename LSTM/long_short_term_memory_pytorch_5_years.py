import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_percentage_error as mape

import numpy as np
import pandas as pd

import os
import random
import time

import multiprocessing

from matplotlib import pyplot as plt

import warnings
from warnings import simplefilter

from metrics_lstm import rmse, pbe, pocid, mase
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler

from functions_forecasting import recursive_multistep_forecasting

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

# Function to convert a date string into a pandas Timestamp
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

def create_dataset_recursive(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

def create_dataset_direct(data, time_steps=1, forecast_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps - forecast_steps + 1):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps:i + time_steps + forecast_steps, 0])
    return np.array(X), np.array(y)

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
            example = np.array(series[i:i + window + 1])
            data.append(example)
    else:
        for i in range(len(series) - window):
            example = np.array(series[i:i + window + 12])
            data.append(example)

    df = pd.DataFrame(data)
    df = df.dropna()
    return df

def train_test_split_cisia(data, horizon, type_predictions):
    if type_predictions == "recursive":
        X = data.iloc[:,:-1] # features
        y = data.iloc[:,-1] # target

        X_train = X[:-horizon] # features train
        X_test =  X[-horizon:] # features test

        y_train = y[:-horizon] # target train
        y_test = y[-horizon:] # target test

    else: 
        X = data.iloc[:, :12]  # primeiras 12 colunas como features
        y = data.iloc[:, 12:]  # últimas 12 colunas como target

        # Definir os conjuntos de treino e teste
        X_train = X[:-24]  # features train
        X_test = X[-1:]   # features test

        y_train = y[:-24]  # target train
        y_test = y[-1:]   # target test
        
    return X_train, X_test, y_train, y_test

def recursive_multistep_forecasting(X_test, model, horizon, device):
    # Converter o X_test para tensor do PyTorch
    model.eval()  # Colocar o modelo em modo de avaliação
    example = torch.tensor(X_test[0].reshape(1, X_test.shape[1], X_test.shape[2]), dtype=torch.float32).to(device)  # Mover o tensor para a GPU

    preds = []
    for i in range(horizon):
        with torch.no_grad():
            # Realizar a previsão com o modelo
            pred = model(example).cpu().numpy()[0]  # Mover para CPU para converter para numpy
        preds.append(pred)

        # Descartar o valor da primeira posição e adicionar o predito
        example = example[:, 1:, :]  # Remove o primeiro valor
        pred_tensor = torch.tensor(pred.reshape(1, 1, -1), dtype=torch.float32).to(device)  # Mover o tensor de previsão para a GPU
        example = torch.cat((example, pred_tensor), dim=1)  # Adiciona a previsão no final

    return np.array(preds)

def train_test_stats(data, horizon):
  train, test = data[:-horizon], data[-horizon:]
  return train, test

def reverse_diff(last_value_train, preds):    
    preds_series = pd.Series(preds)
    preds = pd.concat([last_value_train, preds_series], ignore_index=True)
    preds_cumsum = preds.cumsum()

    return preds_cumsum[1:].values

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

def create_lstm_model(forecast_steps, time_steps, data, epochs, state, product, batch_size, type_predictions='recursive', show_plot=None):
    """
    Create and train an LSTM model for time series forecasting with recursive or direct predictions.

    Parameters:
        - forecast_steps (int): Number of steps to forecast.
        - time_steps (int): Length of the time window for input sequences.
        - data (DataFrame): Input data containing the time series values (m3).
        - epochs (int): Number of training epochs.
        - state (str): State identifier for the data.
        - product (str): Product identifier for the data.
        - batch_size (int): Size of the training batches.
        - type_predictions (str, optional): Prediction mode, either 'recursive' or 'direct'. Default is 'recursive'.
        - show_plot (bool, optional): Flag to display and save plots of predictions.

    Returns:
        - y_pred (np.array): Array of the forecasted points.
    """

    y_preds_5_years = []

    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

    end_date = data['timestamp'].max()

    for years in range(5, 0, -1):
        start_date = end_date - pd.DateOffset(years=years-1)

        df = data[data['timestamp'] <= start_date]
        print(f'\nData filtered for {start_date.date()}\n')
        print(df)
        df = df['m3']

        # Data preparation based on prediction type
        if type_predictions == 'recursive':
            int_dense = 1

            # Prepare data for recursive predictions
            data_recursive = rolling_window(df, time_steps, type_predictions)
            X_train, X_test, y_train, y_test = train_test_split_cisia(data_recursive, forecast_steps, type_predictions)

            y_train = y_train.values.reshape(-1, 1)
            y_test = y_test.values.reshape(-1, 1)
            y_train = np.hstack([y_train, np.zeros((y_train.shape[0], time_steps - 1))])
            y_test = np.hstack([y_test, np.zeros((y_test.shape[0], time_steps - 1))])

            scaler = MinMaxScaler(feature_range=(-1, 1))
            X_train = scaler.fit_transform(X_train)
            y_train = scaler.transform(y_train)[:, 0].reshape(-1, 1)
            X_test = scaler.transform(X_test)
            y_test = scaler.transform(y_test)[:, 0].reshape(-1, 1)

        elif type_predictions == 'direct':
            int_dense = 12

            # Prepare data for direct predictions
            data_direct = df.values.reshape(-1, 1)
            X, y = create_dataset_direct(data_direct, time_steps, forecast_steps)

            X_train, X_test = X[:-24], X[-1:]
            y_train, y_test = y[:-24], y[-1:]

            scaler = MinMaxScaler(feature_range=(-1, 1))
            X_train = scaler.fit_transform(X_train)
            y_train = scaler.transform(y_train)
            X_test = scaler.transform(X_test)
            y_test = scaler.transform(y_test)

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

        # Forecasting and evaluation
        if type_predictions == "recursive":
            y_pred = scaler.inverse_transform(np.hstack([recursive_multistep_forecasting(X_test, model, forecast_steps, device).reshape(-1, 1), np.zeros((forecast_steps, time_steps - 1))]))[:, 0]
        else:
            model.eval()

            X_test = X_test.to(device)

            with torch.no_grad():
                y_pred = model(X_test).cpu().numpy()  

            y_pred = scaler.inverse_transform(y_pred).flatten()

        y_test = df[-forecast_steps:].values

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
                    
def run_lstm(state, product, forecast_steps, time_steps, data_filtered, epochs, bool_save, log_lock, batch_size, type_predictions='recursive'):
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
        create_lstm_model(forecast_steps=forecast_steps, time_steps=time_steps, epochs=epochs, data=data_filtered, state=state, product=product, batch_size=batch_size, type_predictions=type_predictions, show_plot=True)
        
        # Create a DataFrame to store the results
        results_df = pd.DataFrame([{'FORECAST_STEPS': forecast_steps,
                                    'TIME_FORECAST': time_steps,
                                    'TYPE_PREDICTIONS': 'LSTM_PYTORCH_' + type_predictions + '_5_years',
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
                                    'TYPE_PREDICTIONS': 'LSTM_PYTORCH_' + type_predictions + '_5_years',
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
        with log_lock:
            directory = f'results'
            if not os.path.exists(directory):
                os.makedirs(directory)

            file_path = os.path.join(directory, 'lstm_results_pytorch_5_years.xlsx')
            if os.path.exists(file_path):
                existing_df = pd.read_excel(file_path)
            else:
                existing_df = pd.DataFrame()

            combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            combined_df.to_excel(file_path, index=False)

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

def run_lstm_in_thread(forecast_steps, time_steps, epochs, bool_save, batch_size, type_predictions='recursive'):
    """
    Execute LSTM model training in separate processes for different state and product combinations.

    Parameters:
        - forecast_steps (int): Number of steps to forecast in the future.
        - time_steps (int): Length of time steps (window size) for input data generation.
        - epochs (int): Number of training epochs for the LSTM model.
        - bool_save (bool): Whether to save the trained models (True/False).
        - batch_size (int): Batch size for model training.
        - type_predictions (str, optional): Type of prediction method ('recursive' or 'direct_dense12'). Default is 'recursive'.

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
                    data_filtered, epochs, bool_save,
                    log_lock, batch_size, type_predictions
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
    y_pred = \
    create_lstm_model(forecast_steps=12, time_steps=12, data=data_filtered_test, epochs=100, state=state, product=product,
                      batch_size=16, type_predictions='recursive', show_plot=True)

    # Recording end time and calculating execution duration
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nFunction execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")