import torch
from transformers import AutoModelForCausalLM
from time_moe.models.modeling_time_moe import TimeMoeForPrediction

import tensorflow as tf

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

from metrics_time_moe import rmse, pbe, pocid, mase # type: ignore
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler

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

def rolling_window(series, window):
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
   
    for i in range(len(series) - window):
        example = np.array(series[i:i + window + 1])
        data.append(example)

    df = pd.DataFrame(data)
    df = df.dropna()
    return df

def train_test_split_cisia(data, horizon):
    
    X = data.iloc[:,:-1] # features
    y = data.iloc[:,-1] # target

    X_train = X[:-horizon] # features train
    X_test =  X[-horizon:] # features test

    y_train = y[:-horizon] # target train
    y_test = y[-horizon:] # target test

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

def reverse_diff(last_value_train, preds):    
    preds_series = pd.Series(preds)
    preds = pd.concat([last_value_train, preds_series], ignore_index=True)
    preds_cumsum = preds.cumsum()

    return preds_cumsum[1:].values

def get_scaled_data(df, len_data=None):
    df = df[:-12]
    
    if len_data is not None:
        df = df[-len_data:]
    
    # MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_scaled = scaler.fit_transform(df.values.reshape(-1, 1))

    return df_scaled, scaler

def create_time_moe_model(forecast_steps, time_steps, data, state, product, type_model='TimeMoE-50M', show_plot=None):
    """
    Runs an time_moe model for time series forecasting.

    Parameters:
    - forecast_steps (int): Number of steps ahead for forecasting.
    - time_steps (int): Length of the time sequences for generating the attribute-value table.
    - data (pd.DataFrame): DataFrame containing the time series data.
    - epochs (int): Number of epochs for training the time_moe model.
    - state (str): The specific state for which the model is trained.
    - product (str): The specific product for which the model is trained.
    - batch_size: Batch size for training.
    - type_model (str, optional): Type of time_moe to be used. Default is 'TimeMoE-50M'.
    - show_plot (bool or None, optional): Whether to display a plot of the forecasted values. Default is None.

    """

    y_preds_5_years = []

    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

    end_date = data['timestamp'].max()

    for years in range(5, 0, -1):
        start_date = end_date - pd.DateOffset(years=years-1)

        df = data[data['timestamp'] <= start_date]
        print(f'\nData filtered for {start_date.date()}\n')
        df = df['m3']
   
        len_data = None
        df_scaled, scaler = get_scaled_data(df, len_data)

        tensor = torch.tensor(df_scaled, dtype=torch.float32)

        tensor = tensor.squeeze(-1).unsqueeze(0)

        ''' 
        # INFO: ZERO SHOT
        ''' 
        if type_model == "TimeMoE-50M_ZERO_SHOT":
            model = AutoModelForCausalLM.from_pretrained(
                'Maple728/TimeMoE-50M',
                device_map="cpu",  
                trust_remote_code=True,
            )
        elif type_model == "TimeMoE-200M_ZERO_SHOT":
            model = AutoModelForCausalLM.from_pretrained(
                'Maple728/TimeMoE-200M',
                device_map="cpu",  
                trust_remote_code=True,
            )

        # INFO: FINE-TUNING
        elif type_model == "TimeMoE-50M-FINE-TUNING":
            model = TimeMoeForPrediction.from_pretrained(
                "../Time-MoE/fine_tuning_50_10_epochs_normalizados/time_moe",
                device_map="cpu",  # Use "cpu" for CPU inference, and "cuda" for GPU inference.
                trust_remote_code=True,
            )
        elif type_model == "TimeMoE-200M-FINE-TUNING":
            model = TimeMoeForPrediction.from_pretrained(
                "../Time-MoE/fine_tuning_200_10_epochs_normalizados/time_moe",
                device_map="cpu",  # Use "cpu" for CPU inference, and "cuda" for GPU inference.
                trust_remote_code=True,
            )
        
        # INFO: FINE-TUNING DIRECT
        elif type_model == "TimeMoE-50M-FINE-TUNING-DIRECT-1":
            model = TimeMoeForPrediction.from_pretrained(
                "../Time-MoE/fine_tuning_50_1_epochs_normalizados/time_moe",
                device_map="cpu",  # Use "cpu" for CPU inference, and "cuda" for GPU inference.
                trust_remote_code=True,
            )
        elif type_model == "TimeMoE-50M-FINE-TUNING-DIRECT-5":
            model = TimeMoeForPrediction.from_pretrained(
                "../Time-MoE/fine_tuning_50_5_rolling_window_direct/time_moe",
                device_map="cpu",  # Use "cpu" for CPU inference, and "cuda" for GPU inference.
                trust_remote_code=True,
            )
        elif type_model == "TimeMoE-50M-FINE-TUNING-DIRECT-10":
            model = TimeMoeForPrediction.from_pretrained(
                "../Time-MoE/fine_tuning_50_10_rolling_window_direct/time_moe",
                device_map="cpu",  # Use "cpu" for CPU inference, and "cuda" for GPU inference.
                trust_remote_code=True,
            )
        elif type_model == "TimeMoE-200M-FINE-TUNING-DIRECT-1":
            model = TimeMoeForPrediction.from_pretrained(
                "../Time-MoE/fine_tuning_200_1_rolling_window_direct/time_moe",
                device_map="cpu",  # Use "cpu" for CPU inference, and "cuda" for GPU inference.
                trust_remote_code=True,
            )
        elif type_model == "TimeMoE-200M-FINE-TUNING-DIRECT-5":
            model = TimeMoeForPrediction.from_pretrained(
                "../Time-MoE/fine_tuning_200_5_rolling_window_direct/time_moe",
                device_map="cpu",  # Use "cpu" for CPU inference, and "cuda" for GPU inference.
                trust_remote_code=True,
            )
        elif type_model == "TimeMoE-200M-FINE-TUNING-DIRECT-10":
            model = TimeMoeForPrediction.from_pretrained(
                "../Time-MoE/fine_tuning_200_10_rolling_window_direct/time_moe",
                device_map="cpu",  # Use "cpu" for CPU inference, and "cuda" for GPU inference.
                trust_remote_code=True,
            )

        # INFO: TESTE
        elif type_model == "TESTE":
            model = TimeMoeForPrediction.from_pretrained(
                "../Time-MoE/teste_time_moe_fine_tuning/time_moe",
                device_map="cpu",  # Use "cpu" for CPU inference, and "cuda" for GPU inference.
                trust_remote_code=True,
            )

        # INFO: Maple728/TimeMoE-1.1B
        else:
            model = AutoModelForCausalLM.from_pretrained(
                'Maple728/TimeMoE-1.1B',
                device_map="cpu",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
                trust_remote_code=True,
            )

        # forecast
        prediction_length = 12
        output = model.generate(tensor, max_new_tokens=prediction_length)  
        forecast = output[:, -prediction_length:]  
    
        gc.collect() 

        y_pred = scaler.inverse_transform(forecast)
        y_pred = y_pred.flatten()

        y_test = df[-forecast_steps:].values

        # Calculating evaluation metrics
        y_baseline = df[-forecast_steps*2:-forecast_steps].values
        rmse_result_time_moe = rmse(y_test, y_pred)
        mape_result_time_moe = mape(y_test, y_pred)
        pbe_result_time_moe = pbe(y_test, y_pred)
        pocid_result_time_moe = pocid(y_test, y_pred)
        mase_result_time_moe = np.mean(np.abs(y_test - y_pred)) / np.mean(np.abs(y_test - y_baseline))

        print("\nResultados Time-MOE: \n")
        print(f'RMSE: {rmse_result_time_moe}')
        print(f'MAPE: {mape_result_time_moe}')
        print(f'PBE: {pbe_result_time_moe}')
        print(f'POCID: {pocid_result_time_moe}')
        print(f'MASE: {mase_result_time_moe}')
      
        y_preds_5_years.append(y_pred)
    return np.concatenate(y_preds_5_years).tolist()
                    
def run_time_moe_5_years(state, product, forecast_steps, time_steps, data_filtered, bool_save, log_lock, type_model='TimeMoE-50M'):

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    tf.keras.utils.set_random_seed(42)

    # Record the start time of execution
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nExecution started at: {start_timestamp}")
    start_time = time.time()

    try:
        # Run time_moe model training
        y_pred = \
        create_time_moe_model(forecast_steps=forecast_steps, time_steps=time_steps, data=data_filtered, state=state, product=product, type_model=type_model, show_plot=True)

        # Prepare results into a DataFrame
        results_df = pd.DataFrame([{'FORECAST_STEPS': forecast_steps,
                                    'TIME_FORECAST': time_steps,
                                    'TYPE_PREDICTIONS': type_model + "_5_YEARS",
                                    'STATE': state,
                                    'PRODUCT': product,
                                    'RMSE': np.nan,
                                    'MAPE': np.nan,
                                    'PBE': np.nan,
                                    'POCID': np.nan,
                                    'MASE': np.nan,
                                    'PREDICTIONS': y_pred,
                                    'ERROR': np.nan}])
    except Exception as e:
        # Handle any exceptions during model training
        print(f"An error occurred for product '{product}' in state '{state}': {e}")
        
        results_df = pd.DataFrame([{'FORECAST_STEPS': np.nan,
                                    'TIME_FORECAST': np.nan,
                                    'TYPE_PREDICTIONS': type_model + "_5_YEARS",
                                    'STATE': state,
                                    'PRODUCT': product,
                                    'RMSE': np.nan,
                                    'MAPE': np.nan,
                                    'PBE': np.nan,
                                    'POCID': np.nan,
                                    'MASE': np.nan,
                                    'PREDICTIONS': np.nan,
                                    'ERROR': f"An error occurred for product '{product}' in state '{state}': {e}"}])
            
    # Save results to an Excel file if specified
    if bool_save:
        directory = f'results'
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, 'results_time_moe_5_years.xlsx')
        if os.path.exists(file_path):
            existing_df = pd.read_excel(file_path)
        else:
            existing_df = pd.DataFrame()

        combined_df = pd.concat([existing_df, results_df], ignore_index=True)
        combined_df.to_excel(file_path, index=False)

    # Calculate and print the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Function execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

def run_all_in_thread_5_years(forecast_steps, time_steps, bool_save, type_model='TimeMoE-50M'):
    multiprocessing.set_start_method("spawn")

    # Create a lock object
    log_lock = multiprocessing.Lock()

    # Load the combined dataset
    all_data = pd.read_csv('../database/combined_data.csv', sep=";")

    # Initialize a dictionary to store products for each state
    state_product_dict = {}

    # Iterate over unique states
    for state in all_data['state'].unique():
        # Filter products corresponding to this state
        products = all_data[all_data['state'] == state]['product'].unique()
        # Add to the dictionary
        state_product_dict[state] = list(products)

    # Loop through each state and its products
    for state, products in state_product_dict.items():
        for product in products:
            print(f"========== State: {state}, product: {product} ==========")
           
            # Set random seeds for reproducibility
            random.seed(42)
            np.random.seed(42)
            tf.random.set_seed(42)
            os.environ['PYTHONHASHSEED'] = str(42)
            tf.keras.utils.set_random_seed(42)

            # Filter data for the current state and product
            data_filtered = all_data[(all_data['state'] == state) & (all_data['product'] == product)]
            
            thread = multiprocessing.Process(target=run_time_moe_5_years, args=(state, product, forecast_steps, time_steps, data_filtered, bool_save, log_lock, type_model))
            thread.start()
            thread.join()  # Wait for the thread to finish execution

def product_and_single_thread_testing_5_years():    
    """
    Perform a simple training thread using time_moe model for time series forecasting.

    This function initializes random seeds, loads a database, executes an time_moe model,
    evaluates its performance, and prints results.

    Parameters:
    None

    Returns:
    None
    """
    # Setting random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    tf.keras.utils.set_random_seed(42)

    state = "sp"
    product = "gasolinac"
    
    # Loading and preparing data
    data_filtered_test = pd.read_csv(f"../database/venda_process/mensal/uf/{product}/mensal_{state}_{product}.csv", sep=";",  parse_dates=['timestamp'], date_parser=convert_date)

    print(f" ========== Starting univariate test for the state of {state} - {product} ==========")

    # Recording start time of execution
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nExecution started at: {start_timestamp}")
    start_time = time.time()

    # Running the time_moe model
    y_pred = \
    create_time_moe_model(forecast_steps=12, time_steps=12, data=data_filtered_test, state=state, product=product,
                    type_model='TimeMoE-50M_ZERO_SHOT', show_plot=True)

    # Recording end time and calculating execution duration
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nFunction execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")