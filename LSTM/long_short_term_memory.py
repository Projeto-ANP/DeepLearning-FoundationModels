import tensorflow as tf
import keras
import keras.backend as K # type: ignore
import keras_tuner as kt
from keras_tuner import Objective, BayesianOptimization

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_percentage_error as mape

import numpy as np
import pandas as pd

import os
import random
import pickle
import time

import gc
import multiprocessing

from matplotlib import pyplot as plt

import warnings
from warnings import simplefilter

from model_builder import ModelBuilder
from metrics_lstm import rmse, pbe, pocid, mase
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler

from functions_forecasting import recursive_multistep_forecasting

import pycatch22

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

def create_lstm_model(forecast_steps, time_steps, data, epochs, state, product, batch_size, diff=False, type_predictions='recursive', type_lstm='LSTM', show_plot=None, verbose=2):
    """
    Runs an LSTM model for time series forecasting.

    Parameters:
    - forecast_steps (int): Number of steps ahead for forecasting.
    - time_steps (int): Length of the time sequences for generating the attribute-value table.
    - data (pd.DataFrame): DataFrame containing the time series data.
    - epochs (int): Number of epochs for training the LSTM model.
    - state (str): The specific state for which the model is trained.
    - product (str): The specific product for which the model is trained.
    - batch_size: Batch size for training.
    - diff: 
    - type_predictions (str, optional): Type of prediction method. Can be 'recursive' or 'direct'. Default is 'recursive'.
    - type_lstm (str, optional): Type of LSTM to be used. Default is 'LSTM'.
    - show_plot (bool or None, optional): Whether to display a plot of the forecasted values. Default is None.
    - verbose (int, optional): Controls the verbosity of the training process. 0 = silent, 1 = progress bar, 2 = one line per epoch. Default is 2.

    """

    df = data['m3']

    diff = False

    if type_predictions == 'recursive':
        int_dense = 1

        if diff: 
            train, y_test = train_test_stats(df, time_steps)
            last_value_train = train[-1:] 
            train_diff = train.diff().dropna()

            data = rolling_window(pd.concat([train_diff,pd.Series([0,0,0,0,0,0,0,0,0,0,0,0], index=y_test.index)]), time_steps, type_predictions)

            X_train, X_test, y_train, _ = train_test_split_cisia(data, forecast_steps, type_predictions)
        else: 
            data = rolling_window(df, time_steps, type_predictions)

            X_train, X_test, y_train, _ = train_test_split_cisia(data, forecast_steps, type_predictions)
    
        y_train = y_train.values.reshape(-1, 1)
        y_train = np.hstack([y_train, np.zeros((y_train.shape[0], time_steps-1))])

        scaler = MinMaxScaler(feature_range=(-1, 1))

        X_train = scaler.fit_transform(X_train)
        y_train = scaler.transform(y_train)[:, 0].reshape(-1, 1) 
        X_test = scaler.transform(X_test)
        
        # Formato: [instancias, time steps, features]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
    elif type_predictions == 'direct':
        int_dense = 12

        if diff: 
            train, y_test = train_test_stats(df, time_steps)
            last_value_train = train[-1:] 
            train_diff = train.diff().dropna()

            data = rolling_window(pd.concat([train_diff,pd.Series([0,0,0,0,0,0,0,0,0,0,0,0], index=y_test.index)]), time_steps, type_predictions)

            X_train, X_test, y_train, _ = train_test_split_cisia(data, forecast_steps, type_predictions)

        else: 
            data = df.values.reshape(-1, 1)

            X, y = create_dataset_direct(data, time_steps, forecast_steps)

            # Treino e teste (ultima instancia para teste)
            X_train, X_test = X[:-24], X[-1:]
            y_train, y_test = y[:-24], y[-1:]

        scaler = MinMaxScaler(feature_range=(-1, 1))

        X_train = scaler.fit_transform(X_train)
        y_train = scaler.transform(y_train)
        X_test = scaler.transform(X_test)
        
        # Formato: [instancias, time steps, features]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # print("\nX_train\n", X_train)
    # print("\nX_train.shape\n", X_train.shape)
    # print("\ny_train\n", y_train)
    # print("\ny_train\n", y_train.shape)
    # print("\nX_test\n", X_test)
    # print("\nX_test\n", X_test.shape)
    # print("\ny_test\n", y_test)
    # print("\ny_test\n", y_test.shape)

    X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42, shuffle=False)

    tuner = BayesianOptimization(
        hypermodel=ModelBuilder(time_steps=time_steps, dense_predictions=int_dense, type_lstm=type_lstm),
        objective=Objective('mean_absolute_percentage_error', direction='min'),
        num_initial_points=5,
        max_trials=1,
        alpha=0.0001,
        beta=2.6,
        seed=42,
        max_retries_per_trial=1,
        max_consecutive_failed_trials=3,
        directory=f'{type_lstm}_{type_predictions}_{time_steps}',
        overwrite=True,
        project_name=f'lstm_{state}_{product}'
    )

    tuner.search(X_train_val, y_train_val, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                verbose=verbose)

    # Get the best model
    best_model = tuner.get_best_models()[0]
    best_hyperparameters = tuner.get_best_hyperparameters()[0].values

    best_model.summary()

    best_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
   
    gc.collect() 
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    del tuner
    
    if type_predictions == "recursive":
        forecast = recursive_multistep_forecasting(X_test, best_model, forecast_steps)

        # Converter o forecast para um array 2D
        forecast = np.array(forecast).reshape(-1, 1)

        # Adicionar 11 colunas de zeros para corresponder ao scaler
        forecast_padded = np.hstack([forecast, np.zeros((forecast.shape[0], time_steps-1))])

        # Reverter a normalização
        y_pred = scaler.inverse_transform(forecast_padded)[:, 0]

        y_test = df[-forecast_steps:].values

    elif type_predictions == "direct":
        # Predicting
        y_pred = best_model.predict(X_test)

        # Retornando para a escala original
        y_pred = scaler.inverse_transform(y_pred)
        y_pred = y_pred.flatten()

        y_test = df[-forecast_steps:].values
    
    if diff:
        y_pred = reverse_diff(last_value_train, y_pred)

    # Calculating evaluation metrics
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
    print(f'Melhores Parametros: {best_hyperparameters}')

    # ========== Resultados Comparação ==========

    df = pd.read_excel('../00-RANKING.xlsx', sheet_name='ALL')

    filtered_df = df[(df['PRODUCT'] == product) & (df['UF'] == state)]
     
    filtered_df = filtered_df.fillna(0)

    columns = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12']

    # Obter os valores das colunas P1 a P12 como uma lista
    pontos_comp = filtered_df[columns].values.flatten().tolist()
    
    print("\nResultados COMPARAÇÃO: \n")
    rmse_result = rmse(y_test, pontos_comp)
    mape_result = mape(y_test, pontos_comp)
    pbe_result = pbe(y_test, pontos_comp)
    pocid_result = pocid(y_test, pontos_comp)
    mase_result = np.mean(np.abs(y_test - pontos_comp)) / np.mean(np.abs(y_test - y_baseline))

    print(f'RMSE: {rmse_result}')
    print(f'MAPE: {mape_result}')
    print(f'PBE: {pbe_result}')
    print(f'POCID: {pocid_result}')
    print(f'MASE: {mase_result}')
    
    sub_dir = None

    if show_plot:
        plots_dir = f"PLOTS_\\{type_predictions}_{type_lstm}"
        sub_dir = os.path.join(plots_dir, f"plot_{state}_{product}_{time_steps}_{type_predictions}")
    
        os.makedirs(sub_dir, exist_ok=True)

        # Predictions
        plt.figure(figsize=(12, 3))
        plt.title('Normalized Predictions')
        plt.plot(range(len(y_test.T)), y_test.T, label='REAL')
        plt.plot(range(len(y_test.T)), y_pred, label=type_lstm)
        plt.plot(range(len(y_test.T)), pontos_comp, label='COMPARACAO')
        plt.legend()
        plt.savefig(os.path.join(sub_dir, f'normalized_predictions_{type_predictions}.png'))
        plt.close()
        
    return rmse_result_lstm, mape_result_lstm, pbe_result_lstm, pocid_result_lstm, mase_result_lstm, best_hyperparameters, y_pred, batch_size
                    
def run_lstm(state, product, forecast_steps, time_steps, data_filtered, epochs, verbose, bool_save, log_lock, batch_size, type_predictions='recursive', type_lstm='LSTM'):
    """
    Runs LSTM model training and saves results to an Excel file.

    Parameters:
        - state (str): The state for which the LSTM model is being trained.
        - product (str): The product for which the LSTM model is being trained.
        - forecast_steps (int): Number of steps ahead for forecasting.
        - time_steps (int): Length of the time sequences for generating the attribute-value table.
        - data_filtered (pd.DataFrame): Filtered data specific to the state and product.
        - epochs (int): Number of epochs for training the LSTM model.
        - verbose (int): Controls the verbosity of the training process. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        - bool_save (bool): Flag indicating whether to save the results to an Excel file.
        - log_lock: A lock to ensure thread-safe logging.
        - batch_size: Batch size for training.
        - type_predictions (str, optional): Type of prediction method. Can be 'recursive' or 'direct'. Default is 'recursive'.
        - type_lstm (str, optional): Type of LSTM to be used. Default is 'LSTM'.

    Returns:
        None.
    """

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
        # Run LSTM model training
        rmse_result, mape_result, pbe_result, pocid_result, mase_result, best_hyperparameters, y_pred, batch_size = \
        create_lstm_model(forecast_steps=forecast_steps, time_steps=time_steps, epochs=epochs, data=data_filtered, state=state, product=product, batch_size=batch_size, type_predictions=type_predictions, type_lstm=type_lstm, show_plot=True, verbose=verbose)

        # Prepare results into a DataFrame
        results_df = pd.DataFrame([{'FORECAST_STEPS': forecast_steps,
                                    'TIME_FORECAST': time_steps,
                                    'TYPE_PREDICTIONS': "LSTM_" + type_predictions,
                                    'BEST_PARAM': str(best_hyperparameters),
                                    'STATE': state,
                                    'PRODUCT': product,
                                    'RMSE': rmse_result,
                                    'MAPE': mape_result,
                                    'PBE': pbe_result,
                                    'POCID': pocid_result,
                                    'MASE': mase_result,
                                    'PREDICTIONS': y_pred,
                                    'BATCH_SIZE': batch_size,
                                    'ERROR': np.nan}])
    except Exception as e:
        # Handle any exceptions during model training
        print(f"An error occurred for product '{product}' in state '{state}': {e}")
        
        results_df = pd.DataFrame([{'FORECAST_STEPS': np.nan,
                                    'TIME_FORECAST': np.nan,
                                    'TYPE_PREDICTIONS': "LSTM_" + type_predictions,
                                    'BEST_PARAM': np.nan,
                                    'STATE': state,
                                    'PRODUCT': product,
                                    'RMSE': np.nan,
                                    'MAPE': np.nan,
                                    'PBE': np.nan,
                                    'POCID': np.nan,
                                    'MASE': np.nan,
                                    'PREDICTIONS': np.nan,
                                    'BATCH_SIZE': batch_size,
                                    'ERROR': f"An error occurred for product '{product}' in state '{state}': {e}"}])
            
    # Save results to an Excel file if specified
    if bool_save:
        directory = f'result_{time_steps}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, 'lstm_results_simples.xlsx')
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

    # Log the details to a file
    log_entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')}, {state}, {product}, {forecast_steps}, {time_steps}, {type_predictions}, {execution_time:.2f}\n"
    
    # Acquire the lock before writing to the log file
    with log_lock:
        with open("training_log.csv", "a") as log_file:
            log_file.write(log_entry)

def run_lstm_in_thread(forecast_steps, time_steps, epochs, verbose, bool_save, batch_size, type_predictions='recursive', type_lstm='LSTM'):
    """
    Loop through LSTM model with different configurations for each state and product combination.

    Parameters:
        - forecast_steps (int): Prediction forecast_steps.
        - time_steps (int): Length of the time_steps for attribute-value table generation.
        - epochs (int): Number of epochs for training the LSTM model.
        - verbose (int): Controls the verbosity of the training process. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        - bool_save (bool): Flag indicating whether to save the trained models.
        - save_model (bool or None, optional): Save models.
        - type_predictions (str, optional): Type of predictions method. Can be 'recursive' or 'direct_dense12'. Default is 'recursive'.

    Returns:
        None
    """
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
            
            # Create a separate process (thread) to run LSTM model
            thread = multiprocessing.Process(target=run_lstm, args=(state, product, forecast_steps, time_steps, data_filtered, epochs, verbose, bool_save, log_lock, batch_size, type_predictions, type_lstm))
            thread.start()
            thread.join()  # Wait for the thread to finish execution

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

    # Running the LSTM model
    rmse_result, mape_result, pbe_result, pocid_result, mase_result, best_hyperparameters, y_pred, batch_size = \
    create_lstm_model(forecast_steps=12, time_steps=12, data=data_filtered_test, epochs=100, state=state, product=product,
                      batch_size=16, type_predictions='recursive', type_lstm='LSTM', show_plot=True, verbose=1)

    # Recording end time and calculating execution duration
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nFunction execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

product_and_single_thread_testing()