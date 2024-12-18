import torch
from transformers import AutoModelForCausalLM
from time_moe.models.modeling_time_moe import TimeMoeForPrediction
from itertools import product

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

def create_time_moe_model(
    forecast_steps, 
    time_steps, 
    data, 
    state, 
    derivative, 
    type_model='TimeMoE-50M', 
    show_plot=None,
    model='default_model', 
    norm='none', 
    learning_rate=1e-3, 
    lr_scheduler_type='constant',
    sufix='raw_data'
):
    """
    Runs a TimeMoE model for time series forecasting.

    Parameters:
    - forecast_steps (int): Number of steps ahead for forecasting.
    - time_steps (int): Length of the time sequences for generating the attribute-value table.
    - data (pd.DataFrame): DataFrame containing the time series data.
    - state (str): The specific state for which the model is trained.
    - derivative (str): The specific derivative for which the model is trained.
    - type_model (str, optional): Type of TimeMoE to be used. Default is 'TimeMoE-50M'.
    - show_plot (bool or None, optional): Whether to display a plot of the forecasted values. Default is None.
    - model (str): The model architecture to use. Default is 'default_model'.
    - norm (str): Normalization method. Options are ['none', 'zero'].
    - dataset (str): Dataset to use. Options are ['dataset_petroleum_derivatives_normalizados', 'dataset_petroleum_derivatives'].
    - learning_rate (float): Learning rate for the optimizer.
    - lr_scheduler_type (str): Learning rate scheduler type. Options are ['constant', 'linear', 'cosine'].

    Returns:
    - rmse_result (float): Root Mean Square Error of the model's predictions.
    - mape_result (float): Mean Absolute Percentage Error of the model's predictions.
    - pbe_result (float): Percentage Bias Error of the model's predictions.
    - pocid_result (float): Percentage of Correct Increase or Decrease.
    - mase_result (float): Mean Absolute Scaled Error.
    - y_pred (np.ndarray): Array of predicted values.
    """

    df = data['m3']

    len_data = None
    df_scaled, scaler = get_scaled_data(df, len_data)

    tensor = torch.tensor(df_scaled, dtype=torch.float32)

    tensor = tensor.squeeze(-1).unsqueeze(0)

    ''' 
    # INFO: ================== ZERO SHOT ==================
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

    # INFO: ================== FINE-TUNING RAW DATA ==================
    elif type_model == "TimeMoE-50M-FINE-TUNING":
        model_path = f"models_fine_tuning/model_50M/{sufix}/{norm}/fine_tuning_50_10_epochs_{sufix}_{learning_rate}_{lr_scheduler_type}/time_moe"
        model = TimeMoeForPrediction.from_pretrained(
            model_path,
            device_map="cpu",  # Use "cpu" for CPU inference, and "cuda" for GPU inference.
            trust_remote_code=True,
        )
    elif type_model == "TimeMoE-200M-FINE-TUNING":
        model_path = f"models_fine_tuning/model_200M/{sufix}/{norm}/fine_tuning_200_10_epochs_{sufix}_{learning_rate}_{lr_scheduler_type}/time_moe"
        model = TimeMoeForPrediction.from_pretrained(
            model_path,
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
        
    return rmse_result_time_moe, mape_result_time_moe, pbe_result_time_moe, pocid_result_time_moe, mase_result_time_moe, y_pred, 
                    
def run_time_moe(state, derivative, forecast_steps, time_steps, data_filtered, bool_save, log_lock, type_model='TimeMoE-50M'):

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
        if type_model in ['TimeMoE-50M-FINE-TUNING', 'TimeMoE-200M-FINE-TUNING']:

            if type_model == "TimeMoE-50M-FINE-TUNING":
                models = [50]
            else:
                models = [200]

            norms = ['none', 'zero']
            datasets = ['dataset_petroleum_derivatives_normalizados', 'dataset_petroleum_derivatives']
            learning_rates = ['0.001', '0.0001', '5e-05', '2e-05', '1e-06']
            lr_scheduler_types = ['constant', 'linear', 'cosine']

            parameter_combinations = list(product(models, norms, datasets, learning_rates, lr_scheduler_types))

            # Initialize or load existing results DataFrame
            file_path = os.path.join('results', 'results_time_moe_fine_tuning_new2.xlsx')
            if os.path.exists(file_path):
                combined_df = pd.read_excel(file_path)
            else:
                combined_df = pd.DataFrame()

            for model, norm, dataset, learning_rate, lr_scheduler_type in parameter_combinations:
                try:
                    print(f"\n=== Forecasting model: {model} ===")
                    print(f"Type_model: {type_model}")
                    print(f"Dataset: {dataset}")
                    print(f"Normalization: {norm}")
                    print(f"Learning Rate: {learning_rate}")
                    print(f"LR Scheduler: {lr_scheduler_type}")

                    sufix = 'normalizados' if dataset == 'dataset_petroleum_derivatives_normalizados' else 'raw_data'

                    rmse_result, mape_result, pbe_result, pocid_result, mase_result, y_pred = create_time_moe_model(
                        forecast_steps=forecast_steps,
                        time_steps=time_steps,
                        data=data_filtered,
                        state=state,
                        derivative=derivative,
                        type_model=type_model,
                        show_plot=False,
                        model=model,
                        norm=norm,
                        learning_rate=learning_rate,
                        lr_scheduler_type=lr_scheduler_type,
                        sufix=sufix
                    )

                    # Add results to DataFrame
                    results_df = pd.DataFrame([{
                        'FORECAST_STEPS': forecast_steps,
                        'TIME_FORECAST': time_steps,
                        'TYPE_PREDICTIONS': type_model + f"_{sufix}_{norm}_{learning_rate}_{lr_scheduler_type}",
                        'STATE': state,
                        'PRODUCT': derivative,
                        'RMSE': rmse_result,
                        'MAPE': mape_result,
                        'PBE': pbe_result,
                        'POCID': pocid_result,
                        'MASE': mase_result,
                        'PREDICTIONS': y_pred,
                        'ERROR': np.nan
                    }])

                except Exception as e:
                    # Handle any exceptions during model training
                    print(f"An error occurred for derivative '{derivative}' in state '{state}': {e}")

                    results_df = pd.DataFrame([{
                        'FORECAST_STEPS': np.nan,
                        'TIME_FORECAST': np.nan,
                        'TYPE_PREDICTIONS': type_model + f"_{sufix}_{norm}_{learning_rate}_{lr_scheduler_type}",
                        'STATE': state,
                        'PRODUCT': derivative,
                        'RMSE': np.nan,
                        'MAPE': np.nan,
                        'PBE': np.nan,
                        'POCID': np.nan,
                        'MASE': np.nan,
                        'PREDICTIONS': np.nan,
                        'ERROR': f"An error occurred for derivative '{derivative}' in state '{state}': {e}"
                    }])

                # Append current results to the combined DataFrame
                combined_df = pd.concat([combined_df, results_df], ignore_index=True)

        else:
            # Run single TimeMoE model training
            rmse_result, mape_result, pbe_result, pocid_result, mase_result, y_pred = create_time_moe_model(
                forecast_steps=forecast_steps,
                time_steps=time_steps,
                data=data_filtered,
                state=state,
                derivative=derivative,
                type_model=type_model,
                show_plot=True
            )

            results_df = pd.DataFrame([{
                'FORECAST_STEPS': forecast_steps,
                'TIME_FORECAST': time_steps,
                'TYPE_PREDICTIONS': type_model,
                'STATE': state,
                'PRODUCT': derivative,
                'RMSE': rmse_result,
                'MAPE': mape_result,
                'PBE': pbe_result,
                'POCID': pocid_result,
                'MASE': mase_result,
                'PREDICTIONS': y_pred,
                'ERROR': np.nan
            }])

            combined_df = pd.concat([combined_df, results_df], ignore_index=True)

    except Exception as e:
        # Handle exceptions outside of iterations
        print(f"An error occurred for derivative '{derivative}' in state '{state}': {e}")

        results_df = pd.DataFrame([{
            'FORECAST_STEPS': np.nan,
            'TIME_FORECAST': np.nan,
            'TYPE_PREDICTIONS': type_model,
            'STATE': state,
            'PRODUCT': derivative,
            'RMSE': np.nan,
            'MAPE': np.nan,
            'PBE': np.nan,
            'POCID': np.nan,
            'MASE': np.nan,
            'PREDICTIONS': np.nan,
            'ERROR': f"An error occurred for derivative '{derivative}' in state '{state}': {e}"
        }])

        combined_df = pd.concat([combined_df, results_df], ignore_index=True)

    # Save results to Excel file
    if bool_save:
        directory = 'results'
        if not os.path.exists(directory):
            os.makedirs(directory)

        combined_df.to_excel(file_path, index=False)


    # Calculate and print the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Function execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

def run_all_in_thread(forecast_steps, time_steps, bool_save, type_model='TimeMoE-50M'):
    multiprocessing.set_start_method("spawn")

    # Create a lock object
    log_lock = multiprocessing.Lock()

    # Load the combined dataset
    all_data = pd.read_csv('../database/combined_data.csv', sep=";")

    # Initialize a dictionary to store derivatives for each state
    state_derivative_dict = {}

    # Iterate over unique states
    for state in all_data['state'].unique():
        # Filter derivatives corresponding to this state
        derivatives = all_data[all_data['state'] == state]['product'].unique()
        # Add to the dictionary
        state_derivative_dict[state] = list(derivatives)

    # Loop through each state and its derivatives
    for state, derivatives in state_derivative_dict.items():
        for derivative in derivatives:
            print(f"========== State: {state}, derivative: {derivative} ==========")
           
            # Set random seeds for reproducibility
            random.seed(42)
            np.random.seed(42)
            tf.random.set_seed(42)
            os.environ['PYTHONHASHSEED'] = str(42)
            tf.keras.utils.set_random_seed(42)

            # Filter data for the current state and derivative
            data_filtered = all_data[(all_data['state'] == state) & (all_data['product'] == derivative)]
            
            thread = multiprocessing.Process(target=run_time_moe, args=(state, derivative, forecast_steps, time_steps, data_filtered, bool_save, log_lock, type_model))
            thread.start()
            thread.join()  # Wait for the thread to finish execution

def derivative_and_single_thread_testing():    
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
    derivative = "gasolinac"
    
    # Loading and preparing data
    data_filtered_test = pd.read_csv(f"../database/venda_process/mensal/uf/{derivative}/mensal_{state}_{derivative}.csv", sep=";",  parse_dates=['timestamp'], date_parser=convert_date)

    print(f" ========== Starting univariate test for the state of {state} - {derivative} ==========")

    # Recording start time of execution
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nExecution started at: {start_timestamp}")
    start_time = time.time()

    # Running the time_moe model

    # rmse_result, mape_result, pbe_result, pocid_result, mase_result, y_pred = \
    # create_time_moe_model(forecast_steps=12, time_steps=12, data=data_filtered_test, state=state, derivative=derivative,
    #                 type_model='TimeMoE-200M-FINE-TUNING', show_plot=True)

    rmse_result, mape_result, pbe_result, pocid_result, mase_result, y_pred = create_time_moe_model(
                forecast_steps= 12, 
                time_steps= 12, 
                data= data_filtered_test,  
                state= state, 
                derivative= derivative, 
                type_model="TimeMoE-200M-FINE-TUNING", 
                show_plot=True,
                model= "200", 
                norm= "none", 
                learning_rate="1e-06", 
                lr_scheduler_type="linear",
                sufix= "raw_data"
     )

    # Recording end time and calculating execution duration
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nFunction execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")