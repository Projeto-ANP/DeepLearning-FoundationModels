import torch
import tensorflow as tf
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from einops import rearrange

import timesfm


import numpy as np
import pandas as pd

import os
import random
import time

from matplotlib import pyplot as plt

import warnings
from warnings import simplefilter

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

def create_times_fm(
    data, 
    type_model='200M', 
):
    """
    Runs a TimeMoE model for time series forecasting.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the time series data.
    - type_model (str): choose from {200M or 500M}

    """

    y_preds_5_years = []

    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

    end_date = data['timestamp'].max()

    for years in range(5, 0, -1):
        start_date = end_date - pd.DateOffset(years=years-1)

        df = data[data['timestamp'] <= start_date]
        print(f'\nData filtered for {start_date.date()}\n')

        df['unique_id'] = 1
        df.rename(columns={'timestamp': 'ds'}, inplace=True)
        df = df[['unique_id', 'ds', 'm3']]  

        train_data = df.iloc[:-12]  
        test_data = df.iloc[-12:]  
        
        if type_model == '200M':
            model_tfm = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="gpu",
                    per_core_batch_size=16,
                    horizon_len=12,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
            )
        
        else:
            model_tfm = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="gpu",
                    per_core_batch_size=16,
                    horizon_len=12,
                    num_layers=50,
                    use_positional_embedding=False,
                    context_len=2048,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
            )

        # forecast
        forecast_df = model_tfm.forecast_on_df(
                inputs=train_data,
                freq="M",  # monthly
                value_name="m3",
                num_jobs=-1,
            )

        forecast_list = forecast_df["timesfm"].tolist()

        formatted_list = [round(value, 3) for value in forecast_list]

        # Display results
        print("\nMean forecast for the last 12 months:")
        print(formatted_list)

        print("\nActual values for the last 12 months:")
        print(test_data["m3"].tolist())

        # Return predictions as a list
        y_preds_5_years.append(formatted_list)
        
    return np.concatenate(y_preds_5_years).tolist()
            
def run_morai_moe(state, derivative, data_filtered, type_model='200M'):

    # Record the start time of execution
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nExecution started at: {start_timestamp}")
    start_time = time.time()

    try:
        y_pred = \
        create_times_fm(data=data_filtered, type_model=type_model)
        
        # Create a DataFrame to store the results
        results_df = pd.DataFrame([{'TYPE_PREDICTIONS': 'Times-FM_' + type_model,
                                    'STATE': state,
                                    'PRODUCT': derivative,
                                    'PREDICTIONS': y_pred,
                                    'ERROR': np.nan}])
    except Exception as e:
        # Handle exceptions during model training
        print(f"An error occurred for product '{derivative}' in state '{state}': {e}")
        
        results_df = pd.DataFrame([{'TYPE_PREDICTIONS': 'Times-FM_' + type_model,
                                    'STATE': state,
                                    'PRODUCT': derivative,
                                    'PREDICTIONS': np.nan,
                                    'ERROR': f"An error occurred for product '{derivative}' in state '{state}': {e}"}])
            
    # Save the results to an Excel file if requested
    directory = f'results_model_local'
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, 'times_fm_results.xlsx')
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

def run_all_in_thread(type_model='200M'):

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
            
            run_morai_moe(state, derivative, data_filtered, type_model)

def product_and_single_thread_testing():    
    state = "sp"
    derivative = "gasolinac"
    
    # Loading and preparing data
    data_filtered_test = pd.read_csv(f"../database/venda_process/mensal/uf/{derivative}/mensal_{state}_{derivative}.csv", sep=";",  parse_dates=['timestamp'], date_parser=convert_date)

    print(f" ========== Starting univariate test for the state of {state} - {derivative} ==========")

    # Recording start time of execution
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nExecution started at: {start_timestamp}")
    start_time = time.time()
    
    y_pred = \
    create_times_fm(data=data_filtered_test,
                      type_model='500M') # 200M or 500M

    # Recording end time and calculating execution duration
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nFunction execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")