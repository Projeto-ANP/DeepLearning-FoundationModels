import torch
from itertools import product
import tensorflow as tf
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from einops import rearrange


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

def create_morai_model(
    data, 
    type_model='small', 
):
    """
    Runs a TimeMoE model for time series forecasting.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the time series data.
    - type_model (str): choose from {'small', 'base'}

    """

    y_preds_5_years = []

    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

    end_date = data['timestamp'].max()

    for years in range(5, 0, -1):
        start_date = end_date - pd.DateOffset(years=years-1)

        df = data[data['timestamp'] <= start_date]
        print(f'\nData filtered for {start_date.date()}\n')
        print(df)

        # Data separation
        train_data = df.iloc[:-12]  # All data except the last 12 months
        test_data = df.iloc[-12:]   # Last 12 months for testing

        # Prepare training data
        train_target = train_data['m3'].to_numpy()

        # Prepare test data
        test_target = test_data['m3'].to_numpy()

        # Model Initialization
        model = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(
                f"Salesforce/moirai-moe-1.0-R-{type_model}",
            ),
            prediction_length=12,
            context_length=len(train_target),
            patch_size=16,
            num_samples=1,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

        # Prepare data for the model
        past_target = rearrange(torch.as_tensor(train_target, dtype=torch.float32), "t -> 1 t 1")
        past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
        past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)

        # forecast
        forecast = model(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
        )
        forecast = forecast.numpy().flatten().tolist()

        formatted_list = [round(value, 3) for value in forecast]
        
        # Display results
        print("\nMean forecast for the last 12 months:")
        print(formatted_list)

        print("\nActual values for the last 12 months:")
        print(test_target)

        # Return predictions as a list
        y_preds_5_years.append(formatted_list)
        
    return np.concatenate(y_preds_5_years).tolist()
            
def run_morai_moe(state, derivative, data_filtered, type_model='small'):

    # Record the start time of execution
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nExecution started at: {start_timestamp}")
    start_time = time.time()

    try:
        y_pred = \
        create_morai_model(data=data_filtered, type_model=type_model)
        
        # Create a DataFrame to store the results
        results_df = pd.DataFrame([{'TYPE_PREDICTIONS': 'Morai-MoE_' + type_model,
                                    'STATE': state,
                                    'PRODUCT': derivative,
                                    'PREDICTIONS': y_pred,
                                    'ERROR': np.nan}])
    except Exception as e:
        # Handle exceptions during model training
        print(f"An error occurred for product '{derivative}' in state '{state}': {e}")
        
        results_df = pd.DataFrame([{'TYPE_PREDICTIONS': 'Morai-MoE_' + type_model,
                                    'STATE': state,
                                    'PRODUCT': derivative,
                                    'PREDICTIONS': np.nan,
                                    'ERROR': f"An error occurred for product '{derivative}' in state '{state}': {e}"}])
            
    # Save the results to an Excel file if requested
    directory = f'results_model_local'
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, 'morai_moe_results.xlsx')
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

def run_all_in_thread(type_model='small'):

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
    create_morai_model(data=data_filtered_test,
                      type_model='small') # small, base

    # Recording end time and calculating execution duration
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nFunction execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")