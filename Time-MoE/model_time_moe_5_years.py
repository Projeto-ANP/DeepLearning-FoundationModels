from itertools import product
import torch
from transformers import AutoModelForCausalLM
from time_moe.models.modeling_time_moe import TimeMoeForPrediction

import tensorflow as tf

from sklearn.metrics import mean_absolute_percentage_error as mape

import numpy as np
import pandas as pd

import os
import random
import time

import gc
import multiprocessing

import warnings
from warnings import simplefilter

from metrics_time_moe import pbe, pocid # type: ignore
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

def get_scaled_data(df):
    df = df[:-12]

    # MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_scaled = scaler.fit_transform(df.values.reshape(-1, 1))

    return df_scaled, scaler

def create_time_moe_model(
    data, 
    state, 
    derivative, 
    epochs=None,
    type_model='TimeMoE-50M_ZERO_SHOT',
    model=None, 
    learning_rate=None, 
):
    """
    Trains and runs a TimeMoE model for time series forecasting.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the time series data.
    - state (str): The specific state for which the model is trained.
    - derivative (str): The specific derivative for which the model is trained.
    - epochs (int, optional): Number of training epochs. Defaults to None.
    - type_model (str, optional): Type of TimeMoE model to use. Default is 'TimeMoE-50M_ZERO_SHOT'.
    - model (int, optional): Int, 50 or 200. Default is None.
    - learning_rate (float, optional): Learning rate for the optimizer. Defaults to None.

    Returns:
    - y_pred: Array of predicted values.
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

        print(f"Ano atual: {(start_date.date()).year}")
   
        df_scaled, scaler = get_scaled_data(df)

        tensor = torch.tensor(df_scaled, dtype=torch.float32)

        tensor = tensor.squeeze(-1).unsqueeze(0)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tensor = tensor.to(device)
        ''' 
        # INFO: ================== ZERO SHOT ==================
        ''' 
        if type_model == "TimeMoE-50M_ZERO_SHOT":
            model = AutoModelForCausalLM.from_pretrained(
                'Maple728/TimeMoE-50M',
                device_map="cuda",  
                trust_remote_code=True,
            )

        elif type_model == "TimeMoE-200M_ZERO_SHOT":
            model = AutoModelForCausalLM.from_pretrained(
                'Maple728/TimeMoE-200M',
                device_map="cuda",  
                trust_remote_code=True,
            )

        # INFO: ================== FINE-TUNING GLOBAL ==================
        elif type_model == "TimeMoE-50M-FINE-TUNING-GLOBAL":
            model_path = f"models_fine_tuning_global_5_years/model_50M/fine_tuning_50M_{(start_date.date()).year}_{epochs}_epochs_{learning_rate}/time_moe"
            model = TimeMoeForPrediction.from_pretrained(
                model_path,
                device_map="cuda", 
                trust_remote_code=True,
            )

        elif type_model == "TimeMoE-200M-FINE-TUNING-GLOBAL":
            model_path = f"models_fine_tuning_global_5_years/model_50M/fine_tuning_50M_{(start_date.date()).year}_{epochs}_epochs_{learning_rate}/time_moe"
            model = TimeMoeForPrediction.from_pretrained(
                model_path,
                device_map="cuda", 
                trust_remote_code=True,
            )
        
        # INFO: ================== FINE-TUNING INDIVIDUAL ==================

        # INFO: ================== FINE-TUNING PRODUCT ==================
        elif type_model == "TimeMoE-50M-FINE-TUNING-PRODUCT":
            model_path = f"models_fine_tuning_product_5_years/model_50M/fine_tuning_50M_{derivative}_{(start_date.date()).year}/time_moe"
            model = TimeMoeForPrediction.from_pretrained(
                model_path,
                device_map="cuda", 
                trust_remote_code=True,
            )

        elif type_model == "TimeMoE-200M-FINE-TUNING-PRODUCT":
            model_path = f"models_fine_tuning_product_5_years/model_200M/fine_tuning_200M_{derivative}_{(start_date.date()).year}/time_moe"
            model = TimeMoeForPrediction.from_pretrained(
                model_path,
                device_map="cuda", 
                trust_remote_code=True,
            )

        # INFO: Maple728/TimeMoE-1.1B
        elif type_model == "TimeMoE-1.1B":
            model = AutoModelForCausalLM.from_pretrained(
                'Maple728/TimeMoE-1.1B',
                device_map="cuda", 
                trust_remote_code=True,
            )
        
        model.to(device)

        # forecast
        prediction_length = 12
        output = model.generate(tensor, max_new_tokens=prediction_length)  
        forecast = output[:, -prediction_length:]  

        forecast = forecast.cpu().numpy() 
    
        gc.collect() 

        y_pred = scaler.inverse_transform(forecast)
        y_pred = y_pred.flatten()

        y_test = df[-12:].values

        # Calculating evaluation metrics
        y_baseline = df[-12*2:-12].values
        mape_result_time_moe = mape(y_test, y_pred)
        pbe_result_time_moe = pbe(y_test, y_pred)
        pocid_result_time_moe = pocid(y_test, y_pred)
        mase_result_time_moe = np.mean(np.abs(y_test - y_pred)) / np.mean(np.abs(y_test - y_baseline))

        print("\nResultados Time-MOE: \n")
        print(f'MAPE: {mape_result_time_moe}')
        print(f'PBE: {pbe_result_time_moe}')
        print(f'POCID: {pocid_result_time_moe}')
        print(f'MASE: {mase_result_time_moe}')
      
        y_preds_5_years.append(y_pred)

    return np.concatenate(y_preds_5_years).tolist()
                    
def run_time_moe_5_years(state, derivative, data_filtered, type_model='TimeMoE-50M_ZERO_SHOT'):

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

    if type_model in ['TimeMoE-50M-FINE-TUNING-INDIV', 'TimeMoE-200M-FINE-TUNING-INDIV', 'TimeMoE-50M-FINE-TUNING-GLOBAL', 'TimeMoE-200M-FINE-TUNING-GLOBAL']:
        if type_model in ["TimeMoE-50M-FINE-TUNING", 'TimeMoE-50M-FINE-TUNING-GLOBAL']:
            models = [50]
        else:
            models = [200]

        epochs = [10]
        learning_rates = ['1e-06']

        parameter_combinations = list(product(models, epochs, learning_rates))

        for model, epoch, learning_rate in parameter_combinations:
            try:
                print(f"\n=== Forecasting model: {model} ===")
                print(f"Type_model: {type_model}")
                print(f"Epoch: {epoch}")
                print(f"Learning Rate: {learning_rate}")

                y_pred = create_time_moe_model(
                    data=data_filtered,
                    state=state,
                    derivative=derivative,
                    epochs=epoch,
                    type_model=type_model,
                    model=model,
                    learning_rate=learning_rate,
                )

                # Add results to DataFrame
                results_df = pd.DataFrame([{
                    'TYPE_PREDICTIONS': type_model,
                    'STATE': state,
                    'PRODUCT': derivative,
                    'PREDICTIONS': y_pred,
                    'PARAMETERS': [f'model: "{model}"', f'epoch: {epoch}', f'learning_rate: {learning_rate}'],
                    'ERROR': np.nan
                }])

            except Exception as e:
                # Handle any exceptions during model training
                print(f"An error occurred for derivative '{derivative}' in state '{state}': {e}")

                results_df = pd.DataFrame([{
                    'TYPE_PREDICTIONS': type_model,
                    'STATE': state,
                    'PRODUCT': derivative,
                    'PREDICTIONS': np.nan,
                    'PARAMETERS': [f'model: "{model}"', f'epoch: {epoch}', f'learning_rate: {learning_rate}'],
                    'ERROR': f"An error occurred for derivative '{derivative}' in state '{state}': {e}"
                }])

            # Save results to an Excel file if specified
            directory = f'results_model_local'
            if not os.path.exists(directory):
                os.makedirs(directory)

            file_path = os.path.join(directory, 'results_time_moe_product_5_years.xlsx')
            if os.path.exists(file_path):
                existing_df = pd.read_excel(file_path)
            else:
                existing_df = pd.DataFrame()

            combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            combined_df.to_excel(file_path, index=False)

    else:
        # INFO: Only model zero shot
        try: 
            y_pred = create_time_moe_model(
                data=data_filtered,
                state=state,
                derivative=derivative,
                epochs=None,
                type_model=type_model,
                model=None,
                learning_rate=None,
            )

            results_df = pd.DataFrame([{
                'TYPE_PREDICTIONS': type_model,
                'STATE': state,
                'PRODUCT': derivative,
                'PREDICTIONS': y_pred,
                'PARAMETERS': np.nan,
                'ERROR': np.nan
            }])

        except Exception as e:
            # Handle exceptions outside of iterations
            print(f"An error occurred for derivative '{derivative}' in state '{state}': {e}")

            results_df = pd.DataFrame([{
                'TYPE_PREDICTIONS': type_model,
                'STATE': state,
                'PRODUCT': derivative,
                'PREDICTIONS': np.nan,
                'PARAMETERS': np.nan,
                'ERROR': f"An error occurred for derivative '{derivative}' in state '{state}': {e}"
            }])

        # Save results to an Excel file if specified
        directory = f'results_model_local'
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, 'results_time_moe_product_5_years.xlsx')
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

def run_all_in_thread_5_years(type_model='TimeMoE-50M_ZERO_SHOT'):
    multiprocessing.set_start_method("spawn")

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
        for derivative in products:
            print(f"========== State: {state}, product: {derivative} ==========")
           
            # Set random seeds for reproducibility
            random.seed(42)
            np.random.seed(42)
            tf.random.set_seed(42)
            os.environ['PYTHONHASHSEED'] = str(42)
            tf.keras.utils.set_random_seed(42)

            # Filter data for the current state and derivative
            data_filtered = all_data[(all_data['state'] == state) & (all_data['product'] == derivative)]
            
            thread = multiprocessing.Process(target=run_time_moe_5_years, args=(state, derivative, data_filtered, type_model))
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
    derivative = "gasolinac"
    
    # Loading and preparing data
    data_filtered_test = pd.read_csv(f"../database/venda_process/mensal/uf/{derivative}/mensal_{state}_{derivative}.csv", sep=";",  parse_dates=['timestamp'], date_parser=convert_date)

    print(f" ========== Starting univariate test for the state of {state} - {derivative} ==========")

    # Recording start time of execution
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nExecution started at: {start_timestamp}")
    start_time = time.time()

    # Running the time_moe model
    y_pred = create_time_moe_model(
                data= data_filtered_test,  
                state= state, 
                derivative= derivative, 
                type_model="TimeMoE-200M_ZERO_SHOT", 
                epochs=None,
                model= None, 
                learning_rate=None
     )

    # Recording end time and calculating execution duration
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nFunction execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")