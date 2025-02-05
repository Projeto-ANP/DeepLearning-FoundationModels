import torch
from transformers import AutoModelForCausalLM
from time_moe.models.modeling_time_moe import TimeMoeForPrediction
from itertools import product

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

from metrics_time_moe import rrmse, pbe, pocid 
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
    type_model='TimeMoE-50M_ZERO_SHOT', 
    norm='none', 
    learning_rate=None, 
):
    """
    Runs a TimeMoE model for time series forecasting.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the time series data.
    - state (str): The specific state for which the model is trained.
    - derivative (str): The specific derivative for which the model is trained.
    - type_model (str, optional): Type of TimeMoE to be used. Default is 'TimeMoE-50M'.
    - norm (str): Normalization method. Options are ['none', 'zero'].
    - learning_rate (float): Learning rate for the optimizer.

    Returns:
    - rrmse_result (float).
    - mape_result (float).
    - pbe_result (float).
    - pocid_result (float).
    - mase_result (float).
    - y_pred (np.ndarray): Array of predicted values.
    """

    df = data['m3']

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
        model_path = f"models_fine_tuning_global_artigo/model_50M/{norm}/fine_tuning_50_10_epochs_{learning_rate}/time_moe"
        model = TimeMoeForPrediction.from_pretrained(
            model_path,
            device_map="cuda", 
            trust_remote_code=True,
        )

    elif type_model == "TimeMoE-200M-FINE-TUNING-GLOBAL":
        model_path = f"models_fine_tuning_global_artigo/model_200M/{norm}/fine_tuning_200_10_epochs_{learning_rate}/time_moe"
        model = TimeMoeForPrediction.from_pretrained(
            model_path,
            device_map="cuda", 
            trust_remote_code=True,
        )
    
    # INFO: ================== FINE-TUNING INDIVIDUAL ==================
    elif type_model == "TimeMoE-50M-FINE-TUNING-INDIV":
        model_path = f"models_fine_tuning_individual_artigo/model_50M/{norm}/fine_tuning_50_{state}_{derivative}_10_epochs_{learning_rate}/time_moe"
        model = TimeMoeForPrediction.from_pretrained(
            model_path,
            device_map="cuda", 
            trust_remote_code=True,
        )

    elif type_model == "TimeMoE-200M-FINE-TUNING-INDIV":
        model_path = f"models_fine_tuning_individual_artigo/model_200M/{norm}/fine_tuning_200_{state}_{derivative}_10_epochs_{learning_rate}/time_moe"
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
    rrmse_result_time_moe = rrmse(y_test, y_pred, df[:-12].mean())
    mape_result_time_moe = mape(y_test, y_pred)
    pbe_result_time_moe = pbe(y_test, y_pred)
    pocid_result_time_moe = pocid(y_test, y_pred)
    mase_result_time_moe = np.mean(np.abs(y_test - y_pred)) / np.mean(np.abs(y_test - y_baseline))

    print(f"\nResultados Time-MOE modelo: {type_model} \n")
    print(f'RRMSE: {rrmse_result_time_moe}')
    print(f'MAPE: {mape_result_time_moe}')
    print(f'PBE: {pbe_result_time_moe}')
    print(f'POCID: {pocid_result_time_moe}')
    print(f'MASE: {mase_result_time_moe}')
        
    return rrmse_result_time_moe, mape_result_time_moe, pbe_result_time_moe, pocid_result_time_moe, mase_result_time_moe, y_pred, 
                    
def run_time_moe(state, derivative, data_filtered, log_lock, type_model='TimeMoE-50M'):

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
        df_all_results = pd.DataFrame()
            
        if type_model in ['TimeMoE-50M-FINE-TUNING-INDIV', 'TimeMoE-200M-FINE-TUNING-INDIV', 'TimeMoE-50M-FINE-TUNING-GLOBAL', 'TimeMoE-200M-FINE-TUNING-GLOBAL']:

            if type_model in ["TimeMoE-50M-FINE-TUNING", 'TimeMoE-50M-FINE-TUNING-GLOBAL']:
                models = [50]
            else:
                models = [200]

            # norms = ['none', 'zero']
            # datasets = ['dataset_petroleum_derivatives_normalizados', 'dataset_petroleum_derivatives']
            # learning_rates = ['0.001', '0.0001', '5e-05', '2e-05', '1e-06']
            # lr_scheduler_types = ['constant', 'linear', 'cosine']

            # parameter_combinations = list(product(models, norms, datasets, learning_rates, lr_scheduler_types))

            norms = ['zero']
            datasets = ['dataset_petroleum_derivatives_global']
            learning_rates = ['1e-06']

            parameter_combinations = list(product(models, norms, datasets, learning_rates))
            # print(parameter_combinations)

            for model, norm, dataset, learning_rate in parameter_combinations:
                try:
                    print(f"\n=== Forecasting model: {model} ===")
                    print(f"Type_model: {type_model}")
                    print(f"Dataset: {dataset}")
                    print(f"Normalization: {norm}")
                    print(f"Learning Rate: {learning_rate}")

                    rrmse_result, mape_result, pbe_result, pocid_result, mase_result, y_pred = create_time_moe_model(
                        data=data_filtered,
                        state=state,
                        derivative=derivative,
                        type_model=type_model,
                        show_plot=False,
                        model=model,
                        norm=norm,
                        learning_rate=learning_rate,
                    )

                    # Add results to DataFrame
                    results_df = pd.DataFrame([{
                        'TYPE_PREDICTIONS': type_model,
                        'STATE': state,
                        'PRODUCT': derivative,
                        'PREDICTIONS': y_pred,
                        'ERROR': np.nan
                    }])

                except Exception as e:
                    # Handle any exceptions during model training
                    print(f"An error occurred for derivative '{derivative}' in state '{state}': {e}")
                    
                    # 'TYPE_PREDICTIONS': type_model + f"_{sufix}_{norm}_{learning_rate}_{lr_scheduler_type}",

                    results_df = pd.DataFrame([{
                        'TYPE_PREDICTIONS': type_model,
                        'STATE': state,
                        'PRODUCT': derivative,
                        'PREDICTIONS': np.nan,
                        'ERROR': f"An error occurred for derivative '{derivative}' in state '{state}': {e}"
                    }])

                # Append current results to the combined DataFrame
                df_all_results = pd.concat([df_all_results, results_df], ignore_index=True)

        else:
            
            # Run single TimeMoE model training
            rrmse_result, mape_result, pbe_result, pocid_result, mase_result, y_pred = create_time_moe_model(
                data=data_filtered,
                state=state,
                derivative=derivative,
                type_model=type_model,
                show_plot=True
            )

            df_all_results = pd.DataFrame([{
                'TYPE_PREDICTIONS': type_model,
                'STATE': state,
                'PRODUCT': derivative,
                'RRMSE': rrmse_result,
                'MAPE': mape_result,
                'PBE': pbe_result,
                'POCID': pocid_result,
                'MASE': mase_result,
                'PREDICTIONS': y_pred,
                'ERROR': np.nan
            }])

    except Exception as e:
        # Handle exceptions outside of iterations
        print(f"An error occurred for derivative '{derivative}' in state '{state}': {e}")

        df_all_results = pd.DataFrame([{
            'TYPE_PREDICTIONS': type_model,
            'STATE': state,
            'PRODUCT': derivative,
            'RRMSE': np.nan,
            'MAPE': np.nan,
            'PBE': np.nan,
            'POCID': np.nan,
            'MASE': np.nan,
            'PREDICTIONS': np.nan,
            'ERROR': f"An error occurred for derivative '{derivative}' in state '{state}': {e}"
        }])

    # Save results to Excel file
    with log_lock:
        directory = f'results_model_local'
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, 'results_time_moe_fine_last_year.xlsx')
        if os.path.exists(file_path):
            existing_df = pd.read_excel(file_path)
        else:
            existing_df = pd.DataFrame()

        combined_df = pd.concat([existing_df, df_all_results], ignore_index=True)
        combined_df.to_excel(file_path, index=False)

    # Calculate and print the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Function execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

def run_all_in_thread(type_model='TimeMoE-50M'):
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
            
            thread = multiprocessing.Process(target=run_time_moe, args=(state, derivative, data_filtered, log_lock, type_model))
            thread.start()
            thread.join()  

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
    rrmse_result, mape_result, pbe_result, pocid_result, mase_result, y_pred = create_time_moe_model(
                data= data_filtered_test,  
                state= state, 
                derivative= derivative, 
                type_model="TimeMoE-200M_ZERO_SHOT", 
                model= "none", 
                norm= "none", 
                learning_rate="none", 
                lr_scheduler_type="none",
     )

    # Recording end time and calculating execution duration
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nFunction execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")