from darts.models import NBEATSModel
from darts import TimeSeries

from sklearn.metrics import mean_absolute_percentage_error as mape

import numpy as np
import pandas as pd

import os
import time

import multiprocessing

from matplotlib import pyplot as plt

import warnings
from warnings import simplefilter

from metrics_forecasting import rrmse, pbe, pocid, mase
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

def get_scaled_data(df, len_data=None):
    df = df[:-12]
    
    if len_data is not None:
        df = df[-len_data:]
    
    # MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_scaled = scaler.fit_transform(df.values.reshape(-1, 1))

    return df_scaled, scaler

def create_nbeats_model(forecast_steps, time_steps, data):
   
    df = data['m3']
    
    df_scaled, scaler = get_scaled_data(df, None)

    series = TimeSeries.from_values(df_scaled)

    # INFO: First experiment
    # model = NBEATSModel(
    #     input_chunk_length=12,
    #     output_chunk_length=1,
    #     n_epochs=50,
    #     batch_size=16,
    #     random_state=42,
    #     activation='LeakyReLU',
    #     pl_trainer_kwargs={"accelerator": "gpu", "devices": 1}
    # )

    # INFO: Second experiment
    # model = NBEATSModel(
    #     input_chunk_length=12,
    #     output_chunk_length=1,
    #     n_epochs=100,
    #     batch_size=32,
    #     random_state=42,
    #     activation='PReLU',
    #     pl_trainer_kwargs={"accelerator": "gpu", "devices": 1},
    #     num_blocks=2,
    #     num_layers=3,
    #     layer_widths=128
    # )

    # INFO: Third experiment
    # model = NBEATSModel(
    #     input_chunk_length=12,
    #     output_chunk_length=1,
    #     n_epochs=30,
    #     batch_size=16,
    #     random_state=42,
    #     activation='ReLU',
    #     pl_trainer_kwargs={"accelerator": "gpu", "devices": 1},
    #     num_blocks=3,
    #     num_layers=4,
    #     layer_widths=64
    # )

    # INFO: Fourth experiment
    # model = NBEATSModel(
    #     input_chunk_length=12,
    #     output_chunk_length=1,
    #     n_epochs=20,
    #     batch_size=64,
    #     random_state=42,
    #     activation='ReLU',
    #     pl_trainer_kwargs={"accelerator": "gpu", "devices": 1},
    #     num_blocks=4,
    #     num_layers=5,
    #     layer_widths=32
    # )

    # INFO: Fifth experiment
    # model = NBEATSModel(
    #     input_chunk_length=12,
    #     output_chunk_length=1,
    #     n_epochs=75,
    #     batch_size=16,
    #     random_state=42,
    #     activation='ReLU',
    #     pl_trainer_kwargs={"accelerator": "gpu", "devices": 1},
    #     num_blocks=3,
    #     num_layers=2,
    #     layer_widths=256
    # )

    # INFO: Sixth  experiment
    model = NBEATSModel(
        input_chunk_length=12,
        output_chunk_length=1,
        n_epochs=120,
        batch_size=16,
        random_state=42,
        activation='ReLU',
        pl_trainer_kwargs={"accelerator": "gpu", "devices": 1},
        num_blocks=2,
        num_layers=3,
    )

    model.fit(series)

    forecast = model.predict(12)

    y_pred = scaler.inverse_transform(forecast.values().reshape(-1, 1))
    y_pred = y_pred.flatten()

    y_test = df[-12:].values

   # Calculating evaluation metrics
    y_baseline = df[-forecast_steps*2:-forecast_steps].values
    rrmse_result_time_moe = rrmse(y_test, y_pred, df[:-12].mean())
    mape_result_time_moe = mape(y_test, y_pred)
    pbe_result_time_moe = pbe(y_test, y_pred)
    pocid_result_time_moe = pocid(y_test, y_pred)
    mase_result_time_moe = np.mean(np.abs(y_test - y_pred)) / np.mean(np.abs(y_test - y_baseline))

    print(f"\nResultados N-BEATS: \n")
    print(f'RRMSE: {rrmse_result_time_moe}')
    print(f'MAPE: {mape_result_time_moe}')
    print(f'PBE: {pbe_result_time_moe}')
    print(f'POCID: {pocid_result_time_moe}')
    print(f'MASE: {mase_result_time_moe}')
        
    return rrmse_result_time_moe, mape_result_time_moe, pbe_result_time_moe, pocid_result_time_moe, mase_result_time_moe, y_pred, 

def run_nbeats(state, product, forecast_steps, time_steps, data_filtered, bool_save, log_lock):
    """
    Execute nbeats model training and save the results to an Excel file.

    Parameters:
        - state (str): State for which the nbeats model is trained.
        - product (str): Product for which the nbeats model is trained.
        - forecast_steps (int): Number of steps to forecast in the future.
        - time_steps (int): Length of time steps (window size) for input data generation.
        - data_filtered (pd.DataFrame): Filtered dataset containing data for the specific state and product.
        - bool_save (bool): Whether to save the results to an Excel file.
        - log_lock (multiprocessing.Lock): Lock for synchronized logging.

    Returns:
        None
    """

    # Record the start time of execution
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nExecution started at: {start_timestamp}")
    start_time = time.time()

    try:
        # Run nbeats model training and capture performance metrics
        rrmse_result, mape_result, pbe_result, pocid_result, mase_result, y_pred = \
        create_nbeats_model(forecast_steps=forecast_steps, time_steps=time_steps, data=data_filtered)
        
        # Create a DataFrame to store the results
        results_df = pd.DataFrame([{'FORECAST_STEPS': forecast_steps,
                                    'TIME_FORECAST': time_steps,
                                    'TYPE_PREDICTIONS': 'N-BEATS6',
                                    'STATE': state,
                                    'PRODUCT': product,
                                    'RRMSE': rrmse_result,
                                    'MAPE': mape_result,
                                    'PBE': pbe_result,
                                    'POCID': pocid_result,
                                    'MASE': mase_result,
                                    'PREDICTIONS': y_pred,
                                    'ERROR': np.nan}])
        
    except Exception as e:
        # Handle exceptions during model training
        print(f"An error occurred for product '{product}' in state '{state}': {e}")
        
        results_df = pd.DataFrame([{'FORECAST_STEPS': np.nan,
                                    'TIME_FORECAST': np.nan,
                                    'TYPE_PREDICTIONS': 'N-BEATS6',
                                    'STATE': state,
                                    'PRODUCT': product,
                                    'RRMSE': np.nan,
                                    'MAPE': np.nan,
                                    'PBE': np.nan,
                                    'POCID': np.nan,
                                    'MASE': np.nan,
                                    'PREDICTIONS': np.nan,
                                    'ERROR': f"An error occurred for product '{product}' in state '{state}': {e}"}])
            
    # Save the results to an Excel file if requested
    if bool_save:
        with log_lock:
            directory = f'results_model_local'
            if not os.path.exists(directory):
                os.makedirs(directory)

            file_path = os.path.join(directory, 'nbeats_results6.xlsx')
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
    log_entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')}, {state}, {product}, {forecast_steps}, {time_steps}, {execution_time:.2f}\n"

    # Write the log entry with a lock to prevent race conditions
    with log_lock:
        with open("training_log.csv", "a") as log_file:
            log_file.write(log_entry)

def run_nbeats_in_thread(forecast_steps, time_steps, bool_save):

    """
    Execute nbeats model training in separate processes for different state and product combinations.

    Parameters:
        - forecast_steps (int): Number of steps to forecast in the future.
        - time_steps (int): Length of time steps (window size) for input data generation.
        - bool_save (bool): Whether to save the trained models (True/False).

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

            # Create a separate process for running the nbeats model
            process = multiprocessing.Process(
                target=run_nbeats,
                args=(
                    state, product, forecast_steps, time_steps,
                    data_filtered, bool_save,
                    log_lock, 
                )
            )

            # Start and wait for the process to complete
            process.start()
            process.join()

    print("All processes completed successfully.")
    
def product_and_single_thread_testing():    
    """
    Perform a simple training thread using nbeats model for time series forecasting.

    This function initializes random seeds, loads a database, executes an nbeats model,
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

    # Running the nbeats model
    y_pred = \
    create_nbeats_model(forecast_steps=12, time_steps=12, data=data_filtered_test)

    # Recording end time and calculating execution duration
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nFunction execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")