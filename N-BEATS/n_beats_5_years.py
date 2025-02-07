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

from metrics_forecasting import pbe, pocid, mase
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

def get_scaled_data(df):
    df = df[:-12]
    
    # MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_scaled = scaler.fit_transform(df.values.reshape(-1, 1))

    return df_scaled, scaler

def create_nbeats_model_5_years(type_experiment, data):
    y_preds_5_years = []

    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

    end_date = data['timestamp'].max()

    for years in range(5, 0, -1):
        start_date = end_date - pd.DateOffset(years=years-1)

        df = data[data['timestamp'] <= start_date]
        print(f'\nData filtered for {start_date.date()}\n')
        df = df['m3']

        df_scaled, scaler = get_scaled_data(df)

        series = TimeSeries.from_values(df_scaled)


        # INFO: First experiment
        if type_experiment == 1:
            model = NBEATSModel(
                input_chunk_length=12,
                output_chunk_length=1,
                n_epochs=50,
                batch_size=16,
                random_state=42,
                activation='LeakyReLU',
                pl_trainer_kwargs={"accelerator": "gpu", "devices": 1}
            )

        # INFO: Second experiment
        elif type_experiment == 2:
            model = NBEATSModel(
                input_chunk_length=12,
                output_chunk_length=1,
                n_epochs=100,
                batch_size=32,
                random_state=42,
                activation='PReLU',
                pl_trainer_kwargs={"accelerator": "gpu", "devices": 1},
                num_blocks=2,
                num_layers=3,
                layer_widths=128
            )

        # INFO: Third experiment
        elif type_experiment == 3:
            model = NBEATSModel(
                input_chunk_length=12,
                output_chunk_length=1,
                n_epochs=30,
                batch_size=16,
                random_state=42,
                activation='ReLU',
                pl_trainer_kwargs={"accelerator": "gpu", "devices": 1},
                num_blocks=3,
                num_layers=4,
                layer_widths=64
            )

        # INFO: Fourth experiment
        elif type_experiment == 4:
            model = NBEATSModel(
                input_chunk_length=12,
                output_chunk_length=1,
                n_epochs=20,
                batch_size=64,
                random_state=42,
                activation='ReLU',
                pl_trainer_kwargs={"accelerator": "gpu", "devices": 1},
                num_blocks=4,
                num_layers=5,
                layer_widths=32
            )

        # INFO: Fifth experiment
        elif type_experiment == 5:
            model = NBEATSModel(
                input_chunk_length=12,
                output_chunk_length=1,
                n_epochs=75,
                batch_size=16,
                random_state=42,
                activation='ReLU',
                pl_trainer_kwargs={"accelerator": "gpu", "devices": 1},
                num_blocks=3,
                num_layers=2,
                layer_widths=256
            )

        # INFO: Sixth  experiment
        elif type_experiment == 6:
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

        # Evaluation metrics
        y_baseline = df[-12*2:-12].values
        mape_result_n_beats = mape(y_test, y_pred)
        pbe_result_n_beats = pbe(y_test, y_pred)
        pocid_result_n_beats = pocid(y_test, y_pred)
        mase_result_n_beats = np.mean(np.abs(y_test - y_pred)) / np.mean(np.abs(y_test - y_baseline))

        print("\nResultado n_beats: \n")
        print(f'MAPE: {mape_result_n_beats}')
        print(f'PBE: {pbe_result_n_beats}')
        print(f'POCID: {pocid_result_n_beats}')
        print(f'MASE: {mase_result_n_beats}')
    
        y_preds_5_years.append(y_pred)
    
    return np.concatenate(y_preds_5_years).tolist()
                      
def run_nbeats_5_years(state, product, type_experiment, data_filtered):
    """
    Execute nbeats model training and save the results to an Excel file.

    Parameters:
        - state (str): State for which the nbeats model is trained.
        - product (str): Product for which the nbeats model is trained.
        - type_experiment
        - data_filtered (pd.DataFrame): Filtered dataset containing data for the specific state and product.
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
        y_pred = \
        create_nbeats_model_5_years(type_experiment=type_experiment, data=data_filtered)
        
        # Create a DataFrame to store the results
        results_df = pd.DataFrame([{'TYPE_PREDICTIONS': 'N-BEATS',
                                    'EXPERIMENT': 'Parameters_' + type_experiment,
                                    'STATE': state,
                                    'PRODUCT': product,
                                    'RRMSE': np.nan,
                                    'MAPE': np.nan,
                                    'PBE': np.nan,
                                    'POCID': np.nan,
                                    'MASE': np.nan,
                                    'PREDICTIONS': y_pred,
                                    'ERROR': np.nan}])
        
    except Exception as e:
        # Handle exceptions during model training
        print(f"An error occurred for product '{product}' in state '{state}': {e}")
        
        results_df = pd.DataFrame([{'TYPE_PREDICTIONS': 'N-BEATS',
                                    'EXPERIMENT': 'Parameters_' + type_experiment,
                                    'STATE': state,
                                    'PRODUCT': product,
                                    'RRMSE': np.nan,
                                    'MAPE': np.nan,
                                    'PBE': np.nan,
                                    'POCID': np.nan,
                                    'MASE': np.nan,
                                    'PREDICTIONS': np.nan,
                                    'ERROR': f"An error occurred for product '{product}' in state '{state}': {e}"}])
            

    directory = f'results_model_local'
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, 'nbeats_results_5_years.xlsx')
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

def run_nbeats_in_thread_5_years(type_experiment=1):

    """
    Execute nbeats model training in separate processes for different state and product combinations.

    Parameters:
        type_experiment = Default is 1

    Returns:
        None
    """
    # Set the multiprocessing start method
    multiprocessing.set_start_method("spawn")

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
                target=run_nbeats_5_years,
                args=(state, product, type_experiment, data_filtered)
            )

            # Start and wait for the process to complete
            process.start()
            process.join()

    print("All processes completed successfully.")
    
def product_and_single_thread_testing_5_years():    
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
    create_nbeats_model_5_years(type_experiment=1, data=data_filtered_test)

    # Recording end time and calculating execution duration
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nFunction execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")