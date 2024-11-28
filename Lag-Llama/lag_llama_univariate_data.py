from itertools import islice
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from tqdm.autonotebook import tqdm
import torch
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.pandas import PandasDataset
from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.dataset.common import ListDataset
import pandas as pd

# dataset = get_dataset("m4_weekly")

import os
import csv

products = sorted([name for name in os.listdir('../database/venda_process/mensal/uf/') if os.path.isdir(os.path.join('../database/venda_process/mensal/uf/', name))])

def extract_estado(file_name):
    # Split the file name by underscores
    parts = file_name.split('_')
    # Extract the name between underscores
    estado = parts[1]
    return estado

def read_csv_files(folder_path):
    estados = []
    # List all files in the folder
    files = os.listdir(folder_path)
    # Iterate through each file
    for file_name in files:
        # Check if it's a CSV file
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            # Open the CSV file and read the data
            with open(file_path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                # Assuming the first row contains headers
                headers = next(reader)
                # Extract estado from file name and append to estados list
                estado = extract_estado(file_name)
                estados.append(estado)
                estados.sort()
    return estados

# def get_lag_llama_predictions(dataset, prediction_length, context_length=32, num_samples=20, device="cuda", batch_size=64, nonnegative_pred_samples=True):
def get_lag_llama_predictions(dataset, prediction_length, context_length=36, num_samples=100, device="cuda", batch_size=64, nonnegative_pred_samples=True):
    ckpt = torch.load("lag-llama.ckpt", map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
    estimator = LagLlamaEstimator(
        ckpt_path="lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=context_length,
        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        nonnegative_pred_samples=nonnegative_pred_samples,
        # linear positional encoding scaling
        rope_scaling={
            "type": "linear",
            "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
        },
        batch_size=batch_size,
        num_parallel_samples=num_samples,
    )
    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    forecasts = list(tqdm(forecast_it, total=len(dataset), desc="Forecasting batches"))
    tss = list(tqdm(ts_it, total=len(dataset), desc="Ground truth"))
    return forecasts, tss

for product in products:
    folder_path = f'../database/venda_process/mensal/uf/{product}/'
    estados = read_csv_files(folder_path)
    for estado in estados:
        # estado = 'ac'
        # product = 'etanolhidratado'
        dataset_own = pd.read_csv(f"../database/venda_process/mensal/uf/{product}/mensal_{estado}_{product}.csv", header=0, sep=";")
        # filtered_dataset = dataset_own[dataset_own['item_id'] == 'GLP']
        ##########################################3
        # Criação do dataset no formato necessário para o GluonTS
        target_values = dataset_own['m3']
        start_date = pd.to_datetime(dataset_own['timestamp'].iloc[0], format='%Y%m')
        # Divisão em treino e teste (exemplo: usando os últimos 12 meses para teste)
        prediction_length = 12
        train_dataset = ListDataset(
            [{"start": '1990-01', "target": target_values[:-prediction_length], "item_id": product}],
            freq="M"
        )
        test_dataset = ListDataset(
            [{"start": '1990-01', "target": target_values, "item_id": product}],
            freq="M"
        )
        # prediction_length = dataset.metadata.prediction_length
        prediction_length = 12
        context_length = prediction_length*3
        num_samples = 20
        device = "cuda"
        forecasts, tss = get_lag_llama_predictions(
            # dataset.test,
            test_dataset,
            prediction_length=prediction_length,
            num_samples=num_samples,
            context_length=context_length,
            device=device
        )
        plt.figure(figsize=(20, 15))
        date_formater = mdates.DateFormatter('%b, %d')
        plt.rcParams.update({'font.size': 15})
        # Iterate through the first 9 series, and plot the predicted samples
        for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
            ax = plt.subplot(1, 1, idx+1)
            plt.plot(ts[-4 * prediction_length:].to_timestamp(), label="target", )
            forecast.plot( color='g')
            plt.xticks(rotation=60)
            ax.xaxis.set_major_formatter(date_formater)
            ax.set_title(forecast.item_id)
        plt.gcf().tight_layout()
        plt.legend()
        # plt.savefig(f'forecast_plots7_{product}_{estado}.png')  # Save the figure as a PNG file
        plt.show()
        ######################################################################
        ckpt = torch.load("lag-llama.ckpt", map_location=device)
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
        estimator = LagLlamaEstimator(
                ckpt_path="lag-llama.ckpt",
                prediction_length=prediction_length,
                context_length=context_length,
                # distr_output="neg_bin",
                # scaling="mean",
                nonnegative_pred_samples=True,
                aug_prob=0,
                lr=5e-4,
                # estimator args
                input_size=estimator_args["input_size"],
                n_layer=estimator_args["n_layer"],
                n_embd_per_head=estimator_args["n_embd_per_head"],
                n_head=estimator_args["n_head"],
                time_feat=estimator_args["time_feat"],
                # rope_scaling={
                #     "type": "linear",
                #     "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
                # },
                batch_size=64,
                num_parallel_samples=num_samples,
                trainer_kwargs = {"max_epochs": 50,}, # <- lightning trainer arguments
            )
        # predictor = estimator.train(dataset.train, cache_data=True, shuffle_buffer_length=1000)
        predictor = estimator.train(train_dataset, cache_data=True)#, shuffle_buffer_length=1000)
        forecast_it, ts_it = make_evaluation_predictions(
            # dataset=dataset.test,
            dataset=test_dataset,
            predictor=predictor,
            num_samples=num_samples
        )
        # forecasts = list(tqdm(forecast_it, total=len(dataset), desc="Forecasting batches"))
        forecasts = list(tqdm(forecast_it, total=len(test_dataset), desc="Forecasting batches"))
        # tss = list(tqdm(ts_it, total=len(dataset), desc="Ground truth"))
        tss = list(tqdm(ts_it, total=len(test_dataset), desc="Ground truth"))
        plt.figure(figsize=(20, 15))
        date_formater = mdates.DateFormatter('%b, %d')
        plt.rcParams.update({'font.size': 15})
        # Iterate through the first 9 series, and plot the predicted samples
        for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
            ax = plt.subplot(1, 1, idx+1)
            plt.plot(ts[-4 * prediction_length:].to_timestamp(), label="target", )
            forecast.plot( color='g')
            plt.xticks(rotation=60)
            ax.xaxis.set_major_formatter(date_formater)
            ax.set_title(forecast.item_id)
        plt.gcf().tight_layout()
        plt.legend()
        # plt.savefig(f'forecast_plots8_{product}_{estado}.png')
        plt.show()
        # evaluator = Evaluator()
        # agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts))
        # CSV Output VALORES REAIS
        with open(f'LagLlama_output.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            temp = forecast.samples.mean(axis=0)
            values_list = temp.flatten().tolist()
            writer.writerow([product, estado, 'LagLlama', values_list])