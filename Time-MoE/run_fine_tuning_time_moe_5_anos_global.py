# Code Description:

'''
This script performs training (fine-tuning) of the Time-MoE model. For each option, 
a specific dataset is generated with a different configuration. The script iterates 
over all possible combinations of model, normalization, dataset and learning rate, 
and generates the appropriate datasets to fit the model according to the given settings.
'''


import os
import subprocess
from itertools import product
import pandas as pd

# Configurations
models = [50, 200]
epochs = 10
norms = ['zero']

directory = 'dataset_global_5_anos/'
datasets = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

learning_rates = [1e-6]

os.environ["WANDB_MODE"] = "disabled"

# Generate all possible parameter combinations
parameter_combinations = list(product(models, norms, datasets, learning_rates))

for model, norm, dataset, learning_rate in parameter_combinations:
    parts = dataset.split('_')
    # state = parts[1]
    year = parts[2].split('.')[0]
    
    print(f"\n=== Fine-tuning model {model} with parameters: ===")
    print(f"Dataset: {dataset}")
    print(f"Normalization: {norm}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Year: {year}")

    # Define output directory
    output_dir = f"models_fine_tuning_global_article_5_years/model_{model}M/fine_tuning_{model}M_{year}_{epochs}_epochs_{learning_rate}/time_moe"
    os.makedirs(output_dir, exist_ok=True)

    # Define the fine-tuning command
    command = [
        "python",
        "main.py",
        "-d", f"dataset_global_5_anos/{dataset}",
        "-m", f"Maple728/TimeMoE-{model}M",
        "-o", output_dir,
        "--num_train_epochs", str(epochs),
        "--normalization_method", norm,
        "--learning_rate", str(learning_rate),
    ]

    # Execute the command
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(timeout=None)

        print("Process output:")
        print(stdout)

        if stderr:
            print("Process errors:")
            print(stderr)

    except subprocess.SubprocessError as e:
        print(f"Error executing the command for model {model}: {e}")