'''
Code Description:

This script performs training (fine-tuning) of the Time-MoE model. For each option, a specific dataset is generated with a different configuration. 
It iterates over possible combinations of model, normalization method, dataset, and learning rate, 
and fine-tunes the model accordingly. The script can process datasets from different years, 
selecting `.jsonl` files from subdirectories based on the specified year or including all subdirectories.
'''

import os
import subprocess
from itertools import product
import pandas as pd

# Configurations
models = [200]
epochs = 10
norms = ['zero']

directory = 'dataset_individual_5_anos/'
year_folder = '2024'  # Change to '*' to include all subdirectories

# List all .jsonl files within the specified subdirectories
datasets = []
for root, _, files in os.walk(directory):
    # Filters subdirectories by year_folder or includes all if year_folder is '*'
    if year_folder == '*' or os.path.basename(root).endswith(year_folder):
        for file in files:
            if file.endswith('.jsonl'):
                datasets.append(os.path.join(root, file))

learning_rates = [1e-6]

os.environ["WANDB_MODE"] = "disabled"

# Generate all possible parameter combinations
parameter_combinations = list(product(models, norms, datasets, learning_rates))

for model, norm, dataset_path, learning_rate in parameter_combinations:
    dataset_name = os.path.basename(dataset_path) 
    parts = dataset_name.split('_')
    
    try:
        state = parts[1]
        derivative = parts[2].replace('.jsonl', '')
        year = os.path.basename(os.path.dirname(dataset_path)).split('_')[-1]
    except IndexError:
        print(f"Error processing file: {dataset_name}")
        continue

    print(f"\n=== Fine-tuning model {model} with parameters: ===")
    print(f"Dataset: {dataset_name}")
    print(f"Normalization: {norm}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Year: {year}")
    print(f"State: {state}")
    print(f"Derivative: {derivative}")
 
    output_dir = f"models_fine_tuning_individual_article_5_years/model_{model}M/fine_tuning_{model}M_{year}_{state}_{derivative}_{epochs}_epochs_{learning_rate}/time_moe"
    os.makedirs(output_dir, exist_ok=True)

    # Fine-tuning command
    command = [
        "python",
        "main.py",
        "-d", dataset_path,
        "-m", f"Maple728/TimeMoE-{model}M",
        "-o", output_dir,
        "--num_train_epochs", str(epochs),
        "--normalization_method", norm,
        "--learning_rate", str(learning_rate),
    ]

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