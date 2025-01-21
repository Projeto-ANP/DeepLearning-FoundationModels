'''
Code Description:

This script fine-tunes the Time-MoE model using various combinations of configurations. 
It iterates over different models, normalization methods, datasets, learning rates, and learning rate schedulers 
to perform fine-tuning on each configuration. The results are stored in a directory structure based on the parameter values.

'''

import os
import subprocess
from itertools import product

# Configurations
models = [50, 200]
epochs = 10
norms = ['none', 'zero']
datasets = ['dataset_petroleum_derivatives_normalized', 'dataset_petroleum_derivatives']
learning_rates = [1e-3, 1e-4, 5e-5, 2e-5, 1e-6]
lr_scheduler_types = ['constant', 'linear', 'cosine']

os.environ["WANDB_MODE"] = "disabled"

# Generate all possible combinations of parameters
parameter_combinations = list(product(models, norms, datasets, learning_rates, lr_scheduler_types))

for model, norm, dataset, learning_rate, lr_scheduler_type in parameter_combinations:

    print(f"\n=== Fine-tuning model {model} with parameters: ===")
    print(f"Dataset: {dataset}")
    print(f"Normalization: {norm}")
    print(f"Learning Rate: {learning_rate}")
    print(f"LR Scheduler: {lr_scheduler_type}")
    
    if dataset == 'dataset_petroleum_derivatives_normalized':
        suffix = 'normalized'
    else:
        suffix = 'raw_data'

    # Define the output directory
    output_dir = f"models_fine_tuning/model_{model}M/{suffix}/{norm}/fine_tuning_{model}_{epochs}_epochs_{suffix}_{learning_rate}_{lr_scheduler_type}/time_moe"
    os.makedirs(output_dir, exist_ok=True)

    # Define the command for fine-tuning
    command = [
        "python",
        "main.py",
        "-d", f"{dataset}.jsonl",
        "-m", f"Maple728/TimeMoE-{model}M",
        "-o", output_dir,
        "--num_train_epochs", str(epochs),
        "--normalization_method", norm,
        "--learning_rate", str(learning_rate),
        "--lr_scheduler_type", lr_scheduler_type
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