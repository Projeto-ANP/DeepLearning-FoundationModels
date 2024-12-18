import os
import subprocess
from itertools import product

# Configurações
models = [50, 200]
epochs = 10
norms = ['none', 'zero']
datasets = ['dataset_petroleum_derivatives_normalizados', 'dataset_petroleum_derivatives']
learning_rates = [1e-3, 1e-4, 5e-5, 2e-5, 1e-6]
lr_scheduler_types = ['constant', 'linear', 'cosine']

os.environ["WANDB_MODE"] = "disabled"

# Gera todas as combinações possíveis de parâmetros
parameter_combinations = list(product(models, norms, datasets, learning_rates, lr_scheduler_types))

for model, norm, dataset, learning_rate, lr_scheduler_type in parameter_combinations:

    print(f"\n=== Fine-tuning do modelo {model} com parâmetros: ===")
    print(f"Dataset: {dataset}")
    print(f"Normalization: {norm}")
    print(f"Learning Rate: {learning_rate}")
    print(f"LR Scheduler: {lr_scheduler_type}")
    
    if dataset == 'dataset_petroleum_derivatives_normalizados':
        sufix = 'normalizados'
    else:
        sufix = 'raw_data'

    # Define o diretório de saída
    output_dir = f"models_fine_tuning/model_{model}M/{sufix}/{norm}/fine_tuning_{model}_{epochs}_epochs_{sufix}_{learning_rate}_{lr_scheduler_type}/time_moe"
    os.makedirs(output_dir, exist_ok=True)

    # Define o comando para o fine-tuning
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

    # Executa o comando
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(timeout=None)

        print("Saída do processo:")
        print(stdout)

        if stderr:
            print("Erros do processo:")
            print(stderr)

    except subprocess.SubprocessError as e:
        print(f"Erro ao executar o comando para o modelo {model}: {e}")
