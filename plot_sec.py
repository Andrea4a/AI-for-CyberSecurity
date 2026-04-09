import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from utils import get_grid, create_dir
from config import *

# === Argomenti da riga di comando ===
args = parse_args()
# Determina il tipo di attacco specificat
if args.cw:
    attack_name = "cw"
elif args.bim:
    attack_name = "bim"
elif args.fgsm:
    attack_name = "fgsm"
elif args.pgd:
    attack_name = "pgd"
elif args.deepfool:
    attack_name = "deepfool"
else:
    raise ValueError("Devi specificare almeno un attacco")

attack_type = args.type.lower()     # Tipo attacco (generic o targeted)
if attack_type == "generic":
    metric_t="accuracy"
elif attack_type == "targeted":
    metric_t="success_rate"

net_type = args.net.lower()         # Rete target (nn1 o nn2)

# === Costruzione dinamica del percorso al CSV da plottare ===
# Se rete = nn1, il file è OUTPUT_CSV_BIM_GENERIC, ecc.
# Se rete = nn2, viene aggiunto il suffisso _NN2
if net_type == "nn1":
    csv_var = f"OUTPUT_CSV_{attack_name.upper()}_{attack_type.upper()}"
elif net_type == "nn2":
    csv_var = f"OUTPUT_CSV_{attack_name.upper()}_{attack_type.upper()}_{net_type.upper()}"
csv_path = os.path.expanduser(globals()[csv_var])
# Path cartella output per salvare i plot
if net_type == "nn1":
    output_dir = f"plots_sec_nn1/{attack_name}_{attack_type}"
elif net_type == "nn2":
    output_dir = f"plots_sec_nn2/{attack_name}_{attack_type}"
create_dir(output_dir)

if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"CSV non trovato: {csv_path}")
df = pd.read_csv(csv_path)

# === # Path cartella output per salvare i plot ===
# Ottiene tutti i parametri della grid per l’attacco richiesto
grid = get_grid(attack_name)
if not grid:
    raise ValueError(f"Nessuna configurazione trovata per attacco {attack_name}")
# Elenco dei nomi dei parametri (es. eps, max_iter, ecc.)
all_params = list(grid[0].keys())
# Esclude parametri inutili nei plot (non influenzano il risultato)
exclude = {"verbose", "random_eps"}
params_to_plot = [p for p in all_params if p not in exclude]

# === Funzione per generare un singolo grafico SEC ===
def plot_sec_by_param(df, param, save_base, metric=metric_t):
    legend_params = [p for p in params_to_plot if p != param] # Parametri fissi (leggenda)
    plt.figure(figsize=(12, 6))

    if legend_params:
        # Se ci sono più variabili oltre al parametro da variare, separa in curve
        for key, group in df.groupby(legend_params):
            label = ", ".join([f"{p}={v}" for p, v in zip(legend_params, key)])
            grouped = group.groupby(param)[metric].mean().reset_index()
            plt.plot(grouped[param], grouped[metric], marker='o', label=label)
        plt.legend(loc='best', fontsize='small')
    else:
        # Caso semplice: un solo parametro → una curva
        grouped = df.groupby(param)[metric].mean().reset_index()
        plt.plot(grouped[param], grouped[metric], marker='o')
    
    plt.xlabel(param)
    plt.xticks(sorted(df[param].unique()))
    if metric_t == "accuracy":
        plt.ylabel("Accuracy")
        plt.title(f"SEC - Accuracy vs {param} | {attack_name.upper()} {attack_type.upper()}")
    elif metric_t == "success_rate":
        plt.ylabel("Model Targeted Accuracy")
        plt.title(f"SEC - Model Targeted Accuracy vs {param} | {attack_name.upper()} {attack_type.upper()}")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_base + ".png")
    plt.close()
    print(f"✅ Salvato: {save_base}.png")


# === Genera i plot per ogni parametro ===
for param in params_to_plot:
    base_path = os.path.join(output_dir, f"sec_accuracy_vs_{param}")
    plot_sec_by_param(df, param, base_path, metric=metric_t)
