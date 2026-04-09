import os
import time
import random
import re
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import argparse

from resnet.resnet import resnet50
from utils import *
from config import *

args = parse_args()
ATTACK_TYPE = args.type # 'generic' o 'targeted'

# === PARSE ARGOMENTI ===
# Determina quale attacco è stato selezionato
if args.cw:
    ATTACK = "cw"
elif args.bim:
    ATTACK = "bim"
elif args.fgsm:
    ATTACK = "fgsm"
elif args.pgd:
    ATTACK = "pgd"
elif args.deepfool:
    ATTACK = "deepfool"
else:
    parser.error("Devi specificare almeno un attacco: --cw, --bim, --fgsm, --pgd")

# Preprocessore VGGFace: resize 256, crop 224, BGR mean subtraction
preprocessor = VGGPreprocessor()

# === PERCORSI ===
ADV_ROOT = os.path.join("nn1", ATTACK, f"{ATTACK}_{ATTACK_TYPE}", f"sec_adv_{ATTACK}_{ATTACK_TYPE}")
OUT_DIR  = f"nn2/results_adv_resnet50_ft_{ATTACK_TYPE}_{ATTACK}"
os.makedirs(OUT_DIR, exist_ok=True)

# === LABELS E MAPPING ===
labels = load_labels(LABELS_PATH)
matched, id_to_name, name_to_idx  = load_mapping_nn2(CSV_PATH, labels, TESTSET_PATH)

# === MODELLO NN2: ResNet50 ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model_nn2(labels)
print("✅ modello resnet50_ft caricato")

# === VALUTAZIONE IMMAGINI AVVELENATE ===
eps_folders = sorted([d for d in os.listdir(ADV_ROOT)
                      if d.startswith(f'{ATTACK_TYPE}') and
                         os.path.isdir(os.path.join(ADV_ROOT, d))])

summary = []
# === CICLO SU OGNI CONFIGURAZIONE DI ATTACCO ===
for folder in eps_folders:
    path_eps = os.path.join(ADV_ROOT, folder)
    params_dict, params_str = extract_params(folder, ATTACK) # Estrae i parametri dell'attacco dal nome della cartella
    start_time = time.time()
    success, total = 0, 0
    all_paths = [] # Per salvare immagini campione

    # === CICLO SU OGNI IDENTITÀ ===
    for cid in tqdm(os.listdir(path_eps), desc=f"⏳ {params_str}"):
        if cid not in id_to_name:
            continue
        gt_name = id_to_name[cid]
        gt_idx = name_to_idx[gt_name]
        folder_cid = os.path.join(path_eps, cid)

        # === CICLO SU OGNI IMMAGINE ===
        for img_name in os.listdir(folder_cid):
            img_path = os.path.join(folder_cid, img_name)
            try:
                img_rgb = np.asarray(Image.open(img_path).convert('RGB'), np.uint8)
                img_t = preprocessor.transform(img_rgb).unsqueeze(0).to(device) # Preprocessing per ResNet

                # === ESTRAZIONE TARGET INDEX (solo per attacchi targeted) ===
                if ATTACK_TYPE == "targeted":
                    if "TARGET_" in img_name:
                        try:
                            target_idx = int(img_name.split("TARGET_")[-1].split(".")[0])
                        except:
                            continue
                    else:
                        continue

                # === PREDIZIONE DEL MODELLO ===
                with torch.no_grad():
                    pred_idx = torch.argmax(model(img_t), 1).item()

                # Verifica successo attacco
                is_success = (pred_idx != gt_idx) if ATTACK_TYPE == "generic" else (pred_idx == target_idx)
                success += int(is_success)
                total += 1

                all_paths.append((img_path,
                                  labels[target_idx] if ATTACK_TYPE == "targeted" else labels[gt_idx],
                                  labels[pred_idx]))
                del img_t # Libera memoria
            except Exception as e:
                print(f"⚠️ {img_path}: {e}")

    # === CALCOLO METRICHE ===
    elapsed = time.time() - start_time
    if ATTACK_TYPE == "generic":
        acc = 1 - (success / total) if total > 0 else 0
        success_rate = success / total if total > 0 else 0
    elif ATTACK_TYPE == "targeted":
        success_rate = success / total if total > 0 else 0
        acc = 1 - success_rate

    print_metric_row(params_str, acc, success_rate, total)

    # === AGGIUNTA RISULTATI AL RIEPILOGO ===
    summary.append({
        "folder": folder,
        "params": params_dict,
        "accuracy": round(acc, 4),
        "success_rate": round(success_rate, 4),
        "time_sec": round(elapsed, 2),
        "total": total
    })

    # === SALVA IMMAGINI DI ESEMPIO ===
    random.shuffle(all_paths)
    sample = all_paths[:10]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for ax, (p, ref, pred) in zip(axes.flatten(), sample):
        img = Image.open(p).convert('RGB')
        ax.imshow(img)
        ax.axis('off')
        label = "Target" if ATTACK_TYPE == "targeted" else "GT"
        ax.set_title(f"Pred: {pred}\n{label}: {ref}", fontsize=9)
    plt.tight_layout()
    out_img = os.path.join(OUT_DIR, f"pred_{folder}.png")
    plt.savefig(out_img)
    plt.close()

# === RIEPILOGO E SALVATAGGIO CSV ===
print("\n📊 RIEPILOGO METRICHE")
csv_rows = []
for r in summary:
    params = r["params"] if r["params"] is not None else {}
    row = {**params, "accuracy": r["accuracy"], "success_rate": r["success_rate"], "time_sec": r["time_sec"]}
    print(f"{row} | Img: {r['total']}")
    csv_rows.append(row)

csv_df = pd.DataFrame(csv_rows)

# === ORDINAMENTO COLONNE PER ATTACCO ===
if ATTACK == "bim":
    sort_cols = ["eps", "eps_step", "max_iter", "verbose"]
elif ATTACK == "fgsm":
    sort_cols = ["eps"]
elif ATTACK == "cw":
    sort_cols = [
        "binary_search_steps", "confidence", "max_iter", "learning_rate", "initial_const", "batch_size", "verbose"
    ]
elif ATTACK == "pgd":
    sort_cols = [
        "eps", "eps_step", "max_iter", "num_random_init", "random_eps", "verbose"
    ]
elif ATTACK == "deepfool":
    sort_cols = [
        "epsilon", "nb_grads", "max_iter", "verbose"
    ]
else:
    sort_cols = []

# Ordina se le colonne ci sono nel DataFrame (evita errori se manca qualche colonna per caso)
sort_cols = [col for col in sort_cols if col in csv_df.columns]
if sort_cols:
    csv_df = csv_df.sort_values(by=sort_cols).reset_index(drop=True)

# === SALVATAGGIO FINALE CSV ===
csv_path = os.path.join(OUT_DIR, f"metrics_{ATTACK}_{ATTACK_TYPE}.csv")
csv_df.to_csv(csv_path, index=False)
print(f"\n💾 Risultati salvati in: {csv_path}")

