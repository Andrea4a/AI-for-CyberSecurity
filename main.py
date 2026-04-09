import os
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescentPyTorch, CarliniLInfMethod, DeepFool
from art.estimators.classification import PyTorchClassifier
import torch.nn as nn
import random

from config import *
from utils import *

# === PARSE ARGOMENTI DA CLI ===
args = parse_args()
types = [args.type] # 'generic' o 'targeted'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === SETUP MODELLO E DATI ===
set_seed(SEED)  # Imposta seed per riproducibilità
labels = load_labels(LABELS_PATH)  # Carica etichette umane (es. "Brad Pitt")
matched = load_mapping(CSV_PATH, labels, TESTSET_PATH)  # Mapping da ID a nome persona
model, classifier = load_model(labels)  # InceptionResnetV1 + ART PyTorchClassifier

SAVE_IMAGES = True  # Salva immagini avvelenate
results = []        # Lista per salvare risultati della grid

# === DEFINISCI SUFFIX IN BASE AL TIPO DI ATTACCO ===
suffix = "_TARGETED" if args.type == "targeted" else "_GENERIC"

# Selezione tipo di attacco in base agli argomenti da CLI
if args.bim:
    attack_name = "bim"
    attack_class = BasicIterativeMethod
    grid = get_grid("bim")
    save_dir = globals()[f"SAVE_DIR_BIM{suffix}"]
    output_csv = globals()[f"OUTPUT_CSV_BIM{suffix}"]
    tqdm_desc = "Attacco BIM-PyTorch"
elif args.fgsm:
    attack_name = "fgsm"
    attack_class = FastGradientMethod
    grid = get_grid("fgsm")
    save_dir = globals()[f"SAVE_DIR_FGSM{suffix}"]
    output_csv = globals()[f"OUTPUT_CSV_FGSM{suffix}"]
    tqdm_desc = "Attacco FGSM-PyTorch"
elif args.pgd:
    attack_name = "pgd"
    attack_class = ProjectedGradientDescentPyTorch
    grid = get_grid("pgd")
    save_dir = globals()[f"SAVE_DIR_PGD{suffix}"]
    output_csv = globals()[f"OUTPUT_CSV_PGD{suffix}"]
    tqdm_desc = "Attacco PGD-PyTorch"
elif args.cw:
    attack_name = "cw"
    attack_class = CarliniLInfMethod
    grid = get_grid("cw")
    save_dir = globals()[f"SAVE_DIR_CW{suffix}"]
    output_csv = globals()[f"OUTPUT_CSV_CW{suffix}"]
    tqdm_desc = "Attacco CW-PyTorch"
elif args.deepfool:
    attack_name = "deepfool"
    attack_class = DeepFool
    grid = get_grid("deepfool")
    save_dir = globals()[f"SAVE_DIR_DEEPFOOL{suffix}"]
    output_csv = globals()[f"OUTPUT_CSV_DEEPFOOL{suffix}"]
    tqdm_desc = "Attacco DeepFool-PyTorch"
else:
    raise ValueError("Specifica almeno un attacco valido con --bim, --fgsm, --pgd, --cw, o --deepfool")

os.makedirs(save_dir, exist_ok=True)

# === Ciclo su combinazioni della grid ===
for params in grid:
    for attack_type in types: # generic o targeted
        perturbations, linf_norms = [], []
        total, success = 0, 0
        start_time = time.time()

        log_str = " | ".join([f"{k}={v}" for k, v in params.items()])
        print(f"\n🚀 Attacco {attack_type.upper()} | {log_str}")

        # Istanzia l'attacco
        if attack_name == "deepfool":
            attack = attack_class(classifier, **params)
        elif attack_name == "cw":
            attack = attack_class(classifier, **params, targeted=(attack_type == "targeted"))
        else:
            attack = attack_class(
                estimator=classifier,
                targeted=(attack_type == "targeted"),
                **params
            )
        # Loop su ogni immagine del test set
        for id_, gt_name in tqdm(matched.items(), desc=tqdm_desc):
            folder = os.path.join(TESTSET_PATH, id_)
            if not os.path.isdir(folder): continue

            for img_name in os.listdir(folder):
                img_path = os.path.join(folder, img_name)
                img = Image.open(img_path).convert("RGB")
                img_np = np.asarray(img).astype(np.float32) / 255.0
                img_tensor = process_images([img_np])[0].unsqueeze(0).to(device)
                gt_idx = labels.index(gt_name)

                # Verifica che l'immagine sia inizialmente classificata correttamente
                with torch.no_grad():
                    pred_idx = torch.argmax(model(img_tensor), dim=1).item()
                if pred_idx != gt_idx:
                    continue # salta immagini già mal classificate

                # Costruisci label: one-hot target o ground truth
                label_np = np.array([gt_idx])
                image_np = img_tensor.cpu().numpy()

                if attack_type == "targeted":
                    other_indices = [i for i in range(len(labels)) if i != gt_idx]
                    target_idx = random.choice(other_indices)
                    label_np = np.array([target_idx]) 
                    label_np = torch.nn.functional.one_hot(torch.tensor(label_np), num_classes=len(labels)).numpy()

                # Generazione immagine avvelenata
                adv_np = attack.generate(x=image_np, y=label_np if attack_type == "targeted" else None)
                adv_tensor = torch.from_numpy(adv_np[0]).to(device)               

                pert = np.abs(adv_np[0] - image_np[0])
                perturbations.append(np.mean(pert))
                linf_norms.append(np.max(pert))
                #print(f"[DEBUG] {img_name} | L_inf: {np.max(pert):.5f}")

                # Predizione post-attacco
                with torch.no_grad():
                    adv_idx = torch.argmax(model(adv_tensor.unsqueeze(0)), dim=1).item()

                total += 1
                if (attack_type == "generic" and adv_idx != gt_idx) or \
                   (attack_type == "targeted" and adv_idx == target_idx):
                    success += 1
                # Salvataggio immagine avvelenata
                if SAVE_IMAGES:
                    param_folder = f"{attack_type}_{'_'.join([f'{k}{v}' for k, v in params.items()])}"
                    out_dir = os.path.join(save_dir, param_folder, id_)
                    os.makedirs(out_dir, exist_ok=True)

                    # Salvataggio RGB standard senza preprocessing
                    img_uint8 = ((adv_tensor + 1) / 2).clamp(0, 1).cpu().numpy()
                    img_uint8 = (img_uint8 * 255).astype(np.uint8)
                    img_uint8 = np.transpose(img_uint8, (1, 2, 0))  # CHW → HWC

                    adv_filename = f"{os.path.splitext(img_name)[0]}_TARGET_{target_idx}.jpg" if attack_type == "targeted" else img_name
                    Image.fromarray(img_uint8).save(os.path.join(out_dir, adv_filename))

        # Statistiche e salvataggio CSV
        elapsed = time.time() - start_time
        acc = 1 - (success / total) if total > 0 else 0
        linf_max = np.max(linf_norms) if linf_norms else 0

        results.append({
            **params,
            "accuracy": round(acc, 4),
            "success_rate": round(success / total, 4) if total > 0 else 0,
            "l_inf": round(linf_max, 6),
            "time_sec": round(elapsed, 2)
        })

        print(results)
        pd.DataFrame(results).to_csv(output_csv, index=False) 

print(f"\n✅ Grid SEC {attack_name.upper()} completata. Risultati salvati in {output_csv}")
