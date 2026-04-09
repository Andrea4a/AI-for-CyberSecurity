import os
import time
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import argparse

from art.defences.detector.evasion import BinaryInputDetector
from art.estimators.classification import PyTorchClassifier
import torch.nn as nn
import torch.optim as optim

from utils import *
from config import *

# === PARSE ARGOMENTI ===
# Parsing degli argomenti CLI: tipo attacco e nome dell'attacco usato
args = parse_args()
ATTACK_TYPE = args.type
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
    raise ValueError("Devi specificare almeno un attacco")

# === PATH ===
# Percorso delle immagini avvelenate create per NN1 da un attacco specifico
ADV_ROOT = os.path.join("nn1", ATTACK, f"{ATTACK}_{ATTACK_TYPE}", f"sec_adv_{ATTACK}_{ATTACK_TYPE}")
adv_image_paths = []
# Navigazione ricorsiva per ottenere tutti i path delle immagini avvelenate
for subfolder in sorted(os.listdir(ADV_ROOT)):
    sub_path = os.path.join(ADV_ROOT, subfolder)
    if not os.path.isdir(sub_path):
        continue
    for cid in os.listdir(sub_path):
        cid_path = os.path.join(sub_path, cid)
        if not os.path.isdir(cid_path):
            continue
        for img_name in os.listdir(cid_path):
            adv_image_paths.append(os.path.join(cid_path, img_name))

# Path per il file CSV in output che conterrà i risultati del detector
OUT_CSV  = os.path.join("detector_results", f"detector_results_{ATTACK}_{ATTACK_TYPE}.csv")
os.makedirs("detector_results", exist_ok=True)

# === SETUP ===
# Imposta il device a CUDA se disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Carica le label (nomi delle persone) dal file .npy usando `load_labels`, che normalizza nomi e rimuove simboli
labels = load_labels(LABELS_PATH)
# Carica il mapping Class_ID → Nome Persona solo per le identità presenti nel test set e nelle label
matched = load_mapping(CSV_PATH, labels, TESTSET_PATH)

# === LOAD NN1 ===
# Carica il modello InceptionResnetV1 preaddestrato su VGGFace2 e lo wrappa in un ART PyTorchClassifier
nn1_model, _ = load_model(labels)

# === COSTRUZIONE BILANCIATA DEL TRAIN SET ===
print("\n🎯 Costruzione bilanciata del train set...")
max_total = 2000
max_per_class = max_total // 2 # metà immagini clean, metà adversariali

X_train, y_train = [], []

# --- CAMPIONAMENTO CLEAN ---
# Estrazione di tutte le immagini clean per le identità mappate nel test set
all_clean = []
for cid, name in matched.items():
    folder = os.path.join(TESTSET_PATH, cid)
    for img_name in os.listdir(folder):
        all_clean.append(os.path.join(folder, img_name))
# Shuffle e selezione casuale
random.shuffle(all_clean)
selected_clean = all_clean[:max_per_class]

# Caricamento immagini clean preprocessate (resize 256 → crop 224 → normalizzate in [-1, 1])
print("📥 Caricamento immagini clean...")
for path in tqdm(selected_clean, desc="Clean"):
    img = Image.open(path).convert("RGB")
    img_np = np.asarray(img).astype(np.float32) / 255.0
    img_t = process_images([img_np])[0].numpy()
    X_train.append(img_t)
    y_train.append(0)

# --- CAMPIONAMENTO AVVELENATO ---
# Selezione casuale delle immagini avvelenate
all_adv = adv_image_paths
random.shuffle(all_adv)
selected_adv = all_adv[:max_per_class]

# Caricamento immagini avvelenate preprocessate
print("📥 Caricamento immagini avvelenate...")
for path in tqdm(selected_adv, desc="Avvelenate"):
    img = Image.open(path).convert("RGB")
    img_np = np.asarray(img).astype(np.float32) / 255.0
    img_t = process_images([img_np])[0].numpy()
    X_train.append(img_t)
    y_train.append(1)  # Etichetta 1 = avvelenata

# Conversione finale in array NumPy per ART
X_train = np.stack(X_train)
y_train = np.array(y_train)

# === CREA MODELLO DETECTOR CNN ===
print("🛡️  Addestramento BinaryInputDetector...")
# Costruzione del modello CNN semplice + ART PyTorchClassifier tramite load_model_det
_ , classifier = load_model_det(labels)

# Addestramento del detector (BinaryInputDetector) su immagini clean/ad
detector = BinaryInputDetector(classifier)
detector.fit(X_train, y_train, nb_epochs=35, batch_size=16, verbose=True)
# Calcolo e stampa dell'accuratezza sul training set
pred_train = classifier.predict(X_train)
acc_train = np.mean(np.argmax(pred_train, axis=1) == y_train)
print(f"🎓 Accuracy sul train set: {acc_train:.4f}")

# === TEST SU IMMAGINI AVVELENATE ===
print("\n🔍 Valutazione immagini filtrate...")
summary = []
# Selezione delle cartelle contenenti le immagini per i vari parametri dell’attacco (eps, ecc.)
folders = sorted([d for d in os.listdir(ADV_ROOT)
                  if d.startswith(f'{ATTACK_TYPE}') and os.path.isdir(os.path.join(ADV_ROOT, d))])

# Loop su ogni set di immagini per parametri diversi
for folder in folders:
    path_eps = os.path.join(ADV_ROOT, folder)
    params_dict, params_str = extract_params(folder, ATTACK) # Estrazione dei parametri da nome cartella
    start = time.time()
    total, skip, correct = 0, 0, 0

    # Costruzione della lista di immagini con class ID e path completo
    all_image_paths = []
    for cid in os.listdir(path_eps):
        if cid not in matched: continue
        folder_cid = os.path.join(path_eps, cid)
        for img_name in os.listdir(folder_cid):
            all_image_paths.append((cid, os.path.join(folder_cid, img_name), img_name))

    # Valutazione immagine per immagine
    for cid, img_path, img_name in tqdm(all_image_paths, desc=f"Valutazione {folder}"):
        try:
            gt_name = matched[cid]
            gt_idx = labels.index(gt_name)

            # Preprocessing immagine
            img = Image.open(img_path).convert("RGB")
            img_np = np.asarray(img).astype(np.float32) / 255.0
            img_proc = process_images([img_np])[0].numpy()

            # Rilevazione da parte del detector adversarial
            a, b = detector.detect(np.expand_dims(img_proc, axis=0))
            if b == True:
                skip += 1
                continue  # Immagine scartata

            # Predizione con NN1 se immagine non scartata
            img_torch = torch.from_numpy(img_proc).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_nn1 = nn1_model(img_torch).argmax(1).item()
            total += 1

            # Verifica predizione corretta per attacco generic o targeted
            if ATTACK_TYPE == "generic" and pred_nn1 == gt_idx:
                correct += 1
            elif ATTACK_TYPE == "targeted" and "TARGET_" in img_name:
                target_idx = int(img_name.split("TARGET_")[-1].split(".")[0])
                if pred_nn1 == target_idx:
                    correct += 1
        except Exception as e:
            print(f"⚠️ Errore {img_path}: {e}")

    # Calcolo accuracy e tempo per il batch corrente
    acc = correct / total if total > 0 else 0
    elapsed = round(time.time() - start, 2)
    print(f"✅ {params_str} → Acc filtrata: {acc:.4f} | Scartate: {skip} | Immagini: {total}")

    # Salvataggio dei risultati in lista summary
    row = {**params_dict, "filtered_accuracy": round(acc, 4), "images": total, "scartate": skip, "time_sec": elapsed}
    summary.append(row)

# === SALVA CSV ===
df = pd.DataFrame(summary)

# Ordinamento dinamico in base al tipo di attacco
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

# Ordina se le colonne esistono nel DataFrame
sort_cols = [col for col in sort_cols if col in df.columns]
if sort_cols:
    df = df.sort_values(by=sort_cols).reset_index(drop=True)

# Salvataggio finale su disco
df.to_csv(OUT_CSV, index=False)
print(f"\n💾 Risultati salvati in: {OUT_CSV}")

