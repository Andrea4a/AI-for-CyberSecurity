import numpy as np
import tensorflow as tf
import pandas as pd
import torch
import torch.nn.functional as F
import os
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from config import *
from utils import *

# === SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Imposta seed per la riproducibilità dei risultati
set_seed(SEED)
# Carica le label associate alle identità nel dataset di test
labels = load_labels(LABELS_PATH)
# Carica la mappatura ID → Nome Persona per le identità presenti nel test set
matched = load_mapping(CSV_PATH, labels, TESTSET_PATH)
# Carica il modello NN1 (InceptionResnetV1) e il classificatore ART associato
model, classifier = load_model(labels)
# Flag per abilitare la generazione di immagini di esempio
SAVE_IMAGES = True
results = []

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(f"✅ Identità trovate nel test set: {len(matched)}")

# === VALUTAZIONE NN1 ===
correct = 0
total = 0
skipped = 0

# Itera su ciascuna identità e valuta ogni immagine
for identity_id, identity_name in tqdm(matched.items(), desc="🎯 Valutazione classificatore diretto"):
    folder = os.path.join(TESTSET_PATH, identity_id)
    if not os.path.isdir(folder):
        skipped += 1
        continue

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
             # Caricamento e normalizzazione immagine
            img = Image.open(img_path).convert("RGB")
            img_np = np.asarray(img).astype(np.float32) / 255.0
            img_tensor = process_images([img_np])[0].unsqueeze(0).to(device)
            # Inferenza del modello
            with torch.no_grad():
                output = model(img_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                pred_name = labels[pred_idx]
            # Confronta predizione con ground truth
            if pred_name == identity_name:
                correct += 1
            total += 1

            # Pulizia memoria
            del img_tensor
            del output
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"⚠️ Errore con immagine {img_path}: {e}")
            continue
# Calcolo e stampa accuratezza finale
acc = correct / total if total > 0 else 0
print(f"\n✅ Accuratezza finale: {acc * 100:.2f}% su {total} immagini testate")
print(f"📦 Identità saltate (non trovate): {skipped}")

# === VISUALIZZAZIONE ESEMPI ===
print("\n🖼️ Salvo 10 immagini *casuali* con la predizione in predictions_plot.png")
all_img_paths = []
for identity_id, identity_name in matched.items():
    folder = os.path.join(TESTSET_PATH, identity_id)
    if not os.path.isdir(folder):
        continue
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        all_img_paths.append((img_path, identity_id, identity_name))

# Seleziona casualmente 10 immagini
random.shuffle(all_img_paths)
selected_imgs = all_img_paths[:10]

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()

# Visualizza ogni immagine e la predizione associata
for i, (img_path, identity_id, identity_name) in enumerate(selected_imgs):
    try:
        img = Image.open(img_path).convert("RGB")
        img_np = np.asarray(img).astype(np.float32) / 255.0
        img_tensor = process_images([img_np])[0].unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            pred_name = labels[pred_idx]
         # Mostra immagine originale con titolo predizione vs ground truth
        axes[i].imshow(img)  # Mostriamo l'immagine originale
        axes[i].axis('off')
        axes[i].set_title(f"Pred: {pred_name}\nGT: {identity_name}", fontsize=10)

        del img_tensor
        del output
        torch.cuda.empty_cache()

    except Exception as e:
        axes[i].set_visible(False)
        continue

plt.tight_layout()
plt.savefig("predictions_plot.png")
print("✅ Immagine salvata in predictions_plot.png")
