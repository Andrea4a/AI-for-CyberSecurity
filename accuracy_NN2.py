import os
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms

from resnet.resnet import resnet50
from utils import *

# === SETUP ===
# Inizializzazione del preprocessor specifico per il modello VGG
preprocessor = VGGPreprocessor()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Impostazione del seed per la riproducibilità
set_seed(SEED)
# Caricamento delle label utilizzate dal modello
labels = load_labels(LABELS_PATH)
# Caricamento della mappatura tra identità e label, usando file CSV e test set
matched, id_to_name, name_to_idx  = load_mapping_nn2(CSV_PATH, labels, TESTSET_PATH)
# Caricamento del modello ResNet50 
model = load_model_nn2(labels)

# Flag per decidere se salvare esempi visivi
SAVE_IMAGES = True
results = []

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Stampa il numero di identità trovate
print(f"✅ Identità trovate nel test set: {len(matched)}")

# === CARICAMENTO MODELLO ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("✅ Modello resnet50_ft caricato")                                                    # CHW float

# === VALUTAZIONE NN2 ===
correct, total = 0, 0  # Inizializza contatori per accuratezza

# Ciclo sulle identità e le immagini del test set
for cid, cname in tqdm(matched.items(), desc="🎯 Valutazione"):
    gt_idx = name_to_idx[cname]  # Indice della ground truth
    folder = os.path.join(TESTSET_PATH, cid)
    
    # Ciclo su ogni immagine nella cartella dell'identità corrente
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
            # Caricamento e preprocessing dell'immagine
            img_rgb = np.asarray(Image.open(img_path).convert('RGB'), np.uint8)
            img_t = preprocessor.transform(img_rgb).unsqueeze(0).to(device)
        # Inferenza del modello e previsione della classe
            with torch.no_grad():
                pred_idx = torch.argmax(model(img_t), 1).item()
            # Aggiorna i contatori di accuratezza
            correct += (pred_idx == gt_idx)
            total   += 1
            del img_t
        except Exception as e:
            print(f"⚠️ {img_path}: {e}")

# Calcolo e stampa dell'accuracy finale
acc = 100.0 * correct / total if total else 0.0
print(f"\n✅ Accuracy finale: {acc:.2f}% su {total} immagini")

# === VISUALIZZAZIONE ESEMPI ===
print("🖼️ Salvo 10 esempi in predictions_plot_resnet50_ft_fixed.png")
samples = []
for cid, cname in matched.items():
    folder = os.path.join(TESTSET_PATH, cid)
    for n in os.listdir(folder):
        samples.append((os.path.join(folder, n), cname))
random.shuffle(samples)
samples = samples[:10]

fig, axes = plt.subplots(2, 5, figsize=(18,7))
# Ciclo per visualizzare e classificare gli esempi selezionati
for ax, (img_path, cname_gt) in zip(axes.flatten(), samples):
    # Caricamento e conversione dell'immagine in RGB
    img_rgb = np.asarray(Image.open(img_path).convert('RGB'), np.uint8)
    
    #Preprocessing
    img_t = preprocessor.transform(img_rgb).unsqueeze(0).to(device)
     # Predizione dell'identità dell'immagine
    with torch.no_grad():
        pred_name = labels[torch.argmax(model(img_t), 1).item()]
    # Visualizzazione dell'immagine con predizione e ground truth
    ax.imshow(img_rgb)
    ax.axis('off')
    ax.set_title(f"Pred: {pred_name}\nGT: {cname_gt}", fontsize=9)
# Salva e visualizza la figura
plt.tight_layout()
plt.savefig("predictions_plot_resnet50_ft_fixed.png")
print("✅ Immagine salvata in predictions_plot_senet50_ft_fixed.png")
