import os
import shutil
import random
import pandas as pd
import numpy as np
import tensorflow as tf

# === CONFIGURAZIONE ===
min_images = 10 #500                # Minimo di immagini richieste per selezionare un'identità
images_per_identity = 1        # Quante immagini selezionare per ciascuna identità
num_identities = 100            # Quante identità scegliere per il test set

csv_path = os.path.expanduser('~/aicyb/identity_meta.csv')
src_base = os.path.expanduser('~/aicyb/test_set')  # da dove copiare
dst_base = os.path.expanduser('~/aicyb/test_set_100')         # dove salvare

# === CARICA CSV E MAPPING ID ➝ Nome ===
df = pd.read_csv(csv_path, on_bad_lines='skip') # Carica il CSV, salta righe problematiche
df.columns = df.columns.str.strip() # Rimuove spazi dai nomi delle colonne
df['Name'] = df['Name'].str.replace('"', '').str.strip() # Rimuove virgolette e spazi dai nomi
id_to_name = dict(zip(df['Class_ID'], df['Name'])) # Mappa Class_ID → Nome

# === CARICA LABELS DEL MODELLO ===
# Scarica le label ufficiali del modello VGGFace2 da GitHub
fpath = tf.keras.utils.get_file(
    'rcmalli_vggface_labels_v2.npy',
    "https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v2.npy",
    cache_subdir="./"
)

# Carica le label dal file .npy e le normalizza come stringhe
labels = np.load(fpath)
labels = [str(label).strip() for label in labels]

# === FILTRA ID PRESENTI SUL DISCO E COMPATIBILI CON IL MODELLO ===
# Seleziona le identità che:
# 1. Sono presenti nel filesystem (esistono nella cartella src_base)
# 2. Contengono almeno `min_images` immagini
# 3. Sono tra le label supportate dal modello (es. VGGFace2)
available_ids = [
    i for i in df['Class_ID']
    if os.path.isdir(os.path.join(src_base, i))
    and len(os.listdir(os.path.join(src_base, i))) >= min_images
    and id_to_name.get(i) in labels
]

print(f"✅ Identità compatibili trovate: {len(available_ids)} (con almeno {min_images} immagini)")

# === CHECK NUMERO IDENTITÀ ===
if len(available_ids) < num_identities:
    raise ValueError(f"❌ Non ci sono abbastanza identità compatibili (trovate {len(available_ids)}, richieste {num_identities}).")
else:
    print(f"✅ Seleziono {num_identities} identità random tra le {len(available_ids)} compatibili.")

selected_ids = random.sample(available_ids, num_identities)

# === CREA NUOVO TESTSET BILANCIATO ===
os.makedirs(dst_base, exist_ok=True)

created_identities = 0

for identity in selected_ids:
    src_dir = os.path.join(src_base, identity) # Directory sorgente per questa identità
    dst_dir = os.path.join(dst_base, identity) # Directory destinazione
    os.makedirs(dst_dir, exist_ok=True)

    images = os.listdir(src_dir) # Lista immagini disponibili
    if len(images) >= images_per_identity:
        # Se ci sono abbastanza immagini, seleziona un sottoinsieme casuale
        selected_imgs = random.sample(images, images_per_identity)
        for img in selected_imgs:
            shutil.copy(os.path.join(src_dir, img), os.path.join(dst_dir, img))
        created_identities += 1
    else:
        print(f"⚠️ Skipping {identity}: solo {len(images)} immagini disponibili (richieste {images_per_identity})")

print(f"\n✅ Test set bilanciato creato in: {dst_base}")
print(f"📦 Contiene: {created_identities} identità x {images_per_identity} immagini = {created_identities * images_per_identity} immagini totali")
