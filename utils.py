import pandas as pd
import csv
import os
import sys
import torch
import shutil
import pickle
import os
import random
import torch
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from art.estimators.classification import PyTorchClassifier
import torch.nn as nn
import itertools
import matplotlib.pyplot as plt
from config import *
from resnet.resnet import resnet50
import torch.nn.functional as F
import torch.optim as optim
import re

def get_grid(attack_name):
    """
    Restituisce una lista di dizionari con le combinazioni di parametri per ciascun attacco adversariale.
    Serve per eseguire una grid search sui parametri.
    """
    if attack_name == "fgsm":
        eps_values = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        return [{"eps": eps} for eps in eps_values]

    elif attack_name == "bim":
        eps_values = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.08, 0.1]
        eps_step_values = [0.005, 0.01]
        max_iter_values = [2, 4]
        return [
            {"eps": eps, "eps_step": eps_step, "max_iter": max_iter, "verbose": False}
            for eps, eps_step, max_iter in itertools.product(eps_values, eps_step_values, max_iter_values)
        ]

    elif attack_name == "pgd":
        eps_values = [0.02, 0.06, 0.1]
        alpha_values = [0.005]
        iters_values = [10, 20]
        random_init_values = [1, 3]
        return [
            {
                "eps": eps,
                "eps_step": alpha,
                "max_iter": iters,
                "num_random_init": rand_init,
                "random_eps": True,
                "verbose": False
            }
            for eps, alpha, iters, rand_init in itertools.product(
                eps_values, alpha_values, iters_values, random_init_values
            )
        ]

    elif attack_name == "cw":
        #binary_search_steps_values = [1]
        confidence_values = [0, 3, 6, 9]
        max_iter_values = [3]
        learning_rate_values = [0.005, 0.01]
        #initial_const_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        batch_size_values = [16]
        return [
            {
                #"binary_search_steps": b,
                "confidence": c,
                "max_iter": i,
                "learning_rate": lr,
                #"initial_const": const,
                "batch_size": bs,
                "verbose": False
            }
            for c, i, lr, bs in itertools.product(
                #binary_search_steps_values,
                confidence_values,
                max_iter_values,
                learning_rate_values,
                #initial_const_values,
                batch_size_values
            )
        ]

    elif attack_name == "deepfool":
        max_iter_values = [10, 25, 50, 100]
        nb_grads_values = [5, 10, 20]
        epsilon_values   = [1e-4, 1e-5, 1e-6]
        return [
            {
                "epsilon": eps,
                "nb_grads": grads,
                "max_iter": iters,
                "verbose": False
            }
            for eps, grads, iters in itertools.product(epsilon_values, nb_grads_values, max_iter_values)
        ]

    else:
        raise ValueError(f"Attacco '{attack_name}' non supportato")

def set_seed(seed):
    """
    Imposta il seme per garantire la riproducibilità su random, numpy e torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_labels(labels_path):
    """
    Carica le etichette da un file .npy e le pulisce rimuovendo underscore e virgolette.
    """
    labels = np.load(labels_path)
    return [str(label).replace('_', ' ').replace('"', '').strip() for label in labels]

def load_mapping(csv_path, labels, testset_path):
    """
    Crea un dizionario ID immagine ➝ nome, filtrando solo le classi effettivamente presenti nel test set
    e con label contenute nella lista fornita.
    """
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    df.columns = [col.strip() for col in df.columns]
    df['Name'] = df['Name'].str.replace('"', '').str.replace('_', ' ').str.strip()
    id_to_name = dict(zip(df['Class_ID'], df['Name']))
    
    matched = {}
    for id_ in os.listdir(testset_path):
        if id_ in id_to_name and id_to_name[id_] in labels:
            matched[id_] = id_to_name[id_]
    return matched

def load_mapping_nn2(csv_path, labels, testset_path):
    """
    Come load_mapping, ma restituisce anche name_to_idx (nome → indice) e id_to_name.
    Utile per classificatori PyTorch.
    """
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    df.columns = [col.strip() for col in df.columns]
    df['Name'] = df['Name'].str.replace('"', '').str.replace('_', ' ').str.strip()
    name_to_idx = {n: i for i, n in enumerate(labels)}
    id_to_name = dict(zip(df['Class_ID'], df['Name']))
    
    matched = {}
    for id_ in os.listdir(testset_path):
        if id_ in id_to_name and id_to_name[id_] in labels:
            matched[id_] = id_to_name[id_]
    return matched, id_to_name, name_to_idx

def load_model(labels):
    """
    Carica il modello InceptionResnetV1 da facenet_pytorch con classificazione abilitata
    e lo avvolge in un PyTorchClassifier di ART per supportare attacchi adversariali.
    """
    model = InceptionResnetV1(pretrained='vggface2').eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.classify = True
    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224),
        nb_classes=len(labels),
        clip_values=(-1.0, 1.0),
        preprocessing=None,
        device_type="gpu" if torch.cuda.is_available() else "cpu"
    )
    return model, classifier

def load_model_nn2(labels):
    """
    Carica la ResNet50 fine-tuned salvata come dizionario di stato pickle, e la porta in eval.
    """
    model = resnet50(num_classes=len(labels))
    load_state_dict(model, WEIGHTS_PATH)
    model.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    return model

def process_images(images, target_size=256):
    """
    Applica resize, crop, normalizzazione [-1, 1] ad un batch di immagini (array numpy).
    """
    processed_images = []
    mean_rgb = np.array([131.0912, 103.8827, 91.4953], dtype=np.float32).reshape(3, 1, 1) / 255.0

    resize_transform = transforms.Resize(target_size)
    center_crop = transforms.CenterCrop(224)

    for image_np in images:
        image = torch.from_numpy((image_np * 255).astype(np.uint8)).permute(2, 0, 1)
        processed_img = resize_transform(transforms.ToPILImage()(image))
        processed_img = transforms.ToTensor()(processed_img)
        processed_img = center_crop(processed_img)
        processed_img = processed_img.float()
        processed_img = processed_img * 2 - 1
        #processed_img -= torch.from_numpy(mean_rgb).to(processed_img.device)
        processed_images.append(processed_img)

    return torch.stack(processed_images, dim=0)


def load_state_dict(model, fname):
    """
    Carica i pesi da un file pickle e li inserisce nel modello dato.
    Il file pickle contiene pesi convertiti da modelli Caffe (es. VGGFace2).
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))


def get_id_label_map(meta_file):
    """
    Restituisce un dizionario che associa ogni ID ad un indice, filtrando per Flag == 1
    (usato per escludere identità scartate nel dataset VGGFace2).
    """
    N_IDENTITY_PRETRAIN = 8631  # numero di identità usate nel training
    df = pd.read_csv(meta_file, sep=',', quotechar='"', encoding="utf-8", engine='python', on_bad_lines='skip')
    df.columns = [col.strip() for col in df.columns]
    df = df[df["Flag"] == 1].reset_index(drop=True)  # Solo identità con Flag == 1
    id_label_dict = {cid: idx for idx, cid in enumerate(df["Class_ID"])}
    return id_label_dict

def create_dir(dir_name):
    """
    Crea la directory specificata se non esiste già.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

class VGGPreprocessor:
    """
    Preprocessing stile VGGFace2: resize, crop, BGR mean subtraction, trasposizione.
    """
    def __init__(self):
        self.mean_bgr = np.array([91.4953, 103.8827, 131.0912], dtype=np.float32)
        self.resize_crop = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])

    def transform(self, img):
        pil = Image.fromarray(img)
        pil = self.resize_crop(pil)
        img = np.asarray(pil, dtype=np.uint8)
        img = img[:, :, ::-1]  # RGB → BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).float()

def extract_params(folder, attack):
    """
    Estrae i parametri di attacco codificati nel nome della cartella.
    """
    if attack == "cw":
        match = re.search(
            r'confidence([0-9.]+)_max_iter(\d+)_learning_rate([0-9.]+)_batch_size([0-9.]+)_verbose(\w+)',
            folder)
        if match:
            d = match.groups()
            return {
                #"binary_search_steps": int(float(d[0])),
                "confidence": float(d[0]),
                "max_iter": int(d[1]),
                "learning_rate": float(d[2]),
                #"initial_const": float(d[4]),
                "batch_size": int(float(d[3])),
                "verbose": d[4] == "True"
            }, f"conf={d[0]}, iter={d[1]}, lr={d[2]}, batch={d[3]}, verbose={d[4]}"
    elif attack == "fgsm":
        match = re.search(r'eps([0-9.]+)', folder)
        if match:
            eps = match.group(1)
            return {"eps": float(eps)}, f"eps={eps}"
    elif attack == "bim":
        match = re.search(r'eps([0-9.]+)_eps_step([0-9.]+)_max_iter(\d+)_verbose(\w+)', folder)
        if match:
            d = match.groups()
            return {
                "eps": float(d[0]),
                "eps_step": float(d[1]),
                "max_iter": int(d[2]),
                "verbose": d[3] == "True"
            }, f"eps={d[0]}, eps_step={d[1]}, iter={d[2]}, verbose={d[3]}"
    elif attack == "pgd":
        match = re.search(
            r'eps([0-9.]+)_eps_step([0-9.]+)_max_iter(\d+)_num_random_init(\d+)_random_eps(\w+)_verbose(\w+)', folder)
        if match:
            d = match.groups()
            return {
                "eps": float(d[0]),
                "eps_step": float(d[1]),
                "max_iter": int(d[2]),
                "num_random_init": int(d[3]),
                "random_eps": d[4] == "True",
                "verbose": d[5] == "True"
            }, f"eps={d[0]}, eps_step={d[1]}, iter={d[2]}, num_init={d[3]}, random_eps={d[4]}, verbose={d[5]}"
    elif attack == "deepfool":
        match = re.search(
            r'epsilon([0-9eE\+\-\.]+)_nb_grads(\d+)_max_iter(\d+)_verbose(\w+)', folder)
        if match:
            d = match.groups()
            return {
                "epsilon": float(d[0]),
                "nb_grads": int(d[1]),
                "max_iter": int(d[2]),
                "verbose": d[3] == "True"
            }, f"epsilon={d[0]}, nb_grads={d[1]}, iter={d[2]}, verbose={d[3]}"
    return None, "[Parametri non estratti]"


def print_metric_row(params_str, acc, success_rate, total):
    """Stampa un riepilogo leggibile delle metriche ottenute."""
    print(f"✅ {params_str} → accuracy: {acc:.2f} | success rate: {success_rate:.2f} | immagini: {total}")

class DetectorCNN(nn.Module):
    """
    Piccola CNN per classificazione binaria:
    - 2 layer convoluzionali + MaxPool
    - 2 layer fully connected
    """
    def __init__(self):
        super(DetectorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 64)  # 224 → 112 → 56
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # [B, 16, 224, 224]
        x = self.pool1(x)              # [B, 16, 112, 112]
        x = F.relu(self.conv2(x))      # [B, 32, 112, 112]
        x = self.pool2(x)              # [B, 32, 56, 56]
        x = x.view(-1, 32 * 56 * 56)   # Flatten
        x = F.relu(self.fc1(x))       
        x = self.fc2(x)                # No softmax (ART lo gestisce)
        return x

def load_model_det(labels):
    """
    Inizializza DetectorCNN, definisce ottimizzatore e wrapper ART PyTorchClassifier.
    """
    model = DetectorCNN()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    classifier = PyTorchClassifier(
        model=model,
        loss=loss,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=2,
        clip_values=(-1.0, 1.0),
        device_type="gpu" if torch.cuda.is_available() else "cpu"
    )

    return model, classifier

    
