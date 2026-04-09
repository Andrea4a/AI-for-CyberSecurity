# Adversarial Robustness and Defense for Face Recognition Systems

This project analyzes the robustness of face recognition models to different adversary attacks, the transferability of attacks between different networks, and the effectiveness of detector-based defense strategies.

The main models considered are:

- **NN1**: `InceptionResnetV1`
- **NN2**: `ResNet50` fine-tuned

## Project objectives

The project allows you to:

- assess the vulnerability of models to various adversarial attacks
- compare **targeted** and **targeted** attacks
- measure the **transferability** of attacks from one model to another
- apply **defense** techniques via detector
- view performance via **SEC (Success Error Curve)**

## Main parameters

The scripts support some common parameters:

- `--net`  
  Model to use. Supported values:
  - `nn1` → InceptionResnetV1
  - `nn2` → ResNet50 fine-tuned

- `--fgsm`, `--bim`, `--pgd`, `--cw`, `--deepfool`  
  They specify the type of attack to perform.

- `--type`  
  Attack mode:
  - `generic` → untargeted attack
  - `targeted` → targeted attack

## Usage examples

### Performing adversary attacks on NN1

To perform an adversary attack on **NN1**, for example a **Carlini & Wagner targeted** attack:

```bash
python3 main.py --cw --type targeted
```

## Calculating accuracy

To calculate the accuracy of models on an unpoisoned test set:

```bash
python3 accuracy_NN1.py # Accuracy of NN1
python3 accuracy_NN2.py # Accuracy of NN2
```

## Filtering with detector

To apply a detector to poisoned images and measure accuracy after filtering, for example with a **FGSM generic** attack on **NN1**:

```bash
python3 detector.py --fgsm --type generic
```

## Viewing SEC curves

To generate the **Success Error Curve (SEC)** for a specific attack on one of the models:

```bash
python3 plot_sec.py --fgsm --type generic --net nn1
python3 plot_sec.py --fgsm --type generic --net nn2
```

## Creating the test set

To create balanced or custom test sets, change the parameters in the dedicated script and run:

```bash
python3 testset_bil.py
```

## Transferability evaluation on NN2

To test the transferability of an attack from **NN1** to **NN2**, for example a generic **FGSM** attack:

```bash
python3 transfer.py --fgsm --type generic
```

## General structure of the project

The main files include:

- `main.py` → performing adversarial attacks
- `accuracy_NN1.py` → NN1 accuracy calculation
- `accuracy_NN2.py` → NN2 accuracy calculation
- `detector.py` → detection and filtering of poisoned images
- `plot_sec.py` → SEC curve generation
- `testset_bil.py` → creating the test set
- `trasfer.py` → evaluating the transferability of attacks
- `utils.py` → support functions and operating configurations

## Notes

- The test dataset must be placed in the `test_set` folder.
- Pre-trained models must be available locally or downloadable as defined in `config.py`.
- Parameters such as `eps`, `max_iter` and `alpha` can be configured via `utils.py`.
- The `utils.py` file contains auxiliary functions shared between different scripts.

## Requirements

Before running the project, make sure you have all the required dependencies installed and configured correctly:

- test dataset
- pre-trained models
- configuration file

## Possible extensions

The project can be extended with:

- new adversary attacks
- new defense detectors
- additional facial recognition models
- more in-depth quantitative comparisons between robustness and transferability
