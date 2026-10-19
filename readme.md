# 🛡️ Adversarial Attacks & Defense in Face Recognition

**AI Security | Deep Learning | Adversarial Machine Learning | Python**

A project focused on evaluating the robustness of Deep Learning-based face recognition systems against adversarial attacks and exploring defense strategies to improve their reliability.

Developed as a university group project during the Master's Degree in Computer Engineering at the **University of Salerno**.

## 🚀 Key Features

- **Face Recognition:** Evaluation of pre-trained InceptionResNetV1 and ResNet50 models on the VGGFace2 dataset.
- **Adversarial Attacks:** Implementation and analysis of FGSM, BIM, PGD, Carlini & Wagner, and DeepFool.
- **Attack Transferability:** Evaluation of adversarial examples generated against one model on a different architecture.
- **Adversarial Defense:** Development of a lightweight CNN-based detector to identify and filter adversarial images.
- **Robustness Evaluation:** Analysis of classification performance under different attack configurations.

## 🛠️ Technologies Used

- **Language:** Python
- **Deep Learning:** PyTorch, CNNs, InceptionResNetV1, ResNet50
- **AI Security:** Adversarial Robustness Toolbox (ART)
- **Techniques:** Adversarial Attacks, Transfer Learning, Adversarial Detection
- **Dataset:** VGGFace2

## 📊 Results

The two face recognition models achieved the following accuracy on a clean test set of 1,000 images:

| Model | Clean Accuracy |
|---|---|
| InceptionResNetV1 | 91.8% |
| ResNet50 | 87.3% |

The experiments demonstrated that adversarial perturbations can significantly reduce recognition accuracy, with some attacks also transferring between different model architectures.

A CNN-based adversarial detector was implemented to filter potentially manipulated inputs, improving classification accuracy on retained samples under several attack configurations.

## 🔍 Key Takeaways

- Face recognition systems can be highly vulnerable to small adversarial perturbations.
- Adversarial examples may transfer across different neural network architectures.
- Detection-based defenses can mitigate certain attacks, although aggressive filtering may discard legitimate inputs.

## 👥 Project Information

- **University:** University of Salerno
- **Course:** Artificial Intelligence for Cybersecurity
- **Academic Year:** 2024–2025
- **Project Type:** University Group Project
- **Team:** Andrea Rescigno, Pietro Zappullo, Mattia Lamberti, Pasquale Petti
