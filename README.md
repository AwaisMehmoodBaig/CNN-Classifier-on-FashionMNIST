# 🧠 CNN Classifier on FashionMNIST (PyTorch)

A simple yet effective Convolutional Neural Network (CNN) built using **PyTorch** to classify images from the **FashionMNIST** dataset. This project walks through model definition, training, evaluation, and prediction — with visual output.

---

## 📌 Project Overview

- **Goal:** Classify 28x28 grayscale images of clothing items into one of 10 categories.
- **Model Type:** Convolutional Neural Network (CNN)
- **Dataset:** FashionMNIST (provided by `torchvision.datasets`)
- **Accuracy Achieved:** ~90% on test data after 5 epochs
- **Frameworks:** PyTorch, TorchVision, Matplotlib

---

## 🧱 Architecture Summary

The CNN consists of:

- **Conv Layer 1:** 1 input channel → 16 filters (3×3), ReLU, MaxPool(2×2)
- **Conv Layer 2:** 16 → 32 filters (3×3), ReLU, MaxPool(2×2)
- **Fully Connected:** Flattened feature maps (32×7×7) → 10 output classes

```python
Input  → Conv2d → ReLU → MaxPool →
        Conv2d → ReLU → MaxPool →
        Flatten → Linear → Output

# 📦 Installation

Install required packages (if not already):

pip install torch torchvision matplotlib

# 📊 Dataset Description

0: T-shirt/top     5: Sandal
1: Trouser         6: Shirt
2: Pullover        7: Sneaker
3: Dress           8: Bag
4: Coat            9: Ankle boot

# 🏋️ Training Summary

Loss Function: CrossEntropyLoss
Optimizer: Adam
Epochs: 5
Batch Size: 64
