# test_model.py

import torch
from model.cnn_model import CNNModel

# Create an instance of your CNN model
model = CNNModel()

# Create a fake input image batch: 8 images, 1 channel, 28x28
dummy_input = torch.randn(8, 1, 28, 28)

# Forward pass through the model
output = model(dummy_input)

# Print shapes to verify
print("Input shape:", dummy_input.shape)   # [8, 1, 28, 28]
print("Output shape:", output.shape)       # [8, 10]
